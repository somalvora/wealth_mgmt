import json
import sqlite3
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import chromadb
import pytest
from chromadb.config import Settings
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from household_agent.agents import (
    MAX_DISPATCHES,
    MAX_QUERY_REFINEMENTS,
    LEDGER_PROMPT,
    LEDGER_TOOLS,
    AgentResult,
    Budget,
    DispatchItem,
    LedgerAnalyst,
    PolicyCompliance,
    Router,
    RoutingDecision,
    Supervisor,
    TaskAssignment,
    rule_route,
    variance_threshold,
)
from household_agent.approval import build_approval_graph
from household_agent.audit import AuditLog
from household_agent.authz import Authorizer
from household_agent.llm import HashEmbeddings
from household_agent.mcp_server import MEMBERS
from household_agent.retrieval import CANNOT_VERIFY, RetrievalService
from household_agent.synthetic import (
    ACCOUNTS,
    ALEX,
    DEMO_AS_OF,
    JORDAN,
    demo_planning_inputs,
    demo_policy_documents,
    demo_statements,
)

TAX = "San Mateo County Tax Collector"
RENTAL = "acct_rental_checking"
TRACE_QUERY = ("I paid $3,200 to the San Mateo County Tax Collector in October. Is this unusual compared "
               "with last year, and how should it be accounted for?")


class Scripted(BaseChatModel):
    """Plays back steps; a step is an AIMessage or a function of the conversation so far."""

    steps: list[Any]
    i: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        step = self.steps[self.i]
        self.i += 1
        msg = (step(messages) if callable(step) else step).model_copy(update={"id": None})  # fresh, like a real model
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "scripted"


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c_{name}_{len(args)}"}])


def final(*claims):
    return AIMessage(content=json.dumps({"claims": list(claims)}))


def tool_results(messages, name):
    return [json.loads(m.content) for m in messages if isinstance(m, ToolMessage) and m.name == name]


def _txn_ids(messages):
    return [t["transaction_id"] for r in tool_results(messages, "query_ledger") for t in r.get("transactions", [])]


def ledger_script(claim_text="October 2026 payment is $450.00 (16.36%) higher than October 2025."):
    return [
        call("query_ledger", account_ids=[RENTAL], start="2025-10-01", end="2026-10-31", merchant=TAX),
        lambda m: call("run_calculation", fn="variance_between_transactions", transaction_ids=_txn_ids(m)),
        lambda m: final({"text": claim_text,
                         "evidence_ids": [tool_results(m, "run_calculation")[-1]["run_id"], *_txn_ids(m)]}),
    ]


def _chunk_with(messages, needle):
    return next(p["chunk_id"] for r in tool_results(messages, "retrieve_verified") for p in r.get("passages", [])
                if needle in p["text"])


def policy_script():
    return [
        call("retrieve_verified", query="Monthly rent for Unit B lease", collections=["policies_compliance"]),
        call("retrieve_verified", query="rental property tax deductible Schedule E",
             collections=["policies_compliance"], tax_years=[2026]),
        lambda m: final(
            {"text": "Monthly rent for Unit B is $3,400.00.", "evidence_ids": [_chunk_with(m, "3,400.00")]},
            {"text": "Property tax on Unit B is deductible on Schedule E.", "kind": "tax", "tax_year": 2026,
             "evidence_ids": [_chunk_with(m, "3,400.00")]},
        ),
    ]


def assignment(agent="ledger_analyst", purpose="historical_comparison", **kw):
    return TaskAssignment(task_id="task_t", correlation_id="job_t", assigned_agent=agent, purpose=purpose,
                          authorized_scope={"household_id": "hh_demo"}, task="t", **kw)


@pytest.fixture
def retrieval(tmp_path, seeded_store):
    client = chromadb.PersistentClient(str(tmp_path / "chroma"), settings=Settings(anonymized_telemetry=False))
    svc = RetrievalService(client, HashEmbeddings(), seeded_store.authorizer, seeded_store.audit)
    for d in demo_policy_documents():
        svc.index(ALEX, d, correlation_id="idx", at=DEMO_AS_OF)
    return svc


def _ledger(seeded_store, categorizer, steps, actor=ALEX):
    return LedgerAnalyst(Scripted(steps=steps), seeded_store, categorizer, actor, clock=lambda: DEMO_AS_OF)


def _supervisor(seeded_store, specialists, router=None, **kw):
    return Supervisor(ALEX, router or Router(), specialists, seeded_store.audit, clock=lambda: DEMO_AS_OF, **kw)


# --- end to end: the §9 trace ------------------------------------------------------------

def test_supervisor_runs_the_property_tax_trace(seeded_store, categorizer, retrieval):
    specialists = {
        "ledger_analyst": _ledger(seeded_store, categorizer, ledger_script()),
        "policy_compliance": PolicyCompliance(Scripted(steps=policy_script()), retrieval, ALEX, clock=lambda: DEMO_AS_OF),
    }
    out = _supervisor(seeded_store, specialists).run(TRACE_QUERY, correlation_id="job_trace")
    assert out["routing"]["intent"] == "historical_comparison_and_tax_context"
    assert out["dispatches"] == 2 and out["status"] == "needs_review"
    ledger, policy = (AgentResult.model_validate(r) for r in out["results"])
    assert ledger.status == "completed" and ledger.outputs["dollar_variance_minor"] == 45000
    assert ledger.outputs["calc_run_id"] in ledger.evidence_ids
    assert ledger.outputs["category"] == "property_tax" and ledger.outputs["account_ids"] == [RENTAL]
    assert ledger.correlation_id == policy.correlation_id == "job_trace"
    assert policy.claims[0].text == "Monthly rent for Unit B is $3,400.00."
    assert policy.questions == [CANNOT_VERIFY]
    b = out["briefing"]
    assert [s["text"] for s in b["statements"]] == [ledger.claims[0].text, policy.claims[0].text]
    assert b["findings"][0]["type"] == "material_variance" and b["findings"][0]["pct_variance"] == "16.36"
    assert b["findings"][0]["threshold_pct"] == "10" and b["review"] is None  # property-tax override
    assert b["external_action"].startswith("none")
    actions = [e["action"] for e in seeded_store.audit.events("job_trace") if e["acting_agent"] == "supervisor"]
    assert actions == ["route", "dispatch", "dispatch", "threshold_check", "compose_briefing"]
    assert seeded_store.audit.verify() > 0


# --- Ledger Analyst ----------------------------------------------------------------------

def test_ledger_outputs_come_from_tools_not_model_text(seeded_store, categorizer):
    r = _ledger(seeded_store, categorizer, ledger_script())(assignment())
    assert r.status == "completed" and r.outputs["pct_variance"] == "16.36" and len(r.claims) == 1


def test_ledger_claim_with_invented_number_is_removed(seeded_store, categorizer):
    r = _ledger(seeded_store, categorizer, ledger_script("The 2026 payment is $500.00 higher than 2025."))(assignment())
    assert r.claims == [] and r.status == "needs_review"
    assert any("numbers not found" in e and "500" in e for e in r.errors)
    assert r.outputs["dollar_variance_minor"] == 45000  # the real number survives


def test_ledger_claim_citing_only_the_run_can_use_its_input_dates(seeded_store, categorizer):
    steps = [*ledger_script()[:2], lambda m: final({
        "text": "The October 2026 payment is $450 (16.36%) higher than October 2025.",
        "evidence_ids": [tool_results(m, "run_calculation")[-1]["run_id"]]})]
    r = _ledger(seeded_store, categorizer, steps)(assignment())
    assert r.status == "completed" and len(r.claims) == 1


def test_ledger_run_inputs_do_not_vouch_for_other_years(seeded_store, categorizer):
    steps = [*ledger_script()[:2], lambda m: final({
        "text": "The October 2026 payment is $450 higher than October 2024.",
        "evidence_ids": [tool_results(m, "run_calculation")[-1]["run_id"]]})]
    r = _ledger(seeded_store, categorizer, steps)(assignment())
    assert r.claims == [] and any("2024" in e for e in r.errors)


def test_ledger_claim_citing_unknown_evidence_is_removed(seeded_store, categorizer):
    steps = [*ledger_script()[:2], final({"text": "Property tax rose.", "evidence_ids": ["calc_run_made_up"]})]
    r = _ledger(seeded_store, categorizer, steps)(assignment())
    assert r.claims == [] and any("not retrieved" in e for e in r.errors)


def test_ledger_tool_budget(seeded_store, categorizer):
    r = _ledger(seeded_store, categorizer, [call("list_accounts")] * 7)(assignment())
    assert r.status == "budget_exhausted" and r.needs_review
    assert any("budget_exhausted" in e for e in r.errors)


def test_ledger_model_is_told_its_tool_budget_and_today(seeded_store, categorizer):
    seen = []
    steps = [lambda msgs: seen.append(msgs) or final()]
    _ledger(seeded_store, categorizer, steps)(assignment(budget=Budget(max_tool_calls=4)))
    system, task = seen[0][0].content, json.loads(seen[0][1].content)
    assert task["tool_budget"] == 4 and "tool_budget" in system and "merchant" in system
    assert task["today"] == DEMO_AS_OF.date().isoformat() and "relative to today" in system
    assert "must cite the run_id" in system


def test_ledger_retry_limit_per_tool(seeded_store, categorizer):
    bad = call("query_ledger", account_ids=[RENTAL], start="Oct 1", end="2026-10-31")
    r = _ledger(seeded_store, categorizer, [bad, bad, bad, final()])(assignment())
    assert r.status == "needs_review" and len(r.errors) == 3


def test_ledger_cannot_use_tools_outside_its_allowlist(seeded_store, categorizer):
    steps = [call("assess_risk", fn="x"), call("retrieve_verified", query="q"), final()]
    r = _ledger(seeded_store, categorizer, steps)(assignment())
    assert sum("not allowed" in e for e in r.errors) == 2


def test_ledger_cross_member_query_is_denied(seeded_store, categorizer):
    steps = [call("query_ledger", account_ids=["acct_alex_checking"], start="2026-10-01", end="2026-10-31"), final()]
    r = _ledger(seeded_store, categorizer, steps, actor=JORDAN)(assignment(purpose="budgeting"))
    assert r.outputs == {} and r.evidence_ids == [] and "authorization_denied" in r.errors[0]
    assert seeded_store.audit.events("job_t")[-1]["decision"] == "deny"


def test_ledger_unparseable_final_answer_needs_review(seeded_store, categorizer):
    r = _ledger(seeded_store, categorizer, [AIMessage(content="the variance is big")])(assignment())
    assert r.status == "needs_review" and "unparseable" in r.errors[0]


# --- Policy & Compliance -----------------------------------------------------------------

def test_policy_query_refinement_limit(retrieval):
    steps = [call("retrieve_verified", query=f"Unit B rent {i}") for i in range(4)] + [final()]
    r = PolicyCompliance(Scripted(steps=steps), retrieval, ALEX, clock=lambda: DEMO_AS_OF)(
        assignment("policy_compliance"))
    assert any("query refinements" in e for e in r.errors)


def test_policy_model_is_told_its_search_limit(retrieval):
    seen = []
    PolicyCompliance(Scripted(steps=[lambda msgs: seen.append(msgs) or final()]), retrieval, ALEX,
                     clock=lambda: DEMO_AS_OF)(assignment("policy_compliance"))
    assert f"at most {MAX_QUERY_REFINEMENTS + 1} times" in seen[0][0].content


def test_policy_model_is_told_today_from_its_clock(retrieval):
    seen = []
    later = DEMO_AS_OF + timedelta(days=400)
    PolicyCompliance(Scripted(steps=[lambda msgs: seen.append(msgs) or final()]), retrieval, ALEX,
                     clock=lambda: later)(assignment("policy_compliance"))
    assert json.loads(seen[0][1].content)["today"] == later.date().isoformat()


def test_policy_tax_claim_without_current_tier1_becomes_question(retrieval):
    r = PolicyCompliance(Scripted(steps=policy_script()), retrieval, ALEX, clock=lambda: DEMO_AS_OF)(
        assignment("policy_compliance"))
    assert [c.kind for c in r.claims] == ["fact"] and r.questions == [CANNOT_VERIFY] and r.status == "needs_review"


def test_policy_cannot_cite_private_doc_of_another_member(retrieval):
    steps = [call("retrieve_verified", query="Alex private bonus property tax"),
             final({"text": "Alex has a bonus of $9,000.00.", "evidence_ids": ["alex_private_note_000"]})]
    r = PolicyCompliance(Scripted(steps=steps), retrieval, JORDAN, clock=lambda: DEMO_AS_OF)(
        assignment("policy_compliance", purpose="budgeting"))
    assert r.claims == [] and any("not retrieved" in e for e in r.errors)


def test_policy_empty_query_is_an_error_observation(retrieval):
    r = PolicyCompliance(Scripted(steps=[call("retrieve_verified", query=" "), final()]), retrieval, ALEX,
                         clock=lambda: DEMO_AS_OF)(assignment("policy_compliance"))
    assert "invalid_request" in r.errors[0]


# --- Supervisor --------------------------------------------------------------------------

def _stub(status="completed", **kw):
    calls = []

    def run(a: TaskAssignment) -> AgentResult:
        calls.append(a)
        return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent=a.assigned_agent, status=status,
                           **kw)

    run.calls = calls
    return run


def test_external_action_is_refused_without_dispatch(seeded_store):
    stub = _stub()
    out = _supervisor(seeded_store, {"ledger_analyst": stub}).run("Please pay the county tax bill", correlation_id="j")
    assert out["status"] == "refused" and stub.calls == []
    assert any(e["action"] == "external_action" and e["decision"] == "deny" for e in seeded_store.audit.events("j"))


def test_model_router_cannot_clear_the_external_action_flag(seeded_store):
    lie = AIMessage(content=json.dumps({"intent": "historical_comparison", "dispatch_plan": [],
                                        "external_action_requested": False}))
    decision, source = Router(Scripted(steps=[lie]))("transfer $5,000 to brokerage")
    assert source == "model" and decision.external_action_requested


def test_router_reads_replies_split_into_content_blocks(seeded_store):
    # Gemini-style reply: text arrives as blocks, possibly beside non-text (reasoning) blocks
    body = json.dumps({"intent": "historical_comparison", "dispatch_plan": [{"agent": "ledger_analyst", "task": "x"}]})
    blocks = [{"type": "reasoning", "reasoning": "thinking..."},
              {"type": "text", "text": body[:20]}, {"type": "text", "text": body[20:]}]
    decision, source = Router(Scripted(steps=[AIMessage(content=blocks)]))(TRACE_QUERY)
    assert source == "model" and decision.dispatch_plan[0].agent == "ledger_analyst"


@pytest.mark.parametrize("reply", [
    "not json",
    json.dumps({"intent": "historical_comparison", "dispatch_plan": [{"agent": "tax_agent", "task": "x"}]}),
    json.dumps({"intent": "historical_comparison",
                "dispatch_plan": [{"agent": "ledger_analyst", "task": "x"}] * (MAX_DISPATCHES + 1)}),
])
def test_invalid_model_routing_falls_back_to_rules(reply):
    decision, source = Router(Scripted(steps=[AIMessage(content=reply)]))(TRACE_QUERY)
    assert source == "rules_fallback" and decision.intent == "historical_comparison_and_tax_context"


def test_purpose_comes_from_intent_not_from_the_model(seeded_store):
    stub = _stub()
    _supervisor(seeded_store, {"ledger_analyst": stub, "policy_compliance": stub}).run(TRACE_QUERY)
    assert {a.purpose for a in stub.calls} == {"historical_comparison"}
    assert all(a.authorized_scope["requester"] == "alex" for a in stub.calls)


def test_dispatch_budget(seeded_store):
    stub = _stub()
    out = _supervisor(seeded_store, {"ledger_analyst": stub, "policy_compliance": stub}, max_dispatches=1).run(TRACE_QUERY)
    assert len(stub.calls) == 1 and out["results"][1]["status"] == "budget_exhausted"
    assert out["status"] == "needs_review"


def test_job_time_limit(seeded_store):
    now = [DEMO_AS_OF]

    def slow(a):
        now[0] += timedelta(seconds=301)
        return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent=a.assigned_agent,
                           status="completed")

    sup = Supervisor(ALEX, Router(), {"ledger_analyst": slow, "policy_compliance": slow}, seeded_store.audit,
                     clock=lambda: now[0])
    out = sup.run(TRACE_QUERY)
    assert out["results"][1]["errors"] == ["job time limit"]


@pytest.mark.parametrize("specialists", [{"ledger_analyst": lambda a: 1 / 0}, {}])
def test_crashing_or_missing_specialist_becomes_failed_result(seeded_store, specialists):
    out = _supervisor(seeded_store, specialists).run("Is my October spend unusual?")
    assert out["results"][0]["status"] == "failed" and out["status"] == "needs_review"


def test_briefing_contains_only_specialist_claims(seeded_store):
    stub = _stub(claims=[{"text": "Validated claim.", "evidence_ids": ["e1"]}])
    out = _supervisor(seeded_store, {"ledger_analyst": stub}).run("Is my October spend unusual?")
    assert out["status"] == "completed"
    assert out["briefing"]["statements"] == [{"text": "Validated claim.", "evidence_ids": ["e1"], "agent": "ledger_analyst"}]


def test_allocation_request_pauses_for_approval(seeded_store):
    graph = build_approval_graph(inputs_for=lambda hh: demo_planning_inputs(), members=MEMBERS,
                                 audit=seeded_store.audit, checkpointer=InMemorySaver())
    out = _supervisor(seeded_store, {}, approval_graph=graph).run("We have a surplus. Where should the money go?")
    assert out["status"] == "awaiting_approval" and out["approval_thread_id"].startswith("appr_")
    state = graph.get_state({"configurable": {"thread_id": out["approval_thread_id"]}})
    assert state.next == ("approval",)


def test_out_of_scope_request_dispatches_nothing(seeded_store):
    stub = _stub()
    out = _supervisor(seeded_store, {"ledger_analyst": stub}).run("Tell me a joke")
    assert out["routing"]["intent"] == "out_of_scope" and stub.calls == [] and out["status"] == "completed"


@given(st.text(max_size=200))
def test_rule_router_always_produces_a_valid_bounded_plan(text):
    d = rule_route(text)
    RoutingDecision.model_validate(d.model_dump())
    agents = [i.agent for i in d.dispatch_plan]
    assert len(agents) == len(set(agents)) <= 2
    if d.intent == "allocation_planning":
        assert d.approval_required and not agents


# --- handoff contract: correlation IDs and frozen payloads -------------------------------

def test_handoff_payloads_are_frozen_and_carry_the_correlation_id(seeded_store, categorizer):
    a = assignment()
    r = _ledger(seeded_store, categorizer, ledger_script())(a)
    assert r.correlation_id == a.correlation_id == "job_t"
    with pytest.raises(ValidationError):
        a.correlation_id = "job_other"
    with pytest.raises(ValidationError):
        r.status = "completed"
    with pytest.raises(ValidationError):  # a result without the job's ID is not a valid handoff
        AgentResult(task_id="t", agent="ledger_analyst", status="completed")


def test_result_with_another_jobs_correlation_id_is_discarded(seeded_store):
    stub = _stub(claims=[{"text": "Smuggled claim.", "evidence_ids": ["e1"]}], outputs={"pct_variance": "99"})
    impostor = lambda a: stub(a).model_copy(update={"correlation_id": "job_other"})  # noqa: E731
    out = _supervisor(seeded_store, {"ledger_analyst": impostor}).run("Is my October spend unusual?",
                                                                       correlation_id="job_mine")
    [r] = out["results"]
    assert r["status"] == "failed" and r["correlation_id"] == "job_mine" and r["claims"] == [] and r["outputs"] == {}
    assert "correlation_id mismatch" in r["errors"][0] and out["status"] == "needs_review"
    assert out["briefing"]["statements"] == [] and out["findings"] == []
    dispatch = next(e for e in seeded_store.audit.events("job_mine") if e["action"] == "dispatch")
    assert dispatch["reason"] == "failed" and "job_other" in dispatch["details_json"]


# --- threshold rules (design §5.4) -------------------------------------------------------

def _variance_job(audit, pct, category, **kw):
    outputs = {"pct_variance": pct, "calc_run_id": "calc_1", **({"category": category} if category else {})}
    decision = RoutingDecision(intent="historical_comparison",
                               dispatch_plan=[DispatchItem(agent="ledger_analyst", task="x")])
    return Supervisor(ALEX, lambda _r: (decision, "rules"), {"ledger_analyst": _stub(outputs=outputs)}, audit,
                      clock=lambda: DEMO_AS_OF, **kw)


@pytest.mark.parametrize(("category", "pct", "limit"), [
    ("groceries", "12.00", None), ("property_tax", "12.00", "10"), ("groceries", "15.01", "15"),
    ("utilities", "-20", "15"), (None, "15.00", None), ("property_tax", "10.00", None),
])
def test_category_variance_uses_15pct_default_and_10pct_property_tax(seeded_store, category, pct, limit):
    out = _variance_job(seeded_store.audit, pct, category).run("x")
    if limit is None:
        assert out["findings"] == [] and out["status"] == "completed"
    else:
        [f] = out["findings"]
        assert f["threshold_pct"] == limit and f["category"] == category and out["status"] == "needs_review"


@settings(max_examples=40, deadline=None)
@given(st.decimals(min_value=-100, max_value=100, places=2),
       st.sampled_from(["property_tax", "groceries", "utilities", "hoa_dues", None]))
def test_a_variance_is_flagged_exactly_when_above_its_category_threshold(pct, category):
    conn = sqlite3.connect(":memory:")
    out = _variance_job(AuditLog(conn), str(pct), category).run("x")
    limit = Decimal("10") if category == "property_tax" else Decimal("15")
    assert variance_threshold(category) == limit
    assert bool(out["findings"]) == (abs(pct) > limit) == (out["status"] == "needs_review")
    conn.close()


# --- review interrupt (design §5.3), only with a checkpointer ----------------------------

PRIVATE = "acct_alex_checking"


def _review_job(seeded_store, status="completed", accounts=(RENTAL,), pct="16.36"):
    outputs = {"pct_variance": pct, "category": "property_tax", "calc_run_id": "calc_1", "account_ids": list(accounts)}
    stub = _stub(status, outputs=outputs)
    decision = RoutingDecision(intent="historical_comparison",
                               dispatch_plan=[DispatchItem(agent="ledger_analyst", task="x")])
    sup = Supervisor(ALEX, lambda _r: (decision, "rules"), {"ledger_analyst": stub}, seeded_store.audit,
                     clock=lambda: DEMO_AS_OF, members=MEMBERS, authorizer=seeded_store.authorizer)
    return sup, stub


def _reviews(audit, cid):
    return [(e["actor_id"], e["decision"]) for e in audit.events(cid) if e["action"] == "review"]


def test_without_a_checkpointer_there_is_no_review_node(seeded_store):
    sup, _ = _review_job(seeded_store)
    assert "review" not in sup.build().get_graph().nodes and "review" in sup.build(InMemorySaver()).get_graph().nodes
    out = sup.run("x", correlation_id="j_plain")
    assert out["status"] == "needs_review" and "__interrupt__" not in out
    assert _reviews(seeded_store.audit, "j_plain") == []


def test_finding_pauses_for_review_and_an_authorized_member_resolves_it(seeded_store):
    sup, stub = _review_job(seeded_store)
    saver = InMemorySaver()
    out = sup.run("x", correlation_id="j_rev", checkpointer=saver)
    [intr] = out["__interrupt__"]
    assert out["status"] == "awaiting_review" and "briefing" not in out
    assert intr.value["kind"] == "supervisor_review" and intr.value["correlation_id"] == "j_rev"
    assert intr.value["findings"][0]["threshold_pct"] == "10" and intr.value["results"] == []
    done = sup.resume("j_rev", {"reviewer": "jordan", "accepted": True, "note": "new assessment"}, checkpointer=saver)
    assert done["status"] == "reviewed" and done["briefing"]["review"] == {
        "reviewer": "jordan", "accepted": True, "note": "new assessment"}
    assert len(stub.calls) == 1  # resuming does not re-dispatch the specialist
    assert _reviews(seeded_store.audit, "j_rev") == [("jordan", "allow")]
    assert seeded_store.audit.verify() > 0


def test_incomplete_result_pauses_even_without_findings(seeded_store):
    sup, _ = _review_job(seeded_store, status="needs_review", pct="1.00")
    saver = InMemorySaver()
    out = sup.run("x", correlation_id="j_inc", checkpointer=saver)
    assert out["findings"] == [] and out["__interrupt__"][0].value["results"][0]["status"] == "needs_review"
    done = sup.resume("j_inc", {"reviewer": "alex", "accepted": False}, checkpointer=saver)
    assert done["status"] == "review_rejected" and done["review"]["accepted"] is False


def test_clean_job_with_a_checkpointer_does_not_pause(seeded_store):
    sup, _ = _review_job(seeded_store, pct="1.00")
    out = sup.run("x", correlation_id="j_clean", checkpointer=InMemorySaver())
    assert out["status"] == "completed" and "__interrupt__" not in out


@pytest.mark.parametrize("reviewer", ["sam", "casey_cpa", "mallory", ""])
def test_unauthorized_reviewer_is_refused_audited_and_the_job_keeps_waiting(seeded_store, reviewer):
    sup, _ = _review_job(seeded_store)
    saver = InMemorySaver()
    sup.run("x", correlation_id="j_deny", checkpointer=saver)
    out = sup.resume("j_deny", {"reviewer": reviewer, "accepted": True}, checkpointer=saver)
    assert out["status"] == "awaiting_review" and out["review_denied_attempts"] == 1 and out["__interrupt__"]
    assert _reviews(seeded_store.audit, "j_deny") == [(reviewer or "unknown", "deny")]
    assert sup.resume("j_deny", {"reviewer": "alex", "accepted": True}, checkpointer=saver)["status"] == "reviewed"


def test_reviewer_needs_access_to_every_account_the_job_used(seeded_store):
    sup, _ = _review_job(seeded_store, accounts=(RENTAL, PRIVATE))
    saver = InMemorySaver()
    sup.run("x", correlation_id="j_acct", checkpointer=saver)
    out = sup.resume("j_acct", {"reviewer": "jordan", "accepted": True}, checkpointer=saver)
    assert out["status"] == "awaiting_review"
    deny = next(e for e in seeded_store.audit.events("j_acct") if e["action"] == "review")
    assert deny["actor_id"] == "jordan" and deny["reason"].startswith(PRIVATE)
    assert sup.resume("j_acct", {"reviewer": "alex", "accepted": True}, checkpointer=saver)["status"] == "reviewed"


def test_paused_review_survives_restart(tmp_path):
    def open_all():
        ledger = sqlite3.connect(tmp_path / "ledger.db")
        ckpt = sqlite3.connect(tmp_path / "checkpoints.db", check_same_thread=False)  # own connection, as in approval
        store = SimpleNamespace(audit=AuditLog(ledger), authorizer=Authorizer(ACCOUNTS))
        return ledger, ckpt, store

    ledger1, ckpt1, store1 = open_all()
    _review_job(store1)[0].run("x", correlation_id="j_restart", checkpointer=SqliteSaver(ckpt1))
    ledger1.close()
    ckpt1.close()
    ledger2, ckpt2, store2 = open_all()  # "new process"
    sup2, stub2 = _review_job(store2)
    out = sup2.resume("j_restart", {"reviewer": "jordan", "accepted": True}, checkpointer=SqliteSaver(ckpt2))
    assert out["status"] == "reviewed" and stub2.calls == []
    actions = [e["action"] for e in store2.audit.events("j_restart")]
    assert actions == ["route", "dispatch", "threshold_check", "review", "compose_briefing"]
    assert store2.audit.verify() == 5
    ledger2.close()
    ckpt2.close()


# --- Ledger Analyst: ingest_statement ----------------------------------------------------

def test_ledger_tools_include_ingest_but_prompt_prefers_query_ledger():
    names = [t["function"]["name"] for t in LEDGER_TOOLS]
    assert names == ["list_accounts", "query_ledger", "ingest_statement", "run_calculation"]
    assert "prefer query_ledger" in LEDGER_PROMPT


def test_ledger_can_ingest_a_statement_through_its_tool(store, categorizer):
    csv_text = demo_statements()["2026_10_rental_checking.csv"]
    steps = [call("ingest_statement", csv_text=csv_text, source_name="rental.csv"), final()]
    r = _ledger(store, categorizer, steps)(assignment(purpose="statement_ingestion"))
    assert r.status == "completed" and r.errors == [] and r.outputs == {}
    assert any(e["action"] == "upload" and e["decision"] == "allow" for e in store.audit.events("job_t"))


def test_ledger_ingest_of_another_members_statement_is_denied(store, categorizer):
    csv_text = demo_statements()["2026_10_alex_checking.csv"]
    steps = [call("ingest_statement", csv_text=csv_text, source_name="alex.csv"), final()]
    r = _ledger(store, categorizer, steps, actor=JORDAN)(assignment(purpose="statement_ingestion"))
    assert "ingest_statement: authorization_denied" in r.errors[0]
    assert store.audit.events("job_t")[-1]["decision"] == "deny"
