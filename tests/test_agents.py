import hashlib
import json
import sqlite3
import threading
import time
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
    r = _ledger(store, categorizer, steps)(assignment(purpose="statement_ingestion", inputs={"csv_text": csv_text}))
    assert r.status == "completed" and r.errors == [] and r.outputs == {}
    assert any(e["action"] == "upload" and e["decision"] == "allow" for e in store.audit.events("job_t"))


def test_ledger_ingest_of_another_members_statement_is_denied(store, categorizer):
    csv_text = demo_statements()["2026_10_alex_checking.csv"]
    steps = [call("ingest_statement", csv_text=csv_text, source_name="alex.csv"), final()]
    r = _ledger(store, categorizer, steps, actor=JORDAN)(
        assignment(purpose="statement_ingestion", inputs={"csv_text": csv_text}))
    assert "ingest_statement: authorization_denied" in r.errors[0]
    assert store.audit.events("job_t")[-1]["decision"] == "deny"


# --- bounded concurrent dispatch (design §4.2) ---------------------------------------------

def _plan_router(*items, intent="historical_comparison_and_tax_context"):
    decision = RoutingDecision(intent=intent, dispatch_plan=list(items))
    return lambda request: (decision, "test")


def _ledger_item(task="ledger"):
    return DispatchItem(agent="ledger_analyst", task=task)


def _policy_item(task="policy", **kw):
    return DispatchItem(agent="policy_compliance", task=task, **kw)


def _dispatch_agents(audit, cid):
    return [e["acting_agent"] and json.loads(e["details_json"]).get("agent") for e in audit.events(cid)
            if e["action"] == "dispatch"]


def test_independent_specialists_run_at_the_same_time(seeded_store):
    meet = threading.Barrier(2, timeout=5)  # both must be inside their specialist at once, or this raises
    seen = []

    def specialist(a):
        seen.append(threading.current_thread().name)
        meet.wait()
        return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent=a.assigned_agent,
                           status="completed", outputs={"calc_inputs": a.inputs})

    router = _plan_router(_ledger_item(), _policy_item(depends_on_ledger=False))
    out = _supervisor(seeded_store, {"ledger_analyst": specialist, "policy_compliance": specialist},
                      router=router).run("x", correlation_id="j_par")
    assert [r["agent"] for r in out["results"]] == ["ledger_analyst", "policy_compliance"]
    assert [r["status"] for r in out["results"]] == ["completed", "completed"] and out["dispatches"] == 2
    assert len(set(seen)) == 2  # two worker threads
    assert "calc_outputs" not in out["results"][1]["outputs"]["calc_inputs"]  # independent: no Ledger hand-over
    assert _dispatch_agents(seeded_store.audit, "j_par") == ["ledger_analyst", "policy_compliance"]
    assert seeded_store.audit.verify() > 0


def test_dependent_plan_stays_sequential_and_policy_gets_calc_outputs(seeded_store):
    lock, active, peak, inputs = threading.Lock(), [0], [0], {}

    def occupy():  # long enough that two overlapping specialists would certainly be seen together
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1

    def ledger(a):
        occupy()
        return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent="ledger_analyst",
                           status="completed", outputs={"calc_run_id": "calc_1", "pct_variance": "5"})

    def policy(a):
        occupy()
        inputs.update(a.inputs)
        return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent="policy_compliance",
                           status="completed")

    out = _supervisor(seeded_store, {"ledger_analyst": ledger, "policy_compliance": policy},
                      router=_plan_router(_ledger_item(), _policy_item())).run("x")
    assert peak[0] == 1 and out["dispatches"] == 2
    assert inputs["calc_outputs"] == {"calc_1": {"pct_variance": "5"}}


def test_the_overlap_check_would_notice_concurrency(seeded_store):
    """Guards the test above: the same probe on an independent plan does see two specialists at once."""
    lock, active, peak = threading.Lock(), [0], [0]

    def specialist(a):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.2)
        with lock:
            active[0] -= 1
        return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent=a.assigned_agent,
                           status="completed")

    _supervisor(seeded_store, {"ledger_analyst": specialist, "policy_compliance": specialist},
                router=_plan_router(_ledger_item(), _policy_item(depends_on_ledger=False))).run("x")
    assert peak[0] == 2


def test_a_crashing_specialist_does_not_lose_its_partner(seeded_store):
    meet = threading.Barrier(2, timeout=5)

    def crash(a):
        meet.wait()
        raise RuntimeError("boom")

    def fine(a):
        meet.wait()
        return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent=a.assigned_agent,
                           status="completed", claims=[])

    out = _supervisor(seeded_store, {"ledger_analyst": crash, "policy_compliance": fine},
                      router=_plan_router(_ledger_item(), _policy_item(depends_on_ledger=False))).run("x", correlation_id="j_c")
    assert [r["status"] for r in out["results"]] == ["failed", "completed"]
    assert out["results"][0]["errors"] == ["RuntimeError: boom"] and out["status"] == "needs_review"
    assert _dispatch_agents(seeded_store.audit, "j_c") == ["ledger_analyst", "policy_compliance"]


def test_a_concurrent_result_for_another_job_is_discarded(seeded_store):
    meet = threading.Barrier(2, timeout=5)

    def impostor(a):
        meet.wait()
        return AgentResult(task_id=a.task_id, correlation_id="job_other", agent=a.assigned_agent, status="completed",
                           outputs={"pct_variance": "99"})

    def fine(a):
        meet.wait()
        return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent=a.assigned_agent,
                           status="completed")

    out = _supervisor(seeded_store, {"ledger_analyst": fine, "policy_compliance": impostor},
                      router=_plan_router(_ledger_item(), _policy_item(depends_on_ledger=False))).run("x")
    assert out["results"][0]["status"] == "completed"
    assert out["results"][1]["status"] == "failed" and "correlation_id mismatch" in out["results"][1]["errors"][0]
    assert out["findings"] == []


def test_a_specialist_that_outlives_the_job_time_limit_fails_without_losing_the_other(seeded_store):
    release = threading.Event()

    def stuck(a):
        release.wait(10)
        return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent=a.assigned_agent,
                           status="completed")

    def fine(a):
        return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent=a.assigned_agent,
                           status="completed")

    try:
        out = _supervisor(seeded_store, {"ledger_analyst": stuck, "policy_compliance": fine},
                          router=_plan_router(_ledger_item(), _policy_item(depends_on_ledger=False)),
                          job_seconds=1).run("x")
    finally:
        release.set()
    assert [r["status"] for r in out["results"]] == ["failed", "completed"]
    assert "TimeoutError" in out["results"][0]["errors"][0]


def test_budget_is_shared_by_a_concurrent_batch_in_plan_order(seeded_store):
    ran = []

    def fine(a):
        ran.append(a.task)
        return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent=a.assigned_agent,
                           status="completed")

    router = _plan_router(_ledger_item("a"), _ledger_item("b"), _ledger_item("c"))
    out = _supervisor(seeded_store, {"ledger_analyst": fine}, router=router, max_dispatches=2).run("x", correlation_id="j_b")
    assert sorted(ran) == ["a", "b"] and out["dispatches"] == 2
    assert [r["status"] for r in out["results"]] == ["completed", "completed", "budget_exhausted"]
    decisions = [e["decision"] for e in seeded_store.audit.events("j_b") if e["action"] == "dispatch"]
    assert decisions == ["info", "info", "deny"]


def test_time_limit_denies_a_whole_batch_before_it_starts(seeded_store):
    stub = _stub()
    clock = [DEMO_AS_OF + timedelta(seconds=400)]
    sup = Supervisor(ALEX, _plan_router(_ledger_item(), _ledger_item()), {"ledger_analyst": stub}, seeded_store.audit,
                     clock=lambda: clock[0])
    state = sup.build().invoke({"correlation_id": "j_t", "request": "x", "started_at": DEMO_AS_OF.isoformat()})
    assert stub.calls == [] and [r["errors"] for r in state["results"]] == [["job time limit"]] * 2


def test_audit_appends_from_two_threads_keep_the_chain_valid():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    audit = AuditLog(conn)

    def write(n):
        for i in range(25):
            audit.append(correlation_id=f"c{n}", actor_id="alex", household_id="hh", action="x", decision="info",
                         details={"i": i})

    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert audit.verify() == 100


@settings(max_examples=25, deadline=None)
@given(st.permutations([0.0, 0.02, 0.04, 0.06]))
def test_results_and_audit_follow_plan_order_for_any_completion_order(delays):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    audit = AuditLog(conn)

    def specialist(a):
        time.sleep(delays[int(a.task)])
        return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent=a.assigned_agent,
                           status="completed", errors=[a.task])

    plan = [DispatchItem(agent="ledger_analyst" if i % 2 == 0 else "policy_compliance", task=str(i),
                         depends_on_ledger=False) for i in range(4)]
    sup = Supervisor(ALEX, _plan_router(*plan), {"ledger_analyst": specialist, "policy_compliance": specialist}, audit,
                     clock=lambda: DEMO_AS_OF)
    out = sup.run("x", correlation_id="j_p")
    assert [r["errors"] for r in out["results"]] == [["0"], ["1"], ["2"], ["3"]]
    logged = [json.loads(e["details_json"])["errors"] for e in audit.events("j_p") if e["action"] == "dispatch"]
    assert logged == [["0"], ["1"], ["2"], ["3"]] and audit.verify() > 0


@pytest.mark.parametrize(("plan", "batches"), [
    ([_ledger_item(), _policy_item()], [1, 1]),                                  # Policy waits for the Ledger
    ([_policy_item(), _ledger_item()], [2]),                                     # Policy first: nothing to wait for
    ([_ledger_item(), _policy_item(depends_on_ledger=False)], [2]),
    ([_ledger_item(), _ledger_item(), _policy_item(), _policy_item()], [2, 1, 1]),
    ([_ledger_item(), _policy_item(depends_on_ledger=False), _policy_item()], [2, 1]),
])
def test_dependency_rule_batches(plan, batches):
    from household_agent.agents import _batch_end
    sizes, i = [], 0
    while i < len(plan):
        end = _batch_end(plan, i)
        sizes.append(end - i)
        i = end
    assert sizes == batches


# --- review fixes: ingest only on request, all calculations, household-bound resume, timeouts -------------------

from hypothesis import HealthCheck  # noqa: E402

from household_agent.agents import INGEST_PURPOSE, LEDGER_READ_TOOLS, job_thread_id  # noqa: E402
from household_agent.authz import Actor  # noqa: E402
from household_agent.models import Action, Role  # noqa: E402
from household_agent.tools import ToolSession  # noqa: E402

RENTAL_CSV = demo_statements()["2026_10_rental_checking.csv"]
ALEX_CSV = demo_statements()["2026_10_alex_checking.csv"]


class Recorder(Scripted):
    """Scripted model that remembers which tools the agent offered it."""

    seen: list[Any] = []

    def bind_tools(self, tools, **kwargs):
        self.seen.append([t["function"]["name"] for t in tools])
        return self


def _documents(store):
    return store.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


@pytest.mark.parametrize("purpose", ["historical_comparison", "tax_preparation", "capital_allocation", "none"])
def test_ingest_tool_is_not_offered_for_read_only_purposes(store, categorizer, purpose):
    model = Recorder(steps=[final()], seen=[])
    LedgerAnalyst(model, store, categorizer, ALEX, clock=lambda: DEMO_AS_OF)(
        assignment(purpose=purpose, inputs={"csv_text": RENTAL_CSV}))
    assert model.seen == [[t["function"]["name"] for t in LEDGER_READ_TOOLS]]
    assert "ingest_statement" not in model.seen[0] and "query_ledger" in model.seen[0]


def test_ingest_tool_is_offered_only_for_the_ingestion_purpose(store, categorizer):
    model = Recorder(steps=[final()], seen=[])
    LedgerAnalyst(model, store, categorizer, ALEX, clock=lambda: DEMO_AS_OF)(
        assignment(purpose=INGEST_PURPOSE, inputs={"csv_text": RENTAL_CSV}))
    assert "ingest_statement" in model.seen[0]


def test_an_injected_ingest_call_during_a_read_only_job_does_nothing(store, categorizer):
    steps = [call("ingest_statement", csv_text=RENTAL_CSV, source_name="fabricated.csv"), final()]
    r = _ledger(store, categorizer, steps)(assignment(inputs={"csv_text": RENTAL_CSV}))
    assert "ingest_statement: invalid_request: tool 'ingest_statement' is not allowed" in r.errors[0]
    assert _documents(store) == 0
    assert all(e["action"] != "upload" for e in store.audit.events("job_t"))


def test_ingest_of_csv_that_differs_from_the_task_is_refused_and_audited(store, categorizer):
    steps = [call("ingest_statement", csv_text=ALEX_CSV, source_name="swapped.csv"), final()]
    r = _ledger(store, categorizer, steps)(assignment(purpose=INGEST_PURPOSE, inputs={"csv_text": RENTAL_CSV}))
    assert "ingest_statement: statement_mismatch" in r.errors[0] and r.status == "completed"
    assert _documents(store) == 0
    [deny] = [e for e in store.audit.events("job_t") if e["action"] == "ingest_statement"]
    assert deny["decision"] == "deny" and "does not match" in deny["reason"] and deny["purpose"] == INGEST_PURPOSE
    assert all(e["action"] != "upload" for e in store.audit.events("job_t"))
    assert store.audit.verify() == 1


def test_ingest_without_a_task_supplied_csv_is_refused_and_audited(store, categorizer):
    steps = [call("ingest_statement", csv_text=RENTAL_CSV, source_name="x.csv"), final()]
    r = _ledger(store, categorizer, steps)(assignment(purpose=INGEST_PURPOSE))
    assert "statement_mismatch" in r.errors[0] and _documents(store) == 0
    assert store.audit.events("job_t")[-1]["reason"] == "the task supplied no statement to ingest"


def test_ingest_of_the_task_supplied_csv_works(store, categorizer):
    steps = [call("ingest_statement", csv_text=RENTAL_CSV, source_name="rental.csv"), final()]
    r = _ledger(store, categorizer, steps)(assignment(purpose=INGEST_PURPOSE, inputs={"csv_text": RENTAL_CSV}))
    assert r.errors == [] and _documents(store) == 1


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.text(max_size=300))
def test_any_csv_other_than_the_supplied_one_is_refused_before_it_is_parsed(store, categorizer, text):
    supplied = ALEX_CSV
    if text == supplied:
        return
    session = ToolSession(store, categorizer, ALEX, INGEST_PURPOSE, "c_prop", restrict_ingest=True,
                          ingest_csv_sha256=hashlib.sha256(supplied.encode()).hexdigest())
    out = session.call("ingest_statement", {"csv_text": text, "source_name": "x.csv"})
    assert out["ok"] is False and out["error"]["type"] == "statement_mismatch" and _documents(store) == 0


def test_a_host_session_without_the_restriction_still_ingests(store, categorizer):
    out = ToolSession(store, categorizer, ALEX, INGEST_PURPOSE, "c_host").call(
        "ingest_statement", {"csv_text": RENTAL_CSV, "source_name": "r.csv"})
    assert out["ok"] is True


def test_the_model_router_cannot_choose_statement_ingestion():
    reply = AIMessage(content=json.dumps({"intent": "statement_ingestion", "dispatch_plan": [
        {"agent": "ledger_analyst", "task": "x", "inputs": {"csv_text": "fabricated"}}]}))
    decision, source = Router(Scripted(steps=[reply]))("Is my October spend unusual?")
    assert decision.intent != "statement_ingestion" and source == "rules_fallback"


# --- every calculation counts ------------------------------------------------------------

def _two_run_script():
    """Query the property-tax payments (16.36% variance), run it, then also run category_totals on the same IDs."""
    return [
        call("query_ledger", account_ids=[RENTAL], start="2025-10-01", end="2026-10-31", merchant=TAX),
        lambda m: call("run_calculation", fn="variance_between_transactions", transaction_ids=_txn_ids(m)),
        lambda m: call("run_calculation", fn="category_totals", transaction_ids=_txn_ids(m),
                       params={"start": "2025-10-01", "end": "2026-10-31"}),
        final(),
    ]


def _one_job(seeded_store, specialist, **kw):
    decision = RoutingDecision(intent="historical_comparison",
                               dispatch_plan=[DispatchItem(agent="ledger_analyst", task="x")])
    return Supervisor(ALEX, lambda _r: (decision, "rules"), {"ledger_analyst": specialist}, seeded_store.audit,
                      clock=lambda: DEMO_AS_OF, members=MEMBERS, authorizer=seeded_store.authorizer, **kw)


def test_a_later_calculation_does_not_hide_an_earlier_variance_finding(seeded_store, categorizer):
    r = _ledger(seeded_store, categorizer, _two_run_script())(assignment())
    assert [c["calc_run_id"] for c in r.outputs["calculations"]][0] != r.outputs["calc_run_id"]
    assert "pct_variance" in r.outputs["calculations"][0] and "pct_variance" not in r.outputs["calculations"][1]
    out = _one_job(seeded_store, _ledger(seeded_store, categorizer, _two_run_script())).run("x", correlation_id="j_two")
    [f] = out["findings"]
    assert f["type"] == "material_variance" and f["pct_variance"] == "16.36" and f["threshold_pct"] == "10"
    assert f["calc_run_id"] == out["results"][0]["outputs"]["calculations"][0]["calc_run_id"]
    assert out["status"] == "needs_review"


def test_every_run_is_handed_to_policy_as_calc_outputs(seeded_store, categorizer):
    seen = {}

    def policy(a):
        seen.update(a.inputs["calc_outputs"])
        return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent="policy_compliance",
                           status="completed")

    decision = RoutingDecision(intent="historical_comparison_and_tax_context",
                               dispatch_plan=[_ledger_item(), _policy_item()])
    _supervisor(seeded_store, {"ledger_analyst": _ledger(seeded_store, categorizer, _two_run_script()),
                               "policy_compliance": policy},
                router=lambda _r: (decision, "t")).run("x")
    assert len(seen) == 2 and all("calc_run_id" not in v for v in seen.values())


def _two_account_script():
    return [
        call("query_ledger", account_ids=[RENTAL], start="2025-10-01", end="2026-10-31", merchant=TAX),
        call("query_ledger", account_ids=[PRIVATE], start="2026-10-01", end="2026-10-31"),
        lambda m: call("run_calculation", fn="variance_between_transactions",
                       transaction_ids=[t["transaction_id"] for t in tool_results(m, "query_ledger")[0]["transactions"]]),
        final(),
    ]


def test_reviewer_scope_covers_every_account_the_job_read_not_just_the_last_run(seeded_store, categorizer):
    r = _ledger(seeded_store, categorizer, _two_account_script())(assignment())
    assert r.outputs["account_ids"] == sorted([RENTAL, PRIVATE]) and r.outputs["calculations"][0]["account_ids"] == [RENTAL]
    sup = _one_job(seeded_store, _ledger(seeded_store, categorizer, _two_account_script()))
    saver = InMemorySaver()
    out = sup.run("x", correlation_id="j_scope", checkpointer=saver)
    assert out["status"] == "awaiting_review"
    denied = sup.resume("j_scope", {"reviewer": "jordan", "accepted": True}, checkpointer=saver)
    assert denied["status"] == "awaiting_review" and denied["review_denied_attempts"] == 1
    assert sup.resume("j_scope", {"reviewer": "alex", "accepted": True}, checkpointer=saver)["status"] == "reviewed"


def test_a_query_only_job_still_reports_the_accounts_it_read(seeded_store, categorizer):
    steps = [call("query_ledger", account_ids=[PRIVATE], start="2026-10-01", end="2026-10-31"), final()]
    r = _ledger(seeded_store, categorizer, steps)(assignment())
    assert r.outputs["account_ids"] == [PRIVATE] and set(r.outputs) == {"account_ids", "evidence_documents"}


def test_a_job_that_read_nothing_has_no_outputs(seeded_store, categorizer):
    assert _ledger(seeded_store, categorizer, [final()])(assignment()).outputs == {}


@settings(max_examples=30, deadline=None)
@given(st.lists(st.tuples(st.decimals(min_value=-100, max_value=100, places=2),
                          st.sampled_from(["property_tax", "groceries", None])), min_size=1, max_size=5))
def test_every_calculation_above_its_threshold_is_flagged_whatever_its_position(runs):
    conn = sqlite3.connect(":memory:")
    calcs = [{"calc_run_id": f"calc_{i}", "pct_variance": str(p), **({"category": c} if c else {})}
             for i, (p, c) in enumerate(runs)]
    stub = _stub(outputs={**calcs[-1], "calculations": calcs})
    decision = RoutingDecision(intent="historical_comparison",
                               dispatch_plan=[DispatchItem(agent="ledger_analyst", task="x")])
    out = Supervisor(ALEX, lambda _r: (decision, "rules"), {"ledger_analyst": stub}, AuditLog(conn),
                     clock=lambda: DEMO_AS_OF).run("x")
    expected = [f"calc_{i}" for i, (p, c) in enumerate(runs) if abs(p) > variance_threshold(c)]
    assert [f["calc_run_id"] for f in out["findings"]] == expected
    conn.close()


# --- resume is bound to the job's household -----------------------------------------------

EVE = Actor("eve", "hh_other", Role.ADULT_MEMBER)


def _paused(seeded_store, cid="j_hh"):
    sup, _ = _review_job(seeded_store)
    saver = InMemorySaver()
    out = sup.run("x", correlation_id=cid, checkpointer=saver)
    assert out["status"] == "awaiting_review" and out["household_id"] == "hh_demo" and out["requester"] == "alex"
    return sup, saver


def _eve_supervisor(seeded_store):
    return Supervisor(EVE, lambda _r: (RoutingDecision(intent="out_of_scope"), "t"), {}, seeded_store.audit,
                      clock=lambda: DEMO_AS_OF, members={**MEMBERS, "eve": EVE}, authorizer=seeded_store.authorizer)


def test_an_adult_in_another_household_cannot_resume_and_the_refusal_is_audited_under_the_jobs_household(seeded_store):
    sup, saver = _paused(seeded_store)
    out = _eve_supervisor(seeded_store).resume("j_hh", {"reviewer": "eve", "accepted": True}, checkpointer=saver,
                                               household_id="hh_demo")
    assert out["status"] == "awaiting_review" and "review" not in out
    [deny] = [e for e in seeded_store.audit.events("j_hh") if e["action"] == "resume"]
    assert deny["decision"] == "deny" and deny["household_id"] == "hh_demo" and deny["actor_id"] == "eve"
    assert "hh_other" in deny["reason"]
    assert sup.resume("j_hh", {"reviewer": "alex", "accepted": True}, checkpointer=saver)["status"] == "reviewed"


def test_another_household_does_not_even_find_the_job_by_its_id(seeded_store):
    _, saver = _paused(seeded_store, "j_hh2")
    out = _eve_supervisor(seeded_store).resume("j_hh2", {"reviewer": "eve", "accepted": True}, checkpointer=saver)
    assert out["status"] == "not_found"
    [deny] = [e for e in seeded_store.audit.events("j_hh2") if e["action"] == "resume"]
    assert deny["household_id"] == "hh_other" and deny["decision"] == "deny"
    assert saver.get_tuple({"configurable": {"thread_id": job_thread_id("hh_demo", "j_hh2")}}) is not None
    assert saver.get_tuple({"configurable": {"thread_id": "j_hh2"}}) is None


def test_a_reviewer_from_another_household_is_refused_even_through_the_right_supervisor(seeded_store):
    sup, saver = _paused(seeded_store, "j_hh3")
    sup.members = {**sup.members, "eve": EVE}
    out = sup.resume("j_hh3", {"reviewer": "eve", "accepted": True}, checkpointer=saver)
    assert out["status"] == "awaiting_review" and out["review_denied_attempts"] == 1


@given(st.text(max_size=8), st.text(max_size=8), st.text(max_size=8), st.text(max_size=8))
def test_job_thread_ids_never_collide_across_households(h1, c1, h2, c2):
    assert (job_thread_id(h1, c1) == job_thread_id(h2, c2)) == ((h1, c1) == (h2, c2))


# --- timeouts reach the tools -------------------------------------------------------------

def _slow_ledger(seeded_store, categorizer, gate, done):
    model = Scripted(steps=[lambda m: (gate.wait(10), call("query_ledger", account_ids=[RENTAL], start="2026-10-01",
                                                          end="2026-10-31"))[1], final()])
    analyst = _ledger(seeded_store, categorizer, [])
    analyst.model = model

    def run(a):
        try:
            return analyst(a)
        finally:
            done.set()
    return run


def _reads(audit, cid):
    return [e for e in audit.events(cid) if e["action"] == Action.READ.value]


@pytest.mark.parametrize("concurrent", [False, True])
def test_a_specialist_that_wakes_up_after_the_time_limit_can_no_longer_use_tools(seeded_store, categorizer, concurrent):
    gate, done = threading.Event(), threading.Event()
    slow = _slow_ledger(seeded_store, categorizer, gate, done)
    plan = [_ledger_item()] + ([_policy_item(depends_on_ledger=False)] if concurrent else [])
    specialists = {"ledger_analyst": slow, "policy_compliance": _stub()}
    sup = _supervisor(seeded_store, specialists, router=_plan_router(*plan), job_seconds=1)
    if concurrent:
        out = sup.run("x", correlation_id="j_slow")
        gate.set()  # the abandoned thread wakes up only now
    else:
        threading.Timer(1.5, gate.set).start()  # a lone specialist runs inline, so wake it from outside
        out = sup.run("x", correlation_id="j_slow")
    assert done.wait(10)
    assert out["results"][0]["status"] == "failed" and "TimeoutError" in out["results"][0]["errors"][0]
    assert _reads(seeded_store.audit, "j_slow") == []  # the late query_ledger call was refused, not run
    assert seeded_store.audit.verify() > 0


def test_a_fast_specialist_in_a_single_item_batch_is_unaffected_by_the_time_limit(seeded_store, categorizer):
    out = _supervisor(seeded_store, {"ledger_analyst": _ledger(seeded_store, categorizer, ledger_script())},
                      job_seconds=30).run(TRACE_QUERY, correlation_id="j_fast")
    assert out["results"][0]["status"] == "completed" and len(_reads(seeded_store.audit, "j_fast")) == 1


def test_tool_session_refuses_every_call_once_cancelled(store, categorizer):
    cancel = threading.Event()
    session = ToolSession(store, categorizer, ALEX, "historical_comparison", "c_cancel", cancel=cancel)
    assert session.call("list_accounts")["ok"] is True
    cancel.set()
    for name, args in [("list_accounts", {}), ("query_ledger", {"account_ids": [RENTAL], "start": "2026-10-01",
                                                                "end": "2026-10-31"})]:
        out = session.call(name, args)
        assert out["ok"] is False and out["error"]["type"] == "job_cancelled"
    assert store.audit.events("c_cancel") == []


# --- a result must be this task's, from this agent ------------------------------------------

@pytest.mark.parametrize(("field", "value"), [("task_id", "task_other"), ("agent", "policy_compliance")])
def test_result_with_another_task_id_or_agent_is_discarded(seeded_store, field, value):
    stub = _stub(outputs={"pct_variance": "99", "calc_run_id": "c"})
    wrong = lambda a: stub(a).model_copy(update={field: value})  # noqa: E731
    out = _supervisor(seeded_store, {"ledger_analyst": wrong}).run("Is my October spend unusual?", correlation_id="j_w")
    [r] = out["results"]
    assert r["status"] == "failed" and r["outputs"] == {} and f"{field} mismatch" in r["errors"][0]
    assert out["findings"] == []
