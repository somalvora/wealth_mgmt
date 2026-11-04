import json
from datetime import timedelta
from typing import Any

import chromadb
import pytest
from chromadb.config import Settings
from hypothesis import given
from hypothesis import strategies as st
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from household_agent.agents import (
    MAX_DISPATCHES,
    AgentResult,
    Budget,
    LedgerAnalyst,
    PolicyCompliance,
    Router,
    RoutingDecision,
    Supervisor,
    TaskAssignment,
    rule_route,
)
from household_agent.approval import build_approval_graph
from household_agent.llm import HashEmbeddings
from household_agent.mcp_server import MEMBERS
from household_agent.retrieval import CANNOT_VERIFY, RetrievalService
from household_agent.synthetic import ALEX, DEMO_AS_OF, JORDAN, demo_planning_inputs, demo_policy_documents

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
    assert policy.claims[0].text == "Monthly rent for Unit B is $3,400.00."
    assert policy.questions == [CANNOT_VERIFY]
    b = out["briefing"]
    assert [s["text"] for s in b["statements"]] == [ledger.claims[0].text, policy.claims[0].text]
    assert b["findings"][0]["type"] == "material_variance" and b["findings"][0]["pct_variance"] == "16.36"
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
    steps = [call("ingest_statement", csv_text="x", source_name="x"), call("retrieve_verified", query="q"), final()]
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
        return AgentResult(task_id=a.task_id, agent=a.assigned_agent, status=status, **kw)

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
        return AgentResult(task_id=a.task_id, agent=a.assigned_agent, status="completed")

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
