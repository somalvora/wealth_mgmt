"""Failed reconciliation routes to human data review (design §4.1/§4.2, §5.3).

An ``unreconciled``/``unverified`` ingest result becomes a ``data_quality`` finding, which forces
the review interrupt (checkpointer only) and flags the briefing. No figures come from that document.
"""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from langgraph.checkpoint.memory import InMemorySaver

from household_agent.agents import (
    AgentResult,
    DispatchItem,
    LedgerAnalyst,
    RoutingDecision,
    Supervisor,
    compose_briefing,
    data_quality_entries,
)
from household_agent.mcp_server import MEMBERS
from household_agent.synthetic import ALEX, DEMO_AS_OF, demo_statements
from test_agents import Scripted, call, final

CORRUPT = "2026_09_alex_checking_corrupt.csv"


def _job(store, categorizer, steps, actor=ALEX, csv_text=demo_statements()[CORRUPT]):
    ledger = LedgerAnalyst(Scripted(steps=steps), store, categorizer, actor, clock=lambda: DEMO_AS_OF)
    decision = RoutingDecision(intent="statement_ingestion",  # ingest_statement is offered only for this intent
                               dispatch_plan=[DispatchItem(agent="ledger_analyst", task="ingest",
                                                           inputs={"csv_text": csv_text})])
    return Supervisor(ALEX, lambda _r: (decision, "rules"), {"ledger_analyst": ledger}, store.audit,
                      clock=lambda: DEMO_AS_OF, members=MEMBERS, authorizer=store.authorizer)


def _ingest_corrupt(claim=None):
    steps = [call("ingest_statement", csv_text=demo_statements()[CORRUPT], source_name=CORRUPT)]
    steps.append(lambda m: final(claim) if claim else final())
    return steps


def test_unreconciled_ingest_is_routed_to_review_and_resolved_by_an_adult(store, categorizer):
    sup = _job(store, categorizer, _ingest_corrupt())
    saver = InMemorySaver()
    out = sup.run("ingest", correlation_id="j_dq", checkpointer=saver)
    [f] = out["findings"]
    assert f["type"] == "data_quality" and f["status"] == "unreconciled" and f["source_name"] == CORRUPT
    assert f["document_id"] and f["issue_codes"] and out["status"] == "awaiting_review" and out["__interrupt__"]
    done = sup.resume("j_dq", {"reviewer": "jordan", "accepted": True, "note": "ask for a new statement"},
                      checkpointer=saver)
    b = done["briefing"]
    assert done["status"] == "reviewed" and b["needs_review"] is True and b["review"]["reviewer"] == "jordan"
    [alert] = b["alerts"]
    assert CORRUPT in alert["text"] and "No figures were produced" in alert["text"]
    assert b["statements"] == [] and alert["issue_codes"] == f["issue_codes"]
    assert store.audit.verify() > 0


@pytest.mark.parametrize("reviewer", ["sam", "mallory"])
def test_dependent_or_unknown_resolver_is_refused_and_audited(store, categorizer, reviewer):
    sup = _job(store, categorizer, _ingest_corrupt())
    saver = InMemorySaver()
    sup.run("ingest", correlation_id="j_dq2", checkpointer=saver)
    out = sup.resume("j_dq2", {"reviewer": reviewer, "accepted": True}, checkpointer=saver)
    assert out["status"] == "awaiting_review" and out["review_denied_attempts"] == 1
    assert [(e["actor_id"], e["decision"]) for e in store.audit.events("j_dq2") if e["action"] == "review"] == [
        (reviewer, "deny")]


def test_reconciled_document_produces_no_finding_and_no_pause(store, categorizer):
    csv_text = demo_statements()["2026_10_alex_checking.csv"]
    steps = [call("ingest_statement", csv_text=csv_text, source_name="ok.csv"), final()]
    out = _job(store, categorizer, steps, csv_text=csv_text).run("ingest", correlation_id="j_ok", checkpointer=InMemorySaver())
    assert out["findings"] == [] and "__interrupt__" not in out and out["briefing"]["alerts"] == []


def test_a_claim_citing_figures_from_the_unreconciled_document_is_removed(store, categorizer):
    claim = {"text": "September closing balance is $9,999.99.", "evidence_ids": ["doc_placeholder"]}
    r = _job(store, categorizer, _ingest_corrupt(claim)).specialists["ledger_analyst"](
        _assignment())
    assert r.claims == [] and any(e.startswith("claim removed") for e in r.errors)
    assert r.outputs["data_quality"][0]["status"] == "unreconciled"


def test_briefing_drops_a_statement_that_cites_a_flagged_document():
    findings = [{"type": "data_quality", "document_id": "doc_x", "source_name": "bad.csv", "status": "unreconciled",
                 "issue_codes": ["closing_balance_mismatch"]}]
    ok = AgentResult(task_id="t", correlation_id="c", agent="ledger_analyst", status="completed",
                     claims=[{"text": "x $1.00", "evidence_ids": ["doc_x"]}, {"text": "y $2.00", "evidence_ids": ["e"]}])
    state = {"routing": {"intent": "historical_comparison"}, "results": [ok.model_dump(mode="json")],
             "findings": findings}
    assert [s["text"] for s in compose_briefing(state)["statements"]] == ["y $2.00"]


def test_without_a_checkpointer_the_finding_is_reported_but_nothing_pauses(store, categorizer):
    out = _job(store, categorizer, _ingest_corrupt()).run("ingest", correlation_id="j_plain")
    assert "__interrupt__" not in out and out["status"] == "needs_review"
    assert out["findings"][0]["type"] == "data_quality" and out["briefing"]["alerts"]
    assert all(e["action"] != "review" for e in store.audit.events("j_plain"))


def test_unreconciled_errors_from_query_or_calculation_become_entries():
    obs = [{"tool": "run_calculation", "args": {}, "result": {"ok": False, "error": {
        "type": "ineligible_input", "message": "txn_1 is unreconciled"}}},
           {"tool": "query_ledger", "args": {}, "result": {"ok": False, "error": {
               "type": "invalid_request", "message": "bad date"}}}]
    [e] = data_quality_entries(obs)
    assert e["source"] == "run_calculation" and e["status"] == "unreconciled"


def _assignment():
    from household_agent.agents import TaskAssignment
    return TaskAssignment(task_id="task_t", correlation_id="job_t", assigned_agent="ledger_analyst",
                          purpose="statement_ingestion", authorized_scope={"household_id": "hh_demo"}, task="t",
                          inputs={"csv_text": demo_statements()[CORRUPT]})


@settings(max_examples=60, deadline=None)
@given(st.lists(st.sampled_from(["reconciled", "qualified", "unreconciled", "unverified"]), max_size=8))
def test_findings_exist_exactly_for_unreconciled_or_unverified_ingests(kinds):
    obs = [{"tool": "ingest_statement", "args": {"source_name": f"s{i}.csv"},
            "result": {"ok": True, "document_id": f"doc_{i}", "document_confidence": 0.5,
                       "status": k if k != "unverified" else "reconciled", "unverified": k == "unverified",
                       "issues": [{"code": "c", "severity": "material", "message": "m"}]}}
           for i, k in enumerate(kinds)]
    flagged = [e["document_id"] for e in data_quality_entries(obs)]
    assert flagged == [f"doc_{i}" for i, k in enumerate(kinds) if k in ("unreconciled", "unverified")]
