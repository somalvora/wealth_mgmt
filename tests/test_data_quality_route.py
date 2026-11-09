"""Failed reconciliation routes to human data review (design §4.1/§4.2, §5.3).

An ``unreconciled``/``unverified`` ingest result becomes a ``data_quality`` finding, which forces
the review interrupt (checkpointer only) and flags the briefing. No figures come from that document.
"""
import json

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
from household_agent.tools import ToolSession
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
    assert CORRUPT in alert["text"] and "left out of this briefing" in alert["text"]
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


def test_without_a_checkpointer_the_finding_is_reported_but_nothing_pauses(store, categorizer):
    out = _job(store, categorizer, _ingest_corrupt()).run("ingest", correlation_id="j_plain")
    assert "__interrupt__" not in out and out["status"] == "needs_review"
    assert out["findings"][0]["type"] == "data_quality" and out["briefing"]["alerts"]
    assert all(e["action"] != "review" for e in store.audit.events("j_plain"))


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


# --- queries and calculations report what they left out (M3) and the briefing drops derived figures (M4) -----------

PRIVATE = "acct_alex_checking"
RENTAL = "acct_rental_checking"
TAX = "San Mateo County Tax Collector"


def _tools(store, categorizer, actor=ALEX):
    return ToolSession(store, categorizer, actor, "historical_comparison", "job_tools", clock=lambda: DEMO_AS_OF)


def _corrupt_doc_id(store):
    [doc] = [d for d in store.document_coverage(ALEX, "historical_comparison", "c", [PRIVATE], DEMO_AS_OF.date().replace(month=9, day=1),
                                                DEMO_AS_OF.date().replace(month=9, day=30))
             if d["status"] == "unreconciled"]
    return doc["document_id"]


def _read_job(store, categorizer, steps):
    ledger = LedgerAnalyst(Scripted(steps=steps), store, categorizer, ALEX, clock=lambda: DEMO_AS_OF)
    decision = RoutingDecision(intent="historical_comparison",
                               dispatch_plan=[DispatchItem(agent="ledger_analyst", task="read")])
    return Supervisor(ALEX, lambda _r: (decision, "rules"), {"ledger_analyst": ledger}, store.audit,
                      clock=lambda: DEMO_AS_OF, members=MEMBERS, authorizer=store.authorizer)


def test_query_ledger_lists_the_unreconciled_documents_it_left_out(seeded_store, categorizer):
    wide = _tools(seeded_store, categorizer).call("query_ledger", {"account_ids": [PRIVATE], "start": "2026-09-01",
                                                                   "end": "2026-10-31"})
    doc_id = _corrupt_doc_id(seeded_store)
    [entry] = wide["excluded"]["unreconciled"]
    assert entry["document_id"] == doc_id and entry["status"] == "unreconciled" and entry["issue_codes"]
    assert entry["rows_in_range"] > 0 and wide["excluded_rows"] == entry["rows_in_range"]
    assert wide["excluded"]["unverified"] == [] and all(t["source_document_id"] != doc_id for t in wide["transactions"])
    narrow = _tools(seeded_store, categorizer).call("query_ledger", {"account_ids": [PRIVATE], "start": "2026-10-01",
                                                                     "end": "2026-10-31"})
    assert narrow["excluded"] == {"unreconciled": [], "unverified": [], "qualified": []} and narrow["excluded_rows"] == 0


def test_run_calculation_repeats_what_its_inputs_query_left_out(seeded_store, categorizer):
    t = _tools(seeded_store, categorizer)
    q = t.call("query_ledger", {"account_ids": [PRIVATE], "start": "2026-09-01", "end": "2026-10-31"})
    calc = t.call("run_calculation", {"fn": "cash_flow", "transaction_ids": [x["transaction_id"] for x in q["transactions"]], "params": {"start": "2026-09-01", "end": "2026-10-31"}})
    assert calc["ok"] and [d["document_id"] for d in calc["excluded"]["unreconciled"]] == [_corrupt_doc_id(seeded_store)]
    t = _tools(seeded_store, categorizer)  # a session that only ever queried a range the corrupt statement does not touch
    other = t.call("query_ledger", {"account_ids": [PRIVATE], "start": "2026-10-01", "end": "2026-10-31"})
    calc2 = t.call("run_calculation", {"fn": "cash_flow", "transaction_ids": [x["transaction_id"] for x in other["transactions"]], "params": {"start": "2026-10-01", "end": "2026-10-31"}})
    assert calc2["excluded"]["unreconciled"] == []  # only what the queries behind its inputs left out


def test_a_query_over_a_corrupt_statement_becomes_a_finding_a_review_and_an_alert(seeded_store, categorizer):
    steps = [call("query_ledger", account_ids=[PRIVATE], start="2026-09-01", end="2026-10-31"), final()]
    sup, saver = _read_job(seeded_store, categorizer, steps), InMemorySaver()
    out = sup.run("read", correlation_id="j_q", checkpointer=saver)
    [f] = out["findings"]
    assert f["type"] == "data_quality" and f["status"] == "unreconciled" and f["source"] == "query_ledger"
    assert f["document_id"] == _corrupt_doc_id(seeded_store) and f["source_name"] == CORRUPT
    assert out["status"] == "awaiting_review" and out["__interrupt__"]
    done = sup.resume("j_q", {"reviewer": "alex", "accepted": True}, checkpointer=saver)
    b = done["briefing"]
    assert b["needs_review"] is True and [a["document_id"] for a in b["alerts"]] == [f["document_id"]]
    assert CORRUPT in b["alerts"][0]["text"]


def test_a_query_range_that_does_not_touch_the_corrupt_statement_has_no_finding(seeded_store, categorizer):
    steps = [call("query_ledger", account_ids=[PRIVATE], start="2026-10-01", end="2026-10-31"), final()]
    out = _read_job(seeded_store, categorizer, steps).run("read", correlation_id="j_q2", checkpointer=InMemorySaver())
    assert out["findings"] == [] and "__interrupt__" not in out and out["briefing"]["alerts"] == []
    assert out["status"] == "completed"


def _flag_unverified(store, doc_id):
    # Test-only: documents are append-only, so lift that trigger on this throwaway in-memory database to simulate a
    # statement that came out of reconciliation with low confidence.
    store.conn.execute("DROP TRIGGER documents_no_update")
    store.conn.execute("UPDATE documents SET unverified = 1, document_confidence = 0.5 WHERE document_id = ?", (doc_id,))


def _rental_variance_steps(include_provisional):
    return [
        call("query_ledger", account_ids=[RENTAL], start="2025-10-01", end="2026-10-31", merchant=TAX,
             include_provisional=include_provisional),
        lambda m: call("run_calculation", fn="variance_between_transactions", transaction_ids=_ids(m)),
        lambda m: final({"text": "October 2026 payment is $450.00 higher than October 2025.",
                         "evidence_ids": [_results(m, "run_calculation")[-1]["run_id"]]})]


def _results(messages, name):
    from langchain_core.messages import ToolMessage
    return [json.loads(m.content) for m in messages if isinstance(m, ToolMessage) and m.name == name]


def _ids(messages):
    return [t["transaction_id"] for r in _results(messages, "query_ledger") for t in r.get("transactions", [])]


def test_a_claim_citing_a_calculation_run_built_on_a_flagged_document_is_dropped(seeded_store, categorizer):
    """Real path: the claim cites the calc run ID (not a document ID), whose inputs come from the flagged statement."""
    control = _read_job(seeded_store, categorizer, _rental_variance_steps(True)).run("read", correlation_id="j_ctl")
    [kept] = control["briefing"]["statements"]  # unflagged: the claim reaches the briefing
    assert kept["evidence_ids"][0].startswith("calc_run_") and not any(f["type"] == "data_quality" for f in control["findings"])
    [oct_doc] = [d for d in seeded_store.document_coverage(ALEX, "historical_comparison", "c", [RENTAL], DEMO_AS_OF.date().replace(month=10, day=1),
                                                           DEMO_AS_OF.date().replace(month=10, day=31))
                 if d["source_name"] == "2026_10_rental_checking.csv"]
    _flag_unverified(seeded_store, oct_doc["document_id"])
    out = _read_job(seeded_store, categorizer, _rental_variance_steps(True)).run("read", correlation_id="j_flag")
    [f] = [f for f in out["findings"] if f["type"] == "data_quality"]
    assert f["document_id"] == oct_doc["document_id"] and f["status"] == "unverified" and f["source"] == "query_ledger"
    b = out["briefing"]
    assert b["statements"] == [] and b["needs_review"] is True and "left out of this briefing" in b["alerts"][0]["text"]
    assert out["results"][0]["claims"], "the specialist still produced the claim; it is the briefing that excludes it"


def test_without_include_provisional_an_unverified_statement_is_left_out_and_reported(seeded_store, categorizer):
    doc = next(d for d in seeded_store.document_coverage(ALEX, "historical_comparison", "c", [RENTAL], DEMO_AS_OF.date().replace(month=10, day=1),
                                                         DEMO_AS_OF.date().replace(month=10, day=31))
               if d["source_name"] == "2026_10_rental_checking.csv")
    _flag_unverified(seeded_store, doc["document_id"])
    r = _tools(seeded_store, categorizer).call("query_ledger", {"account_ids": [RENTAL], "start": "2026-10-01", "end": "2026-10-31"})
    assert [d["document_id"] for d in r["excluded"]["unverified"]] == [doc["document_id"]] and r["unverified_included"] == []
    r = _tools(seeded_store, categorizer).call("query_ledger", {"account_ids": [RENTAL], "start": "2026-10-01", "end": "2026-10-31",
                                                                "include_provisional": True})
    assert r["excluded"]["unverified"] == [] and [d["document_id"] for d in r["unverified_included"]] == [doc["document_id"]]


def test_briefing_drops_a_claim_whose_calc_run_traces_to_a_flagged_document():
    findings = [{"type": "data_quality", "document_id": "doc_x", "source_name": "bad.csv", "status": "unverified",
                 "issue_codes": []}]
    ledger = AgentResult(task_id="t", correlation_id="c", agent="ledger_analyst", status="completed",
                         outputs={"evidence_documents": {"calc_run_1": ["doc_x"], "calc_run_2": ["doc_y"],
                                                         "txn_1": ["doc_x", "doc_y"]}},
                         claims=[{"text": "from x $1.00", "evidence_ids": ["calc_run_1"]},
                                 {"text": "from y $2.00", "evidence_ids": ["calc_run_2"]},
                                 {"text": "mixed $3.00", "evidence_ids": ["calc_run_2", "txn_1"]}])
    state = {"routing": {"intent": "historical_comparison"}, "results": [ledger.model_dump(mode="json")],
             "findings": findings}
    assert [s["text"] for s in compose_briefing(state)["statements"]] == ["from y $2.00"]
    assert [s["text"] for s in compose_briefing({**state, "findings": []})["statements"]] == [
        "from x $1.00", "from y $2.00", "mixed $3.00"]


def test_entries_come_from_the_structured_excluded_field_not_from_error_text():
    doc = {"document_id": "doc_1", "source_name": "a.csv", "issue_codes": ["c"], "document_confidence": 0.4}
    ok = {"tool": "query_ledger", "args": {}, "result": {"ok": True, "transactions": [], "excluded": {
        "unreconciled": [doc], "unverified": [], "qualified": []}, "unverified_included": []}}
    text_only = {"tool": "run_calculation", "args": {}, "result": {"ok": False, "error": {
        "type": "ineligible_input", "message": "txn_1 is unreconciled"}}}
    [e] = data_quality_entries([ok, text_only])
    assert e["document_id"] == "doc_1" and e["status"] == "unreconciled" and e["source"] == "query_ledger"
    assert data_quality_entries([text_only]) == []  # wording alone no longer creates an entry


_KINDS = st.sampled_from(["reconciled", "qualified", "unreconciled", "unverified", "unverified_included"])


@settings(max_examples=80, deadline=None)
@given(st.lists(_KINDS, max_size=8), st.booleans())
def test_query_entries_exist_exactly_for_unreconciled_or_unverified_documents(kinds, repeat):
    def doc(i):
        return {"document_id": f"doc_{i}", "source_name": f"s{i}.csv", "issue_codes": [], "document_confidence": 0.5}
    excluded = {"unreconciled": [], "unverified": [], "qualified": []}
    included = []
    for i, k in enumerate(kinds):
        if k in excluded:
            excluded[k].append(doc(i))
        elif k == "unverified_included":
            included.append(doc(i))
    obs = [{"tool": "query_ledger", "args": {}, "result": {"ok": True, "excluded": excluded, "unverified_included": included}}]
    if repeat:  # the same documents reported again by the calculation do not create a second entry
        obs.append({"tool": "run_calculation", "args": {}, "result": {"ok": True, "excluded": excluded,
                                                                      "unverified_included": included}})
    got = {e["document_id"]: e["status"] for e in data_quality_entries(obs)}
    assert len(got) == len(data_quality_entries(obs))
    assert got == {f"doc_{i}": ("unreconciled" if k == "unreconciled" else "unverified")
                   for i, k in enumerate(kinds) if k in ("unreconciled", "unverified", "unverified_included")}
