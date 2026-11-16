import sqlite3
from dataclasses import replace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from household_agent.approval import build_approval_graph, start_request
from household_agent.audit import AuditLog
from household_agent.synthetic import SOMAL, CASEY_CPA, HOUSEHOLD, SAYALI, VIAAN, demo_planning_inputs

MEMBERS = {a.member_id: a for a in (SOMAL, SAYALI, CASEY_CPA, VIAAN)}


@pytest.fixture
def audit(conn):
    return AuditLog(conn)


def _graph(audit, saver, inputs=None):
    box = {"inputs": inputs or demo_planning_inputs()}
    graph = build_approval_graph(inputs_for=lambda hh: box["inputs"], members=MEMBERS, audit=audit, checkpointer=saver)
    return graph, box


def _resume(graph, thread_id, **answer):
    return graph.invoke(Command(resume=answer), {"configurable": {"thread_id": thread_id}})


def test_plan_pauses_for_approval(audit):
    graph, _ = _graph(audit, InMemorySaver())
    thread_id, out = start_request(graph, HOUSEHOLD, "somal")
    [intr] = out["__interrupt__"]
    assert intr.value["kind"] == "allocation_approval"
    assert intr.value["plan"]["best_node_id"] == "d1.balanced/d2.tax_aggressive/d3.tax_aggressive"
    assert sum(intr.value["plan"]["allocation_minor"].values()) == demo_planning_inputs().surplus_minor
    assert out["status"] == "awaiting_approval" and "decision" not in out
    assert graph.get_state({"configurable": {"thread_id": thread_id}}).next == ("approval",)


def test_approve_and_reject(audit):
    graph, _ = _graph(audit, InMemorySaver())
    t1, _ = start_request(graph, HOUSEHOLD, "somal")
    assert _resume(graph, t1, approver="sayali", approved=True, note="ok")["status"] == "approved"
    t2, _ = start_request(graph, HOUSEHOLD, "somal")
    out = _resume(graph, t2, approver="somal", approved=False)
    assert out["status"] == "rejected" and out["decision"]["approver"] == "somal"
    approvals = [e for e in audit.events(t1) if e["action"] == "approve"]
    assert [e["decision"] for e in approvals] == ["allow"]


def test_only_truthy_true_counts_as_approval(audit):
    graph, _ = _graph(audit, InMemorySaver())
    t, _ = start_request(graph, HOUSEHOLD, "somal")
    assert _resume(graph, t, approver="somal", approved="yes")["status"] == "rejected"


@pytest.mark.parametrize("approver", ["casey_cpa", "viaan", "mallory", ""])
def test_non_approvers_denied_and_graph_keeps_waiting(audit, approver):
    graph, _ = _graph(audit, InMemorySaver())
    t, _ = start_request(graph, HOUSEHOLD, "somal")
    out = _resume(graph, t, approver=approver, approved=True)
    assert out["status"] == "awaiting_approval" and out["denied_attempts"] == 1 and out["__interrupt__"]
    denies = [e for e in audit.events(t) if e["action"] == "approve"]
    assert [e["decision"] for e in denies] == ["deny"]
    assert _resume(graph, t, approver="sayali", approved=True)["status"] == "approved"


def test_other_household_cannot_approve(audit):
    outsider = replace(SAYALI, member_id="pat", household_id="hh_other")
    graph = build_approval_graph(inputs_for=lambda hh: demo_planning_inputs(), members={**MEMBERS, "pat": outsider},
                                 audit=audit, checkpointer=InMemorySaver())
    t, _ = start_request(graph, HOUSEHOLD, "somal")
    assert _resume(graph, t, approver="pat", approved=True)["status"] == "awaiting_approval"


def test_changed_inputs_make_approval_stale(audit):
    graph, box = _graph(audit, InMemorySaver())
    t, _ = start_request(graph, HOUSEHOLD, "somal")
    box["inputs"] = replace(box["inputs"], liquid_cash_minor=box["inputs"].liquid_cash_minor + 100_00)
    out = _resume(graph, t, approver="somal", approved=True)
    assert out["status"] == "stale"
    assert [e["reason"] for e in audit.events(t) if e["action"] == "approve"] == ["plan inputs changed since review"]


def test_no_surplus_ends_without_interrupt(audit):
    inputs = demo_planning_inputs()
    graph, _ = _graph(audit, InMemorySaver(), replace(inputs, liquid_cash_minor=inputs.required_reserve_minor))
    _, out = start_request(graph, HOUSEHOLD, "somal")
    assert out["status"] == "not_eligible" and "__interrupt__" not in out


def _open(tmp_path):
    # The checkpointer manages its own transactions, so it gets its own database/connection.
    ledger = sqlite3.connect(tmp_path / "ledger.db")
    checkpoints = sqlite3.connect(tmp_path / "checkpoints.db", check_same_thread=False)
    return ledger, checkpoints


@pytest.mark.parametrize("attempt", range(5))  # repeated: this once failed intermittently
def test_paused_request_survives_restart(tmp_path, attempt):
    ledger1, ckpt1 = _open(tmp_path)
    graph1, _ = _graph(AuditLog(ledger1), SqliteSaver(ckpt1))
    thread_id, _ = start_request(graph1, HOUSEHOLD, "somal")
    ledger1.close()
    ckpt1.close()

    ledger2, ckpt2 = _open(tmp_path)  # "new process"
    audit2 = AuditLog(ledger2)
    graph2, _ = _graph(audit2, SqliteSaver(ckpt2))
    out = _resume(graph2, thread_id, approver="sayali", approved=True)
    assert out["status"] == "approved"
    assert [e["action"] for e in audit2.events(thread_id)] == ["plan_allocation", "approve"]
    assert audit2.verify() == 2
    ledger2.close()
    ckpt2.close()
