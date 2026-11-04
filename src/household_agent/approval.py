"""Approval-gated allocation workflow (design §5.8, §10; LangGraph interrupt + checkpointer).

    plan ──(awaiting_approval | needs_review)──► approval ──► END   (approved | rejected | stale)
      └──(not_eligible | no_compliant_plan)──► END

``approval`` pauses the graph with ``interrupt()``. The SQLite checkpointer persists the
paused state, so a human can resume it later, even from a new process. On resume:

- the approver must be a household admin or adult member of the same household; any other
  decision is audited as a deny and the graph keeps waiting;
- the plan is recomputed from current inputs, and if the input hash changed since the plan
  was shown, the request ends ``stale`` instead of approving numbers nobody reviewed.

"Approved" records a human decision. Nothing in this workflow moves money.

Give the checkpointer its own SQLite connection (and file): ``SqliteSaver`` manages its own
transactions and may write from a worker thread, so sharing the ledger/audit connection
races with their commits.
"""

import uuid
from collections.abc import Callable, Mapping
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from household_agent.audit import AuditLog
from household_agent.authz import Actor
from household_agent.models import Action, Role
from household_agent.planning import Critic, PlanResult, ThoughtGenerator, inputs_hash, plan_allocation
from household_finance.allocation import PlanningInputs

APPROVER_ROLES = frozenset({Role.HOUSEHOLD_ADMIN, Role.ADULT_MEMBER})
WORKFLOW = "allocation_approval"

Status = Literal["awaiting_approval", "approved", "rejected", "stale", "not_eligible", "no_compliant_plan"]


class ApprovalState(TypedDict, total=False):
    correlation_id: str
    household_id: str
    requested_by: str
    status: Status
    plan: dict[str, Any]
    decision: dict[str, Any]
    denied_attempts: int


def plan_summary(result: PlanResult) -> dict[str, Any]:
    """JSON-safe view of a plan result: integer cents and decimal strings only."""
    best = result.best
    return {
        "status": result.status,
        "message": result.message,
        "input_hash": result.input_hash,
        "evaluations": result.evaluations,
        "planner_version": result.planner_version,
        "best_node_id": best.node_id if best else None,
        "allocation_minor": dict(sorted(best.allocation.items())) if best else {},
        "composite": str(best.score.composite) if best else None,
        "discrepancy_node_ids": list(result.discrepancy_node_ids),
    }


def build_approval_graph(
    *,
    inputs_for: Callable[[str], PlanningInputs],
    members: Mapping[str, Actor],
    audit: AuditLog,
    checkpointer: BaseCheckpointSaver,
    generator: ThoughtGenerator | None = None,
    critic: Critic | None = None,
):
    def _audit(state: ApprovalState, actor_id: str, action: str, decision: str, reason: str, **details: Any) -> None:
        audit.append(
            correlation_id=state["correlation_id"], actor_id=actor_id, household_id=state["household_id"],
            acting_agent=WORKFLOW, action=action, purpose="capital_allocation", decision=decision,
            reason=reason, resource=f"plan:{state['plan']['input_hash']}", details=details,
        )

    def plan(state: ApprovalState) -> ApprovalState:
        summary = plan_summary(plan_allocation(inputs_for(state["household_id"]), generator, critic))
        state = {**state, "plan": summary}
        _audit(state, state["requested_by"], "plan_allocation", "info", summary["status"],
               best_node_id=summary["best_node_id"], evaluations=summary["evaluations"])
        if summary["status"] in ("not_eligible", "no_compliant_plan"):
            return {"plan": summary, "status": summary["status"]}
        return {"plan": summary, "status": "awaiting_approval", "denied_attempts": 0}

    def approval(state: ApprovalState) -> ApprovalState:
        answer = interrupt({"kind": WORKFLOW, "correlation_id": state["correlation_id"], "plan": state["plan"]})
        approver_id = str(answer.get("approver", ""))
        approver = members.get(approver_id)
        if approver is None or approver.household_id != state["household_id"] or approver.role not in APPROVER_ROLES:
            _audit(state, approver_id or "unknown", Action.APPROVE.value, "deny", "approver lacks approval rights")
            return {"denied_attempts": state.get("denied_attempts", 0) + 1}

        decision = {"approver": approver_id, "approved": answer.get("approved") is True,
                    "note": str(answer.get("note", ""))}
        current = inputs_hash(inputs_for(state["household_id"]), generator)  # no model calls on approve
        if current != state["plan"]["input_hash"]:
            _audit(state, approver_id, Action.APPROVE.value, "deny", "plan inputs changed since review",
                   current_input_hash=current)
            return {"decision": decision, "status": "stale"}
        status: Status = "approved" if decision["approved"] else "rejected"
        _audit(state, approver_id, Action.APPROVE.value, "allow", status, note=decision["note"])
        return {"decision": decision, "status": status}

    def after_plan(state: ApprovalState) -> str:
        return "approval" if state["status"] == "awaiting_approval" else END

    def after_approval(state: ApprovalState) -> str:
        return "approval" if state["status"] == "awaiting_approval" else END

    g = StateGraph(ApprovalState)
    g.add_node("plan", plan)
    g.add_node("approval", approval)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", after_plan, ["approval", END])
    g.add_conditional_edges("approval", after_approval, ["approval", END])
    return g.compile(checkpointer=checkpointer)


def start_request(graph, household_id: str, requested_by: str) -> tuple[str, dict[str, Any]]:
    """Start a new approval thread; returns ``(thread_id, result)``."""
    thread_id = f"appr_{uuid.uuid4().hex[:12]}"
    result = graph.invoke(
        {"correlation_id": thread_id, "household_id": household_id, "requested_by": requested_by},
        {"configurable": {"thread_id": thread_id}},
    )
    return thread_id, result
