"""Planner inputs derived from the ledger, and planner runs recorded as evidence (design §5.8, §3.4).

``derive_planning_inputs`` builds ``PlanningInputs`` for one actor from data that actor may
read, never from model text:

- liquid cash: the closing balance of the latest verified statement of each checking or
  savings account in the look-back window;
- monthly essential outflow: per account, the essential-category outflow from a recorded
  ``category_totals`` calculation run, divided by the number of verified statement months,
  then summed (rounded up, so the reserve errs high);
- near-term liabilities: explicit items, each with an evidence id (for example the county tax
  bill), plus the amount owed on verified credit-card statements.

Accounts the actor may not read, or without a verified statement in the window, are left out
and listed in the provenance; the result is then ``complete=False`` and the plan run is
recorded as provisional. Contribution room and return assumptions are inputs: in a real run
they come from Tier-1 retrieval and the household, not from the ledger.

``record_plan_run`` stores the plan as a ``calculation_runs`` row (``fn="plan_allocation"``)
with the planner's input hash, so an approved plan can be traced to its exact inputs.
"""

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from household_agent.approval import plan_summary
from household_agent.authz import Actor
from household_agent.calc_runner import CalculationRunner, _hash
from household_agent.ledger import LedgerStore
from household_agent.models import Action
from household_agent.planning import (
    PLANNER_VERSION,
    SCORING_VERSION,
    Critic,
    PlanResult,
    ThoughtGenerator,
    plan_allocation,
)
from household_finance.allocation import PlanningInputs

PURPOSE = "capital_allocation"
LOOKBACK_MONTHS = 3
LIQUID_KINDS = frozenset({"checking", "savings"})
CARD_KINDS = frozenset({"credit_card"})
# Recurring essentials. Property tax is lumpy, so it enters as an explicit near-term liability
# instead of being averaged into a monthly figure.
ESSENTIAL_CATEGORIES = frozenset({"mortgage", "rent", "utilities", "groceries", "hoa_dues", "insurance"})


@dataclass(frozen=True)
class Liability:
    evidence_id: str
    amount_minor: int
    description: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("a liability needs an evidence id")
        if not isinstance(self.amount_minor, int) or isinstance(self.amount_minor, bool) or self.amount_minor < 0:
            raise ValueError("liability amount must be a non-negative int of minor units")


@dataclass(frozen=True)
class DerivedInputs:
    inputs: PlanningInputs
    window: tuple[date, date]
    accounts: tuple[dict[str, Any], ...]
    excluded_accounts: Mapping[str, str]
    calc_run_ids: tuple[str, ...]
    liability_evidence_ids: tuple[str, ...]
    complete: bool = field(default=True)

    @property
    def evidence_ids(self) -> list[str]:
        docs = [a["document_id"] for a in self.accounts]
        return sorted({*docs, *self.calc_run_ids, *self.liability_evidence_ids, self.inputs.limits_evidence_id})

    def provenance(self) -> dict[str, Any]:
        return {
            "window": [d.isoformat() for d in self.window],
            "accounts": list(self.accounts),
            "excluded_accounts": dict(sorted(self.excluded_accounts.items())),
            "calc_run_ids": list(self.calc_run_ids),
            "liability_evidence_ids": list(self.liability_evidence_ids),
            "limits_evidence_id": self.inputs.limits_evidence_id,
            "complete": self.complete,
        }


def _months_back(d: date, months: int) -> date:
    """First day of the month ``months`` before ``d``'s month."""
    y, m = divmod(d.year * 12 + d.month - 1 - months, 12)
    return date(y, m + 1, 1)


def derive_planning_inputs(
    store: LedgerStore,
    runner: CalculationRunner,
    actor: Actor,
    *,
    as_of: datetime,
    correlation_id: str,
    contribution_room_minor: Mapping[str, int],
    limits_evidence_id: str,
    real_return: Mapping[str, Decimal],
    liabilities: Sequence[Liability] = (),
    lookback_months: int = LOOKBACK_MONTHS,
    **assumptions: Any,
) -> DerivedInputs:
    if lookback_months < 1:
        raise ValueError("lookback_months must be at least 1")
    window = (_months_back(as_of.date(), lookback_months), as_of.date())
    excluded: dict[str, str] = {}
    readable = []
    for account in store.authorizer.accounts(actor.household_id):
        if store.authorizer.authorize(actor, account.account_id, Action.READ, PURPOSE, as_of):
            readable.append(account)
        else:
            excluded[account.account_id] = "not readable by this member for capital_allocation"

    liquid = card_owed = monthly_essential = 0
    used: list[dict[str, Any]] = []
    run_ids: list[str] = []
    for account in readable:
        statements = store.verified_statements(actor, PURPOSE, correlation_id, [account.account_id], *window, at=as_of)
        if not statements:
            excluded[account.account_id] = "no verified statement in the window"
            continue
        latest = statements[0]
        months = sorted({s["period_start"][:7] for s in statements})
        if account.kind in LIQUID_KINDS:
            liquid += max(latest["closing_balance_minor"], 0)
        elif account.kind in CARD_KINDS:
            card_owed += max(-latest["closing_balance_minor"], 0)

        start = date.fromisoformat(min(s["period_start"] for s in statements))
        end = date.fromisoformat(max(s["period_end"] for s in statements))
        txns = store.query_transactions(actor, PURPOSE, correlation_id, [account.account_id], start, end, at=as_of)
        run = runner.run("category_totals", txns, actor_id=actor.member_id, household_id=actor.household_id,
                         correlation_id=correlation_id, params={"start": start.isoformat(), "end": end.isoformat()},
                         acting_agent="planner_inputs")
        outflow = -sum(min(v, 0) for k, v in run.output.items() if k in ESSENTIAL_CATEGORIES)
        monthly_essential += -(-outflow // len(months))  # ceiling division
        run_ids.append(run.run_id)
        used.append({"account_id": account.account_id, "kind": account.kind, "document_id": latest["document_id"],
                     "period_end": latest["period_end"], "closing_balance_minor": latest["closing_balance_minor"],
                     "months": months, "essential_outflow_minor": outflow, "calc_run_id": run.run_id})

    inputs = PlanningInputs(
        liquid_cash_minor=liquid,
        monthly_essential_outflow_minor=monthly_essential,
        near_term_liabilities_minor=sum(li.amount_minor for li in liabilities) + card_owed,
        contribution_room_minor=dict(contribution_room_minor),
        limits_evidence_id=limits_evidence_id,
        real_return=dict(real_return),
        **assumptions,
    )
    return DerivedInputs(inputs, window, tuple(used), excluded, tuple(run_ids),
                         tuple(li.evidence_id for li in liabilities), complete=not excluded)


def record_plan_run(
    store: LedgerStore, actor: Actor, result: PlanResult, derived: DerivedInputs, *, correlation_id: str,
    at: datetime | None = None,
) -> str:
    """Store the plan as calculation evidence. Provisional when some account was left out."""
    at = at or datetime.now(UTC)
    i = derived.inputs
    output = plan_summary(result)
    run_id = f"plan_run_{uuid.uuid4().hex[:12]}"
    output_hash = _hash(output)
    store.record_calculation_run({
        "run_id": run_id, "fn": "plan_allocation", "library_version": f"{PLANNER_VERSION}+{SCORING_VERSION}",
        "input_ids": derived.evidence_ids,
        "params": {"reserve_months": i.reserve_months, "buffer_target_pct": str(i.buffer_target_pct),
                   "horizon_years": i.horizon_years, "taxable_drag": str(i.taxable_drag),
                   "provenance": derived.provenance()},
        "input_hash": result.input_hash, "output": output, "output_hash": output_hash,
        "provisional": not derived.complete, "correlation_id": correlation_id, "created_at": at.isoformat(),
    })
    store.audit.append(
        correlation_id=correlation_id, actor_id=actor.member_id, household_id=actor.household_id,
        action="calculate", purpose=PURPOSE, decision="info", resource=f"calc:{run_id}", acting_agent="planner",
        details={"fn": "plan_allocation", "input_hash": result.input_hash, "output_hash": output_hash,
                 "status": result.status, "provisional": not derived.complete,
                 "excluded_accounts": sorted(derived.excluded_accounts)},
        at=at,
    )
    return run_id


def plan_from_ledger(
    store: LedgerStore, runner: CalculationRunner, actor: Actor, *, as_of: datetime, correlation_id: str,
    generator: ThoughtGenerator | None = None, critic: Critic | None = None, **kwargs: Any,
) -> tuple[PlanResult, DerivedInputs, str]:
    """Derive inputs, run the planner, record the run. Returns ``(result, derived, run_id)``."""
    derived = derive_planning_inputs(store, runner, actor, as_of=as_of, correlation_id=correlation_id, **kwargs)
    result = plan_allocation(derived.inputs, generator, critic)
    return result, derived, record_plan_run(store, actor, result, derived, correlation_id=correlation_id, at=as_of)


def ledger_inputs_for(
    store: LedgerStore, runner: CalculationRunner, actor: Actor, *, clock: Callable[[], datetime],
    correlation_id: str = "allocation_inputs", **kwargs: Any,
) -> Callable[[str], PlanningInputs]:
    """``inputs_for`` for ``build_approval_graph``: fresh ledger-derived inputs on every call,
    so a ledger change between showing and approving a plan makes the request ``stale``."""
    def inputs_for(household_id: str) -> PlanningInputs:
        if household_id != actor.household_id:
            raise PermissionError("planner inputs requested for another household")
        return derive_planning_inputs(store, runner, actor, as_of=clock(), correlation_id=correlation_id,
                                      **kwargs).inputs
    return inputs_for
