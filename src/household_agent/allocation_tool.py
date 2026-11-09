"""The Supervisor-side ``evaluate_allocation_candidates`` tool (design §4.1, §5.8).

Like ``draft_reminder`` this is not on any specialist's allowlist (``ToolSession.TOOL_NAMES`` and the Ledger Analyst
and Policy & Compliance tool lists do not contain it): the Supervisor owns the planner. It wraps the pure evaluator in
``allocation.py`` with the project's usual guards:

- **Authorization.** Only a household admin or adult member, in their own household, for the ``capital_allocation``
  purpose. Dependents, delegates and any other purpose are refused, and the refusal is audited.
- **Inputs.** Monthly essential expenses and liquid cash come from the ledger when the actor may read it (verified
  statements only, the same derivation the §5.8 planner uses). APRs, minimum payments, balances, the emergency-fund
  target, retirement headroom and the household's minimum buffer are never invented: they must be supplied by the host
  in ``AllocationFacts``. Anything missing is refused with a message that names it.
- **Evidence.** Each successful run is stored as a ``calculation_runs`` row (``fn = evaluate_allocation_candidates``)
  and audited. ``run_id`` is the evidence ID that claims cite; ``evidence_id`` is returned as the same value.
"""

import uuid
from dataclasses import dataclass, field
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from household_agent.allocation import (
    EVALUATOR_VERSION,
    MAX_DEBTS,
    MAX_HORIZON_MONTHS,
    MAX_MINOR,
    AllocationEvaluation,
    AllocationInputError,
    CandidatePlan,
    Debt,
    HouseholdFinancialState,
    RetirementAccount,
    evaluate_candidates,
    evaluation_output,
)
from household_agent.approval import APPROVER_ROLES
from household_agent.authz import Actor
from household_agent.calc_runner import CalculationRunner, _hash
from household_agent.ledger import LedgerStore
from household_agent.plan_inputs import PURPOSE, derive_planning_inputs
from household_finance.allocation import BUCKET_TIER
from household_finance.money import MoneyError, to_minor

TOOL_NAME = "evaluate_allocation_candidates"
SUPERVISOR_ONLY_TOOLS = (TOOL_NAME, "draft_reminder")
DEFAULT_HORIZON_MONTHS = 12
_UNUSED_RETURNS = {bucket: Decimal(0) for bucket in BUCKET_TIER}  # placeholder the shared derivation requires


class AllocationFacts(BaseModel):
    """What the host knows about the household that the ledger cannot say. Supplied by the host, never by a model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    debts: tuple[Debt, ...] = Field(max_length=MAX_DEBTS)
    emergency_fund_balance_minor: StrictInt = Field(ge=0, le=MAX_MINOR)
    emergency_fund_target_months: Decimal = Field(gt=0, le=120, max_digits=8, decimal_places=2, allow_inf_nan=False)
    retirement_accounts: tuple[RetirementAccount, ...] = Field(max_length=MAX_DEBTS)
    min_cash_buffer_months: Decimal = Field(ge=0, le=120, max_digits=8, decimal_places=2, allow_inf_nan=False)
    horizon_months: StrictInt = Field(default=DEFAULT_HORIZON_MONTHS, ge=1, le=MAX_HORIZON_MONTHS)
    # Optional overrides. When left out they are read from the ledger; if the ledger cannot supply them either, the
    # tool refuses and says so.
    monthly_essential_expenses_minor: StrictInt | None = Field(default=None, gt=0, le=MAX_MINOR)
    liquid_cash_minor: StrictInt | None = Field(default=None, ge=0, le=MAX_MINOR)


def access_denial(actor: Actor, purpose: str) -> str | None:
    """Why ``actor`` may not use the tool for ``purpose``, or ``None``."""
    if actor.role not in APPROVER_ROLES:
        return f"{actor.role.value} may not request allocation evaluations"
    if purpose != PURPOSE:
        return f"purpose {purpose!r} is not {PURPOSE!r}"
    return None


def audit_denial(store: LedgerStore, actor: Actor, purpose: str | None, correlation_id: str, reason: str, *,
                 at: datetime, acting_agent: str | None = "supervisor") -> None:
    store.audit.append(correlation_id=correlation_id, actor_id=actor.member_id, household_id=actor.household_id,
                       acting_agent=acting_agent, action=TOOL_NAME, purpose=purpose, decision="deny", reason=reason,
                       at=at)


def _error(kind: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"type": kind, "message": message}}


def _validation_error(exc: ValidationError) -> dict[str, Any]:
    errs = exc.errors()
    where = [".".join(str(p) for p in e["loc"]) for e in errs]
    if all(e["type"] == "missing" for e in errs):
        return _error("missing_input", "missing required input(s): " + ", ".join(where))
    return _error("invalid_request", "; ".join(f"{w}: {e['msg']}" for w, e in zip(where, errs, strict=True)))


def parse_amount(value: str) -> int:
    """Dollars text ("$2,000", "2000.50") to cents; ``ValueError`` if it is not an exact amount."""
    try:
        return to_minor(value)
    except MoneyError as exc:
        raise ValueError(str(exc)) from exc


@dataclass(frozen=True)
class ResolvedState:
    """The household snapshot plus where each input came from (``ledger`` or ``task_input``)."""

    state: HouseholdFinancialState
    sources: dict[str, str]
    ledger_runs: tuple[str, ...] = ()
    ledger_docs: tuple[str, ...] = ()
    excluded_accounts: tuple[str, ...] = ()
    provisional: bool = field(default=False)


def resolve_state(store: LedgerStore, actor: Actor, facts: AllocationFacts | dict[str, Any] | None, *,
                  correlation_id: str, at: datetime) -> ResolvedState | dict[str, Any]:
    """Build the snapshot: explicit inputs, plus ledger-derived cash and essentials where not given.
    Returns an error dict (``ok`` False) when something required is missing or invalid. Does not check access."""
    if facts is None:
        return _error("missing_input", "missing required input(s): allocation facts (debts with APR and minimum payment, "
                                       "emergency-fund balance and target months, retirement headroom, minimum cash buffer)")
    try:
        f = facts if isinstance(facts, AllocationFacts) else AllocationFacts.model_validate(facts)
    except ValidationError as exc:
        return _validation_error(exc)
    sources: dict[str, str] = {}
    cash, essentials = f.liquid_cash_minor, f.monthly_essential_expenses_minor
    runs: list[str] = []
    docs: list[str] = []
    excluded: list[str] = []
    provisional = False
    if cash is None or essentials is None:
        derived = derive_planning_inputs(
            store, CalculationRunner(store), actor, as_of=at, correlation_id=correlation_id,
            contribution_room_minor={}, limits_evidence_id=None,  # type: ignore[arg-type]
            real_return=_UNUSED_RETURNS)  # only balances and essential outflow are read; no return is ever used
        runs, docs = list(derived.calc_run_ids), [a["document_id"] for a in derived.accounts]
        provisional, excluded = not derived.complete, sorted(derived.excluded_accounts)
        if cash is None and derived.accounts:
            cash = derived.inputs.liquid_cash_minor
            sources["liquid_cash_minor"] = "ledger"
        if essentials is None and derived.inputs.monthly_essential_outflow_minor > 0:
            essentials = derived.inputs.monthly_essential_outflow_minor
            sources["monthly_essential_expenses_minor"] = "ledger"
    missing = [n for n, v in (("liquid_cash_minor", cash), ("monthly_essential_expenses_minor", essentials)) if v is None]
    if missing:
        return _error("missing_input", "missing required input(s) the ledger could not supply: " + ", ".join(missing))
    for name in ("liquid_cash_minor", "monthly_essential_expenses_minor", "debts", "emergency_fund_balance_minor",
                 "emergency_fund_target_months", "retirement_accounts", "min_cash_buffer_months", "horizon_months"):
        sources.setdefault(name, "task_input")
    try:
        state = HouseholdFinancialState(
            monthly_essential_expenses_minor=essentials, liquid_cash_minor=cash,  # type: ignore[arg-type]
            emergency_fund_balance_minor=f.emergency_fund_balance_minor,
            emergency_fund_target_months=f.emergency_fund_target_months, min_cash_buffer_months=f.min_cash_buffer_months,
            debts=f.debts, retirement_accounts=f.retirement_accounts, horizon_months=f.horizon_months)
    except ValidationError as exc:
        return _validation_error(exc)
    return ResolvedState(state, sources, tuple(runs), tuple(docs), tuple(excluded), provisional)


def evaluate_allocation_candidates(
    store: LedgerStore, actor: Actor, *, purpose: str, correlation_id: str, extra_amount_minor: int,
    candidates: Sequence[CandidatePlan | dict[str, Any]],
    facts: AllocationFacts | ResolvedState | dict[str, Any] | None, at: datetime,
    acting_agent: str | None = "supervisor",
) -> dict[str, Any]:
    """Evaluate candidate plans. Never raises: failures are ``{"ok": False, "error": {"type", "message"}}``.

    Success returns ``run_id`` / ``evidence_id`` (equal), ``output`` (the full evaluation with ``*_display`` strings),
    ``input_hash``, ``output_hash``, ``provisional``, ``input_sources`` (``ledger`` or ``task_input`` per field) and
    ``ledger_excluded_accounts``. ``facts`` may be a ``ResolvedState`` the caller already built with ``resolve_state``.
    """
    denial = access_denial(actor, purpose)
    if denial:
        audit_denial(store, actor, purpose, correlation_id, denial, at=at, acting_agent=acting_agent)
        return _error("authorization_denied", denial)
    resolved = facts if isinstance(facts, ResolvedState) else resolve_state(
        store, actor, facts, correlation_id=correlation_id, at=at)
    if isinstance(resolved, dict):
        return resolved
    try:
        plans = [c if isinstance(c, CandidatePlan) else CandidatePlan.model_validate(c) for c in candidates]
        evaluation: AllocationEvaluation = evaluate_candidates(resolved.state, extra_amount_minor, plans)
    except ValidationError as exc:
        return _validation_error(exc)
    except AllocationInputError as exc:
        store.audit.append(correlation_id=correlation_id, actor_id=actor.member_id, household_id=actor.household_id,
                           acting_agent=acting_agent, action=TOOL_NAME, purpose=purpose, decision="deny",
                           reason=str(exc), at=at)
        return _error("invalid_request", str(exc))

    output = evaluation_output(evaluation)
    run_id = f"alloc_run_{uuid.uuid4().hex[:12]}"
    output_hash = _hash(output)
    store.record_calculation_run({
        "run_id": run_id, "fn": TOOL_NAME, "library_version": EVALUATOR_VERSION,
        "input_ids": sorted({*resolved.ledger_runs, *resolved.ledger_docs}),
        "params": {"extra_amount_minor": extra_amount_minor, "horizon_months": resolved.state.horizon_months,
                   "input_sources": resolved.sources, "candidates": [p.name for p in plans],
                   "ledger_excluded_accounts": list(resolved.excluded_accounts)},
        "input_hash": evaluation.input_hash, "output": output, "output_hash": output_hash,
        "provisional": resolved.provisional, "correlation_id": correlation_id, "created_at": at.isoformat()})
    store.audit.append(
        correlation_id=correlation_id, actor_id=actor.member_id, household_id=actor.household_id,
        acting_agent=acting_agent, action="calculate", purpose=purpose, decision="info", resource=f"calc:{run_id}",
        details={"fn": TOOL_NAME, "input_hash": evaluation.input_hash, "output_hash": output_hash,
                 "provisional": resolved.provisional, "candidates": len(plans), "best_valid": evaluation.best_valid,
                 "input_sources": resolved.sources}, at=at)
    return {"ok": True, "run_id": run_id, "evidence_id": run_id, "output": output,
            "input_hash": evaluation.input_hash, "output_hash": output_hash, "provisional": resolved.provisional,
            "input_sources": resolved.sources, "ledger_excluded_accounts": list(resolved.excluded_accounts),
            "evaluator_version": EVALUATOR_VERSION}
