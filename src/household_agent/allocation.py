"""Deterministic evaluator for "where should our extra money go?" (design §4.1 / §5.8 companion).

Pure code: no model, no clock, no I/O. Given a frozen snapshot of the household's finances and a few
*candidate plans* (how many cents each plan puts in each bucket) it returns exact outcomes per candidate and a
transparent ranking. The Supervisor's Tree-of-Thoughts step proposes candidate *shapes*; every number the
household sees comes from here, so the claim validator can check it against this output.

Money is integer cents (``*_minor``). Rates and ratios are ``Decimal``; they serialize as strings.

Assumptions (also returned verbatim in the output, ``assumptions``):

- The extra amount is already counted in ``liquid_cash_minor`` (it is cash on hand that has not been moved yet).
- ``liquid_cash_minor`` is spendable cash outside the emergency fund; the emergency fund is a separate balance.
- Interest: each month, interest = balance * APR / 12, rounded half-up to the cent, is added; then the minimum
  payment (capped at balance + interest) is paid. The extra principal is applied at the start of month 1. The
  minimum payment never changes (it does not shrink when the balance does), and no other payments are made.
  Interest saved = baseline interest over the horizon (no extra principal) minus interest with the extra principal.
- Liquid reserve after a plan = cash left after money leaves for debt, retirement and the emergency fund, plus the
  emergency fund after its contribution. Buffer months = liquid reserve / monthly essential expenses.
- The retirement bucket is contribution/headroom accounting only. NO investment return is forecast, nothing is
  picked. The only tax effect is the optional flat ``tax_deferred_rate_percent`` the host gives an account:
  tax benefit = contribution x rate, rounded half-up to the cent, counted once as a saving for the horizon. With the
  rate at its default of 0 the benefit is 0 and the ranking is exactly the interest-only ranking.
- A candidate that over-pays a debt is evaluated with the payment clamped to the balance (its figures stay defined)
  and is reported as invalid.

Ranking (``alloc_rank_1.1``), lexicographic, lower tuple first, so it is total and stable (names are unique):
(number of violations, emergency-fund shortfall to target, -(interest saved + tax benefit), -retirement used,
-liquid reserve after, name). Safety first (valid, emergency fund funded), then cheaper debt, then retirement room.
The minimum cash buffer is a hard constraint, so it is covered by the first key.
"""

import hashlib
import json
from collections.abc import Sequence
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from household_finance.money import format_minor

EVALUATOR_VERSION = "alloc_eval_1.1"
RANKING_VERSION = "alloc_rank_1.1"
MAX_HORIZON_MONTHS = 360
MAX_CANDIDATES = 12
MAX_DEBTS = 20
MAX_MINOR = 10**12  # $10 billion: a sanity bound, far above any household
_ID = r"^[A-Za-z0-9_.:-]{1,40}$"

ASSUMPTIONS: tuple[str, ...] = (
    "The extra amount is already included in liquid cash and has not been moved yet.",
    "Liquid cash excludes the emergency fund, which is a separate balance; the liquid reserve is cash plus emergency fund.",
    "Interest accrues monthly at APR/12 on the balance, rounded half-up to the cent; the minimum payment is paid each month "
    "and never shrinks; extra principal is applied at the start of month 1; no other payments are modelled.",
    "Interest saved is measured only over the stated horizon, not over the life of the debt.",
    "The retirement bucket is contribution and headroom accounting only: no investment return is forecast, "
    "no investments are picked. The only tax effect is the flat pre-tax rate the household states for an account "
    "(tax benefit = contribution x rate, once); there are no brackets, employer match, phase-outs or state taxes.",
    "Nothing here moves money; the household carries out any transfer itself.",
)

Minor = Annotated[int, StrictInt]


class AllocationInputError(ValueError):
    """The request itself is unusable (zero extra, no candidates, duplicate names)."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Debt(_Frozen):
    debt_id: str = Field(pattern=_ID)
    balance_minor: StrictInt = Field(ge=0, le=MAX_MINOR)
    apr_percent: Decimal = Field(ge=0, le=100, max_digits=8, decimal_places=4, allow_inf_nan=False)
    minimum_payment_minor: StrictInt = Field(ge=0, le=MAX_MINOR)


class RetirementAccount(_Frozen):
    account_id: str = Field(pattern=_ID)
    headroom_minor: StrictInt = Field(ge=0, le=MAX_MINOR)  # contribution room left this tax year
    # Host-supplied, never inferred: the share of a contribution that is deducted from taxable income (0 = after-tax).
    tax_deferred_rate_percent: Decimal = Field(default=Decimal(0), ge=0, le=100, max_digits=6, decimal_places=2,
                                               allow_inf_nan=False)


class HouseholdFinancialState(_Frozen):
    """Frozen snapshot the evaluation is computed from. Every field is explicit; nothing has a hidden default."""

    monthly_essential_expenses_minor: StrictInt = Field(gt=0, le=MAX_MINOR)
    liquid_cash_minor: StrictInt = Field(ge=0, le=MAX_MINOR)
    emergency_fund_balance_minor: StrictInt = Field(ge=0, le=MAX_MINOR)
    emergency_fund_target_months: Decimal = Field(gt=0, le=120, max_digits=8, decimal_places=2, allow_inf_nan=False)
    min_cash_buffer_months: Decimal = Field(ge=0, le=120, max_digits=8, decimal_places=2, allow_inf_nan=False)
    debts: tuple[Debt, ...] = Field(max_length=MAX_DEBTS)
    retirement_accounts: tuple[RetirementAccount, ...] = Field(max_length=MAX_DEBTS)
    horizon_months: StrictInt = Field(ge=1, le=MAX_HORIZON_MONTHS)

    @model_validator(mode="after")
    def _unique_ids(self) -> "HouseholdFinancialState":
        for what, ids in (("debt_id", [d.debt_id for d in self.debts]),
                          ("account_id", [a.account_id for a in self.retirement_accounts])):
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {what}")
        return self


class CandidatePlan(_Frozen):
    """One way to split the extra amount, in cents. Negative or oversized amounts are accepted here on purpose:
    the evaluator reports them as violations instead of refusing the whole request."""

    name: str = Field(min_length=1, max_length=60)
    debt_principal_minor: dict[str, StrictInt] = Field(default_factory=dict)
    emergency_fund_minor: StrictInt = 0
    retirement_minor: dict[str, StrictInt] = Field(default_factory=dict)
    hold_minor: StrictInt = 0  # kept as cash on purpose


class Violation(_Frozen):
    code: str
    message: str


class DebtOutcome(_Frozen):
    debt_id: str
    extra_principal_minor: int  # after clamping to [0, balance]
    baseline_interest_minor: int
    plan_interest_minor: int
    interest_saved_minor: int
    plan_balance_at_horizon_minor: int
    paid_off_in_month: int | None


class CandidateOutcome(_Frozen):
    name: str
    rank: int
    valid: bool
    allocated_minor: int
    unallocated_minor: int  # extra - allocated; negative means over-allocated
    interest_saved_minor: int
    tax_benefit_minor: int
    debts: tuple[DebtOutcome, ...]
    liquid_reserve_after_minor: int
    buffer_months: Decimal
    buffer_meets_minimum: bool
    emergency_fund_after_minor: int
    emergency_fund_target_minor: int
    emergency_fund_pct_of_target: Decimal
    emergency_fund_shortfall_minor: int
    retirement_used_minor: int
    retirement_headroom_minor: int
    retirement_headroom_remaining_minor: int
    violations: tuple[Violation, ...]
    score_key: tuple[Any, ...]
    rationale: str


class AllocationEvaluation(_Frozen):
    evaluator_version: str = EVALUATOR_VERSION
    ranking_version: str = RANKING_VERSION
    extra_amount_minor: int
    horizon_months: int
    inputs: HouseholdFinancialState
    candidates: tuple[CandidateOutcome, ...]  # ranked, best first
    ranking: tuple[str, ...]
    best_valid: str | None
    no_valid_candidate: bool
    assumptions: tuple[str, ...] = ASSUMPTIONS
    input_hash: str


def _dec(d: Decimal) -> str:
    return f"{d:f}"


def _simulate(balance: int, apr_percent: Decimal, payment: int, months: int) -> tuple[int, int, int | None]:
    """(total interest, balance at horizon, month paid off) under the module's amortization assumptions."""
    rate = apr_percent / Decimal(1200)
    total, paid_off = 0, None
    for month in range(1, months + 1):
        if balance <= 0:
            break
        interest = int((Decimal(balance) * rate).quantize(Decimal(1), rounding=ROUND_HALF_UP))
        balance = balance + interest - min(payment, balance + interest)
        total += interest
        if balance == 0:
            paid_off = month
            break
    return total, balance, paid_off


def _key_text(index: int) -> str:
    return ("number of constraint violations", "emergency-fund shortfall", "interest saved plus tax benefit",
            "retirement headroom used", "liquid reserve after the plan", "name")[index]


def _evaluate_one(state: HouseholdFinancialState, extra: int, plan: CandidatePlan,
                  baseline: dict[str, tuple[int, int, int | None]]) -> dict[str, Any]:
    violations: list[Violation] = []
    debt_by_id = {d.debt_id: d for d in state.debts}
    acct_by_id = {a.account_id: a for a in state.retirement_accounts}

    def bad(code: str, message: str) -> None:
        violations.append(Violation(code=code, message=message))

    for label, amount in (("emergency fund", plan.emergency_fund_minor), ("cash hold", plan.hold_minor),
                          *((f"debt {k}", v) for k, v in sorted(plan.debt_principal_minor.items())),
                          *((f"retirement {k}", v) for k, v in sorted(plan.retirement_minor.items()))):
        if amount < 0:
            bad("negative_amount", f"{label}: amount {format_minor(amount)} is negative")
    for k in sorted(plan.debt_principal_minor):
        if k not in debt_by_id:
            bad("unknown_debt", f"debt {k!r} is not one of the household's debts")
    for k in sorted(plan.retirement_minor):
        if k not in acct_by_id:
            bad("unknown_retirement_account", f"retirement account {k!r} is not one of the household's accounts")

    debt_total = sum(plan.debt_principal_minor.values())
    retirement_total = sum(plan.retirement_minor.values())
    allocated = debt_total + retirement_total + plan.emergency_fund_minor + plan.hold_minor
    unallocated = extra - allocated
    if allocated > extra:
        bad("over_allocation", f"allocates {format_minor(allocated)}, more than the extra {format_minor(extra)}")

    outcomes: list[DebtOutcome] = []
    for d in state.debts:
        put = max(0, min(plan.debt_principal_minor.get(d.debt_id, 0), d.balance_minor))
        if plan.debt_principal_minor.get(d.debt_id, 0) > d.balance_minor:
            bad("debt_overpayment", f"debt {d.debt_id}: {format_minor(plan.debt_principal_minor[d.debt_id])} is more than "
                                    f"its balance {format_minor(d.balance_minor)}")
        base_interest, _, _ = baseline[d.debt_id]
        plan_interest, end_balance, paid = _simulate(d.balance_minor - put, d.apr_percent, d.minimum_payment_minor,
                                                     state.horizon_months)
        outcomes.append(DebtOutcome(
            debt_id=d.debt_id, extra_principal_minor=put, baseline_interest_minor=base_interest,
            plan_interest_minor=plan_interest, interest_saved_minor=base_interest - plan_interest,
            plan_balance_at_horizon_minor=end_balance, paid_off_in_month=paid))

    retirement_used = sum(v for k, v in plan.retirement_minor.items() if k in acct_by_id and v > 0)
    for k, a in sorted(acct_by_id.items()):
        if plan.retirement_minor.get(k, 0) > a.headroom_minor:
            bad("over_contribution_limit", f"retirement {k}: {format_minor(plan.retirement_minor[k])} is more than the "
                                           f"remaining headroom {format_minor(a.headroom_minor)}")
    headroom_total = sum(a.headroom_minor for a in state.retirement_accounts)
    tax_benefit = sum(
        int((Decimal(plan.retirement_minor[k]) * a.tax_deferred_rate_percent / 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))
        for k, a in acct_by_id.items() if plan.retirement_minor.get(k, 0) > 0)

    cash_after = state.liquid_cash_minor - debt_total - retirement_total - plan.emergency_fund_minor
    if cash_after < 0:
        bad("insufficient_cash", f"leaves liquid cash at {format_minor(cash_after)}")
    ef_after = state.emergency_fund_balance_minor + plan.emergency_fund_minor
    reserve_after = cash_after + ef_after
    monthly = state.monthly_essential_expenses_minor
    buffer_months = (Decimal(reserve_after) / Decimal(monthly)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    meets = Decimal(reserve_after) >= state.min_cash_buffer_months * Decimal(monthly)
    if not meets:
        bad("buffer_below_minimum", f"liquid buffer {_dec(buffer_months)} months is below the household minimum "
                                    f"{_dec(state.min_cash_buffer_months)} months")
    ef_target = int((state.emergency_fund_target_months * Decimal(monthly)).quantize(Decimal(1), rounding=ROUND_CEILING))
    ef_pct = (Decimal(ef_after) * 100 / Decimal(ef_target)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    shortfall = max(0, ef_target - ef_after)
    interest_saved = sum(o.interest_saved_minor for o in outcomes)
    return {
        "name": plan.name, "valid": not violations, "allocated_minor": allocated, "unallocated_minor": unallocated,
        "interest_saved_minor": interest_saved, "tax_benefit_minor": tax_benefit, "debts": tuple(outcomes), "liquid_reserve_after_minor": reserve_after,
        "buffer_months": buffer_months, "buffer_meets_minimum": meets, "emergency_fund_after_minor": ef_after,
        "emergency_fund_target_minor": ef_target, "emergency_fund_pct_of_target": ef_pct,
        "emergency_fund_shortfall_minor": shortfall, "retirement_used_minor": retirement_used,
        "retirement_headroom_minor": headroom_total,
        "retirement_headroom_remaining_minor": headroom_total - retirement_used,
        "violations": tuple(violations),
        "score_key": (len(violations), shortfall, -(interest_saved + tax_benefit), -retirement_used, -reserve_after, plan.name),
    }


def _rationale(o: dict[str, Any], state: HouseholdFinancialState, extra: int, rank: int, total: int,
               top: dict[str, Any]) -> str:
    d = format_minor
    parts = [f"{o['name']}: " + ("no constraint violations." if o["valid"] else
                                  "violates constraints: " + "; ".join(v.message for v in o["violations"]) + ".")]
    parts.append(f"Allocates {d(o['allocated_minor'])} of the extra {d(extra)} ({d(o['unallocated_minor'])} unallocated).")
    parts.append(f"Saves {d(o['interest_saved_minor'])} in interest over {state.horizon_months} months.")
    parts.append(f"Emergency fund {_dec(o['emergency_fund_pct_of_target'])}% of its {d(o['emergency_fund_target_minor'])} target.")
    if o["tax_benefit_minor"]:
        parts.append(f"Pre-tax contributions save an estimated {d(o['tax_benefit_minor'])} in tax at the stated rate.")
    parts.append(f"Uses {d(o['retirement_used_minor'])} of {d(o['retirement_headroom_minor'])} retirement headroom.")
    parts.append(f"Liquid buffer {_dec(o['buffer_months'])} months (household minimum {_dec(state.min_cash_buffer_months)}).")
    if rank == 1:
        parts.append(f"Ranked 1 of {total}.")
    else:
        why = next(i for i, (a, b) in enumerate(zip(o["score_key"], top["score_key"], strict=True)) if a != b)
        parts.append(f"Ranked {rank} of {total}, behind {top['name']} on {_key_text(why)}.")
    return " ".join(parts)


def evaluate_candidates(state: HouseholdFinancialState, extra_amount_minor: int,
                        candidates: Sequence[CandidatePlan]) -> AllocationEvaluation:
    """Evaluate and rank candidate plans. Pure; the order of ``candidates`` does not change the result."""
    if isinstance(extra_amount_minor, bool) or not isinstance(extra_amount_minor, int):
        raise AllocationInputError("extra_amount_minor must be an int number of cents")
    if extra_amount_minor <= 0:
        raise AllocationInputError("the extra amount must be greater than zero")
    if extra_amount_minor > MAX_MINOR:
        raise AllocationInputError("the extra amount is implausibly large")
    if not candidates:
        raise AllocationInputError("at least one candidate plan is required")
    if len(candidates) > MAX_CANDIDATES:
        raise AllocationInputError(f"at most {MAX_CANDIDATES} candidate plans can be evaluated at once")
    names = [c.name for c in candidates]
    if len(names) != len(set(names)):
        raise AllocationInputError("candidate names must be unique")

    baseline = {d.debt_id: _simulate(d.balance_minor, d.apr_percent, d.minimum_payment_minor, state.horizon_months)
                for d in state.debts}
    raw = sorted((_evaluate_one(state, extra_amount_minor, c, baseline) for c in candidates),
                 key=lambda o: o["score_key"])
    top = raw[0]
    ranked = tuple(CandidateOutcome(rank=i, rationale=_rationale(o, state, extra_amount_minor, i, len(raw), top), **o)
                   for i, o in enumerate(raw, start=1))
    best_valid = next((c.name for c in ranked if c.valid), None)
    payload = {"extra": extra_amount_minor, "state": state.model_dump(mode="json"),
               "candidates": sorted((c.model_dump(mode="json") for c in candidates), key=lambda c: c["name"]),
               "evaluator": EVALUATOR_VERSION, "ranking": RANKING_VERSION}
    input_hash = "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return AllocationEvaluation(
        extra_amount_minor=extra_amount_minor, horizon_months=state.horizon_months, inputs=state, candidates=ranked,
        ranking=tuple(c.name for c in ranked), best_valid=best_valid, no_valid_candidate=best_valid is None,
        input_hash=input_hash)


def with_display(value: Any) -> Any:
    """Add a formatted ``*_display`` next to every ``*_minor`` int, so "$2,000.00" in a claim matches 200000."""
    if isinstance(value, dict):
        extra = {f"{k[:-6]}_display": format_minor(v) for k, v in value.items()
                 if k.endswith("_minor") and isinstance(v, int) and not isinstance(v, bool)}
        return {**{k: with_display(v) for k, v in value.items()}, **extra}
    if isinstance(value, list):
        return [with_display(v) for v in value]
    return value


def evaluation_output(evaluation: AllocationEvaluation) -> dict[str, Any]:
    """JSON-safe output with display strings: what is stored as evidence and what claims are checked against."""
    return with_display(evaluation.model_dump(mode="json"))
