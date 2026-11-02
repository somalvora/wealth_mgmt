"""Capital-allocation plan expansion, validation, and scoring (design §5.8).

The Tree-of-Thought planner proposes *strategies* (which tier, what share of the remaining
surplus, which buckets); it never writes dollar amounts. These pure functions turn a strategy
into integer cents, decide whether a plan is allowed, and score it. Scores are ``Decimal``
values in [0, 1].
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, Decimal
from typing import Literal

from household_finance.calcs import CalculationError

SCORING_VERSION = "alloc_score_1.0"

TIER_BUCKETS: dict[int, frozenset[str]] = {
    1: frozenset({"liquidity_buffer"}),            # Safety tier: cash above the required reserve
    2: frozenset({"hsa", "ira", "plan_529"}),      # Tax-advantaged tier (statutory caps apply)
    3: frozenset({"brokerage", "debt_prepayment"}),  # Discretionary growth tier
}
BUCKET_TIER = {b: t for t, buckets in TIER_BUCKETS.items() for b in buckets}
TAX_ADVANTAGED = TIER_BUCKETS[2]
FINAL_TIER = max(TIER_BUCKETS)

SCORE_WEIGHTS = {
    "tax_efficiency": Decimal("0.40"),
    "liquidity_safety": Decimal("0.35"),
    "expected_net_yield": Decimal("0.25"),
}
_SCORE_Q = Decimal("0.0001")


@dataclass(frozen=True)
class PlanningInputs:
    """Reconciled household facts plus labeled assumptions for one planning run."""

    liquid_cash_minor: int
    monthly_essential_outflow_minor: int
    near_term_liabilities_minor: int
    # Remaining statutory room per tax-advantaged bucket, from a verified Tier-1 source for the
    # tax year. A bucket missing here is unverified and may receive nothing.
    contribution_room_minor: Mapping[str, int]
    limits_evidence_id: str | None
    # Assumptions (shown to the household): annual real return per bucket and taxable drag.
    real_return: Mapping[str, Decimal]
    taxable_drag: Decimal = Decimal("0.01")
    horizon_years: int = 5
    reserve_months: int = 6
    buffer_target_pct: Decimal = Decimal("25")

    def __post_init__(self) -> None:
        for name in ("liquid_cash_minor", "monthly_essential_outflow_minor", "near_term_liabilities_minor"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise CalculationError(f"{name} must be a non-negative int of minor units")
        for b, room in self.contribution_room_minor.items():
            if b not in TAX_ADVANTAGED or not isinstance(room, int) or room < 0:
                raise CalculationError(f"invalid contribution room for {b!r}")
        for b, r in self.real_return.items():
            if b not in BUCKET_TIER or not isinstance(r, Decimal):
                raise CalculationError(f"real_return for {b!r} must be a Decimal for a known bucket")
        if set(self.real_return) != set(BUCKET_TIER):
            raise CalculationError("real_return must cover every bucket")
        if self.horizon_years < 1:
            raise CalculationError("horizon_years must be >= 1")

    @property
    def required_reserve_minor(self) -> int:
        return self.reserve_months * self.monthly_essential_outflow_minor + self.near_term_liabilities_minor

    @property
    def surplus_minor(self) -> int:
        return self.liquid_cash_minor - self.required_reserve_minor


@dataclass(frozen=True)
class AllocationStep:
    """A strategy for one tier. ``fraction`` is the share of the remaining surplus to commit
    (the final tier always commits everything left). ``buckets`` are weights for ``split`` mode
    or a priority order for ``fill_in_order`` mode (filled up to each bucket's statutory room)."""

    tier: int
    fraction: Decimal
    buckets: tuple[tuple[str, Decimal], ...]
    mode: Literal["split", "fill_in_order"] = "split"


@dataclass(frozen=True)
class Violation:
    code: str
    message: str


@dataclass(frozen=True)
class PlanScore:
    tax_efficiency: Decimal
    liquidity_safety: Decimal
    expected_net_yield: Decimal
    composite: Decimal
    violations: tuple[Violation, ...] = field(default=())

    @property
    def compliant(self) -> bool:
        return not self.violations


def allocated_total(allocation: Mapping[str, int]) -> int:
    return sum(allocation.values())


def expand_step(inputs: PlanningInputs, allocation: Mapping[str, int], step: AllocationStep) -> dict[str, int]:
    """Apply one strategy step to an allocation and return the new allocation (cents)."""
    if step.tier not in TIER_BUCKETS:
        raise CalculationError(f"unknown tier {step.tier}")
    if not isinstance(step.fraction, Decimal) or not Decimal(0) <= step.fraction <= Decimal(1):
        raise CalculationError("fraction must be a Decimal in [0, 1]")
    remaining = inputs.surplus_minor - allocated_total(allocation)
    if remaining < 0:
        raise CalculationError("allocation already exceeds surplus")
    fraction = Decimal(1) if step.tier == FINAL_TIER else step.fraction
    target = int((Decimal(remaining) * fraction).to_integral_value(rounding=ROUND_DOWN))
    out = dict(allocation)
    if target == 0 or not step.buckets:
        return out

    if step.mode == "fill_in_order":
        left = target
        for bucket, _ in step.buckets:
            room = inputs.contribution_room_minor.get(bucket, 0) - out.get(bucket, 0) if bucket in TAX_ADVANTAGED else left
            give = max(0, min(left, room))
            if give:
                out[bucket] = out.get(bucket, 0) + give
                left -= give
        return out

    # split: largest-remainder apportionment so the parts sum exactly to target.
    total_w = sum((w for _, w in step.buckets), Decimal(0))
    if total_w <= 0:
        raise CalculationError("split weights must sum to a positive value")
    shares = [(b, Decimal(target) * w / total_w) for b, w in step.buckets]
    floors = {b: int(s.to_integral_value(rounding=ROUND_DOWN)) for b, s in shares}
    short = target - sum(floors.values())
    for b, _ in sorted(shares, key=lambda bs: (-(bs[1] - floors[bs[0]]), bs[0]))[:short]:
        floors[b] += 1
    for b, amt in floors.items():
        if amt:
            out[b] = out.get(b, 0) + amt
    return out


def validate_allocation(inputs: PlanningInputs, allocation: Mapping[str, int], *, complete: bool) -> list[Violation]:
    """Hard constraints. Any violation prunes the branch (score 0)."""
    v: list[Violation] = []
    if inputs.surplus_minor <= 0:
        v.append(Violation("no_surplus", "liquid cash does not exceed the required reserve"))
    for bucket, amt in allocation.items():
        if bucket not in BUCKET_TIER:
            v.append(Violation("unknown_bucket", f"{bucket!r} is not an allocation bucket"))
        elif not isinstance(amt, int) or isinstance(amt, bool) or amt < 0:
            v.append(Violation("invalid_amount", f"{bucket}: amount must be a non-negative int"))
        elif bucket in TAX_ADVANTAGED:
            room = inputs.contribution_room_minor.get(bucket)
            if room is None and amt > 0:
                v.append(Violation("limit_unverified", f"{bucket}: no verified statutory limit for this tax year"))
            elif room is not None and amt > room:
                v.append(Violation("statutory_cap_exceeded", f"{bucket}: {amt} exceeds remaining room {room}"))
    if v:
        return v
    total = allocated_total(allocation)
    if total > inputs.surplus_minor:
        v.append(Violation("liquidity_floor_breached", "plan would draw cash below the required reserve"))
    elif complete and total != inputs.surplus_minor:
        v.append(Violation("unallocated_surplus", f"{inputs.surplus_minor - total} minor units left unallocated"))
    return v


def _clamp01(x: Decimal) -> Decimal:
    return min(Decimal(1), max(Decimal(0), x))


def _effective_return(inputs: PlanningInputs, bucket: str) -> Decimal:
    r = inputs.real_return[bucket]
    return r - inputs.taxable_drag if bucket == "brokerage" else r


def score_allocation(inputs: PlanningInputs, allocation: Mapping[str, int]) -> PlanScore:
    """Score a complete plan. Non-compliant plans score 0 on every dimension."""
    violations = tuple(validate_allocation(inputs, allocation, complete=True))
    if violations:
        zero = Decimal(0).quantize(_SCORE_Q)
        return PlanScore(zero, zero, zero, zero, violations)
    surplus = inputs.surplus_minor

    room_total = sum(inputs.contribution_room_minor.values())
    shelter_cap = min(room_total, surplus)
    sheltered = sum(allocation.get(b, 0) for b in TAX_ADVANTAGED)
    tax = Decimal(1) if shelter_cap == 0 else Decimal(sheltered) / Decimal(shelter_cap)

    required = inputs.required_reserve_minor
    target = Decimal(required) * (1 + inputs.buffer_target_pct / 100)
    liquid_after = Decimal(required + allocation.get("liquidity_buffer", 0))
    liquidity = Decimal(1) if target == 0 else liquid_after / target

    h = inputs.horizon_years
    growth = sum((Decimal(amt) * ((1 + _effective_return(inputs, b)) ** h - 1) for b, amt in allocation.items()), Decimal(0))
    best_r = max(_effective_return(inputs, b) for b in BUCKET_TIER)
    best = Decimal(surplus) * ((1 + best_r) ** h - 1)
    net_yield = Decimal(0) if best <= 0 else growth / best

    tax, liquidity, net_yield = (_clamp01(x).quantize(_SCORE_Q, rounding=ROUND_HALF_EVEN) for x in (tax, liquidity, net_yield))
    composite = (
        SCORE_WEIGHTS["tax_efficiency"] * tax
        + SCORE_WEIGHTS["liquidity_safety"] * liquidity
        + SCORE_WEIGHTS["expected_net_yield"] * net_yield
    ).quantize(_SCORE_Q, rounding=ROUND_HALF_EVEN)
    return PlanScore(tax, liquidity, net_yield, composite)


def rollout(inputs: PlanningInputs, allocation: Mapping[str, int], default_steps: Sequence[AllocationStep], from_tier: int) -> dict[str, int]:
    """Complete a partial plan with default steps for tiers >= ``from_tier`` (used to score
    intermediate nodes on a whole-plan basis instead of penalizing tiers not yet decided)."""
    out = dict(allocation)
    for step in default_steps:
        if step.tier >= from_tier:
            out = expand_step(inputs, out, step)
    return out
