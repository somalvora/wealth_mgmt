"""Bounded Tree-of-Thought capital-allocation planner (design §5.8).

Beam search over three allocation tiers (Safety -> Tax-advantaged -> Growth) with b=3 thoughts
per node and a beam of k=2: 3 + 6 + 6 = 15 node evaluations per run. A *thought* is a
strategy (persona, share of remaining surplus, target buckets); it never contains a dollar
amount. ``household_finance.allocation`` expands it into cents, enforces the hard constraints,
and scores the plan. Phase 1 ships a deterministic template generator; Phase 2 swaps in LLM
persona nodes behind the same ``ThoughtGenerator`` protocol.

The result is a *scenario for human approval*. Nothing here moves money.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Literal, Protocol

from household_finance.allocation import (
    FINAL_TIER,
    SCORING_VERSION,
    TIER_BUCKETS,
    AllocationStep,
    PlanningInputs,
    PlanScore,
    Violation,
    expand_step,
    rollout,
    score_allocation,
    validate_allocation,
)

PLANNER_VERSION = "tot_planner_1.0"
BRANCHING = 3
BEAM_WIDTH = 2
SOFT_PRUNE_BELOW = Decimal("0.65")
TIE_MARGIN = Decimal("0.02")
DISCREPANCY_THRESHOLD = Decimal("0.20")
MAX_DISCREPANCY_RETAINED = 2  # extra branches kept per depth when the critic disagrees
PERSONAS = ("conservative", "balanced", "tax_aggressive")

NodeStatus = Literal["selected", "hard_pruned", "soft_pruned", "beam_cut", "retained_discrepancy", "leaf"]


@dataclass(frozen=True)
class Thought:
    persona: str
    step: AllocationStep
    rationale: str


@dataclass(frozen=True)
class PlanNode:
    node_id: str
    depth: int
    allocation: Mapping[str, int]
    thoughts: tuple[Thought, ...]
    score: PlanScore
    status: NodeStatus = "selected"
    critic_score: Decimal | None = None
    discrepancy: bool = False


@dataclass(frozen=True)
class PlanResult:
    status: Literal["awaiting_approval", "needs_review", "no_compliant_plan", "not_eligible"]
    message: str
    best: PlanNode | None
    ranked_leaves: tuple[PlanNode, ...]
    trace: tuple[PlanNode, ...]
    evaluations: int
    input_hash: str
    planner_version: str = PLANNER_VERSION
    scoring_version: str = SCORING_VERSION
    requires_approval: bool = True
    discrepancy_node_ids: tuple[str, ...] = field(default=())


class ThoughtGenerator(Protocol):
    version: str

    def propose(self, inputs: PlanningInputs, node: PlanNode, tier: int) -> Sequence[Thought]: ...


class Critic(Protocol):
    """Optional qualitative evaluator (Phase 2: LLM). Advisory only: it can keep a branch alive
    for human review but can never prune one (design §5.8 risk mitigation)."""

    def score(self, inputs: PlanningInputs, node: PlanNode) -> Decimal: ...


_TEMPLATES: dict[int, dict[str, tuple[AllocationStep, str]]] = {
    1: {
        "conservative": (AllocationStep(1, Decimal("0.30"), (("liquidity_buffer", Decimal(1)),)),
                         "Hold 30% of surplus as extra cash above the six-month reserve."),
        "balanced": (AllocationStep(1, Decimal("0.10"), (("liquidity_buffer", Decimal(1)),)),
                     "Hold a modest 10% cash buffer above the reserve."),
        "tax_aggressive": (AllocationStep(1, Decimal("0"), (("liquidity_buffer", Decimal(1)),)),
                           "Keep only the required reserve; deploy the rest."),
    },
    2: {
        "conservative": (AllocationStep(2, Decimal("0.40"), (("hsa", Decimal(1)), ("ira", Decimal(1)), ("plan_529", Decimal(1))), "fill_in_order"),
                         "Use 40% of what remains for tax-advantaged room, HSA first."),
        "balanced": (AllocationStep(2, Decimal("0.70"), (("hsa", Decimal(1)), ("ira", Decimal(1)), ("plan_529", Decimal(1))), "fill_in_order"),
                     "Use 70% of what remains for tax-advantaged room, HSA first."),
        "tax_aggressive": (AllocationStep(2, Decimal("1"), (("hsa", Decimal(1)), ("ira", Decimal(1)), ("plan_529", Decimal(1))), "fill_in_order"),
                           "Fill all verified tax-advantaged room before anything else."),
    },
    3: {
        "conservative": (AllocationStep(3, Decimal(1), (("debt_prepayment", Decimal(1)),)),
                         "Put the remainder toward debt principal (guaranteed return)."),
        "balanced": (AllocationStep(3, Decimal(1), (("debt_prepayment", Decimal(1)), ("brokerage", Decimal(1)))),
                     "Split the remainder between debt principal and taxable brokerage."),
        "tax_aggressive": (AllocationStep(3, Decimal(1), (("brokerage", Decimal(1)),)),
                           "Invest the remainder in taxable brokerage for growth."),
    },
}
DEFAULT_ROLLOUT = tuple(_TEMPLATES[t]["balanced"][0] for t in sorted(TIER_BUCKETS))


class TemplateThoughtGenerator:
    """Deterministic Conservative / Balanced / Tax-Aggressive personas (Phase 1)."""

    version = "template_generator_1.0"

    def propose(self, inputs: PlanningInputs, node: PlanNode, tier: int) -> Sequence[Thought]:
        return [Thought(p, *_TEMPLATES[tier][p]) for p in PERSONAS]


def _inputs_hash(inputs: PlanningInputs, generator_version: str) -> str:
    payload = {
        "liquid_cash_minor": inputs.liquid_cash_minor,
        "monthly_essential_outflow_minor": inputs.monthly_essential_outflow_minor,
        "near_term_liabilities_minor": inputs.near_term_liabilities_minor,
        "contribution_room_minor": dict(sorted(inputs.contribution_room_minor.items())),
        "limits_evidence_id": inputs.limits_evidence_id,
        "real_return": {k: str(v) for k, v in sorted(inputs.real_return.items())},
        "taxable_drag": str(inputs.taxable_drag),
        "horizon_years": inputs.horizon_years,
        "reserve_months": inputs.reserve_months,
        "buffer_target_pct": str(inputs.buffer_target_pct),
        "generator": generator_version,
        "planner": PLANNER_VERSION,
        "scoring": SCORING_VERSION,
    }
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _evaluate(inputs: PlanningInputs, parent: PlanNode, thought: Thought) -> PlanNode:
    tier = thought.step.tier
    node_id = f"{parent.node_id}/d{tier}.{thought.persona}" if parent.depth else f"d{tier}.{thought.persona}"
    zero = Decimal("0.0000")
    if thought.step.tier != parent.depth + 1:
        bad = PlanScore(zero, zero, zero, zero, (Violation("wrong_tier", f"thought targets tier {tier} at depth {parent.depth + 1}"),))
        return PlanNode(node_id, parent.depth + 1, parent.allocation, (*parent.thoughts, thought), bad, "hard_pruned")
    if any(b not in TIER_BUCKETS[tier] for b, _ in thought.step.buckets):
        bad = PlanScore(zero, zero, zero, zero, (Violation("bucket_not_in_tier", f"tier {tier} allows {sorted(TIER_BUCKETS[tier])}"),))
        return PlanNode(node_id, tier, parent.allocation, (*parent.thoughts, thought), bad, "hard_pruned")

    allocation = expand_step(inputs, parent.allocation, thought.step)
    violations = validate_allocation(inputs, allocation, complete=tier == FINAL_TIER)
    if violations:
        bad = PlanScore(zero, zero, zero, zero, tuple(violations))
        return PlanNode(node_id, tier, allocation, (*parent.thoughts, thought), bad, "hard_pruned")
    # Intermediate nodes are scored on a whole-plan basis: finish the undecided tiers with the
    # Balanced default, then score. Leaves are scored as-is.
    completed = allocation if tier == FINAL_TIER else rollout(inputs, allocation, DEFAULT_ROLLOUT, tier + 1)
    return PlanNode(node_id, tier, allocation, (*parent.thoughts, thought), score_allocation(inputs, completed))


def _rank(nodes: Sequence[PlanNode], k: int | None = None) -> list[PlanNode]:
    """Highest composite first; among candidates within TIE_MARGIN of the current best, prefer
    higher liquidity safety, then higher composite, then node_id (deterministic)."""
    pool = list(nodes)
    out: list[PlanNode] = []
    while pool and (k is None or len(out) < k):
        top = max(n.score.composite for n in pool)
        contenders = [n for n in pool if top - n.score.composite <= TIE_MARGIN]
        pick = min(contenders, key=lambda n: (-n.score.liquidity_safety, -n.score.composite, n.node_id))
        out.append(pick)
        pool.remove(pick)
    return out


def plan_allocation(inputs: PlanningInputs, generator: ThoughtGenerator | None = None, critic: Critic | None = None) -> PlanResult:
    generator = generator or TemplateThoughtGenerator()
    input_hash = _inputs_hash(inputs, generator.version)
    if inputs.surplus_minor <= 0:
        return PlanResult("not_eligible", "Reconciled liquid cash does not exceed the required reserve; build the reserve first.",
                          None, (), (), 0, input_hash)

    zero = Decimal("0.0000")
    root = PlanNode("root", 0, {}, (), PlanScore(zero, zero, zero, zero))
    beam = [root]
    trace: list[PlanNode] = []
    evaluations = 0
    discrepancies: list[str] = []
    overflow = False
    leaves: list[PlanNode] = []  # ranked: compliant leaves first, then discrepancy-retained ones

    for tier in sorted(TIER_BUCKETS):
        children: list[PlanNode] = []
        for parent in beam:
            thoughts = list(generator.propose(inputs, parent, tier))
            if len(thoughts) != BRANCHING:
                raise ValueError(f"generator must propose exactly {BRANCHING} thoughts, got {len(thoughts)}")
            for t in thoughts:
                children.append(_evaluate(inputs, parent, t))
                evaluations += 1

        if critic is not None:
            children = [
                n if n.status == "hard_pruned" else replace(
                    n, critic_score=(c := critic.score(inputs, n)),
                    discrepancy=abs(c - n.score.composite) > DISCREPANCY_THRESHOLD)
                for n in children
            ]

        alive = [n for n in children if n.status != "hard_pruned"]
        passing = [n for n in alive if n.score.composite >= SOFT_PRUNE_BELOW]
        selected_ids = {n.node_id for n in (passing if tier == FINAL_TIER else _rank(passing, BEAM_WIDTH))}
        flagged = [n for n in alive if n.discrepancy and n.node_id not in selected_ids]
        retained_ids = {n.node_id for n in flagged[:MAX_DISCREPANCY_RETAINED]}
        overflow |= len(flagged) > MAX_DISCREPANCY_RETAINED
        discrepancies += [n.node_id for n in alive if n.discrepancy]

        def status_of(n: PlanNode) -> NodeStatus:
            if n.status == "hard_pruned":
                return "hard_pruned"
            if n.node_id in selected_ids:
                return "leaf" if tier == FINAL_TIER else "selected"
            if n.node_id in retained_ids:
                return "retained_discrepancy"
            return "soft_pruned" if n.score.composite < SOFT_PRUNE_BELOW else "beam_cut"

        children = [replace(n, status=status_of(n)) for n in children]
        trace += children
        if tier == FINAL_TIER:
            leaves = (_rank([n for n in children if n.status == "leaf"])
                      + _rank([n for n in children if n.status == "retained_discrepancy"]))
        else:
            beam = [n for n in children if n.status in ("selected", "retained_discrepancy")]
            if not beam:
                break

    best = next((n for n in leaves if n.status == "leaf"), None)
    if best is None:
        return PlanResult("needs_review" if discrepancies else "no_compliant_plan",
                          "No branch satisfied the hard constraints and the 0.65 score floor.",
                          None, tuple(leaves), tuple(trace), evaluations, input_hash,
                          discrepancy_node_ids=tuple(discrepancies))
    status = "needs_review" if overflow or discrepancies else "awaiting_approval"
    message = ("Critic and deterministic scores diverged; both branches kept for human review."
               if discrepancies else "Top plan ready for household approval (scenario only; no money moves).")
    return PlanResult(status, message, best, tuple(leaves), tuple(trace), evaluations, input_hash,
                      discrepancy_node_ids=tuple(discrepancies))
