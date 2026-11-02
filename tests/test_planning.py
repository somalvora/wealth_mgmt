from dataclasses import replace
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from household_finance.allocation import (
    AllocationStep,
    PlanningInputs,
    expand_step,
    score_allocation,
    validate_allocation,
)
from household_finance.calcs import CalculationError
from household_agent.planning import (
    BRANCHING,
    TemplateThoughtGenerator,
    Thought,
    plan_allocation,
)
from household_agent.synthetic import demo_planning_inputs

INPUTS = demo_planning_inputs()


# --- finance library: expansion, validation, scoring --------------------------------

def test_surplus_is_cash_minus_six_months_and_liabilities():
    assert INPUTS.required_reserve_minor == 6 * 700_000 + 320_000
    assert INPUTS.surplus_minor == 6_600_000 - 4_520_000


def test_split_apportions_exact_cents():
    step = AllocationStep(3, Decimal(1), (("brokerage", Decimal(1)), ("debt_prepayment", Decimal(2))))
    out = expand_step(INPUTS, {"liquidity_buffer": 1}, step)
    assert out["brokerage"] + out["debt_prepayment"] == INPUTS.surplus_minor - 1


def test_fill_in_order_respects_statutory_room():
    step = AllocationStep(2, Decimal(1), (("hsa", Decimal(1)), ("ira", Decimal(1)), ("plan_529", Decimal(1))), "fill_in_order")
    out = expand_step(INPUTS, {}, step)
    assert out == {"hsa": 400_000, "ira": 700_000, "plan_529": 500_000}


def test_float_fraction_rejected():
    with pytest.raises(CalculationError):
        expand_step(INPUTS, {}, AllocationStep(1, 0.5, (("liquidity_buffer", Decimal(1)),)))  # type: ignore[arg-type]


def test_hard_constraints():
    codes = lambda alloc, complete=True: {v.code for v in validate_allocation(INPUTS, alloc, complete=complete)}  # noqa: E731
    assert codes({"ira": 700_001}) == {"statutory_cap_exceeded"}
    assert codes({"brokerage": INPUTS.surplus_minor + 1}) == {"liquidity_floor_breached"}
    assert codes({"brokerage": 1}) == {"unallocated_surplus"}
    assert codes({"brokerage": 1}, complete=False) == set()
    assert codes({"crypto": 1}) == {"unknown_bucket"}
    unverified = replace(INPUTS, contribution_room_minor={"ira": 700_000})
    assert {v.code for v in validate_allocation(unverified, {"hsa": 1, "brokerage": INPUTS.surplus_minor - 1}, complete=True)} == {"limit_unverified"}


def test_noncompliant_plan_scores_zero():
    s = score_allocation(INPUTS, {"ira": 10_000_000})
    assert s.composite == 0 and not s.compliant


@settings(max_examples=60)
@given(st.lists(st.integers(min_value=0, max_value=10_000), min_size=6, max_size=6))
def test_compliant_scores_are_bounded(weights):
    buckets = ["liquidity_buffer", "brokerage", "debt_prepayment"]
    w = weights[:3] if sum(weights[:3]) else [1, 0, 0]
    step = AllocationStep(3, Decimal(1), tuple((b, Decimal(x)) for b, x in zip(buckets, w, strict=True)))
    alloc = expand_step(INPUTS, {}, step)
    s = score_allocation(INPUTS, alloc)
    assert s.compliant
    for x in (s.tax_efficiency, s.liquidity_safety, s.expected_net_yield, s.composite):
        assert Decimal(0) <= x <= Decimal(1)


# --- planner: beam search ----------------------------------------------------------

def test_beam_search_evaluates_exactly_fifteen_nodes():
    r = plan_allocation(INPUTS)
    assert r.evaluations == 15 == len(r.trace)
    assert sum(n.depth == 3 for n in r.trace) == 6


def test_best_plan_is_complete_compliant_and_needs_approval():
    r = plan_allocation(INPUTS)
    assert r.status == "awaiting_approval" and r.requires_approval
    assert sum(r.best.allocation.values()) == INPUTS.surplus_minor
    assert r.best.score.compliant
    assert all(n.score.composite >= Decimal("0.65") for n in r.ranked_leaves)


def test_tie_break_prefers_liquidity_within_margin():
    r = plan_allocation(INPUTS)
    top_raw = max(r.ranked_leaves, key=lambda n: n.score.composite)
    assert top_raw.score.composite - r.best.score.composite <= Decimal("0.02")
    assert r.best.score.liquidity_safety > top_raw.score.liquidity_safety
    assert r.best.node_id == "d1.balanced/d2.tax_aggressive/d3.tax_aggressive"


def test_intermediate_nodes_are_not_penalized_for_undecided_tiers():
    # Without the rollout, a depth-1 node has tax_efficiency 0 and could never reach 0.65.
    r = plan_allocation(INPUTS)
    depth1 = [n for n in r.trace if n.depth == 1]
    assert all(n.score.tax_efficiency > 0 for n in depth1)
    assert any(n.status == "selected" for n in depth1)


def test_planner_is_reproducible():
    a, b = plan_allocation(INPUTS), plan_allocation(INPUTS)
    assert a.input_hash == b.input_hash
    assert [(n.node_id, n.score.composite) for n in a.trace] == [(n.node_id, n.score.composite) for n in b.trace]


def test_not_eligible_without_surplus():
    r = plan_allocation(replace(INPUTS, liquid_cash_minor=INPUTS.required_reserve_minor))
    assert r.status == "not_eligible" and r.evaluations == 0 and r.best is None


class _OverCapGenerator(TemplateThoughtGenerator):
    """Simulates an LLM persona proposing a split that ignores statutory room."""

    version = "overcap_test"

    def propose(self, inputs, node, tier):
        thoughts = list(super().propose(inputs, node, tier))
        if tier == 2:
            thoughts[2] = Thought("tax_aggressive", AllocationStep(2, Decimal(1), (("ira", Decimal(1)),)),
                                  "Put everything in the IRA.")
        return thoughts


def test_cap_breaching_thought_is_hard_pruned():
    r = plan_allocation(INPUTS, _OverCapGenerator())
    pruned = [n for n in r.trace if n.status == "hard_pruned"]
    assert pruned and all(n.score.composite == 0 for n in pruned)
    assert {v.code for n in pruned for v in n.score.violations} == {"statutory_cap_exceeded"}
    assert all("d2.tax_aggressive" not in n.node_id for n in r.ranked_leaves)


def test_wrong_tier_and_wrong_bucket_are_hard_pruned():
    class Bad(TemplateThoughtGenerator):
        version = "bad"

        def propose(self, inputs, node, tier):
            t = list(super().propose(inputs, node, tier))
            if tier == 1:
                t[0] = Thought("conservative", AllocationStep(1, Decimal("0.5"), (("brokerage", Decimal(1)),)), "x")
                t[1] = Thought("balanced", AllocationStep(3, Decimal(1), (("brokerage", Decimal(1)),)), "x")
            return t

    r = plan_allocation(INPUTS, Bad())
    codes = {v.code for n in r.trace if n.status == "hard_pruned" for v in n.score.violations}
    assert codes == {"bucket_not_in_tier", "wrong_tier"}


def test_generator_must_propose_exactly_b_thoughts():
    class Two(TemplateThoughtGenerator):
        def propose(self, inputs, node, tier):
            return list(super().propose(inputs, node, tier))[:BRANCHING - 1]

    with pytest.raises(ValueError):
        plan_allocation(INPUTS, Two())


class _ContrarianCritic:
    """Strongly prefers the conservative cash buffer, disagreeing with the deterministic score."""

    def score(self, inputs, node):
        return Decimal("0.99") if node.node_id.startswith("d1.conservative") else node.score.composite


def test_critic_disagreement_keeps_branch_for_review_but_cannot_prune():
    base = plan_allocation(INPUTS)
    r = plan_allocation(INPUTS, critic=_ContrarianCritic())
    assert "d1.conservative" in r.discrepancy_node_ids
    kept = next(n for n in r.trace if n.node_id == "d1.conservative")
    assert kept.status == "retained_discrepancy" and kept.discrepancy
    assert r.status == "needs_review"
    # The critic never removes a branch the deterministic score selected.
    assert {n.node_id for n in base.ranked_leaves} <= {n.node_id for n in r.ranked_leaves}
    assert r.evaluations > base.evaluations
