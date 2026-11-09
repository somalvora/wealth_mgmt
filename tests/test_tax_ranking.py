"""Tax-aware ranking: an optional, host-stated flat pre-tax rate per retirement account."""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from household_agent.allocation import (
    CandidatePlan,
    Debt,
    HouseholdFinancialState,
    RetirementAccount,
    evaluate_candidates,
)
from household_agent.allocation_tool import evaluate_allocation_candidates
from household_agent.synthetic import ALEX, DEMO_AS_OF
from tests.test_allocation import facts

LOW_APR = Debt(debt_id="card", balance_minor=500_000, apr_percent="2", minimum_payment_minor=20_000)
X = 200_000


def _state(rate="0", **kw):
    base = dict(monthly_essential_expenses_minor=300_000, liquid_cash_minor=1_000_000,
                emergency_fund_balance_minor=1_800_000, emergency_fund_target_months="6",  # fund already at target
                min_cash_buffer_months="3", debts=(LOW_APR,), horizon_months=12,
                retirement_accounts=(RetirementAccount(account_id="k401", headroom_minor=700_000,
                                                       tax_deferred_rate_percent=rate),))
    return HouseholdFinancialState(**{**base, **kw})


PAY = CandidatePlan(name="pay_debt", debt_principal_minor={"card": X})
RET = CandidatePlan(name="to_401k", retirement_minor={"k401": X})


def _by_name(ev):
    return {c.name: c for c in ev.candidates}


def test_tax_benefit_is_hand_computed():
    # $2,000 pre-tax at 22% = $440.00, shown in the rationale and used in the ranking key.
    o = _by_name(evaluate_candidates(_state("22"), X, [RET]))["to_401k"]
    assert o.tax_benefit_minor == 44_000 and "$440.00" in o.rationale
    assert o.score_key[2] == -44_000


def test_tax_benefit_rounds_half_up_to_the_cent():
    # 1,005c x 12.5% = 125.625c -> 126c; 1,001c x 0.5% = 5.005c -> 5c.
    r = CandidatePlan(name="r", retirement_minor={"k401": 1_005})
    assert _by_name(evaluate_candidates(_state("12.5"), 1_005, [r]))["r"].tax_benefit_minor == 126
    r = CandidatePlan(name="r", retirement_minor={"k401": 1_001})
    assert _by_name(evaluate_candidates(_state("0.5"), 1_001, [r]))["r"].tax_benefit_minor == 5


def test_the_default_rate_is_zero_and_changes_nothing():
    assert RetirementAccount(account_id="a", headroom_minor=1).tax_deferred_rate_percent == 0
    ev = evaluate_candidates(_state(), X, [PAY, RET])
    c = _by_name(ev)
    assert c["to_401k"].tax_benefit_minor == 0 and "save an estimated" not in c["to_401k"].rationale
    assert ev.ranking[0] == "pay_debt"  # interest-only ordering, as before


def test_a_stated_tax_rate_can_move_retirement_ahead_of_a_cheap_debt():
    assert evaluate_candidates(_state("0"), X, [PAY, RET]).ranking == ("pay_debt", "to_401k")
    ev = evaluate_candidates(_state("30"), X, [PAY, RET])
    assert ev.ranking == ("to_401k", "pay_debt") and ev.evaluator_version == "alloc_eval_1.1"
    assert "interest saved plus tax benefit" in _by_name(ev)["pay_debt"].rationale


def test_safety_still_beats_tax_benefit():
    # Emergency fund below target: funding it outranks even a large tax benefit.
    ev = evaluate_candidates(_state("37", emergency_fund_balance_minor=600_000), X,
                             [RET, CandidatePlan(name="ef", emergency_fund_minor=X)])
    assert ev.ranking[0] == "ef"


def test_an_after_tax_account_earns_no_benefit():
    r = CandidatePlan(name="r", retirement_minor={"k401": 700_000})
    assert _by_name(evaluate_candidates(_state("0"), 700_000, [r]))["r"].tax_benefit_minor == 0


@pytest.mark.parametrize("bad", ["-1", "100.01", "NaN", "Infinity", "12.345"])
def test_invalid_tax_rates_are_rejected(bad):
    with pytest.raises(ValidationError):
        RetirementAccount(account_id="a", headroom_minor=1, tax_deferred_rate_percent=bad)


def test_the_rate_reaches_the_stored_run_through_the_tool(seeded_store):
    f = facts(retirement_accounts=[{"account_id": "roth", "headroom_minor": 700_000, "tax_deferred_rate_percent": "22"}])
    r = evaluate_allocation_candidates(
        seeded_store, ALEX, purpose="capital_allocation", correlation_id="alloc_tax", extra_amount_minor=X,
        candidates=[CandidatePlan(name="r", retirement_minor={"roth": X})], facts=f, at=DEMO_AS_OF)
    c = r["output"]["candidates"][0]
    assert r["ok"] and c["tax_benefit_minor"] == 44_000 and c["tax_benefit_display"] == "$440.00"
    assert any("flat pre-tax rate" in a for a in r["output"]["assumptions"])


@settings(max_examples=60, deadline=None)
@given(rate=st.decimals(min_value=0, max_value=100, places=2), amount=st.integers(0, 700_000))
def test_property_tax_benefit_is_bounded_by_the_contribution(rate, amount):
    plan = CandidatePlan(name="r", retirement_minor={"k401": amount})
    o = _by_name(evaluate_candidates(_state(str(rate)), max(amount, 1), [plan]))["r"]
    assert 0 <= o.tax_benefit_minor <= amount
    if rate == 0:
        assert o.tax_benefit_minor == 0


@settings(max_examples=60, deadline=None)
@given(rate=st.decimals(min_value=0, max_value=100, places=2), a=st.integers(0, 350_000), b=st.integers(0, 350_000))
def test_property_tax_benefit_never_falls_as_the_contribution_grows(rate, a, b):
    lo, hi = sorted((a, b))
    plans = [CandidatePlan(name="lo", retirement_minor={"k401": lo}),
             CandidatePlan(name="hi", retirement_minor={"k401": hi})]
    c = _by_name(evaluate_candidates(_state(str(rate)), 700_000, plans))
    assert c["lo"].tax_benefit_minor <= c["hi"].tax_benefit_minor


@settings(max_examples=40, deadline=None)
@given(rate=st.decimals(min_value=0, max_value=100, places=2), order=st.permutations([0, 1, 2]))
def test_property_ranking_with_tax_is_independent_of_candidate_order(rate, order):
    plans = [PAY, RET, CandidatePlan(name="hold", hold_minor=X)]
    ev = evaluate_candidates(_state(str(rate)), X, [plans[i] for i in order])
    assert ev.ranking == evaluate_candidates(_state(str(rate)), X, plans).ranking
