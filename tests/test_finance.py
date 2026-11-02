from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from household_finance import (
    CalculationError,
    MoneyError,
    amortization_schedule,
    cash_flow,
    category_totals,
    format_minor,
    net_worth_snapshot,
    recurring_drift,
    to_minor,
    variance,
)


@dataclass(frozen=True)
class Entry:
    posted_at: date
    amount_minor: int
    category: str = "misc"
    merchant_normalized: str = "Merchant"


OCT = (date(2026, 10, 1), date(2026, 10, 31))


# --- money ---------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "minor"),
    [("2,750.00", 275000), ("-42.99", -4299), ("$3,200", 320000), ("0.01", 1), ("-$15.00", -1500)],
)
def test_to_minor_parses_amounts(text, minor):
    assert to_minor(text) == minor


@pytest.mark.parametrize("bad", ["12.345", "abc", "NaN", 1.5, 10])
def test_to_minor_rejects_inexact_or_untyped_values(bad):
    with pytest.raises(MoneyError):
        to_minor(bad)


def test_format_minor():
    assert format_minor(-45000) == "-$450.00"
    assert format_minor(123456789) == "$1,234,567.89"


# --- variance (design §9 worked example) ---------------------------------

def test_variance_matches_design_trace():
    v = variance(275_000, 320_000)
    assert v.dollar_variance_minor == 45_000
    assert v.pct_variance == Decimal("16.36")


def test_variance_rejects_zero_prior():
    with pytest.raises(CalculationError):
        variance(0, 100)


@given(st.integers(-10**9, 10**9).filter(bool), st.integers(-10**9, 10**9).filter(bool))
def test_variance_dollar_amount_is_antisymmetric(a, b):
    assert variance(a, b).dollar_variance_minor == -variance(b, a).dollar_variance_minor


# --- cash flow and categories -------------------------------------------

def test_cash_flow_ignores_entries_outside_period():
    entries = [
        Entry(date(2026, 10, 1), 500_000),
        Entry(date(2026, 10, 15), -120_000),
        Entry(date(2026, 11, 1), -999_999),
    ]
    cf = cash_flow(entries, *OCT)
    assert (cf.inflow_minor, cf.outflow_minor, cf.net_minor) == (500_000, 120_000, 380_000)


def test_cash_flow_rejects_inverted_period():
    with pytest.raises(CalculationError):
        cash_flow([], date(2026, 10, 31), date(2026, 10, 1))


entries_strategy = st.lists(
    st.builds(
        Entry,
        posted_at=st.dates(date(2026, 9, 25), date(2026, 11, 5)),
        amount_minor=st.integers(-10**7, 10**7),
        category=st.sampled_from(["groceries", "rent", "utilities", "salary"]),
    ),
    max_size=60,
)


@given(entries_strategy)
def test_category_totals_sum_to_net_cash_flow(entries):
    assert sum(category_totals(entries, *OCT).values()) == cash_flow(entries, *OCT).net_minor


# --- net worth -------------------------------------------------------------

def test_net_worth_snapshot():
    nw = net_worth_snapshot({"checking": 1_250_000, "brokerage": 8_000_000, "card": -230_000, "mortgage": -40_000_000})
    assert nw.assets_minor == 9_250_000
    assert nw.liabilities_minor == 40_230_000
    assert nw.net_minor == -30_980_000


# --- recurring drift --------------------------------------------------------

def test_recurring_drift_flags_small_charges_growing_over_threshold():
    prior = [Entry(date(2025, m, 5), -999, merchant_normalized="StreamCo") for m in range(1, 13)]
    current = [Entry(date(2026, m, 5), -1299, merchant_normalized="StreamCo") for m in range(1, 13)]
    current += [Entry(date(2026, 3, 1), -899, merchant_normalized="NewApp")]
    findings = {f.merchant: f for f in recurring_drift(prior, current)}
    assert findings["StreamCo"].pct_change == Decimal("30.03")
    assert findings["StreamCo"].flagged
    assert findings["NewApp"].pct_change is None and not findings["NewApp"].flagged


def test_recurring_drift_ignores_large_charges():
    prior = [Entry(date(2025, 1, 1), -10_000, merchant_normalized="Gym")]
    current = [Entry(date(2026, 1, 1), -20_000, merchant_normalized="Gym")]
    assert recurring_drift(prior, current) == []


# --- amortization -----------------------------------------------------------

def test_amortization_30yr_mortgage():
    rows = amortization_schedule(40_000_000, Decimal("0.065"), 360)
    assert rows[0].payment_minor == 252_827  # $2,528.27 standard 6.5% / 30yr / $400k payment
    assert rows[-1].balance_minor == 0
    assert sum(r.principal_minor for r in rows) == 40_000_000


@given(
    st.integers(100_000, 200_000_000),
    st.decimals(Decimal("0"), Decimal("0.15"), places=4),
    st.integers(1, 480),
)
def test_amortization_always_repays_exact_principal(principal, rate, term):
    rows = amortization_schedule(principal, rate, term)
    assert rows[-1].balance_minor == 0
    assert sum(r.principal_minor for r in rows) == principal
    assert all(r.payment_minor == r.principal_minor + r.interest_minor for r in rows)


def test_amortization_rejects_float_rate():
    with pytest.raises(CalculationError):
        amortization_schedule(100_000, 0.05, 12)  # type: ignore[arg-type]
