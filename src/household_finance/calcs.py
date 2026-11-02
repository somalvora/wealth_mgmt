"""Deterministic finance functions (design §3.4).

Agents never do arithmetic; they ask the calculation runner to call one of these named,
versioned, tested functions. Inputs are integer cents; percentages are ``Decimal``.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from typing import Protocol

import numpy_financial as npf

from household_finance.money import percent


class CalculationError(ValueError):
    """Raised for inputs a calculation cannot handle (returned to the agent as an observation)."""


class LedgerEntry(Protocol):
    @property
    def posted_at(self) -> date: ...
    @property
    def amount_minor(self) -> int: ...
    @property
    def category(self) -> str: ...
    @property
    def merchant_normalized(self) -> str: ...


def _check_period(start: date, end: date) -> None:
    if end < start:
        raise CalculationError(f"period end {end} is before start {start}")


@dataclass(frozen=True)
class CashFlow:
    inflow_minor: int
    outflow_minor: int  # positive magnitude
    net_minor: int


def cash_flow(entries: Iterable[LedgerEntry], start: date, end: date) -> CashFlow:
    _check_period(start, end)
    inflow = outflow = 0
    for e in entries:
        if start <= e.posted_at <= end:
            if e.amount_minor >= 0:
                inflow += e.amount_minor
            else:
                outflow -= e.amount_minor
    return CashFlow(inflow, outflow, inflow - outflow)


def category_totals(entries: Iterable[LedgerEntry], start: date, end: date) -> dict[str, int]:
    """Signed total per category within the period, keys sorted for stable output."""
    _check_period(start, end)
    totals: dict[str, int] = defaultdict(int)
    for e in entries:
        if start <= e.posted_at <= end:
            totals[e.category] += e.amount_minor
    return dict(sorted(totals.items()))


@dataclass(frozen=True)
class NetWorth:
    assets_minor: int
    liabilities_minor: int  # positive magnitude
    net_minor: int


def net_worth_snapshot(balances_minor: Mapping[str, int]) -> NetWorth:
    """Balances are signed: assets positive, liabilities (cards, mortgages) negative."""
    assets = sum(b for b in balances_minor.values() if b > 0)
    liabilities = -sum(b for b in balances_minor.values() if b < 0)
    return NetWorth(assets, liabilities, assets - liabilities)


@dataclass(frozen=True)
class Variance:
    prior_minor: int
    current_minor: int
    dollar_variance_minor: int
    pct_variance: Decimal


def variance(prior_minor: int, current_minor: int) -> Variance:
    """Change from prior to current; the percentage is relative to ``|prior|``."""
    if prior_minor == 0:
        raise CalculationError("prior amount is zero; percentage variance is undefined")
    delta = current_minor - prior_minor
    return Variance(prior_minor, current_minor, delta, percent(delta, abs(prior_minor)))


@dataclass(frozen=True)
class DriftFinding:
    merchant: str
    prior_total_minor: int
    current_total_minor: int
    pct_change: Decimal | None
    flagged: bool


def recurring_drift(
    prior: Iterable[LedgerEntry],
    current: Iterable[LedgerEntry],
    small_charge_max_minor: int = 1500,
    increase_threshold_pct: Decimal = Decimal("20"),
) -> list[DriftFinding]:
    """Flag merchants whose small recurring debits grew more than the threshold year over year.

    Only debits smaller than ``small_charge_max_minor`` count. Merchants with no prior-year
    charges are reported unflagged; novelty is the cluster-risk rule's job, not this one.
    """

    def totals(entries: Iterable[LedgerEntry]) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for e in entries:
            if e.amount_minor < 0 and -e.amount_minor < small_charge_max_minor:
                out[e.merchant_normalized] += -e.amount_minor
        return out

    prior_totals, current_totals = totals(prior), totals(current)
    findings = []
    for merchant in sorted(current_totals):
        before, now = prior_totals.get(merchant, 0), current_totals[merchant]
        pct = percent(now - before, before) if before else None
        findings.append(
            DriftFinding(merchant, before, now, pct, pct is not None and pct > increase_threshold_pct)
        )
    return findings


@dataclass(frozen=True)
class AmortizationRow:
    period: int
    payment_minor: int
    interest_minor: int
    principal_minor: int
    balance_minor: int


def amortization_schedule(
    principal_minor: int, annual_rate: Decimal, term_months: int
) -> list[AmortizationRow]:
    """Fixed-rate monthly schedule.

    ``numpy-financial`` supplies the level payment (a float formula); everything after that is
    exact integer/Decimal math, and the final payment absorbs rounding so the balance ends at 0.
    """
    if principal_minor <= 0 or term_months <= 0:
        raise CalculationError("principal and term must be positive")
    if not isinstance(annual_rate, Decimal) or annual_rate < 0:
        raise CalculationError("annual_rate must be a non-negative Decimal, e.g. Decimal('0.065')")

    monthly_rate = annual_rate / 12
    if monthly_rate == 0:
        payment = -(-principal_minor // term_months)  # ceiling division
    else:
        level = -npf.pmt(float(monthly_rate), term_months, principal_minor / 100)
        payment = int((Decimal(str(level)) * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))

    rows, balance = [], principal_minor
    for period in range(1, term_months + 1):
        interest = int((Decimal(balance) * monthly_rate).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
        principal_part = payment - interest
        if period == term_months or principal_part > balance:
            principal_part = balance
        balance -= principal_part
        rows.append(AmortizationRow(period, principal_part + interest, interest, principal_part, balance))
        if balance == 0:
            break
    return rows
