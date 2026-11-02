"""Versioned, deterministic household finance library (design §3.4)."""

from household_finance.calcs import (
    AmortizationRow,
    CalculationError,
    CashFlow,
    DriftFinding,
    NetWorth,
    Variance,
    amortization_schedule,
    cash_flow,
    category_totals,
    net_worth_snapshot,
    recurring_drift,
    variance,
)
from household_finance.money import MoneyError, format_minor, percent, to_minor

__version__ = "0.2.0"

__all__ = [
    "AmortizationRow",
    "CalculationError",
    "CashFlow",
    "DriftFinding",
    "MoneyError",
    "NetWorth",
    "Variance",
    "__version__",
    "amortization_schedule",
    "cash_flow",
    "category_totals",
    "format_minor",
    "net_worth_snapshot",
    "percent",
    "recurring_drift",
    "to_minor",
    "variance",
]
