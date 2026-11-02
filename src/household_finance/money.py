"""Money conversion helpers.

Money is always integer minor units (cents). Floats are rejected at the boundary so
binary rounding error can never enter a ledger total.
"""

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

CENT = Decimal("0.01")
PERCENT_QUANTUM = Decimal("0.01")


class MoneyError(ValueError):
    """Raised when a value cannot be represented exactly as minor units."""


def to_minor(amount: str | Decimal) -> int:
    """Convert a decimal amount such as ``"-2,750.00"`` or ``Decimal("42.99")`` to cents.

    Rejects floats, non-finite values, and anything with more than two decimal places.
    """
    if isinstance(amount, (float, bool, int)):
        raise MoneyError(f"expected str or Decimal, got {type(amount).__name__}")
    if isinstance(amount, str):
        text = amount.strip().replace(",", "").replace("$", "")
        try:
            value = Decimal(text)
        except InvalidOperation as exc:
            raise MoneyError(f"not a decimal amount: {amount!r}") from exc
    else:
        value = amount
    if not value.is_finite():
        raise MoneyError(f"non-finite amount: {amount!r}")
    if value != value.quantize(CENT):
        raise MoneyError(f"more than two decimal places: {amount!r}")
    return int(value * 100)


def format_minor(minor: int) -> str:
    """Format cents for display, e.g. ``-45000`` -> ``"-$450.00"``."""
    sign = "-" if minor < 0 else ""
    return f"{sign}${Decimal(abs(minor)) / 100:,.2f}"


def percent(numerator_minor: int, denominator_minor: int) -> Decimal:
    """Exact percentage rounded half-up to two places. Raises on a zero denominator."""
    if denominator_minor == 0:
        raise ZeroDivisionError("percentage denominator is zero")
    ratio = Decimal(numerator_minor) / Decimal(denominator_minor) * 100
    return ratio.quantize(PERCENT_QUANTUM, rounding=ROUND_HALF_UP)
