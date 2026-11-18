"""A fictional brokerage account: monthly statements with holdings, and the two calculations the demo asks about.

The ledger holds bank transactions; a brokerage statement is a different shape (positions, not rows), so it has its own
small tables (see ``LedgerStore.ingest_brokerage_month``) and its own calculations. They follow the same rules as the
ledger ones: the model names a function and an account and month, never a number; the figures are computed here in
integer cents; every run is recorded with an input hash and an output hash so a claim can cite it.

The account is invented: the quantities, prices and amounts are fictional and do not describe any real account. The
tickers are real, widely held symbols chosen so a viewer can look them up; they were not taken from anyone's statements.
"""

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from household_agent.retrieval import SourceDocument

BROKERAGE_ACCOUNT_ID = "acct_somal_brokerage"
BROKERAGE_HOUSEHOLD = "hh_demo"
CASH = "cash"

# The household's concentration policy, in basis points of the portfolio. The policy document the Policy agent retrieves
# is written from these numbers, and the calculation below flags against the same numbers, so the two cannot disagree.
LIMITS_BPS = {"single_stock": 2000, "technology_sector": 5000}


class BrokerageStatementError(ValueError):
    """A brokerage statement that cannot be stored (does not continue the previous month, or is malformed)."""


@dataclass(frozen=True)
class Holding:
    symbol: str
    name: str
    sector: str
    kind: str  # "stock" or "etf"
    quantity: int
    price_minor: int

    @property
    def value_minor(self) -> int:
        return self.quantity * self.price_minor


@dataclass(frozen=True)
class BrokerageMonth:
    """One monthly statement. ``closing_minor`` is derived (cash plus every position), so a statement always adds up;
    what can go wrong is the month before it, which ``LedgerStore.ingest_brokerage_month`` checks."""
    account_id: str
    period: str  # YYYY-MM
    opening_minor: int
    deposits_minor: int
    dividends_minor: int
    cash_minor: int
    holdings: tuple[Holding, ...]

    @property
    def closing_minor(self) -> int:
        return self.cash_minor + sum(h.value_minor for h in self.holdings)

    @property
    def statement_id(self) -> str:
        return f"bstmt_{self.account_id}_{self.period}"


# --- calculations -----------------------------------------------------------------------------------------------

def _pct(numerator: int, denominator: int) -> str:
    """``numerator / denominator`` as a percentage with one decimal, rounded half up ("-5.8")."""
    return str((Decimal(numerator) * 100 / Decimal(denominator)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def performance(m: BrokerageMonth) -> dict[str, Any]:
    """Opening, closing, what was deposited, and what the market (and everything else) did, in cents.

    ``market_effect_minor`` is what is left of the change after deposits and dividends: closing - opening - deposits -
    dividends. It is negative when the account lost value."""
    market = m.closing_minor - m.opening_minor - m.deposits_minor - m.dividends_minor
    return {
        "period": m.period, "statement_id": m.statement_id, "opening_minor": m.opening_minor,
        "closing_minor": m.closing_minor, "change_minor": m.closing_minor - m.opening_minor,
        "deposits_minor": m.deposits_minor, "dividends_minor": m.dividends_minor, "market_effect_minor": market,
        "market_effect_pct": _pct(market, m.opening_minor), "market_effect_note": "percent of the opening value",
    }


def concentration(m: BrokerageMonth) -> dict[str, Any]:
    """How much of the portfolio each position is, the largest, the top four, the technology share, and which of them
    break the household's limits. Shares are of the whole portfolio, cash included."""
    total = m.closing_minor
    if total <= 0:
        raise BrokerageStatementError(f"{m.period}: the portfolio has no value to measure")
    invested = sorted((h for h in m.holdings if h.sector != CASH), key=lambda h: (-h.value_minor, h.symbol))
    positions = [{"symbol": h.symbol, "name": h.name, "sector": h.sector, "kind": h.kind, "value_minor": h.value_minor,
                  "share_pct": _pct(h.value_minor, total)} for h in invested]
    tech = sum(h.value_minor for h in invested if h.sector == "technology")
    top4 = invested[:4]
    over = [h.symbol for h in invested
            if h.kind == "stock" and h.value_minor * 10_000 > LIMITS_BPS["single_stock"] * total]
    return {
        "period": m.period, "statement_id": m.statement_id, "total_minor": total, "positions": positions,
        "largest_symbol": invested[0].symbol, "largest_share_pct": positions[0]["share_pct"],
        "top4_symbols": [h.symbol for h in top4], "top4_share_pct": _pct(sum(h.value_minor for h in top4), total),
        "technology_value_minor": tech, "technology_share_pct": _pct(tech, total),
        "limits_pct": {k: str(Decimal(v) / 100) for k, v in LIMITS_BPS.items()},
        "stocks_over_single_stock_limit": over,
        "technology_over_limit": tech * 10_000 > LIMITS_BPS["technology_sector"] * total,
    }


BROKERAGE_CALCS = {"brokerage_performance": performance, "brokerage_concentration": concentration}


# --- synthetic statements -----------------------------------------------------------------------------------------

def _h(symbol: str, name: str, sector: str, kind: str, quantity: int, price: str) -> Holding:
    dollars, _, cents = price.partition(".")
    return Holding(symbol, name, sector, kind, quantity, int(dollars) * 100 + int(cents.ljust(2, "0")))


def _holdings(prices: tuple[str, ...], voo_qty: int) -> tuple[Holding, ...]:
    """Eleven positions (four technology stocks, three other stocks, four funds) priced for one month. The monthly deposit
    buys more of the broad-market fund, so only its quantity changes from month to month."""
    nvda, msft, aapl, amd, jnj, xom, jpm, voo, vxus, schd, bnd = prices
    return (
        _h("NVDA", "NVIDIA Corp", "technology", "stock", 42, nvda),
        _h("MSFT", "Microsoft Corp", "technology", "stock", 26, msft),
        _h("AAPL", "Apple Inc", "technology", "stock", 34, aapl),
        _h("AMD", "Advanced Micro Devices", "technology", "stock", 18, amd),
        _h("JNJ", "Johnson & Johnson", "healthcare", "stock", 20, jnj),
        _h("XOM", "Exxon Mobil Corp", "energy", "stock", 28, xom),
        _h("JPM", "JPMorgan Chase & Co", "financials", "stock", 35, jpm),
        _h("VOO", "Vanguard S&P 500 ETF", "broad_market", "etf", voo_qty, voo),
        _h("VXUS", "Vanguard Total International Stock ETF", "international", "etf", 40, vxus),
        _h("SCHD", "Schwab US Dividend Equity ETF", "broad_market", "etf", 25, schd),
        _h("BND", "Vanguard Total Bond Market ETF", "bonds", "etf", 55, bnd),
    )


def demo_brokerage_months() -> list[BrokerageMonth]:
    """August to October 2026. The deposits are the 1,500.00 ``ACME BROKERAGE TRANSFER`` rows in Somal's checking
    statements, and the September closing value is the one the quarterly summary in ``history`` states (31,334.23), so the
    two sources agree. October is a bad month: the account ends below where it began even after a deposit."""
    a = BROKERAGE_ACCOUNT_ID
    return [
        BrokerageMonth(a, "2026-08", 2895000, 150000, 0, 187340, _holdings(
            ("191.20", "138.50", "93.40", "61.20", "84.00", "41.00", "47.50", "108.50", "47.30", "69.00", "52.10"), 8)),
        BrokerageMonth(a, "2026-09", 2962300, 150000, 8412, 200823, _holdings(
            ("190.00", "140.00", "95.00", "60.00", "85.00", "42.00", "48.00", "110.00", "48.00", "70.00", "52.00"), 21)),
        BrokerageMonth(a, "2026-10", 3133423, 150000, 3140, 210973, _holdings(
            ("171.00", "126.00", "86.00", "54.00", "84.00", "40.00", "47.00", "108.00", "47.00", "69.00", "52.00"), 34)),
    ]


def _money(minor: int) -> str:
    return f"{'-' if minor < 0 else ''}{abs(minor) // 100}.{abs(minor) % 100:02d}"


def statement_text(m: BrokerageMonth) -> str:
    """The statement as retrievable text: header lines, then one ``symbol,name,sector,quantity,price,value`` row each."""
    lines = [f"# account_id: {m.account_id}", f"# period: {m.period}", f"# opening_value: {_money(m.opening_minor)}",
             f"# deposits: {_money(m.deposits_minor)}", f"# dividends: {_money(m.dividends_minor)}",
             f"# closing_value: {_money(m.closing_minor)}", f"# cash: {_money(m.cash_minor)}",
             "symbol,name,sector,quantity,price,value"]
    lines += [f"{h.symbol},{h.name},{h.sector},{h.quantity},{_money(h.price_minor)},{_money(h.value_minor)}"
              for h in m.holdings]
    return "\n".join(lines)


def policy_text() -> str:
    single, sector = (LIMITS_BPS[k] // 100 for k in ("single_stock", "technology_sector"))
    return ("# Concentration limits\n"
            f"No single stock may be more than {single} percent of the brokerage portfolio. "
            "Broad-market, international and bond funds are not single stocks and are exempt from this limit.\n"
            f"Technology holdings together may not be more than {sector} percent of the portfolio.\n"
            "# What a breach means\n"
            "A position above a limit is flagged for review at the next monthly statement. It is never sold automatically; "
            "trimming it is a decision for the account owner.")


def brokerage_documents() -> list[SourceDocument]:
    """Monthly statements (``financial_history``) and the concentration policy (``policies_compliance``), all scoped to
    the brokerage account so only its owner, or someone the owner shared it with, can retrieve them."""
    docs = [SourceDocument(
        doc_id=f"brokerage_stmt_{m.period}", household_id=BROKERAGE_HOUSEHOLD, collection="financial_history",
        account_id=m.account_id, source_name=f"Brokerage statement {m.period} (SYNTHETIC)", doc_type="brokerage_statement",
        authority_tier=2, publisher="ACME Brokerage (SYNTHETIC)", statement_period=m.period, text=statement_text(m))
        for m in demo_brokerage_months()]
    docs.append(SourceDocument(
        doc_id="brokerage_concentration_policy", household_id=BROKERAGE_HOUSEHOLD, collection="policies_compliance",
        account_id=BROKERAGE_ACCOUNT_ID, source_name="Brokerage concentration policy", doc_type="investment_policy",
        authority_tier=2, publisher="Household", jurisdiction="US", effective_from=date(2026, 1, 1), text=policy_text()))
    return docs
