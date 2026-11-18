"""Brokerage eval areas: the two calculations the demo's brokerage chips ask for, checked against an independent oracle.

- brokerage_performance:   random monthly statements (and the demo's own three months) run through the real ledger and
                           tool layer; opening, closing, deposits, dividends, market effect and its percentage must equal
                           an oracle built on exact fractions, the months must chain (a gap is refused), and the demo's
                           deposits must equal the ``ACME BROKERAGE TRANSFER`` amounts the checking history is built from
- brokerage_concentration: shares, ranking, technology share and the limit flags must equal the oracle, with the limits
                           read from the written policy document (not from ``LIMITS_BPS``), so a policy that says one
                           thing while the calculation flags another fails. Positions sit exactly on, one cent under
                           and one cent over each limit, and a fund over 20% must not be flagged as a single stock.

The oracle never imports ``brokerage.py`` logic: only its data classes (to build a statement) and the policy text it
publishes are used, and every figure is recomputed here from the quantities and prices.
"""

import random
import re
import sqlite3
from fractions import Fraction
from functools import lru_cache
from typing import Any

from household_agent.audit import AuditLog
from household_agent.authz import Authorizer
from household_agent.brokerage import (
    BROKERAGE_ACCOUNT_ID,
    BrokerageMonth,
    BrokerageStatementError,
    Holding,
    brokerage_documents,
    demo_brokerage_months,
)
from household_agent.categorize import Categorizer
from household_agent.history import CONTRIBUTION_BY_MONTH, history_statements
from household_agent.ledger import LedgerStore
from household_agent.synthetic import DEMO_ACCOUNTS, DEMO_AS_OF, RULES, SOMAL
from household_agent.tools import ToolSession

N_RANDOM = 24  # random portfolios per area; the boundary and demo cases come on top of these

_SECTORS = ("technology", "healthcare", "energy", "financials", "consumer", "utilities")
_FUND_SECTORS = ("broad_market", "international", "bonds")
_TICKERS = ("AXLR", "BRVN", "CLDX", "DRMT", "EQNS", "FLUX", "GRNT", "HLXA", "IVRY", "JUNO", "KRTL", "LMNA", "MRDN",
            "NVTR", "ORBX", "PLSR", "QNTA", "RVLT", "SLCA", "TRNX", "UMBR", "VLTA", "WYND", "XPRT", "YNDR", "ZNTH")


# --- the oracle (exact fractions; nothing here calls brokerage.py) ----------------------------------------------

def _tenth_pct(f: Fraction) -> str:
    """A ratio (0.231 is 23.1 percent) as a percentage with one decimal, half away from zero ("-5.8")."""
    tenths = (abs(f) * 1000 * 2 + 1) // 2  # floor(|f| * 1000 + 1/2): tenths of a percent
    return f"{'-' if f < 0 else ''}{tenths // 10}.{tenths % 10}"


def _value(h: dict[str, Any]) -> int:
    return h["quantity"] * h["price_minor"]


def _closing(m: dict[str, Any]) -> int:
    return m["cash"] + sum(_value(h) for h in m["holdings"])


def policy_limits() -> tuple[Fraction, Fraction]:
    """(single stock, technology) limits as fractions of the portfolio, read from the policy document's own sentences."""
    text = next(d.text for d in brokerage_documents() if d.doc_id == "brokerage_concentration_policy")
    single = re.search(r"single stock may be more than (\d+) percent", text)
    sector = re.search(r"Technology holdings together may not be more than (\d+) percent", text)
    if not (single and sector):
        raise ValueError("the policy document does not state both limits in the expected sentences")
    return Fraction(int(single.group(1)), 100), Fraction(int(sector.group(1)), 100)


def oracle_performance(m: dict[str, Any]) -> dict[str, Any]:
    closing = _closing(m)
    market = closing - m["opening"] - m["deposits"] - m["dividends"]
    return {"period": m["period"], "opening_minor": m["opening"], "closing_minor": closing,
            "change_minor": closing - m["opening"], "deposits_minor": m["deposits"], "dividends_minor": m["dividends"],
            "market_effect_minor": market, "market_effect_pct": _tenth_pct(Fraction(market, m["opening"]))}


def oracle_concentration(m: dict[str, Any]) -> dict[str, Any]:
    """Everything except the limit flags, which depend on the policy and are decided when the case is scored."""
    total = _closing(m)
    ranked = sorted(m["holdings"], key=lambda h: (-_value(h), h["symbol"]))
    pct = lambda v: _tenth_pct(Fraction(v, total))  # noqa: E731
    tech = sum(_value(h) for h in m["holdings"] if h["sector"] == "technology")
    return {"period": m["period"], "total_minor": total,
            "positions": [[h["symbol"], _value(h), pct(_value(h))] for h in ranked],
            "largest_symbol": ranked[0]["symbol"], "largest_share_pct": pct(_value(ranked[0])),
            "top4_symbols": [h["symbol"] for h in ranked[:4]], "top4_share_pct": pct(sum(_value(h) for h in ranked[:4])),
            "technology_value_minor": tech, "technology_share_pct": pct(tech),
            "stock_values": {h["symbol"]: _value(h) for h in ranked if h["kind"] == "stock"}}


# --- scenario generation ----------------------------------------------------------------------------------------

def _holding(symbol: str, sector: str, kind: str, quantity: int, price_minor: int) -> dict[str, Any]:
    return {"symbol": symbol, "name": f"{symbol} (SYNTHETIC)", "sector": sector, "kind": kind, "quantity": quantity,
            "price_minor": price_minor}


def _random_holdings(rng: random.Random, symbols: list[str]) -> list[dict[str, Any]]:
    out = []
    for s in symbols:
        if rng.random() < 0.3:
            out.append(_holding(s, rng.choice(_FUND_SECTORS + ("technology",)), "etf", rng.randrange(5, 200),
                                rng.randrange(2_000, 40_000)))
        else:
            out.append(_holding(s, rng.choice(_SECTORS), "stock", rng.randrange(1, 120), rng.randrange(500, 90_000)))
    return out


def _month(period: str, opening: int, deposits: int, dividends: int, cash: int, holdings: list[dict[str, Any]]) -> dict[str, Any]:
    return {"period": period, "opening": opening, "deposits": deposits, "dividends": dividends, "cash": cash,
            "holdings": holdings}


def _random_month(rng: random.Random, period: str, opening: int | None) -> dict[str, Any]:
    holdings = _random_holdings(rng, rng.sample(_TICKERS, rng.randrange(5, 13)))
    worth = sum(_value(h) for h in holdings)
    return _month(period, opening if opening is not None else max(1, int(worth * rng.uniform(0.6, 1.4))),
                  rng.choice([0, 50_000, 150_000, rng.randrange(0, 300_000)]), rng.choice([0, rng.randrange(0, 20_000)]),
                  rng.randrange(0, max(2, worth // 5)), holdings)


def _chain(rng: random.Random, periods: tuple[str, str]) -> list[dict[str, Any]]:
    first = _random_month(rng, periods[0], None)
    return [first, _random_month(rng, periods[1], _closing(first))]


def _tie_month(rng: random.Random, negative: bool) -> dict[str, Any]:
    """A month whose market effect is exactly a half of the last shown decimal (x.x5 percent), so the rounding rule
    decides the answer: opening = 2000 t and market = ±t (2 j + 1) give (2 j + 1) / 20 percent."""
    holdings = _random_holdings(rng, rng.sample(_TICKERS, 6))
    worth = sum(_value(h) for h in holdings)
    t, j = worth // 1000 + 50 + rng.randrange(100), rng.randrange(50)
    deposits, dividends = rng.randrange(0, 50_000), rng.randrange(0, 5_000)
    market = (-1 if negative else 1) * t * (2 * j + 1)
    return _month("2026-10", 2000 * t, deposits, dividends, 2000 * t + deposits + dividends + market - worth, holdings)


def _demo_months() -> list[dict[str, Any]]:
    return [_month(m.period, m.opening_minor, m.deposits_minor, m.dividends_minor, m.cash_minor,
                   [_holding(h.symbol, h.sector, h.kind, h.quantity, h.price_minor) | {"name": h.name} for h in m.holdings])
            for m in demo_brokerage_months()]


@lru_cache(maxsize=1)
def _history_transfers() -> dict[str, int]:
    """Cents moved to the brokerage per month in the checking statements the demo holds (only the months it holds)."""
    out: dict[str, int] = {}
    for name, _member, text in history_statements():
        if "somal_checking" not in name:
            continue
        period = next(ln for ln in text.splitlines() if ln.startswith("# period_start:")).split(":", 1)[1].strip()[:7]
        for ln in text.splitlines():
            if ",ACME BROKERAGE TRANSFER," in ln:
                out[period] = out.get(period, 0) - int(Fraction(ln.rsplit(",", 1)[1]) * 100)
    return out


def _boundary_month(rng: random.Random, kind: str) -> dict[str, Any]:
    """A portfolio built so one position (or the technology total) lands exactly on, a cent under or a cent over a limit."""
    single, sector = policy_limits()
    others = [{**h, "sector": "healthcare" if h["kind"] == "stock" else "bonds"}  # nothing technology, so only the subject counts
              for h in _random_holdings(rng, rng.sample(_TICKERS, 5))]
    rest_value = sum(_value(h) for h in others)
    where, _, edge = kind.rpartition("_")  # e.g. "stock_at", "tech_over"
    delta = {"under": -1, "at": 0, "over": 1}[edge]
    limit = single if where == "stock" else sector
    n, d = limit.numerator, limit.denominator
    # position / (position + R) == n / d  <=>  position == R n / (d - n); R = k (d - n) makes that an integer
    k = rest_value // (d - n) + 1 + rng.randrange(50)
    rest, position = k * (d - n), k * n
    cash = rest - rest_value
    if where == "stock":
        holdings = others + [_holding("LMTX", "healthcare", "stock", 1, position + delta)]
    else:
        half = position // 2
        holdings = others + [_holding("TCHA", "technology", "stock", 1, half + delta),
                             _holding("TCHB", "technology", "etf", 1, position - half)]
    return _month("2026-10", max(1, rest_value + position), 0, 0, cash, holdings)


def _fund_over_limit(rng: random.Random) -> dict[str, Any]:
    stocks = [_holding(s, "healthcare", "stock", rng.randrange(1, 30), rng.randrange(1_000, 20_000)) for s in _TICKERS[:4]]
    big = sum(_value(h) for h in stocks) * 2
    return _month("2026-10", big, 0, 0, 1_000, stocks + [_holding("BIGF", "broad_market", "etf", 1, big)])


def brokerage_cases(seed: int) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[tuple[dict[str, Any], dict[str, Any]]]]:
    """(performance cases, concentration cases) as ``(payload, expected)``. Own generator, so adding them moves no other area."""
    rng = random.Random(f"brokerage-{seed}")
    perf: list[tuple[dict[str, Any], dict[str, Any]]] = []
    conc: list[tuple[dict[str, Any], dict[str, Any]]] = []
    periods = ("2026-04", "2026-05")

    def add_perf(kind: str, months: list[dict[str, Any]], **extra: Any) -> None:
        perf.append(({"kind": kind, "months": months, **extra}, {"months": [oracle_performance(m) for m in months]}))

    def add_conc(kind: str, months: list[dict[str, Any]]) -> None:
        conc.append(({"kind": kind, "months": months}, {"months": [oracle_concentration(m) for m in months]}))

    for i in range(N_RANDOM):
        add_perf("random_chain_with_gap" if i % 6 == 5 else "random_chain", _chain(rng, periods), gap=i % 6 == 5)
        add_conc("random", _chain(rng, periods))
    for negative in (False, True, False, True):
        add_perf("rounding_tie_" + ("down" if negative else "up"), [_tie_month(rng, negative)])
    demo = _demo_months()
    transfers = _history_transfers()
    rule = {m["period"]: int(Fraction(CONTRIBUTION_BY_MONTH(m["period"])) * 100) for m in demo}
    add_perf("demo_months", demo, rule_deposits=rule, ledger_deposits={p: transfers[p] for p in rule if p in transfers})
    add_conc("demo_months", demo)
    for kind in ("stock_under", "stock_at", "stock_over", "tech_under", "tech_at", "tech_over"):
        for _ in range(2):
            add_conc("boundary_" + kind, [_boundary_month(rng, kind)])
    for _ in range(2):
        add_conc("fund_over_limit", [_fund_over_limit(rng)])
    return perf, conc


# --- scoring ----------------------------------------------------------------------------------------------------

def _to_month(m: dict[str, Any]) -> BrokerageMonth:
    return BrokerageMonth(BROKERAGE_ACCOUNT_ID, m["period"], m["opening"], m["deposits"], m["dividends"], m["cash"],
                          tuple(Holding(**h) for h in m["holdings"]))


def _world(months: list[dict[str, Any]], corr: str) -> tuple[LedgerStore, ToolSession, list[str]]:
    conn = sqlite3.connect(":memory:")
    store = LedgerStore(conn, Authorizer(DEMO_ACCOUNTS), AuditLog(conn))
    problems = []
    for m in months:
        try:
            store.ingest_brokerage_month(SOMAL, "statement_ingestion", corr, _to_month(m), at=DEMO_AS_OF)
        except BrokerageStatementError as exc:
            problems.append(f"{m['period']} was refused: {exc}")
    session = ToolSession(store, Categorizer(RULES), SOMAL, "historical_comparison", corr, clock=lambda: DEMO_AS_OF,
                          acting_agent="ledger_analyst")
    return store, session, problems


def _run(store: LedgerStore, session: ToolSession, fn: str, period: str, problems: list[str]) -> dict[str, Any] | None:
    out = session.call(fn, {"account_id": BROKERAGE_ACCOUNT_ID, "month": period})
    if not out.get("ok"):
        problems.append(f"{fn} {period} failed: {out.get('error')}")
        return None
    row = store.conn.execute("SELECT output_json, output_hash FROM calculation_runs WHERE run_id = ?", (out["run_id"],)).fetchone()
    if row is None or row[1] != out["output_hash"]:
        problems.append(f"{fn} {period} left no recorded run with the same output hash")
    return out["output"]


def performance_problems(payload: dict[str, Any], expected: dict[str, Any], corr: str) -> list[str]:
    months = payload["months"]
    store, session, problems = _world(months, corr)
    try:
        if payload.get("gap"):  # a month that does not open where the previous one closed must not be stored
            bad = {**months[-1], "period": "2026-06", "opening": _closing(months[-1]) + 1}
            try:
                store.ingest_brokerage_month(SOMAL, "statement_ingestion", corr, _to_month(bad), at=DEMO_AS_OF)
                problems.append("a month opening one cent off the previous close was stored")
            except BrokerageStatementError:
                pass
        outs = []
        for m, want in zip(months, expected["months"]):
            got = _run(store, session, "brokerage_performance", m["period"], problems)
            outs.append(got)
            if got is None:
                continue
            for key, value in want.items():
                if got.get(key) != value:
                    problems.append(f"{m['period']} {key}: got {got.get(key)!r}, want {value!r}")
            if got.get("deposits_minor", 0) + got.get("dividends_minor", 0) + got.get("market_effect_minor", 0) != got.get("change_minor"):
                problems.append(f"{m['period']}: deposits + dividends + market effect do not add up to the change")
        for a, b in zip(outs, outs[1:]):
            if a and b and b["opening_minor"] != a["closing_minor"]:
                problems.append("a month does not open at the previous month's closing value")
        latest = session.call("brokerage_performance", {"account_id": BROKERAGE_ACCOUNT_ID})
        if not latest.get("ok") or latest["output"]["period"] != months[-1]["period"]:
            problems.append("the default month is not the latest statement")
        for period, minor in payload.get("rule_deposits", {}).items():
            got = next((o for o in outs if o and o["period"] == period), None)
            if got and got["deposits_minor"] != minor:
                problems.append(f"{period}: deposits {got['deposits_minor']} != the {minor} the checking history transfers")
        for period, minor in payload.get("ledger_deposits", {}).items():
            got = next((o for o in outs if o and o["period"] == period), None)
            if got and got["deposits_minor"] != minor:
                problems.append(f"{period}: deposits {got['deposits_minor']} != the {minor} in the ledger rows")
    finally:
        store.conn.close()
    return problems


def concentration_problems(payload: dict[str, Any], expected: dict[str, Any], corr: str) -> list[str]:
    months = payload["months"]
    store, session, problems = _world(months, corr)
    try:
        single, sector = policy_limits()
        for m, want in zip(months, expected["months"]):
            got = _run(store, session, "brokerage_concentration", m["period"], problems)
            if got is None:
                continue
            total = want["total_minor"]
            checks = {
                "total_minor": want["total_minor"], "positions": want["positions"],
                "largest_symbol": want["largest_symbol"], "largest_share_pct": want["largest_share_pct"],
                "top4_symbols": want["top4_symbols"], "top4_share_pct": want["top4_share_pct"],
                "technology_value_minor": want["technology_value_minor"],
                "technology_share_pct": want["technology_share_pct"],
                "stocks_over_single_stock_limit": sorted(s for s, v in want["stock_values"].items()
                                                         if Fraction(v, total) > single),
                "technology_over_limit": Fraction(want["technology_value_minor"], total) > sector,
                "limits_pct": {"single_stock": str(single * 100), "technology_sector": str(sector * 100)},
            }
            reported = {**got, "positions": [[p["symbol"], p["value_minor"], p["share_pct"]] for p in got["positions"]],
                        "stocks_over_single_stock_limit": sorted(got["stocks_over_single_stock_limit"])}
            for key, value in checks.items():
                if reported.get(key) != value:
                    problems.append(f"{m['period']} {key}: got {reported.get(key)!r}, want {value!r}")
    finally:
        store.conn.close()
    return problems
