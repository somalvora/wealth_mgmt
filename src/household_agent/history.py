"""Two years of fictional past statements, and the step that puts any ingested statement in the vector index.

The ledger (SQLite) holds the exact rows and answers "how much"; the vector collection ``financial_history`` holds the
same statements as text chunks so a question can find similar months by meaning. Everything here is invented.
History runs October 2024 to September 2026 (24 months), one statement per account per month, except two months the
demo already owns: the October 2025 rental statement (seeded separately) and Somal's September 2026 checking (the
deliberately corrupt sample).
"""

import calendar
import random

from household_agent.ingest import ParsedStatement
from household_agent.retrieval import SourceDocument
from household_agent.synthetic import HOUSEHOLD, Row, realistic_card_statement, render_statement

HISTORY_PREFIX = "hist_"  # file-name prefix: lets the demo list its own documents without these
HISTORY_FIRST = (2024, 10)
HISTORY_MONTHS = 24
SKIPPED = {("acct_rental_checking", "2025-10"), ("acct_somal_checking", "2026-09")}
RENTAL, SOMAL_CHECKING, SAYALI_CARD = "acct_rental_checking", "acct_somal_checking", "acct_sayali_card"
OWNER = {RENTAL: "somal", SOMAL_CHECKING: "somal", SAYALI_CARD: "sayali"}  # who uploads each account's statements
# Monthly transfer into the (fictional) brokerage from Somal's checking: 1,000 in year one, 1,500 in year two.
CONTRIBUTION_BY_MONTH = lambda ym: "1000.00" if ym < "2025-10" else "1500.00"  # noqa: E731


def contributed_minor(ym: str) -> int:
    """What the ledger holds for the month: nothing where Somal's statement is not part of the history."""
    return 0 if (SOMAL_CHECKING, ym) in SKIPPED else int(float(CONTRIBUTION_BY_MONTH(ym)) * 100)


def months() -> list[str]:
    y, m = HISTORY_FIRST
    out = []
    for _ in range(HISTORY_MONTHS):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def history_summary() -> str:
    """One line for the UI: what is already on file, so nobody wonders why the document list is short."""
    ms = months()
    label = lambda ym: calendar.month_abbr[int(ym[5:])] + " " + ym[:4]  # noqa: E731
    return f"History on file: {label(ms[0])} to {label(ms[-1])}, monthly statements for all three accounts."


def _money(cents: int) -> str:
    return f"{'-' if cents < 0 else ''}{abs(cents) // 100}.{abs(cents) % 100:02d}"


def _day(ym: str, d: int) -> str:
    return f"{ym}-{d:02d}"


def _rental_rows(ym: str, rng: random.Random) -> list[Row]:
    rent = "3000.00" if ym < "2025-04" else "3100.00"
    hoa = "410.00" if ym < "2025-04" else "425.00" if ym < "2026-04" else "450.00"
    rows = [Row(_day(ym, 1), "RENT DEPOSIT UNIT B", rent), Row(_day(ym, 5), "MAPLE ST HOA DUES", "-" + hoa),
            Row(_day(ym, 9), "CITY WATER UTILITY", "-" + _money(rng.randint(9500, 12500)))]
    if ym in ("2025-01", "2026-03"):
        rows.append(Row(_day(ym, 12), f"BAYSIDE PLUMBING INV {rng.randint(1000, 9999)}", "-640.00" if ym == "2025-01" else "-520.00"))
    tax = {"2024-10": "2600.00", "2025-04": "2600.00", "2026-04": "2750.00"}.get(ym)
    if tax:
        rows.append(Row(_day(ym, 18), "SAN MATEO TAX COLLECTOR - SECURED PROPERTY TAX - UNIT B", "-" + tax))
    return rows


def _somal_rows(ym: str, rng: random.Random) -> list[Row]:
    pay = "3950.00" if ym < "2025-04" else "4100.00"
    mortgage = "3400.00" if ym < "2025-10" else "3450.00"
    winter = int(ym[5:]) in (11, 12, 1, 2)
    return [
        Row(_day(ym, 1), "SUMMIT MORTGAGE PMT", "-" + mortgage), Row(_day(ym, 3), "PAYROLL ACME CORP", pay),
        Row(_day(ym, 4), "ACME BROKERAGE TRANSFER", "-" + CONTRIBUTION_BY_MONTH(ym)),
        Row(_day(ym, 11), "PG&E ENERGY BILL", "-" + _money(rng.randint(22000 if winter else 12000, 26000 if winter else 17000))),
        Row(_day(ym, 14), "SAFEWAY #1182", "-" + _money(rng.randint(12000, 20000))),
        Row(_day(ym, 17), "PAYROLL ACME CORP", pay), Row(_day(ym, 26), "SAFEWAY #1182", "-" + _money(rng.randint(9000, 16000))),
    ]


def _closing(csv_text: str) -> int:
    line = next(ln for ln in csv_text.splitlines() if ln.startswith("# closing_balance:"))
    value = line.split(":", 1)[1].strip()
    sign = -1 if value.startswith("-") else 1
    whole, _, cents = value.lstrip("-").partition(".")
    return sign * (int(whole) * 100 + int(cents.ljust(2, "0")[:2]))


def history_statements(seed: int = 7) -> list[tuple[str, str, str]]:
    """``(file name, member who uploads it, CSV text)`` oldest first. Deterministic for a seed; each statement
    balances, and each opens where the previous one for the account closed."""
    out: list[tuple[str, str, str]] = []
    opening = {RENTAL: 9000_00, SOMAL_CHECKING: 6000_00, SAYALI_CARD: -900_00}
    for ym in months():
        end = f"{ym}-{calendar.monthrange(int(ym[:4]), int(ym[5:]))[1]:02d}"
        for acct, label in ((RENTAL, "rental_checking"), (SOMAL_CHECKING, "somal_checking"), (SAYALI_CARD, "sayali_card")):
            rng = random.Random(f"{seed}-{acct}-{ym}")
            if acct == SAYALI_CARD:
                text = realistic_card_statement(rng.randint(1, 10**6), ym, acct, _money(opening[acct]))
            else:
                rows = (_rental_rows if acct == RENTAL else _somal_rows)(ym, rng)
                rows.sort(key=lambda r: r.posted_at)
                text = render_statement(acct, f"{ym}-01", end, _money(opening[acct]), rows)
            opening[acct] = _closing(text)  # chain even through a month the demo already owns, so the account stays continuous
            if (acct, ym) not in SKIPPED:
                out.append((f"{HISTORY_PREFIX}{ym.replace('-', '_')}_{label}.csv", OWNER[acct], text))
    return out


def statement_source(parsed: ParsedStatement, source_name: str, document_id: str) -> SourceDocument:
    """The statement as retrievable text: the header lines, then one ``date,description,amount`` row each."""
    h = parsed.header
    fmt = lambda minor: f"{'-' if minor < 0 else ''}{abs(minor) // 100}.{abs(minor) % 100:02d}"  # noqa: E731
    lines = [f"# account_id: {h.account_id}", f"# period: {h.period_start.isoformat()} to {h.period_end.isoformat()}",
             f"# opening_balance: {fmt(h.opening_balance_minor)}", f"# closing_balance: {fmt(h.closing_balance_minor)}",
             "date,description,amount"]
    lines += [f"{r.posted_at.isoformat()},{r.description},{fmt(r.amount_minor)}" for r in parsed.rows]
    return SourceDocument(
        doc_id=f"stmt_{document_id}", household_id=h.household_id, collection="financial_history", account_id=h.account_id,
        source_name=source_name, doc_type="bank_statement", authority_tier=2, publisher="Statement",
        text="\n".join(lines), statement_period=h.period_start.strftime("%Y-%m"))


# Fictional quarterly returns on the brokerage account, one per quarter from Q4 2024.
_QUARTER_RETURN = [-0.04, -0.02, 0.01, 0.06, 0.05, 0.035, -0.01, 0.03]
_QUARTER_LABEL = [("Q4 2024", "2024-12"), ("Q1 2025", "2025-03"), ("Q2 2025", "2025-06"), ("Q3 2025", "2025-09"),
                  ("Q4 2025", "2025-12"), ("Q1 2026", "2026-03"), ("Q2 2026", "2026-06"), ("Q3 2026", "2026-09")]


def brokerage_summaries() -> list[SourceDocument]:
    """Eight fictional quarterly summaries of a brokerage account funded from Somal's checking. Contributions match the
    ``ACME BROKERAGE TRANSFER`` rows in the ledger; the portfolio value follows an invented return path."""
    ms = months()
    value, total, out = 0.0, 0, []
    for q, ((label, period), ret) in enumerate(zip(_QUARTER_LABEL, _QUARTER_RETURN, strict=True)):
        contrib = sum(contributed_minor(m) for m in ms[3 * q:3 * q + 3])
        total += contrib
        value = (value + contrib / 100) * (1 + ret)
        gain = value - total / 100
        out.append(SourceDocument(
            doc_id=f"brokerage_{period}", household_id=HOUSEHOLD, collection="financial_history", account_id=SOMAL_CHECKING,
            source_name=f"ACME Brokerage summary {label} (SYNTHETIC)", doc_type="brokerage_summary", authority_tier=2,
            publisher="ACME Brokerage (SYNTHETIC)", statement_period=period,
            text=f"# Brokerage summary {label}\nFictional ACME Brokerage account funded from acct_somal_checking. "
                 f"Contributions this quarter: {contrib / 100:,.2f} dollars. Total contributed to date: {total / 100:,.2f} dollars. "
                 f"Portfolio value at quarter end: {value:,.2f} dollars. Gain or loss to date: {gain:+,.2f} dollars. "
                 f"Individual positions are listed in the monthly brokerage statements."))
    return out
