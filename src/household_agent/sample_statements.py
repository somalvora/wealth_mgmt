"""Generated PDF statements for the web demo (``hh_statement_pdf_v1`` text layout).

Every file is synthetic. Three are good statements, and two show the guardrails:

- ``2026_10_rental_checking.pdf``  Alex's joint rental account for October 2026 (reconciles);
- ``2026_10_alex_checking.pdf``    Alex's private checking (reconciles);
- ``2026_10_jordan_card.pdf``      Jordan's private card: Alex uploading it is refused;
- ``2026_09_alex_checking_tampered.pdf``  the stated closing balance was edited, so it does not reconcile;
- ``scanned_statement.pdf``        a page with no text layer (what a scan looks like to the parser): rejected.

Output is deterministic (reportlab ``invariant``), so the same bytes (and source hash) come out every time.
"""

import io
from dataclasses import dataclass

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from household_agent.synthetic import HOUSEHOLD

Row = tuple[str, str, int]  # (date, description, signed amount in cents)


def _fmt(minor: int) -> str:
    return f"{'-' if minor < 0 else ''}{abs(minor) // 100:,}.{abs(minor) % 100:02d}"


def statement_lines(account_id: str, start: str, end: str, opening: int, rows: list[Row], *,
                    stated_closing: int | None = None) -> list[str]:
    amounts = [a for _, _, a in rows]
    closing = opening + sum(amounts) if stated_closing is None else stated_closing
    lines = [
        f"Household ID: {HOUSEHOLD}", f"Account ID: {account_id}", "Currency: USD",
        f"Statement Period: {start} to {end}", f"Opening Balance: {_fmt(opening)}",
        f"Closing Balance: {_fmt(closing)}", f"Total Credits: {_fmt(sum(a for a in amounts if a > 0))}",
        f"Total Debits: {_fmt(-sum(a for a in amounts if a < 0))}", "Date Description Amount Balance",
    ]
    running = opening
    for posted, description, amount in rows:
        running += amount
        lines.append(f"{posted} {description} {_fmt(amount)} {_fmt(running)}")
    return lines


def _pdf(lines: list[str], title: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter, invariant=1)
    c.setTitle(title)
    y = 750
    for line in lines:
        c.drawString(40, y, line)
        y -= 16
    c.save()
    return buf.getvalue()


def _image_only_pdf() -> bytes:
    """A page with ruled boxes and no text: pypdf finds no text layer, as with a scanned statement."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter, invariant=1)
    for i in range(14):
        c.rect(40, 700 - 40 * i, 520, 28)
    c.save()
    return buf.getvalue()


@dataclass(frozen=True)
class Sample:
    name: str
    member: str  # who uploads it in the demo
    description: str
    expect: str  # what the system should do with it
    content: bytes


def samples() -> dict[str, Sample]:
    rental = statement_lines("acct_rental_checking", "2026-10-01", "2026-10-31", 1724000, [
        ("2026-10-01", "RENT DEPOSIT UNIT B", 310000),
        ("2026-10-05", "MAPLE ST HOA DUES", -45000),
        ("2026-10-09", "CITY WATER UTILITY", -11840),
        ("2026-10-18", "SAN MATEO TAX COLLECTOR - SECURED PROPERTY TAX - UNIT B", -320000),
    ])
    alex = statement_lines("acct_alex_checking", "2026-10-01", "2026-10-31", 920000, [
        ("2026-10-01", "SUMMIT MORTGAGE PMT", -345000),
        ("2026-10-03", "PAYROLL ACME CORP", 410000),
        ("2026-10-11", "PG&E ENERGY BILL", -21237),
        ("2026-10-14", "SAFEWAY #1182", -18652),
        ("2026-10-17", "PAYROLL ACME CORP", 410000),
        ("2026-10-28", "SAFEWAY #1182", -14308),
    ])
    jordan = statement_lines("acct_jordan_card", "2026-10-01", "2026-10-31", -82000, [
        ("2026-10-02", "STREAMCO MONTHLY", -1299),
        ("2026-10-07", "SAFEWAY #0457", -9610),
        ("2026-10-15", "CARD PAYMENT THANK YOU", 82000),
        ("2026-10-21", "NOVAPIXEL DIGITAL", -4299),
    ])
    tampered = statement_lines("acct_alex_checking", "2026-09-01", "2026-09-30", 800000, [
        ("2026-09-03", "PAYROLL ACME CORP", 410000),
        ("2026-09-10", "SUMMIT MORTGAGE PMT", -345000),
    ], stated_closing=999999)  # the rows end at 8,650.00; someone edited the stated closing balance
    out = [
        Sample("2026_10_rental_checking.pdf", "alex", "October 2026 joint rental account (includes the $3,200 county tax payment)",
               "reconciled", _pdf(rental, "rental checking Oct 2026")),
        Sample("2026_10_alex_checking.pdf", "alex", "Alex's private checking, October 2026", "reconciled",
               _pdf(alex, "alex checking Oct 2026")),
        Sample("2026_10_jordan_card.pdf", "alex", "Jordan's private card: upload it as Alex",
               "refused: Alex may not write to Jordan's private account", _pdf(jordan, "jordan card Oct 2026")),
        Sample("2026_09_alex_checking_tampered.pdf", "alex", "September statement with an edited closing balance",
               "stored as unreconciled and blocked from every calculation", _pdf(tampered, "alex checking Sep 2026")),
        Sample("scanned_statement.pdf", "alex", "A scanned page: no text layer",
               "rejected: OCR is not built", _image_only_pdf()),
    ]
    return {s.name: s for s in out}
