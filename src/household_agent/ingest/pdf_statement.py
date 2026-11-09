"""Parser for text-based PDF statements (``hh_statement_pdf_v1``), producing the same
``ParsedStatement`` as the CSV parser so reconciliation, categorization, dedup and audit are unchanged.

Expected text layer (one item per line, as most bank PDFs extract)::

    Household ID: hh_demo
    Account ID: acct_rental_checking
    Currency: USD
    Statement Period: 2026-10-01 to 2026-10-31
    Opening Balance: 12,500.00
    Closing Balance: 12,000.00
    Total Credits: 3,100.00
    Total Debits: 3,600.00
    Date Description Amount            (optionally ``... Amount Balance``)
    2026-10-01 RENT DEPOSIT UNIT B 3,100.00

Limits and safety:
- Only an extractable text layer is read. A PDF with none (a scan) is rejected: OCR is not built.
- Encrypted, oversize (bytes or pages) and corrupt files are rejected before any text is used.
- The text is untrusted data. It is only matched against fixed patterns; nothing in it is executed
  or interpreted as an instruction, and pypdf does not run embedded JavaScript or launch actions.
- A date-led line that does not match the row pattern, or a running balance that disagrees with the
  amounts, becomes a ``RowError`` (so the statement is unreconciled) rather than being dropped.
- Amounts go through ``to_minor`` (integer cents, never floats). Extraction is deterministic.
"""

import hashlib
import io
import re
from datetime import date

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from household_finance import MoneyError, to_minor

from household_agent.ingest.csv_statement import (
    ParsedRow,
    ParsedStatement,
    RowError,
    StatementHeader,
    StatementParseError,
)

FORMAT_ID = "hh_statement_pdf_v1"
PARSER_VERSION = "hh_pdf_parser_1.0"
PDF_MAGIC = b"%PDF-"
MAX_PDF_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 20

_AMT = r"-?\$?-?[\d,]+\.\d{2}"
_ROW_2 = re.compile(rf"^(\d{{4}}-\d{{2}}-\d{{2}})\s+(.+?)\s+({_AMT})$")
_ROW_3 = re.compile(rf"^(\d{{4}}-\d{{2}}-\d{{2}})\s+(.+?)\s+({_AMT})\s+({_AMT})$")
_DATE_LED = re.compile(r"^\d{4}-\d{2}-\d{2}\b")
_TABLE_HEADER = re.compile(r"^date\s+description\s+amount(\s+balance)?$", re.IGNORECASE)
_PAGE_FOOTER = re.compile(r"^page \d+ of \d+$", re.IGNORECASE)
_HEADER_KEYS = {
    "household id": "household_id", "account id": "account_id", "currency": "currency",
    "statement period": "period", "opening balance": "opening_balance",
    "closing balance": "closing_balance", "total credits": "total_credits", "total debits": "total_debits",
}


def is_pdf(content: bytes) -> bool:
    return content.startswith(PDF_MAGIC)


def _extract_lines(content: bytes, source_name: str) -> list[str]:
    if len(content) > MAX_PDF_BYTES:
        raise StatementParseError(f"{source_name}: PDF larger than {MAX_PDF_BYTES} bytes")
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise StatementParseError(f"{source_name}: encrypted PDF is not supported")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise StatementParseError(f"{source_name}: more than {MAX_PDF_PAGES} pages")
        texts = [page.extract_text() or "" for page in reader.pages]
    except StatementParseError:
        raise
    except (PyPdfError, ValueError, KeyError, TypeError, IndexError, OSError, RecursionError) as exc:
        raise StatementParseError(f"{source_name}: unreadable or corrupt PDF ({type(exc).__name__})") from exc
    lines = [" ".join(ln.split()) for ln in "\n".join(texts).splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise StatementParseError(f"{source_name}: PDF has no extractable text layer; scanned statements "
                                  "need OCR, which is not built (unreconciled)")
    return lines


def parse_statement_pdf(content: bytes, source_name: str) -> ParsedStatement:
    source_hash = "sha256:" + hashlib.sha256(content).hexdigest()
    lines = _extract_lines(content, source_name)

    meta: dict[str, str] = {}
    table_at = None
    for i, line in enumerate(lines):
        if _TABLE_HEADER.match(line):
            table_at = i
            break
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() in _HEADER_KEYS:
            meta.setdefault(_HEADER_KEYS[key.strip().lower()], value.strip())
    if table_at is None:
        raise StatementParseError(f"{source_name}: no 'Date Description Amount' table header found")
    has_balance = lines[table_at].lower().endswith("balance")
    missing = [k for k in ("household_id", "account_id", "currency", "period", "opening_balance",
                           "closing_balance", "total_credits", "total_debits") if not meta.get(k)]
    if missing:
        raise StatementParseError(f"{source_name}: missing header fields {missing}")

    try:
        start_raw, sep, end_raw = meta["period"].partition(" to ")
        if not sep:
            raise ValueError("period must read 'YYYY-MM-DD to YYYY-MM-DD'")
        header = StatementHeader(
            household_id=meta["household_id"], account_id=meta["account_id"],
            currency=meta["currency"].upper(),
            period_start=date.fromisoformat(start_raw.strip()), period_end=date.fromisoformat(end_raw.strip()),
            opening_balance_minor=to_minor(meta["opening_balance"]),
            closing_balance_minor=to_minor(meta["closing_balance"]),
            total_credits_minor=to_minor(meta["total_credits"]),
            total_debits_minor=abs(to_minor(meta["total_debits"])),
        )
    except (ValueError, MoneyError) as exc:
        raise StatementParseError(f"{source_name}: invalid header value ({exc})") from exc

    rows: list[ParsedRow] = []
    errors: list[RowError] = []
    running = header.opening_balance_minor
    for n, line in enumerate(lines[table_at + 1:], start=table_at + 2):
        if _PAGE_FOOTER.match(line):
            continue
        m = (_ROW_3 if has_balance else _ROW_2).match(line)
        if not m:
            errors.append(RowError(n, line, "line does not match the date/description/amount layout"
                                   if _DATE_LED.match(line) else "unexpected text in the transaction table"))
            continue
        try:
            posted_at = date.fromisoformat(m.group(1))
            amount = to_minor(m.group(3))
            balance = to_minor(m.group(4)) if has_balance else None
        except (ValueError, MoneyError) as exc:
            errors.append(RowError(n, line, str(exc)))
            continue
        running += amount
        if balance is not None and balance != running:
            errors.append(RowError(n, line, f"running balance {balance} does not follow from the amounts"))
            running = balance
            continue
        rows.append(ParsedRow(n, line, posted_at, m.group(2).strip(), amount))

    return ParsedStatement(source_name, source_hash, PARSER_VERSION, header, tuple(rows), tuple(errors))
