"""Parser for the single supported synthetic statement format, ``hh_statement_csv_v1``.

Example::

    # format: hh_statement_csv_v1
    # household_id: hh_demo
    # account_id: acct_rental_checking
    # currency: USD
    # period_start: 2026-10-01
    # period_end: 2026-10-31
    # opening_balance: 12500.00
    # closing_balance: 12000.00
    # total_credits: 3100.00
    # total_debits: 3600.00
    date,description,amount
    2026-10-01,RENT DEPOSIT UNIT B,3100.00

A malformed header makes the whole file unusable (``StatementParseError``). A malformed
row is kept as a ``RowError`` so reconciliation can mark the statement unreconciled with
the exact line numbers instead of silently dropping data.
"""

import csv
import hashlib
from dataclasses import dataclass
from datetime import date

from household_finance import MoneyError, to_minor

FORMAT_ID = "hh_statement_csv_v1"
PARSER_VERSION = "hh_csv_parser_1.0"
COLUMNS = ["date", "description", "amount"]
REQUIRED_HEADER = (
    "format", "household_id", "account_id", "currency", "period_start", "period_end",
    "opening_balance", "closing_balance", "total_credits", "total_debits",
)


class StatementParseError(ValueError):
    pass


@dataclass(frozen=True)
class StatementHeader:
    household_id: str
    account_id: str
    currency: str
    period_start: date
    period_end: date
    opening_balance_minor: int
    closing_balance_minor: int
    total_credits_minor: int
    total_debits_minor: int  # positive magnitude


@dataclass(frozen=True)
class ParsedRow:
    line_number: int
    raw_text: str
    posted_at: date
    description: str
    amount_minor: int


@dataclass(frozen=True)
class RowError:
    line_number: int
    raw_text: str
    error: str


@dataclass(frozen=True)
class ParsedStatement:
    source_name: str
    source_hash: str
    parser_version: str
    header: StatementHeader
    rows: tuple[ParsedRow, ...]
    row_errors: tuple[RowError, ...]


def parse_statement_csv(content: bytes, source_name: str) -> ParsedStatement:
    source_hash = "sha256:" + hashlib.sha256(content).hexdigest()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StatementParseError(f"{source_name}: not UTF-8") from exc

    meta: dict[str, str] = {}
    lines = text.splitlines()
    idx = 0
    while idx < len(lines) and lines[idx].startswith("#"):
        key, sep, value = lines[idx][1:].partition(":")
        if sep:
            meta[key.strip()] = value.strip()
        idx += 1

    missing = [k for k in REQUIRED_HEADER if not meta.get(k)]
    if missing:
        raise StatementParseError(f"{source_name}: missing header fields {missing}")
    if meta["format"] != FORMAT_ID:
        raise StatementParseError(f"{source_name}: unsupported format {meta['format']!r}")
    if idx >= len(lines) or [c.strip() for c in lines[idx].split(",")] != COLUMNS:
        raise StatementParseError(f"{source_name}: expected column header {','.join(COLUMNS)}")

    try:
        header = StatementHeader(
            household_id=meta["household_id"],
            account_id=meta["account_id"],
            currency=meta["currency"].upper(),
            period_start=date.fromisoformat(meta["period_start"]),
            period_end=date.fromisoformat(meta["period_end"]),
            opening_balance_minor=to_minor(meta["opening_balance"]),
            closing_balance_minor=to_minor(meta["closing_balance"]),
            total_credits_minor=to_minor(meta["total_credits"]),
            total_debits_minor=abs(to_minor(meta["total_debits"])),
        )
    except (ValueError, MoneyError) as exc:
        raise StatementParseError(f"{source_name}: invalid header value ({exc})") from exc

    rows: list[ParsedRow] = []
    errors: list[RowError] = []
    for offset, line in enumerate(lines[idx + 1:], start=idx + 2):
        if not line.strip():
            continue
        fields = next(csv.reader([line]))
        if len(fields) != len(COLUMNS):
            errors.append(RowError(offset, line, f"expected {len(COLUMNS)} columns, got {len(fields)}"))
            continue
        raw_date, description, raw_amount = (f.strip() for f in fields)
        try:
            posted_at = date.fromisoformat(raw_date)
            amount = to_minor(raw_amount)
        except (ValueError, MoneyError) as exc:
            errors.append(RowError(offset, line, str(exc)))
            continue
        if not description:
            errors.append(RowError(offset, line, "empty description"))
            continue
        rows.append(ParsedRow(offset, line, posted_at, description, amount))

    return ParsedStatement(source_name, source_hash, PARSER_VERSION, header, tuple(rows), tuple(errors))
