from household_agent.ingest.csv_statement import (
    FORMAT_ID,
    PARSER_VERSION,
    ParsedRow,
    ParsedStatement,
    RowError,
    StatementHeader,
    StatementParseError,
    parse_statement_csv,
)

from household_agent.ingest.pdf_statement import is_pdf, parse_statement_pdf


def parse_statement(content: bytes, source_name: str, *, allow_pdf: bool = True) -> ParsedStatement:
    """Choose the parser from the bytes (PDF magic ``%PDF-``), not the file name."""
    if is_pdf(content):
        if not allow_pdf:
            raise StatementParseError(f"{source_name}: PDF is not accepted on this path (CSV text only)")
        return parse_statement_pdf(content, source_name)
    return parse_statement_csv(content, source_name)


__all__ = [
    "is_pdf",
    "parse_statement",
    "parse_statement_pdf",
    "FORMAT_ID",
    "PARSER_VERSION",
    "ParsedRow",
    "ParsedStatement",
    "RowError",
    "StatementHeader",
    "StatementParseError",
    "parse_statement_csv",
]
