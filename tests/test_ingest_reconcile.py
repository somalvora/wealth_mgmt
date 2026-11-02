import pytest

from household_agent.categorize import Categorizer
from household_agent.ingest import StatementParseError, parse_statement_csv
from household_agent.models import ReconciliationStatus
from household_agent.reconcile import reconcile
from household_agent.synthetic import RULES, Row, demo_statements, render_statement


def _recon(text, historical=frozenset()):
    parsed = parse_statement_csv(text.encode(), "test.csv")
    cats = Categorizer(RULES)
    return parsed, reconcile(parsed, [cats.categorize(r.description).mapped for r in parsed.rows], set(historical))


def _stmt(rows, **kw):
    return render_statement("acct_alex_checking", "2026-10-01", "2026-10-31", "1000.00", rows, **kw)


# --- parser ---------------------------------------------------------------

def test_parser_reads_header_and_rows_with_line_provenance():
    parsed = parse_statement_csv(demo_statements()["2026_10_rental_checking.csv"].encode(), "x.csv")
    assert parsed.header.account_id == "acct_rental_checking"
    assert parsed.header.opening_balance_minor == 1_724_000
    assert len(parsed.rows) == 4
    tax = parsed.rows[-1]
    assert (tax.amount_minor, tax.line_number) == (-320_000, 15)
    assert parsed.source_hash.startswith("sha256:")


def test_parser_rejects_missing_header_field():
    text = _stmt([Row("2026-10-02", "SAFEWAY", "-10.00")]).replace("# total_debits: 10.00\n", "")
    with pytest.raises(StatementParseError, match="total_debits"):
        parse_statement_csv(text.encode(), "bad.csv")


def test_parser_rejects_unknown_format():
    text = _stmt([]).replace("hh_statement_csv_v1", "bank_xyz_v9")
    with pytest.raises(StatementParseError, match="unsupported format"):
        parse_statement_csv(text.encode(), "bad.csv")


def test_bad_row_is_kept_as_error_and_blocks_reconciliation():
    text = _stmt([Row("2026-10-02", "SAFEWAY", "-10.00")]) + "2026-10-03,SAFEWAY,12.345\n"
    parsed, result = _recon(text)
    assert [e.line_number for e in parsed.row_errors] == [13]
    assert result.status is ReconciliationStatus.UNRECONCILED
    assert any(i.code == "row_parse_error" for i in result.issues)


# --- reconciliation -----------------------------------------------------------

def test_clean_statement_reconciles_with_full_confidence():
    _, result = _recon(_stmt([Row("2026-10-02", "SAFEWAY #1", "-10.00"), Row("2026-10-03", "PAYROLL", "50.00")]))
    assert result.status is ReconciliationStatus.RECONCILED
    assert result.document_confidence == 1.0 and not result.unverified


def test_balance_mismatch_is_material():
    _, result = _recon(_stmt([Row("2026-10-02", "SAFEWAY", "-10.00")], closing_override="1000.00"))
    assert result.status is ReconciliationStatus.UNRECONCILED
    issue = next(i for i in result.issues if i.code == "balance_mismatch")
    assert issue.amount_minor == -1000


def test_row_outside_period_is_material():
    _, result = _recon(_stmt([Row("2026-11-02", "SAFEWAY", "-10.00")]))
    assert result.status is ReconciliationStatus.UNRECONCILED
    assert any(i.code == "outside_period" for i in result.issues)


def test_identical_rows_make_statement_qualified():
    rows = [Row("2026-10-07", "SAFEWAY #0457", "-96.10")] * 2
    _, result = _recon(_stmt(rows))
    assert result.status is ReconciliationStatus.QUALIFIED
    dup = next(i for i in result.issues if i.code == "possible_duplicate")
    assert dup.amount_minor == -9610


def test_reversal_pair_is_informational_only():
    _, result = _recon(_stmt([Row("2026-10-07", "SAFEWAY", "-20.00"), Row("2026-10-08", "SAFEWAY REFUND", "20.00")]))
    assert result.status is ReconciliationStatus.RECONCILED
    assert any(i.code == "reversal_pair" for i in result.issues)


def test_rows_already_ingested_are_detected_by_hash():
    text = _stmt([Row("2026-10-02", "SAFEWAY", "-10.00")])
    _, first = _recon(text)
    _, second = _recon(text, historical=set(first.row_hashes))
    assert second.status is ReconciliationStatus.UNRECONCILED
    assert any(i.code == "duplicate_of_ingested" for i in second.issues)


def test_unmapped_merchants_lower_confidence_to_unverified():
    rows = [Row("2026-10-02", "MYSTERY VENDOR A", "-10.00"), Row("2026-10-03", "MYSTERY VENDOR B", "-11.00")]
    _, result = _recon(_stmt(rows))
    assert result.status is ReconciliationStatus.RECONCILED
    assert result.document_confidence == 0.80 and result.unverified
