import sqlite3
from datetime import date
from decimal import Decimal

import pytest

from household_agent.authz import AuthorizationError
from household_agent.calc_runner import CalculationRunner, IneligibleInputError
from household_agent.ledger import DuplicateDocumentError
from household_agent.models import ReconciliationStatus
from household_agent.pipeline import ingest_statement
from household_agent.synthetic import ALEX, DEMO_AS_OF, HOUSEHOLD, JORDAN, demo_statements

CORR = "corr_test"
OCTS = (date(2025, 10, 1), date(2026, 10, 31))


def _ingest(store, categorizer, name, actor=ALEX):
    return ingest_statement(store, categorizer, actor, demo_statements()[name].encode(), name,
                            purpose="statement_ingestion", correlation_id=CORR, at=DEMO_AS_OF)


@pytest.fixture
def loaded(store, categorizer):
    for name in demo_statements():
        _ingest(store, categorizer, name, JORDAN if "jordan" in name else ALEX)
    return store


def test_non_owner_upload_is_denied_and_audited(store, categorizer):
    with pytest.raises(AuthorizationError):
        _ingest(store, categorizer, "2026_10_jordan_card.csv", actor=ALEX)
    denied = [e for e in store.audit.events(CORR) if e["decision"] == "deny"]
    assert denied and denied[0]["action"] == "upload"


def test_same_file_cannot_be_ingested_twice(store, categorizer):
    _ingest(store, categorizer, "2026_10_alex_checking.csv")
    with pytest.raises(DuplicateDocumentError):
        _ingest(store, categorizer, "2026_10_alex_checking.csv")


def test_query_returns_only_reconciled_by_default(loaded):
    card = loaded.query_transactions(JORDAN, "budgeting", CORR, ["acct_jordan_card"], *OCTS, at=DEMO_AS_OF)
    assert card == []  # qualified statement is provisional
    card = loaded.query_transactions(JORDAN, "budgeting", CORR, ["acct_jordan_card"], *OCTS,
                                     include_provisional=True, at=DEMO_AS_OF)
    assert len(card) == 5 and all(t.reconciliation_status is ReconciliationStatus.QUALIFIED for t in card)


def test_unreconciled_rows_are_never_returned(loaded):
    sept = loaded.query_transactions(ALEX, "budgeting", CORR, ["acct_alex_checking"],
                                     date(2026, 9, 1), date(2026, 9, 30), include_provisional=True, at=DEMO_AS_OF)
    assert sept == []


def test_cross_member_read_is_denied(loaded):
    with pytest.raises(AuthorizationError):
        loaded.query_transactions(JORDAN, "budgeting", CORR, ["acct_alex_checking"], *OCTS, at=DEMO_AS_OF)


def test_raw_ledger_rows_are_append_only(loaded):
    with pytest.raises(sqlite3.IntegrityError):
        loaded.conn.execute("UPDATE transactions SET amount_minor = 0")
    with pytest.raises(sqlite3.IntegrityError):
        loaded.conn.execute("DELETE FROM raw_rows")


def test_correction_adds_a_version(loaded):
    txn = loaded.query_transactions(ALEX, "budgeting", CORR, ["acct_rental_checking"], *OCTS, at=DEMO_AS_OF)[0]
    fixed = loaded.correct_transaction(ALEX, "categorization", CORR, txn.transaction_id, "user confirmed",
                                       category="rental_income_verified", at=DEMO_AS_OF)
    assert fixed.version == 2 and fixed.category_confidence == 1.0
    again = loaded.query_transactions(ALEX, "budgeting", CORR, ["acct_rental_checking"], *OCTS, at=DEMO_AS_OF)
    assert next(t for t in again if t.transaction_id == txn.transaction_id).category == "rental_income_verified"
    versions = loaded.conn.execute("SELECT COUNT(*) FROM transactions WHERE transaction_id = ?", (txn.transaction_id,)).fetchone()
    assert versions[0] == 2


def test_property_tax_variance_end_to_end(loaded):
    txns = loaded.query_transactions(ALEX, "historical_comparison", CORR, ["acct_rental_checking"], *OCTS,
                                     merchant_normalized="San Mateo County Tax Collector", at=DEMO_AS_OF)
    runner = CalculationRunner(loaded)
    run = runner.run("variance_between_transactions", txns, actor_id="alex", household_id=HOUSEHOLD, correlation_id=CORR)
    assert run.output["dollar_variance_minor"] == 45_000
    assert Decimal(run.output["pct_variance"]) == Decimal("16.36")
    assert not run.provisional

    rerun = runner.run("variance_between_transactions", txns, actor_id="alex", household_id=HOUSEHOLD, correlation_id=CORR)
    assert (rerun.input_hash, rerun.output_hash) == (run.input_hash, run.output_hash)
    assert rerun.run_id != run.run_id
    stored = loaded.conn.execute("SELECT COUNT(*) FROM calculation_runs").fetchone()[0]
    assert stored == 2
    assert loaded.audit.verify() > 0


def test_calculation_rejects_provisional_unless_allowed(loaded):
    card = loaded.query_transactions(JORDAN, "budgeting", CORR, ["acct_jordan_card"], *OCTS,
                                     include_provisional=True, at=DEMO_AS_OF)
    runner = CalculationRunner(loaded)
    params = {"start": "2026-10-01", "end": "2026-10-31"}
    with pytest.raises(IneligibleInputError):
        runner.run("cash_flow", card, actor_id="jordan", household_id=HOUSEHOLD, correlation_id=CORR, params=params)
    run = runner.run("cash_flow", card, actor_id="jordan", household_id=HOUSEHOLD, correlation_id=CORR,
                     params=params, allow_provisional=True)
    assert run.provisional and run.output["net_minor"] == 820_00 - 1299 - 9610 * 2 - 4299


def test_unknown_calculation_is_rejected(loaded):
    with pytest.raises(KeyError):
        CalculationRunner(loaded).run("exec_python", [], actor_id="alex", household_id=HOUSEHOLD, correlation_id=CORR)
