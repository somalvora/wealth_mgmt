import sqlite3
from datetime import date
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from household_agent.authz import AuthorizationError
from household_agent.calc_runner import CalculationRunner, IneligibleInputError
from household_agent.ledger import DuplicateDocumentError
from household_agent.models import ReconciliationStatus
from household_agent.pipeline import ingest_statement
from household_agent.synthetic import SOMAL, DEMO_AS_OF, HOUSEHOLD, SAYALI, demo_statements

CORR = "corr_test"
OCTS = (date(2025, 10, 1), date(2026, 10, 31))


def _ingest(store, categorizer, name, actor=SOMAL):
    return ingest_statement(store, categorizer, actor, demo_statements()[name].encode(), name,
                            purpose="statement_ingestion", correlation_id=CORR, at=DEMO_AS_OF)


@pytest.fixture
def loaded(store, categorizer):
    for name in demo_statements():
        _ingest(store, categorizer, name, SAYALI if "sayali" in name else SOMAL)
    return store


def test_non_owner_upload_is_denied_and_audited(store, categorizer):
    with pytest.raises(AuthorizationError):
        _ingest(store, categorizer, "2026_10_sayali_card.csv", actor=SOMAL)
    denied = [e for e in store.audit.events(CORR) if e["decision"] == "deny"]
    assert denied and denied[0]["action"] == "upload"


def test_same_file_cannot_be_ingested_twice(store, categorizer):
    _ingest(store, categorizer, "2026_10_somal_checking.csv")
    with pytest.raises(DuplicateDocumentError):
        _ingest(store, categorizer, "2026_10_somal_checking.csv")


def test_query_returns_only_reconciled_by_default(loaded):
    card = loaded.query_transactions(SAYALI, "budgeting", CORR, ["acct_sayali_card"], *OCTS, at=DEMO_AS_OF)
    assert card == []  # qualified statement is provisional
    card = loaded.query_transactions(SAYALI, "budgeting", CORR, ["acct_sayali_card"], *OCTS,
                                     include_provisional=True, at=DEMO_AS_OF)
    assert len(card) == 5 and all(t.reconciliation_status is ReconciliationStatus.QUALIFIED for t in card)


def test_unreconciled_rows_are_never_returned(loaded):
    sept = loaded.query_transactions(SOMAL, "budgeting", CORR, ["acct_somal_checking"],
                                     date(2026, 9, 1), date(2026, 9, 30), include_provisional=True, at=DEMO_AS_OF)
    assert sept == []


def test_cross_member_read_is_denied(loaded):
    with pytest.raises(AuthorizationError):
        loaded.query_transactions(SAYALI, "budgeting", CORR, ["acct_somal_checking"], *OCTS, at=DEMO_AS_OF)


def test_raw_ledger_rows_are_append_only(loaded):
    with pytest.raises(sqlite3.IntegrityError):
        loaded.conn.execute("UPDATE transactions SET amount_minor = 0")
    with pytest.raises(sqlite3.IntegrityError):
        loaded.conn.execute("DELETE FROM raw_rows")


def test_correction_adds_a_version(loaded):
    txn = loaded.query_transactions(SOMAL, "budgeting", CORR, ["acct_rental_checking"], *OCTS, at=DEMO_AS_OF)[0]
    fixed = loaded.correct_transaction(SOMAL, "categorization", CORR, txn.transaction_id, "user confirmed",
                                       category="rental_income_verified", at=DEMO_AS_OF)
    assert fixed.version == 2 and fixed.category_confidence == 1.0
    again = loaded.query_transactions(SOMAL, "budgeting", CORR, ["acct_rental_checking"], *OCTS, at=DEMO_AS_OF)
    assert next(t for t in again if t.transaction_id == txn.transaction_id).category == "rental_income_verified"
    versions = loaded.conn.execute("SELECT COUNT(*) FROM transactions WHERE transaction_id = ?", (txn.transaction_id,)).fetchone()
    assert versions[0] == 2


def test_property_tax_variance_end_to_end(loaded):
    txns = loaded.query_transactions(SOMAL, "historical_comparison", CORR, ["acct_rental_checking"], *OCTS,
                                     merchant_normalized="San Mateo County Tax Collector", at=DEMO_AS_OF)
    runner = CalculationRunner(loaded)
    run = runner.run("variance_between_transactions", txns, actor_id="somal", household_id=HOUSEHOLD, correlation_id=CORR)
    assert run.output["dollar_variance_minor"] == 45_000
    assert Decimal(run.output["pct_variance"]) == Decimal("16.36")
    assert not run.provisional

    rerun = runner.run("variance_between_transactions", txns, actor_id="somal", household_id=HOUSEHOLD, correlation_id=CORR)
    assert (rerun.input_hash, rerun.output_hash) == (run.input_hash, run.output_hash)
    assert rerun.run_id != run.run_id
    stored = loaded.conn.execute("SELECT COUNT(*) FROM calculation_runs").fetchone()[0]
    assert stored == 2
    assert loaded.audit.verify() > 0


def test_calculation_rejects_provisional_unless_allowed(loaded):
    card = loaded.query_transactions(SAYALI, "budgeting", CORR, ["acct_sayali_card"], *OCTS,
                                     include_provisional=True, at=DEMO_AS_OF)
    runner = CalculationRunner(loaded)
    params = {"start": "2026-10-01", "end": "2026-10-31"}
    with pytest.raises(IneligibleInputError):
        runner.run("cash_flow", card, actor_id="sayali", household_id=HOUSEHOLD, correlation_id=CORR, params=params)
    run = runner.run("cash_flow", card, actor_id="sayali", household_id=HOUSEHOLD, correlation_id=CORR,
                     params=params, allow_provisional=True)
    assert run.provisional and run.output["net_minor"] == 820_00 - 1299 - 9610 * 2 - 4299


def _card(loaded):
    return loaded.query_transactions(SAYALI, "budgeting", CORR, ["acct_sayali_card"], *OCTS,
                                     include_provisional=True, at=DEMO_AS_OF)


def _flow(loaded, card, **extra):
    return CalculationRunner(loaded).run(
        "cash_flow", card, actor_id="sayali", household_id=HOUSEHOLD, correlation_id=CORR, allow_provisional=True,
        params={"start": "2026-10-01", "end": "2026-10-31", **extra}).output


def test_cash_flow_can_leave_out_transfers_such_as_a_card_payment(loaded):
    card = _card(loaded)
    assert [t.category for t in card if t.amount_minor > 0] == ["transfer"]
    everything = _flow(loaded, card)
    spending_only = _flow(loaded, card, exclude_categories=["transfer"])
    assert everything["inflow_minor"] == 820_00  # the payment looks like money in
    assert spending_only["inflow_minor"] == 0 and spending_only["outflow_minor"] == everything["outflow_minor"]
    assert spending_only["net_minor"] == -everything["outflow_minor"]


@pytest.mark.parametrize("bad", ["transfer", {"transfer": 1}, 5])
def test_exclude_categories_must_be_a_list(loaded, bad):
    with pytest.raises(ValueError, match="list of category names"):
        _flow(loaded, _card(loaded), exclude_categories=bad)


@given(skip=st.lists(st.sampled_from(["transfer", "groceries", "entertainment", "nothing_like_this"]), max_size=4))
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_excluding_categories_never_adds_money(skip, loaded):
    card = _card(loaded)
    base, less = _flow(loaded, card), _flow(loaded, card, exclude_categories=skip)
    assert less["inflow_minor"] <= base["inflow_minor"] and less["outflow_minor"] <= base["outflow_minor"]
    assert less["net_minor"] == less["inflow_minor"] - less["outflow_minor"]


def test_unknown_calculation_is_rejected(loaded):
    with pytest.raises(KeyError):
        CalculationRunner(loaded).run("exec_python", [], actor_id="somal", household_id=HOUSEHOLD, correlation_id=CORR)
