from dataclasses import replace

import pytest

from household_agent.consent import ConsentStore
from household_agent.synthetic import ALEX, CASEY_CPA, DEMO_AS_OF, JORDAN, SAM, cpa_grant, demo_statements
from household_agent.tools import ToolSession

TAX = "San Mateo County Tax Collector"
RENTAL = "acct_rental_checking"


def _session(store, categorizer, actor=ALEX, purpose="historical_comparison", **kw):
    return ToolSession(store, categorizer, actor, purpose, correlation_id="t1", clock=lambda: DEMO_AS_OF, **kw)


def _tax_rows(s):
    return s.call("query_ledger", {"account_ids": [RENTAL], "start": "2025-10-01", "end": "2026-10-31", "merchant": TAX})


def test_query_then_calculate_property_tax_variance(seeded_store, categorizer):
    s = _session(seeded_store, categorizer, acting_agent="ledger_analyst")
    rows = _tax_rows(s)
    assert rows["ok"] and [t["amount_minor"] for t in rows["transactions"]] == [-275000, -320000]
    assert all(t["source_locator"].startswith("line=") for t in rows["transactions"])
    calc = s.call("run_calculation", {"fn": "variance_between_transactions",
                                      "transaction_ids": [t["transaction_id"] for t in rows["transactions"]]})
    assert calc["ok"] and calc["output"]["dollar_variance_minor"] == 45000
    assert calc["output"]["pct_variance"] == "16.36" and not calc["provisional"]
    assert calc["run_id"].startswith("calc_run_") and calc["input_hash"].startswith("sha256:")


def test_list_accounts_respects_ownership_and_consent(seeded_store, categorizer, conn):
    ids = lambda s: [a["account_id"] for a in s.call("list_accounts")["accounts"]]  # noqa: E731
    assert ids(_session(seeded_store, categorizer, ALEX)) == ["acct_alex_checking", RENTAL]
    assert ids(_session(seeded_store, categorizer, JORDAN)) == ["acct_jordan_card", RENTAL]
    assert ids(_session(seeded_store, categorizer, SAM)) == []
    cpa = _session(seeded_store, categorizer, CASEY_CPA, purpose="tax_preparation")
    assert ids(cpa) == []
    consents = ConsentStore(conn, seeded_store.authorizer, seeded_store.audit)
    expires = DEMO_AS_OF.replace(year=2027)
    consents.grant(ALEX, cpa_grant(expires), correlation_id="g", at=DEMO_AS_OF)
    consents.grant(JORDAN, replace(cpa_grant(expires), grant_id="g_j", grantor_id="jordan"), correlation_id="g", at=DEMO_AS_OF)
    assert ids(cpa) == [RENTAL]
    # Consent is purpose-scoped: the same CPA in a budgeting session sees nothing.
    assert ids(_session(seeded_store, categorizer, CASEY_CPA, purpose="budgeting")) == []


def test_cross_member_read_is_denied_and_audited(seeded_store, categorizer):
    s = _session(seeded_store, categorizer, JORDAN, purpose="budgeting")
    out = s.call("query_ledger", {"account_ids": ["acct_alex_checking"], "start": "2026-10-01", "end": "2026-10-31"})
    assert out == {"ok": False, "error": {"type": "authorization_denied", "message": out["error"]["message"]}}
    assert seeded_store.audit.events("t1")[-1]["decision"] == "deny"


def test_actor_cannot_be_passed_as_a_tool_argument(seeded_store, categorizer):
    s = _session(seeded_store, categorizer, JORDAN)
    out = s.call("query_ledger", {"actor": "alex", "account_ids": ["acct_alex_checking"],
                                  "start": "2026-10-01", "end": "2026-10-31"})
    assert out["error"]["type"] == "invalid_request" and "actor" in out["error"]["message"]


def test_calculation_only_on_transactions_this_session_queried(seeded_store, categorizer):
    alex = _session(seeded_store, categorizer)
    ids = [t["transaction_id"] for t in _tax_rows(alex)["transactions"]]
    fresh = _session(seeded_store, categorizer)  # did not query
    out = fresh.call("run_calculation", {"fn": "variance_between_transactions", "transaction_ids": ids})
    assert out["error"]["type"] == "ineligible_input"
    out = alex.call("run_calculation", {"fn": "variance_between_transactions", "transaction_ids": [ids[0], "txn_made_up"]})
    assert out["error"]["type"] == "ineligible_input" and "txn_made_up" in out["error"]["message"]


def test_provisional_rows_need_explicit_opt_in(seeded_store, categorizer):
    s = _session(seeded_store, categorizer, JORDAN, purpose="budgeting")
    q = {"account_ids": ["acct_jordan_card"], "start": "2026-10-01", "end": "2026-10-31"}
    assert s.call("query_ledger", q)["transactions"] == []  # card statement is only "qualified"
    rows = s.call("query_ledger", {**q, "include_provisional": True})["transactions"]
    assert len(rows) == 5 and {t["reconciliation_status"] for t in rows} == {"qualified"}
    args = {"fn": "cash_flow", "transaction_ids": [t["transaction_id"] for t in rows],
            "params": {"start": "2026-10-01", "end": "2026-10-31"}}
    assert s.call("run_calculation", args)["error"]["type"] == "ineligible_input"
    out = s.call("run_calculation", {**args, "allow_provisional": True})
    assert out["ok"] and out["provisional"]


def test_tool_budget_is_enforced(seeded_store, categorizer):
    s = _session(seeded_store, categorizer, max_calls=2)
    assert s.call("list_accounts")["ok"] and s.call("list_accounts")["ok"]
    assert s.call("list_accounts")["error"]["type"] == "budget_exhausted"
    assert s.calls == 2


@pytest.mark.parametrize(
    "name, args, kind",
    [
        ("delete_ledger", {}, "invalid_request"),
        ("query_ledger", {"account_ids": [RENTAL]}, "invalid_request"),
        ("query_ledger", {"account_ids": [], "start": "2026-10-01", "end": "2026-10-31"}, "invalid_request"),
        ("query_ledger", {"account_ids": [RENTAL], "start": "Oct 1", "end": "2026-10-31"}, "invalid_request"),
        ("run_calculation", {"fn": "eval", "transaction_ids": []}, "invalid_request"),
        ("query_ledger", {"account_ids": ["acct_nope"], "start": "2026-10-01", "end": "2026-10-31"}, "authorization_denied"),
    ],
)
def test_bad_requests_return_structured_errors(seeded_store, categorizer, name, args, kind):
    assert _session(seeded_store, categorizer).call(name, args)["error"]["type"] == kind


def test_ingest_tool_guards(store, categorizer):
    statements = demo_statements()
    jordan = _session(store, categorizer, JORDAN, purpose="statement_ingestion")
    out = jordan.call("ingest_statement", {"csv_text": statements["2026_10_alex_checking.csv"], "source_name": "a.csv"})
    assert out["error"]["type"] == "authorization_denied"
    ok = jordan.call("ingest_statement", {"csv_text": statements["2026_10_jordan_card.csv"], "source_name": "j.csv"})
    assert ok["ok"] and ok["status"] == "qualified" and ok["issues"][0]["code"] == "possible_duplicate"
    again = jordan.call("ingest_statement", {"csv_text": statements["2026_10_jordan_card.csv"], "source_name": "j.csv"})
    assert again["error"]["type"] == "duplicate_document"
    bad = jordan.call("ingest_statement", {"csv_text": "not,a,statement\n", "source_name": "x.csv"})
    assert bad["error"]["type"] == "invalid_statement"
