from dataclasses import replace

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from household_agent.agents import LEDGER_TOOLS
from household_agent.calc_runner import REGISTRY, REQUIRED_PARAMS
from household_agent.consent import ConsentStore
from household_agent.synthetic import SOMAL, CASEY_CPA, DEMO_AS_OF, SAYALI, VIAAN, cpa_grant, demo_statements
from household_agent.tools import ToolSession

TAX = "San Mateo County Tax Collector"
RENTAL = "acct_rental_checking"


def _session(store, categorizer, actor=SOMAL, purpose="historical_comparison", **kw):
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
    assert ids(_session(seeded_store, categorizer, SOMAL)) == [RENTAL, "acct_somal_checking"]
    assert ids(_session(seeded_store, categorizer, SAYALI)) == [RENTAL, "acct_sayali_card"]
    assert ids(_session(seeded_store, categorizer, VIAAN)) == []
    cpa = _session(seeded_store, categorizer, CASEY_CPA, purpose="tax_preparation")
    assert ids(cpa) == []
    consents = ConsentStore(conn, seeded_store.authorizer, seeded_store.audit)
    expires = DEMO_AS_OF.replace(year=2027)
    consents.grant(SOMAL, cpa_grant(expires), correlation_id="g", at=DEMO_AS_OF)
    consents.grant(SAYALI, replace(cpa_grant(expires), grant_id="g_j", grantor_id="sayali"), correlation_id="g", at=DEMO_AS_OF)
    assert ids(cpa) == [RENTAL]
    # Consent is purpose-scoped: the same CPA in a budgeting session sees nothing.
    assert ids(_session(seeded_store, categorizer, CASEY_CPA, purpose="budgeting")) == []


def test_cross_member_read_is_denied_and_audited(seeded_store, categorizer):
    s = _session(seeded_store, categorizer, SAYALI, purpose="budgeting")
    out = s.call("query_ledger", {"account_ids": ["acct_somal_checking"], "start": "2026-10-01", "end": "2026-10-31"})
    assert out == {"ok": False, "error": {"type": "authorization_denied", "message": out["error"]["message"]}}
    assert seeded_store.audit.events("t1")[-1]["decision"] == "deny"


def test_actor_cannot_be_passed_as_a_tool_argument(seeded_store, categorizer):
    s = _session(seeded_store, categorizer, SAYALI)
    out = s.call("query_ledger", {"actor": "somal", "account_ids": ["acct_somal_checking"],
                                  "start": "2026-10-01", "end": "2026-10-31"})
    assert out["error"]["type"] == "invalid_request" and "actor" in out["error"]["message"]


def test_calculation_only_on_transactions_this_session_queried(seeded_store, categorizer):
    somal = _session(seeded_store, categorizer)
    ids = [t["transaction_id"] for t in _tax_rows(somal)["transactions"]]
    fresh = _session(seeded_store, categorizer)  # did not query
    out = fresh.call("run_calculation", {"fn": "variance_between_transactions", "transaction_ids": ids})
    assert out["error"]["type"] == "ineligible_input"
    out = somal.call("run_calculation", {"fn": "variance_between_transactions", "transaction_ids": [ids[0], "txn_made_up"]})
    assert out["error"]["type"] == "ineligible_input" and "txn_made_up" in out["error"]["message"]


def test_provisional_rows_need_explicit_opt_in(seeded_store, categorizer):
    s = _session(seeded_store, categorizer, SAYALI, purpose="budgeting")
    q = {"account_ids": ["acct_sayali_card"], "start": "2026-10-01", "end": "2026-10-31"}
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


def _rental_ids(s):
    rows = s.call("query_ledger", {"account_ids": [RENTAL], "start": "2025-10-01", "end": "2026-10-31"})
    return [t["transaction_id"] for t in rows["transactions"]]


@pytest.mark.parametrize("fn", [f for f, keys in REQUIRED_PARAMS.items() if keys])
def test_missing_calculation_params_are_named_in_the_error(seeded_store, categorizer, fn):
    s = _session(seeded_store, categorizer)
    out = s.call("run_calculation", {"fn": fn, "transaction_ids": _rental_ids(s)})
    message = out["error"]["message"]
    assert out["error"]["type"] == "invalid_request"
    assert fn in message and all(k in message for k in REQUIRED_PARAMS[fn]) and "ISO dates" in message


def test_partly_supplied_params_report_only_what_is_missing(seeded_store, categorizer):
    s = _session(seeded_store, categorizer)
    out = s.call("run_calculation", {"fn": "category_totals", "transaction_ids": _rental_ids(s),
                                     "params": {"start": "2026-10-01"}})
    assert out["error"]["type"] == "invalid_request" and out["error"]["message"].endswith("missing: end")


def test_calculation_with_its_params_still_runs(seeded_store, categorizer):
    s = _session(seeded_store, categorizer)
    out = s.call("run_calculation", {"fn": "category_totals", "transaction_ids": _rental_ids(s),
                                     "params": {"start": "2025-10-01", "end": "2026-10-31"}})
    assert out["ok"] and out["fn"] == "category_totals"


def test_params_sent_as_a_json_string_are_read(seeded_store, categorizer):
    s = _session(seeded_store, categorizer)
    ids = _rental_ids(s)
    as_dict = s.call("run_calculation", {"fn": "category_totals", "transaction_ids": ids,
                                         "params": {"start": "2025-10-01", "end": "2026-10-31"}})
    as_text = s.call("run_calculation", {"fn": "category_totals", "transaction_ids": ids,
                                         "params": '{"start": "2025-10-01", "end": "2026-10-31"}'})
    assert as_text["ok"] and as_text["output"] == as_dict["output"]


@pytest.mark.parametrize("params", ["not json", "[1, 2]", '"just a string"', 5, ["start", "end"]])
def test_params_that_are_not_an_object_get_a_clear_error(seeded_store, categorizer, params):
    s = _session(seeded_store, categorizer)
    out = s.call("run_calculation", {"fn": "category_totals", "transaction_ids": _rental_ids(s), "params": params})
    assert out["error"]["type"] == "invalid_request" and "JSON object" in out["error"]["message"]


@given(params=st.one_of(st.text(max_size=20), st.integers(), st.lists(st.integers(), max_size=3)))
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_non_object_params_never_raise(params, seeded_store, categorizer):
    s = _session(seeded_store, categorizer)
    out = s.call("run_calculation", {"fn": "cash_flow", "transaction_ids": _rental_ids(s), "params": params})
    assert not out["ok"] and out["error"]["type"] == "invalid_request"


def test_tool_description_lists_every_calculations_params():
    assert set(REQUIRED_PARAMS) == set(REGISTRY)  # a new calculation must declare its params
    desc = next(t["function"]["description"] for t in LEDGER_TOOLS if t["function"]["name"] == "run_calculation")
    assert all(fn in desc and all(k in desc for k in keys) for fn, keys in REQUIRED_PARAMS.items())


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(extra=st.dictionaries(st.text(min_size=1, max_size=8), st.text(max_size=8), max_size=3))
def test_property_params_without_required_keys_are_always_refused(extra, seeded_store, categorizer):
    s = _session(seeded_store, categorizer)
    ids = _rental_ids(s)
    for fn, keys in REQUIRED_PARAMS.items():
        if not keys:
            continue
        params = {k: v for k, v in extra.items() if k not in keys}
        out = s.call("run_calculation", {"fn": fn, "transaction_ids": ids, "params": params})
        assert not out["ok"] and out["error"]["type"] == "invalid_request"


def test_ingest_tool_guards(store, categorizer):
    statements = demo_statements()
    sayali = _session(store, categorizer, SAYALI, purpose="statement_ingestion")
    out = sayali.call("ingest_statement", {"csv_text": statements["2026_10_somal_checking.csv"], "source_name": "a.csv"})
    assert out["error"]["type"] == "authorization_denied"
    ok = sayali.call("ingest_statement", {"csv_text": statements["2026_10_sayali_card.csv"], "source_name": "j.csv"})
    assert ok["ok"] and ok["status"] == "qualified" and ok["issues"][0]["code"] == "possible_duplicate"
    again = sayali.call("ingest_statement", {"csv_text": statements["2026_10_sayali_card.csv"], "source_name": "j.csv"})
    assert again["error"]["type"] == "duplicate_document"
    bad = sayali.call("ingest_statement", {"csv_text": "not,a,statement\n", "source_name": "x.csv"})
    assert bad["error"]["type"] == "invalid_statement"
