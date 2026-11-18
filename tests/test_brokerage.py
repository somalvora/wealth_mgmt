"""Brokerage statements: the two calculations, the ledger rules for storing them, and who may read them."""

import json
from dataclasses import replace
from fractions import Fraction

import pytest

from household_agent.audit import AuditLog
from household_agent.authz import AuthorizationError, Authorizer
from household_agent.brokerage import (
    BROKERAGE_ACCOUNT_ID,
    LIMITS_BPS,
    BrokerageStatementError,
    brokerage_documents,
    concentration,
    demo_brokerage_months,
    performance,
    policy_text,
)
from household_agent.ledger import DuplicateDocumentError, LedgerStore
from household_agent.models import Action
from household_agent.synthetic import DEMO_ACCOUNTS, DEMO_AS_OF, SAYALI, SOMAL, VIAAN
from household_agent.tools import ToolSession

AUG, SEP, OCT = demo_brokerage_months()


@pytest.fixture
def bstore(conn):
    return LedgerStore(conn, Authorizer(DEMO_ACCOUNTS), AuditLog(conn))


@pytest.fixture
def loaded(bstore):
    for m in (AUG, SEP, OCT):
        bstore.ingest_brokerage_month(SOMAL, "statement_ingestion", "seed", m, at=DEMO_AS_OF)
    return bstore


def _session(store, categorizer, actor=SOMAL, purpose="historical_comparison"):
    return ToolSession(store, categorizer, actor, purpose, correlation_id="t1", clock=lambda: DEMO_AS_OF,
                       acting_agent="ledger_analyst")


# --- the numbers -----------------------------------------------------------------------------------------------

def _pct(n, d):  # an independent rounding, half up, on exact fractions
    x = Fraction(n * 1000, d)
    return f"{(x.numerator * 2 + x.denominator) // (x.denominator * 2) / 10:.1f}" if n >= 0 else \
        f"-{(-x.numerator * 2 + x.denominator) // (x.denominator * 2) / 10:.1f}"


def test_the_statements_chain_and_match_the_quarterly_summary():
    assert [m.opening_minor for m in (SEP, OCT)] == [AUG.closing_minor, SEP.closing_minor]
    assert SEP.closing_minor == 3133423  # 31,334.23, the Q3 2026 figure in history.py
    assert all(m.deposits_minor == 150000 for m in (AUG, SEP, OCT))  # the ACME BROKERAGE TRANSFER rows


def test_performance_separates_deposits_from_the_market():
    out = performance(OCT)
    assert (out["opening_minor"], out["closing_minor"], out["deposits_minor"], out["dividends_minor"]) == \
           (3133423, 3104573, 150000, 3140)
    assert out["change_minor"] == -28850 and out["market_effect_minor"] == -181990
    assert out["market_effect_pct"] == _pct(-181990, 3133423) == "-5.8"
    # deposits + dividends + market effect always explain the whole change
    for m in (AUG, SEP, OCT):
        p = performance(m)
        assert p["deposits_minor"] + p["dividends_minor"] + p["market_effect_minor"] == p["change_minor"]


def test_concentration_matches_an_independent_recomputation():
    for m in (AUG, SEP, OCT):
        out = concentration(m)
        total = m.cash_minor + sum(Fraction(h.quantity) * h.price_minor for h in m.holdings)
        assert out["total_minor"] == total
        share = {h.symbol: Fraction(h.quantity * h.price_minor, 1) / total for h in m.holdings}
        ranked = sorted(share, key=lambda s: (-share[s], s))
        assert out["largest_symbol"] == ranked[0] and out["top4_symbols"] == ranked[:4]
        assert out["top4_share_pct"] == _pct(sum(share[s] * total for s in ranked[:4]), total)
        tech = sum(h.quantity * h.price_minor for h in m.holdings if h.sector == "technology")
        assert out["technology_value_minor"] == tech and out["technology_share_pct"] == _pct(tech, total)
        stocks_over = sorted(h.symbol for h in m.holdings
                             if h.kind == "stock" and Fraction(h.quantity * h.price_minor, total) > Fraction(20, 100))
        assert sorted(out["stocks_over_single_stock_limit"]) == stocks_over
        assert out["technology_over_limit"] is (Fraction(tech, total) > Fraction(1, 2))


def test_october_concentration_story():
    out = concentration(OCT)
    assert out["largest_symbol"] == "NVDA" and out["largest_share_pct"] == "23.1"
    assert out["stocks_over_single_stock_limit"] == ["NVDA"] and out["technology_share_pct"] == "46.2"
    assert out["technology_over_limit"] is False and out["top4_share_pct"] == "54.9"
    assert concentration(SEP)["technology_over_limit"] is True  # September was over the sector limit


def test_funds_are_exempt_from_the_single_stock_limit():
    big_fund = replace(OCT, holdings=tuple(replace(h, quantity=h.quantity * 20) if h.symbol == "VOO" else h
                                           for h in OCT.holdings))
    assert concentration(big_fund)["largest_symbol"] == "VOO"
    assert "VOO" not in concentration(big_fund)["stocks_over_single_stock_limit"]


def test_the_policy_document_states_the_same_limits_the_calculation_uses():
    text = policy_text()
    assert f"{LIMITS_BPS['single_stock'] // 100} percent" in text and f"{LIMITS_BPS['technology_sector'] // 100} percent" in text
    docs = {d.doc_id: d for d in brokerage_documents()}
    assert docs["brokerage_concentration_policy"].collection == "policies_compliance"
    assert {d.account_id for d in docs.values()} == {BROKERAGE_ACCOUNT_ID}
    assert sum(d.doc_type == "brokerage_statement" for d in docs.values()) == 3


# --- storing them -----------------------------------------------------------------------------------------------

def test_ingest_enforces_continuity_duplicates_and_account_kind(bstore):
    ingest = lambda m, actor=SOMAL, **kw: bstore.ingest_brokerage_month(  # noqa: E731
        actor, "statement_ingestion", "c1", m, at=DEMO_AS_OF, **kw)
    ingest(AUG)
    with pytest.raises(DuplicateDocumentError):
        ingest(AUG)
    with pytest.raises(BrokerageStatementError, match="opens at"):
        ingest(replace(OCT, opening_minor=OCT.opening_minor + 1))  # does not continue the previous month... (no Sep yet)
    ingest(SEP)
    with pytest.raises(BrokerageStatementError, match="opens at"):
        ingest(replace(OCT, opening_minor=OCT.opening_minor + 1))
    with pytest.raises(BrokerageStatementError):
        ingest(replace(OCT, account_id="acct_somal_checking"))  # not a brokerage account
    assert [r[0] for r in bstore.conn.execute("SELECT period FROM brokerage_statements ORDER BY period")] == \
           ["2026-08", "2026-09"]


def test_only_the_owner_can_store_or_read_a_brokerage_statement(loaded):
    with pytest.raises(AuthorizationError):
        loaded.ingest_brokerage_month(SAYALI, "statement_ingestion", "c2", replace(AUG, period="2026-07"), at=DEMO_AS_OF)
    with pytest.raises(AuthorizationError):
        loaded.brokerage_month(SAYALI, "historical_comparison", "c2", BROKERAGE_ACCOUNT_ID)
    assert loaded.brokerage_month(SOMAL, "historical_comparison", "c2", BROKERAGE_ACCOUNT_ID).period == "2026-10"
    assert loaded.brokerage_month(SOMAL, "historical_comparison", "c2", BROKERAGE_ACCOUNT_ID, "2026-08").period == "2026-08"
    with pytest.raises(KeyError):
        loaded.brokerage_month(SOMAL, "historical_comparison", "c2", BROKERAGE_ACCOUNT_ID, "2025-01")


def test_stored_statements_are_append_only(loaded):
    with pytest.raises(Exception, match="append-only"):
        loaded.conn.execute("UPDATE brokerage_statements SET cash_minor = 1")


# --- the tools --------------------------------------------------------------------------------------------------

def test_tools_return_a_recorded_run_for_the_latest_month(loaded, categorizer):
    s = _session(loaded, categorizer)
    out = s.call("brokerage_performance", {"account_id": BROKERAGE_ACCOUNT_ID})
    assert out["ok"] and out["output"]["period"] == "2026-10" and out["output"]["market_effect_minor"] == -181990
    row = loaded.conn.execute("SELECT fn, output_json, input_hash, output_hash FROM calculation_runs WHERE run_id = ?",
                              (out["run_id"],)).fetchone()
    assert row[0] == "brokerage_performance" and json.loads(row[1]) == out["output"]
    assert (row[2], row[3]) == (out["input_hash"], out["output_hash"])
    assert any(e["resource"] == f"calc:{out['run_id']}" for e in loaded.audit.events("t1"))
    sep = s.call("brokerage_concentration", {"account_id": BROKERAGE_ACCOUNT_ID, "month": "2026-09"})
    assert sep["ok"] and sep["output"]["technology_over_limit"] is True and sep["run_id"] != out["run_id"]


def test_tools_refuse_other_members_and_unknown_months(loaded, categorizer):
    for actor in (SAYALI, VIAAN):
        out = _session(loaded, categorizer, actor).call("brokerage_concentration", {"account_id": BROKERAGE_ACCOUNT_ID})
        assert out["ok"] is False and out["error"]["type"] == "authorization_denied"
    assert [e["decision"] for e in loaded.audit.events("t1")].count("deny") >= 2
    out = _session(loaded, categorizer).call("brokerage_performance", {"account_id": BROKERAGE_ACCOUNT_ID, "month": "2020-01"})
    assert out["ok"] is False and out["error"]["type"] == "invalid_request"

