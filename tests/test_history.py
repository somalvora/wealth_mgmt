"""Two years of fictional history: generated statements, the vector index of them, and who may search what."""

import re
from collections import defaultdict

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from household_agent.history import (
    contributed_minor,
    HISTORY_MONTHS,
    HISTORY_PREFIX,
    SKIPPED,
    brokerage_summaries,
    history_statements,
    months,
    statement_source,
)
from household_agent.ingest import parse_statement
from household_agent.retrieval import chunk_document
from household_agent.synthetic import DEMO_AS_OF, SAYALI, SOMAL, VIAAN, demo_statements
from household_agent.webapp import DemoService

HIST = history_statements()


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    svc = DemoService(tmp_path_factory.mktemp("history"))
    yield svc
    svc.close()


def parsed(name, text):
    return parse_statement(text.encode(), name)


# --- the generated statements -------------------------------------------------------------------------------------

def test_there_are_24_months_and_every_account_month_except_the_two_the_demo_owns():
    assert len(months()) == HISTORY_MONTHS == 24 and months()[0] == "2024-10" and months()[-1] == "2026-09"
    assert len(HIST) == 3 * 24 - len(SKIPPED) == 70
    assert len({name for name, _, _ in HIST}) == 70 and all(n.startswith(HISTORY_PREFIX) for n, _, _ in HIST)
    names = {n for n, _, _ in HIST}
    assert "hist_2025_10_rental_checking.csv" not in names and "hist_2026_09_somal_checking.csv" not in names


def test_every_statement_parses_balances_and_belongs_to_the_uploader_who_may_write_it():
    owner = {"acct_rental_checking": {"somal"}, "acct_somal_checking": {"somal"}, "acct_sayali_card": {"sayali"}}
    for name, member, text in HIST:
        p = parsed(name, text)
        assert not p.row_errors and p.rows
        assert member in owner[p.header.account_id]
        credits = sum(r.amount_minor for r in p.rows if r.amount_minor > 0)
        assert p.header.opening_balance_minor + sum(r.amount_minor for r in p.rows) == p.header.closing_balance_minor
        assert credits == p.header.total_credits_minor


def test_each_account_continues_from_the_previous_months_closing_balance():
    by_account = defaultdict(list)
    for name, _, text in HIST:
        p = parsed(name, text)
        by_account[p.header.account_id].append(p.header)
    for headers in by_account.values():
        headers.sort(key=lambda h: h.period_start)
        for prev, nxt in zip(headers, headers[1:]):
            gap = (nxt.period_start.year - prev.period_start.year) * 12 + nxt.period_start.month - prev.period_start.month
            if gap == 1:  # a skipped month breaks the chain by design
                assert nxt.opening_balance_minor == prev.closing_balance_minor


def test_the_history_is_deterministic_and_a_different_seed_changes_it():
    assert history_statements() == HIST
    assert history_statements(seed=8) != HIST


def test_the_brokerage_transfers_in_the_ledger_match_the_summaries():
    transfers = 0
    for name, _, text in HIST:
        transfers += sum(-r.amount_minor for r in parsed(name, text).rows if "ACME BROKERAGE" in r.description)
    summaries = brokerage_summaries()
    assert len(summaries) == 8
    total = float(re.search(r"Total contributed to date: ([\d,.]+) dollars", summaries[-1].text).group(1).replace(",", ""))
    assert int(total * 100) == transfers == sum(contributed_minor(m) for m in months())


def test_every_brokerage_summary_states_value_and_gain_and_is_private_to_somal():
    for d in brokerage_summaries():
        assert d.account_id == "acct_somal_checking" and d.collection == "financial_history"
        assert "Portfolio value at quarter end" in d.text and "Gain or loss to date" in d.text
        assert len(chunk_document(d)) == 1


@given(st.integers(min_value=0, max_value=50))
@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_any_seed_gives_balanced_statements_for_the_same_months(seed):
    out = history_statements(seed=seed)
    assert [n for n, _, _ in out] == [n for n, _, _ in HIST]
    for name, _, text in out[:6]:
        p = parsed(name, text)
        assert p.header.opening_balance_minor + sum(r.amount_minor for r in p.rows) == p.header.closing_balance_minor


# --- statement_source ---------------------------------------------------------------------------------------------

def test_a_statement_becomes_a_searchable_document_with_its_period_and_rows():
    name, text = "2025_10_rental_checking.csv", demo_statements()["2025_10_rental_checking.csv"]
    doc = statement_source(parsed(name, text), name, "doc_abc")
    assert doc.doc_id == "stmt_doc_abc" and doc.collection == "financial_history" and doc.doc_type == "bank_statement"
    assert doc.account_id == "acct_rental_checking" and doc.statement_period == "2025-10"
    assert "SAN MATEO TAX COLLECTOR" in doc.text and "-2750.00" in doc.text and "2025-10-31" in doc.text
    chunks = chunk_document(doc)
    assert len(chunks) == 1 and chunks[0].text.startswith("# account_id")


def test_a_long_statement_is_split_into_chunks_that_each_repeat_the_header():
    name, text = next((n, t) for n, _, t in HIST if n.endswith("sayali_card.csv"))
    chunks = chunk_document(statement_source(parsed(name, text), name, "d"))
    assert len(chunks) >= 2 and all(c.text.startswith("# account_id") and "date,description,amount" in c.text for c in chunks)


# --- seeding the demo ---------------------------------------------------------------------------------------------

def test_seeding_fills_the_ledger_and_the_vector_index_but_not_the_document_list(seeded):
    assert [d["name"] for d in seeded.documents] == ["2025_10_rental_checking.csv"]
    rows = seeded.conn.execute("SELECT reconciliation_status, count(*) FROM documents GROUP BY 1").fetchall()
    assert rows == [("reconciled", 71)]
    fh = seeded.retrieval.client.get_collection("hh_demo__financial_history")
    metas = fh.get(include=["metadatas"])["metadatas"]
    assert {m["statement_period"] for m in metas} >= {"2024-10", "2026-09", "2025-12"}
    assert {m["doc_type"] for m in metas} == {"bank_statement", "brokerage_summary"}
    assert seeded.audit_status()["intact"]


def test_a_reused_workdir_is_not_seeded_twice(tmp_path):
    first = DemoService(tmp_path)
    count, chunks = (first.conn.execute("SELECT count(*) FROM documents").fetchone()[0],
                     first.retrieval.client.get_collection("hh_demo__financial_history").count())
    first.close()
    again = DemoService(tmp_path)
    assert again.conn.execute("SELECT count(*) FROM documents").fetchone()[0] == count
    assert again.retrieval.client.get_collection("hh_demo__financial_history").count() == chunks
    assert [d["name"] for d in again.documents] == ["2025_10_rental_checking.csv"]  # still no history in the list
    again.close()


def test_history_can_be_turned_off(tmp_path):
    svc = DemoService(tmp_path, history=False)
    assert svc.conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 1
    svc.close()


def test_a_new_upload_is_searchable_when_it_reconciles_and_not_when_it_does_not(tmp_path):
    svc = DemoService(tmp_path, history=False)
    good = svc.upload("somal", "2026_10_somal_checking.csv", demo_statements()["2026_10_somal_checking.csv"].encode())
    bad = svc.upload("somal", "2026_09_somal_checking_corrupt.csv", demo_statements()["2026_09_somal_checking_corrupt.csv"].encode())
    assert good[1]["outcome"] == "reconciled" and bad[1]["outcome"] != "reconciled"
    ids = svc.retrieval.client.get_collection("hh_demo__financial_history").get()["ids"]
    assert any(i.startswith(f"stmt_{good[1]['document_id']}") for i in ids)
    assert not any(i.startswith(f"stmt_{bad[1]['document_id']}") for i in ids)  # a statement that does not balance is not evidence
    svc.close()


# --- who may find what --------------------------------------------------------------------------------------------

def search(svc, actor, query, **kw):
    return svc.retrieval.retrieve_verified(actor, "historical_comparison", query, correlation_id="corr_t", at=DEMO_AS_OF,
                                           collections=("financial_history",), **kw)["passages"]


def test_each_member_finds_only_the_accounts_they_may_read(seeded):
    somal = {p["metadata"]["account_id"] for p in search(seeded, SOMAL, "grocery and clinic charges card statement")}
    sayali = {p["metadata"]["account_id"] for p in search(seeded, SAYALI, "ACME brokerage transfer portfolio value")}
    assert "acct_sayali_card" not in somal  # Sayali's private card
    assert "acct_somal_checking" not in sayali  # Somal's private checking and brokerage summaries
    assert search(seeded, VIAAN, "rent deposit") == []


def test_a_statement_period_filter_returns_only_that_month(seeded):
    passages = search(seeded, SOMAL, "rent deposit and HOA dues", statement_periods=["2025-03"])
    assert passages and {p["metadata"]["statement_period"] for p in passages} == {"2025-03"}


def test_searching_the_brokerage_finds_the_summaries_and_the_contribution_rows(seeded):
    passages = search(seeded, SOMAL, "how much did we contribute to the brokerage and what is it worth now")
    kinds = {p["metadata"]["doc_type"] for p in passages}
    assert "brokerage_summary" in kinds and all(p["verification"] == "verified" for p in passages)
