"""The fictional card statement generator: it reconciles, it is repeatable, and the demo accepts it."""

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from household_agent.ingest import parse_statement
from household_agent.synthetic import realistic_card_statement
from household_agent.webapp import DemoService, create_app


def test_the_statement_reconciles_by_construction():
    parsed = parse_statement(realistic_card_statement().encode(), "2026_10_card.csv")
    h, rows = parsed.header, parsed.rows
    assert h.account_id == "acct_sayali_card" and not parsed.row_errors
    total = sum(r.amount_minor for r in rows)
    assert h.opening_balance_minor + total == h.closing_balance_minor
    assert h.total_credits_minor == sum(r.amount_minor for r in rows if r.amount_minor > 0)
    assert h.total_debits_minor == -sum(r.amount_minor for r in rows if r.amount_minor < 0)


def test_it_looks_like_a_card_month():
    rows = parse_statement(realistic_card_statement().encode(), "c.csv").rows
    assert len(rows) >= 30
    assert sum(r.amount_minor > 0 for r in rows) == 3  # two payments and a return
    assert any("SAFEWAY" in r.description for r in rows) and any("PG&E" in r.description for r in rows)
    assert all(r.posted_at.isoformat().startswith("2026-10-") for r in rows)


def test_the_same_seed_gives_the_same_file_and_another_seed_differs():
    assert realistic_card_statement(seed=3) == realistic_card_statement(seed=3)
    assert realistic_card_statement(seed=3) != realistic_card_statement(seed=4)


@settings(deadline=None, max_examples=40)
@given(st.integers(0, 10_000), st.sampled_from(["2026-02", "2026-09", "2026-10", "2027-12"]))
def test_any_seed_and_month_reconciles_inside_the_month(seed, month):
    parsed = parse_statement(realistic_card_statement(seed, month).encode(), "c.csv")
    assert not parsed.row_errors
    assert parsed.header.opening_balance_minor + sum(r.amount_minor for r in parsed.rows) == parsed.header.closing_balance_minor
    assert all(r.posted_at.isoformat().startswith(month) for r in parsed.rows)


def test_the_demo_accepts_it_for_sayali_and_refuses_it_for_somal(tmp_path):
    svc = DemoService(tmp_path)
    try:
        client = TestClient(create_app(svc))
        content = realistic_card_statement().encode()

        def up(member):
            return client.post("/api/upload", data={"member": member}, files={"file": ("card.csv", content, "text/csv")})

        assert up("somal").status_code == 403
        ok = up("sayali")
        assert ok.status_code == 200 and ok.json()["outcome"] == "reconciled"
    finally:
        svc.close()


def _ask_for_the_month_by_category(tmp_path, claims):
    """Sayali uploads the card month; a scripted analyst runs category_totals and states ``claims`` in dollars."""
    from household_agent import webapp
    from household_agent.vertical_slice import PlaybackChatModel, _call, _final, _results, _txn_ids

    run = lambda m: _call("run_calculation", fn="category_totals", transaction_ids=_txn_ids(m),  # noqa: E731
                          params={"start": "2026-10-01", "end": "2026-10-31"})
    final = lambda m: _final(*[{"text": t, "evidence_ids": [_results(m, "run_calculation")[-1]["run_id"]]}  # noqa: E731
                               for t in claims])
    model = PlaybackChatModel(steps=[
        _call("query_ledger", account_ids=["acct_sayali_card"], start="2026-10-01", end="2026-10-31"), run, final])
    svc = DemoService(tmp_path)
    try:
        assert svc.upload("sayali", "card.csv", realistic_card_statement().encode())[0] == 200
        original, webapp.ledger_playback = webapp.ledger_playback, lambda: model
        try:
            return svc.chat("sayali", "break down what we spent last month")
        finally:
            webapp.ledger_playback = original
    finally:
        svc.close()


def test_a_category_total_quoted_in_dollars_is_kept(tmp_path):
    parsed = parse_statement(realistic_card_statement().encode(), "c.csv")
    groceries = -sum(r.amount_minor for r in parsed.rows if "SAFEWAY" in r.description)
    text = f"Groceries were ${groceries // 100:,}.{groceries % 100:02d} last month."
    out = _ask_for_the_month_by_category(tmp_path, [text])
    assert [s["text"] for s in out["statements"]] == [text]
    assert not any("no verified figures" in q for q in out["questions"])


def test_a_category_total_that_is_not_in_the_ledger_is_still_removed(tmp_path):
    out = _ask_for_the_month_by_category(tmp_path, ["Groceries were $999,999.99 last month."])
    assert out["statements"] == [] and out["needs_review"]


def test_an_impossible_month_is_rejected():
    with pytest.raises(ValueError):
        realistic_card_statement(month="2026-13")
