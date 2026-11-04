import json
from datetime import timedelta
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from household_agent.approval import build_approval_graph, start_request
from household_agent.calc_runner import CalculationRunner
from household_agent.mcp_server import MEMBERS
from household_agent.pipeline import ingest_statement
from household_agent.plan_inputs import (
    Liability,
    _months_back,
    derive_planning_inputs,
    ledger_inputs_for,
    plan_from_ledger,
)
from household_agent.synthetic import ALEX, DEMO_AS_OF, HOUSEHOLD, JORDAN, SAM, Row, demo_planning_inputs, render_statement

COUNTY_BILL = Liability("county_bill_2026", 320000, "next Unit B property-tax installment")


def _kwargs(**kw):
    d = demo_planning_inputs()
    return {"contribution_room_minor": d.contribution_room_minor, "limits_evidence_id": d.limits_evidence_id,
            "real_return": d.real_return, "liabilities": [COUNTY_BILL], **kw}


def _derive(store, actor=ALEX, **kw):
    return derive_planning_inputs(store, CalculationRunner(store), actor, as_of=DEMO_AS_OF, correlation_id="plan",
                                  **_kwargs(**kw))


def _runs(store, fn):
    cur = store.conn.execute("SELECT run_id, input_hash, output_json, provisional, params_json FROM calculation_runs "
                             "WHERE fn = ?", (fn,))
    return [dict(zip(("run_id", "input_hash", "output", "provisional", "params"), r, strict=True)) for r in cur]


def _ingest(store, categorizer, text, name):
    return ingest_statement(store, categorizer, ALEX, text.encode(), name, purpose="statement_ingestion",
                            correlation_id="seed2", at=DEMO_AS_OF)


# --- happy path ------------------------------------------------------------------------

def test_alex_inputs_come_from_verified_statements(seeded_store):
    d = _derive(seeded_store)
    # closing balances: rental 16,571.60 + Alex checking 13,408.03 (October; the corrupt September is ignored)
    assert d.inputs.liquid_cash_minor == 1657160 + 1340803
    # essentials: rental HOA 450.00 + water 118.40; Alex mortgage 3,450.00 + PG&E 212.37 + groceries 329.60
    assert d.inputs.monthly_essential_outflow_minor == 56840 + 399197
    assert d.inputs.near_term_liabilities_minor == 320000
    assert {a["account_id"] for a in d.accounts} == {"acct_rental_checking", "acct_alex_checking"}
    assert all(a["months"] == ["2026-10"] for a in d.accounts)
    assert d.window == (DEMO_AS_OF.date().replace(month=8, day=1), DEMO_AS_OF.date())


def test_evidence_ids_cover_documents_runs_and_liabilities(seeded_store):
    d = _derive(seeded_store)
    assert set(d.calc_run_ids) <= set(d.evidence_ids)
    assert {"county_bill_2026", "SYNTHETIC_limits_2026"} <= set(d.evidence_ids)
    assert {a["document_id"] for a in d.accounts} <= set(d.evidence_ids)
    assert len(_runs(seeded_store, "category_totals")) == 2


def test_plan_run_is_recorded_and_reproducible(seeded_store):
    runner = CalculationRunner(seeded_store)
    r1, d1, id1 = plan_from_ledger(seeded_store, runner, ALEX, as_of=DEMO_AS_OF, correlation_id="p1",
                                   **_kwargs(reserve_months=3))
    r2, _, id2 = plan_from_ledger(seeded_store, runner, ALEX, as_of=DEMO_AS_OF, correlation_id="p2",
                                  **_kwargs(reserve_months=3))
    assert r1.status == "awaiting_approval" and id1 != id2
    runs = {r["run_id"]: r for r in _runs(seeded_store, "plan_allocation")}
    assert runs[id1]["input_hash"] == runs[id2]["input_hash"] == r1.input_hash == r2.input_hash
    assert json.loads(runs[id1]["output"])["allocation_minor"] == dict(sorted(r1.best.allocation.items()))
    assert json.loads(runs[id1]["params"])["provenance"]["excluded_accounts"] == d1.excluded_accounts
    event = next(e for e in seeded_store.audit.events("p1") if e["resource"] == f"calc:{id1}")
    assert event["purpose"] == "capital_allocation" and json.loads(event["details_json"])["fn"] == "plan_allocation"


def test_card_balance_owed_becomes_a_liability(store, categorizer):
    card = render_statement("acct_jordan_card", "2026-10-01", "2026-10-31", "-500.00",
                            [Row("2026-10-07", "SAFEWAY #0457", "-120.00")])
    ingest_statement(store, categorizer, JORDAN, card.encode(), "card.csv", purpose="statement_ingestion",
                     correlation_id="s", at=DEMO_AS_OF)
    d = derive_planning_inputs(store, CalculationRunner(store), JORDAN, as_of=DEMO_AS_OF, correlation_id="p",
                               **_kwargs(liabilities=[]))
    assert d.inputs.near_term_liabilities_minor == 62000 and d.inputs.liquid_cash_minor == 0
    assert d.inputs.monthly_essential_outflow_minor == 12000


def test_multi_month_average_rounds_up(store, categorizer):
    for month, water in (("09", "-100.01"), ("10", "-100.00")):
        _ingest(store, categorizer, render_statement(
            "acct_rental_checking", f"2026-{month}-01", f"2026-{month}-30", "1000.00",
            [Row(f"2026-{month}-09", "CITY WATER UTILITY", water)]), f"r{month}.csv")
    d = _derive(store)
    rental = next(a for a in d.accounts if a["account_id"] == "acct_rental_checking")
    assert rental["months"] == ["2026-09", "2026-10"] and rental["closing_balance_minor"] == 90000
    assert d.inputs.monthly_essential_outflow_minor == 10001  # 20,001 / 2 rounded up


# --- negative paths --------------------------------------------------------------------

def test_jordan_never_sees_alex_private_balance(seeded_store):
    d = _derive(seeded_store, actor=JORDAN)
    assert "acct_alex_checking" in d.excluded_accounts and not d.complete
    assert d.inputs.liquid_cash_minor == 1657160
    assert all(a["account_id"] != "acct_alex_checking" for a in d.accounts)


def test_dependent_gets_empty_inputs_and_not_eligible(seeded_store):
    result, d, run_id = plan_from_ledger(seeded_store, CalculationRunner(seeded_store), SAM, as_of=DEMO_AS_OF,
                                         correlation_id="p", **_kwargs())
    assert d.accounts == () and d.inputs.liquid_cash_minor == 0 and result.status == "not_eligible"
    assert _runs(seeded_store, "plan_allocation")[0]["provisional"] == 1


def test_corrupt_statement_never_supplies_a_balance(store, categorizer):
    _ingest(store, categorizer, render_statement(
        "acct_alex_checking", "2026-10-01", "2026-10-31", "1000.00", [Row("2026-10-03", "PAYROLL ACME CORP", "10.00")],
        closing_override="99999.99"), "corrupt.csv")
    d = _derive(store)
    assert d.excluded_accounts["acct_alex_checking"] == "no verified statement in the window"
    assert d.inputs.liquid_cash_minor == 0


def test_statements_outside_the_window_are_ignored(store, categorizer):
    _ingest(store, categorizer, render_statement(
        "acct_rental_checking", "2026-06-01", "2026-06-30", "50000.00", []), "june.csv")
    assert _derive(store).inputs.liquid_cash_minor == 0
    assert _derive(store, lookback_months=6).inputs.liquid_cash_minor == 5000000


def test_statement_balances_are_append_only(seeded_store):
    with pytest.raises(Exception, match="append-only"):
        seeded_store.conn.execute("UPDATE statement_balances SET closing_balance_minor = 1")


@pytest.mark.parametrize("bad", [dict(evidence_id="", amount_minor=1), dict(evidence_id="x", amount_minor=-1),
                                 dict(evidence_id="x", amount_minor=1.5)])
def test_liability_requires_evidence_and_minor_units(bad):
    with pytest.raises(ValueError):
        Liability(**bad)


def test_inputs_for_rejects_another_household(seeded_store):
    fn = ledger_inputs_for(seeded_store, CalculationRunner(seeded_store), ALEX, clock=lambda: DEMO_AS_OF, **_kwargs())
    assert fn(HOUSEHOLD).liquid_cash_minor == 2997963
    with pytest.raises(PermissionError):
        fn("hh_other")


# --- approval graph wiring ---------------------------------------------------------------

def test_new_statement_after_plan_makes_approval_stale(seeded_store, categorizer):
    now = [DEMO_AS_OF]
    inputs_for = ledger_inputs_for(seeded_store, CalculationRunner(seeded_store), ALEX, clock=lambda: now[0],
                                   **_kwargs(reserve_months=3))
    graph = build_approval_graph(inputs_for=inputs_for, members=MEMBERS, audit=seeded_store.audit,
                                 checkpointer=InMemorySaver())
    thread_id, out = start_request(graph, HOUSEHOLD, "alex")
    assert out["status"] == "awaiting_approval"
    _ingest(seeded_store, categorizer, render_statement(
        "acct_rental_checking", "2026-09-01", "2026-09-30", "15000.00",
        [Row("2026-09-05", "MAPLE ST HOA DUES", "-450.00")]), "sept_rental.csv")
    now[0] += timedelta(hours=1)
    out = graph.invoke(Command(resume={"approver": "jordan", "approved": True}),
                       {"configurable": {"thread_id": thread_id}})
    assert out["status"] == "stale"


# --- properties --------------------------------------------------------------------------

@given(st.dates(), st.integers(min_value=0, max_value=36))
def test_months_back_is_first_of_month_and_not_after_date(d, months):
    m = _months_back(d, months) if d.year > 4 else None
    if m:
        assert m.day == 1 and m <= d and (d.year - m.year) * 12 + d.month - m.month == months


@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(st.lists(st.integers(min_value=1, max_value=10**8), max_size=4))
def test_liabilities_sum_exactly_and_raise_the_reserve(seeded_store, amounts):
    items = [Liability(f"ev_{i}", a) for i, a in enumerate(amounts)]
    d = _derive(seeded_store, liabilities=items)
    assert d.inputs.near_term_liabilities_minor == sum(amounts)
    assert set(d.liability_evidence_ids) == {li.evidence_id for li in items}
    assert d.inputs.required_reserve_minor == 6 * d.inputs.monthly_essential_outflow_minor + sum(amounts)
    assert isinstance(d.inputs.real_return["hsa"], Decimal)
