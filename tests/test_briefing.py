import json
import sqlite3
from datetime import date

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from household_agent.agents import AgentResult, ResultClaim
from household_agent.audit import AuditLog
from household_agent.authz import Authorizer
from household_agent.briefing import (
    DISCLAIMER,
    SECTIONS,
    BriefingItem,
    MonthlyBriefing,
    draft_monthly_briefing,
    render_markdown,
)
from household_agent.calc_runner import CalculationRunner
from household_agent.categorize import Categorizer
from household_agent.ledger import LedgerStore
from household_agent.pipeline import ingest_statement
from household_agent.plan_inputs import Liability, plan_from_ledger
from household_agent.synthetic import (
    ACCOUNTS,
    ALEX,
    CASEY_CPA,
    DEMO_AS_OF,
    JORDAN,
    RULES,
    SAM,
    Row,
    cpa_grant,
    demo_planning_inputs,
    render_statement,
)

OCT = date(2026, 10, 1)


def _brief(store, actor=ALEX, month=OCT, **kw):
    return draft_monthly_briefing(store, CalculationRunner(store), actor, month=month, as_of=DEMO_AS_OF,
                                  correlation_id="brief", **kw)


def _kinds(items):
    return [i.kind for i in items]


# --- happy path ------------------------------------------------------------------------

def test_alex_october_figures_are_hand_computed(seeded_store):
    b = _brief(seeded_store)
    flow = next(i for i in b.net_worth_and_cash_flow if i.kind == "cash_flow")
    # inflow: rent 3,100 + payroll 2 x 4,100; outflow: mortgage 3,450 + PG&E 212.37 + groceries 329.60
    #         + HOA 450 + water 118.40 + property tax 3,200
    assert flow.amounts_minor == {"inflow": 1130000, "outflow": 776037, "net": 353963}
    cats = next(i for i in b.budget_and_variance if i.kind == "category_totals")
    assert cats.amounts_minor["property_tax"] == -320000 and cats.amounts_minor["groceries"] == -32960
    assert sum(cats.amounts_minor.values()) == flow.amounts_minor["net"]
    closing = {i.text.split()[0]: i.amounts_minor["closing"] for i in b.net_worth_and_cash_flow
               if i.kind == "closing_balance"}
    assert closing == {"acct_rental_checking": 1657160, "acct_alex_checking": 1340803}
    assert _kinds(b.data_coverage) == ["statement", "statement", "not_shared"]


def test_calculations_are_recorded_and_the_briefing_is_audited(seeded_store):
    b = _brief(seeded_store)
    rows = seeded_store.conn.execute("SELECT run_id FROM calculation_runs").fetchall()
    assert set(b.calc_run_ids) <= {r[0] for r in rows} and len(b.calc_run_ids) == 3
    event = next(e for e in seeded_store.audit.events("brief") if e["action"] == "compose_briefing")
    assert event["resource"] == "briefing:2026-10" and json.loads(event["details_json"])["calc_run_ids"] == b.calc_run_ids


def test_material_events_skip_routine_items_and_list_property_tax(seeded_store):
    events = _brief(seeded_store).material_events
    assert [i.text for i in events] == ["2026-10-18 San Mateo County Tax Collector (property_tax)"]
    assert events[0].amounts_minor == {"amount": -320000} and len(events[0].evidence_ids) == 2


def test_unmapped_high_value_item_is_a_material_event(store, categorizer):
    ingest_statement(store, categorizer, ALEX, render_statement(
        "acct_alex_checking", "2026-10-01", "2026-10-31", "5000.00",
        [Row("2026-10-04", "MYSTERY VENDOR 99", "-600.00"), Row("2026-10-05", "MYSTERY VENDOR 98", "-499.99")]
    ).encode(), "x.csv", purpose="statement_ingestion", correlation_id="s", at=DEMO_AS_OF)
    events = _brief(store).material_events
    assert [(i.kind, i.amounts_minor["amount"]) for i in events] == [("unmapped_high_value", -60000)]


def test_plan_and_pending_approval_appear_under_actions(seeded_store):
    d = demo_planning_inputs()
    plan = plan_from_ledger(seeded_store, CalculationRunner(seeded_store), ALEX, as_of=DEMO_AS_OF, correlation_id="p",
                            contribution_room_minor=d.contribution_room_minor, limits_evidence_id=d.limits_evidence_id,
                            real_return=d.real_return, liabilities=[Liability("county_bill_2026", 320000)],
                            reserve_months=3)
    b = _brief(seeded_store, plan=plan, approval_thread_id="appr_123")
    best, runner_up, pending = b.actions_for_approval
    result, derived, run_id = plan
    assert best.kind == "allocation_scenario" and sum(best.amounts_minor.values()) == derived.inputs.surplus_minor
    assert run_id in best.evidence_ids and best.provisional  # Jordan's card is left out of Alex's inputs
    assert runner_up.amounts_minor != best.amounts_minor and pending.evidence_ids == ["appr_123"]
    assumptions = next(i for i in b.assumptions_and_limitations if i.kind == "planning_assumptions")
    assert assumptions.amounts_minor["surplus"] == derived.inputs.surplus_minor


def test_supervisor_results_are_sorted_into_sections(seeded_store):
    ledger = AgentResult(task_id="t1", correlation_id="brief", agent="ledger_analyst", status="completed",
                         evidence_ids=["calc_1"],
                         claims=[ResultClaim(text="Property tax rose 16.36%.", evidence_ids=["calc_1"])])
    policy = AgentResult(task_id="t2", correlation_id="brief", agent="policy_compliance", status="needs_review",
                         needs_review=True,
                         claims=[ResultClaim(text="Second installment is due by April 10.", evidence_ids=["doc_p"])],
                         questions=["I cannot verify the rule for this tax year."])
    finding = {"type": "material_variance", "pct_variance": "16.36", "threshold_pct": "10", "calc_run_id": "calc_1"}
    b = _brief(seeded_store, agent_results=[ledger, policy.model_dump()], findings=[finding])
    assert "validated_claim" in _kinds(b.budget_and_variance)
    assert _kinds(b.policy_and_deadlines) == ["validated_claim", "unverified_question"]
    assert b.material_events[-1].kind == "material_variance" and b.material_events[-1].evidence_ids == ["calc_1"]
    assert "agent_needs_review" in _kinds(b.assumptions_and_limitations)


def test_markdown_has_every_section_and_the_disclaimer(seeded_store):
    md = render_markdown(_brief(seeded_store))
    assert md.startswith("# Monthly briefing 2026-10 (draft)")
    assert all(f"## {title}" in md for _, title in SECTIONS)
    assert "Nothing to report." in md and md.rstrip().endswith(DISCLAIMER)
    assert "inflow $11,300.00, outflow $7,760.37, net $3,539.63" in md


# --- negative paths --------------------------------------------------------------------

def test_jordan_never_sees_alex_private_account(seeded_store):
    b = _brief(seeded_store, actor=JORDAN)
    dumped = b.model_dump_json()
    assert "acct_alex_checking" not in dumped and "1340803" not in dumped and "Summit" not in dumped
    assert _kinds(b.data_coverage).count("not_shared") == 1


def test_qualified_card_is_provisional_and_needs_confirmation(seeded_store):
    b = _brief(seeded_store, actor=JORDAN)
    card = next(i for i in b.data_coverage if "jordan_card" in i.text)
    assert card.provisional and "possible_duplicate" in card.text
    dup = next(i for i in b.material_events if i.kind == "possible_duplicate")
    assert dup.amounts_minor == {"amount": -9610} and dup.provisional
    assert _kinds(b.actions_for_approval) == ["confirm_statement"]
    left_out = next(i for i in b.budget_and_variance if i.kind == "provisional_transactions")
    assert len(left_out.evidence_ids) == 5
    cats = next(i for i in b.budget_and_variance if i.kind == "category_totals")
    assert "groceries" not in cats.amounts_minor and "entertainment" not in cats.amounts_minor


def test_corrupt_statement_contributes_no_numbers(seeded_store):
    b = _brief(seeded_store, month=date(2026, 9, 1))
    assert b.net_worth_and_cash_flow == [] and b.calc_run_ids == []
    assert "unreconciled" in b.data_coverage[0].text
    assert "unreconciled_excluded" in _kinds(b.assumptions_and_limitations)
    assert "999999" not in b.model_dump_json()


def test_month_without_statements_reports_missing_sources(seeded_store):
    b = _brief(seeded_store, month=date(2026, 8, 15))
    assert _kinds(b.data_coverage) == ["missing_statement", "missing_statement", "not_shared"]
    assert b.material_events == [] and b.calc_run_ids == []


@pytest.mark.parametrize("actor", [SAM, CASEY_CPA])
def test_dependent_and_tax_only_delegate_get_no_ledger_data(seeded_store, actor):
    seeded_store.authorizer.add_grant(cpa_grant(expires_at=DEMO_AS_OF.replace(year=2027)))
    b = _brief(seeded_store, actor=actor)
    assert _kinds(b.data_coverage) == ["not_shared"] and "3 household account" in b.data_coverage[0].text
    assert b.net_worth_and_cash_flow == b.budget_and_variance == b.material_events == []
    assert b.assumptions_and_limitations[-1].text == DISCLAIMER


def test_amounts_without_evidence_are_rejected():
    with pytest.raises(ValidationError, match="must cite evidence"):
        BriefingItem(kind="cash_flow", text="Net", amounts_minor={"net": 1})
    with pytest.raises(ValidationError):
        BriefingItem(kind="x", text="y", amounts_minor={"net": 1.5}, evidence_ids=["e"])
    with pytest.raises(ValidationError):
        BriefingItem(kind="x", text="y", advice="buy")


def test_briefing_schema_requires_draft_and_disclaimer(seeded_store):
    data = json.loads(_brief(seeded_store).model_dump_json())
    assert MonthlyBriefing.model_validate(data).draft is True
    for patch in ({"draft": False}, {"external_action": "transfer"},
                  {"assumptions_and_limitations": data["assumptions_and_limitations"][:-1]}):
        with pytest.raises(ValidationError):
            MonthlyBriefing.model_validate({**data, **patch})


# --- properties --------------------------------------------------------------------------

_rows = st.lists(st.tuples(st.integers(1, 28), st.sampled_from(["SAFEWAY #1", "PAYROLL ACME", "PG&E", "ODD SHOP"]),
                           st.integers(-500_000, 500_000).filter(bool)), max_size=12)


@settings(max_examples=25, deadline=None)
@given(_rows)
def test_category_totals_sum_to_net_cash_flow_and_amounts_cite_evidence(rows):
    conn = sqlite3.connect(":memory:")
    store = LedgerStore(conn, Authorizer(ACCOUNTS), AuditLog(conn))
    text = render_statement("acct_alex_checking", "2026-10-01", "2026-10-31", "100000.00",
                            [Row(f"2026-10-{d:02d}", m, f"{a / 100:.2f}") for d, m, a in rows])
    ingest_statement(store, Categorizer(RULES), ALEX, text.encode(), "p.csv", purpose="statement_ingestion",
                     correlation_id="s", at=DEMO_AS_OF)
    b = _brief(store)
    flow = next((i for i in b.net_worth_and_cash_flow if i.kind == "cash_flow"), None)
    if flow is not None:
        cats = next(i for i in b.budget_and_variance if i.kind == "category_totals")
        assert sum(cats.amounts_minor.values()) == flow.amounts_minor["net"]
    for key, _ in SECTIONS:
        assert all(i.evidence_ids for i in getattr(b, key) if i.amounts_minor)
