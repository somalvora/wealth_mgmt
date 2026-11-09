"""evaluate_allocation_candidates: the deterministic evaluator, the Supervisor-side tool, and the allocate flow.

Numbers in the happy-path tests are worked out by hand in the comments. Models are fakes: the model may propose
candidate shapes and choose/explain, but every figure comes from code and every claim is re-checked against it.
"""
import json
from dataclasses import dataclass
from typing import Any, Callable

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from household_agent.agents import Router, Supervisor, rule_route
from household_agent.allocation import (
    AllocationInputError,
    CandidatePlan,
    Debt,
    HouseholdFinancialState,
    RetirementAccount,
    evaluate_candidates,
)
from household_agent.allocation_advice import (
    BPS,
    CHOOSE_PROMPT,
    PROPOSE_PROMPT,
    CandidateShape,
    ShapeLine,
    expand_shape,
    extra_amount_from_request,
    standard_candidates,
)
from household_agent.allocation_tool import AllocationFacts, evaluate_allocation_candidates
from household_agent.synthetic import ALEX, CASEY_CPA, DEMO_AS_OF, SAM

CARD = Debt(debt_id="card", balance_minor=500_000, apr_percent="24", minimum_payment_minor=20_000)  # $5,000 at 24%
ROTH = RetirementAccount(account_id="roth", headroom_minor=700_000)  # $7,000 of room left this year
EXTRA = 200_000  # $2,000


def state(**kw) -> HouseholdFinancialState:
    base = dict(monthly_essential_expenses_minor=300_000,  # $3,000 a month
                liquid_cash_minor=1_000_000,  # $10,000 outside the emergency fund
                emergency_fund_balance_minor=600_000,  # $6,000 saved
                emergency_fund_target_months="6",  # target $18,000
                min_cash_buffer_months="3", debts=(CARD,), retirement_accounts=(ROTH,), horizon_months=12)
    return HouseholdFinancialState(**{**base, **kw})


def facts(**kw) -> dict[str, Any]:
    base = dict(debts=[CARD.model_dump(mode="json")], emergency_fund_balance_minor=600_000,
                emergency_fund_target_months="6", retirement_accounts=[ROTH.model_dump(mode="json")],
                min_cash_buffer_months="3", horizon_months=12,
                monthly_essential_expenses_minor=300_000, liquid_cash_minor=1_000_000)
    return {**base, **kw}


# --- the pure evaluator: hand-worked numbers ---------------------------------------------

def test_interest_saved_is_hand_computed():
    # $1,000 at 12% APR = 1% a month; a $1,100 minimum payment clears it in month 1.
    # Baseline month-1 interest is 1,000c. With $1,000 extra principal the balance is 0, so interest is 0: saved 1,000c.
    s = state(debts=(Debt(debt_id="d", balance_minor=100_000, apr_percent="12", minimum_payment_minor=110_000),))
    ev = evaluate_candidates(s, 100_000, [CandidatePlan(name="pay", debt_principal_minor={"d": 100_000})])
    [o] = ev.candidates
    assert o.interest_saved_minor == 1_000 and o.debts[0].baseline_interest_minor == 1_000
    assert o.debts[0].plan_interest_minor == 0 and o.valid and o.unallocated_minor == 0


def test_buffer_months_and_emergency_fund_are_hand_computed():
    # Whole $2,000 into the emergency fund: fund $8,000 of the $18,000 target = 44.44%; cash $8,000 + fund $8,000
    # = $16,000 reserve = 5.33 months of $3,000.
    o = evaluate_candidates(state(), EXTRA, [CandidatePlan(name="ef", emergency_fund_minor=EXTRA)]).candidates[0]
    assert o.emergency_fund_after_minor == 800_000 and o.emergency_fund_target_minor == 1_800_000
    assert str(o.emergency_fund_pct_of_target) == "44.44" and o.emergency_fund_shortfall_minor == 1_000_000
    assert o.liquid_reserve_after_minor == 1_600_000 and str(o.buffer_months) == "5.33"
    assert o.buffer_meets_minimum and o.valid


def test_retirement_headroom_accounting():
    o = evaluate_candidates(state(), EXTRA, [CandidatePlan(name="r", retirement_minor={"roth": EXTRA})]).candidates[0]
    assert o.retirement_used_minor == EXTRA and o.retirement_headroom_remaining_minor == 500_000 and o.valid


# --- negative paths: violations, not crashes ---------------------------------------------

@pytest.mark.parametrize("plan,code", [
    (CandidatePlan(name="p", hold_minor=EXTRA + 1), "over_allocation"),
    (CandidatePlan(name="p", hold_minor=-5, emergency_fund_minor=EXTRA + 5), "negative_amount"),
    (CandidatePlan(name="p", debt_principal_minor={"card": 500_001}), "debt_overpayment"),
    (CandidatePlan(name="p", debt_principal_minor={"nope": 100}), "unknown_debt"),
    (CandidatePlan(name="p", retirement_minor={"nope": 100}), "unknown_retirement_account"),
    (CandidatePlan(name="p", retirement_minor={"roth": 700_001}), "over_contribution_limit"),
])
def test_each_constraint_is_reported(plan, code):
    o = evaluate_candidates(state(liquid_cash_minor=5_000_000), EXTRA, [plan]).candidates[0]
    assert not o.valid and code in {v.code for v in o.violations}


def test_buffer_below_household_minimum_is_a_violation():
    s = state(min_cash_buffer_months="5.2")  # reserve $16,000 = 5.33 months; spending $2,000 on debt leaves 4.67
    ev = evaluate_candidates(s, EXTRA, [CandidatePlan(name="debt", debt_principal_minor={"card": EXTRA}),
                                        CandidatePlan(name="hold", hold_minor=EXTRA)])
    by = {c.name: c for c in ev.candidates}
    assert "buffer_below_minimum" in {v.code for v in by["debt"].violations} and by["hold"].valid
    assert ev.best_valid == "hold" and ev.ranking[0] == "hold"


def test_insufficient_cash_is_a_violation():
    o = evaluate_candidates(state(liquid_cash_minor=100_000), EXTRA,
                            [CandidatePlan(name="r", retirement_minor={"roth": EXTRA})]).candidates[0]
    assert "insufficient_cash" in {v.code for v in o.violations}


@pytest.mark.parametrize("extra,cands", [(0, [CandidatePlan(name="a")]), (-1, [CandidatePlan(name="a")]),
                                         (10**13, [CandidatePlan(name="a")]), (EXTRA, []),
                                         (EXTRA, [CandidatePlan(name="a"), CandidatePlan(name="a")]),
                                         (EXTRA, [CandidatePlan(name=f"n{i}") for i in range(13)]),
                                         (True, [CandidatePlan(name="a")])])
def test_unusable_requests_are_refused(extra, cands):
    with pytest.raises(AllocationInputError):
        evaluate_candidates(state(), extra, cands)


def test_state_rejects_bad_inputs():
    with pytest.raises(ValueError):
        state(debts=(CARD, CARD))  # duplicate debt id
    with pytest.raises(ValueError):
        state(monthly_essential_expenses_minor=0)
    with pytest.raises(ValueError):
        state(liquid_cash_minor=-1)
    with pytest.raises(ValueError):
        state(horizon_months=0)


def test_no_valid_candidate_is_reported():
    s = state(min_cash_buffer_months="6")  # reserve is 5.33 months at best, so nothing can satisfy 6
    ev = evaluate_candidates(s, EXTRA, standard_candidates(s, EXTRA))
    assert ev.no_valid_candidate and ev.best_valid is None


# --- properties ----------------------------------------------------------------------------

@st.composite
def plans(draw, name="p"):
    n = draw(st.integers(0, 4))
    return CandidatePlan(
        name=name, debt_principal_minor={"card": draw(st.integers(-1000, 600_000))} if n > 0 else {},
        emergency_fund_minor=draw(st.integers(-1000, 400_000)) if n > 1 else 0,
        retirement_minor={"roth": draw(st.integers(-1000, 800_000))} if n > 2 else {},
        hold_minor=draw(st.integers(-1000, 400_000)) if n > 3 else 0)


@settings(max_examples=80, deadline=None)
@given(plans())
def test_conservation_allocated_plus_unallocated_is_extra(plan):
    o = evaluate_candidates(state(liquid_cash_minor=5_000_000), EXTRA, [plan]).candidates[0]
    assert o.allocated_minor + o.unallocated_minor == EXTRA


@settings(max_examples=80, deadline=None)
@given(plans())
def test_valid_iff_no_violations_and_ranking_key_agrees(plan):
    o = evaluate_candidates(state(liquid_cash_minor=5_000_000), EXTRA, [plan]).candidates[0]
    assert o.valid == (not o.violations) and o.score_key[0] == len(o.violations)


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 250_000), st.integers(0, 250_000))
def test_interest_saved_is_monotone_in_extra_principal(a, b):
    lo, hi = sorted((a, b))
    ev = evaluate_candidates(state(liquid_cash_minor=5_000_000), 250_000, [
        CandidatePlan(name="lo", debt_principal_minor={"card": lo}),
        CandidatePlan(name="hi", debt_principal_minor={"card": hi})])
    by = {c.name: c.interest_saved_minor for c in ev.candidates}
    assert by["hi"] >= by["lo"] >= 0


@settings(max_examples=40, deadline=None)
@given(st.lists(plans(), min_size=1, max_size=6), st.randoms())
def test_ranking_is_deterministic_stable_and_order_independent(raw, rnd):
    cands = [p.model_copy(update={"name": f"c{i}"}) for i, p in enumerate(raw)]
    shuffled = list(cands)
    rnd.shuffle(shuffled)
    s = state(liquid_cash_minor=5_000_000)
    a, b = evaluate_candidates(s, EXTRA, cands), evaluate_candidates(s, EXTRA, shuffled)
    assert a.ranking == b.ranking and a.input_hash == b.input_hash and a == evaluate_candidates(s, EXTRA, cands)
    keys = [c.score_key for c in a.candidates]
    assert keys == sorted(keys) and [c.rank for c in a.candidates] == list(range(1, len(cands) + 1))
    assert a.best_valid == next((c.name for c in a.candidates if c.valid), None)


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 150_000), st.integers(0, 50_000), st.integers(0, 50_000))
def test_a_plan_inside_every_limit_has_no_violations(debt, ef, hold):
    plan = CandidatePlan(name="ok", debt_principal_minor={"card": debt}, emergency_fund_minor=ef, hold_minor=hold)
    o = evaluate_candidates(state(liquid_cash_minor=5_000_000), 250_000, [plan]).candidates[0]
    assert o.valid and o.violations == ()


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 10**9), st.lists(st.integers(0, BPS), min_size=1, max_size=8))
def test_expand_shape_cents_add_up_to_extra(extra, shares):
    total = sum(shares)
    if total == 0:
        return
    scaled = [s * BPS // total for s in shares]  # rescale so the shares add up to exactly 10,000 basis points
    scaled[0] += BPS - sum(scaled)
    shape = CandidateShape(name="x", lines=[ShapeLine(bucket="hold", share_bps=s) for s in scaled])
    assert expand_shape(shape, extra).hold_minor == extra


def test_expand_shape_refuses_shares_that_do_not_add_up():
    with pytest.raises(ValueError):
        expand_shape(CandidateShape(name="x", lines=[ShapeLine(bucket="hold", share_bps=9_000)]), EXTRA)


def test_expand_shape_routes_buckets_and_apportions_remainders():
    shape = CandidateShape(name="Mix It", lines=[
        ShapeLine(bucket="debt", target_id="card", share_bps=3_333), ShapeLine(bucket="emergency_fund", share_bps=3_333),
        ShapeLine(bucket="retirement", target_id="roth", share_bps=3_334)])
    p = expand_shape(shape, 1_000)  # 333.3 / 333.3 / 333.4: the largest remainders get the spare cents
    assert p.name == "model_mix_it"
    assert p.debt_principal_minor["card"] + p.emergency_fund_minor + p.retirement_minor["roth"] == 1_000


@pytest.mark.parametrize("text,cents", [("Where should our extra $2,000 go?", 200_000), ("we have $1,234.50 extra", 123_450),
                                        ("allocate $2000 and $2,000.00", 200_000)])
def test_extra_amount_is_read_from_the_request(text, cents):
    assert extra_amount_from_request(text) == cents


@pytest.mark.parametrize("text", ["where should our extra money go?", "move $100 or maybe $200"])
def test_extra_amount_refuses_missing_or_ambiguous(text):
    with pytest.raises(ValueError):
        extra_amount_from_request(text)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(1, 3_000_000))
def test_standard_candidates_never_break_a_balance_or_headroom_limit(extra):
    s = state(liquid_cash_minor=5_000_000)
    ev = evaluate_candidates(s, extra, standard_candidates(s, extra))
    for c in ev.candidates:
        assert not ({"over_allocation", "debt_overpayment", "over_contribution_limit", "negative_amount"}
                    & {v.code for v in c.violations})
        assert c.allocated_minor + c.unallocated_minor == extra
    assert len({c.name for c in ev.candidates}) == len(ev.candidates)


def test_standard_candidates_are_deterministic():
    s = state()
    assert standard_candidates(s, EXTRA) == standard_candidates(s, EXTRA)


# --- the Supervisor-side tool ------------------------------------------------------------

def _call(store, actor=ALEX, **kw):
    args = dict(purpose="capital_allocation", correlation_id="alloc_t", extra_amount_minor=EXTRA,
                candidates=[CandidatePlan(name="ef", emergency_fund_minor=EXTRA)], facts=facts(), at=DEMO_AS_OF)
    return evaluate_allocation_candidates(store, actor, **{**args, **kw})


def _stored(store, run_id):
    return json.loads(store.conn.execute("SELECT output_json FROM calculation_runs WHERE run_id=?", (run_id,)).fetchone()[0])


def test_tool_happy_path_records_a_run_and_audits_it(seeded_store):
    r = _call(seeded_store)
    assert r["ok"] and r["run_id"] == r["evidence_id"] and r["run_id"].startswith("alloc_run_")
    assert r["output"]["extra_amount_display"] == "$2,000.00"
    assert r["output"]["candidates"][0]["interest_saved_display"] == "$0.00"
    fn, h = seeded_store.conn.execute("SELECT fn, output_hash FROM calculation_runs WHERE run_id=?", (r["run_id"],)).fetchone()
    assert (fn, h) == ("evaluate_allocation_candidates", r["output_hash"])
    assert _stored(seeded_store, r["run_id"]) == r["output"]
    ev = [e for e in seeded_store.audit.events("alloc_t") if e["action"] == "calculate"]
    assert ev and ev[-1]["resource"] == f"calc:{r['run_id']}" and seeded_store.audit.verify() > 0


@pytest.mark.parametrize("actor", [SAM, CASEY_CPA])
def test_dependents_and_delegates_are_refused_and_audited(seeded_store, actor):
    before = seeded_store.conn.execute("SELECT COUNT(*) FROM calculation_runs").fetchone()[0]
    r = _call(seeded_store, actor)
    assert not r["ok"] and r["error"]["type"] == "authorization_denied"
    assert seeded_store.conn.execute("SELECT COUNT(*) FROM calculation_runs").fetchone()[0] == before
    deny = [e for e in seeded_store.audit.events("alloc_t") if e["decision"] == "deny"]
    assert deny and deny[-1]["actor_id"] == actor.member_id and deny[-1]["action"] == "evaluate_allocation_candidates"


def test_wrong_purpose_is_refused(seeded_store):
    r = _call(seeded_store, purpose="historical_comparison")
    assert not r["ok"] and r["error"]["type"] == "authorization_denied"


@pytest.mark.parametrize("drop", ["debts", "emergency_fund_balance_minor", "emergency_fund_target_months",
                                  "retirement_accounts", "min_cash_buffer_months"])
def test_missing_inputs_are_named_never_invented(seeded_store, drop):
    f = facts()
    del f[drop]
    r = _call(seeded_store, facts=f)
    assert not r["ok"] and r["error"]["type"] == "missing_input" and drop in r["error"]["message"]


def test_no_facts_at_all_is_refused(seeded_store):
    r = _call(seeded_store, facts=None)
    assert not r["ok"] and r["error"]["type"] == "missing_input"


def test_unusable_request_is_refused_and_audited_not_raised(seeded_store):
    r = _call(seeded_store, extra_amount_minor=0)
    assert not r["ok"] and r["error"]["type"] == "invalid_request"
    assert [e for e in seeded_store.audit.events("alloc_t") if e["decision"] == "deny"]


def test_invalid_candidate_payload_is_an_error_not_an_exception(seeded_store):
    r = _call(seeded_store, candidates=[{"name": "x", "surprise": 1}])
    assert not r["ok"] and r["error"]["type"] in {"invalid_request", "missing_input"}


def test_cash_and_essentials_come_from_the_ledger_when_not_given(seeded_store):
    f = facts()
    del f["monthly_essential_expenses_minor"], f["liquid_cash_minor"]
    r = _call(seeded_store, facts=f)
    if r["ok"]:  # the demo ledger supplies both
        assert r["input_sources"]["liquid_cash_minor"] == "ledger"
        assert r["input_sources"]["monthly_essential_expenses_minor"] == "ledger"
        assert r["input_sources"]["debts"] == "task_input"
    else:  # or the tool says exactly what it could not find
        assert r["error"]["type"] == "missing_input"


def test_facts_reject_unexpected_fields():
    with pytest.raises(ValueError):
        AllocationFacts.model_validate({**facts(), "returns": "0.07"})  # there is no return-forecast input


# --- routing -------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["Where should our extra $2,000 go this month?", "allocate our surplus",
                                  "should we pay down the card or fund the roth with the extra $2,000"])
def test_allocation_questions_route_to_allocation_without_refusal(text):
    d = rule_route(text)
    assert d.intent == "allocation_planning" and not d.external_action_requested


@pytest.mark.parametrize("text", ["allocate $2,000 and transfer it to savings", "allocate the surplus and pay the invoice now",
                                  "where should our extra $2,000 go? wire it to the broker"])
def test_money_movement_inside_an_allocation_question_is_still_refused(text):
    d = rule_route(text)
    assert d.intent == "allocation_planning" and d.external_action_requested


# --- the allocate flow ----------------------------------------------------------------------

@dataclass
class FnModel:
    """Fake model: ``fn(system_prompt, payload) -> reply text``; only ``invoke`` is used."""
    fn: Callable[[str, dict], str]

    def invoke(self, messages):
        return AIMessage(content=self.fn(messages[0].content, json.loads(messages[1].content)))


def _sup(seeded_store, model=None, actor=ALEX, **f):
    return Supervisor(actor, Router(), {}, seeded_store.audit, clock=lambda: DEMO_AS_OF, ledger=seeded_store,
                      allocation_facts=facts(**f), allocation_model=model)


ASK = "Where should our extra $2,000 go this month?"


def test_allocation_question_runs_the_tool_and_every_number_traces_to_the_run(seeded_store):
    out = _sup(seeded_store).run(ASK, correlation_id="j_alloc")
    a = out["briefing"]["allocation"]
    assert out["status"] == "completed" and a["status"] == "recommended" and a["recommended"] == "all_to_emergency_fund"
    assert a["extra_amount_display"] == "$2,000.00" and a["best_valid"] == a["ranked_first"] == a["recommended"]
    assert "not financial advice" in a["framing"] and any("return" in x for x in a["assumptions"])
    stored = _stored(seeded_store, a["run_id"])
    assert stored["ranking"][0] == "all_to_emergency_fund"
    [s] = out["briefing"]["statements"]
    assert s["evidence_ids"] == [a["run_id"]] and "all_to_emergency_fund" in s["text"]
    assert seeded_store.audit.verify() > 0


def test_a_checkpointer_does_not_pause_an_allocation_answer(seeded_store):
    out = _sup(seeded_store).run(ASK, correlation_id="j_alloc_cp", checkpointer=InMemorySaver())
    assert out["status"] == "completed" and "__interrupt__" not in out


def test_without_facts_the_older_planner_path_is_used(seeded_store):
    sup = Supervisor(ALEX, Router(), {}, seeded_store.audit, clock=lambda: DEMO_AS_OF, ledger=seeded_store)
    out = sup.run(ASK, correlation_id="j_old")
    assert out["status"] == "failed" and "allocation" not in out


def test_a_dependent_is_refused_before_anything_is_evaluated(seeded_store):
    before = seeded_store.conn.execute("SELECT COUNT(*) FROM calculation_runs").fetchone()[0]
    out = _sup(seeded_store, actor=SAM).run(ASK, correlation_id="j_sam")
    assert out["status"] == "refused" and out["allocation"]["status"] == "denied"
    assert seeded_store.conn.execute("SELECT COUNT(*) FROM calculation_runs").fetchone()[0] == before


def test_a_transfer_request_is_refused_as_an_external_action(seeded_store):
    out = _sup(seeded_store).run("allocate our extra $2,000 and transfer it to savings", correlation_id="j_x")
    assert out["status"] == "refused" and not out.get("allocation")


def test_no_amount_in_the_question_asks_instead_of_guessing(seeded_store):
    out = _sup(seeded_store).run("where should our extra money go?", correlation_id="j_noamt")
    assert out["status"] == "needs_review" and out["allocation"]["status"] == "missing_inputs"
    assert "dollar amount" in out["briefing"]["questions"][0] and not out["briefing"]["statements"]


def test_missing_household_facts_are_asked_for_not_invented(seeded_store):
    f = facts()
    del f["debts"]
    sup = Supervisor(ALEX, Router(), {}, seeded_store.audit, clock=lambda: DEMO_AS_OF, ledger=seeded_store,
                     allocation_facts=f)
    out = sup.run(ASK, correlation_id="j_nofacts")
    assert out["allocation"]["status"] == "missing_inputs" and "debts" in out["briefing"]["questions"][0]


def test_when_no_candidate_is_valid_nothing_is_recommended(seeded_store):
    out = _sup(seeded_store, min_cash_buffer_months="6").run(ASK, correlation_id="j_none")
    a = out["briefing"]["allocation"]
    assert out["status"] == "needs_review" and a["recommended"] is None and a["best_valid"] is None
    assert out["briefing"]["needs_review"] and not out["briefing"]["statements"]


# --- a misbehaving model ---------------------------------------------------------------------

def _model(propose=None, choose=None):
    def fn(system, payload):
        if system == PROPOSE_PROMPT:
            return propose if propose is not None else json.dumps({"candidates": []})
        assert system == CHOOSE_PROMPT
        if callable(choose):
            return choose(payload)
        return choose or json.dumps({"chosen": None, "claims": []})
    return FnModel(fn)


def test_a_model_choice_with_violations_is_flagged_and_replaced(seeded_store):
    pick_debt = _model(choose=lambda p: json.dumps({"chosen": "all_to_highest_apr_debt", "claims": []}))
    out = _sup(seeded_store, pick_debt, min_cash_buffer_months="5.2").run(ASK, correlation_id="j_bad_pick")
    a = out["briefing"]["allocation"]
    assert a["model_choice"] == "all_to_highest_apr_debt" and a["recommended"] == "all_to_emergency_fund"
    assert out["status"] == "needs_review" and any("violates constraints" in f for f in a["flags"])


def test_a_valid_model_choice_is_honoured(seeded_store):
    pick = _model(choose=lambda p: json.dumps({"chosen": "all_to_cash_hold", "claims": []}))
    a = _sup(seeded_store, pick).run(ASK, correlation_id="j_pick")["briefing"]["allocation"]
    assert a["recommended"] == "all_to_cash_hold" and a["best_valid"] == "all_to_emergency_fund"
    assert any("differs from the top-ranked" in f for f in a["flags"])


def test_a_model_choice_that_was_never_evaluated_is_flagged(seeded_store):
    out = _sup(seeded_store, _model(choose=lambda p: json.dumps({"chosen": "bitcoin", "claims": []}))).run(
        ASK, correlation_id="j_ghost")
    a = out["briefing"]["allocation"]
    assert a["recommended"] == "all_to_emergency_fund" and out["status"] == "needs_review"
    assert any("not evaluated" in f for f in a["flags"])


def test_a_claim_with_an_invented_number_is_removed(seeded_store):
    def choose(p):
        return json.dumps({"chosen": "all_to_emergency_fund", "claims": [
            {"text": "This saves $987.65 in interest.", "evidence_ids": [p["evidence_id"]]},
            {"text": "The emergency fund reaches 44.44% of its target.", "evidence_ids": [p["evidence_id"]]}]})
    out = _sup(seeded_store, _model(choose=choose)).run(ASK, correlation_id="j_claims")
    texts = [s["text"] for s in out["briefing"]["statements"]]
    assert "This saves $987.65 in interest." not in texts and "The emergency fund reaches 44.44% of its target." in texts
    assert out["status"] == "needs_review" and any("claim removed" in f for f in out["briefing"]["allocation"]["flags"])


def test_a_claim_citing_an_unknown_run_is_removed(seeded_store):
    choose = json.dumps({"chosen": None, "claims": [{"text": "It saves $1.00.", "evidence_ids": ["alloc_run_fake"]}]})
    out = _sup(seeded_store, _model(choose=choose)).run(ASK, correlation_id="j_fakeid")
    assert all("$1.00" not in s["text"] for s in out["briefing"]["statements"]) and out["status"] == "needs_review"


def test_an_unparseable_model_answer_falls_back_to_the_ranking(seeded_store):
    out = _sup(seeded_store, _model(propose="not json", choose="still not json")).run(ASK, correlation_id="j_junk")
    a = out["briefing"]["allocation"]
    assert a["recommended"] == "all_to_emergency_fund" and out["status"] == "needs_review"
    assert any("unparseable" in f for f in a["flags"]) and any("proposal ignored" in f for f in a["flags"])


def test_model_proposed_shapes_are_evaluated_in_cents_by_code(seeded_store):
    shapes = {"candidates": [{"name": "Half and half", "lines": [
        {"bucket": "emergency_fund", "share_bps": 5_000}, {"bucket": "debt", "target_id": "card", "share_bps": 5_000}]},
        {"name": "Broken", "lines": [{"bucket": "hold", "share_bps": 9_000}]}]}
    out = _sup(seeded_store, _model(propose=json.dumps(shapes))).run(ASK, correlation_id="j_shape")
    a = out["briefing"]["allocation"]
    names = [c["name"] for c in a["candidates"]]
    assert "model_half_and_half" in names and "model_broken" not in names
    assert any("dropped" in f for f in a["flags"])
    row = next(c for c in _stored(seeded_store, a["run_id"])["candidates"] if c["name"] == "model_half_and_half")
    assert row["allocated_minor"] == EXTRA and row["unallocated_minor"] == 0 and row["valid"]


def test_a_model_cannot_duplicate_a_candidate_name(seeded_store):
    line = [{"bucket": "hold", "share_bps": 10_000}]
    shapes = {"candidates": [{"name": "x", "lines": line}, {"name": "x", "lines": line}]}
    out = _sup(seeded_store, _model(propose=json.dumps(shapes))).run(ASK, correlation_id="j_dup")
    names = [c["name"] for c in out["briefing"]["allocation"]["candidates"]]
    assert names.count("model_x") == 1
