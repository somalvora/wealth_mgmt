"""The brokerage eval areas: they pass on the real calculations and fail when a calculation, a rounding rule, a limit or the
statement-chaining rule is wrong (an eval that cannot fail proves nothing)."""

from fractions import Fraction

import pytest

from household_agent import brokerage, tools
from household_agent import eval_brokerage as eb
from household_agent.evals import evaluate, generate_scenarios

SCENARIOS = generate_scenarios(0)


def _only(area):
    return [s for s in SCENARIOS if s.area == area]


def _gate(report, name):
    return next(g for g in report.gates if g.name == name)


def test_the_areas_cover_random_boundary_tie_and_demo_cases():
    perf, conc = _only("brokerage_performance"), _only("brokerage_concentration")
    assert len(perf) == eb.N_RANDOM + 5 and len(conc) == eb.N_RANDOM + 15
    kinds = {s.payload["kind"] for s in perf}
    assert {"random_chain", "random_chain_with_gap", "rounding_tie_up", "rounding_tie_down", "demo_months"} == kinds
    kinds = {s.payload["kind"] for s in conc}
    assert {f"boundary_{w}_{e}" for w in ("stock", "tech") for e in ("under", "at", "over")} <= kinds
    assert {"random", "demo_months", "fund_over_limit"} <= kinds
    demo = next(s for s in perf if s.payload["kind"] == "demo_months")
    assert set(demo.payload["rule_deposits"].values()) == {150_000} and demo.payload["ledger_deposits"]  # August is held


def test_the_boundary_positions_sit_exactly_on_a_cent_under_and_over_the_limit():
    single, sector = eb.policy_limits()
    for s in _only("brokerage_concentration"):
        kind = s.payload["kind"]
        if not kind.startswith("boundary_"):
            continue
        want = s.expected["months"][0]
        total = want["total_minor"]
        share = (Fraction(want["stock_values"]["LMTX"], total) if "stock" in kind
                 else Fraction(want["technology_value_minor"], total))
        limit = single if "stock" in kind else sector
        assert {"under": share < limit, "at": share == limit, "over": share > limit}[kind.rsplit("_", 1)[1]], kind


def test_both_areas_pass_with_the_real_calculations():
    report = evaluate(_only("brokerage_performance") + _only("brokerage_concentration"))
    assert report.passed and not report.failures
    assert _gate(report, "brokerage_performance_match").value == 1.0
    assert _gate(report, "brokerage_concentration_match").value == 1.0


def test_the_oracle_rounds_a_tie_away_from_zero_and_agrees_with_the_demo_story():
    assert eb._tenth_pct(Fraction(5, 2000)) == "0.3" and eb._tenth_pct(Fraction(-5, 2000)) == "-0.3"
    assert eb._tenth_pct(Fraction(-181990, 3133423)) == "-5.8"
    demo = next(s for s in _only("brokerage_concentration") if s.payload["kind"] == "demo_months")
    october = demo.expected["months"][-1]
    assert (october["largest_symbol"], october["largest_share_pct"], october["technology_share_pct"]) == ("NVDA", "23.1", "46.2")


# --- negative paths ---------------------------------------------------------------------------------------------

def _failed(report, gate):
    assert not _gate(report, gate).passed
    return " ".join(f["detail"] for f in report.failures)


def test_a_performance_that_ignores_dividends_fails_the_gate(monkeypatch):
    real = brokerage.performance

    def forgets_dividends(m):
        out = real(m)
        out["market_effect_minor"] += m.dividends_minor
        return out

    monkeypatch.setitem(tools.BROKERAGE_CALCS, "brokerage_performance", forgets_dividends)
    detail = _failed(evaluate(_only("brokerage_performance")), "brokerage_performance_match")
    assert "market_effect_minor" in detail


def test_banker_rounding_fails_on_the_tie_months(monkeypatch):
    real = brokerage.performance

    def half_even(m):
        out = real(m)
        out["market_effect_pct"] = f"{Fraction(out['market_effect_minor'] * 100, m.opening_minor):.1f}"
        return out

    monkeypatch.setitem(tools.BROKERAGE_CALCS, "brokerage_performance", half_even)
    report = evaluate(_only("brokerage_performance"))
    detail = _failed(report, "brokerage_performance_match")
    assert "rounding_tie" in detail and "market_effect_pct" in detail


def test_deposits_that_differ_from_the_checking_transfers_fail_the_demo_case(monkeypatch):
    real = brokerage.demo_brokerage_months
    monkeypatch.setattr(eb, "demo_brokerage_months", lambda: [m.__class__(**{**m.__dict__, "deposits_minor": 100_000})
                                                              for m in real()])
    cases, _ = eb.brokerage_cases(0)
    demo = next(c for c in cases if c[0]["kind"] == "demo_months")
    problems = eb.performance_problems(demo[0], demo[1], "t")
    assert any("the 150000 the checking history transfers" in p for p in problems)
    assert any("in the ledger rows" in p for p in problems)


def test_a_gap_between_months_that_the_ledger_accepts_fails_the_gate(monkeypatch):
    monkeypatch.setattr("household_agent.ledger.LedgerStore.ingest_brokerage_month",
                        lambda self, actor, purpose, corr, month, at=None: None)  # stores nothing, refuses nothing
    report = evaluate(_only("brokerage_performance"))
    assert not _gate(report, "brokerage_performance_match").passed


def test_a_flag_that_fires_at_the_limit_fails_the_boundary_cases(monkeypatch):
    real = brokerage.concentration

    def greater_or_equal(m):
        out = real(m)
        total = m.closing_minor
        out["stocks_over_single_stock_limit"] = [
            h.symbol for h in m.holdings
            if h.kind == "stock" and h.value_minor * 10_000 >= brokerage.LIMITS_BPS["single_stock"] * total]
        return out

    monkeypatch.setitem(tools.BROKERAGE_CALCS, "brokerage_concentration", greater_or_equal)
    report = evaluate(_only("brokerage_concentration"))
    detail = _failed(report, "brokerage_concentration_match")
    assert "boundary_stock_at" in detail and "boundary_stock_under" not in detail


def test_a_fund_flagged_as_a_single_stock_fails(monkeypatch):
    real = brokerage.concentration

    def counts_funds(m):
        out = real(m)
        out["stocks_over_single_stock_limit"] = [
            h.symbol for h in m.holdings if h.value_minor * 10_000 > brokerage.LIMITS_BPS["single_stock"] * m.closing_minor]
        return out

    monkeypatch.setitem(tools.BROKERAGE_CALCS, "brokerage_concentration", counts_funds)
    assert "fund_over_limit" in _failed(evaluate(_only("brokerage_concentration")), "brokerage_concentration_match")


def test_a_policy_text_that_states_other_limits_than_the_calculation_flags_fails(monkeypatch):
    """The calculation still flags at 20 and 50; the written policy now says 25 and 55."""
    monkeypatch.setattr(eb, "policy_limits", lambda: (Fraction(25, 100), Fraction(55, 100)))
    detail = _failed(evaluate(_only("brokerage_concentration")), "brokerage_concentration_match")
    assert "limits_pct" in detail


def test_a_calculation_and_policy_changed_together_still_pass(monkeypatch):
    """One source of truth: change LIMITS_BPS and the generated policy text moves with it, so the eval follows."""
    monkeypatch.setitem(brokerage.LIMITS_BPS, "single_stock", 2500)
    monkeypatch.setitem(brokerage.LIMITS_BPS, "technology_sector", 5500)
    scenarios = generate_scenarios(0)
    report = evaluate([s for s in scenarios if s.area == "brokerage_concentration"])
    assert report.passed, report.failures[:2]


def test_an_unparseable_policy_is_reported_not_guessed(monkeypatch):
    monkeypatch.setattr(brokerage, "policy_text", lambda: "Keep things balanced.")
    with pytest.raises(ValueError, match="does not state both limits"):
        eb.policy_limits()
