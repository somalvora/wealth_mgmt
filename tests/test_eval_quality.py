import random
from dataclasses import dataclass, replace
from datetime import date, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

import household_finance as hf
from household_agent import eval_quality as quality
from household_agent.authz import Authorizer, Decision
from household_agent.calc_runner import REGISTRY
from household_agent.evals import evaluate, generate_scenarios
from household_agent.retrieval import PUBLIC, Claim, RetrievalService, validate_claims
from household_agent.synthetic import ACCOUNTS

SCENARIOS = generate_scenarios(0)


def _only(area):
    return [s for s in SCENARIOS if s.area == area]


def _gate(report, name):
    return next(g for g in report.gates if g.name == name)


@dataclass(frozen=True)
class E:
    posted_at: date
    amount_minor: int
    merchant_normalized: str
    category: str = "uncategorized"


D = date(2026, 10, 1)


def _day(n):
    return D + timedelta(days=n)


# --- happy path: cluster rules ---------------------------------------------------------

def test_small_charges_over_300_in_seven_days_are_one_cluster():
    current = [E(_day(k), -7_600, f"New {k}") for k in range(4)]  # $304 over 4 days
    [alert] = hf.cluster_risk(current)
    assert alert.rule == "small_cluster_7d" and alert.total_minor == 30_400 and alert.count == 4
    assert alert.entry_indexes == (0, 1, 2, 3) and "$304.00" in alert.explanation


def test_ten_small_charges_on_one_day_are_a_burst():
    alerts = hf.cluster_risk([E(D, -100 - k, f"Shop {k}") for k in range(10)])
    assert [a.rule for a in alerts] == ["burst_24h"] and alerts[0].count == 10


def test_probes_then_a_larger_charge_from_one_merchant_is_card_testing():
    current = [E(_day(0), -199, "Quickpay"), E(_day(1), -250, "Quickpay"), E(_day(2), -101, "Quickpay"),
               E(_day(5), -42_000, "Quickpay")]
    [alert] = hf.cluster_risk(current)
    assert alert.rule == "small_then_large" and alert.merchants == ("Quickpay",) and alert.count == 4


def test_runner_splits_baseline_and_reports_transaction_ids():
    from household_agent.models import Transaction

    def txn(i, day, merchant, amount):
        return Transaction.model_construct(transaction_id=f"t{i}", posted_at=day, amount_minor=amount,
                                           merchant_normalized=merchant, category="uncategorized")

    known = txn(0, date(2026, 9, 3), "Old", -100)  # before ``start``: baseline, not current
    current = [txn(1, _day(0), "Old", -8_000), *(txn(i, _day(i), f"New {i}", -8_000) for i in range(2, 6))]
    [alert] = REGISTRY["cluster_risk"]([known, *current], {"start": "2026-10-01", "end": "2026-10-31"})
    assert alert["rule"] == "small_cluster_7d" and alert["transaction_ids"] == ["t2", "t3", "t4", "t5"]
    assert alert["total_minor"] == 32_000


# --- happy path: eval areas ------------------------------------------------------------

def test_quality_areas_pass_with_the_offline_components():
    report = evaluate([s for s in SCENARIOS if s.area in ("calculation", "anomaly", "retrieval", "privacy")])
    assert report.passed, report.failures
    m = report.metrics
    assert m["calculation"] == m["anomaly_recall"] == m["citation_precision"] == m["citation_support_rate"] == 1.0
    assert m["anomaly_false_alerts"] == 0 and m["privacy_canary_exposures"] == 0 and m["privacy_positive_controls"] == 1
    assert m["retrieval_recall_at_5"] >= 0.9 and m["cross_member_leaks"] == 0


def test_cluster_rules_are_what_close_the_anomaly_gap():
    report = evaluate(_only("anomaly"))
    # per-transaction limits alone catch only the two single-event types out of five
    assert report.metrics["anomaly_recall"] == 1.0 and report.metrics["anomaly_recall_per_transaction_rules"] == 0.4


def test_calculation_oracle_is_independent_of_the_finance_library():
    for s in _only("calculation"):
        cats = s.expected["categories"]
        assert sum(cats.values()) == s.expected["net"] == s.expected["closing"] - s.expected["opening"]
        assert set(cats) <= set(quality.CALC_MERCHANTS.values())


def test_word_overlap_keeps_a_paraphrase_but_drops_an_unrelated_source():
    passages = [{"chunk_id": "a", "text": "The HOA maintains the common landscaping and the pool.",
                 "verification": "verified", "authority_tier": 2, "metadata": {"effective_tax_year": 0}},
                {"chunk_id": "b", "text": "Monthly rent for Unit B is 3,400.00 dollars.",
                 "verification": "verified", "authority_tier": 2, "metadata": {"effective_tax_year": 0}}]
    claim = "The HOA maintains the pool and landscaping."
    ok, wrong = validate_claims([Claim(claim, "fact", ("a",)), Claim(claim, "fact", ("b",))], passages)
    assert ok["verdict"] == "supported"
    assert wrong["verdict"] == "rephrase_as_question" and "do not say" in wrong["reason"]


# --- negative paths --------------------------------------------------------------------

def test_without_cluster_rules_anomaly_recall_fails(monkeypatch):
    monkeypatch.setitem(REGISTRY, "cluster_risk", lambda txns, params: [])
    report = evaluate(_only("anomaly"))
    assert not _gate(report, "anomaly_recall").passed and report.metrics["anomaly_recall"] == 0.4
    assert {f["detail"].split(":")[0] for f in report.failures} == {"small_cluster", "burst", "small_then_large"}


def test_a_wrong_oracle_value_fails_calculation_match():
    s = _only("calculation")[0]
    bad = replace(s, expected={**s.expected, "net": s.expected["net"] + 1})
    report = evaluate([bad])
    assert not _gate(report, "calculation_match").passed and "cash flow" in report.failures[0]["detail"]


def test_a_validator_that_trusts_every_citation_fails_citation_precision(monkeypatch):
    monkeypatch.setattr(quality, "validate_claims", lambda claims, passages: [{"verdict": "supported"}] * len(claims))
    report = evaluate(_only("retrieval"))
    assert not _gate(report, "citation_precision").passed and report.metrics["citation_precision"] < 0.6


def test_retrieval_without_scope_filtering_leaks_and_exposes_canaries(monkeypatch):
    monkeypatch.setattr(RetrievalService, "allowed_scopes",
                        lambda self, actor, purpose, at: [*(a.account_id for a in ACCOUNTS), PUBLIC])
    report = evaluate(_only("retrieval") + _only("privacy"))
    assert not _gate(report, "cross_member_leaks").passed and not _gate(report, "privacy_canary_exposures").passed
    surfaces = {f["detail"] for f in report.failures if f["metric"] == "privacy_canary_exposures"}
    assert any("retrieval" in d for d in surfaces) and any("supervisor" in d for d in surfaces)


def test_an_allow_all_authorizer_exposes_canaries_in_briefings_and_tools(monkeypatch):
    monkeypatch.setattr(Authorizer, "authorize", lambda self, *a, **k: Decision(True, "allow_all"))
    report = evaluate(_only("privacy"))
    assert not _gate(report, "privacy_canary_exposures").passed
    assert any("briefing" in f["detail"] and "tools" in f["detail"] for f in report.failures)


def test_baseline_merchants_and_categorized_or_credit_rows_never_alert():
    current = [E(D, -9_000, f"Known {k}") for k in range(12)]
    assert hf.cluster_risk(current, baseline=[E(D - timedelta(days=30), -500, f"Known {k}") for k in range(12)]) == []
    assert hf.cluster_risk([replace(e, category="groceries") for e in current]) == []
    assert hf.cluster_risk([replace(e, amount_minor=9_000) for e in current]) == []


def test_limits_are_strict():
    assert hf.cluster_risk([E(_day(k), -7_500, f"N {k}") for k in range(4)]) == []  # exactly $300
    assert hf.cluster_risk([E(D, -100 - k, f"S {k}") for k in range(9)]) == []  # 9 in a day
    probes = [E(_day(k), -500, "Q") for k in range(2)]
    assert hf.cluster_risk([*probes, E(_day(3), -20_000, "Q")]) == []  # only 2 probes
    late = [E(_day(k), -500, "Q") for k in range(3)] + [E(_day(10), -20_000, "Q")]
    assert hf.cluster_risk(late) == []  # large charge 8 days after the last probe


# --- properties --------------------------------------------------------------------------

_entries = st.lists(st.tuples(st.integers(0, 27), st.integers(1, 30_000), st.integers(0, 6)), max_size=40)


@settings(max_examples=60, deadline=None)
@given(_entries)
def test_every_alert_meets_its_rule_and_a_full_baseline_silences_all(rows):
    current = [E(_day(d), -a, f"M{m}") for d, a, m in rows]
    for alert in hf.cluster_risk(current):
        picked = [current[i] for i in alert.entry_indexes]
        assert alert.total_minor == -sum(e.amount_minor for e in picked) and alert.count == len(picked)
        if alert.rule == "small_cluster_7d":
            assert alert.total_minor > 30_000 and (alert.end - alert.start).days < 7
            assert all(-e.amount_minor < 10_000 for e in picked)
        elif alert.rule == "burst_24h":
            assert alert.count >= 10 and alert.start == alert.end
        else:
            assert alert.rule == "small_then_large" and len(alert.merchants) == 1
    assert hf.cluster_risk(current, baseline=current) == []


@settings(max_examples=5, deadline=None)
@given(st.integers(0, 10_000))
def test_any_seed_scores_numbers_and_anomalies_perfectly(seed):
    rng = random.Random(seed)
    for p, e in quality.calculation_cases(rng, 4):
        assert quality.calculation_problems(p, e, "prop") == []
    for p, e in quality.anomaly_cases(rng, 8):
        alerts = quality.anomaly_alerts(p, "prop")
        assert (e["alert"] in alerts) if e["alert"] else not alerts
