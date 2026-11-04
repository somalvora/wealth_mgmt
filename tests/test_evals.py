import json
from collections import Counter
from dataclasses import replace

from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from household_agent.agents import DispatchItem, Router, RoutingDecision
from household_agent.authz import Authorizer, Decision
from household_agent.evals import NOT_COVERED, evaluate, generate_scenarios, main
from household_agent.llm_planner import LLMThoughtGenerator
from household_agent.planning import _TEMPLATES, PERSONAS
from household_agent.synthetic import ACCOUNTS

SCENARIOS = generate_scenarios(0)


class Canned(BaseChatModel):
    """Replies with ``reply(last_message_text)``."""

    reply: object

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = self.reply(str(messages[-1].content))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    @property
    def _llm_type(self) -> str:
        return "canned"


def _oracle():
    """A router that returns exactly the labeled answer for every scenario."""
    answers = {}
    for s in SCENARIOS:
        if s.area == "routing":
            e = s.expected
            answers[s.payload["request"]] = RoutingDecision(
                intent=e["intent"], dispatch_plan=[DispatchItem(agent=a, task="t") for a in e["agents"]],
                approval_required=e["approval_required"])
        elif s.area == "adversarial":
            answers[s.payload["request"]] = RoutingDecision(intent="out_of_scope", external_action_requested=True)
    return lambda request: (answers[request], "model")


def _gate(report, name):
    return next(g for g in report.gates if g.name == name)


def _only(area, scenarios=SCENARIOS):
    return [s for s in scenarios if s.area == area]


# --- happy path ------------------------------------------------------------------------

def test_scenario_set_is_large_covers_every_area_and_is_deterministic():
    counts = Counter(s.area for s in SCENARIOS)
    assert len(SCENARIOS) >= 300 and set(counts) == {"routing", "adversarial", "authorization", "allocation",
                                                      "ingestion", "calculation", "anomaly", "retrieval", "privacy"}
    assert len({s.scenario_id for s in SCENARIOS}) == len(SCENARIOS)
    assert generate_scenarios(0) == SCENARIOS and generate_scenarios(1) != SCENARIOS
    allocation = _only("allocation")
    assert {s.expected["eligible"] for s in allocation} == {True, False}
    assert {s.expected["status"] for s in _only("ingestion")} == {"reconciled", "unreconciled"}
    assert sum(s.expected["allowed"] for s in _only("authorization")) > 0


def test_oracle_router_passes_every_gate():
    report = evaluate(SCENARIOS, router=_oracle())
    # the one remaining failure is a retrieval miss by the offline hash embeddings, not routing
    assert report.passed and [f["metric"] for f in report.failures] == ["retrieval_recall_at_5"]
    assert report.metrics["routing"] == 1.0 and report.metrics["schema_validity"] == 1.0
    assert report.not_covered == NOT_COVERED


def test_keyword_baseline_is_safe_and_exposes_its_false_positive():
    report = evaluate(SCENARIOS)
    assert report.passed
    assert report.metrics["unsafe_actions"] == 0 and report.metrics["cross_member_leaks"] == 0
    assert report.metrics["allocation"] == report.metrics["ingestion"] == 1.0
    assert 0 < report.metrics["max_node_evaluations"] <= 27
    # "Did we pay more ...?" is a question, but the keyword router treats it as a payment request
    assert [f["scenario_id"] for f in report.failures] == ["route_054", "retr_005"]
    assert 0.95 <= report.metrics["routing"] < 1.0


def test_cli_writes_the_json_report(tmp_path, capsys):
    out = tmp_path / "report.json"
    assert main(["--seed", "0", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["passed"] is True and len(data["gates"]) == 13 and data["counts"]["routing"] == 55
    assert "PASS  unsafe_actions" in capsys.readouterr().out


def test_model_persona_generator_keeps_allocation_invariants():
    def thoughts(text):
        tier = json.loads(text)["tier"]
        return json.dumps({"thoughts": [
            {"persona": p, "tier": s.tier, "fraction": str(s.fraction), "buckets": [{"bucket": b, "weight": str(w)}
             for b, w in s.buckets], "mode": s.mode, "rationale": r}
            for p in PERSONAS for s, r in [_TEMPLATES[tier][p]]]})
    report = evaluate(_only("allocation")[:10], generator=LLMThoughtGenerator(Canned(reply=thoughts)))
    assert report.passed and report.metrics["allocation"] == 1.0


# --- negative paths --------------------------------------------------------------------

def test_router_that_never_flags_payments_fails_the_unsafe_gate():
    report = evaluate(_only("adversarial"), router=lambda r: (RoutingDecision(intent="out_of_scope"), "model"))
    assert not _gate(report, "unsafe_actions").passed and report.metrics["unsafe_actions"] == 14


def test_model_router_bypass_is_blocked_by_the_keyword_guard():
    # a model that says "no external action" is overridden when the request names a payment
    lie = RoutingDecision(intent="out_of_scope").model_dump_json()
    report = evaluate(_only("adversarial"), router=Router(Canned(reply=lambda _: lie)))
    assert _gate(report, "unsafe_actions").passed and report.metrics["schema_validity"] == 1.0


def test_unparseable_model_router_fails_schema_validity():
    report = evaluate(_only("routing"), router=Router(Canned(reply=lambda _: "Sure! Route to the ledger.")))
    assert not _gate(report, "schema_validity").passed and report.metrics["schema_validity"] == 0.0
    assert _gate(report, "routing_accuracy").passed  # the keyword fallback still answers


def test_misrouted_allocation_requests_fail_approval_gating():
    report = evaluate(_only("routing"), router=lambda r: (RoutingDecision(intent="out_of_scope"), "model"))
    assert not _gate(report, "approval_gating").passed and not _gate(report, "routing_accuracy").passed
    assert any(f["metric"] == "approval_gating" for f in report.failures)


def test_allow_all_authorizer_is_reported_as_leaks():
    class AllowAll(Authorizer):
        def authorize(self, *args, **kwargs):
            return Decision(True, "allow_all")

    report = evaluate(_only("authorization"), authorizer_factory=lambda: AllowAll(ACCOUNTS))
    assert not _gate(report, "cross_member_leaks").passed and report.metrics["cross_member_leaks"] > 0


def test_wrong_label_is_caught_as_a_failure():
    s = _only("ingestion")[0]
    flipped = replace(s, expected={"status": "reconciled" if s.expected["status"] != "reconciled" else "unreconciled"})
    report = evaluate([flipped])
    assert not _gate(report, "statement_integrity").passed and report.failures[0]["scenario_id"] == s.scenario_id


def test_empty_scenario_set_does_not_divide_by_zero():
    assert evaluate([]).passed


# --- properties --------------------------------------------------------------------------

@settings(max_examples=6, deadline=None)
@given(st.integers(0, 10_000))
def test_any_seed_keeps_the_safety_and_integrity_gates(seed):
    sizes = dict(n_allocation=15, n_ingestion=9, n_calculation=6, n_anomaly=16)
    scenarios = generate_scenarios(seed, **sizes)
    assert scenarios == generate_scenarios(seed, **sizes)
    report = evaluate([s for s in scenarios if s.area != "routing"])
    assert report.passed, report.failures
