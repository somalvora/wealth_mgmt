"""The eval aborts when a hosted model is mostly falling back, instead of grinding through a meaningless run."""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from household_agent.agents import rule_route
from household_agent.evals import EARLY_STOP, ModelUnhealthy, evaluate, generate_scenarios, main

ROUTING = [s for s in generate_scenarios(0) if s.area == "routing"]


def _router(bad_every: int):
    """Fake model router: every `bad_every`-th call falls back (0 = never)."""
    n = {"i": 0}

    def route(request):
        n["i"] += 1
        if bad_every and n["i"] % bad_every == 0:
            return rule_route(request), "rules_fallback"
        return rule_route(request), "model"
    return route


def test_healthy_model_is_not_aborted():
    report = evaluate(ROUTING[:30], router=_router(0), fallback_guard=EARLY_STOP)
    assert report.metrics["router_fallbacks"] == 0


def test_dead_model_aborts_after_the_window_with_counts_only():
    with pytest.raises(ModelUnhealthy) as e:
        evaluate(ROUTING[:40], router=_router(1), fallback_guard=(10, 0.8))
    assert "10 of 10" in str(e.value) and "router" in str(e.value)


def test_no_guard_runs_everything():
    report = evaluate(ROUTING[:40], router=_router(1))
    assert report.metrics["router_fallbacks"] == 40


def test_below_the_window_is_never_judged():
    report = evaluate(ROUTING[:9], router=_router(1), fallback_guard=(10, 0.8))
    assert report.counts["routing"] == 9


def test_cli_exits_2_with_a_clear_message_and_flag_disables(monkeypatch, capsys):
    class Dead:
        def invoke(self, messages, *a, **k):
            if messages == "Reply with the single word: ok":
                return None  # preflight passes, then the model dies
            raise RuntimeError("quota secret-detail")

    monkeypatch.setattr("household_agent.llm.get_chat_model", lambda spec: Dead())
    assert main(["--model", "x:y", "--allocation-scenarios", "0"]) == 2
    err = capsys.readouterr().err
    assert "aborted early" in err and "--no-early-stop" in err and "secret-detail" not in err
    assert main(["--model", "x:y", "--allocation-scenarios", "0", "--no-early-stop"]) in (0, 1)


@settings(max_examples=25, deadline=None)
@given(bad_every=st.integers(1, 6), window=st.integers(1, 12), share=st.sampled_from([0.3, 0.5, 0.8]))
def test_property_aborts_iff_share_exceeds_limit_after_window(bad_every, window, share):
    route, n = _router(bad_every), 20
    expect = any(k >= window and (k // bad_every) / k > share for k in range(1, n + 1))
    try:
        evaluate(ROUTING[:n], router=route, fallback_guard=(window, share))
        aborted = False
    except ModelUnhealthy:
        aborted = True
    assert aborted == expect


def test_dead_planner_critic_aborts_after_an_allocation_scenario():
    from household_agent.llm_planner import LLMCritic

    class Dead:
        def invoke(self, *a, **k):
            raise RuntimeError("down")

    critic = LLMCritic(Dead(), model_id="fake")
    alloc = [s for s in generate_scenarios(0) if s.area == "allocation"][:3]
    with pytest.raises(ModelUnhealthy) as e:
        evaluate(alloc, critic=critic, fallback_guard=(1, 0.5))
    assert "planner critic" in str(e.value) and critic.calls > 0
