"""The eval CLI checks each model once before a long run and reports how much of a run was really fallbacks."""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.language_models import FakeListChatModel

from household_agent import evals, llm
from household_agent.evals import evaluate, fallback_summary, generate_scenarios, main, preflight
from household_agent.llm_planner import LLMCritic, LLMThoughtGenerator
from tests.test_model_call_errors import Broken


def test_preflight_passes_when_every_call_works():
    assert preflight({"a": lambda: 1, "b": lambda: 2}) == []


def test_preflight_names_each_failing_model_by_error_type_only():
    def retired():
        raise RuntimeError("key=AIza-SECRET-123 model gone")

    problems = preflight({"--model gemini:old": retired, "--embed-model ok": lambda: 1})
    assert problems == ["--model gemini:old: RuntimeError"] and "SECRET" not in repr(problems)


def test_cli_stops_before_evaluating_when_a_model_is_dead(monkeypatch, capsys):
    monkeypatch.setattr(llm, "get_chat_model", lambda spec, **kw: Broken(exc=RuntimeError("404 gone")))
    called = []
    monkeypatch.setattr(evals, "evaluate", lambda *a, **k: called.append(1))
    assert main(["--model", "gemini:retired"]) == 2
    assert not called and "preflight failed" in capsys.readouterr().err


def test_cli_runs_when_the_model_answers(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(llm, "get_chat_model", lambda spec, **kw: FakeListChatModel(responses=["ok"]))
    main(["--model", "fake:m", "--no-early-stop", "--out", str(tmp_path / "r.json")])
    out = capsys.readouterr().out
    assert "model fallbacks" in out and "router" in out


def test_report_counts_router_fallbacks():
    from household_agent.agents import Router
    scenarios = [s for s in generate_scenarios(0) if s.area == "routing"][:6]
    report = evaluate(scenarios, router=Router(Broken(exc=RuntimeError("down"))))
    assert report.metrics["router_model_calls"] == 6 and report.metrics["router_fallbacks"] == 6
    assert report.metrics["schema_validity"] == 0.0


def test_summary_includes_planner_and_critic_counts():
    report = evaluate(generate_scenarios(0)[:3])
    gen, critic = LLMThoughtGenerator(Broken(exc=RuntimeError("x"))), LLMCritic(Broken(exc=RuntimeError("x")))
    text = fallback_summary(report, gen, critic)
    assert "router 0/0" in text and "planner thoughts 0" in text and "critic 0" in text


@settings(max_examples=20, deadline=None)
@given(names=st.lists(st.text(alphabet="abcdef", min_size=1, max_size=8), max_size=5, unique=True), bad=st.sets(st.integers(0, 4)))
def test_property_preflight_reports_exactly_the_failing_checks(names, bad):
    def make(i):
        def call():
            if i in bad:
                raise ValueError("boom")
        return call

    checks = {n: make(i) for i, n in enumerate(names)}
    assert [p.split(":")[0] for p in preflight(checks)] == [n for i, n in enumerate(names) if i in bad]
