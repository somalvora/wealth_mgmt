"""The eval harness reports progress per scenario, so a slow hosted model is visible rather than silent."""
from hypothesis import given, settings
from hypothesis import strategies as st

from household_agent.evals import evaluate, generate_scenarios, main


def test_progress_is_called_once_per_scenario_in_order():
    scenarios = generate_scenarios(0)[:12]
    seen = []
    evaluate(scenarios, progress=lambda i, n, s: seen.append((i, n, s.scenario_id)))
    assert seen == [(i, 12, s.scenario_id) for i, s in enumerate(scenarios, 1)]


def test_no_progress_callback_is_fine():
    assert evaluate(generate_scenarios(0)[:5]).counts


def test_cli_prints_progress_to_stderr_not_stdout(capsys, tmp_path):
    main(["--out", str(tmp_path / "r.json")])
    out, err = capsys.readouterr()
    assert "1/321" in err and "321/321" in err
    assert "321/321" not in out and "scenarios" in out


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 50), k=st.integers(1, 15))
def test_property_progress_count_equals_scenario_count(seed, k):
    scenarios = generate_scenarios(seed)[:k]
    calls = []
    evaluate(scenarios, progress=lambda i, n, s: calls.append(i))
    assert calls == list(range(1, k + 1))
