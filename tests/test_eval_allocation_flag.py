"""--allocation-scenarios: skip or shrink the slow model-driven allocation scenarios without hiding that it was done."""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from household_agent.evals import main


def _run(capsys, tmp_path, *flags):
    code = main(["--out", str(tmp_path / "r.json"), *flags])
    return code, capsys.readouterr().out


def test_default_runs_all_sixty_allocation_scenarios(capsys, tmp_path):
    _, out = _run(capsys, tmp_path)
    assert "'allocation': 60" in out and "NOTE: no allocation" not in out


def test_zero_skips_allocation_and_says_the_gates_are_vacuous(capsys, tmp_path):
    code, out = _run(capsys, tmp_path, "--allocation-scenarios", "0")
    assert "'allocation'" not in out and "NOTE: no allocation scenarios were run" in out
    assert "scenarios {" in out and code == 0  # everything else still runs and passes offline


def test_a_small_number_runs_exactly_that_many(capsys, tmp_path):
    _, out = _run(capsys, tmp_path, "--allocation-scenarios", "5")
    assert "'allocation': 5" in out and "NOTE: no allocation" not in out


def test_negative_is_rejected(capsys, tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["--allocation-scenarios", "-1"])
    assert e.value.code == 2


@settings(max_examples=6, deadline=None)
@given(n=st.integers(1, 8))
def test_property_count_matches_the_request(n):
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["--allocation-scenarios", str(n)])
    assert f"'allocation': {n}" in buf.getvalue()
