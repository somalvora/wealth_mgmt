"""The vertical-slice report says why a specialist failed, not just that it did."""
from hypothesis import given, settings
from hypothesis import strategies as st

from household_agent.agents import AgentResult
from household_agent.vertical_slice import _why, run_vertical_slice
from tests.test_vertical_slice import _check, _models


def _result(status="failed", errors=()):
    return AgentResult(task_id="t", correlation_id="c", agent="ledger_analyst", status=status, errors=list(errors))


def test_a_completed_result_has_no_reason():
    assert _why(_result("completed", ["ignored"])) == ""


def test_a_failure_without_errors_has_no_reason():
    assert _why(_result("failed", [])) == ""


def test_only_the_first_line_of_the_first_error_is_shown():
    assert _why(_result(errors=["TimeoutError: too slow\ntraceback line", "second"])) == " (TimeoutError: too slow)"


def test_a_crashing_specialist_model_shows_its_error_in_the_trace_detail(tmp_path):
    class Boom:
        def bind_tools(self, *a, **k):
            return self

        def invoke(self, *a, **k):
            raise ConnectionError("model server unreachable")

    report = run_vertical_slice(tmp_path, models=_models(ledger=lambda: Boom()))
    detail = _check(report, "three_agent_trace").detail
    assert "ledger_analyst=failed" in detail and "unreachable" in detail


@settings(max_examples=30, deadline=None)
@given(msg=st.text(min_size=1, max_size=500))
def test_property_reason_is_one_short_line(msg):
    why = _why(_result(errors=[msg]))
    assert "\n" not in why and len(why) <= 160 + 3
