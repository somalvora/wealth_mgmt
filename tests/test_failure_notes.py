"""A failed specialist is explained in the briefing: a busy model says so, a crash names its type only."""
import pytest
from hypothesis import given
from hypothesis import strategies as st

from household_agent.agents import NO_LEDGER_FIGURES, AgentResult, compose_briefing, failure_notes


def _failed(*errors, agent="ledger_analyst"):
    return AgentResult(task_id="t", correlation_id="c", agent=agent, status="failed", needs_review=True,
                       errors=list(errors))


def _briefing(*results):
    return compose_briefing({"results": [r.model_dump() for r in results], "routing": {"intent": "historical_comparison"}})


@pytest.mark.parametrize("error", [
    "GoogleAPIError: 503 UNAVAILABLE. This model is currently experiencing high demand.",
    "GoogleRateLimitError: 429 RESOURCE_EXHAUSTED",
    "RateLimitError: slow down",
])
def test_a_busy_or_over_quota_model_is_called_out_in_the_briefing(error):
    out = _briefing(_failed(error))
    [note] = out["questions"]
    assert "busy or over its quota" in note and "ledger analyst" in note and "ask again" in note
    assert NO_LEDGER_FIGURES not in out["questions"] and out["needs_review"] and out["statements"] == []


def test_another_crash_names_its_type_and_never_its_message():
    out = _briefing(_failed("ConnectionError: key=AIza-SECRET-123 refused"))
    [note] = out["questions"]
    assert "ConnectionError" in note and "SECRET" not in note and "busy" not in note


def test_a_tool_name_or_stray_number_is_not_taken_for_an_error_type_or_a_busy_model():
    [note] = failure_notes([_failed("get_transactions: 503 rows came back")])
    assert "busy" not in note and "get_transactions" not in note and "(error)" in note


def test_a_failure_with_no_error_text_still_gets_a_note():
    assert "unknown error" in failure_notes([_failed()])[0]


def test_a_finished_specialist_adds_no_failure_note():
    ok = AgentResult(task_id="t", correlation_id="c", agent="ledger_analyst", status="completed")
    assert failure_notes([ok]) == []


def test_each_failed_specialist_gets_its_own_note():
    notes = failure_notes([_failed("ValueError: x"), _failed("TimeoutError: y", agent="policy_compliance")])
    assert len(notes) == 2 and "policy compliance" in notes[1]


@given(message=st.text(max_size=60))
def test_property_the_error_message_never_reaches_the_note(message):
    note = failure_notes([_failed(f"WeirdError: SECRET_{message}")])[0]
    assert "WeirdError" in note and "SECRET_" not in note
