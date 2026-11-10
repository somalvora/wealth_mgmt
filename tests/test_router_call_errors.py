"""A failed router model call (dropped connection, API 500) falls back to the keyword rules, not a crash."""
import httpcore
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from household_agent.agents import Router, rule_route
from tests.test_model_call_errors import Broken

DROPPED = httpcore.RemoteProtocolError("Server disconnected without sending a response.")


def test_a_failed_call_falls_back_to_the_rules():
    request = "transfer $5,000 to brokerage"
    decision, source = Router(Broken(exc=DROPPED))(request)
    assert source == "rules_fallback" and decision == rule_route(request)


def test_the_fallback_still_refuses_money_movement():
    decision, _ = Router(Broken(exc=RuntimeError("500 INTERNAL")))("please pay the electric bill now")
    assert decision.external_action_requested


def test_keyboard_interrupt_is_not_swallowed():
    with pytest.raises(KeyboardInterrupt):
        Router(Broken(exc=KeyboardInterrupt()))("how much did we spend on groceries")


def test_no_model_still_uses_rules_source():
    assert Router()("how much did we spend on groceries")[1] == "rules"


@settings(max_examples=20, deadline=None)
@given(msg=st.text(max_size=40), req=st.text(min_size=1, max_size=60),
       kind=st.sampled_from([RuntimeError, ValueError, ConnectionError, TimeoutError]))
def test_property_any_ordinary_error_gives_the_rules_decision(msg, req, kind):
    decision, source = Router(Broken(exc=kind(msg)))(req)
    assert source == "rules_fallback" and decision == rule_route(req)
