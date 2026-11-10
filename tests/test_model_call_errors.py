"""A failed model call (dropped connection, API error) is a recorded fallback, not a crash."""
import httpcore
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult

from household_agent.llm_planner import LLMCritic, LLMThoughtGenerator
from household_agent.planning import PERSONAS, plan_allocation
from tests.test_llm_planner import INPUTS, _node


class Broken(BaseChatModel):
    """Every call raises ``exc``."""

    exc: BaseException
    calls: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls += 1
        raise self.exc

    @property
    def _llm_type(self) -> str:
        return "broken"


DROPPED = httpcore.RemoteProtocolError("Server disconnected without sending a response.")


def test_critic_gives_no_opinion_when_the_call_fails():
    critic = LLMCritic(Broken(exc=DROPPED))
    node = _node("0.8000")
    assert critic.score(INPUTS, node) == node.score.composite
    assert critic.failures == [{"node_id": node.node_id, "reason": "model call failed: RemoteProtocolError"}]


def test_generator_falls_back_to_templates_for_every_persona_when_the_call_fails():
    gen = LLMThoughtGenerator(Broken(exc=DROPPED))
    thoughts = gen.propose(INPUTS, _node(), 1)
    assert len(thoughts) == len(PERSONAS) and all("[template fallback]" in t.rationale for t in thoughts)
    assert len(gen.fallbacks) == len(PERSONAS)
    assert all(f["reason"] == "model call failed: RemoteProtocolError" for f in gen.fallbacks)


def test_a_whole_planning_run_survives_a_dead_model():
    gen, critic = LLMThoughtGenerator(Broken(exc=DROPPED)), LLMCritic(Broken(exc=DROPPED))
    result = plan_allocation(INPUTS, gen, critic)
    assert result.status == "awaiting_approval" and len(gen.fallbacks) == 15 and critic.failures


def test_the_error_message_is_not_recorded_only_its_type():
    secret = RuntimeError("key=AIza-SECRET-123 rejected")
    gen = LLMThoughtGenerator(Broken(exc=secret))
    gen.propose(INPUTS, _node(), 1)
    assert "SECRET" not in repr(gen.fallbacks)
    critic = LLMCritic(Broken(exc=secret))
    critic.score(INPUTS, _node())
    assert "SECRET" not in repr(critic.failures)


def test_keyboard_interrupt_is_not_swallowed():
    with pytest.raises(KeyboardInterrupt):
        LLMCritic(Broken(exc=KeyboardInterrupt())).score(INPUTS, _node())


@settings(max_examples=20, deadline=None)
@given(msg=st.text(max_size=40), kind=st.sampled_from([RuntimeError, ValueError, ConnectionError, TimeoutError]))
def test_property_any_ordinary_error_yields_one_thought_per_persona_and_a_clamped_score(msg, kind):
    gen, critic = LLMThoughtGenerator(Broken(exc=kind(msg))), LLMCritic(Broken(exc=kind(msg)))
    assert [t.persona for t in gen.propose(INPUTS, _node(), 1)] == list(PERSONAS)
    assert 0 <= critic.score(INPUTS, _node()) <= 1
