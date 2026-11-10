"""Hosted clients carry a per-request timeout, so one unanswered request cannot hang a run."""
import pytest

from household_agent import llm
from household_agent.llm import get_chat_model
from tests.test_llm import groq_env  # noqa: F401  (fixture)


def test_the_timeout_is_finite_and_shorter_than_ten_minutes():
    assert 0 < llm.HOSTED_TIMEOUT_S <= 600


def test_gemini_client_has_the_timeout(groq_env, monkeypatch):  # noqa: F811
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-123")
    assert get_chat_model("gemini:gemini-2.5-flash").timeout == llm.HOSTED_TIMEOUT_S


def test_groq_client_has_the_timeout(groq_env):  # noqa: F811
    assert get_chat_model("groq:qwen/qwen3-32b").request_timeout == llm.HOSTED_TIMEOUT_S


@pytest.mark.parametrize("spec", ["gemini:gemini-2.5-flash", "gemini:gemma-4-26b-a4b-it"])
def test_every_gemini_model_gets_it(groq_env, monkeypatch, spec):  # noqa: F811
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-123")
    assert get_chat_model(spec).timeout == llm.HOSTED_TIMEOUT_S
