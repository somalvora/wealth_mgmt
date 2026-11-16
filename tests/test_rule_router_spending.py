"""The keyword fallback router (used when the model router fails or is rate limited) knows everyday spending words."""

import pytest

from household_agent.agents import rule_route


@pytest.mark.parametrize("text", [
    "Summary of last months card.",
    "spending history is what I was askig.",
    "Can you tell me how much we spend last month?",
    "Give me a breakdown of expenses by category",
    "Summarize my transactions for October",
])
def test_spending_questions_reach_the_ledger_analyst(text):
    d = rule_route(text)
    assert d.intent == "historical_comparison" and [i.agent for i in d.dispatch_plan] == ["ledger_analyst"]


@pytest.mark.parametrize("text", ["What is the weather in Paris?", "Write me a poem about cats", "Hello there"])
def test_unrelated_text_is_still_out_of_scope(text):
    d = rule_route(text)
    assert d.intent == "out_of_scope" and d.dispatch_plan == []


def test_an_allocation_question_with_how_much_stays_allocation_planning():
    assert rule_route("How much of the surplus should we invest?").intent == "allocation_planning"


def test_moving_money_is_still_refused_when_a_spending_word_is_present():
    assert rule_route("Transfer $500 from checking to savings and summarize my spending").external_action_requested
