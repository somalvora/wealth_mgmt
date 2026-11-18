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


@pytest.mark.parametrize("text", [
    "Did we pay more for utilities in October than last year?",
    "Did I pay the electric bill twice this year?",
    "Did you pay more for groceries this year?",
    "Did we ever pay the plumber?",
])
def test_a_past_tense_did_we_pay_question_is_not_a_payment_request(text):
    d = rule_route(text)
    assert not d.external_action_requested and not d.approval_required


def test_a_did_we_pay_comparison_reaches_the_ledger_analyst():
    d = rule_route("Did we pay more for utilities in October than last year?")
    assert d.intent == "historical_comparison" and [i.agent for i in d.dispatch_plan] == ["ledger_analyst"]


@pytest.mark.parametrize("text", [
    "Pay the HOA dues from the rental account.",
    "Did we pay the plumber? Pay him $400 now.",
    "Please pay the electric bill today",
    "Make a payment of $500 to the plumber.",
    "Transfer $500 to savings.",
    "Pay the property tax now",
])
def test_real_payment_requests_are_still_refused(text):
    assert rule_route(text).external_action_requested
