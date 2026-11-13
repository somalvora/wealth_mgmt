"""The router prompt defines every intent, because a real model otherwise guesses (Gemma sent tax questions to
out_of_scope and added a ledger dispatch to allocation requests)."""
import typing

from household_agent.agents import ROUTER_PROMPT, RoutingDecision


def _intents() -> list[str]:
    return [i for i in typing.get_args(RoutingDecision.model_fields["intent"].annotation) if i != "statement_ingestion"]


def test_every_model_chosen_intent_is_named_with_a_definition():
    for intent in _intents():
        assert f"{intent} =" in ROUTER_PROMPT, intent


def test_statement_ingestion_is_not_offered_to_the_model():
    assert "statement_ingestion" not in ROUTER_PROMPT


def test_tax_questions_are_never_out_of_scope_and_allocation_has_no_dispatch():
    assert "never out_of_scope" in ROUTER_PROMPT
    assert "empty dispatch_plan" in ROUTER_PROMPT and "approval_required true" in ROUTER_PROMPT


def test_external_actions_are_still_forbidden():
    assert "Never plan payments, transfers" in ROUTER_PROMPT and "external_action_requested true only" in ROUTER_PROMPT
