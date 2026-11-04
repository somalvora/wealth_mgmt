import json
import sqlite3
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from household_agent.approval import build_approval_graph, start_request
from household_agent.audit import AuditLog
from household_agent.llm_planner import LLMCritic, LLMThoughtGenerator
from household_agent.mcp_server import MEMBERS
from household_agent.planning import _TEMPLATES, PERSONAS, PlanNode, inputs_hash, plan_allocation
from household_agent.synthetic import HOUSEHOLD, demo_planning_inputs
from household_finance.allocation import PlanScore

INPUTS = demo_planning_inputs()


class Fake(BaseChatModel):
    """Answers each call with ``respond(request_json)``; counts calls."""

    respond: Callable[[dict[str, Any]], str]
    calls: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        ask = json.loads(next(m.content for m in messages if isinstance(m, HumanMessage)))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.respond(ask)))])

    @property
    def _llm_type(self) -> str:
        return "fake"


def _template_json(tier: int, persona: str, override: dict[str, Any] | None = None) -> dict[str, Any]:
    step, rationale = _TEMPLATES[tier][persona]
    return {"persona": persona, "tier": step.tier, "fraction": str(step.fraction),
            "buckets": [{"bucket": b, "weight": str(w)} for b, w in step.buckets], "mode": step.mode,
            "rationale": rationale, **(override or {})}


def mirror(**overrides: dict[str, Any]) -> Callable[[dict[str, Any]], str]:
    """Replies with the template thoughts; ``overrides[f"{tier}.{persona}"]`` patches one of them."""
    def respond(ask):
        t = ask["tier"]
        return json.dumps({"thoughts": [_template_json(t, p, overrides.get(f"{t}.{p}")) for p in PERSONAS]})
    return respond


def _node(composite="0.8000"):
    c = Decimal(composite)
    return PlanNode("d1.balanced", 1, {"liquidity_buffer": 1000}, (), PlanScore(c, c, c, c))


# --- happy path ------------------------------------------------------------------------

def test_model_thoughts_drive_the_same_search():
    model = Fake(respond=mirror())
    gen = LLMThoughtGenerator(model, model_id="ollama:qwen3:8b")
    result = plan_allocation(INPUTS, gen)
    template = plan_allocation(INPUTS)
    assert result.status == "awaiting_approval" and result.evaluations == 15
    assert result.best.allocation == template.best.allocation and result.best.node_id == template.best.node_id
    assert model.calls == 5 and gen.fallbacks == []  # one call per beam node: 1 + 2 + 2
    assert gen.version == "llm_generator_1.0+ollama:qwen3:8b"
    assert result.input_hash == inputs_hash(INPUTS, gen) != template.input_hash


def test_model_strategy_and_rationale_are_used():
    gen = LLMThoughtGenerator(Fake(respond=mirror(**{"1.conservative": {
        "fraction": "0.5", "rationale": "Twins arriving in spring; hold half as extra cash."}})))
    result = plan_allocation(INPUTS, gen)
    node = next(n for n in result.trace if n.node_id == "d1.conservative")
    assert node.allocation["liquidity_buffer"] == INPUTS.surplus_minor // 2
    assert node.thoughts[-1].rationale.startswith("Twins arriving")


def test_reasoning_block_and_code_fence_are_stripped():
    def respond(ask):
        return "<think>tier 1 means cash</think>\n```json\n" + mirror()(ask) + "\n```"
    gen = LLMThoughtGenerator(Fake(respond=respond))
    assert plan_allocation(INPUTS, gen).status == "awaiting_approval" and gen.fallbacks == []


# --- negative paths: the model can't put money or bad structure into a thought ------------

def test_unparseable_reply_falls_back_to_templates_for_every_persona():
    gen = LLMThoughtGenerator(Fake(respond=lambda ask: "Sure! Here are some ideas..."))
    result = plan_allocation(INPUTS, gen)
    assert result.best.allocation == plan_allocation(INPUTS).best.allocation
    assert len(gen.fallbacks) == 15 and all(f["reason"].startswith("unparseable") for f in gen.fallbacks)
    assert all(t.rationale.startswith("[template fallback]") for t in result.best.thoughts)


def test_amount_fields_are_rejected():
    gen = LLMThoughtGenerator(Fake(respond=mirror(**{"1.tax_aggressive": {"amount_minor": 500000}})))
    result = plan_allocation(INPUTS, gen)
    assert {(f["persona"], f["tier"]) for f in gen.fallbacks} == {("tax_aggressive", "1")}
    assert "Extra inputs" in gen.fallbacks[0]["reason"]
    node = next(n for n in result.trace if n.node_id == "d1.tax_aggressive")
    assert node.thoughts[-1].rationale.startswith("[template fallback]")


def test_rationale_with_a_dollar_amount_is_rejected():
    gen = LLMThoughtGenerator(Fake(respond=mirror(**{"3.balanced": {"rationale": "Put $5,000 in brokerage."}})))
    plan_allocation(INPUTS, gen)
    assert gen.fallbacks and {f["persona"] for f in gen.fallbacks} == {"balanced"}
    assert all("money amounts" in f["reason"] for f in gen.fallbacks)


def test_fraction_outside_zero_one_is_rejected():
    gen = LLMThoughtGenerator(Fake(respond=mirror(**{"2.balanced": {"fraction": "1.5"}})))
    plan_allocation(INPUTS, gen)
    assert {f["persona"] for f in gen.fallbacks} == {"balanced"}


def test_missing_and_duplicate_personas_fall_back():
    def respond(ask):
        t = ask["tier"]
        return json.dumps({"thoughts": [_template_json(t, "balanced"), _template_json(t, "balanced", {"fraction": "0"})]})
    gen = LLMThoughtGenerator(Fake(respond=respond))
    thoughts = gen.propose(INPUTS, PlanNode("root", 0, {}, (), _node().score), 1)
    assert [t.persona for t in thoughts] == list(PERSONAS)
    assert thoughts[1].step.fraction == Decimal("0.10")  # first balanced kept
    assert {f["persona"]: f["reason"] for f in gen.fallbacks} == {
        "conservative": "persona missing", "tax_aggressive": "persona missing"}


def test_bucket_outside_the_tier_is_hard_pruned_not_repaired():
    gen = LLMThoughtGenerator(Fake(respond=mirror(**{"1.tax_aggressive": {
        "buckets": [{"bucket": "brokerage", "weight": "1"}]}})))
    result = plan_allocation(INPUTS, gen)
    node = next(n for n in result.trace if n.node_id == "d1.tax_aggressive")
    assert node.status == "hard_pruned" and node.score.violations[0].code == "bucket_not_in_tier"
    assert gen.fallbacks == [] and result.status == "awaiting_approval"


def test_wrong_tier_is_hard_pruned():
    gen = LLMThoughtGenerator(Fake(respond=mirror(**{"1.balanced": {"tier": 3}})))
    node = next(n for n in plan_allocation(INPUTS, gen).trace if n.node_id.startswith("d3.balanced")
                and n.depth == 1)
    assert node.status == "hard_pruned" and node.score.violations[0].code == "wrong_tier"


# --- critic --------------------------------------------------------------------------------

def test_agreeing_critic_keeps_the_plan_ready_for_approval():
    critic = LLMCritic(Fake(respond=lambda ask: json.dumps({"score": "0.85", "reason": "Reasonable."})))
    result = plan_allocation(INPUTS, critic=critic)
    assert result.status == "awaiting_approval" and critic.failures == []
    assert all(n.critic_score == Decimal("0.8500") for n in result.trace if n.status != "hard_pruned")


def test_disagreeing_critic_flags_needs_review_but_prunes_nothing():
    critic = LLMCritic(Fake(respond=lambda ask: json.dumps({"score": "0.1", "reason": "Too little cash."})))
    result, baseline = plan_allocation(INPUTS, critic=critic), plan_allocation(INPUTS)
    assert result.status == "needs_review" and result.discrepancy_node_ids
    assert result.best.allocation == baseline.best.allocation  # advisory only
    assert critic.reasons[result.best.node_id] == "Too little cash."


def test_unusable_critic_reply_has_no_opinion():
    critic = LLMCritic(Fake(respond=lambda ask: "I think it's fine"))
    result = plan_allocation(INPUTS, critic=critic)
    assert result.status == "awaiting_approval" and not result.discrepancy_node_ids
    assert len(critic.failures) == len([n for n in result.trace if n.status != "hard_pruned"])


def test_critic_rejects_nan_and_extra_fields():
    for reply in ('{"score": "NaN"}', '{"score": "0.5", "amount_minor": 1}'):
        critic = LLMCritic(Fake(respond=lambda ask, r=reply: r))
        assert critic.score(INPUTS, _node("0.8000")) == Decimal("0.8000") and critic.failures


# --- approval wiring -----------------------------------------------------------------------

def test_approving_a_model_plan_makes_no_further_model_calls():
    model = Fake(respond=mirror())
    graph = build_approval_graph(inputs_for=lambda hh: INPUTS, members=MEMBERS,
                                 audit=AuditLog(sqlite3.connect(":memory:")), checkpointer=InMemorySaver(),
                                 generator=LLMThoughtGenerator(model))
    thread_id, out = start_request(graph, HOUSEHOLD, "alex")
    assert out["status"] == "awaiting_approval" and model.calls == 5
    out = graph.invoke(Command(resume={"approver": "jordan", "approved": True}),
                       {"configurable": {"thread_id": thread_id}})
    assert out["status"] == "approved" and model.calls == 5


# --- properties ----------------------------------------------------------------------------

@settings(max_examples=40, deadline=None)
@given(st.text(max_size=200) | st.builds(lambda o: json.dumps({"thoughts": o}),
                                         st.lists(st.dictionaries(st.sampled_from(
                                             ["persona", "tier", "fraction", "buckets", "rationale", "amount_minor"]),
                                             st.one_of(st.integers(), st.text(max_size=10), st.sampled_from(PERSONAS))),
                                             max_size=4)))
def test_any_reply_yields_one_thought_per_persona(reply):
    gen = LLMThoughtGenerator(Fake(respond=lambda ask: reply))
    thoughts = gen.propose(INPUTS, PlanNode("root", 0, {}, (), _node().score), 2)
    assert [t.persona for t in thoughts] == list(PERSONAS)
    assert all(Decimal(0) <= t.step.fraction <= 1 and t.step.buckets for t in thoughts)


@settings(max_examples=60, deadline=None)
@given(st.decimals(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000, places=6))
def test_critic_score_is_clamped_to_unit_interval(score):
    critic = LLMCritic(Fake(respond=lambda ask: json.dumps({"score": str(score)})))
    s = critic.score(INPUTS, _node())
    assert Decimal(0) <= s <= Decimal(1) and s == s.quantize(Decimal("0.0001"))
