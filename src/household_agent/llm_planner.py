"""Model-driven persona thoughts and critic for the Tree-of-Thought planner (design §5.8).

Both classes sit behind the planner's existing protocols, so the beam search, the hard
constraints and the scoring do not change:

- ``LLMThoughtGenerator`` asks the chat model, once per beam node, for one strategy per persona
  (Conservative / Balanced / Tax-Aggressive) as JSON: tier, fraction of the remaining surplus,
  bucket weights, mode and a short rationale. The model **never supplies amounts**: the schema
  forbids extra fields (so ``amount_minor`` and friends are rejected) and a rationale that quotes
  a currency amount is rejected too. ``household_finance.allocation`` still turns the strategy
  into cents. A persona whose output is missing or malformed falls back to its template thought,
  and the fallback is recorded. Output that is well-formed but breaks a rule (wrong tier, bucket
  not in the tier, over a statutory cap) is passed through on purpose: the planner hard-prunes it,
  and the trace shows why.
- ``LLMCritic`` asks the model for a qualitative 0-1 score of a node. It is clamped to [0, 1]
  and stays advisory: a large gap from the deterministic score keeps a branch alive for human
  review (``needs_review``); it can never prune one. If the model's reply is unusable the critic
  returns the deterministic score, so a broken critic cannot raise a false discrepancy.
"""

import json
import re
from collections.abc import Sequence
from decimal import Decimal
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from household_agent.llm import chat_model_spec, get_chat_model
from household_agent.planning import _TEMPLATES, PERSONAS, PlanNode, Thought
from household_finance.allocation import TIER_BUCKETS, AllocationStep, PlanningInputs

GENERATOR_VERSION = "llm_generator_1.0"
CRITIC_VERSION = "llm_critic_1.0"
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)  # qwen3 reasoning block
_CURRENCY = re.compile(r"[$€£]\s*\d|\d[\d,]*(?:\.\d+)?\s*(?:usd|dollars?|cents?)\b", re.IGNORECASE)
_SCORE_Q = Decimal("0.0001")

GENERATOR_SYSTEM = """You propose capital-allocation strategies for a household planner.
Reply with JSON only: {"thoughts": [<one object per persona>]}.
Each object has exactly these fields:
  persona   - one of "conservative", "balanced", "tax_aggressive"
  tier      - the tier you were asked about
  fraction  - share of the remaining surplus to commit, a decimal string in [0, 1]
  buckets   - list of {"bucket": <name>, "weight": <decimal string > 0>}, only buckets allowed for the tier
  mode      - "split" (weights) or "fill_in_order" (priority order, filled up to each bucket's room)
  rationale - one plain sentence; no dollar amounts
Never write amounts of money. The calculator turns your strategy into amounts."""

CRITIC_SYSTEM = """You review a proposed household allocation for qualitative risks the formula may miss
(e.g. too little cash for a family with irregular income, debt left at a high rate).
Reply with JSON only: {"score": <decimal string between 0 and 1>, "reason": <one sentence>}."""


def _clean(text: str) -> str:
    return _FENCE.sub("", _THINK.sub("", text).strip()).strip()


class _BucketWeight(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bucket: str = Field(min_length=1, max_length=40)
    weight: Decimal = Field(gt=0, le=100)


class _ThoughtModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    persona: Literal["conservative", "balanced", "tax_aggressive"]
    tier: int
    fraction: Decimal = Field(ge=0, le=1)
    buckets: list[_BucketWeight] = Field(min_length=1, max_length=3)
    mode: Literal["split", "fill_in_order"] = "split"
    rationale: str = Field(min_length=1, max_length=400)

    @field_validator("rationale")
    @classmethod
    def _no_amounts(cls, v: str) -> str:
        if _CURRENCY.search(v):
            raise ValueError("rationale must not state money amounts")
        return v

    def to_thought(self) -> Thought:
        step = AllocationStep(self.tier, self.fraction, tuple((b.bucket, b.weight) for b in self.buckets), self.mode)
        return Thought(self.persona, step, self.rationale)


class _ThoughtsReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thoughts: list[dict[str, Any]] = Field(max_length=len(PERSONAS) * 2)


class _CriticReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: Decimal
    reason: str = Field(default="", max_length=400)


def _model_id(model: BaseChatModel | None, model_id: str | None) -> str:
    if model_id:
        return model_id
    return str(chat_model_spec()) if model is None else getattr(model, "_llm_type", type(model).__name__)


def _context(inputs: PlanningInputs, node: PlanNode) -> dict[str, Any]:
    return {
        "surplus_minor": inputs.surplus_minor,
        "required_reserve_minor": inputs.required_reserve_minor,
        "verified_contribution_room_minor": dict(sorted(inputs.contribution_room_minor.items())),
        "allocated_so_far_minor": dict(sorted(node.allocation.items())),
        "path": [{"persona": t.persona, "tier": t.step.tier, "rationale": t.rationale} for t in node.thoughts],
    }


class LLMThoughtGenerator:
    """One model call per beam node (5 per run); one validated thought per persona."""

    def __init__(self, model: BaseChatModel | None = None, *, model_id: str | None = None):
        self.version = f"{GENERATOR_VERSION}+{_model_id(model, model_id)}"
        self.model = model or get_chat_model()
        self.fallbacks: list[dict[str, str]] = []  # {"node_id", "tier", "persona", "reason"}

    def _fallback(self, node: PlanNode, tier: int, persona: str, reason: str) -> Thought:
        self.fallbacks.append({"node_id": node.node_id, "tier": str(tier), "persona": persona, "reason": reason})
        step, rationale = _TEMPLATES[tier][persona]
        return Thought(persona, step, f"[template fallback] {rationale}")

    def propose(self, inputs: PlanningInputs, node: PlanNode, tier: int) -> Sequence[Thought]:
        ask = {"tier": tier, "allowed_buckets": sorted(TIER_BUCKETS[tier]), "personas": list(PERSONAS),
               **_context(inputs, node)}
        reply = self.model.invoke([SystemMessage(GENERATOR_SYSTEM), HumanMessage(json.dumps(ask))])
        try:
            raw = _ThoughtsReply.model_validate_json(_clean(str(reply.content))).thoughts
        except ValidationError as e:
            return [self._fallback(node, tier, p, f"unparseable reply: {e.errors()[0]['msg']}") for p in PERSONAS]

        by_persona: dict[str, Thought] = {}
        errors: dict[str, str] = {}
        for item in raw:
            persona = item.get("persona") if isinstance(item, dict) else None
            try:
                thought = _ThoughtModel.model_validate(item).to_thought()
            except ValidationError as e:
                if isinstance(persona, str):
                    errors.setdefault(persona, f"invalid thought: {e.errors()[0]['msg']}")
                continue
            if thought.persona in by_persona:
                errors.setdefault(thought.persona, "duplicate persona; first kept")
                continue
            by_persona[thought.persona] = thought
        return [by_persona[p] if p in by_persona else self._fallback(node, tier, p, errors.get(p, "persona missing"))
                for p in PERSONAS]


class LLMCritic:
    """Advisory 0-1 score from the model, clamped and quantized; never prunes."""

    def __init__(self, model: BaseChatModel | None = None, *, model_id: str | None = None):
        self.version = f"{CRITIC_VERSION}+{_model_id(model, model_id)}"
        self.model = model or get_chat_model()
        self.failures: list[dict[str, str]] = []
        self.reasons: dict[str, str] = {}

    def score(self, inputs: PlanningInputs, node: PlanNode) -> Decimal:
        ask = {**_context(inputs, node), "node_id": node.node_id,
               "deterministic_scores": {"tax_efficiency": str(node.score.tax_efficiency),
                                        "liquidity_safety": str(node.score.liquidity_safety),
                                        "expected_net_yield": str(node.score.expected_net_yield)}}
        reply = self.model.invoke([SystemMessage(CRITIC_SYSTEM), HumanMessage(json.dumps(ask))])
        try:
            parsed = _CriticReply.model_validate_json(_clean(str(reply.content)))
            if not parsed.score.is_finite():
                raise ValueError("score is not finite")
        except (ValidationError, ValueError) as e:
            self.failures.append({"node_id": node.node_id, "reason": str(e).splitlines()[0]})
            return node.score.composite  # no opinion: cannot create a discrepancy
        self.reasons[node.node_id] = parsed.reason
        return min(max(parsed.score, Decimal(0)), Decimal(1)).quantize(_SCORE_Q)
