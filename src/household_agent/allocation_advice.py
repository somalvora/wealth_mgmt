"""Supervisor-side "where should our extra money go?" flow (design §5.8 companion).

    request -> access check -> candidate plans -> evaluate_allocation_candidates -> model chooses/explains -> validate

What the model does: it may propose candidate *shapes* (shares of the extra amount per bucket, in basis points,
never dollars) and then choose among the evaluated candidates and write claims about them. What code does: the extra
amount comes from the request text, shapes become cents by largest-remainder apportionment, the deterministic set of
standard candidates is always evaluated, every outcome comes from ``evaluate_allocation_candidates``, and every
claim the model writes is re-checked by ``validate_claims`` against that evaluation. A model choice with constraint
violations is flagged and replaced by the best valid candidate; if no candidate is valid the answer is
``needs_review``. Nothing here moves money; the output is decision support, not financial advice.
"""

import json
import re
from collections.abc import Callable
from decimal import ROUND_CEILING, Decimal
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from household_agent.allocation import MAX_CANDIDATES, CandidatePlan, HouseholdFinancialState
from household_agent.allocation_tool import ResolvedState
from household_agent.retrieval import Claim, validate_claims

FRAMING = ("Decision support only, not financial advice. Every figure comes from the cited deterministic evaluation "
           "run; no investment returns are forecast and no money is moved.")
MAX_MODEL_SHAPES = 5
BPS = 10_000
_AMOUNT = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
Bucket = Literal["debt", "emergency_fund", "retirement", "hold"]

PROPOSE_PROMPT = (
    "You propose candidate ways to split a household's extra money. Reply with only JSON: "
    '{"candidates": [{"name": "short_name", "rationale": "one sentence", "lines": [{"bucket": '
    '"debt|emergency_fund|retirement|hold", "target_id": "debt id or retirement account id, else null", '
    '"share_bps": 0-10000}]}]}. Each candidate\'s share_bps must add to exactly 10000. Never write dollar amounts or '
    "numbers other than share_bps: code computes every figure. Use only the debt and account ids you are given. "
    "At most " + str(MAX_MODEL_SHAPES) + " candidates."
)
CHOOSE_PROMPT = (
    "You choose among already-evaluated candidate plans and explain the choice. Reply with only JSON: "
    '{"chosen": "candidate name or null", "claims": [{"text": "...", "evidence_ids": ["<evidence_id>"]}]}. '
    "Use only candidate names from the evaluation. Every claim must cite the evaluation's evidence_id and may only "
    "state numbers that appear in it. Never do arithmetic. This is decision support, not financial advice."
)


class ShapeLine(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bucket: Bucket
    target_id: str | None = None
    share_bps: int = Field(ge=0, le=BPS)


class CandidateShape(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=40)
    rationale: str = Field(default="", max_length=300)
    lines: list[ShapeLine] = Field(min_length=1, max_length=8)


class ProposedShapes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[CandidateShape] = Field(max_length=MAX_MODEL_SHAPES)


class _Claim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class ModelChoice(BaseModel):
    chosen: str | None = None
    claims: list[_Claim] = Field(default_factory=list)


def extra_amount_from_request(request: str) -> int:
    """The one dollar amount the request names, in cents. ``ValueError`` if there is none or it is ambiguous."""
    amounts = {int((Decimal(m.replace(",", "")) * 100).to_integral_value()) for m in _AMOUNT.findall(request)}
    if not amounts:
        raise ValueError("the question names no dollar amount; say how much extra money there is, for example $2,000")
    if len(amounts) > 1:
        raise ValueError("the question names more than one dollar amount; say which one is the extra money to place")
    return amounts.pop()


def expand_shape(shape: CandidateShape, extra: int) -> CandidatePlan:
    """Shape (shares) to cents by largest-remainder apportionment: the cents add up to exactly ``extra``.
    ``ValueError`` if the shares do not add up to 10000."""
    if sum(line.share_bps for line in shape.lines) != BPS:
        raise ValueError(f"shares add up to {sum(line.share_bps for line in shape.lines)}, not {BPS}")
    floors = [extra * line.share_bps // BPS for line in shape.lines]
    order = sorted(range(len(floors)), key=lambda i: (-(extra * shape.lines[i].share_bps % BPS), i))
    for i in order[: extra - sum(floors)]:
        floors[i] += 1
    debts: dict[str, int] = {}
    retirement: dict[str, int] = {}
    ef = hold = 0
    for line, cents in zip(shape.lines, floors, strict=True):
        if line.bucket == "emergency_fund":
            ef += cents
        elif line.bucket == "hold":
            hold += cents
        elif line.bucket == "debt":
            debts[line.target_id or ""] = debts.get(line.target_id or "", 0) + cents
        else:
            retirement[line.target_id or ""] = retirement.get(line.target_id or "", 0) + cents
    return CandidatePlan(name=_model_name(shape.name), debt_principal_minor=debts, emergency_fund_minor=ef,
                         retirement_minor=retirement, hold_minor=hold)


def _model_name(name: str) -> str:
    return "model_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]


def standard_candidates(state: HouseholdFinancialState, extra: int) -> list[CandidatePlan]:
    """The deterministic fallback set; it is always evaluated, whatever the model proposes.

    ``all_to_highest_apr_debt``, ``all_to_debt_<id>`` for each other debt, ``all_to_emergency_fund``,
    ``all_to_retirement`` (the account with the most headroom), ``all_to_cash_hold`` and a waterfall (emergency fund
    up to its target, then the highest-APR debt, then retirement, the rest held). Debt and retirement amounts are
    capped at the balance or headroom with the rest held, so these candidates break no limit by construction; a
    violation can still come from the household's own minimum buffer.
    """
    by_apr = sorted(state.debts, key=lambda d: (-d.apr_percent, d.debt_id))
    top_acct = min(state.retirement_accounts, key=lambda a: (-a.headroom_minor, a.account_id), default=None)
    plans: list[CandidatePlan] = []
    for i, d in enumerate(by_apr):
        put = min(extra, d.balance_minor)
        plans.append(CandidatePlan(name="all_to_highest_apr_debt" if i == 0 else f"all_to_debt_{d.debt_id}",
                                   debt_principal_minor={d.debt_id: put}, hold_minor=extra - put))
    plans.append(CandidatePlan(name="all_to_emergency_fund", emergency_fund_minor=extra))
    if top_acct is not None:
        put = min(extra, top_acct.headroom_minor)
        plans.append(CandidatePlan(name="all_to_retirement", retirement_minor={top_acct.account_id: put},
                                   hold_minor=extra - put))
    plans.append(CandidatePlan(name="all_to_cash_hold", hold_minor=extra))
    target = int((state.emergency_fund_target_months * state.monthly_essential_expenses_minor).quantize(
        Decimal(1), rounding=ROUND_CEILING))
    left = extra
    ef = min(left, max(0, target - state.emergency_fund_balance_minor))
    left -= ef
    debts: dict[str, int] = {}
    if by_apr and left:
        debts[by_apr[0].debt_id] = put = min(left, by_apr[0].balance_minor)
        left -= put
    retirement: dict[str, int] = {}
    if top_acct is not None and left:
        retirement[top_acct.account_id] = put = min(left, top_acct.headroom_minor)
        left -= put
    plans.append(CandidatePlan(name="waterfall_ef_then_debt_then_retirement", emergency_fund_minor=ef,
                               debt_principal_minor=debts, retirement_minor=retirement, hold_minor=left))
    return plans[:MAX_CANDIDATES]


def _json_reply(model: BaseChatModel, system: str, payload: dict[str, Any]) -> str:
    reply = model.invoke([SystemMessage(system), HumanMessage(json.dumps(payload, default=str))])
    return _FENCE.sub("", str(reply.text).strip())


def propose_shapes(model: BaseChatModel, request: str, state: HouseholdFinancialState, extra: int) -> tuple[list[CandidatePlan], list[str]]:
    """Model-proposed shapes expanded to cents, plus notes on what was dropped. Never raises."""
    notes: list[str] = []
    try:
        proposed = ProposedShapes.model_validate_json(_json_reply(model, PROPOSE_PROMPT, {
            "request": request, "debt_ids": [d.debt_id for d in state.debts],
            "retirement_account_ids": [a.account_id for a in state.retirement_accounts]}))
    except (ValidationError, ValueError) as exc:
        return [], [f"model candidate proposal ignored: {type(exc).__name__}"]
    plans: list[CandidatePlan] = []
    for shape in proposed.candidates:
        try:
            plans.append(expand_shape(shape, extra))
        except ValueError as exc:
            notes.append(f"model candidate {shape.name!r} dropped: {exc}")
    return plans, notes


def _row(c: dict[str, Any]) -> dict[str, Any]:
    keep = ("name", "rank", "valid", "interest_saved_minor", "interest_saved_display", "buffer_months",
            "emergency_fund_pct_of_target", "retirement_used_minor", "retirement_used_display", "unallocated_minor",
            "unallocated_display")
    return {**{k: c[k] for k in keep}, "violations": [v["message"] for v in c["violations"]]}


def advise(
    *, request: str, resolve: Callable[[], ResolvedState | dict[str, Any]], model: BaseChatModel | None,
    evaluate: Callable[..., dict[str, Any]], access_denial: Callable[[], str | None],
) -> dict[str, Any]:
    """Run the flow and return the ``allocation`` record for the job state and briefing. Never raises.

    ``access_denial()`` is the pre-flight authorization check (before any model call), ``resolve()`` builds the
    household snapshot or returns a tool-style error dict, and ``evaluate(extra_amount_minor=, candidates=)`` is the
    bound ``evaluate_allocation_candidates`` call. ``status`` is ``recommended``, ``needs_review``, ``missing_inputs``
    or ``denied``.
    """
    base: dict[str, Any] = {"framing": FRAMING, "flags": [], "questions": [], "statements": [], "needs_review": False}

    def denied(reason: str) -> dict[str, Any]:
        return {**base, "status": "denied", "denial": reason, "questions": [f"Allocation advice refused: {reason}."]}

    def missing(message: str) -> dict[str, Any]:
        return {**base, "status": "missing_inputs", "needs_review": True, "questions": [message]}

    reason = access_denial()
    if reason:
        return denied(reason)
    resolved = resolve()
    if isinstance(resolved, dict):
        err = resolved["error"]
        return missing(f"Allocation advice needs household details that are never guessed ({err['type']}): {err['message']}")
    try:
        extra = extra_amount_from_request(request)
    except ValueError as exc:
        return missing(str(exc))

    plans = standard_candidates(resolved.state, extra)
    flags: list[str] = []
    if model is not None:
        shaped, notes = propose_shapes(model, request, resolved.state, extra)
        flags += notes
        known = {p.name for p in plans}
        for p in shaped:
            if p.name in known:
                flags.append(f"model candidate {p.name!r} dropped: duplicate name")
            elif len(plans) >= MAX_CANDIDATES:
                flags.append(f"model candidate {p.name!r} dropped: candidate limit")
            else:
                plans.append(p)
                known.add(p.name)

    result = evaluate(extra_amount_minor=extra, candidates=plans, facts=resolved)
    if not result["ok"]:
        err = result["error"]
        if err["type"] == "authorization_denied":
            return denied(err["message"])
        return missing(f"Allocation evaluation refused ({err['type']}): {err['message']}")

    out, run_id = result["output"], result["run_id"]
    by_name = {c["name"]: c for c in out["candidates"]}
    best_valid = out["best_valid"]
    recommended, model_choice, needs_review = best_valid, None, False
    kept: list[dict[str, Any]] = []
    removed: list[str] = []
    if model is not None:
        try:
            choice: ModelChoice | None = ModelChoice.model_validate_json(_json_reply(model, CHOOSE_PROMPT, {
                "request": request, "evidence_id": run_id, "evaluation": out}))
        except (ValidationError, ValueError):
            choice = None
            needs_review = True
            flags.append("the model's answer was unparseable; the best valid candidate by the deterministic ranking is used")
        if choice is not None:
            model_choice = choice.chosen
            fallback = (f"the best valid candidate {best_valid!r} is recommended instead" if best_valid
                        else "no valid candidate exists")
            if choice.chosen is not None and choice.chosen not in by_name:
                flags.append(f"the model chose {choice.chosen!r}, which was not evaluated; {fallback}")
                needs_review = True
            elif choice.chosen is not None and not by_name[choice.chosen]["valid"]:
                reasons = "; ".join(v["message"] for v in by_name[choice.chosen]["violations"])
                flags.append(f"the model chose {choice.chosen!r}, which violates constraints ({reasons}); {fallback}")
                needs_review = True
            elif choice.chosen is not None:
                recommended = choice.chosen
            verdicts = validate_claims([Claim(c.text, "calculated", calc_run_ids=tuple(c.evidence_ids))
                                        for c in choice.claims], [], {run_id: out})
            for c, v in zip(choice.claims, verdicts, strict=True):
                if v["verdict"] == "supported":
                    kept.append({"text": c.text, "evidence_ids": c.evidence_ids, "agent": "supervisor"})
                else:
                    removed.append(f"claim removed: {v['reason']}")
                    needs_review = True
    if best_valid is None:
        needs_review = True
        flags.append("no candidate plan satisfies every constraint; nothing is recommended until the household reviews them")
    elif recommended != best_valid:
        flags.append(f"the model's valid choice {recommended!r} differs from the top-ranked valid candidate {best_valid!r}")

    statements = ([{"text": by_name[recommended]["rationale"], "evidence_ids": [run_id], "agent": "supervisor"}]
                  if recommended else []) + kept
    return {**base, "status": "needs_review" if needs_review else "recommended", "needs_review": needs_review,
            "run_id": run_id, "evidence_id": run_id, "extra_amount_minor": extra,
            "extra_amount_display": out["extra_amount_display"], "recommended": recommended,
            "ranked_first": out["ranking"][0], "best_valid": best_valid, "model_choice": model_choice,
            "candidates": [_row(c) for c in out["candidates"]], "assumptions": out["assumptions"],
            "input_sources": result["input_sources"], "ledger_excluded_accounts": result["ledger_excluded_accounts"],
            "provisional": result["provisional"], "flags": flags + removed, "statements": statements}
