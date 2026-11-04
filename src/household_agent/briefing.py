"""Draft-only monthly briefing (design §10, §12 step 6).

``draft_monthly_briefing`` builds the seven §10 sections for one member and one month. It is
deterministic: it uses only

- ledger data the member may read for ``monthly_briefing`` (coverage metadata, verified
  balances, verified transactions),
- calculation runs it records itself (``cash_flow``, ``category_totals``),
- validated Supervisor results (claims, questions, threshold findings), and
- an optional §5.8 plan from ``plan_from_ledger``.

No model writes any part of it. Every item that carries an amount must cite evidence (the
schema rejects one that doesn't), amounts stay integer cents, and unreconciled statements
contribute coverage notes but never numbers. The briefing is a draft: nothing is sent and no
money moves.
"""

import calendar
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from household_agent.agents import AgentResult
from household_agent.authz import Actor
from household_agent.calc_runner import CalculationRunner
from household_agent.categorize import UNCATEGORIZED
from household_agent.ledger import LedgerStore
from household_agent.models import Action
from household_agent.plan_inputs import DerivedInputs
from household_agent.planning import PlanResult
from household_agent.reconcile import CONFIDENCE_THRESHOLD
from household_finance.money import format_minor

BRIEFING_VERSION = "monthly_briefing_1.0"
PURPOSE = "monthly_briefing"
LARGE_TXN_MINOR = 250_000  # $2,500: non-routine movements at or above this are material events
UNMAPPED_MIN_MINOR = 50_000  # $500: uncategorized items at or above this are material events
BASELINE_DAYS = 90  # merchants seen in this look-back are not "new" for the cluster rules
ROUTINE_CATEGORIES = frozenset({"salary", "mortgage", "rent", "rental_income", "transfer"})
DUPLICATE_CODES = frozenset({"possible_duplicate", "duplicate_of_ingested"})
DISCLAIMER = ("This is a draft for the household. It is informational only, not financial, legal or tax "
              "advice, and nothing in it moves money or contacts anyone.")
SECTIONS: tuple[tuple[str, str], ...] = (
    ("data_coverage", "Data Coverage"),
    ("net_worth_and_cash_flow", "Net Worth and Cash Flow"),
    ("budget_and_variance", "Budget and Variance"),
    ("material_events", "Material Events"),
    ("policy_and_deadlines", "Policy and Deadline Review"),
    ("actions_for_approval", "Actions for Approval"),
    ("assumptions_and_limitations", "Assumptions and Limitations"),
)


class BriefingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    text: str
    amounts_minor: dict[str, int] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    provisional: bool = False

    @model_validator(mode="after")
    def _amounts_need_evidence(self) -> "BriefingItem":
        if self.amounts_minor and not self.evidence_ids:
            raise ValueError("an item with amounts must cite evidence")
        return self


class MonthlyBriefing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    briefing_version: Literal["monthly_briefing_1.0"] = BRIEFING_VERSION
    household_id: str
    prepared_for: str
    month: str = Field(pattern=r"^\d{4}-\d{2}$")
    correlation_id: str
    draft: Literal[True] = True
    external_action: Literal["none"] = "none"
    data_coverage: list[BriefingItem]
    net_worth_and_cash_flow: list[BriefingItem]
    budget_and_variance: list[BriefingItem]
    material_events: list[BriefingItem]
    policy_and_deadlines: list[BriefingItem]
    actions_for_approval: list[BriefingItem]
    assumptions_and_limitations: list[BriefingItem] = Field(min_length=1)
    calc_run_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _disclaimer_last(self) -> "MonthlyBriefing":
        if self.assumptions_and_limitations[-1].kind != "disclaimer":
            raise ValueError("the briefing must end with the disclaimer")
        return self


def _month_bounds(month: date) -> tuple[date, date]:
    return month.replace(day=1), month.replace(day=calendar.monthrange(month.year, month.month)[1])


def _plan_items(result: PlanResult, derived: DerivedInputs, run_id: str) -> tuple[list[BriefingItem], list[BriefingItem]]:
    """(actions, assumptions) for a §5.8 scenario: best plan, runner-up, scores, assumptions."""
    evidence = [run_id, *([derived.inputs.limits_evidence_id] if derived.inputs.limits_evidence_id else [])]
    if result.best is None:
        return [BriefingItem(kind="allocation_scenario", text=f"Allocation planner: {result.status}. {result.message}",
                             evidence_ids=[run_id])], []
    s = result.best.score
    actions = [BriefingItem(
        kind="allocation_scenario", amounts_minor=dict(sorted(result.best.allocation.items())), evidence_ids=evidence,
        provisional=not derived.complete,
        text=(f"Proposed scenario {result.best.node_id} ({result.status}). Scores: tax {s.tax_efficiency}, "
              f"liquidity {s.liquidity_safety}, yield {s.expected_net_yield}, composite {s.composite}."))]
    runner_up = next((n for n in result.ranked_leaves if n.allocation != result.best.allocation), None)
    if runner_up is not None:
        actions.append(BriefingItem(
            kind="allocation_runner_up", amounts_minor=dict(sorted(runner_up.allocation.items())),
            evidence_ids=evidence, provisional=not derived.complete,
            text=f"Runner-up {runner_up.node_id}, composite {runner_up.score.composite}."))
    i = derived.inputs
    assumptions = [BriefingItem(
        kind="planning_assumptions", evidence_ids=evidence,
        amounts_minor={"required_reserve": i.required_reserve_minor, "surplus": i.surplus_minor},
        text=(f"Planner assumptions: {i.reserve_months}-month reserve plus known liabilities, buffer target "
              f"{i.buffer_target_pct}%, {i.horizon_years}-year horizon, taxable drag {i.taxable_drag}, real returns "
              + ", ".join(f"{k} {v}" for k, v in sorted(i.real_return.items())) + "."))]
    if not derived.complete:
        assumptions.append(BriefingItem(
            kind="planning_incomplete", text=f"The scenario leaves out {len(derived.excluded_accounts)} account(s) "
                                             "you can't read or that lack a verified statement, so it is provisional."))
    return actions, assumptions


def draft_monthly_briefing(
    store: LedgerStore,
    runner: CalculationRunner,
    actor: Actor,
    *,
    month: date,
    as_of: datetime,
    correlation_id: str,
    agent_results: Sequence[AgentResult | Mapping[str, Any]] = (),
    findings: Sequence[Mapping[str, Any]] = (),
    plan: tuple[PlanResult, DerivedInputs, str] | None = None,
    approval_thread_id: str | None = None,
) -> MonthlyBriefing:
    start, end = _month_bounds(month)
    label = start.strftime("%Y-%m")
    s: dict[str, list[BriefingItem]] = {key: [] for key, _ in SECTIONS}
    run_ids: list[str] = []

    readable, not_shared = [], 0
    for account in store.authorizer.accounts(actor.household_id):
        if store.authorizer.authorize(actor, account.account_id, Action.READ, PURPOSE, as_of):
            readable.append(account.account_id)
        else:
            not_shared += 1

    # 1. Data coverage (status only; unreconciled statements never contribute numbers)
    docs = store.document_coverage(actor, PURPOSE, correlation_id, readable, start, end, at=as_of) if readable else []
    by_account: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in docs:
        by_account[d["account_id"]].append(d)
    for account_id in readable:
        if not by_account[account_id]:
            s["data_coverage"].append(BriefingItem(kind="missing_statement",
                                                   text=f"{account_id}: no statement received for {label}."))
            s["assumptions_and_limitations"].append(BriefingItem(
                kind="missing_source", text=f"{account_id} has no statement for {label}; its figures are absent."))
        for d in by_account[account_id]:
            verified = d["status"] == "reconciled" and d["document_confidence"] >= CONFIDENCE_THRESHOLD
            codes = sorted({i["code"] for i in d["issues"]})
            s["data_coverage"].append(BriefingItem(
                kind="statement", evidence_ids=[d["document_id"]], provisional=not verified,
                text=(f"{account_id}: {d['source_name']} ({d['period_start']} to {d['period_end']}) is {d['status']}, "
                      f"confidence {d['document_confidence']:.2f}" + (f"; issues: {', '.join(codes)}." if codes else "."))))
            if d["status"] == "unreconciled":
                s["assumptions_and_limitations"].append(BriefingItem(
                    kind="unreconciled_excluded", evidence_ids=[d["document_id"]],
                    text=f"{d['source_name']} did not reconcile and is excluded from every figure."))
            elif d["status"] == "qualified":
                s["actions_for_approval"].append(BriefingItem(
                    kind="confirm_statement", evidence_ids=[d["document_id"]], provisional=True,
                    text=f"Please confirm {d['source_name']}: {', '.join(codes) or 'needs review'}."))
            for issue in d["issues"]:
                if issue["code"] in DUPLICATE_CODES:
                    s["material_events"].append(BriefingItem(
                        kind="possible_duplicate", evidence_ids=[d["document_id"]], provisional=True,
                        amounts_minor={"amount": issue["amount_minor"]} if issue.get("amount_minor") is not None else {},
                        text=f"Possible duplicate in {d['source_name']}: {issue['message']}"))
    if not_shared:
        s["data_coverage"].append(BriefingItem(
            kind="not_shared", text=f"{not_shared} household account(s) are not shared with you for briefings."))

    # 2-4. Balances, cash flow, categories, large movements (verified data only)
    if readable:
        for st in store.verified_statements(actor, PURPOSE, correlation_id, readable, start, end, at=as_of):
            s["net_worth_and_cash_flow"].append(BriefingItem(
                kind="closing_balance", evidence_ids=[st["document_id"]],
                amounts_minor={"opening": st["opening_balance_minor"], "closing": st["closing_balance_minor"]},
                text=f"{st['account_id']} balance on {st['period_end']}."))
        verified = store.query_transactions(actor, PURPOSE, correlation_id, readable, start, end, at=as_of)
        everything = store.query_transactions(actor, PURPOSE, correlation_id, readable, start, end,
                                              include_provisional=True, at=as_of)
        params = {"start": start.isoformat(), "end": end.isoformat()}
        if verified:
            common = dict(actor_id=actor.member_id, household_id=actor.household_id, correlation_id=correlation_id,
                          params=params, acting_agent="briefing")
            flow = runner.run("cash_flow", verified, **common)
            cats = runner.run("category_totals", verified, **common)
            run_ids += [flow.run_id, cats.run_id]
            s["net_worth_and_cash_flow"].append(BriefingItem(
                kind="cash_flow", evidence_ids=[flow.run_id], text=f"Verified cash flow for {label}.",
                amounts_minor={"inflow": flow.output["inflow_minor"], "outflow": flow.output["outflow_minor"],
                               "net": flow.output["net_minor"]}))
            s["budget_and_variance"].append(BriefingItem(
                kind="category_totals", evidence_ids=[cats.run_id], amounts_minor=dict(cats.output),
                text=f"Verified totals by category for {label}."))
        verified_ids = {t.transaction_id for t in verified}
        provisional = [t for t in everything if t.transaction_id not in verified_ids]
        if provisional:
            s["budget_and_variance"].append(BriefingItem(
                kind="provisional_transactions", provisional=True, evidence_ids=[t.transaction_id for t in provisional],
                text=f"{len(provisional)} transaction(s) from statements awaiting confirmation are left out of the totals."))
        for t in everything:
            kind = ("unmapped_high_value" if t.category == UNCATEGORIZED and abs(t.amount_minor) >= UNMAPPED_MIN_MINOR
                    else "large_transaction" if t.category not in ROUTINE_CATEGORIES
                    and abs(t.amount_minor) >= LARGE_TXN_MINOR else None)
            if kind:
                s["material_events"].append(BriefingItem(
                    kind=kind, provisional=t.transaction_id not in verified_ids,
                    evidence_ids=[t.transaction_id, t.source_document_id], amounts_minor={"amount": t.amount_minor},
                    text=f"{t.posted_at} {t.merchant_normalized} ({t.category})"))
        if everything:
            # Cluster rules (design §5.5): new unmapped small charges that each stay under the limits above
            history = store.query_transactions(actor, PURPOSE, correlation_id, readable,
                                               start - timedelta(days=BASELINE_DAYS), start - timedelta(days=1),
                                               include_provisional=True, at=as_of)
            risk = runner.run("cluster_risk", [*history, *everything], actor_id=actor.member_id,
                              household_id=actor.household_id, correlation_id=correlation_id, params=params,
                              acting_agent="briefing", allow_provisional=True)
            run_ids.append(risk.run_id)
            for a in risk.output:
                s["material_events"].append(BriefingItem(
                    kind=a["rule"], evidence_ids=[risk.run_id, *a["transaction_ids"]],
                    provisional=any(i not in verified_ids for i in a["transaction_ids"]),
                    amounts_minor={"total": a["total_minor"]}, text=f"{a['explanation']}; please review."))

    # Supervisor results: validated claims, open questions and threshold findings
    for raw in agent_results:
        r = raw if isinstance(raw, AgentResult) else AgentResult.model_validate(raw)
        section = "policy_and_deadlines" if r.agent == "policy_compliance" else "budget_and_variance"
        for c in r.claims:
            s[section].append(BriefingItem(kind="validated_claim", text=c.text, evidence_ids=list(c.evidence_ids)))
        for q in r.questions:
            s["policy_and_deadlines"].append(BriefingItem(kind="unverified_question", text=q))
        if r.status != "completed":
            s["assumptions_and_limitations"].append(BriefingItem(
                kind="agent_needs_review", text=f"{r.agent} finished with status {r.status}."))
    for f in findings:
        s["material_events"].append(BriefingItem(
            kind=str(f.get("type", "finding")), evidence_ids=[str(f["calc_run_id"])] if f.get("calc_run_id") else [],
            text=(f"Change of {f['pct_variance']}% is above the {f['threshold_pct']}% review threshold."
                  if f.get("type") == "material_variance" else f"Finding: {f.get('type', 'unspecified')}.")))

    # 6. Actions for approval
    if plan is not None:
        actions, assumptions = _plan_items(*plan)
        s["actions_for_approval"] += actions
        s["assumptions_and_limitations"] += assumptions
    if approval_thread_id:
        s["actions_for_approval"].append(BriefingItem(
            kind="approval_pending", evidence_ids=[approval_thread_id],
            text="The allocation scenario is waiting for a household admin or adult member to approve or reject it."))

    # 7. Limitations, ending with the disclaimer
    s["assumptions_and_limitations"].append(BriefingItem(
        kind="net_worth_scope", text="Balances cover statement accounts only; property, loans and investments are "
                                     "not in the ledger yet, so no household net-worth total is shown."))
    s["assumptions_and_limitations"].append(BriefingItem(kind="disclaimer", text=DISCLAIMER))

    briefing = MonthlyBriefing(household_id=actor.household_id, prepared_for=actor.member_id, month=label,
                               correlation_id=correlation_id, calc_run_ids=run_ids, **s)
    store.audit.append(
        correlation_id=correlation_id, actor_id=actor.member_id, household_id=actor.household_id,
        action="compose_briefing", purpose=PURPOSE, decision="info", resource=f"briefing:{label}",
        acting_agent="supervisor", at=as_of,
        details={"version": BRIEFING_VERSION, "calc_run_ids": run_ids, "accounts": readable,
                 "not_shared": not_shared, "items": {k: len(v) for k, v in s.items()}},
    )
    return briefing


def render_markdown(briefing: MonthlyBriefing) -> str:
    lines = [f"# Monthly briefing {briefing.month} (draft)", "",
             f"Prepared for {briefing.prepared_for} · household {briefing.household_id} · "
             f"correlation {briefing.correlation_id}", ""]
    for key, title in SECTIONS:
        lines += [f"## {title}", ""]
        items: list[BriefingItem] = getattr(briefing, key)
        if not items:
            lines.append("- Nothing to report.")
        for it in items:
            amounts = ", ".join(f"{k} {format_minor(v)}" for k, v in it.amounts_minor.items())
            line = f"- {it.text.rstrip('.')}: {amounts}." if amounts else f"- {it.text}"
            if it.provisional:
                line += " *(provisional)*"
            if it.evidence_ids:
                line += f" [evidence: {', '.join(it.evidence_ids)}]"
            lines.append(line)
        lines.append("")
    return "\n".join(lines)
