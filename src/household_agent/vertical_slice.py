"""Vertical-slice exit run (design §12, "Vertical-Slice Exit Criteria").

``python -m household_agent.vertical_slice`` runs one synthetic job under a single
correlation ID and turns each exit criterion into a ``Check`` with evidence IDs:

- ``parse -> authorize -> reconcile -> calculate -> retrieve -> briefing`` across the
  Supervisor, Ledger Analyst and Policy & Compliance agents, traced from each statement's
  source hash to the audit events of the job;
- an authorization denial, a consent-revocation denial, an unreconciled-data block and a
  stale-source escalation;
- an interrupt-and-resume approval, resumed from a reopened checkpoint database, that does
  not run the completed planning step again;
- a §5.8 planning run (same input hash, same tree), a hard-pruned cap breach and a critic
  discrepancy routed to review;
- no external action is available.

By default the agents use ``PlaybackChatModel``: scripted tool calls that stand in for a
model, so the run works offline. ``--model ollama:qwen2.5:7b`` uses a real model for the
router and both specialists; the checks test system guarantees, not model wording.
"""

import argparse
import json
import re
import sqlite3
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from household_agent.agents import (
    _EXTERNAL,
    LEDGER_TOOLS,
    POLICY_TOOLS,
    AgentResult,
    LedgerAnalyst,
    PolicyCompliance,
    Router,
    Supervisor,
)
from household_agent.approval import build_approval_graph
from household_agent.audit import AuditLog
from household_agent.authz import AuthorizationError, Authorizer
from household_agent.briefing import DISCLAIMER, MonthlyBriefing, draft_monthly_briefing, render_markdown
from household_agent.calc_runner import CalculationRunner, IneligibleInputError
from household_agent.categorize import Categorizer
from household_agent.consent import ConsentStore
from household_agent.ledger import LedgerStore
from household_agent.mcp_server import MEMBERS
from household_agent.models import Action
from household_agent.pipeline import IngestResult, ingest_statement
from household_agent.plan_inputs import Liability, derive_planning_inputs, ledger_inputs_for, plan_from_ledger
from household_agent.planning import TemplateThoughtGenerator, Thought, inputs_hash, plan_allocation
from household_agent.retrieval import CANNOT_VERIFY, Claim, RetrievalService, validate_claims
from household_agent.synthetic import (
    ACCOUNTS,
    ALEX,
    CASEY_CPA,
    DEMO_AS_OF,
    DEMO_OCTOBERS,
    JORDAN,
    RULES,
    SAM,
    cpa_grant,
    demo_planning_inputs,
    demo_policy_documents,
    demo_statements,
)
from household_agent.tools import TOOL_NAMES
from household_finance.allocation import AllocationStep

SLICE_VERSION = "vertical_slice_1.0"
TAX = "San Mateo County Tax Collector"
RENTAL = "acct_rental_checking"
TRACE_REQUEST = ("I paid $3,200 to the San Mateo County Tax Collector in October. Is this unusual compared "
                 "with last year, and how should it be accounted for?")
PLAN_REQUEST = "We have a surplus this month; how should we allocate it?"
EXTERNAL_REQUEST = "Please transfer $5,000 from checking to the brokerage."
OCTOBER = date(2026, 10, 1)
EXPECTED_STATUS = {
    "2025_10_rental_checking.csv": "reconciled",
    "2026_10_rental_checking.csv": "reconciled",
    "2026_10_alex_checking.csv": "reconciled",
    "2026_10_jordan_card.csv": "qualified",  # duplicate Safeway charge
    "2026_09_alex_checking_corrupt.csv": "unreconciled",
}
REQUIRED_JOB_ACTIONS = frozenset({"ingest_document", Action.INDEX.value, Action.READ.value, Action.RETRIEVE.value,
                                  "calculate", "route", "dispatch", "threshold_check", "compose_briefing"})


@dataclass(frozen=True)
class Check:
    name: str
    criterion: str
    passed: bool
    detail: str
    evidence_ids: tuple[str, ...] = ()


@dataclass
class SliceReport:
    version: str
    mode: str
    correlation_id: str
    approval_thread_id: str
    checks: list[Check] = field(default_factory=list)
    briefing_markdown: str = ""
    audit_events: int = 0

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "passed": self.passed}, indent=2, default=str)


# --- offline stand-in for a model ----------------------------------------------------------

class PlaybackChatModel(BaseChatModel):
    """Plays back scripted steps; a step is an AIMessage or a function of the conversation."""

    steps: list[Any]
    i: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.i >= len(self.steps):
            msg = AIMessage(content='{"claims": []}')
        else:
            step = self.steps[self.i]
            msg = (step(messages) if callable(step) else step).model_copy(update={"id": None})
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "playback"


def _call(name: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}_{len(args)}"}])


def _final(*claims: dict[str, Any]) -> AIMessage:
    return AIMessage(content=json.dumps({"claims": list(claims)}))


def _results(messages, name: str) -> list[dict[str, Any]]:
    return [json.loads(m.content) for m in messages if isinstance(m, ToolMessage) and m.name == name]


def _txn_ids(messages) -> list[str]:
    return [t["transaction_id"] for r in _results(messages, "query_ledger") for t in r.get("transactions", [])]


def _chunk_with(messages, needle: str) -> str:
    return next((p["chunk_id"] for r in _results(messages, "retrieve_verified") for p in r.get("passages", [])
                 if needle in p["text"]), "missing")


def ledger_playback() -> PlaybackChatModel:
    """The tool calls a Ledger Analyst model makes for the §9 property-tax trace."""
    return PlaybackChatModel(steps=[
        _call("query_ledger", account_ids=[RENTAL], start="2025-10-01", end="2026-10-31", merchant=TAX),
        lambda m: _call("run_calculation", fn="variance_between_transactions", transaction_ids=_txn_ids(m)),
        lambda m: _final({"text": "October 2026 payment is $450.00 (16.36%) higher than October 2025.",
                          "evidence_ids": [_results(m, "run_calculation")[-1].get("run_id", "missing"), *_txn_ids(m)]}),
    ])


def policy_playback() -> PlaybackChatModel:
    """The tool calls a Policy & Compliance model makes; its 2026 tax claim has no 2026 source."""
    return PlaybackChatModel(steps=[
        _call("retrieve_verified", query="Monthly rent for Unit B lease", collections=["policies_compliance"]),
        _call("retrieve_verified", query="rental property tax deductible Schedule E",
              collections=["policies_compliance"], tax_years=[2026]),
        lambda m: _final(
            {"text": "Monthly rent for Unit B is $3,400.00.", "evidence_ids": [_chunk_with(m, "3,400.00")]},
            {"text": "Property tax on Unit B is deductible on Schedule E.", "kind": "tax", "tax_year": 2026,
             "evidence_ids": [_chunk_with(m, "3,400.00")]},
        ),
    ])


# --- planner stand-ins for misbehaving model nodes ------------------------------------------

class OverCapPersona(TemplateThoughtGenerator):
    """A persona node that ignores statutory room: puts all tax-advantaged money in the IRA."""

    version = "overcap_persona_slice"

    def propose(self, inputs, node, tier):
        thoughts = list(super().propose(inputs, node, tier))
        if tier == 2:
            thoughts[2] = Thought("tax_aggressive", AllocationStep(2, Decimal(1), (("ira", Decimal(1)),)),
                                  "Put everything in the IRA.")
        return thoughts


class CashLovingCritic:
    """A critic that strongly prefers the conservative cash buffer, disagreeing with the score."""

    def score(self, inputs, node):
        return Decimal("0.99") if node.node_id.startswith("d1.conservative") else node.score.composite


# --- the run ---------------------------------------------------------------------------------

@dataclass
class _Env:
    workdir: Path
    conn: sqlite3.Connection
    store: LedgerStore
    consent: ConsentStore
    runner: CalculationRunner

    @classmethod
    def open(cls, workdir: Path) -> "_Env":
        # specialists run on worker threads; AuditLog/LedgerStore/ConsentStore serialize with one shared lock
        conn = sqlite3.connect(workdir / "ledger.db", check_same_thread=False)
        authorizer, audit = Authorizer(ACCOUNTS), AuditLog(conn)
        consent = ConsentStore(conn, authorizer, audit)  # reloads persisted grants and revocations
        store = LedgerStore(conn, authorizer, audit)
        return cls(workdir, conn, store, consent, CalculationRunner(store))


def _planner_kwargs() -> dict[str, Any]:
    d = demo_planning_inputs()
    return {"contribution_room_minor": d.contribution_room_minor, "limits_evidence_id": d.limits_evidence_id,
            "real_return": d.real_return, "liabilities": [Liability("county_bill_2026", 320000, "2nd installment")],
            "reserve_months": 3}


def _approval_graph(env: _Env, ckpt: sqlite3.Connection, correlation_id: str):
    inputs_for = ledger_inputs_for(env.store, env.runner, ALEX, clock=lambda: DEMO_AS_OF,
                                   correlation_id=correlation_id, **_planner_kwargs())
    return build_approval_graph(inputs_for=inputs_for, members=MEMBERS, audit=env.store.audit,
                                checkpointer=SqliteSaver(ckpt))


def run_vertical_slice(
    workdir: Path,
    *,
    models: Callable[[str], BaseChatModel] | None = None,
    embeddings: Embeddings | None = None,
    correlation_id: str = "corr_vertical_slice",
) -> SliceReport:
    """Run the slice in ``workdir`` (ledger.db, checkpoints.db, chroma/). ``models(role)`` returns
    the chat model for ``router``, ``ledger_analyst`` or ``policy_compliance``; ``None`` = playback."""
    from household_agent.llm import HashEmbeddings

    workdir.mkdir(parents=True, exist_ok=True)
    corr, at = correlation_id, DEMO_AS_OF
    env = _Env.open(workdir)
    store, runner = env.store, env.runner
    categorizer = Categorizer(RULES)
    checks: list[Check] = []

    def check(name: str, criterion: str, passed: bool, detail: str, evidence: Sequence[str] = ()) -> None:
        checks.append(Check(name, criterion, bool(passed), detail, tuple(evidence)))

    # 1. parse -> authorize -> reconcile (each owner uploads their own statements)
    ingested: dict[str, IngestResult] = {}
    for name, content in demo_statements().items():
        actor = JORDAN if "jordan_card" in name else ALEX
        ingested[name] = ingest_statement(store, categorizer, actor, content.encode(), name,
                                          purpose="statement_ingestion", correlation_id=corr, at=at)
    got = {n: r.status.value for n, r in ingested.items()}
    check("ingest_and_reconcile", "parse -> authorize -> reconcile", got == EXPECTED_STATUS,
          ", ".join(f"{n}={s}" for n, s in got.items()), [r.document_id for r in ingested.values()])

    # 2. index the synthetic lease, county bills and policy sources (authorized per document)
    client = chromadb.PersistentClient(str(workdir / "chroma"), settings=Settings(anonymized_telemetry=False))
    retrieval = RetrievalService(client, embeddings or HashEmbeddings(), store.authorizer, store.audit)
    indexed, refused = [], []
    for doc in demo_policy_documents():
        indexed += retrieval.index(ALEX, doc, correlation_id=corr, at=at)
    try:
        retrieval.index(SAM, demo_policy_documents()[0], correlation_id=corr, at=at)
    except AuthorizationError as exc:
        refused.append(str(exc))

    # 3. the three-agent job (Supervisor -> Ledger Analyst -> Policy & Compliance)
    mode = "offline playback models" if models is None else "live models"
    router_model = models("router") if models else None  # None = keyword router
    specialists = {
        "ledger_analyst": LedgerAnalyst(models("ledger_analyst") if models else ledger_playback(),
                                        store, categorizer, ALEX, clock=lambda: at),
        "policy_compliance": PolicyCompliance(models("policy_compliance") if models else policy_playback(),
                                              retrieval, ALEX, clock=lambda: at),
    }
    job = Supervisor(ALEX, Router(router_model), specialists, store.audit, clock=lambda: at).run(
        TRACE_REQUEST, correlation_id=corr)
    results = [AgentResult.model_validate(r) for r in job.get("results", [])]
    by_agent = {r.agent: r for r in results}
    sup_actions = [e["action"] for e in store.audit.events(corr) if e["acting_agent"] == "supervisor"]
    ledger = by_agent.get("ledger_analyst")
    calc_run_id = (ledger.outputs.get("calc_run_id") if ledger else None) or ""
    statements = job.get("briefing", {}).get("statements", [])
    check("three_agent_trace", "Supervisor, Ledger Analyst and Policy & Compliance on one job",
          job.get("routing", {}).get("intent") == "historical_comparison_and_tax_context"
          and set(by_agent) == {"ledger_analyst", "policy_compliance"}
          and all(r.status != "failed" for r in results) and bool(calc_run_id)
          and sup_actions[:1] == ["route"] and sup_actions.count("dispatch") == 2
          and sup_actions[-1] == "compose_briefing" and all(s["evidence_ids"] for s in statements),
          f"{mode}; intent={job.get('routing', {}).get('intent')}; "
          + "; ".join(f"{r.agent}={r.status}" for r in results) + f"; supervisor steps={sup_actions}",
          [calc_run_id, *(i for r in results for i in r.evidence_ids)])

    # 4. provenance: calculation -> transactions -> source documents (hashes) -> audit events
    row = env.conn.execute("SELECT input_ids_json, input_hash, correlation_id FROM calculation_runs WHERE run_id = ?",
                           (calc_run_id,)).fetchone()
    txn_ids = json.loads(row[0]) if row else []
    docs = env.conn.execute(
        f"SELECT DISTINCT d.document_id, d.source_hash FROM transactions t JOIN documents d "
        f"ON d.document_id = t.source_document_id WHERE t.transaction_id IN ({','.join('?' * len(txn_ids))})",
        txn_ids).fetchall() if txn_ids else []
    events = store.audit.events(corr)
    resources = {(e["action"], e["resource"]) for e in events}
    traced = (bool(row) and row[2] == corr and bool(docs) and all(h.startswith("sha256:") for _, h in docs)
              and ("calculate", f"calc:{calc_run_id}") in resources
              and all(("ingest_document", f"document:{d}") in resources for d, _ in docs))
    check("provenance_chain", "source hash -> calculation -> audit correlation ID", traced,
          f"{calc_run_id} <- {len(txn_ids)} transactions <- " + ", ".join(f"{d} ({h[:15]}...)" for d, h in docs)
          + f"; correlation_id={corr}", [calc_run_id, *txn_ids, *(d for d, _ in docs)])

    # 5. authorization denial: Jordan asks for Alex's private checking
    try:
        store.query_transactions(JORDAN, "budgeting", corr, ["acct_alex_checking"], *DEMO_OCTOBERS, at=at)
        denied = ""
    except AuthorizationError as exc:
        denied = str(exc)
    deny_events = [e for e in store.audit.events(corr) if e["decision"] == "deny"
                   and e["actor_id"] == "jordan" and e["resource"] == "account:acct_alex_checking"]
    check("authorization_denial", "authorization denial", bool(denied) and bool(deny_events) and bool(refused),
          f"jordan -> acct_alex_checking: {denied or 'ALLOWED'}; sam indexing: {refused[0] if refused else 'ALLOWED'}",
          [e["event_id"] for e in deny_events])

    # 6. consent revocation: both owners let the CPA read the rental account; Jordan revokes
    g = cpa_grant(at + timedelta(days=180))
    env.consent.grant(ALEX, g, correlation_id=corr, at=at)
    env.consent.grant(JORDAN, replace(g, grant_id="grant_cpa_rental_jordan", grantor_id="jordan", version="cv_2"),
                      correlation_id=corr, at=at)
    before = len(store.query_transactions(CASEY_CPA, "tax_preparation", corr, [RENTAL], *DEMO_OCTOBERS, at=at))
    revoked_at = at + timedelta(minutes=5)
    env.consent.revoke(JORDAN, "grant_cpa_rental_jordan", correlation_id=corr, at=revoked_at)
    try:
        store.query_transactions(CASEY_CPA, "tax_preparation", corr, [RENTAL], *DEMO_OCTOBERS,
                                 at=revoked_at + timedelta(seconds=1))
        after = "ALLOWED"
    except AuthorizationError as exc:
        after = str(exc)
    check("consent_revocation_denial", "consent-revocation denial", before > 0 and after != "ALLOWED",
          f"CPA read {before} rental transactions with both owners' consent; after Jordan revoked: {after}",
          ["grant_cpa_rental", "grant_cpa_rental_jordan"])

    # 7. unreconciled data is blocked from every calculation
    corrupt = ingested["2026_09_alex_checking_corrupt.csv"]
    sept = store.query_transactions(ALEX, "budgeting", corr, ["acct_alex_checking"], date(2026, 9, 1),
                                    date(2026, 9, 30), include_provisional=True, at=at)
    raw_ids = [r[0] for r in env.conn.execute("SELECT transaction_id FROM transactions WHERE source_document_id = ?",
                                              (corrupt.document_id,))]
    try:
        runner.run("cash_flow", [store._latest(t) for t in raw_ids], actor_id="alex", household_id=ALEX.household_id,
                   correlation_id=corr, params={"start": "2026-09-01", "end": "2026-09-30"})
        blocked = ""
    except IneligibleInputError as exc:
        blocked = str(exc)
    sept_brief = draft_monthly_briefing(store, runner, ALEX, month=date(2026, 9, 1), as_of=at, correlation_id=corr)
    check("unreconciled_block", "unreconciled-data block",
          corrupt.status.value == "unreconciled" and sept == [] and bool(raw_ids) and bool(blocked)
          and sept_brief.net_worth_and_cash_flow == [] and sept_brief.calc_run_ids == [],
          f"{corrupt.source_name}: queries return {len(sept)} rows; forcing {len(raw_ids)} rows into cash_flow: "
          f"{blocked or 'ALLOWED'}; September briefing has no figures", [corrupt.document_id])

    # 8. stale sources are escalated, never cited
    old = retrieval.retrieve_verified(ALEX, "tax_preparation", "2024 county tax bill first installment parcel",
                                      correlation_id=corr, at=at, collections=["policies_compliance"])
    irs = retrieval.retrieve_verified(ALEX, "tax_preparation", "rental property tax deductible Schedule E",
                                      correlation_id=corr, at=at, collections=["policies_compliance"])
    passages = {p["chunk_id"]: p for p in old["passages"] + irs["passages"]}
    expired = [i for i, p in passages.items() if p["verification"] == "expired"]
    irs_ids = [i for i, p in passages.items() if p["metadata"]["doc_id"] == "irs_rental_2025"]
    verdicts = validate_claims([
        Claim("The first installment was 2,600.00 dollars.", "fact", tuple(expired[:1])),
        Claim("Property tax on Unit B is deductible on Schedule E.", "tax", tuple(irs_ids[:1]), tax_year=2026),
    ], list(passages.values()))
    policy = by_agent.get("policy_compliance")
    check("stale_source_escalation", "stale-source escalation",
          bool(expired) and bool(irs_ids) and [v["verdict"] for v in verdicts] == ["rephrase_as_question"] * 2
          and verdicts[1]["message"] == CANNOT_VERIFY,
          f"expired: {expired}; 2025-only IRS source for a 2026 claim -> {verdicts[1]['verdict'] if verdicts else '-'}; "
          f"agent questions: {policy.questions if policy else []}", [*expired, *irs_ids])

    # 9. §5.8 planning: reproducible tree, hard-pruned cap breach, critic discrepancy
    plan = plan_from_ledger(store, runner, ALEX, as_of=at, correlation_id=corr, **_planner_kwargs())
    result, derived, plan_run_id = plan
    again = plan_allocation(derived.inputs)
    rederived = derive_planning_inputs(store, runner, ALEX, as_of=at, correlation_id=corr, **_planner_kwargs())
    tree = lambda r: [(n.node_id, n.status, str(n.score.composite)) for n in r.trace]  # noqa: E731
    check("planning_reproducible", "same input hash -> same tree",
          result.status == "awaiting_approval" and again.input_hash == result.input_hash == inputs_hash(rederived.inputs)
          and tree(again) == tree(result) and result.evaluations == 15,
          f"input_hash={result.input_hash[:23]}...; {result.evaluations} nodes; best={result.best.node_id if result.best else None}; "
          f"provisional={not derived.complete} (excluded: {sorted(derived.excluded_accounts)})", [plan_run_id])

    capped = plan_allocation(derived.inputs, OverCapPersona())
    pruned = [n for n in capped.trace if n.status == "hard_pruned"]
    codes = {v.code for n in pruned for v in n.score.violations}
    check("cap_breach_hard_pruned", "hard-pruned cap breach",
          bool(pruned) and codes == {"statutory_cap_exceeded"} and capped.best is not None
          and not any("d2.tax_aggressive" in n.node_id for n in capped.ranked_leaves),
          f"{len(pruned)} branch(es) hard-pruned ({sorted(codes)}); best={capped.best.node_id if capped.best else None}",
          [n.node_id for n in pruned])

    reviewed = plan_allocation(derived.inputs, critic=CashLovingCritic())
    check("critic_discrepancy_review", "critic discrepancy routed to review",
          reviewed.status == "needs_review" and bool(reviewed.discrepancy_node_ids)
          and {n.node_id for n in result.ranked_leaves} <= {n.node_id for n in reviewed.ranked_leaves},
          f"status={reviewed.status}; discrepancies={list(reviewed.discrepancy_node_ids)}; "
          f"{reviewed.evaluations} nodes (critic kept branches, removed none)", list(reviewed.discrepancy_node_ids))

    # 10. interrupt: the allocation request pauses for approval (checkpoint on disk)
    ckpt = sqlite3.connect(workdir / "checkpoints.db", check_same_thread=False)
    paused = Supervisor(ALEX, Router(router_model), {}, store.audit, clock=lambda: at,
                        approval_graph=_approval_graph(env, ckpt, f"{corr}_plan")).run(
                            PLAN_REQUEST, correlation_id=f"{corr}_plan")
    thread_id = paused.get("approval_thread_id") or ""

    # 11. the monthly briefing (draft only) with the job's results, findings and plan
    briefing = draft_monthly_briefing(store, runner, ALEX, month=OCTOBER, as_of=at, correlation_id=corr,
                                      agent_results=results, findings=job.get("findings", []), plan=plan,
                                      approval_thread_id=thread_id or None)
    markdown = render_markdown(briefing)
    MonthlyBriefing.model_validate_json(briefing.model_dump_json())
    sections = [briefing.data_coverage, briefing.net_worth_and_cash_flow, briefing.budget_and_variance,
                briefing.material_events, briefing.policy_and_deadlines, briefing.actions_for_approval,
                briefing.assumptions_and_limitations]
    kinds = {i.kind for s in sections for i in s}
    check("draft_briefing", "calculate -> retrieve -> briefing (draft only)",
          briefing.draft and briefing.external_action == "none" and markdown.rstrip().endswith(DISCLAIMER)
          and {"cash_flow", "category_totals", "material_variance", "allocation_scenario", "approval_pending"} <= kinds
          and all(i.evidence_ids for s in sections for i in s if i.amounts_minor)
          and "acct_jordan_card" not in briefing.model_dump_json(),
          f"{sum(map(len, sections))} items; calc runs {briefing.calc_run_ids}; Jordan's private card not shown",
          briefing.calc_run_ids)

    # 12. resume from a reopened checkpoint and ledger ("new process"); the plan is not re-run
    ckpt.close()
    env.conn.close()
    env = _Env.open(workdir)
    ckpt = sqlite3.connect(workdir / "checkpoints.db", check_same_thread=False)
    graph = _approval_graph(env, ckpt, f"{corr}_plan")
    config = {"configurable": {"thread_id": thread_id}}
    waiting = graph.get_state(config).next if thread_id else ()
    final: dict[str, Any] = {}
    if waiting == ("approval",):  # nothing to resume if the request never paused
        graph.invoke(Command(resume={"approver": "sam", "approved": True}), config)
        final = graph.invoke(Command(resume={"approver": "jordan", "approved": True, "note": "slice"}), config)
    thread_events = [(e["action"], e["decision"]) for e in env.store.audit.events(thread_id)]
    check("interrupt_and_resume", "interrupt-and-resume approval without re-running completed steps",
          paused.get("status") == "awaiting_approval" and waiting == ("approval",) and final.get("status") == "approved"
          and thread_events == [("plan_allocation", "info"), ("approve", "deny"), ("approve", "allow")],
          f"paused at {waiting}; after reopening: sam denied, jordan approved; thread events={thread_events}",
          [thread_id])
    ckpt.close()

    # 13. no external action is available
    store = env.store
    refused_job = Supervisor(ALEX, Router(router_model), {}, store.audit, clock=lambda: at).run(
        EXTERNAL_REQUEST, correlation_id=f"{corr}_external")
    tool_names = {*TOOL_NAMES, *(t["function"]["name"] for t in LEDGER_TOOLS + POLICY_TOOLS)}
    risky = sorted(n for n in tool_names if _EXTERNAL.search(n.replace("_", " ")))
    check("no_external_actions", "all external actions unavailable",
          refused_job.get("status") == "refused" and not risky
          and not re.search(r"\b(pay|transfer|wire)\b", final.get("status", "")),
          f"'{EXTERNAL_REQUEST}' -> {refused_job.get('status')}; tools={sorted(tool_names)}", [f"{corr}_external"])

    # 14. the audit chain is intact and the job's trail covers every stage
    count = store.audit.verify()
    actions = {e["action"] for e in store.audit.events(corr)}
    check("audit_trail", "tamper-evident audit trail for the whole job",
          count > 0 and REQUIRED_JOB_ACTIONS <= actions,
          f"{count} chained events; missing stages: {sorted(REQUIRED_JOB_ACTIONS - actions) or 'none'}", [corr])
    env.conn.close()

    (workdir / "briefing_2026-10.md").write_text(markdown)
    return SliceReport(SLICE_VERSION, mode, corr, thread_id, checks, markdown, count)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m household_agent.vertical_slice", description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="chat model spec for router and specialists, e.g. ollama:qwen2.5:7b or groq:qwen/qwen3-32b "
                                    "(default: offline playback)")
    ap.add_argument("--embed-model", default="hash", help="embedding spec (default: hash, offline)")
    ap.add_argument("--workdir", help="keep ledger.db, checkpoints.db, chroma/ and the briefing here")
    ap.add_argument("--out", help="write the JSON report here")
    args = ap.parse_args(argv)

    from household_agent.llm import get_chat_model, get_embeddings

    models = (lambda role: get_chat_model(args.model)) if args.model else None
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(args.workdir) if args.workdir else Path(tmp)
        report = run_vertical_slice(workdir, models=models, embeddings=get_embeddings(args.embed_model))
        print(f"Vertical slice ({report.mode}), correlation_id={report.correlation_id}")
        for c in report.checks:
            print(f"  {'PASS' if c.passed else 'FAIL'}  {c.name:<27} {c.detail}")
        print(f"  audit events: {report.audit_events}; briefing: {workdir / 'briefing_2026-10.md'}"
              + ("" if args.workdir else " (temporary; pass --workdir to keep it)"))
        if args.out:
            Path(args.out).write_text(report.to_json())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
