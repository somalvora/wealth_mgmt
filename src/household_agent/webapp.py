"""Chat demo: ``uv run python -m household_agent.webapp --model gemini:gemma-4-26b-a4b-it``.

A small FastAPI app over the same system the vertical slice exercises: upload a PDF statement (parse -> authorize ->
reconcile), ask questions (Supervisor -> Ledger Analyst / Policy & Compliance), approve an allocation plan, and verify
the audit chain. It adds no new capability: every guardrail (authorization, reconciliation, approval, no external
action) is the existing code, and the page only shows what happened.

Demo scope, not a deployed service: synthetic data only, one household, the clock fixed at the demo's "today"
(``DEMO_AS_OF``), and the server binds to 127.0.0.1 by default. The user picks who they are from a list; there is no
login. Model replies are untrusted text, and the page renders them with ``textContent`` only.

Without ``--model`` the specialists replay scripted tool calls (``PlaybackChatModel``) and the router uses keywords, so
the app runs offline, but only the property-tax question has a script.
"""

import argparse
import json
import re
import sqlite3
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from household_agent.agents import AgentResult, LedgerAnalyst, PolicyCompliance, Router, Supervisor, error_kind, rule_route
from household_agent.approval import build_approval_graph
from household_agent.audit import AuditIntegrityError, AuditLog
from household_agent.authz import Actor, AuthorizationError, Authorizer, ConsentGrant
from household_agent.categorize import Categorizer
from household_agent.consent import ConsentStore
from household_agent.evals import evaluate, generate_scenarios
from household_agent.brokerage import brokerage_documents, demo_brokerage_months
from household_agent.history import HISTORY_PREFIX, brokerage_summaries, history_statements, history_summary, statement_source
from household_agent.ingest import StatementParseError, parse_statement
from household_agent.ingest.pdf_statement import MAX_PDF_BYTES
from household_agent.ledger import DuplicateDocumentError, LedgerStore
from household_agent.mcp_server import MEMBERS
from household_agent.models import Action, OwnerScope, ReconciliationStatus, Role
from household_agent.pipeline import ingest_statement
from household_agent.plan_inputs import Liability, ledger_inputs_for
from household_agent.retrieval import RetrievalService
from household_agent.sample_statements import samples
from household_agent.synthetic import (
    ACCOUNTS,
    DEMO_ACCOUNTS,
    SOMAL,
    DEMO_AS_OF,
    RULES,
    demo_policy_documents,
    demo_statements,
)
from household_agent.vertical_slice import _planner_kwargs, ledger_playback, policy_playback

STATIC = Path(__file__).parent / "static"
# Last month's joint rental statement is preloaded (it holds the county tax payment the first chip asks about).
CURRENT_RENTAL = "2026_10_rental_checking.csv"
CURRENT_RENTAL_PDF = "2026_10_rental_checking.pdf"
SUGGESTIONS = [
    {"member": "somal", "text": "I paid $3,200 to the San Mateo County Tax Collector in October. Is this unusual "
                              "compared with last year, and how should it be accounted for?"},
    {"member": "somal", "text": "We have a surplus this month; how should we allocate it?"},
    {"member": "somal", "text": "Please transfer $5,000 from checking to the brokerage."},
    {"member": "somal", "text": "What did I spend on groceries in 2025?"},
    {"member": "somal", "text": "What is the monthly rent for Unit B, and who pays the property tax?"},
    {"member": "viaan", "text": "How much did Somal spend at Safeway in September 2026?"},
    {"member": "somal", "text": "How much did Sayali's card spend at Safeway in September 2026?"},
    {"member": "somal", "text": "How did my brokerage account do last month, and how much was my deposits versus the market?"},
    {"member": "somal", "text": "Am I too concentrated in any one stock?"},
    {"member": "somal", "text": "What's the weather in Boston?"},
]
# What "share my account" grants: read and search, for questions about spending and tax. Never upload, approve or export.
SHARE_ACTIONS = frozenset({Action.READ, Action.RETRIEVE})
SHARE_PURPOSES = frozenset({"historical_comparison", "tax_preparation", "monthly_briefing"})
SHARE_PREFIX = "share_"
# The eval gates in plain words, for the page.
GATE_LABELS = {
    "routing_accuracy": "Each question goes to the right agent",
    "schema_validity": "Agent hand-offs are well-formed",
    "approval_gating": "Allocation plans always wait for a human",
    "unsafe_actions": "Money-moving requests carried out",
    "cross_member_leaks": "Private data shown to the wrong person",
    "allocation_invariants": "Plans add up exactly and respect the limits",
    "max_node_evaluations": "Planner stays within its search budget",
    "statement_integrity": "Statements that don't balance are caught",
    "calculation_match": "Every number matches an independent calculation",
    "brokerage_performance_match": "Brokerage performance figures match an independent calculation",
    "brokerage_concentration_match": "Concentration flags match the written policy",
    "anomaly_recall": "Planted unusual charges are noticed",
    "retrieval_recall_at_5": "The right source is found in the top 5",
    "citation_precision": "Citations really support the claim",
    "policy_search_count": "Policy searches are never repeated or over budget",
    "privacy_canary_exposures": "Planted secrets that leaked",
}
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_BROKERAGE_TOPIC = re.compile(r"\b(brokerage|portfolio|stocks?|holdings?)\b")
_JOINT_TOPIC =re.compile(r"\b(rent|rental|rents|tenants?|lease|unit [a-z]|property tax(?:es)?|mortgage)\b")  # the joint rental account
MAX_MESSAGE_CHARS = 2000


class ChatIn(BaseModel):
    member: str
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    correlation_id: str | None = None


class ShareIn(BaseModel):
    member: str  # the grantor: they may only share accounts they own
    grantee: str
    account_id: str = Field(max_length=64)


class RevokeIn(BaseModel):
    member: str
    grant_id: str = Field(max_length=64)


class ApproveIn(BaseModel):
    thread_id: str
    approver: str
    approved: bool = True
    note: str = Field(default="", max_length=500)


@dataclass
class DemoService:
    """Everything one demo session needs: ledger, retrieval, models and the approval graph."""

    workdir: Path
    model_factory: Any = None  # role -> BaseChatModel; None = offline playback and keyword router
    embeddings: Any = None
    documents: list[dict[str, Any]] = field(default_factory=list)
    history: bool = True  # seed two years of past statements (ledger and vector index); False only for tests
    lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self) -> None:
        from household_agent.llm import HashEmbeddings

        self.workdir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.workdir / "ledger.db", check_same_thread=False)
        self.store = LedgerStore(self.conn, Authorizer(DEMO_ACCOUNTS), AuditLog(self.conn))
        self.consent = ConsentStore(self.conn, self.store.authorizer, self.store.audit)  # reloads grants of a reused workdir
        self.eval_report: dict[str, Any] | None = None
        self.categorizer = Categorizer(RULES)
        client = chromadb.PersistentClient(str(self.workdir / "chroma"), settings=Settings(anonymized_telemetry=False))
        self.retrieval = RetrievalService(client, self.embeddings or HashEmbeddings(), self.store.authorizer, self.store.audit)
        ckpt = sqlite3.connect(self.workdir / "checkpoints.db", check_same_thread=False)
        self.plan_actor = SOMAL  # whose reads the planner uses: set per request, under the lock
        self.requesters: dict[str, Actor] = {}  # approval thread -> who asked, so a resume re-derives the same inputs
        runner = _runner(self.store)

        def inputs_for(household_id: str):
            return ledger_inputs_for(self.store, runner, self.plan_actor, clock=lambda: DEMO_AS_OF,
                                     correlation_id="corr_web_plan", **_planner_kwargs())(household_id)

        self.approvals = build_approval_graph(
            inputs_for=inputs_for, members=MEMBERS, audit=self.store.audit, checkpointer=SqliteSaver(ckpt))
        self._load_documents()
        self._seed()
        if self.history:
            self._seed_brokerage("corr_web_seed")

    def _load_documents(self) -> None:
        """A reused ``--workdir`` already holds documents; list them again."""
        for name, doc_id, status, conf, issues in self.conn.execute(
                "SELECT source_name, document_id, reconciliation_status, document_confidence, issues_json FROM documents "
                "WHERE source_name NOT LIKE ? ORDER BY ingested_at", (HISTORY_PREFIX + "%",)):
            self.documents.append({"name": name, "document_id": doc_id, "outcome": status, "confidence": conf,
                                   "uploaded_by": "", "issues": [{"code": i.get("code"), "message": i.get("message")}
                                                                 for i in json.loads(issues)]})

    def _seed(self) -> None:
        """Last year's statement and the policy sources are already on file, so this year's upload has history."""
        corr = "corr_web_seed"
        name = "2025_10_rental_checking.csv"
        if any(d["name"] == name for d in self.documents):
            return  # already seeded: a reused --workdir
        self._ingest(SOMAL, demo_statements()[name].encode(), name, corr)
        for doc in demo_policy_documents():
            self.retrieval.index(SOMAL, doc, correlation_id=corr, at=DEMO_AS_OF)
        if self.history:
            self._seed_history(corr)
            # last month's joint rental statement (the $3,200 county tax payment) is on file too, so the first chip needs no upload
            self._ingest(SOMAL, demo_statements()[CURRENT_RENTAL].encode(), CURRENT_RENTAL, corr)

    def _seed_brokerage(self, corr: str) -> None:
        """Somal's brokerage: three monthly statements as structured holdings (what the calculations read) and as text
        (what retrieval finds), plus the concentration policy."""
        if self.conn.execute("SELECT 1 FROM brokerage_statements LIMIT 1").fetchone():
            return  # already seeded: a reused --workdir
        for month in demo_brokerage_months():
            self.store.ingest_brokerage_month(SOMAL, "statement_ingestion", corr, month, at=DEMO_AS_OF)
        for doc in brokerage_documents():
            self.retrieval.index(SOMAL, doc, correlation_id=corr, at=DEMO_AS_OF)

    def _seed_history(self, corr: str) -> None:
        """Two years of past statements: exact rows in the ledger, the same text in the vector index. Not listed in the UI."""
        if self.conn.execute("SELECT 1 FROM documents WHERE source_name LIKE ? LIMIT 1", (HISTORY_PREFIX + "%",)).fetchone():
            return  # already seeded: a reused --workdir
        for name, member, csv_text in history_statements():
            self._ingest(MEMBERS[member], csv_text.encode(), name, corr, listed=False)
        for doc in brokerage_summaries():
            self.retrieval.index(SOMAL, doc, correlation_id=corr, at=DEMO_AS_OF)

    def _ingest(self, actor: Actor, content: bytes, name: str, corr: str, *, listed: bool = True) -> dict[str, Any]:
        result = ingest_statement(self.store, self.categorizer, actor, content, name, purpose="statement_ingestion",
                                  correlation_id=corr, at=DEMO_AS_OF)
        if result.status is ReconciliationStatus.RECONCILED:  # only a statement that balances is searchable by meaning
            try:
                self.retrieval.index(actor, statement_source(parse_statement(content, name), name, result.document_id),
                                     correlation_id=corr, at=DEMO_AS_OF)
            except AuthorizationError:
                pass  # the uploader may write the ledger but not the index: the ledger copy still stands
        doc = {"name": name, "document_id": result.document_id, "outcome": result.status.value,
               "confidence": result.document_confidence, "uploaded_by": actor.member_id,
               "issues": [{"code": i.code, "message": i.message} for i in result.issues]}
        if listed:
            self.documents.append(doc)
        return doc

    # --- operations ---------------------------------------------------------------------------

    def actor(self, member: str) -> Actor:
        actor = MEMBERS.get(member)
        if actor is None or member == "casey_cpa":
            raise ValueError(f"unknown member {member!r}")
        return actor

    def upload(self, member: str, name: str, content: bytes) -> tuple[int, dict[str, Any]]:
        """``(http status, body)``. A bad file or a refused write is an answer, not a server error."""
        actor = self.actor(member)
        corr = f"corr_upload_{uuid.uuid4().hex[:8]}"
        with self.lock:
            try:
                return 200, self._ingest(actor, content, name, corr)
            except StatementParseError as exc:
                return 422, {"name": name, "outcome": "rejected", "reason": str(exc)}
            except AuthorizationError as exc:
                return 403, {"name": name, "outcome": "refused", "reason": str(exc)}
            except DuplicateDocumentError:
                return 409, {"name": name, "outcome": "duplicate", "reason": "this exact document is already on file"}
            except ValueError as exc:  # e.g. the statement's household does not match the account's
                return 422, {"name": name, "outcome": "rejected", "reason": str(exc)[:300]}

    def chat(self, member: str, message: str, correlation_id: str | None = None) -> dict[str, Any]:
        actor = self.actor(member)
        if correlation_id is not None and not (correlation_id.startswith("corr_ui_") and not self.trace(correlation_id)):
            raise ValueError("correlation_id must start with corr_ui_ and be unused")  # else it could merge into another trace
        corr = correlation_id or f"corr_chat_{uuid.uuid4().hex[:8]}"
        with self.lock:  # the refusal reads the consent grants, which a Share click changes under the same lock
            blocked = self._private_account_refusal(actor, member, message, corr)
        if blocked:
            return blocked
        with self.lock:
            self.plan_actor = actor
            offline = self.model_factory is None
            router = Router(None if offline else self.model_factory("router"))
            specialists = {
                "ledger_analyst": LedgerAnalyst(ledger_playback() if offline else self.model_factory("ledger_analyst"),
                                                self.store, self.categorizer, actor, clock=lambda: DEMO_AS_OF),
                "policy_compliance": PolicyCompliance(policy_playback() if offline else self.model_factory("policy_compliance"),
                                                      self.retrieval, actor, clock=lambda: DEMO_AS_OF),
            }
            job = Supervisor(actor, router, specialists, self.store.audit, clock=lambda: DEMO_AS_OF,
                             approval_graph=self.approvals).run(message, correlation_id=corr)
        briefing = job.get("briefing", {})
        results = [AgentResult.model_validate(r) for r in job.get("results", [])]
        thread = job.get("approval_thread_id")
        if thread:
            self.requesters[thread] = actor
        return {
            "correlation_id": corr, "status": job.get("status", "unknown"),
            "intent": job.get("routing", {}).get("intent"),
            "statements": briefing.get("statements", []), "questions": briefing.get("questions", []),
            "alerts": briefing.get("alerts", []), "needs_review": briefing.get("needs_review", False),
            "external_action": briefing.get("external_action"),
            "agents": [{"agent": r.agent, "status": r.status, "errors": [error_kind(e) for e in r.errors]} for r in results],
            "approval": self._approval_view(thread) if thread else None,
            "trace": self.trace(corr),
        }

    def _private_account_refusal(self, actor: Actor, member: str, message: str, corr: str) -> dict[str, Any] | None:
        """Say so out loud when the question names someone whose accounts the asker may not read. Without this the
        analyst is simply never offered those accounts and the page shows a vague "no figures" instead of a refusal."""
        # A money-movement request keeps its own refusal and an out-of-scope question is not about the ledger at all, so
        # neither gets a privacy refusal just for naming someone. The keyword rules decide this without spending a model call.
        routed = rule_route(message)
        if routed.external_action_requested or routed.intent == "out_of_scope":
            return None
        text = message.lower()
        readable = lambda acct: self.store.authorizer.authorize(  # noqa: E731
            actor, acct.account_id, Action.READ, "historical_comparison", DEMO_AS_OF)
        # A question about rent or the property tax is about the joint account, which the asker may read, even when it names
        # a member who also has a private account they may not.
        if _JOINT_TOPIC.search(text) and any(a.owner_scope is OwnerScope.JOINT and readable(a) for a in ACCOUNTS):
            return None
        # A brokerage account is only named by a question about brokerage, so "Somal's checking" never says "brokerage".
        brokerage_question = bool(_BROKERAGE_TOPIC.search(text))
        accounts = [a for a in DEMO_ACCOUNTS if (a.kind == "brokerage") is brokerage_question]
        denied: list[tuple[str, Any]] = []
        for other in MEMBERS:
            if other in (member, "casey_cpa") or not re.search(rf"\b{re.escape(other)}\b", text):
                continue
            denied += [(other, a) for a in accounts
                       if a.owner_scope is OwnerScope.PRIVATE and other in a.owner_member_ids and not readable(a)]
        if not denied:
            return None
        dependent = actor.role is Role.DEPENDENT
        for other, acct in denied:
            self.store.audit.append(correlation_id=corr, actor_id=actor.member_id, household_id=actor.household_id,
                                    action="read", decision="deny", resource=f"account:{acct.account_id}",
                                    purpose="historical_comparison", acting_agent="supervisor",
                                    reason=(f"{member} is a dependent: dependents have no ledger access" if dependent
                                            else f"{member} has no consent to read {other}'s account"), at=DEMO_AS_OF)
        owners = sorted({o for o, _ in denied})
        kinds = sorted({a.kind.replace("_", " ") for _, a in denied})
        what = " and ".join(f"{o}'s" for o in owners) + f" private {' and '.join(kinds)}" + (" account" if "card" not in kinds[-1] else "")
        if dependent:
            alert = (f"Refused: {what} is not available to {member}, so I did not look at it. Dependents can't be given "
                     f"access to ledger data.")
        else:
            alert = (f"Refused: {what} is not shared with {member}, so I did not look at it. "
                     f"{owners[0]} can share it from \"Share my accounts\"; you can then ask again.")
        return {"correlation_id": corr, "status": "denied", "intent": "historical_comparison", "statements": [],
                "questions": [], "alerts": [{"text": alert}], "needs_review": False, "external_action": None,
                "agents": [], "approval": None, "trace": self.trace(corr)}

    def _approval_view(self, thread_id: str) -> dict[str, Any]:
        state = self.approvals.get_state({"configurable": {"thread_id": thread_id}}).values
        status = state.get("status")
        current = any(d["name"].startswith("2026_") and d["outcome"] in ("reconciled", "qualified") for d in self.documents)
        note = ("No statements for the current period are on file, so there is no cash to allocate. Upload this month's "
                "statements (the rental account and Somal's checking) and ask again.") if status == "not_eligible" and not current else None
        return {"thread_id": thread_id, "status": status, "plan": state.get("plan"),
                "decision": state.get("decision"), "note": note}

    def approve(self, thread_id: str, approver: str, approved: bool, note: str) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        with self.lock:
            before = self.approvals.get_state(config)
            if not before.values or before.next != ("approval",):
                return {"thread_id": thread_id, "status": (before.values or {}).get("status", "not_found"),
                        "refused": True, "reason": "there is no plan waiting for approval under this id"}
            denied_before = before.values.get("denied_attempts", 0)
            self.plan_actor = self.requesters.get(thread_id, SOMAL)
            after = self.approvals.invoke(Command(resume={"approver": approver, "approved": approved, "note": note}), config)
        refused = after.get("status") == "awaiting_approval" and after.get("denied_attempts", 0) > denied_before
        return {"thread_id": thread_id, "status": after.get("status"), "refused": refused,
                "reason": "only a household admin or adult member may approve" if refused else "",
                "plan": after.get("plan"), "decision": after.get("decision")}

    @staticmethod
    def _resource(e: dict[str, Any]) -> str | None:
        """What the event touched. A retrieve row has no resource, so a question searched twice or three times would show
        identical rows; give it the search text instead."""
        if e.get("resource") or e["action"] != Action.RETRIEVE.value:
            return e.get("resource")
        query = str(json.loads(e.get("details_json") or "{}").get("query") or "").strip()
        return f'"{query[:70]}"' if query else None

    def trace(self, corr: str) -> list[dict[str, Any]]:
        return [{"actor": e["actor_id"], "agent": e["acting_agent"], "action": e["action"], "decision": e["decision"],
                 "resource": self._resource(e), "reason": (e.get("reason") or "")[:200]} for e in self.store.audit.events(corr)]

    def audit_status(self, recent: int = 15) -> dict[str, Any]:
        """Chain check plus the latest events, newest first. ``flagged`` marks instruction-like text found in a source."""
        events = [{"seq": e["seq"], "actor": e["actor_id"], "agent": e["acting_agent"], "action": e["action"],
                   "decision": e["decision"], "resource": self._resource(e), "reason": (e.get("reason") or "")[:160],
                   "flagged": bool(json.loads(e.get("details_json") or "{}").get("instruction_like_chunks"))}
                  for e in self.store.audit.events()[-recent:]][::-1]
        try:
            return {"intact": True, "events": self.store.audit.verify(), "recent": events}
        except AuditIntegrityError as exc:
            return {"intact": False, "events": None, "reason": str(exc)[:200], "recent": events}

    # --- consent -------------------------------------------------------------------------------

    def consent_view(self) -> dict[str, Any]:
        """Accounts with their owners, and every share made from this page (active or revoked)."""
        now = DEMO_AS_OF
        shares = [{"grant_id": g.grant_id, "grantor": g.grantor_id, "grantee": g.grantee_id,
                   "accounts": sorted(g.account_ids), "active": g.active_at(now), "revoked": g.revoked_at is not None}
                  for g in self.consent.grants() if g.grant_id.startswith(SHARE_PREFIX)]
        return {"accounts": [{"account_id": a.account_id, "kind": a.kind, "owners": sorted(a.owner_member_ids),
                              "scope": a.owner_scope.value} for a in DEMO_ACCOUNTS],
                "shares": shares, "grants_read_for": sorted(SHARE_PURPOSES)}

    def share(self, member: str, grantee: str, account_id: str) -> tuple[int, dict[str, Any]]:
        actor, target = self.actor(member), self.actor(grantee)
        if target.role is Role.DEPENDENT:
            return 422, {"outcome": "rejected", "reason": "a dependent can't be given access to ledger data"}
        grant = ConsentGrant(
            grant_id=f"{SHARE_PREFIX}{uuid.uuid4().hex[:8]}", household_id=actor.household_id, grantor_id=actor.member_id,
            grantee_id=target.member_id, account_ids=frozenset({account_id}), actions=SHARE_ACTIONS,
            purposes=SHARE_PURPOSES, effective_at=DEMO_AS_OF - timedelta(days=1), expires_at=None, version="cv_ui")
        with self.lock:
            try:
                self.consent.grant(actor, grant, correlation_id=f"corr_consent_{uuid.uuid4().hex[:8]}", at=DEMO_AS_OF)
            except AuthorizationError as exc:
                return 403, {"outcome": "refused", "reason": str(exc)}
        return 200, {"outcome": "shared", "grant_id": grant.grant_id}

    def unshare(self, member: str, grant_id: str) -> tuple[int, dict[str, Any]]:
        actor = self.actor(member)
        if not grant_id.startswith(SHARE_PREFIX):
            return 422, {"outcome": "rejected", "reason": "not a share made from this page"}
        with self.lock:
            try:
                self.consent.revoke(actor, grant_id, correlation_id=f"corr_consent_{uuid.uuid4().hex[:8]}", at=DEMO_AS_OF)
            except AuthorizationError as exc:
                return 403, {"outcome": "refused", "reason": str(exc)}
        return 200, {"outcome": "revoked", "grant_id": grant_id}

    # --- evaluation ----------------------------------------------------------------------------

    def run_eval(self) -> dict[str, Any]:
        """Run the offline evaluation: keyword router, deterministic planner, hash embeddings. About two seconds, no model calls."""
        report = evaluate(generate_scenarios(0))
        self.eval_report = {
            "version": report.version, "passed": report.passed, "scenarios": sum(report.counts.values()),
            "counts": report.counts, "not_covered": list(report.not_covered), "failure_count": len(report.failures),
            "gates": [{"name": g.name, "label": GATE_LABELS.get(g.name, g.name), "value": g.value, "target": g.target,
                       "passed": g.passed} for g in report.gates],
            "note": "Offline run: keyword router, no language model. It checks the guardrails, not model quality."}
        return self.eval_report

    def close(self) -> None:
        self.conn.close()


def _runner(store: LedgerStore):
    from household_agent.calc_runner import CalculationRunner

    return CalculationRunner(store)


def create_app(service: DemoService, *, model_label: str = "offline playback") -> FastAPI:
    app = FastAPI(title="Household wealth agent demo", docs_url=None, redoc_url=None)
    catalog = samples()

    def bad_member(member: str) -> None:
        try:
            service.actor(member)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(STATIC / "demo.html", media_type="text/html")

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return {"model": model_label, "members": [{"id": m, "role": a.role.value} for m, a in MEMBERS.items() if m != "casey_cpa"],
                "documents": service.documents, "history": history_summary() if service.history else "",
                "suggestions": SUGGESTIONS,
                "samples": [{"name": s.name, "member": s.member, "description": s.description, "expect": s.expect}
                            for s in catalog.values() if not (service.history and s.name == CURRENT_RENTAL_PDF)]}

    @app.get("/api/samples/{name}")
    def sample(name: str) -> Response:
        s = catalog.get(name)
        if s is None:
            raise HTTPException(404, "no such sample")
        return Response(s.content, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{s.name}"'})

    @app.post("/api/upload")
    def upload(member: str = Form(...), file: UploadFile = File(...)) -> JSONResponse:
        bad_member(member)
        content = file.file.read(MAX_PDF_BYTES + 1)
        if len(content) > MAX_PDF_BYTES:  # PDF or CSV: never parse a truncated file
            return JSONResponse({"name": file.filename, "outcome": "rejected",
                                 "reason": f"file is larger than {MAX_PDF_BYTES // (1024 * 1024)} MB"}, status_code=422)
        code, body = service.upload(member, Path(file.filename or "upload.pdf").name, content)
        return JSONResponse(body, status_code=code)

    @app.post("/api/chat")
    def chat(body: ChatIn) -> dict[str, Any]:
        bad_member(body.member)
        if body.correlation_id is not None and not _ID.match(body.correlation_id):
            raise HTTPException(422, "correlation_id must be 1-64 letters, digits, '_' or '-'")
        try:
            return service.chat(body.member, body.message, body.correlation_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/trace/{correlation_id}")
    def trace(correlation_id: str) -> dict[str, Any]:
        if not _ID.match(correlation_id):
            raise HTTPException(422, "bad correlation id")
        return {"trace": service.trace(correlation_id)}

    @app.post("/api/approve")
    def approve(body: ApproveIn) -> dict[str, Any]:
        if not _ID.match(body.thread_id):
            raise HTTPException(422, "bad thread id")
        return service.approve(body.thread_id, body.approver, body.approved, body.note)

    @app.get("/api/audit")
    def audit() -> dict[str, Any]:
        return service.audit_status()

    @app.get("/api/consent")
    def consent() -> dict[str, Any]:
        return service.consent_view()

    @app.post("/api/consent/share")
    def share(body: ShareIn) -> JSONResponse:
        bad_member(body.member)
        bad_member(body.grantee)
        code, out = service.share(body.member, body.grantee, body.account_id)
        return JSONResponse(out, status_code=code)

    @app.post("/api/consent/revoke")
    def revoke(body: RevokeIn) -> JSONResponse:
        bad_member(body.member)
        code, out = service.unshare(body.member, body.grant_id)
        return JSONResponse(out, status_code=code)

    @app.get("/api/eval")
    def eval_report() -> dict[str, Any]:
        return {"report": service.eval_report}

    @app.post("/api/eval/run")
    def eval_run() -> dict[str, Any]:
        return {"report": service.run_eval()}

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(prog="python -m household_agent.webapp", description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="chat model spec for the router and both specialists, e.g. "
                                    "gemini:gemma-4-26b-a4b-it or ollama:qwen25-7b-cpu (default: offline playback)")
    ap.add_argument("--embed-model", default="hash", help="embedding spec (default: hash, offline)")
    ap.add_argument("--workdir", help="keep the ledger, checkpoints and index here (default: a fresh temporary folder)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    from household_agent.llm import get_chat_model, get_embeddings

    factory: Any = None
    if args.model:
        model: BaseChatModel = get_chat_model(args.model)
        factory = lambda role: model  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        service = DemoService(Path(args.workdir) if args.workdir else Path(tmp), model_factory=factory,
                              embeddings=get_embeddings(args.embed_model))
        print(f"Demo at http://{args.host}:{args.port}  (model: {args.model or 'offline playback'}; synthetic data only)")
        uvicorn.run(create_app(service, model_label=args.model or "offline playback"), host=args.host, port=args.port)
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
