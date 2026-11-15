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

from household_agent.agents import AgentResult, LedgerAnalyst, PolicyCompliance, Router, Supervisor
from household_agent.approval import build_approval_graph
from household_agent.audit import AuditIntegrityError, AuditLog
from household_agent.authz import Actor, AuthorizationError, Authorizer
from household_agent.categorize import Categorizer
from household_agent.ingest import StatementParseError
from household_agent.ingest.pdf_statement import MAX_PDF_BYTES
from household_agent.ledger import DuplicateDocumentError, LedgerStore
from household_agent.mcp_server import MEMBERS
from household_agent.pipeline import ingest_statement
from household_agent.plan_inputs import Liability, ledger_inputs_for
from household_agent.retrieval import RetrievalService
from household_agent.sample_statements import samples
from household_agent.synthetic import (
    ACCOUNTS,
    ALEX,
    DEMO_AS_OF,
    RULES,
    demo_policy_documents,
    demo_statements,
)
from household_agent.vertical_slice import _planner_kwargs, ledger_playback, policy_playback

STATIC = Path(__file__).parent / "static"
SUGGESTIONS = [
    {"member": "alex", "text": "I paid $3,200 to the San Mateo County Tax Collector in October. Is this unusual "
                              "compared with last year, and how should it be accounted for?"},
    {"member": "alex", "text": "We have a surplus this month; how should we allocate it?"},
    {"member": "alex", "text": "Please transfer $5,000 from checking to the brokerage."},
]
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_MESSAGE_CHARS = 2000


class ChatIn(BaseModel):
    member: str
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    correlation_id: str | None = None


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
    lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self) -> None:
        from household_agent.llm import HashEmbeddings

        self.workdir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.workdir / "ledger.db", check_same_thread=False)
        self.store = LedgerStore(self.conn, Authorizer(ACCOUNTS), AuditLog(self.conn))
        self.categorizer = Categorizer(RULES)
        client = chromadb.PersistentClient(str(self.workdir / "chroma"), settings=Settings(anonymized_telemetry=False))
        self.retrieval = RetrievalService(client, self.embeddings or HashEmbeddings(), self.store.authorizer, self.store.audit)
        ckpt = sqlite3.connect(self.workdir / "checkpoints.db", check_same_thread=False)
        self.plan_actor = ALEX  # whose reads the planner uses: set per request, under the lock
        self.requesters: dict[str, Actor] = {}  # approval thread -> who asked, so a resume re-derives the same inputs
        runner = _runner(self.store)

        def inputs_for(household_id: str):
            return ledger_inputs_for(self.store, runner, self.plan_actor, clock=lambda: DEMO_AS_OF,
                                     correlation_id="corr_web_plan", **_planner_kwargs())(household_id)

        self.approvals = build_approval_graph(
            inputs_for=inputs_for, members=MEMBERS, audit=self.store.audit, checkpointer=SqliteSaver(ckpt))
        self._load_documents()
        self._seed()

    def _load_documents(self) -> None:
        """A reused ``--workdir`` already holds documents; list them again."""
        for name, doc_id, status, conf, issues in self.conn.execute(
                "SELECT source_name, document_id, reconciliation_status, document_confidence, issues_json FROM documents "
                "ORDER BY ingested_at"):
            self.documents.append({"name": name, "document_id": doc_id, "outcome": status, "confidence": conf,
                                   "uploaded_by": "", "issues": [{"code": i.get("code"), "message": i.get("message")}
                                                                 for i in json.loads(issues)]})

    def _seed(self) -> None:
        """Last year's statement and the policy sources are already on file, so this year's upload has history."""
        corr = "corr_web_seed"
        name = "2025_10_rental_checking.csv"
        if any(d["name"] == name for d in self.documents):
            return  # already seeded: a reused --workdir
        self._ingest(ALEX, demo_statements()[name].encode(), name, corr)
        for doc in demo_policy_documents():
            self.retrieval.index(ALEX, doc, correlation_id=corr, at=DEMO_AS_OF)

    def _ingest(self, actor: Actor, content: bytes, name: str, corr: str) -> dict[str, Any]:
        result = ingest_statement(self.store, self.categorizer, actor, content, name, purpose="statement_ingestion",
                                  correlation_id=corr, at=DEMO_AS_OF)
        doc = {"name": name, "document_id": result.document_id, "outcome": result.status.value,
               "confidence": result.document_confidence, "uploaded_by": actor.member_id,
               "issues": [{"code": i.code, "message": i.message} for i in result.issues]}
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
            "agents": [{"agent": r.agent, "status": r.status, "errors": [str(e)[:200] for e in r.errors]} for r in results],
            "approval": self._approval_view(thread) if thread else None,
            "trace": self.trace(corr),
        }

    def _approval_view(self, thread_id: str) -> dict[str, Any]:
        state = self.approvals.get_state({"configurable": {"thread_id": thread_id}}).values
        return {"thread_id": thread_id, "status": state.get("status"), "plan": state.get("plan"),
                "decision": state.get("decision")}

    def approve(self, thread_id: str, approver: str, approved: bool, note: str) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        with self.lock:
            before = self.approvals.get_state(config)
            if not before.values or before.next != ("approval",):
                return {"thread_id": thread_id, "status": (before.values or {}).get("status", "not_found"),
                        "refused": True, "reason": "there is no plan waiting for approval under this id"}
            denied_before = before.values.get("denied_attempts", 0)
            self.plan_actor = self.requesters.get(thread_id, ALEX)
            after = self.approvals.invoke(Command(resume={"approver": approver, "approved": approved, "note": note}), config)
        refused = after.get("status") == "awaiting_approval" and after.get("denied_attempts", 0) > denied_before
        return {"thread_id": thread_id, "status": after.get("status"), "refused": refused,
                "reason": "only a household admin or adult member may approve" if refused else "",
                "plan": after.get("plan"), "decision": after.get("decision")}

    def trace(self, corr: str) -> list[dict[str, Any]]:
        return [{"actor": e["actor_id"], "agent": e["acting_agent"], "action": e["action"], "decision": e["decision"],
                 "resource": e.get("resource"), "reason": (e.get("reason") or "")[:200]} for e in self.store.audit.events(corr)]

    def audit_status(self) -> dict[str, Any]:
        try:
            return {"intact": True, "events": self.store.audit.verify()}
        except AuditIntegrityError as exc:
            return {"intact": False, "events": None, "reason": str(exc)[:200]}

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
                "documents": service.documents, "suggestions": SUGGESTIONS,
                "samples": [{"name": s.name, "member": s.member, "description": s.description, "expect": s.expect}
                            for s in catalog.values()]}

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
