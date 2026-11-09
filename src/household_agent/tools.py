"""Agent-facing tools (design §3.2), shared by the MCP server and the LangGraph agents.

A ``ToolSession`` is created by the host for one authenticated request. It binds the actor,
purpose, correlation ID and clock, so none of them is a tool argument: a model cannot claim
to be another member or pick a purpose that unlocks someone else's consent. Every tool
takes and returns JSON-safe values and reports failures as ``{"ok": False, "error": ...}``
so agents can route on them (design §5.4) instead of crashing.

``run_calculation`` accepts transaction IDs only, and only IDs this session already received
from an authorized ``query_ledger`` call, so a model can never feed invented numbers into a
calculation.
"""

import contextvars
import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from household_agent.authz import Actor, AuthorizationError
from household_agent.calc_runner import REGISTRY, CalculationRunner, IneligibleInputError
from household_agent.categorize import Categorizer
from household_agent.ingest import StatementParseError
from household_agent.ledger import DuplicateDocumentError, LedgerStore
from household_agent.models import Action, Transaction
from household_agent.pipeline import ingest_statement
from household_finance import CalculationError

TOOL_NAMES = ("list_accounts", "query_ledger", "run_calculation", "ingest_statement")


class ToolBudgetExceeded(RuntimeError):
    pass


class JobCancelled(RuntimeError):
    """The job's time limit passed (or the job was cancelled): the tool refuses and does nothing."""


class StatementMismatchError(PermissionError):
    """``ingest_statement`` was called with CSV text other than the statement the task supplied."""


# Set by the Supervisor for each dispatched specialist (in that task's own context). A specialist that outlives the job
# time limit cannot be killed, so the Supervisor sets this event and every tool call after that is refused.
CANCEL_EVENT: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar("cancel_event", default=None)


_ERROR_TYPES: tuple[tuple[type[BaseException], str], ...] = (
    (AuthorizationError, "authorization_denied"),
    (ToolBudgetExceeded, "budget_exhausted"),
    (JobCancelled, "job_cancelled"),
    (StatementMismatchError, "statement_mismatch"),
    (IneligibleInputError, "ineligible_input"),
    (DuplicateDocumentError, "duplicate_document"),
    (StatementParseError, "invalid_statement"),
    (CalculationError, "calculation_error"),
    (KeyError, "invalid_request"),
    (ValueError, "invalid_request"),
)


def _error(exc: BaseException) -> dict[str, Any]:
    kind = next(name for t, name in _ERROR_TYPES if isinstance(exc, t))
    message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
    return {"ok": False, "error": {"type": kind, "message": str(message)}}


def _txn_view(t: Transaction) -> dict[str, Any]:
    return {
        "transaction_id": t.transaction_id, "version": t.version, "account_id": t.account_id,
        "posted_at": t.posted_at.isoformat(), "amount_minor": t.amount_minor, "currency": t.currency,
        "merchant": t.merchant_normalized, "category": t.category,
        "reconciliation_status": t.reconciliation_status.value, "document_confidence": t.document_confidence,
        "source_document_id": t.source_document_id, "source_locator": t.source_locator,
    }


def _left_out_view(notes: list[dict[str, Any]]) -> dict[str, Any]:
    """What a ledger tool result says about documents it did not (fully) use, merged over ``notes`` (one per query).

    ``excluded``: documents whose rows were left out, by status (``unreconciled`` / ``unverified`` / ``qualified``);
    ``excluded_rows``: how many rows that is; ``unverified_included``: unverified documents whose provisional rows
    *were* returned. Agents read this structured field; nothing depends on matching error text.
    """
    excluded: dict[str, dict[str, Any]] = {"unreconciled": {}, "unverified": {}, "qualified": {}}
    included: dict[str, Any] = {}
    for n in notes:
        for kind, docs in n["excluded"].items():
            excluded[kind].update({d["document_id"]: d for d in docs})
        included.update({d["document_id"]: d for d in n["unverified_included"]})
    return {"excluded": {k: list(v.values()) for k, v in excluded.items()},
            "excluded_rows": sum(d["rows_in_range"] for v in excluded.values() for d in v.values()),
            "unverified_included": list(included.values())}


@dataclass
class ToolSession:
    store: LedgerStore
    categorizer: Categorizer
    actor: Actor
    purpose: str
    correlation_id: str
    acting_agent: str | None = None
    max_calls: int | None = None
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    cancel: threading.Event | None = None  # set => every tool call is refused before it touches anything
    # Agent sessions pin ingestion to the statement the task supplied (its SHA-256); None with
    # ``restrict_ingest`` means the task supplied none, so ingest_statement is refused. Host/MCP sessions leave it off.
    restrict_ingest: bool = False
    ingest_csv_sha256: str | None = None
    calls: int = 0
    _seen: dict[str, Transaction] = field(default_factory=dict, repr=False)
    _seen_in: dict[str, set[int]] = field(default_factory=dict, repr=False)  # transaction_id -> queries that returned it
    _query_notes: list[dict[str, Any]] = field(default_factory=list, repr=False)  # per query: documents left out

    def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Dispatch by name (used by agents and the MCP server); never raises."""
        if name not in TOOL_NAMES:
            return _error(KeyError(f"unknown tool {name!r}"))
        try:
            return getattr(self, name)(**(args or {}))
        except TypeError as exc:  # wrong/missing arguments from the model
            return _error(ValueError(f"bad arguments for {name}: {exc}"))

    def _charge(self) -> None:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise ToolBudgetExceeded(f"tool budget of {self.max_calls} calls exhausted")
        self.calls += 1

    def _guard(self, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            if self.cancel is not None and self.cancel.is_set():
                raise JobCancelled("the job time limit passed; this tool call was not run")
            self._charge()
            return {"ok": True, **fn()}
        except tuple(t for t, _ in _ERROR_TYPES) as exc:
            return _error(exc)

    # --- tools ---------------------------------------------------------------------

    def list_accounts(self) -> dict[str, Any]:
        """Accounts the actor may read for this session's purpose (checks are not audited)."""
        def run() -> dict[str, Any]:
            at = self.clock()
            accounts = [
                {"account_id": a.account_id, "kind": a.kind, "owner_scope": a.owner_scope.value}
                for a in self.store.authorizer.accounts(self.actor.household_id)
                if self.store.authorizer.authorize(self.actor, a.account_id, Action.READ, self.purpose, at)
            ]
            return {"accounts": accounts}
        return self._guard(run)

    def query_ledger(
        self, account_ids: list[str], start: str, end: str,
        merchant: str | None = None, include_provisional: bool = False,
    ) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            if not account_ids:
                raise ValueError("account_ids must not be empty")
            txns = self.store.query_transactions(
                self.actor, self.purpose, self.correlation_id, list(account_ids),
                date.fromisoformat(start), date.fromisoformat(end),
                include_provisional=include_provisional, merchant_normalized=merchant, at=self.clock(),
            )
            left_out = self.store.left_out_documents(
                self.actor, self.purpose, self.correlation_id, list(account_ids),
                date.fromisoformat(start), date.fromisoformat(end), include_provisional=include_provisional,
                at=self.clock())
            query_no = len(self._query_notes)
            self._query_notes.append(left_out)
            for t in txns:
                self._seen[t.transaction_id] = t
                self._seen_in.setdefault(t.transaction_id, set()).add(query_no)
            return {"transactions": [_txn_view(t) for t in txns], "provisional_included": include_provisional,
                    **_left_out_view([left_out])}
        return self._guard(run)

    def run_calculation(
        self, fn: str, transaction_ids: list[str], params: dict[str, Any] | None = None,
        allow_provisional: bool = False,
    ) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            if fn not in REGISTRY:
                raise KeyError(f"unknown calculation {fn!r}; allowed: {sorted(REGISTRY)}")
            unknown = [i for i in transaction_ids if i not in self._seen]
            if unknown:
                raise IneligibleInputError(f"transactions not returned by query_ledger in this session: {unknown}")
            r = CalculationRunner(self.store).run(
                fn, [self._seen[i] for i in transaction_ids], actor_id=self.actor.member_id,
                household_id=self.actor.household_id, correlation_id=self.correlation_id,
                params=params or {}, allow_provisional=allow_provisional, acting_agent=self.acting_agent,
            )
            return {"run_id": r.run_id, "fn": r.fn, "output": r.output, "input_hash": r.input_hash,
                    "output_hash": r.output_hash, "library_version": r.library_version,
                    "provisional": r.provisional, "input_ids": r.input_ids,
                    **_left_out_view([self._query_notes[q] for q in sorted(
                        {q for i in r.input_ids for q in self._seen_in.get(i, ())})])}
        return self._guard(run)

    def ingest_statement(self, csv_text: str, source_name: str) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            if self.restrict_ingest:
                digest = hashlib.sha256(csv_text.encode()).hexdigest()
                if self.ingest_csv_sha256 is None or digest != self.ingest_csv_sha256:
                    reason = ("the task supplied no statement to ingest" if self.ingest_csv_sha256 is None
                              else "csv_text does not match the statement the task supplied")
                    self.store.audit.append(
                        correlation_id=self.correlation_id, actor_id=self.actor.member_id,
                        household_id=self.actor.household_id, acting_agent=self.acting_agent,
                        action="ingest_statement", purpose=self.purpose, decision="deny", reason=reason,
                        details={"csv_sha256": digest, "source_name": source_name}, at=self.clock())
                    raise StatementMismatchError(reason)
            r = ingest_statement(self.store, self.categorizer, self.actor, csv_text.encode(), source_name,
                                 purpose=self.purpose, correlation_id=self.correlation_id, at=self.clock(),
                                 allow_pdf=False)  # the agent tool is CSV text only
            return {"document_id": r.document_id, "status": r.status.value,
                    "document_confidence": r.document_confidence, "unverified": r.unverified,
                    "issues": [{"code": i.code, "severity": i.severity, "message": i.message} for i in r.issues]}
        return self._guard(run)
