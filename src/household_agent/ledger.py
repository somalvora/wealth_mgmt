"""Canonical ledger and document store (design §3.1, §6).

SQLite for Phase 1. Raw rows, transactions, and calculation runs are append-only (triggers
reject UPDATE/DELETE); a correction inserts a new transaction version. Every write and read
is authorized first and audited, whether allowed or denied.
"""

import json
import sqlite3
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, date, datetime

from household_agent.audit import AuditLog
from household_agent.authz import Actor, AuthorizationError, Authorizer, Decision
from household_agent.categorize import Categorized
from household_agent.ingest import ParsedStatement
from household_agent.models import Action, ReconciliationStatus, Transaction
from household_agent.reconcile import CONFIDENCE_THRESHOLD, ReconciliationResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    household_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_hash TEXT NOT NULL UNIQUE,
    parser_version TEXT NOT NULL,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    currency TEXT NOT NULL,
    reconciliation_status TEXT NOT NULL,
    document_confidence REAL NOT NULL,
    unverified INTEGER NOT NULL,
    issues_json TEXT NOT NULL,
    correlation_id TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS statement_balances (
    document_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    opening_balance_minor INTEGER NOT NULL,
    closing_balance_minor INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS raw_rows (
    document_id TEXT NOT NULL,
    line_number INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    PRIMARY KEY (document_id, line_number)
);
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    household_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    owner_scope TEXT NOT NULL,
    posted_at TEXT NOT NULL,
    amount_minor INTEGER NOT NULL,
    currency TEXT NOT NULL,
    merchant_raw TEXT NOT NULL,
    merchant_normalized TEXT NOT NULL,
    category TEXT NOT NULL,
    category_confidence REAL NOT NULL,
    document_confidence REAL NOT NULL,
    source_document_id TEXT NOT NULL,
    source_locator TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    reconciliation_status TEXT NOT NULL,
    row_hash TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    correction_reason TEXT,
    PRIMARY KEY (transaction_id, version)
);
CREATE INDEX IF NOT EXISTS idx_txn_account ON transactions (account_id, posted_at);
CREATE TABLE IF NOT EXISTS calculation_runs (
    run_id TEXT PRIMARY KEY,
    fn TEXT NOT NULL,
    library_version TEXT NOT NULL,
    input_ids_json TEXT NOT NULL,
    params_json TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    output_json TEXT NOT NULL,
    output_hash TEXT NOT NULL,
    provisional INTEGER NOT NULL,
    correlation_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

_APPEND_ONLY_TABLES = ("documents", "statement_balances", "raw_rows", "transactions", "calculation_runs")

_TXN_COLUMNS = tuple(Transaction.model_fields)


class DuplicateDocumentError(ValueError):
    pass


class LedgerStore:
    def __init__(self, conn: sqlite3.Connection, authorizer: Authorizer, audit: AuditLog):
        self.conn = conn
        self.authorizer = authorizer
        self.audit = audit
        conn.executescript(_SCHEMA)
        for table in _APPEND_ONLY_TABLES:
            for op in ("UPDATE", "DELETE"):
                conn.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {table}_no_{op.lower()} BEFORE {op} ON {table} "
                    f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END;"
                )
        conn.commit()

    # --- authorization helper -------------------------------------------------

    def _require(
        self, actor: Actor, account_id: str, action: Action, purpose: str, correlation_id: str, at: datetime
    ) -> Decision:
        decision = self.authorizer.authorize(actor, account_id, action, purpose, at)
        self.audit.append(
            correlation_id=correlation_id,
            actor_id=actor.member_id,
            household_id=actor.household_id,
            action=action.value,
            purpose=purpose,
            resource=f"account:{account_id}",
            decision="allow" if decision else "deny",
            reason=decision.reason,
            policy_version=decision.policy_version,
            consent_version=decision.consent_version,
            at=at,
        )
        if not decision:
            raise AuthorizationError(decision)
        return decision

    # --- writes -----------------------------------------------------------------

    def ingest_document(
        self,
        actor: Actor,
        purpose: str,
        correlation_id: str,
        parsed: ParsedStatement,
        categorized: Sequence[Categorized],
        recon: ReconciliationResult,
        at: datetime | None = None,
    ) -> str:
        at = at or datetime.now(UTC)
        h = parsed.header
        self._require(actor, h.account_id, Action.UPLOAD, purpose, correlation_id, at)
        account = self.authorizer.account(h.account_id)
        assert account is not None  # authorize() denies unknown accounts
        if h.household_id != account.household_id:
            raise ValueError("statement household does not match account household")
        if self.conn.execute("SELECT 1 FROM documents WHERE source_hash = ?", (parsed.source_hash,)).fetchone():
            raise DuplicateDocumentError(f"{parsed.source_name} was already ingested")

        document_id = f"doc_{uuid.uuid4().hex[:16]}"
        with self.conn:
            self.conn.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    document_id, h.household_id, h.account_id, parsed.source_name, parsed.source_hash,
                    parsed.parser_version, h.period_start.isoformat(), h.period_end.isoformat(), h.currency,
                    recon.status.value, recon.document_confidence, int(recon.unverified),
                    json.dumps([asdict(i) for i in recon.issues]), correlation_id, at.isoformat(),
                ),
            )
            self.conn.execute(
                "INSERT INTO statement_balances VALUES (?,?,?,?,?,?)",
                (document_id, h.account_id, h.period_start.isoformat(), h.period_end.isoformat(),
                 h.opening_balance_minor, h.closing_balance_minor),
            )
            raw = [(r.line_number, r.raw_text) for r in parsed.rows] + [(e.line_number, e.raw_text) for e in parsed.row_errors]
            self.conn.executemany(
                "INSERT INTO raw_rows VALUES (?,?,?)", [(document_id, n, t) for n, t in sorted(raw)]
            )
            for row, cat, rh in zip(parsed.rows, categorized, recon.row_hashes, strict=True):
                txn = Transaction(
                    transaction_id=f"txn_{rh[:16]}",
                    household_id=h.household_id,
                    account_id=h.account_id,
                    owner_scope=account.owner_scope,
                    posted_at=row.posted_at,
                    amount_minor=row.amount_minor,
                    currency=h.currency,
                    merchant_raw=row.description,
                    merchant_normalized=cat.merchant_normalized,
                    category=cat.category,
                    category_confidence=cat.confidence,
                    document_confidence=recon.document_confidence,
                    source_document_id=document_id,
                    source_locator=f"line={row.line_number}",
                    parser_version=parsed.parser_version,
                    reconciliation_status=recon.status,
                    row_hash=rh,
                    ingested_at=at,
                )
                self._insert_txn(txn)
        self.audit.append(
            correlation_id=correlation_id, actor_id=actor.member_id, household_id=actor.household_id,
            action="ingest_document", decision="info", resource=f"document:{document_id}", at=at,
            details={
                "source_hash": parsed.source_hash, "status": recon.status.value,
                "document_confidence": recon.document_confidence, "unverified": recon.unverified,
                "rows": len(parsed.rows), "row_errors": len(parsed.row_errors),
            },
        )
        return document_id

    def correct_transaction(
        self,
        actor: Actor,
        purpose: str,
        correlation_id: str,
        transaction_id: str,
        reason: str,
        *,
        category: str | None = None,
        merchant_normalized: str | None = None,
        at: datetime | None = None,
    ) -> Transaction:
        at = at or datetime.now(UTC)
        current = self._latest(transaction_id)
        self._require(actor, current.account_id, Action.WRITE, purpose, correlation_id, at)
        changes: dict = {"version": current.version + 1, "correction_reason": reason, "ingested_at": at}
        if category is not None:
            changes |= {"category": category, "category_confidence": 1.0}
        if merchant_normalized is not None:
            changes["merchant_normalized"] = merchant_normalized
        corrected = current.model_copy(update=changes)
        with self.conn:
            self._insert_txn(corrected)
        return corrected

    def record_calculation_run(self, row: dict) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO calculation_runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row["run_id"], row["fn"], row["library_version"], json.dumps(row["input_ids"]),
                    json.dumps(row["params"], sort_keys=True, default=str), row["input_hash"],
                    json.dumps(row["output"], sort_keys=True), row["output_hash"], int(row["provisional"]),
                    row["correlation_id"], row["created_at"],
                ),
            )

    # --- reads ------------------------------------------------------------------

    def historical_row_hashes(self, account_id: str) -> set[str]:
        cur = self.conn.execute("SELECT DISTINCT row_hash FROM transactions WHERE account_id = ?", (account_id,))
        return {r[0] for r in cur}

    def query_transactions(
        self,
        actor: Actor,
        purpose: str,
        correlation_id: str,
        account_ids: Sequence[str],
        start: date,
        end: date,
        *,
        include_provisional: bool = False,
        merchant_normalized: str | None = None,
        at: datetime | None = None,
    ) -> list[Transaction]:
        """Latest version of each authorized transaction in the period.

        By default only ``reconciled`` rows from verified documents are returned. With
        ``include_provisional`` the caller also receives ``qualified`` and ``unverified`` rows,
        which must be labeled provisional downstream. ``unreconciled`` rows are never returned.
        """
        at = at or datetime.now(UTC)
        for account_id in account_ids:
            self._require(actor, account_id, Action.READ, purpose, correlation_id, at)

        allowed = [ReconciliationStatus.RECONCILED.value]
        if include_provisional:
            allowed.append(ReconciliationStatus.QUALIFIED.value)
        sql = f"""
            SELECT {', '.join('t.' + c for c in _TXN_COLUMNS)} FROM transactions t
            JOIN (SELECT transaction_id, MAX(version) v FROM transactions GROUP BY transaction_id) latest
              ON latest.transaction_id = t.transaction_id AND latest.v = t.version
            WHERE t.account_id IN ({','.join('?' * len(account_ids))})
              AND t.posted_at BETWEEN ? AND ?
              AND t.reconciliation_status IN ({','.join('?' * len(allowed))})
        """
        params: list = [*account_ids, start.isoformat(), end.isoformat(), *allowed]
        if not include_provisional:
            sql += " AND t.document_confidence >= ?"
            params.append(CONFIDENCE_THRESHOLD)
        if merchant_normalized:
            sql += " AND t.merchant_normalized = ?"
            params.append(merchant_normalized)
        cur = self.conn.execute(sql + " ORDER BY t.posted_at, t.transaction_id", params)
        return [Transaction(**dict(zip(_TXN_COLUMNS, r, strict=True))) for r in cur]

    def verified_statements(
        self,
        actor: Actor,
        purpose: str,
        correlation_id: str,
        account_ids: Sequence[str],
        start: date,
        end: date,
        *,
        at: datetime | None = None,
    ) -> list[dict]:
        """Statements fully inside ``[start, end]`` that passed reconciliation, newest first.

        Same eligibility as ``query_transactions``: ``reconciled`` and at or above the confidence
        threshold. A statement whose stated closing balance did not reconcile never supplies a
        balance.
        """
        at = at or datetime.now(UTC)
        for account_id in account_ids:
            self._require(actor, account_id, Action.READ, purpose, correlation_id, at)
        cur = self.conn.execute(
            f"""
            SELECT b.document_id, b.account_id, b.period_start, b.period_end,
                   b.opening_balance_minor, b.closing_balance_minor
            FROM statement_balances b JOIN documents d ON d.document_id = b.document_id
            WHERE b.account_id IN ({','.join('?' * len(account_ids))})
              AND b.period_start >= ? AND b.period_end <= ?
              AND d.reconciliation_status = ? AND d.document_confidence >= ?
            ORDER BY b.period_end DESC, b.document_id
            """,
            [*account_ids, start.isoformat(), end.isoformat(), ReconciliationStatus.RECONCILED.value,
             CONFIDENCE_THRESHOLD],
        )
        cols = ("document_id", "account_id", "period_start", "period_end", "opening_balance_minor",
                "closing_balance_minor")
        return [dict(zip(cols, r, strict=True)) for r in cur]

    def document_status(self, document_id: str) -> dict:
        cur = self.conn.execute(
            "SELECT reconciliation_status, document_confidence, unverified, issues_json FROM documents WHERE document_id = ?",
            (document_id,),
        )
        status, confidence, unverified, issues = cur.fetchone()
        return {"status": status, "document_confidence": confidence, "unverified": bool(unverified), "issues": json.loads(issues)}

    # --- internals ----------------------------------------------------------------

    def _insert_txn(self, txn: Transaction) -> None:
        values = txn.model_dump(mode="json")
        self.conn.execute(
            f"INSERT INTO transactions ({', '.join(_TXN_COLUMNS)}) VALUES ({', '.join('?' * len(_TXN_COLUMNS))})",
            tuple(values[c] for c in _TXN_COLUMNS),
        )

    def _latest(self, transaction_id: str) -> Transaction:
        cur = self.conn.execute(
            f"SELECT {', '.join(_TXN_COLUMNS)} FROM transactions WHERE transaction_id = ? ORDER BY version DESC LIMIT 1",
            (transaction_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(transaction_id)
        return Transaction(**dict(zip(_TXN_COLUMNS, row, strict=True)))
