"""Append-only, hash-chained audit log (design §7, "Tamper-Evident Audit Record").

Each event stores the hash of the previous event. SQLite triggers reject UPDATE/DELETE,
and ``verify()`` recomputes the chain so any out-of-band edit is detected.
"""

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

GENESIS_HASH = "0" * 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    correlation_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    acting_agent TEXT,
    household_id TEXT NOT NULL,
    action TEXT NOT NULL,
    purpose TEXT,
    resource TEXT,
    decision TEXT NOT NULL,
    reason TEXT,
    policy_version TEXT,
    consent_version TEXT,
    details_json TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
"""

_HASHED_FIELDS = (
    "event_id", "correlation_id", "ts", "actor_id", "acting_agent", "household_id", "action",
    "purpose", "resource", "decision", "reason", "policy_version", "consent_version", "details_json",
)


class AuditIntegrityError(RuntimeError):
    pass


def _event_hash(fields: dict[str, Any], prev_hash: str) -> str:
    payload = json.dumps({k: fields[k] for k in _HASHED_FIELDS}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((prev_hash + payload).encode()).hexdigest()


class AuditLog:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        conn.executescript(_SCHEMA)

    def append(
        self,
        *,
        correlation_id: str,
        actor_id: str,
        household_id: str,
        action: str,
        decision: str,  # "allow", "deny", or "info"
        purpose: str | None = None,
        resource: str | None = None,
        reason: str | None = None,
        acting_agent: str | None = None,
        policy_version: str | None = None,
        consent_version: str | None = None,
        details: dict[str, Any] | None = None,
        at: datetime | None = None,
    ) -> str:
        fields = {
            "event_id": f"evt_{uuid.uuid4().hex}",
            "correlation_id": correlation_id,
            "ts": (at or datetime.now(UTC)).isoformat(),
            "actor_id": actor_id,
            "acting_agent": acting_agent,
            "household_id": household_id,
            "action": action,
            "purpose": purpose,
            "resource": resource,
            "decision": decision,
            "reason": reason,
            "policy_version": policy_version,
            "consent_version": consent_version,
            "details_json": json.dumps(details or {}, sort_keys=True, default=str),
        }
        row = self._conn.execute("SELECT hash FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
        prev_hash = row[0] if row else GENESIS_HASH
        fields["prev_hash"] = prev_hash
        fields["hash"] = _event_hash(fields, prev_hash)
        cols = ", ".join(fields)
        self._conn.execute(
            f"INSERT INTO audit_events ({cols}) VALUES ({', '.join('?' * len(fields))})",
            tuple(fields.values()),
        )
        self._conn.commit()
        return fields["event_id"]

    def verify(self) -> int:
        """Recompute the chain; returns the number of events or raises ``AuditIntegrityError``."""
        prev_hash, count = GENESIS_HASH, 0
        cur = self._conn.execute(
            f"SELECT seq, {', '.join(_HASHED_FIELDS)}, prev_hash, hash FROM audit_events ORDER BY seq"
        )
        for row in cur:
            seq, values, stored_prev, stored_hash = row[0], row[1:-2], row[-2], row[-1]
            fields = dict(zip(_HASHED_FIELDS, values, strict=True))
            if stored_prev != prev_hash or _event_hash(fields, prev_hash) != stored_hash:
                raise AuditIntegrityError(f"audit chain broken at seq {seq}")
            prev_hash, count = stored_hash, count + 1
        return count

    def events(self, correlation_id: str | None = None) -> list[dict[str, Any]]:
        sql = f"SELECT seq, {', '.join(_HASHED_FIELDS)} FROM audit_events"
        params: tuple = ()
        if correlation_id:
            sql += " WHERE correlation_id = ?"
            params = (correlation_id,)
        cur = self._conn.execute(sql + " ORDER BY seq", params)
        return [dict(zip(("seq", *_HASHED_FIELDS), r, strict=True)) for r in cur]
