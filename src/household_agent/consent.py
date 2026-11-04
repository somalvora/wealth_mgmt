"""Persistent consent grants (design §7, "Consent Lifecycle").

Grants and revocations live in append-only SQLite tables: a revocation is a new row, never
an edit. On startup ``ConsentStore`` loads every grant into the ``Authorizer`` so consent
survives restarts. Only the grantor may create or revoke a grant, and only for accounts
they own; every attempt, allowed or denied, is audited.
"""

import json
import sqlite3
from datetime import datetime

from household_agent.audit import AuditLog
from household_agent.authz import POLICY_VERSION, Actor, AuthorizationError, Authorizer, ConsentGrant, Decision
from household_agent.models import Action, Role

_SCHEMA = """
CREATE TABLE IF NOT EXISTS consent_grants (
    grant_id TEXT PRIMARY KEY,
    household_id TEXT NOT NULL,
    grantor_id TEXT NOT NULL,
    grantee_id TEXT NOT NULL,
    account_ids_json TEXT NOT NULL,
    actions_json TEXT NOT NULL,
    purposes_json TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    expires_at TEXT,
    version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS consent_revocations (
    grant_id TEXT PRIMARY KEY REFERENCES consent_grants (grant_id),
    revoked_by TEXT NOT NULL,
    revoked_at TEXT NOT NULL
);
"""

_APPEND_ONLY_TABLES = ("consent_grants", "consent_revocations")


class ConsentStore:
    def __init__(self, conn: sqlite3.Connection, authorizer: Authorizer, audit: AuditLog):
        self._conn = conn
        self._authorizer = authorizer
        self._audit = audit
        conn.executescript(_SCHEMA)
        for table in _APPEND_ONLY_TABLES:
            for op in ("UPDATE", "DELETE"):
                conn.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {table}_no_{op.lower()} BEFORE {op} ON {table} "
                    f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END;"
                )
        conn.commit()
        for g in self.grants():
            authorizer.add_grant(g)

    def grants(self) -> list[ConsentGrant]:
        cur = self._conn.execute(
            "SELECT g.grant_id, g.household_id, g.grantor_id, g.grantee_id, g.account_ids_json, "
            "g.actions_json, g.purposes_json, g.effective_at, g.expires_at, g.version, r.revoked_at "
            "FROM consent_grants g LEFT JOIN consent_revocations r USING (grant_id) ORDER BY g.grant_id"
        )
        return [
            ConsentGrant(
                grant_id=r[0], household_id=r[1], grantor_id=r[2], grantee_id=r[3],
                account_ids=frozenset(json.loads(r[4])),
                actions=frozenset(Action(a) for a in json.loads(r[5])),
                purposes=frozenset(json.loads(r[6])),
                effective_at=datetime.fromisoformat(r[7]),
                expires_at=datetime.fromisoformat(r[8]) if r[8] else None,
                version=r[9],
                revoked_at=datetime.fromisoformat(r[10]) if r[10] else None,
            )
            for r in cur
        ]

    def _record(self, actor: Actor, action: str, grant_id: str, decision: Decision,
                correlation_id: str, at: datetime) -> None:
        self._audit.append(
            correlation_id=correlation_id, actor_id=actor.member_id, household_id=actor.household_id,
            action=action, resource=f"consent:{grant_id}", decision="allow" if decision else "deny",
            reason=decision.reason, policy_version=POLICY_VERSION, at=at,
        )
        if not decision:
            raise AuthorizationError(decision)

    def _check_grant(self, actor: Actor, grant: ConsentGrant) -> Decision:
        if actor.role in (Role.DELEGATE, Role.DEPENDENT):
            return Decision(False, f"{actor.role} may not grant consent")
        if actor.member_id != grant.grantor_id or actor.household_id != grant.household_id:
            return Decision(False, "only the grantor may grant their own consent")
        if grant.grantee_id == grant.grantor_id:
            return Decision(False, "cannot grant consent to oneself")
        if not grant.account_ids or not grant.actions or not grant.purposes:
            return Decision(False, "grant must name accounts, actions and purposes")
        for account_id in sorted(grant.account_ids):
            account = self._authorizer.account(account_id)
            if account is None or actor.member_id not in account.owner_member_ids:
                return Decision(False, f"grantor does not own {account_id}")
        if grant.expires_at is not None and grant.expires_at <= grant.effective_at:
            return Decision(False, "grant expires before it takes effect")
        if grant.revoked_at is not None:
            return Decision(False, "new grants cannot be pre-revoked")
        if self._conn.execute("SELECT 1 FROM consent_grants WHERE grant_id = ?", (grant.grant_id,)).fetchone():
            return Decision(False, f"grant {grant.grant_id} already exists")
        return Decision(True, "grantor owns every account in the grant")

    def grant(self, actor: Actor, grant: ConsentGrant, *, correlation_id: str, at: datetime) -> None:
        self._record(actor, "consent_grant", grant.grant_id, self._check_grant(actor, grant), correlation_id, at)
        self._conn.execute(
            "INSERT INTO consent_grants VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                grant.grant_id, grant.household_id, grant.grantor_id, grant.grantee_id,
                json.dumps(sorted(grant.account_ids)), json.dumps(sorted(a.value for a in grant.actions)),
                json.dumps(sorted(grant.purposes)), grant.effective_at.isoformat(),
                grant.expires_at.isoformat() if grant.expires_at else None, grant.version, at.isoformat(),
            ),
        )
        self._conn.commit()
        self._authorizer.add_grant(grant)

    def revoke(self, actor: Actor, grant_id: str, *, correlation_id: str, at: datetime) -> None:
        row = self._conn.execute(
            "SELECT g.grantor_id, r.grant_id FROM consent_grants g "
            "LEFT JOIN consent_revocations r USING (grant_id) WHERE g.grant_id = ?",
            (grant_id,),
        ).fetchone()
        if row is None:
            decision = Decision(False, f"unknown grant {grant_id}")
        elif row[0] != actor.member_id:
            decision = Decision(False, "only the grantor may revoke a grant")
        elif row[1] is not None:
            decision = Decision(False, f"grant {grant_id} is already revoked")
        else:
            decision = Decision(True, "grantor revoked consent")
        self._record(actor, "consent_revoke", grant_id, decision, correlation_id, at)
        self._conn.execute("INSERT INTO consent_revocations VALUES (?, ?, ?)", (grant_id, actor.member_id, at.isoformat()))
        self._conn.commit()
        self._authorizer.revoke(grant_id, at)
