import sqlite3
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from household_agent.audit import AuditLog
from household_agent.authz import AuthorizationError, Authorizer
from household_agent.consent import ConsentStore
from household_agent.models import Action
from household_agent.synthetic import ACCOUNTS, SOMAL, CASEY_CPA, SAYALI, VIAAN, cpa_grant

NOW = datetime(2026, 11, 3, tzinfo=UTC)
EXPIRES = datetime(2027, 4, 15, tzinfo=UTC)
RENTAL = "acct_rental_checking"


def _store(conn):
    authz = Authorizer(ACCOUNTS)
    return ConsentStore(conn, authz, AuditLog(conn)), authz


def _cpa_reads(authz, at=NOW):
    return authz.authorize(CASEY_CPA, RENTAL, Action.READ, "tax_preparation", at)


def _both_owners_grant(store):
    somal = cpa_grant(EXPIRES)
    sayali = replace(somal, grant_id="grant_cpa_rental_sayali", grantor_id="sayali", version="cv_2")
    store.grant(SOMAL, somal, correlation_id="c1", at=NOW)
    store.grant(SAYALI, sayali, correlation_id="c1", at=NOW)


def test_joint_account_needs_every_owner_grant(conn):
    store, authz = _store(conn)
    store.grant(SOMAL, cpa_grant(EXPIRES), correlation_id="c1", at=NOW)
    assert not _cpa_reads(authz)
    sayali = replace(cpa_grant(EXPIRES), grant_id="g_j", grantor_id="sayali", version="cv_2")
    store.grant(SAYALI, sayali, correlation_id="c1", at=NOW)
    assert _cpa_reads(authz).consent_version == "sayali:cv_2,somal:cv_1"


def test_grants_survive_restart(conn):
    store, _ = _store(conn)
    _both_owners_grant(store)
    _, fresh_authz = _store(conn)  # new process, same database
    assert _cpa_reads(fresh_authz)
    assert not _cpa_reads(fresh_authz, at=EXPIRES)  # expiry still applies after reload


def test_revocation_persists(conn):
    store, authz = _store(conn)
    _both_owners_grant(store)
    store.revoke(SAYALI, "grant_cpa_rental_sayali", correlation_id="c2", at=NOW)
    assert not _cpa_reads(authz)
    _, fresh_authz = _store(conn)
    assert not _cpa_reads(fresh_authz)


@pytest.mark.parametrize(
    "actor, grant, reason",
    [
        (SAYALI, cpa_grant(EXPIRES), "only the grantor"),
        (SOMAL, replace(cpa_grant(EXPIRES), account_ids=frozenset({"acct_sayali_card"})), "does not own"),
        (SOMAL, replace(cpa_grant(EXPIRES), account_ids=frozenset({"acct_unknown"})), "does not own"),
        (CASEY_CPA, replace(cpa_grant(EXPIRES), grantor_id="casey_cpa"), "may not grant"),
        (VIAAN, replace(cpa_grant(EXPIRES), grantor_id="viaan"), "may not grant"),
        (SOMAL, replace(cpa_grant(EXPIRES), grantee_id="somal"), "oneself"),
        (SOMAL, replace(cpa_grant(EXPIRES), expires_at=datetime(2025, 1, 1, tzinfo=UTC)), "expires before"),
        (SOMAL, replace(cpa_grant(EXPIRES), purposes=frozenset()), "must name"),
        (SOMAL, replace(cpa_grant(EXPIRES), revoked_at=NOW), "pre-revoked"),
    ],
)
def test_invalid_grants_denied_and_audited(conn, actor, grant, reason):
    store, _ = _store(conn)
    with pytest.raises(AuthorizationError, match=reason):
        store.grant(actor, grant, correlation_id="bad", at=NOW)
    assert store.grants() == []
    [event] = AuditLog(conn).events("bad")
    assert event["decision"] == "deny" and event["action"] == "consent_grant"


def test_duplicate_grant_id_denied(conn):
    store, _ = _store(conn)
    store.grant(SOMAL, cpa_grant(EXPIRES), correlation_id="c1", at=NOW)
    with pytest.raises(AuthorizationError, match="already exists"):
        store.grant(SOMAL, cpa_grant(EXPIRES), correlation_id="c1", at=NOW)


def test_only_grantor_can_revoke_once(conn):
    store, authz = _store(conn)
    _both_owners_grant(store)
    with pytest.raises(AuthorizationError, match="only the grantor"):
        store.revoke(SOMAL, "grant_cpa_rental_sayali", correlation_id="c3", at=NOW)
    assert _cpa_reads(authz)
    with pytest.raises(AuthorizationError, match="unknown grant"):
        store.revoke(SOMAL, "nope", correlation_id="c3", at=NOW)
    store.revoke(SOMAL, "grant_cpa_rental", correlation_id="c3", at=NOW)
    with pytest.raises(AuthorizationError, match="already revoked"):
        store.revoke(SOMAL, "grant_cpa_rental", correlation_id="c3", at=NOW)


def test_consent_tables_are_append_only(conn):
    store, _ = _store(conn)
    _both_owners_grant(store)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE consent_grants SET expires_at = NULL")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM consent_grants")
    assert AuditLog(conn).verify() == 2
