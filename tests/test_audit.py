import sqlite3

import pytest

from household_agent.audit import AuditIntegrityError, AuditLog


def _log(conn, n=3):
    log = AuditLog(conn)
    for i in range(n):
        log.append(correlation_id="c1", actor_id="alex", household_id="hh", action="read",
                   decision="allow", details={"i": i})
    return log


def test_chain_verifies(conn):
    assert _log(conn).verify() == 3


def test_update_and_delete_are_blocked(conn):
    _log(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE audit_events SET decision = 'deny' WHERE seq = 1")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM audit_events WHERE seq = 1")


def test_out_of_band_tampering_is_detected(conn):
    log = _log(conn)
    conn.execute("DROP TRIGGER audit_no_update")
    conn.execute("UPDATE audit_events SET decision = 'deny' WHERE seq = 2")
    with pytest.raises(AuditIntegrityError, match="seq 2"):
        log.verify()


def test_events_filter_by_correlation(conn):
    log = _log(conn)
    log.append(correlation_id="c2", actor_id="jordan", household_id="hh", action="read", decision="deny")
    assert [e["actor_id"] for e in log.events("c2")] == ["jordan"]
