"""One lock for the audit log, ledger and consent store (they share one SQLite connection)."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from household_agent.agents import AgentResult, DispatchItem, RoutingDecision, Supervisor
from household_agent.audit import AuditLog
from household_agent.authz import AuthorizationError, Authorizer
from household_agent.categorize import Categorizer
from household_agent.consent import ConsentStore
from household_agent.ledger import DuplicateDocumentError, LedgerStore
from household_agent.pipeline import ingest_statement
from household_agent.synthetic import ACCOUNTS, ALEX, DEMO_AS_OF, RULES, cpa_grant, demo_statements
from household_agent.vertical_slice import _Env

CSV = demo_statements()["2026_10_alex_checking.csv"].encode()


def _stores():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    authorizer, audit = Authorizer(ACCOUNTS), AuditLog(conn)
    return conn, LedgerStore(conn, authorizer, audit), ConsentStore(conn, authorizer, audit)


def _race_ingests(store, n):
    barrier, outcomes = threading.Barrier(n, timeout=10), []

    def go(i):
        barrier.wait()
        try:
            ingest_statement(store, Categorizer(RULES), ALEX, CSV, "s.csv", purpose="statement_ingestion",
                             correlation_id=f"c{i}", at=DEMO_AS_OF)
            outcomes.append("ok")
        except DuplicateDocumentError:
            outcomes.append("duplicate")
        except Exception as exc:  # noqa: BLE001  anything else (e.g. IntegrityError) is a failure of the lock
            outcomes.append(type(exc).__name__)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return outcomes


def _audit_actions(audit):
    actions = [e["action"] for e in audit.events()]
    return {a: actions.count(a) for a in set(actions)}


def test_concurrent_ingests_of_one_statement_exactly_one_wins_and_nothing_is_lost():
    conn, store, _ = _stores()
    outcomes = _race_ingests(store, 8)
    assert sorted(outcomes) == ["duplicate"] * 7 + ["ok"]
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    # every attempt was authorized and audited; only the winner ingested; the chain has no gap
    assert _audit_actions(store.audit) == {"upload": 8, "ingest_document": 1}
    assert store.audit.verify() == 9


@settings(max_examples=6, deadline=None)
@given(st.integers(min_value=2, max_value=6))
def test_any_number_of_racing_ingests_yields_one_document_and_a_valid_chain(n):
    conn, store, _ = _stores()
    assert sorted(_race_ingests(store, n)) == ["duplicate"] * (n - 1) + ["ok"]
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert store.audit.verify() == n + 1


def test_a_failing_ingest_rolls_back_fully_while_other_threads_append_audit_events(monkeypatch):
    conn, store, _ = _stores()
    calls = []
    real = store._insert_txn

    def explode_on_third_row(txn):
        calls.append(txn)
        if len(calls) == 3:
            raise RuntimeError("disk full")
        real(txn)

    monkeypatch.setattr(store, "_insert_txn", explode_on_third_row)
    stop, appended = threading.Event(), []

    def appender():
        while not stop.is_set() or len(appended) < 100:
            store.audit.append(correlation_id="bg", actor_id="alex", household_id="hh_demo", action="bg",
                               decision="info")
            appended.append(1)

    t = threading.Thread(target=appender)
    t.start()
    try:
        with pytest.raises(RuntimeError, match="disk full"):
            ingest_statement(store, Categorizer(RULES), ALEX, CSV, "s.csv", purpose="statement_ingestion",
                             correlation_id="c", at=DEMO_AS_OF)
    finally:
        stop.set()
        t.join()
    for table in ("documents", "statement_balances", "raw_rows", "transactions"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table
    actions = _audit_actions(store.audit)
    assert actions["bg"] == len(appended) and actions["upload"] == 1 and "ingest_document" not in actions
    assert store.audit.verify() == len(appended) + 1  # no background row was erased by the rollback


def test_the_failed_statement_can_be_ingested_again_afterwards(monkeypatch):
    _, store, _ = _stores()
    real = store._insert_txn
    monkeypatch.setattr(store, "_insert_txn", lambda t: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        ingest_statement(store, Categorizer(RULES), ALEX, CSV, "s.csv", purpose="statement_ingestion",
                         correlation_id="c1", at=DEMO_AS_OF)
    monkeypatch.setattr(store, "_insert_txn", real)
    ingest_statement(store, Categorizer(RULES), ALEX, CSV, "s.csv", purpose="statement_ingestion",
                     correlation_id="c2", at=DEMO_AS_OF)
    assert store.audit.verify() == 3


def test_the_three_stores_share_one_reentrant_lock():
    _, store, consent = _stores()
    assert consent._audit.lock is store.audit.lock
    with store.audit.lock, store.audit.lock:  # re-entrant: a store may append to the audit log while holding it
        store.audit.append(correlation_id="c", actor_id="alex", household_id="hh_demo", action="x", decision="info")


def test_concurrent_identical_consent_grants_one_succeeds_and_the_rest_are_refused_cleanly():
    conn, _, consent = _stores()
    grant = cpa_grant(datetime(2027, 4, 15, tzinfo=UTC))
    barrier, outcomes = threading.Barrier(6, timeout=10), []

    def go(i):
        barrier.wait()
        try:
            consent.grant(ALEX, grant, correlation_id=f"c{i}", at=datetime(2026, 11, 3, tzinfo=UTC))
            outcomes.append("ok")
        except AuthorizationError:
            outcomes.append("refused")
        except Exception as exc:  # noqa: BLE001
            outcomes.append(type(exc).__name__)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(outcomes) == ["ok"] + ["refused"] * 5
    assert conn.execute("SELECT COUNT(*) FROM consent_grants").fetchone()[0] == 1
    assert consent._audit.verify() == 6


def test_a_parallel_batch_on_the_vertical_slice_connection_does_not_raise_programming_error(tmp_path):
    env = _Env.open(tmp_path)
    try:
        def specialist(a):  # touches the shared connection from a worker thread
            env.store.audit.append(correlation_id=a.correlation_id, actor_id="alex", household_id="hh_demo",
                                   action="probe", decision="info")
            env.store.historical_row_hashes("acct_alex_checking")
            return AgentResult(task_id=a.task_id, correlation_id=a.correlation_id, agent=a.assigned_agent,
                               status="completed")

        plan = [DispatchItem(agent="ledger_analyst", task="a"),
                DispatchItem(agent="policy_compliance", task="b", depends_on_ledger=False)]
        decision = RoutingDecision(intent="historical_comparison_and_tax_context", dispatch_plan=plan)
        sup = Supervisor(ALEX, lambda _r: (decision, "test"),
                         {"ledger_analyst": specialist, "policy_compliance": specialist}, env.store.audit,
                         clock=lambda: DEMO_AS_OF)
        out = sup.run("x", correlation_id="j_slice")
        assert [r["status"] for r in out["results"]] == ["completed", "completed"]
        assert env.store.audit.verify() > 0
        with ThreadPoolExecutor(2) as pool:  # and a plain cross-thread read
            assert pool.submit(lambda: env.conn.execute("SELECT 1").fetchone()).result() == (1,)
    finally:
        env.conn.close()
