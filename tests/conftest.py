import sqlite3

import pytest

from household_agent.audit import AuditLog
from household_agent.authz import Authorizer
from household_agent.categorize import Categorizer
from household_agent.ledger import LedgerStore
from household_agent.synthetic import ACCOUNTS, RULES


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def store(conn):
    return LedgerStore(conn, Authorizer(ACCOUNTS), AuditLog(conn))


@pytest.fixture
def categorizer():
    return Categorizer(RULES)


@pytest.fixture
def seeded_store(store, categorizer):
    """Ledger with every demo statement ingested by its owner (as in the demo)."""
    from household_agent.pipeline import ingest_statement
    from household_agent.synthetic import SOMAL, DEMO_AS_OF, SAYALI, demo_statements

    for name, content in demo_statements().items():
        actor = SAYALI if "sayali_card" in name else SOMAL
        ingest_statement(store, categorizer, actor, content.encode(), name,
                         purpose="statement_ingestion", correlation_id="seed", at=DEMO_AS_OF)
    return store
