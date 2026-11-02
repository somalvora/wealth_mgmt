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
