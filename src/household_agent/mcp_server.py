"""``household-finance-tools`` MCP server (design §3.2, §4.2): ``python -m household_agent.mcp_server``.

A thin stdio wrapper over ``ToolSession``. The host that launches the server has already
authenticated the user and passes identity through the environment, never through tool
arguments:

- ``HH_LEDGER_DB``  path to the ledger SQLite file (required)
- ``HH_ACTOR``      member ID of the authenticated user (required)
- ``HH_PURPOSE``    purpose of this session, e.g. ``historical_comparison`` (required)

One server process serves one actor and purpose. Accounts and members come from the
synthetic household until account onboarding exists.
"""

import os
import sqlite3
import sys
import uuid
from typing import Any

from mcp.server.fastmcp import FastMCP

from household_agent.audit import AuditLog
from household_agent.authz import Authorizer
from household_agent.categorize import Categorizer
from household_agent.consent import ConsentStore
from household_agent.ledger import LedgerStore
from household_agent.synthetic import ACCOUNTS, ALEX, CASEY_CPA, JORDAN, RULES, SAM
from household_agent.tools import ToolSession

SERVER_NAME = "household-finance-tools"
MEMBERS = {a.member_id: a for a in (ALEX, JORDAN, CASEY_CPA, SAM)}


def build_server(session: ToolSession) -> FastMCP:
    server = FastMCP(SERVER_NAME, instructions=(
        "Household ledger tools. Amounts are integer minor units (cents). Never compute money "
        "yourself: use run_calculation on transaction IDs returned by query_ledger."
    ))

    @server.tool()
    def list_accounts() -> dict[str, Any]:
        """List the accounts you may read for this session's purpose."""
        return session.call("list_accounts")

    @server.tool()
    def query_ledger(account_ids: list[str], start: str, end: str, merchant: str | None = None,
                     include_provisional: bool = False) -> dict[str, Any]:
        """Return transactions (latest versions) for accounts in [start, end] (ISO dates).
        Reconciled data only unless include_provisional; unreconciled data is never returned."""
        return session.call("query_ledger", {"account_ids": account_ids, "start": start, "end": end,
                                             "merchant": merchant, "include_provisional": include_provisional})

    @server.tool()
    def run_calculation(fn: str, transaction_ids: list[str], params: dict[str, Any] | None = None,
                        allow_provisional: bool = False) -> dict[str, Any]:
        """Run an allowlisted finance function (cash_flow, category_totals,
        variance_between_transactions, recurring_drift) on transactions from query_ledger."""
        return session.call("run_calculation", {"fn": fn, "transaction_ids": transaction_ids,
                                                "params": params, "allow_provisional": allow_provisional})

    @server.tool()
    def ingest_statement(csv_text: str, source_name: str) -> dict[str, Any]:
        """Parse, reconcile and store an hh_statement_csv_v1 statement you own."""
        return session.call("ingest_statement", {"csv_text": csv_text, "source_name": source_name})

    return server


def session_from_env(env: dict[str, str] | None = None) -> ToolSession:
    env = dict(os.environ if env is None else env)
    missing = [k for k in ("HH_LEDGER_DB", "HH_ACTOR", "HH_PURPOSE") if not env.get(k)]
    if missing:
        raise SystemExit(f"{SERVER_NAME}: missing environment {missing}")
    actor = MEMBERS.get(env["HH_ACTOR"])
    if actor is None:
        raise SystemExit(f"{SERVER_NAME}: unknown actor {env['HH_ACTOR']!r}")
    conn = sqlite3.connect(env["HH_LEDGER_DB"], check_same_thread=False)
    authorizer, audit = Authorizer(ACCOUNTS), AuditLog(conn)
    ConsentStore(conn, authorizer, audit)  # loads persisted grants
    return ToolSession(LedgerStore(conn, authorizer, audit), Categorizer(RULES), actor, env["HH_PURPOSE"],
                       correlation_id=f"mcp_{uuid.uuid4().hex[:12]}", acting_agent="mcp_client")


def main() -> None:
    build_server(session_from_env()).run("stdio")


if __name__ == "__main__":
    sys.exit(main())
