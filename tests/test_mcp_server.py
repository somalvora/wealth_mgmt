import asyncio
import json
import os
import sqlite3
import sys

import pytest
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from household_agent.audit import AuditLog
from household_agent.authz import Authorizer
from household_agent.categorize import Categorizer
from household_agent.ledger import LedgerStore
from household_agent.mcp_server import build_server, session_from_env
from household_agent.pipeline import ingest_statement
from household_agent.synthetic import ACCOUNTS, ALEX, DEMO_AS_OF, JORDAN, RULES, demo_statements
from household_agent.tools import TOOL_NAMES, ToolSession

TAX = "San Mateo County Tax Collector"
TAX_QUERY = {"account_ids": ["acct_rental_checking"], "start": "2025-10-01", "end": "2026-10-31", "merchant": TAX}


def _result(call_result) -> dict:
    return json.loads(call_result.content[0].text)


def _session(store, categorizer, actor=ALEX, purpose="historical_comparison"):
    return ToolSession(store, categorizer, actor, purpose, correlation_id="mcp_t", clock=lambda: DEMO_AS_OF)


def test_server_lists_exactly_the_tools_without_identity_arguments(seeded_store, categorizer):
    async def go():
        async with create_connected_server_and_client_session(build_server(_session(seeded_store, categorizer))) as c:
            return (await c.list_tools()).tools

    tools = asyncio.run(go())
    assert sorted(t.name for t in tools) == sorted(TOOL_NAMES)
    for t in tools:
        assert not {"actor", "actor_id", "member_id", "purpose", "household_id"} & set(t.inputSchema.get("properties", {}))


def test_query_and_calculate_over_mcp(seeded_store, categorizer):
    async def go():
        async with create_connected_server_and_client_session(build_server(_session(seeded_store, categorizer))) as c:
            rows = _result(await c.call_tool("query_ledger", TAX_QUERY))
            ids = [t["transaction_id"] for t in rows["transactions"]]
            return _result(await c.call_tool("run_calculation", {"fn": "variance_between_transactions",
                                                                 "transaction_ids": ids}))

    out = asyncio.run(go())
    assert out["ok"] and out["output"]["dollar_variance_minor"] == 45000 and out["output"]["pct_variance"] == "16.36"


def test_denial_comes_back_as_data_over_mcp(seeded_store, categorizer):
    async def go():
        session = _session(seeded_store, categorizer, JORDAN, "budgeting")
        async with create_connected_server_and_client_session(build_server(session)) as c:
            return _result(await c.call_tool("query_ledger", {"account_ids": ["acct_alex_checking"],
                                                              "start": "2026-10-01", "end": "2026-10-31"}))

    assert asyncio.run(go())["error"]["type"] == "authorization_denied"


def test_langchain_adapter_tools(seeded_store, categorizer):
    async def go():
        async with create_connected_server_and_client_session(build_server(_session(seeded_store, categorizer))) as c:
            tools = {t.name: t for t in await load_mcp_tools(c)}
            raw = await tools["query_ledger"].ainvoke(TAX_QUERY)
            return tools, raw

    tools, raw = asyncio.run(go())
    assert set(tools) == set(TOOL_NAMES)
    [block] = raw  # adapter returns LangChain content blocks
    assert json.loads(block["text"])["transactions"][1]["amount_minor"] == -320000


@pytest.mark.parametrize("env", [{}, {"HH_LEDGER_DB": "x.db", "HH_ACTOR": "alex"},
                                 {"HH_LEDGER_DB": "x.db", "HH_ACTOR": "mallory", "HH_PURPOSE": "budgeting"}])
def test_server_refuses_to_start_without_a_known_identity(env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        session_from_env(env)


def test_stdio_subprocess_end_to_end(tmp_path):
    db = tmp_path / "ledger.db"
    conn = sqlite3.connect(db)
    store = LedgerStore(conn, Authorizer(ACCOUNTS), AuditLog(conn))
    ingest_statement(store, Categorizer(RULES), ALEX, demo_statements()["2026_10_rental_checking.csv"].encode(),
                     "r.csv", purpose="statement_ingestion", correlation_id="seed", at=DEMO_AS_OF)
    conn.close()

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "household_agent.mcp_server"],
        env={**os.environ, "HH_LEDGER_DB": str(db), "HH_ACTOR": "jordan", "HH_PURPOSE": "budgeting"},
    )

    async def go():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as c:
            await c.initialize()
            accounts = _result(await c.call_tool("list_accounts", {}))
            denied = _result(await c.call_tool("query_ledger", {"account_ids": ["acct_alex_checking"],
                                                                "start": "2026-10-01", "end": "2026-10-31"}))
            joint = _result(await c.call_tool("query_ledger", {"account_ids": ["acct_rental_checking"],
                                                               "start": "2026-10-01", "end": "2026-10-31"}))
            return accounts, denied, joint

    accounts, denied, joint = asyncio.run(asyncio.wait_for(go(), timeout=60))
    assert [a["account_id"] for a in accounts["accounts"]] == ["acct_jordan_card", "acct_rental_checking"]
    assert denied["error"]["type"] == "authorization_denied"
    assert len(joint["transactions"]) == 4

    events = AuditLog(sqlite3.connect(db)).events()
    assert any(e["actor_id"] == "jordan" and e["decision"] == "deny" for e in events)
