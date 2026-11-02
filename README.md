# Household Wealth Agent

This is a local-first, multi-agent assistant for household finances. It reads bank statements and reconciles them against their own totals. It answers questions such as *"why did our property tax go up?"* with deterministic, auditable calculations, and every number cites its source line. Each household member's private data stays private unless that member gives consent.

The design is the single source of truth:
[`agentic_ai_course_design.md`](../../agentic_ai_course_design.md) (on the Desktop). The code-level view is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Status

| Phase | Scope (design §12) | State |
|---|---|---|
| 1 | Steps 1–5 below | **Done**: 59 tests passing |
| 2 | LangGraph supervisor + Ledger Analyst + Policy & Compliance agents, ChromaDB retrieval, local MCP server, Ollama models | Not started |
| 3 | Evaluation harness (≥200 scenarios), briefing generation, hardening | Not started |

What's in Phase 1:

1. **Finance library** (`household_finance`). Money is stored as integer cents and floats are rejected. It covers cash flow, category totals, variance, recurring-charge drift, net worth and amortization (`numpy-financial`). The library is tested with `pytest` and `hypothesis` property tests.
2. **Ingestion**. A strict CSV statement parser that keeps line-level provenance. It feeds rule-based merchant categorization.
3. **Reconciliation**. Each statement is checked for balance continuity, period bounds, matching totals and duplicates. It then gets one of three states, `reconciled` / `qualified` / `unreconciled`, plus a weighted `document_confidence` score.
4. **Ledger, privacy and audit**. The SQLite ledger is append-only. Corrections are added as new versions instead of overwriting old rows. Authorization is deny-by-default and requires per-owner consent; joint accounts need consent from every owner. The audit log is hash-chained and tamper-evident.
5. **Calculation runner**. Only allowlisted calculations can run. Unreconciled data can never be used. Provisional data is used only when the caller explicitly opts in. Every run records input and output hashes so it can be reproduced.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                    # create .venv and install deps
uv run pytest                              # run all tests
uv run python -m household_agent.demo      # end-to-end demo on synthetic data
uv run python scripts/generate_synthetic_data.py   # (re)write data/synthetic/*.csv
```

The demo walks through five steps:

1. **Ingests five synthetic statements.** Three reconcile cleanly. Jordan's card statement is `qualified` because it contains two identical Safeway charges that need a human to confirm. The September file is `unreconciled` because its balances don't add up.
2. **Compares the October property tax year over year.** It went from $2,750.00 to $3,200.00, a rise of **$450.00 (16.36%)**. The result comes with a calculation `run_id` and an input hash.
3. **Blocks a cross-member read.** Jordan tries to read Alex's private checking account and is refused for lack of consent.
4. **Excludes the corrupt statement.** No rows from the corrupt September statement reach calculations.
5. **Verifies the audit chain.** Every allow or deny decision is in it.

## Project layout

```
src/household_finance/     Pure, deterministic finance library (no I/O)
  money.py                 to_minor / format_minor / percent — integer cents
  calcs.py                 cash_flow, category_totals, variance, recurring_drift,
                           net_worth_snapshot, amortization_schedule
src/household_agent/       Agent foundation (agents arrive in Phase 2)
  models.py                Account, Transaction, enums (roles, actions, status)
  authz.py                 Deny-by-default authorizer + consent grants
  audit.py                 Hash-chained, append-only audit log
  ingest/csv_statement.py  Statement parser (hh_statement_csv_v1)
  categorize.py            Rule-based merchant normalization
  reconcile.py             Reconciliation checks + document confidence
  ledger.py                Append-only SQLite ledger (LedgerStore)
  calc_runner.py           Allowlisted calculation runner with provenance
  pipeline.py              parse → categorize → reconcile → persist
  synthetic.py             Demo household, accounts, rules, statements
  demo.py                  `python -m household_agent.demo`
scripts/                   generate_synthetic_data.py
data/synthetic/            Generated sample statements (synthetic only)
tests/                     pytest + hypothesis suites
```

## Statement format (`hh_statement_csv_v1`)

```
# format: hh_statement_csv_v1
# household_id: hh_demo
# account_id: acct_rental_checking
# currency: USD
# period_start: 2026-10-01
# period_end: 2026-10-31
# opening_balance: 17240.00
# closing_balance: 16571.60
# total_credits: 3100.00
# total_debits: 3768.40
date,description,amount
2026-10-01,RENT DEPOSIT UNIT B,3100.00
2026-10-05,MAPLE ST HOA DUES,-450.00
2026-10-09,CITY WATER UTILITY,-118.40
2026-10-18,SAN MATEO TAX COLLECTOR - SECURED PROPERTY TAX - UNIT B,-3200.00
```

Every header field is required. Amounts are signed decimal strings with at most 2 decimal places. A broken row is kept as an error, which marks the whole statement unreconciled; it is never silently dropped.

## Known gaps (Phase 1)

These parts of the design are not implemented yet:

- **Encryption at rest and per-household keys.** The SQLite database is plaintext. Use synthetic data only.
- **Statement formats.** There is no PDF/OCR ingestion; only the CSV format above is supported.
- **Reviewer override.** A reviewer can't yet mark a `qualified` statement as confirmed.
- **Agents and orchestration.** The LangGraph agents, checkpointer/interrupts, ChromaDB retrieval and the MCP server are planned for Phase 2.
- **Consent management.** Consent grants live in memory and are not persisted.

## Data policy

Use **synthetic data only** until encryption at rest lands. Never commit real statements; `*.db` and `*.sqlite` are git-ignored.
