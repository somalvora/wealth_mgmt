# Architecture

This file describes the architecture as it exists in code. It tracks
[`agentic_ai_course_design.md`](../../agentic_ai_course_design.md), and the design doc wins whenever the two disagree. Update this file whenever the code changes shape.

## 1. Big picture

```
                 ┌──────────────────────────── Phase 2 (planned) ───────────────────────────┐
  user ──► Supervisor agent ──task_assignment──► Ledger Analyst ──► tools ─┐                 │
                 │            ◄──agent_result───                            │                 │
                 └──────────► Policy & Compliance ──► retrieve_verified ───┤ (ChromaDB)      │
                 └──────────────────────────────────────────────────────────┼─────────────────┘
                                                                            ▼
 ┌──────────────────────────── Phase 1 (implemented) ─────────────────────────────────────┐
 │  pipeline.ingest_statement                                                             │
 │    parse_statement_csv ─► Categorizer ─► reconcile ─► LedgerStore.ingest_document      │
 │                                                          │  (Authorizer + AuditLog)    │
 │  LedgerStore.query_transactions ─► CalculationRunner.run ─► household_finance.calcs    │
 │                                          │                                              │
 │                                          └─► calculation_runs + audit_events (SQLite)   │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

Phase 1 builds the **deterministic core** that the agents will call as tools. An LLM never does arithmetic, never decides access, and never writes to the ledger directly. Agents can only choose which tool to call and how to explain the result (design §1.1, §3.2).

## 2. Module map

| Module | Responsibility | Design § | Future agent tool |
|---|---|---|---|
| `household_finance.money` | Parse and format money as integer minor units; `Decimal` percentages with ROUND_HALF_UP | 3.4 | — |
| `household_finance.calcs` | Pure finance functions; no I/O, no floats | 3.4 | `run_calculation` |
| `household_agent.models` | Pydantic `Account` and `Transaction`; enums | 6 | — |
| `household_agent.authz` | Deny-by-default `Authorizer`, `ConsentGrant` | 7 | (enforced inside every tool) |
| `household_agent.audit` | Hash-chained `AuditLog` | 7 | — |
| `household_agent.ingest` | `parse_statement_csv` with line provenance | 3.1, 6 | `parse_statement` |
| `household_agent.categorize` | Rule-based merchant normalization | 6 | (part of ingest) |
| `household_agent.reconcile` | Checks, status, `document_confidence` | 3.3 | `reconcile_statement` |
| `household_agent.ledger` | Append-only `LedgerStore` | 6 | `query_ledger` |
| `household_agent.calc_runner` | Allowlist, eligibility, provenance | 3.2, 3.4 | `run_calculation` |
| `household_agent.pipeline` | Orchestrates ingest end to end | 12 | — |

## 3. Data flow: ingesting a statement

1. **Parse.** The parser validates header fields, column names and every row. Row problems become `RowError`s and are kept rather than dropped. The parser computes `source_hash = sha256(file bytes)`.
2. **Categorize.** Each description goes through a case-insensitive rule match. A match gives `merchant_normalized`, a category and confidence 0.98. No match gives `uncategorized` with confidence 0.0, which lowers `mapped_merchant_rate`.
3. **Reconcile.** These checks run (tolerance is 0 cents):

   | Check | Severity |
   |---|---|
   | row parse errors | material |
   | unsupported currency | material |
   | invalid period | material |
   | rows outside the period | material |
   | opening + Σrows ≠ closing | material |
   | credit/debit totals ≠ header | material |
   | row hash already ingested | material |
   | identical rows in the same file | bounded |
   | reversal pairs within 3 days | info |

   - **Status.** Any material issue → `unreconciled`. Otherwise any bounded issue → `qualified`. Otherwise → `reconciled`.
   - **Confidence.**
     `document_confidence = 0.35·row_parse_rate + 0.25·balance_continuity + 0.10·totals_match + 0.10·duplicate_free + 0.20·mapped_merchant_rate`.
     If it is below 0.85, the statement gets the `unverified` flag.
4. **Persist.** `LedgerStore.ingest_document` authorizes `upload` against the account and rejects a file whose `source_hash` is already stored. It writes the `documents`, `raw_rows` and `transactions` rows, then records an audit event.

## 4. Data flow: answering with a number

1. **Query.** `LedgerStore.query_transactions` authorizes `read` for **each** account and returns the latest version of each transaction.
   - By default it returns only `reconciled` rows with confidence ≥ 0.85.
   - `include_provisional=True` adds `qualified` and `unverified` rows.
   - `unreconciled` rows are never returned.
2. **Calculate.** `CalculationRunner.run(fn, txns, ...)` looks up `fn` in `REGISTRY`; an unknown name raises `KeyError`.
   - **Eligibility.** It rejects any `unreconciled` input. Provisional input is accepted only with `allow_provisional=True`, and the result is then flagged `provisional`.
   - **Recording.** It calls the pure function and stores a `calculation_runs` row with the canonical-JSON `input_hash` and `output_hash` and `household_finance.__version__`, then appends a `calculate` audit event.
3. **Explain** (Phase 2). The agent quotes `output` and cites `run_id` plus each input's `source_locator` (for example `line=15`).

## 5. Storage schema (SQLite)

All tables are **append-only**: `BEFORE UPDATE` and `BEFORE DELETE` triggers raise an error.

| Table | Key | Notes |
|---|---|---|
| `documents` | `document_id` | `source_hash` UNIQUE, status, confidence, issues JSON |
| `raw_rows` | (`document_id`, `line_number`) | Exact source text for provenance |
| `transactions` | (`transaction_id`, `version`) | Corrections insert `version + 1` with `correction_reason` |
| `calculation_runs` | `run_id` | fn, params, input IDs, hashes, library version, provisional |
| `audit_events` | `seq` | `prev_hash` → `hash` chain from a genesis of `"0"*64` |

`transaction_id = "txn_" + row_hash[:16]`. The row hash covers the account, date, amount, description and an **occurrence index**, so two genuinely identical charges still get distinct IDs.

## 6. Authorization rules (`authz_1.0`)

Rules are evaluated in order, and the first match wins:

1. Unknown account → **deny**.
2. Actor from another household → **deny**.
3. `dependent` role → **deny**.
4. Actor owns the account (and is not a delegate) → **allow**. Owning one account does not give a member any access to someone else's private account, even if that member is the admin.
5. `delegate` asking for anything other than `read` / `report` / `report_aggregate` → **deny**.
6. Otherwise the actor needs an active `ConsentGrant` from **every** owner of the account, matching the action and purpose.
   - Delegate grants must have an `expires_at`.
   - Revoked or expired grants don't count.

Every decision, allow or deny, is written to the audit log along with `policy_version` and the `consent_version`s used.

## 7. Guardrails summary

| Risk | Guardrail | Where |
|---|---|---|
| Float rounding error | Floats rejected; integer cents; explicit rounding | `money.py` |
| LLM does arithmetic | Allowlisted `REGISTRY` only | `calc_runner.py` |
| Bad statement corrupts answers | `unreconciled` never queryable or calculable | `ledger.py`, `calc_runner.py` |
| Double import | `source_hash` UNIQUE + historical row hashes | `ledger.py`, `reconcile.py` |
| Silent edits | Append-only triggers, versioned corrections | `ledger.py` |
| Cross-member leak | Deny-by-default and all-owner consent | `authz.py` |
| Untraceable actions | Hash-chained audit log, `verify()` | `audit.py` |

## 8. Phase 2 plan (design §4, §12 steps 6–10)

- **Agents.** A LangGraph supervisor graph with two specialists.
  - Ledger Analyst: `parse_statement`, `reconcile_statement`, `query_ledger`, `run_calculation`, `assess_risk`.
  - Policy & Compliance: `retrieve_verified`, `validate_claims`.
  - The supervisor itself owns `draft_reminder` and `compose_briefing`.
- **Handoffs.** Typed `task_assignment` / `agent_result` handoffs.
- **Budgets.**
  - Supervisor: 8 dispatches per job and a 5-minute job limit.
  - Specialists: 6 tool calls each, with 2 retries per tool.
- **Durability.** A SQLite checkpointer, plus `interrupt()` for human approval.
- **Retrieval.** A ChromaDB index of verified documents, using 500/100 chunking and `nomic-embed-text`. Every retrieval is filtered by the authorizer.
- **MCP.** A `household-finance-tools` stdio MCP server exposes the Phase 1 functions, consumed through `langchain-mcp-adapters`.
- **Models.** Local Ollama models: `qwen2.5:7b` as the baseline, with `qwen3:8b` for comparison.
