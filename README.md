# Household Wealth Agent

This is a local-first, multi-agent assistant for household finances. It reads bank statements and reconciles them against their own totals. It answers questions such as *"why did our property tax go up?"* with deterministic, auditable calculations, and every number cites its source line. Each household member's private data stays private unless that member gives consent.

The design is the single source of truth:
[`agentic_ai_course_design.md`](../../agentic_ai_course_design.md) (on the Desktop). The code-level view is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Status

| Phase | Scope (design §12) | State |
|---|---|---|
| 1 | Steps 1–6 below | **Done** |
| 2 | LangGraph supervisor + Ledger Analyst + Policy & Compliance agents, ChromaDB retrieval, local MCP server, Ollama models | **In progress**: model config, persistent consent, approval workflow with checkpoints done |
| 3 | Evaluation harness (≥200 scenarios), briefing generation, hardening | Not started |

What's in Phase 1:

1. **Finance library** (`household_finance`). Money is stored as integer cents and floats are rejected. It covers cash flow, category totals, variance, recurring-charge drift, net worth and amortization (`numpy-financial`). The library is tested with `pytest` and `hypothesis` property tests.
2. **Ingestion**. A strict CSV statement parser that keeps line-level provenance. It feeds rule-based merchant categorization.
3. **Reconciliation**. Each statement is checked for balance continuity, period bounds, matching totals and duplicates. It then gets one of three states, `reconciled` / `qualified` / `unreconciled`, plus a weighted `document_confidence` score.
4. **Ledger, privacy and audit**. The SQLite ledger is append-only. Corrections are added as new versions instead of overwriting old rows. Authorization is deny-by-default and requires per-owner consent; joint accounts need consent from every owner. The audit log is hash-chained and tamper-evident.
5. **Calculation runner**. Only allowlisted calculations can run. Unreconciled data can never be used. Provisional data is used only when the caller explicitly opts in. Every run records input and output hashes so it can be reproduced.
6. **Tree-of-Thought capital-allocation planner** (design §5.8). It splits a surplus across three tiers: a cash buffer, then tax-advantaged accounts, then growth or debt prepayment.
   - The search is a beam search (b=3 personas, k=2, depth 3): 15 node evaluations per run.
   - Thoughts are *strategies*. The finance library turns them into cents, enforces caps and the reserve floor, and scores them (0.40 tax, 0.35 liquidity, 0.25 yield).
   - The result is an approval-gated scenario; nothing moves money.

What's in Phase 2 so far:

1. **Model configuration** (`llm.py`). `HH_CHAT_MODEL` (default `ollama:qwen2.5:7b`) and `HH_EMBED_MODEL` (default `ollama:nomic-embed-text`, or `hash` for offline) choose the models. No other module names a model, and **no test needs Ollama**: tests use LangChain fake chat models and the deterministic `HashEmbeddings`.
2. **Persistent consent** (`consent.py`). Grants and revocations are stored in append-only SQLite tables and reloaded on startup. Only the grantor can grant or revoke, and only for accounts they own. Every attempt is audited.
3. **Approval workflow** (`approval.py`). A LangGraph graph plans an allocation, then pauses with `interrupt()` until a household admin or adult member approves or rejects it. The paused request is checkpointed with `SqliteSaver`, so it survives a restart. If the inputs changed since the plan was shown, the approval ends `stale`.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Ollama is **not** needed to run the tests or the demo. It's only needed for real-model runs:

```bash
ollama pull qwen2.5:7b && ollama pull qwen3:8b && ollama pull nomic-embed-text
export HH_CHAT_MODEL=ollama:qwen3:8b       # optional; default is ollama:qwen2.5:7b
```

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
5. **Plans a capital allocation.** The household has a **$20,800 surplus** above a $45,200 reserve (six months of essentials plus the next property-tax installment). The planner evaluates 15 nodes and picks *balanced buffer → fill tax room → brokerage*. Within the 0.02 tie margin it prefers this plan over the slightly higher-scoring all-aggressive plan because it keeps more liquidity. It then stops at `awaiting_approval`.
6. **Verifies the audit chain.** Every allow or deny decision is in it.

## Project layout

```
src/household_finance/     Pure, deterministic finance library (no I/O)
  money.py                 to_minor / format_minor / percent — integer cents
  calcs.py                 cash_flow, category_totals, variance, recurring_drift,
                           net_worth_snapshot, amortization_schedule
  allocation.py            expand_step / validate_allocation / score_allocation / rollout
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
  planning.py              Tree-of-Thought beam-search planner (template generator)
  llm.py                   Model selection from env (HH_CHAT_MODEL / HH_EMBED_MODEL), HashEmbeddings
  consent.py               Persistent, append-only consent grants and revocations
  approval.py              LangGraph plan → interrupt() approval graph with checkpointing
  synthetic.py             Demo household, accounts, rules, statements
  demo.py                  `python -m household_agent.demo`
scripts/                   generate_synthetic_data.py
data/synthetic/            Generated sample statements (synthetic only)
tests/                     pytest + hypothesis suites
```

## Testing

Every feature ships with tests in the same change. A step isn't done until `uv run pytest` passes, and no test may require Ollama or the network. Each new module gets:

- **Happy-path tests** using hand-computed expected values.
- **Negative-path tests** covering denials, rejected input, blocked data and pruned branches. The design's guardrails are only real if a test shows them refusing.
- **Property tests** (`hypothesis`) for any money math, e.g. totals sum exactly and scores stay in [0, 1].
- **An end-to-end check** through the demo flow when the feature is visible to users.

| Test file | Tests | Covers |
|---|---|---|
| `test_finance.py` | 15 (+ parametrized cases) | Money parsing and formatting, cash flow, category totals, variance, drift, amortization; hypothesis properties |
| `test_authz.py` | 11 | Owner access, cross-member denial, joint all-owner consent, delegate expiry and revocation, dependents |
| `test_audit.py` | 4 | Chain verification, UPDATE/DELETE blocked, tamper detection |
| `test_ingest_reconcile.py` | 11 | Parser provenance and errors; reconciled, qualified and unreconciled outcomes; duplicates, reversals, confidence |
| `test_ledger_and_calcs.py` | 10 | Upload denial audited, re-import blocked, provisional and unreconciled filtering, versioned corrections, reproducible calculations |
| `test_planning.py` | 17 | Allocation expansion and constraints, 15-node beam search, rollout scoring, liquidity tie-break, hard pruning, advisory critic |
| `test_llm.py` | 9 | Model spec parsing, env override, unsupported specs rejected, HashEmbeddings similarity, determinism (hypothesis) |
| `test_consent.py` | 15 | Joint all-owner grants, grants and revocations survive a restart, 9 invalid-grant denials audited, grantor-only revocation, append-only tables |
| `test_approval.py` | 15 | Interrupt payload, approve/reject, non-approvers denied while the graph keeps waiting, other household denied, stale inputs, not-eligible ends early, restart resume (repeated 5×) |

Total: **115 passing**.

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
- **Agents and orchestration.** The LangGraph supervisor and specialist agents, ChromaDB retrieval and the MCP server are next in Phase 2. The approval graph is not yet wired to the supervisor.
- **Planner inputs.**
  - Planner inputs are synthetic. Contribution room values are placeholders, not IRS limits.
  - Planning inputs are not yet derived from the ledger, and planner runs are not yet stored as `calculation_runs`.
  - The LLM persona generator and the critic arrive in Phase 2.

## Data policy

Use **synthetic data only** until encryption at rest lands. Never commit real statements; `*.db` and `*.sqlite` are git-ignored.
