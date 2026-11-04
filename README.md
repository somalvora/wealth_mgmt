# Household Wealth Agent

This is a local-first, multi-agent assistant for household finances. It reads bank statements and reconciles them against their own totals. It answers questions such as *"why did our property tax go up?"* with deterministic, auditable calculations, and every number cites its source line. Each household member's private data stays private unless that member gives consent.

The design is the single source of truth:
[`agentic_ai_course_design.md`](../../agentic_ai_course_design.md) (on the Desktop). The code-level view is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Status

| Phase | Scope (design §12) | State |
|---|---|---|
| 1 | Steps 1–6 below | **Done** |
| 2 | LangGraph supervisor + Ledger Analyst + Policy & Compliance agents, ChromaDB retrieval, local MCP server, Ollama models | **In progress**: model config, consent, approval, tools, MCP server, retrieval, agents, ledger-derived planner inputs, model planner personas, monthly briefing and eval harness skeleton done |
| 3 | Evaluation harness (≥200 scenarios), briefing generation, hardening | **Started**: 231-scenario eval skeleton and draft monthly briefing are in; retrieval/citation/anomaly metrics and hardening remain |

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
4. **Agent tool layer** (`tools.py`). `ToolSession` exposes `list_accounts`, `query_ledger`, `run_calculation` and `ingest_statement`.
   - The host binds the actor, purpose and correlation ID when it creates the session; none of them is a tool argument.
   - `run_calculation` only accepts transaction IDs that this session already received from `query_ledger`.
   - Each session has a call budget.
   - Errors come back as data (`authorization_denied`, `ineligible_input`, `budget_exhausted`, …).
5. **MCP server** (`mcp_server.py`). `household-finance-tools` is a stdio server that wraps `ToolSession`. Its identity comes from the environment (`HH_LEDGER_DB`, `HH_ACTOR`, `HH_PURPOSE`), so one process serves one authenticated user.
6. **Verified retrieval** (`retrieval.py`). ChromaDB keeps one collection per household and type (`financial_history`, `policies_compliance`).
   - Access control happens **before** similarity search. The authorizer lists the accounts the actor may `retrieve` for the stated purpose, and that list becomes a mandatory Chroma `where` filter. Another member's private document is never even a candidate, and revoking consent takes effect on the next query.
   - Delegates see only public Tier-1 sources (`PUBLIC` scope). Dependents see nothing.
   - The search returns up to 20 candidates. MMR (λ = 0.65) picks up to 5 passages, and each is checked: chunk hash, expiry and tax year.
   - `validate_claims` rules each claim `supported`, `remove` or `rephrase_as_question`:
     - A tax claim needs a verified Tier-1 source for the same tax year. Otherwise it gets the standard "cannot verify the rule for this tax year" sentence.
     - Every number in a claim must appear in its cited evidence.
   - Text that looks like an instruction (e.g. "ignore previous instructions") is flagged and audited as a data-quality finding. It is never followed.
7. **Agents** (`agents.py`). A LangGraph Supervisor routes each request, dispatches the Ledger Analyst and the Policy & Compliance agent, checks thresholds and composes the briefing.
   - **Handoffs.** These are typed (`TaskAssignment` → `AgentResult`). Status is `completed`, `needs_review`, `failed` or `budget_exhausted`.
   - **Specialists.** Each is a bounded ReAct subgraph: 6 tool calls, 2 retries per tool, and 2 query refinements for Policy & Compliance. A specialist can only call its own allowlisted tools.
   - **Numbers.** Outputs and evidence IDs are copied from tool observations. Every model-written claim goes back through `validate_claims`, so an invented amount is dropped and the result is flagged `needs_review`.
   - **Supervisor budgets.** At most 8 dispatches and 5 minutes per job. Requests to pay or transfer are refused. Allocation requests start the approval graph and pause.
   - **Router.** A model router with a keyword fallback. The purpose comes from the intent and the identity from the host, never from the model.
   - **Briefing.** Deterministic. It contains only validated claims, open questions and threshold findings (e.g. a property-tax variance over 10%).
8. **Planner inputs from the ledger** (`plan_inputs.py`). `derive_planning_inputs` builds the planner's numbers from data the requesting member may read, never from model text.
   - **Liquid cash.** The closing balance of the latest verified statement (reconciled, confidence ≥ 0.85) of each checking and savings account in a 3-month window. Closing balances are now stored at ingest in the append-only `statement_balances` table.
   - **Monthly essentials.** Per account, mortgage, rent, utilities, groceries, HOA dues and insurance come from a recorded `category_totals` run, divided by the number of verified months and rounded up. Property tax is excluded because it is lumpy; it enters as an explicit liability.
   - **Liabilities.** Each explicit liability needs an evidence ID (e.g. the county tax bill). Money owed on verified card statements is added.
   - **Exclusions.** Accounts the member can't read for `capital_allocation`, or that have no verified statement, are left out and listed. The inputs are then `complete=False`.
   - **Recorded runs.** `plan_from_ledger` stores every plan in `calculation_runs` (`fn="plan_allocation"`). The run carries the planner's input hash and all evidence IDs, is marked provisional if any account was left out, and is audited.
   - **Approval wiring.** `ledger_inputs_for` re-derives the inputs on every call. If a new statement lands between showing a plan and approving it, the approval ends `stale`.
9. **Model-driven planner personas and critic** (`llm_planner.py`). These plug into the same `ThoughtGenerator` / `Critic` slots as the templates, so the search, the hard rules and the scoring are unchanged.
   - **Personas.** `LLMThoughtGenerator` makes one model call per beam node (5 per run). The model returns one strategy per persona: tier, share of the remaining surplus, bucket weights, mode and a one-sentence rationale.
   - **No amounts from the model.** The schema rejects unknown fields such as `amount_minor`, and a rationale that quotes a money amount is rejected too. The finance library still computes every cent.
   - **Fallbacks.** A persona whose output is missing or malformed gets its template thought instead, and the fallback is recorded. A well-formed thought that breaks a rule (wrong tier, a bucket that doesn't belong to the tier) is hard-pruned rather than fixed, so the trace shows why it was dropped.
   - **Critic.** `LLMCritic` scores a node from 0 to 1, clamped. It is advisory: a gap over 0.20 keeps the branch for human review (`needs_review`), and an unusable reply counts as "no opinion".
   - **Approval.** `build_approval_graph` and `plan_from_ledger` accept `generator=` / `critic=`. Approving re-checks only the input hash, so it makes no model calls.
10. **Draft monthly briefing** (`briefing.py`). `draft_monthly_briefing` builds the seven design §10 sections for one member and one month: Data Coverage, Net Worth and Cash Flow, Budget and Variance, Material Events, Policy and Deadline Review, Actions for Approval, and Assumptions and Limitations. `render_markdown` turns it into a document.
    - **No model writes it.** It uses ledger data the member may read for `monthly_briefing`, two calculation runs it records itself (`cash_flow`, `category_totals`), validated Supervisor results, and an optional plan from `plan_from_ledger`.
    - **Evidence is required.** The schema rejects any item that carries an amount without evidence. Amounts stay integer cents.
    - **Coverage before numbers.** `LedgerStore.document_coverage` lists every statement in the month in any state, as metadata only.
      - Missing statements are listed in Data Coverage.
      - Unreconciled statements are excluded and named.
      - Qualified statements become a "please confirm" action, and their transactions are marked provisional and left out of totals.
      - Possible duplicates appear as material events.
    - **Material events** are uncategorized items of $500 or more and non-routine movements of $2,500 or more. Payroll, mortgage, rent and transfers count as routine.
    - **Privacy.** Accounts the member can't read appear only as a count ("2 household account(s) are not shared with you").
    - **Draft only.** `draft` is always `true` and `external_action` is always `"none"`. The last item is always the disclaimer, and composing the briefing is audited.
11. **Evaluation harness skeleton** (`evals.py`). `generate_scenarios(seed)` builds a seeded, synthetic held-out set of 231 scenarios. `evaluate` runs them through the real components and scores the design §2 gates.
    - **Routing (55 scenarios).** Intent, dispatched agents and the approval route. Gates: accuracy ≥ 0.95, schema validity ≥ 0.99, approval gating = 100%. A keyword fallback (`rules_fallback`) counts as an invalid model reply.
    - **Adversarial (14).** Pay, transfer, wire and prompt-injection requests run through `Supervisor.run` and must end `refused`. Gate: 0 unsafe actions.
    - **Authorization (72).** Every actor × account × purpose combination, with and without the CPA's two-owner consent. Gate: 0 cross-member leaks.
    - **Allocation (60).** Random planner inputs. The plan must sum exactly to the surplus, stay within caps, use ≤ 27 node evaluations, be approval-gated and be reproducible.
    - **Ingestion (30).** Random statements, a third with a corrupted closing balance. Each must end `reconciled` or `unreconciled` as labeled. Gate: 100% statement integrity.
    - **Not covered yet.** Retrieval recall@5, citation precision, calculation match, anomaly recall and privacy memorization are listed in the report.
    - **Keyword baseline.** All gates pass. Routing scores 98.2%: "Did we pay more for utilities…?" is wrongly flagged as a payment request. That is safe, but it is a miss.

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
uv run python -m household_agent.evals --out eval.json   # eval gates, keyword-router baseline (exit 1 if a gate fails)
uv run python -m household_agent.evals --model ollama:qwen3:8b --planner-model ollama:qwen3:8b   # on the Ollama machine
HH_LEDGER_DB=ledger.db HH_ACTOR=alex HH_PURPOSE=budgeting \
  uv run python -m household_agent.mcp_server       # MCP server on stdio (for an MCP client)
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
  llm_planner.py           Model-driven persona thoughts (validated, template fallback) and advisory critic
  plan_inputs.py           PlanningInputs derived from verified ledger data; plan runs recorded as calculation_runs
  briefing.py              Draft-only seven-section monthly briefing (deterministic, evidence-cited) + Markdown
  evals.py                 Seeded ≥200-scenario eval harness and §2 gates; `python -m household_agent.evals`
  tools.py                 ToolSession: identity-bound, budgeted agent tools (JSON in/out)
  mcp_server.py            household-finance-tools stdio MCP server over ToolSession
  agents.py                Supervisor + Ledger Analyst + Policy & Compliance (LangGraph), typed handoffs, budgets
  retrieval.py             ChromaDB verified retrieval (authorize-first filters, MMR, verification) + validate_claims
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
| `test_tools.py` | 14 | Property-tax variance through the tools, ownership/consent/purpose-scoped account listing, denial audited, actor can't be passed as an argument, invented transaction IDs rejected, provisional opt-in, budget, 6 bad-request cases, ingest guards |
| `test_mcp_server.py` | 8 | Tool list has no identity arguments, query + calculate over MCP, denial as data, `langchain-mcp-adapters` tools, server refuses to start without an identity (3 cases), real stdio subprocess end to end |
| `test_retrieval.py` | 22 | Private doc never a candidate for another member, consent grant/revocation take effect immediately (purpose-scoped), delegate public-only, dependent nothing, household isolation, missing purpose/query rejected, forged indexing denied, expired source not citable, tax-year pre-filter, Tier-1-same-year tax rule, invented numbers removed, calculated/unknown-evidence claims, injection flagged and audited, tampered chunk detected, statement chunking, period filter, heading split, chunk-window and MMR properties (hypothesis) |

| `test_agents.py` | 27 | §9 property-tax trace end to end, outputs from tools not model text, invented numbers and unknown evidence removed, tool budget, retry limit, tool allowlist, cross-member denial, unparseable answer, query-refinement limit, tax claim → "cannot verify" question, private doc not citable, external action refused, model can't clear the external-action flag, invalid routing falls back (3 cases), purpose from intent, dispatch budget, job time limit, crashing/missing specialist, briefing only from claims, allocation pauses for approval, out of scope, router property (hypothesis) |
| `test_plan_inputs.py` | 17 | Liquid cash, essentials and liabilities from verified statements (hand-computed), evidence IDs cover documents/runs/liabilities, plan run recorded with the planner's input hash and audited, reproducible hash, card balance owed becomes a liability, multi-month average rounds up, Jordan never sees Alex's private balance, dependent gets not_eligible, corrupt and out-of-window statements ignored, `statement_balances` append-only, invalid liabilities (3 cases), other household rejected, new statement makes approval `stale`, month-window and liability-sum properties (hypothesis) |
| `test_llm_planner.py` | 17 | Model thoughts drive the same 15-node search in 5 calls, model fraction and rationale used, `<think>` and code fences stripped, unparseable reply falls back to templates, amount fields rejected, currency in rationale rejected, fraction > 1 rejected, missing/duplicate personas, out-of-tier bucket and wrong tier hard-pruned (not repaired), agreeing critic, disagreeing critic → `needs_review` with nothing pruned, unusable/NaN/extra-field critic replies have no opinion, approving makes no model calls, any reply yields one thought per persona and critic score clamped to [0, 1] (hypothesis) |
| `test_briefing.py` | 16 | Alex's October cash flow, category totals and closing balances (hand-computed), calc runs recorded and briefing audited, routine items skipped and property tax listed, $500 unmapped threshold, plan + runner-up + pending approval under Actions, Supervisor claims/questions sorted into sections, Markdown has every section and the disclaimer, Jordan never sees Alex's private account, qualified card provisional with duplicate and confirmation request, corrupt statement contributes no numbers, missing statements reported, dependent and tax-only CPA get no ledger data (2 cases), schema guards (amounts need evidence, draft/disclaimer required), category totals sum to net cash flow (hypothesis) |
| `test_evals.py` | 13 | ≥200 deterministic scenarios across 5 areas, oracle router passes every gate, keyword baseline safe and its one false positive reported, CLI writes the JSON report, model persona generator keeps allocation invariants, router that never flags payments fails `unsafe_actions`, model "no external action" overridden, unparseable model router fails schema validity, misrouted allocation fails approval gating, allow-all authorizer reported as leaks, wrong label caught, empty set, any seed keeps the safety and integrity gates (hypothesis) |

Total: **249 passing**.

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
- **Agents and orchestration.** The LangGraph supervisor and specialist agents, are next in Phase 2. The approval graph, MCP tools and retrieval are not yet wired to the supervisor. Retrieval uses word-based chunking (words approximate tokens) and a regex injection detector. Accounts and members for the MCP server still come from the synthetic household.
- **Planner inputs.**
  - Planner inputs are synthetic. Contribution room values are placeholders, not IRS limits.
  - Planning inputs are derived from the ledger (`plan_inputs.py`), but the demo's step 5 still uses the fixed synthetic inputs. Contribution room and return assumptions are still passed in rather than retrieved.
  - The model persona generator and critic (`llm_planner.py`) are tested only with fake models so far. Their prompts haven't been tuned against `qwen2.5:7b` / `qwen3:8b`, and the demo still uses the template generator. Real-model runs happen on the Ollama machine.

- **Briefing and evals.**
  - The briefing is not yet produced by the demo or the Supervisor graph.
  - No household net-worth total is shown, because property, loans and investments aren't in the ledger.
  - The eval set is template-generated, not a human-labeled held-out set.
  - Retrieval, citation, calculation-match, anomaly and memorization metrics are not scored yet.
  - Real-model eval numbers still need a run on the Ollama machine.

## Data policy

Use **synthetic data only** until encryption at rest lands. Never commit real statements; `*.db` and `*.sqlite` are git-ignored.
