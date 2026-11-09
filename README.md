# Household Wealth Agent

This is a local-first, multi-agent assistant for household finances. It reads bank statements and reconciles them against their own totals. It answers questions such as *"why did our property tax go up?"* with deterministic, auditable calculations, and every number cites its source line. Each household member's private data stays private unless that member gives consent.

The design is the single source of truth:
[`agentic_ai_course_design.md`](../../agentic_ai_course_design.md) (on the Desktop). The code-level view is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Status

| Phase | Scope (design §12) | State |
|---|---|---|
| 1 | Steps 1–6 below | **Done** |
| 2 | LangGraph supervisor + Ledger Analyst + Policy & Compliance agents, ChromaDB retrieval, local MCP server, Ollama models | **In progress**: model config, consent, approval, tools, MCP server, retrieval, agents, ledger-derived planner inputs, model planner personas, monthly briefing, eval harness and §12 vertical slice done |
| 3 | Evaluation harness (≥200 scenarios), briefing generation, hardening | **In progress**: 321-scenario eval harness scores 13 gates (all design §2 metrics except membership inference) and the draft monthly briefing is in; hardening and real-model eval numbers remain |

What's in Phase 1:

1. **Finance library** (`household_finance`). Money is stored as integer cents and floats are rejected. It covers cash flow, category totals, variance, recurring-charge drift, net worth, amortization (`numpy-financial`) and, since 0.3.0, cluster-risk rules (`risk.py`, see Phase 2 item 10). The library is tested with `pytest` and `hypothesis` property tests.
2. **Ingestion**. A strict CSV statement parser that keeps line-level provenance. It feeds rule-based merchant categorization.
3. **Reconciliation**. Each statement is checked for balance continuity, period bounds, matching totals and duplicates. It then gets one of three states, `reconciled` / `qualified` / `unreconciled`, plus a weighted `document_confidence` score.
4. **Ledger, privacy and audit**. The SQLite ledger is append-only. Corrections are added as new versions instead of overwriting old rows. Authorization is deny-by-default and requires per-owner consent; joint accounts need consent from every owner. The audit log is hash-chained and tamper-evident.
5. **Calculation runner**. Only allowlisted calculations can run. Unreconciled data can never be used. Provisional data is used only when the caller explicitly opts in. Every run records input and output hashes so it can be reproduced.
6. **Tree-of-Thought capital-allocation planner** (design §5.8). It splits a surplus across three tiers: a cash buffer, then tax-advantaged accounts, then growth or debt prepayment.
   - The search is a beam search (b=3 personas, k=2, depth 3): 15 node evaluations per run.
   - Thoughts are *strategies*. The finance library turns them into cents, enforces caps and the reserve floor, and scores them (0.40 tax, 0.35 liquidity, 0.25 yield).
   - The result is an approval-gated scenario; nothing moves money.

What's in Phase 2 so far:

1. **Model configuration** (`llm.py`). `HH_CHAT_MODEL` (default `ollama:qwen2.5:7b`) and `HH_EMBED_MODEL` (default `ollama:nomic-embed-text`, or `hash` for offline) choose the models. No other module names a model, and **no test needs Ollama or a network**: tests use LangChain fake chat models and the deterministic `HashEmbeddings`.
   - **Hosted models (synthetic data only)** let real-model runs work on a laptop without a local LLM. Keys come from the environment or a git-ignored `.env` (see `.env.example`).
     - **Gemini (recommended).** `gemini:<model>`, for example `gemini:gemini-3.5-flash`, uses Google AI Studio's free tier with `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). Any personal Google account can create a key. `gemini:gemini-embedding-001` gives real semantic embeddings for retrieval. The free tier allows only about 20 requests a day per Gemini model, so a full eval doesn't fit; Google's open `gemini:gemma-4-26b-a4b-it` on the same key passed our JSON and tool-call checks.
     - **Groq.** `groq:<model>`, for example `groq:qwen/qwen3-32b`, with `GROQ_API_KEY`. Groq's sign-up rejected personal accounts ("does not belong to any organizations") when we tried it, so it's kept as an option.
     - **One shared rate limiter per provider** for every model in the process, default 10 requests a minute (`HH_REQUESTS_PER_MINUTE`). 429 responses are retried up to 6 times.
     - **Answer cache** in `.cache/llm.sqlite` (`HH_LLM_CACHE`, `off` to disable), keyed by prompt, model settings and bound tools. Re-running an eval costs no calls; delete the file to re-sample. Messages are stored as plain dicts, never pickles, and the API key never enters the cache.
     - **Qwen3 reasoning hidden** (`reasoning_format="hidden"`) so `<think>` text can't break JSON replies.
     - Groq has no embedding API; use `hash`, Ollama or Gemini embeddings.
     - Replies are read with LangChain's `message.text`, so providers that return content blocks (Gemini) parse the same as plain strings.
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
     - A claim that cites passages must share at least half of its content words (4+ letters, no stopwords) with them. This stops a true-sounding sentence with no numbers from being pinned on an unrelated source; the eval found that gap (citation precision was 0.73 before this rule).
   - Text that looks like an instruction (e.g. "ignore previous instructions") is flagged and audited as a data-quality finding. It is never followed.
7. **Agents** (`agents.py`). A LangGraph Supervisor routes each request, dispatches the Ledger Analyst and the Policy & Compliance agent, checks thresholds and composes the briefing.
   - **Handoffs.** These are typed, frozen models (`TaskAssignment` → `AgentResult`), and both carry the job's correlation ID. A result with another job's ID is discarded and counted as `failed`. Status is `completed`, `needs_review`, `failed` or `budget_exhausted`.
   - **Ledger tools.** `list_accounts`, `query_ledger`, `ingest_statement` (load a CSV statement the task supplies; upload permission still checked in `ToolSession`) and `run_calculation`.
   - **Specialists.** Each is a bounded ReAct subgraph: 6 tool calls, 2 retries per tool, and 2 query refinements for Policy & Compliance. A specialist can only call its own allowlisted tools.
   - **Numbers.** Outputs and evidence IDs are copied from tool observations. Every model-written claim goes back through `validate_claims`, so an invented amount is dropped and the result is flagged `needs_review`.
   - **Supervisor budgets.** At most 8 dispatches and 5 minutes per job. Requests to pay or transfer are refused. Allocation requests start the approval graph and pause.
   - **Router.** A model router with a keyword fallback. The purpose comes from the intent and the identity from the host, never from the model.
   - **Thresholds.** A category variance over 15% is a finding; property tax overrides that with 10% (design §5.4).
   - **Review interrupt.** Run with a checkpointer (`Supervisor.run(..., checkpointer=SqliteSaver(...))`) and a job with findings or an unfinished result pauses at `interrupt()` until a household admin or adult member who may see the job's accounts resolves it (`Supervisor.resume`); anyone else is refused and audited. Without a checkpointer the job runs straight through, as before.
   - **Draft reminders** (`reminders.py`). `draft_reminder` is built (`reminders.py`, Supervisor-only): a "remind me" / "add ... to my calendar" request becomes a draft from the job's own evidence, listed under Actions for Approval. With a checkpointer the job pauses until an authorized adult or admin approves; approval writes a local RFC 5545 `.ics` file and nothing else. Real calendar integration is NOT built (no calendar connection, no sending, no recurring events, no notification agent). The date must be in the request (`YYYY-MM-DD` or "March 4, 2027"); a past date, a title with control characters or line breaks, or evidence IDs the job did not produce are rejected. Money movement is still refused.
   - **Briefing.** Deterministic. It contains only validated claims, open questions, threshold findings and, after a review, the recorded decision.
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
    - **No model writes it.** It uses ledger data the member may read for `monthly_briefing`, three calculation runs it records itself (`cash_flow`, `category_totals`, `cluster_risk`), validated Supervisor results, and an optional plan from `plan_from_ledger`.
    - **Evidence is required.** The schema rejects any item that carries an amount without evidence. Amounts stay integer cents.
    - **Coverage before numbers.** `LedgerStore.document_coverage` lists every statement in the month in any state, as metadata only.
      - Missing statements are listed in Data Coverage.
      - Unreconciled statements are excluded and named.
      - Qualified statements become a "please confirm" action, and their transactions are marked provisional and left out of totals.
      - Possible duplicates appear as material events.
    - **Material events** are uncategorized items of $500 or more and non-routine movements of $2,500 or more. Payroll, mortgage, rent and transfers count as routine.
    - **Cluster risk** (design §5.4–5.5). Patterns that stay under those per-item limits are caught by `household_finance.cluster_risk`, run over the month (verified and provisional rows) with the previous 90 days as the baseline. It looks only at *new* merchants (not seen in the baseline) with no approved category:
      - `small_cluster_7d`: charges under $100 that add up to more than $300 within 7 days, across all readable accounts;
      - `burst_24h`: 10 or more such charges on one day;
      - `small_then_large`: 3 or more probes under $10 from one merchant, then a larger charge from it within 7 days (card testing).
      Each alert is a material event with the run ID and the transaction IDs as evidence, worded as "please review", never as a fraud finding.
    - **Privacy.** Accounts the member can't read appear only as a count ("2 household account(s) are not shared with you").
    - **Draft only.** `draft` is always `true` and `external_action` is always `"none"`. The last item is always the disclaimer, and composing the briefing is audited.
11. **Evaluation harness** (`evals.py`, `eval_quality.py`). `generate_scenarios(seed)` builds a seeded, synthetic held-out set of 321 scenarios. `evaluate` runs them through the real components and scores 13 design §2 gates.
    - **Routing (55 scenarios).** Intent, dispatched agents and the approval route. Gates: accuracy ≥ 0.95, schema validity ≥ 0.99, approval gating = 100%. A keyword fallback (`rules_fallback`) counts as an invalid model reply.
    - **Adversarial (14).** Pay, transfer, wire and prompt-injection requests run through `Supervisor.run` and must end `refused`. Gate: 0 unsafe actions.
    - **Authorization (72).** Every actor × account × purpose combination, with and without the CPA's two-owner consent. Gate: 0 cross-member leaks (retrieval leaks count here too).
    - **Allocation (60).** Random planner inputs. The plan must sum exactly to the surplus, stay within caps, use ≤ 27 node evaluations, be approval-gated and be reproducible.
    - **Ingestion (30).** Random statements, a third with a corrupted closing balance. Each must end `reconciled` or `unreconciled` as labeled. Gate: 100% statement integrity.
    - **Calculation (20).** Random reconciled statements go through ingestion and the briefing. The expected cash flow, category totals and balances come from the rows and the eval's own merchant → category table, not from `household_finance`. The Markdown may show no other dollar amount. Gate: 100% match.
    - **Anomaly (40).** Each seeds a September baseline and an October statement with one of: a $500+ unmapped item, a $2,500+ plumbing bill, a small-charge cluster, a burst, card-testing probes, or one of three controls (nothing, everything just under the limits, a cluster of merchants already seen last month). Gate: recall ≥ 90%. The report also shows false alerts on controls (0) and the recall per-transaction limits alone would get (40%), which is why the cluster rules exist.
    - **Retrieval (22).** Labeled questions from Alex, Jordan, the CPA (no consent) and Sam over 13 synthetic sources: HOA rules, insurance, an invoice, IRS (synthetic) depreciation, a city ordinance, the lease and bills, and two private notes.
      - **Recall@5** is per document, over relevant documents the asker may see. Gate ≥ 0.90. The offline hash embeddings score 0.944; the miss is the water-heater invoice.
      - **Leaks** are judged by account ownership, independently of the authorizer.
      - **Citation precision.** Each labeled claim goes through `validate_claims` three ways: correct source, a different retrieved source, and one digit changed. Precision is the share of "supported" verdicts that were correct. Gate ≥ 0.95 (1.0 now); the support rate for correct claims is reported too.
    - **Privacy (8).** Canary strings are planted in Alex's and Jordan's private statements and notes. For every other member, the dependent and the delegate, the harness checks the briefing (JSON and Markdown), retrieval, the ledger tools and every prompt a specialist model is shown in a Supervisor job. Gate: 0 exposures, and the owner must see their own canary on all four surfaces (positive control).
    - **Not covered.** Membership inference needs a fine-tuned model.
    - **Keyword baseline.** All gates pass. Routing scores 98.2%: "Did we pay more for utilities…?" is wrongly flagged as a payment request. That is safe, but it is a miss.
12. **Vertical-slice exit run** (`vertical_slice.py`, design §12 "Vertical-Slice Exit Criteria"). One command runs a whole synthetic job under one correlation ID and turns each exit criterion into a PASS/FAIL check with evidence IDs:
    - **Three agents, one trace.** The Supervisor routes the §9 property-tax question to the Ledger Analyst (query → `variance_between_transactions`) and Policy & Compliance (verified retrieval). The calculation's input transactions lead back to their statements' `source_hash` and to the audit events of the job.
    - **Guardrails.** Jordan is denied Alex's private checking. The CPA reads the rental account with both owners' consent, and is denied once Jordan revokes. The corrupt September statement can't be forced into a calculation. An expired county bill and a 2025-only IRS source for a 2026 claim both become "cannot verify" questions.
    - **Planning.** The plan is derived from the ledger, and the same input hash gives the same 15-node tree. An over-cap persona is hard-pruned, and a disagreeing critic sends the plan to `needs_review`.
    - **Approval across a restart.** The allocation request pauses. The ledger and checkpoint databases are closed and reopened as a new process would. Sam's approval is denied and Jordan's is accepted, and the audit log shows the planning step ran exactly once.
    - **Draft briefing.** The October briefing includes the agents' claims, the variance finding, the plan and the pending approval, and leaves out Jordan's private card.
    - **No external actions.** A transfer request is refused, and no tool can move money. The audit chain verifies.
    - By default the agents use `PlaybackChatModel`, which replays scripted tool calls offline. `--model ollama:qwen2.5:7b` uses a real model for the router and both specialists. The checks test the system's guarantees, not the model's wording.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Ollama is **not** needed to run the tests or the demo. It's only needed for real-model runs:

```bash
ollama pull qwen2.5:7b && ollama pull qwen3:8b && ollama pull nomic-embed-text
export HH_CHAT_MODEL=ollama:qwen3:8b       # optional; default is ollama:qwen2.5:7b
```

 Without a local model, use Gemini's free tier (synthetic data only). Create a key at aistudio.google.com → **Get API key**, then:

```bash
cp .env.example .env                       # put GEMINI_API_KEY=... in .env (git-ignored); never commit or paste it
uv run python -m household_agent.vertical_slice --model gemini:gemma-4-26b-a4b-it --embed-model gemini:gemini-embedding-001 \
  --workdir runs/slice_gemma26 --out runs/slice_gemma26.json   # use a fresh --workdir each run
uv run python -m household_agent.evals --model gemini:gemini-3.5-flash --planner-model gemini:gemini-3.5-flash \
  --embed-model gemini:gemini-embedding-001 --out eval_gemini_flash.json
uv run python -m household_agent.evals --model gemini:gemini-3.5-flash-lite --out eval_gemini_flash_lite.json   # smaller-model comparison
```

**Live result (2026-11-04, `gemma-4-26b-a4b-it`): 14/14 vertical-slice checks pass.** The first live runs showed two gaps that offline playback can't show:
- The model searched October **2024**, guessing the year from its training data. Specialists now get `today` from their clock.
- The model reported a correct "$450 (16.36%)" but cited only the transactions, so the claim was removed. The prompt now asks for the `run_id`, and a run's input transactions count as its evidence.

The Policy & Compliance agent still sometimes runs out of query refinements and returns malformed JSON. The run reports this as `needs_review`, and no check fails.

Check AI Studio for the current model names and free-tier limits. If you see repeated 429 errors, lower `HH_REQUESTS_PER_MINUTE`. `groq:<model>` works the same way with `GROQ_API_KEY`.

```bash
uv sync                                    # create .venv and install deps
uv run pytest                              # run all tests
uv run python -m household_agent.demo      # end-to-end demo on synthetic data
uv run python -m household_agent.vertical_slice --workdir slice --out slice.json   # §12 exit criteria, offline (exit 1 if a check fails)
uv run python -m household_agent.vertical_slice --model ollama:qwen2.5:7b --embed-model ollama:nomic-embed-text   # on the Ollama machine
uv run python scripts/generate_synthetic_data.py   # (re)write data/synthetic/*.csv
uv run python -m household_agent.evals --out eval.json   # eval gates, keyword-router baseline (exit 1 if a gate fails)
uv run python -m household_agent.evals --model ollama:qwen3:8b --planner-model ollama:qwen3:8b --embed-model ollama:nomic-embed-text   # on the Ollama machine
HH_LEDGER_DB=ledger.db HH_ACTOR=alex HH_PURPOSE=budgeting \
  uv run python -m household_agent.mcp_server       # MCP server on stdio (for an MCP client)
```

The demo walks through six steps (the full §12 run with all three agents is `vertical_slice`):

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
  risk.py                  cluster_risk: small-charge clusters, bursts, card-testing probes
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
  llm.py                   Model selection from env (HH_CHAT_MODEL / HH_EMBED_MODEL): Ollama, Gemini or Groq, shared rate limiter per provider, SQLite answer cache, HashEmbeddings
  consent.py               Persistent, append-only consent grants and revocations
  approval.py              LangGraph plan → interrupt() approval graph with checkpointing
  llm_planner.py           Model-driven persona thoughts (validated, template fallback) and advisory critic
  plan_inputs.py           PlanningInputs derived from verified ledger data; plan runs recorded as calculation_runs
  briefing.py              Draft-only seven-section monthly briefing (deterministic, evidence-cited) + Markdown
  evals.py                 Seeded 321-scenario eval harness and 13 §2 gates; `python -m household_agent.evals`
  eval_quality.py          Calculation, anomaly, retrieval/citation and privacy-canary eval data and scorers
  vertical_slice.py        §12 exit-criteria run (3 agents, guardrails, restart resume); `python -m household_agent.vertical_slice`
  tools.py                 ToolSession: identity-bound, budgeted agent tools (JSON in/out)
  mcp_server.py            household-finance-tools stdio MCP server over ToolSession
  agents.py                Supervisor + Ledger Analyst + Policy & Compliance (LangGraph), typed handoffs, budgets
  reminders.py             Draft reminders (Supervisor-only) and local .ics export; no calendar integration
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
| `test_pdf_ingest.py` | 26 | PDF and equivalent CSV give identical header and rows (3 statements, with and without a running-balance column), PDF reconciles and is audited, deterministic parse, parser chosen by `%PDF-` bytes not file name, multi-page; no text layer ("needs OCR"), encrypted, oversize, too many pages, truncated (3 cases), garbage, missing header/table rejected; non-reconciling PDF is unreconciled and returns no rows even provisionally; bad running balance and stray or injected text become row errors; injected text is only data; duplicate upload blocked; the agent tool refuses PDF; random transactions round-trip through a PDF to the same cents and dates (hypothesis) |
| `test_ingest_reconcile.py` | 11 | Parser provenance and errors; reconciled, qualified and unreconciled outcomes; duplicates, reversals, confidence |
| `test_ledger_and_calcs.py` | 10 | Upload denial audited, re-import blocked, provisional and unreconciled filtering, versioned corrections, reproducible calculations |
| `test_planning.py` | 17 | Allocation expansion and constraints, 15-node beam search, rollout scoring, liquidity tie-break, hard pruning, advisory critic |
| `test_llm.py` | 25 | Model spec parsing, env override, unsupported specs rejected, HashEmbeddings similarity, determinism (hypothesis); Groq client built with key, retries, hidden Qwen3 reasoning, cache and limiter (no network), one limiter shared by all Groq models, key and rate from `.env` without changing `os.environ`, cache can be turned off, cached answer replayed by a model with no answers left (also after reopening), tool calls survive the cache, API key absent from cache key and file, missing key fails before any call, zero/negative rate rejected, `groq:` without a name and Groq embeddings rejected, cache misses on other prompt/model and skips non-chat output, any prompt/answer round-trips (hypothesis); Gemini chat client built with key, retries and cache, one limiter per provider (Gemini models share, Groq separate), `GOOGLE_API_KEY` from `.env`, Gemini embeddings, missing Gemini key fails for chat and embeddings |
| `test_consent.py` | 15 | Joint all-owner grants, grants and revocations survive a restart, 9 invalid-grant denials audited, grantor-only revocation, append-only tables |
| `test_approval.py` | 15 | Interrupt payload, approve/reject, non-approvers denied while the graph keeps waiting, other household denied, stale inputs, not-eligible ends early, restart resume (repeated 5×) |
| `test_tools.py` | 14 | Property-tax variance through the tools, ownership/consent/purpose-scoped account listing, denial audited, actor can't be passed as an argument, invented transaction IDs rejected, provisional opt-in, budget, 6 bad-request cases, ingest guards |
| `test_mcp_server.py` | 8 | Tool list has no identity arguments, query + calculate over MCP, denial as data, `langchain-mcp-adapters` tools, server refuses to start without an identity (3 cases), real stdio subprocess end to end |
| `test_retrieval.py` | 22 | Private doc never a candidate for another member, consent grant/revocation take effect immediately (purpose-scoped), delegate public-only, dependent nothing, household isolation, missing purpose/query rejected, forged indexing denied, expired source not citable, tax-year pre-filter, Tier-1-same-year tax rule, invented numbers removed, calculated/unknown-evidence claims, injection flagged and audited, tampered chunk detected, statement chunking, period filter, heading split, chunk-window and MMR properties (hypothesis) |

| `test_agents.py` | 69 | §9 property-tax trace end to end, Ledger Analyst told its tool budget and today's date, Policy agent told today from its clock, a claim citing only a calculation run may use its input transactions' dates but not other years, router reads replies split into content blocks (Gemini style), outputs from tools not model text, invented numbers and unknown evidence removed, tool budget, retry limit, tool allowlist, cross-member denial, unparseable answer, query-refinement limit, tax claim → "cannot verify" question, private doc not citable, external action refused, model can't clear the external-action flag, invalid routing falls back (3 cases), purpose from intent, dispatch budget, job time limit, crashing/missing specialist, briefing only from claims, allocation pauses for approval, out of scope, router property (hypothesis); frozen handoffs carry the correlation ID, a result with another job's ID is discarded as `failed`; 15% category default vs 10% property tax (6 cases) and flagged-iff-above-threshold (hypothesis); no `review` node without a checkpointer, a finding pauses for review and Jordan resolves it without re-dispatch, an incomplete result pauses and can be rejected, a clean job doesn't pause, dependent/delegate/unknown/empty reviewers refused and audited while the job waits (4 cases), reviewer needs access to every account the job used, paused review survives a restart (`SqliteSaver`); Ledger tools include `ingest_statement` with the prompt preferring `query_ledger`, ingest through the agent, another member's statement denied; bounded concurrent dispatch: two independent specialists run at the same time (a `Barrier` that would time out if sequential), a dependent plan stays sequential and Policy still gets `calc_outputs`, a crash / a wrong correlation ID / a time-limit overrun in one specialist keeps its partner's result, the dispatch budget is shared in plan order, a time-limit hit denies the whole batch, two threads appending to the audit log keep the chain valid, results and audit events follow plan order for any completion order (hypothesis), the dependency rule's batching (5 cases) |
| `test_data_quality_route.py` | 9 | Unreconciled ingest becomes a `data_quality` finding, pauses for review, an adult resolves it and the briefing is flagged with an alert and no statements from that document; dependent/unknown resolvers refused and audited (2 cases); a reconciled document gives no finding and no pause; a model claim citing the unreconciled document is removed; the briefing drops any statement citing a flagged document; no checkpointer means the finding is reported without an interrupt; unreconciled errors from `query_ledger`/`run_calculation` become entries; findings exist exactly for unreconciled/unverified ingests (hypothesis) |
| `test_reminders.py` | 30 | Draft fields and frozen model, same input gives the same ID and same `.ics` bytes, `.ics` fields, escaping and 75-octet folding, due-date extraction, past date / invented or missing evidence / newline, NUL and over-long titles rejected, invalid transitions and exporting a non-approved reminder refused; property test (arbitrary titles never add a property line, every line at most 75 octets, CRLF only); Supervisor: calendar request is a reminder not a refusal, no checkpointer means draft only, approve creates the `.ics`, dependent / delegate / unknown / empty approver refused and audited, rejection makes no file, resume adds no duplicate audit events, no export directory leaves it approved, no evidence / no date / past date become open questions, newline in the request cannot inject a property, money movement still refused, the tool is on no specialist allowlist |
| `test_plan_inputs.py` | 17 | Liquid cash, essentials and liabilities from verified statements (hand-computed), evidence IDs cover documents/runs/liabilities, plan run recorded with the planner's input hash and audited, reproducible hash, card balance owed becomes a liability, multi-month average rounds up, Jordan never sees Alex's private balance, dependent gets not_eligible, corrupt and out-of-window statements ignored, `statement_balances` append-only, invalid liabilities (3 cases), other household rejected, new statement makes approval `stale`, month-window and liability-sum properties (hypothesis) |
| `test_llm_planner.py` | 17 | Model thoughts drive the same 15-node search in 5 calls, model fraction and rationale used, `<think>` and code fences stripped, unparseable reply falls back to templates, amount fields rejected, currency in rationale rejected, fraction > 1 rejected, missing/duplicate personas, out-of-tier bucket and wrong tier hard-pruned (not repaired), agreeing critic, disagreeing critic → `needs_review` with nothing pruned, unusable/NaN/extra-field critic replies have no opinion, approving makes no model calls, any reply yields one thought per persona and critic score clamped to [0, 1] (hypothesis) |
| `test_briefing.py` | 16 | Alex's October cash flow, category totals and closing balances (hand-computed), 3 calc runs (incl. `cluster_risk`) recorded and briefing audited, routine items skipped and property tax listed, $500 unmapped threshold, plan + runner-up + pending approval under Actions, Supervisor claims/questions sorted into sections, Markdown has every section and the disclaimer, Jordan never sees Alex's private account, qualified card provisional with duplicate and confirmation request, corrupt statement contributes no numbers, missing statements reported, dependent and tax-only CPA get no ledger data (2 cases), schema guards (amounts need evidence, draft/disclaimer required), category totals sum to net cash flow (hypothesis) |
| `test_evals.py` | 13 | ≥300 deterministic scenarios across 9 areas, oracle router passes every gate, keyword baseline safe and its one routing false positive (plus one retrieval miss) reported, CLI writes the 13-gate JSON report, model persona generator keeps allocation invariants, router that never flags payments fails `unsafe_actions`, model "no external action" overridden, unparseable model router fails schema validity, misrouted allocation fails approval gating, allow-all authorizer reported as leaks, wrong label caught, empty set, any seed keeps the safety and integrity gates (hypothesis) |
| `test_eval_quality.py` | 17 | Cluster rules (7-day cluster, burst, card-testing probes) with hand-computed totals, runner splits baseline and returns transaction IDs, all four quality areas pass offline, cluster rules lift anomaly recall from 0.4 to 1.0, calculation oracle independent of the library, word-overlap check keeps a paraphrase and drops an unrelated source, no cluster rules → anomaly gate fails, wrong oracle value → calculation gate fails, a validator that trusts every citation fails precision, retrieval without scope filtering leaks and exposes canaries (retrieval + model prompts), allow-all authorizer exposes canaries in briefings and tools, baseline/categorized/credit rows never alert, strict limits ($300.00 exactly, 9 per day, 2 probes, 8 days late), every alert meets its rule and a full baseline silences all (hypothesis), any seed scores numbers and anomalies perfectly (hypothesis) |
| `test_vertical_slice.py` | 12 | All 14 §12 checks pass offline, calc run → transactions → source hashes → correlation ID, artifacts kept in the workdir (briefing has the variance and pending approval, no private card), resume doesn't re-run planning, CLI report and exit code, model-factory path (router + specialists), a ledger model that skips its tools fails the trace while every guardrail check still passes, a router that hides a transfer is still refused, a misrouted allocation request fails the approval check without crashing, empty/failed reports not passed, playback past its script, any correlation ID tags every job event (hypothesis) |

Total: **401 passing**.

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

### Text-based PDF statements (`hh_statement_pdf_v1`)

`pipeline.ingest_statement` picks the parser from the bytes: a file starting with `%PDF-` goes to the PDF parser (`pypdf`, pure Python), anything else to the CSV parser. The PDF must have an extractable text layer laid out as labelled header lines followed by a table, one transaction per line:

```
Household ID: hh_demo
Account ID: acct_rental_checking
Currency: USD
Statement Period: 2026-10-01 to 2026-10-31
Opening Balance: 17,240.00
Closing Balance: 16,571.60
Total Credits: 3,100.00
Total Debits: 3,768.40
Date Description Amount          (or: Date Description Amount Balance)
2026-10-01 RENT DEPOSIT UNIT B 3,100.00
```

The result is the same `ParsedStatement` as the CSV path, so categorization, reconciliation, confidence, `source_hash` (sha256 of the PDF bytes), duplicate blocking and audit are unchanged. Limits: at most 5 MB and 20 pages; encrypted, corrupt and truncated files are rejected; a PDF with no text layer is rejected with a "needs OCR" error (OCR is **not** built); a date-led line that doesn't match the layout, any other stray text in the table, or a running balance that disagrees with the amounts becomes a row error (statement unreconciled). PDF text is treated as data only. `reportlab` (dev dependency) is used by the tests to generate PDFs.

## Known gaps (Phase 1)

These parts of the design are not implemented yet:

- **Encryption at rest and per-household keys.** The SQLite database is plaintext. Use synthetic data only.
- **Statement formats.** CSV and text-based PDF statements are supported (see below). **OCR is not built**: a scanned PDF with no text layer is rejected. The agent's `ingest_statement` tool still takes CSV text only; PDFs go in through `pipeline.ingest_statement`/`ingest_file`.
- **Reviewer override.** A reviewer can't yet mark a `qualified` statement as confirmed.
- **Agents and orchestration.** The Supervisor and both specialists run end to end in `vertical_slice`, but only with offline playback models so far; a real-model run on the Ollama machine is still needed. Retrieval uses word-based chunking (words approximate tokens) and a regex injection detector. Accounts and members for the MCP server still come from the synthetic household.
- **Planner inputs.**
  - Planner inputs are synthetic. Contribution room values are placeholders, not IRS limits.
  - Planning inputs are derived from the ledger (`plan_inputs.py`), but the demo's step 5 still uses the fixed synthetic inputs. Contribution room and return assumptions are still passed in rather than retrieved.
  - The model persona generator and critic (`llm_planner.py`) are tested only with fake models so far. Their prompts haven't been tuned against `qwen2.5:7b` / `qwen3:8b`, and the demo still uses the template generator. Real-model runs happen on the Ollama machine.

- **Briefing and evals.**
  - The briefing is produced by `vertical_slice` from the Supervisor's results, but the Supervisor graph's own `compose_briefing` step still returns only the claim list.
  - No household net-worth total is shown, because property, loans and investments aren't in the ledger.
  - The eval set is template-generated, not a human-labeled held-out set.
  - Membership inference is not scored (no fine-tuned model). The canary test covers runtime surfaces only.
  - Retrieval eval numbers use offline hash embeddings; `nomic-embed-text` numbers need the Ollama machine.
  - Real-model eval numbers still need a run, either with Gemini (`GEMINI_API_KEY`), Groq or on the Ollama machine. The hosted clients are tested only offline so far; tool calling has not been checked against the live APIs.

## Data policy

Use **synthetic data only** until encryption at rest lands. Never commit real statements; `*.db`, `*.sqlite`, `.env` and `.cache/` are git-ignored. Hosted models (Gemini, Groq) must only ever see synthetic data: their free tiers are outside the household-controlled environment, and Google may use free-tier prompts to improve its products (design §11).
