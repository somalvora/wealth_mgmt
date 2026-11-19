# Architecture

The Household Wealth Agent is a local-first multi-agent assistant for household finances (Python, LangGraph, FastAPI demo, offline eval harness). This file describes the system as built. Setup, commands and known gaps are in [README.md](README.md); the live routing run is in the Evaluation notes there.

**Core rule:** a model never does arithmetic, never decides access and never writes to the ledger. Code computes every number and enforces every permission; a model only chooses which tool to call and how to explain the result, and every claim it makes is re-checked against recorded evidence. All data in the repo is synthetic.

## 1. Big picture

```
 user -> Router (model JSON, else keyword rules) -> Supervisor graph
            |- money movement ........................ refuse (audited deny)
            |- allocation request .................... plan_allocation -> approval interrupt
            '- dispatch (<= 8 items, <= 300 s)
                 |- Ledger Analyst  (ReAct, <= 6 tool calls) -> ToolSession -> ledger + calculations
                 '- Policy & Compliance (ReAct)             -> retrieve_verified -> ChromaDB
               -> threshold check -> [review interrupt] -> compose briefing (claims re-validated)

 ingest: parse (CSV/PDF) -> categorize -> reconcile -> LedgerStore (SQLite, append-only)
 every step: Authorizer (deny by default) and hash-chained AuditLog
```

Layers:

- `household_finance`: pure, deterministic library (no I/O, no floats).
- `household_agent`: ingestion, ledger, authorization, audit, tools, agents, retrieval, planning, briefing, evals, web demo.

## 2. Module map

| Module | Responsibility |
|---|---|
| `household_finance.money` | Money as integer minor units; `Decimal` percentages, half-up rounding |
| `household_finance.calcs` / `risk` / `allocation` | Cash flow, category totals, variance, recurring drift, amortization; `cluster_risk`; allocation expansion, constraints, scoring |
| `models`, `categorize` | Pydantic `Account` / `Transaction`; rule-based merchant normalization |
| `ingest` | `parse_statement` picks CSV or PDF by `%PDF-` magic bytes; line provenance, `source_hash` |
| `reconcile`, `pipeline` | Reconciliation checks and status; end-to-end ingest |
| `ledger` | Append-only `LedgerStore` (statements, transactions, calculation runs, brokerage months) |
| `calc_runner` | Allowlisted `REGISTRY`, eligibility rules, provenance, `REQUIRED_PARAMS` |
| `brokerage` | Fictional brokerage account: `brokerage_performance`, `brokerage_concentration`, limits in `LIMITS_BPS` |
| `authz`, `consent`, `audit` | Deny-by-default `Authorizer`; persistent `ConsentStore`; hash-chained `AuditLog` |
| `tools`, `mcp_server` | `ToolSession` (identity and purpose bound by the host) and the stdio MCP server over it |
| `agents` | `Router`, `Supervisor` graph, `LedgerAnalyst`, `PolicyCompliance`, typed `TaskAssignment` / `AgentResult` |
| `retrieval` | `RetrievalService` over ChromaDB: authorize-first filters, MMR, per-passage verification, `validate_claims` |
| `planning`, `llm_planner`, `approval`, `plan_inputs` | Tree-of-Thought allocation planner, model personas and critic, durable approval graph, ledger-derived inputs |
| `allocation_tool`, `allocation_advice` | `evaluate_allocation_candidates` (Supervisor-side decision support) and the model's bounded part |
| `reminders` | Draft reminders and local `.ics` export |
| `briefing`, `history` | Deterministic monthly briefing; two years of fictional statements for the demo |
| `llm` | The only place models are named (`HH_CHAT_MODEL`, `HH_EMBED_MODEL`: Ollama, Gemini, Groq); rate limiter, answer cache, offline `HashEmbeddings` |
| `webapp`, `static/demo.html`, `sample_statements`, `synthetic` | FastAPI chat demo, generated sample PDFs, synthetic households |
| `evals`, `eval_quality`, `eval_brokerage`, `eval_resume` | Eval harness, scorers and oracles, resumable live routing eval |
| `vertical_slice`, `demo` | Design exit-criteria run (14 checks) and a short terminal demo |

## 3. Ingestion and reconciliation

1. **Parse.** CSV (`hh_statement_csv_v1`) or text-layer PDF (`pypdf`; at most 5 MB and 20 pages; encrypted, corrupt, scanned or over-limit files are rejected; text is matched against fixed patterns only and never followed as instructions). Both produce one `ParsedStatement` with every row's source line. Row problems are kept as errors, not dropped. `source_hash = sha256(bytes)`.
2. **Categorize.** Case-insensitive rules give a merchant, category and confidence; no match gives `uncategorized` at 0.0.
3. **Reconcile** (tolerance 0 cents). Material: row errors, unsupported currency, invalid period, rows outside the period, opening + rows != closing, credit/debit totals != header, already-ingested row hash. Bounded: identical rows in one file. Info: reversal pairs within 3 days. Any material issue gives `unreconciled`; otherwise any bounded issue gives `qualified`; otherwise `reconciled`. `document_confidence = 0.35*row_parse + 0.25*balance_continuity + 0.10*totals_match + 0.10*duplicate_free + 0.20*mapped_merchant_rate`; below 0.85 the statement is flagged `unverified`.
4. **Persist.** `LedgerStore.ingest_document` authorizes `upload` on the account, rejects a duplicate `source_hash`, writes document, raw rows and transactions, and audits. The agent's `ingest_statement` tool takes CSV only and only for the host-only `statement_ingestion` purpose, matched by sha256 to the task's own input; PDFs enter through `pipeline.ingest_statement`.

Reading back: `query_transactions` authorizes `read` per account and by default returns only `reconciled` rows with confidence >= 0.85; `include_provisional=True` adds `qualified` and `unverified` rows; `unreconciled` rows are never returned.

## 4. Calculations

`CalculationRunner.run` looks the function up in the allowlisted `REGISTRY` (`cash_flow`, `category_totals`, `variance_between_transactions`, `recurring_drift`, `cluster_risk`), rejects unreconciled input (provisional input only with `allow_provisional`, and the result is flagged), and stores a `calculation_runs` row with canonical-JSON input and output hashes and the library version. The `run_calculation` tool description is built from `REQUIRED_PARAMS`, and a missing param is refused by name.

**Brokerage** (`brokerage.py`). A brokerage statement is positions, not rows, so it has its own table (`brokerage_statements`) and two calculations, `brokerage_performance` (opening, deposits, dividends, market effect, closing, percent change) and `brokerage_concentration` (largest position, top-four share, technology share, breaches). A month must continue the previous month's closing. Concentration limits live in one place (`LIMITS_BPS`: 20% single stock, 50% technology, funds exempt); both the calculation and the generated policy document are written from it, so they cannot disagree. The account exists only in the demo household.

## 5. Authorization, consent, audit

Authorization rules (`authz_1.0`), first match wins: unknown account, other household, or `dependent` role all deny; an owner of the account is allowed (owning one account gives no access to another member's private account, even for an admin); a `delegate` may only `read` / `report` / `report_aggregate`; otherwise an active `ConsentGrant` from **every** owner is needed, matching action and purpose (delegate grants must expire; revoked or expired grants do not count). Every decision, allow or deny, is audited with the policy and consent versions.

`ConsentStore` persists grants and revocations as append-only rows (a revocation is a new row, never an edit). Only the grantor can grant for an owned account, every change is audited, and grants reload into the `Authorizer` on restart.

`AuditLog` is a hash chain (`prev_hash` -> `hash` from a genesis of 64 zeros) with `verify()`. Append takes a lock so concurrent specialists cannot fork the chain; the ledger, consent store and audit log share one SQLite connection and one re-entrant lock held across whole transactions.

## 6. Tools and agents

**ToolSession.** Identity, purpose, correlation ID and call budget are bound by the host and never appear in a tool schema. Tools: `list_accounts`, `query_ledger`, `run_calculation`, `ingest_statement`, `brokerage_performance`, `brokerage_concentration`. `run_calculation` resolves transaction IDs only from what this session already queried. Errors are structured (`authorization_denied`, `budget_exhausted`, `job_cancelled`, `statement_mismatch`, `ineligible_input`, `duplicate_document`, `invalid_statement`, `calculation_error`, `invalid_request`), and tool errors name the fix so a model can self-correct. The MCP server exposes the same tools; it reads `HH_ACTOR` / `HH_PURPOSE` once at startup and refuses to start without them.

**Router.** One model call returns a validated `RoutingDecision` (intent, dispatch plan of at most 8 items, external-action flag); an unusable reply falls back to keyword rules (`rules_fallback`). Intents a model may choose: historical comparison, tax context, both, allocation planning, out of scope (`statement_ingestion` is host-only); brokerage questions route to the Ledger Analyst, and concentration questions also to Policy.

**Supervisor graph.** `classify` -> one of `refuse`, `plan` / `allocate`, `dispatch` -> `threshold` -> `review` (only with a checkpointer) -> `remind` -> `compose`. Each step is audited under the job's correlation ID.

- **Refusal.** A money-movement request ("pay", "transfer", "wire", ...) ends `refused` with an audited deny and nothing is dispatched. The model can set but not clear the external-action flag (rules and model are OR-ed). In the demo, a question naming a member whose private account the asker cannot read is refused before any model runs, with an audited deny (`DemoService._private_account_refusal`); rent and property-tax questions are about the joint account and are exempt.
- **Handoffs.** `TaskAssignment` and `AgentResult` are frozen pydantic models carrying the correlation ID, task ID and agent. A result that does not match is discarded as `failed`. A specialist that raises becomes a `failed` result.
- **Budgets.** At most 8 dispatches and 300 seconds per job; each specialist gets 6 tool calls and 2 retries per tool. Independent plan items run concurrently in a thread pool; dependent items (Policy after Ledger) run alone. After the deadline a per-task cancel event makes later tool calls refuse (`job_cancelled`).
- **Ledger Analyst.** ReAct over the `ToolSession`. Outputs and evidence IDs are copied from tool observations, not from model text; claims may cite only IDs from this assignment.
- **Policy & Compliance.** ReAct over `retrieve_verified`: at most 3 searches (1 plus `MAX_QUERY_REFINEMENTS = 2`). An exact repeat of a search (case and spacing ignored) returns the first result with no new search, no audit row and no budget use; a fourth distinct search returns `budget_exhausted`. Claims go through `validate_claims`; an invalid final answer gets one repair call (field paths and error types only) and is re-validated.
- **Threshold check.** A variance above 15% in a category is a `material_variance` finding (property tax: 10%). Any non-`completed` result or any finding makes the job `needs_review`. A failed reconciliation (unreconciled or unverified statement) becomes a `data_quality` finding; figures derived from a flagged document are left out of the briefing.
- **Review interrupt (checkpointer only).** The job pauses at `interrupt()` with status `awaiting_review`. `Supervisor.resume` continues it, even from a new process with a SQLite saver. The reviewer must be an admin or adult of the same household and be authorized for the job's purpose on every account it used; anyone else is audited as a deny and the job keeps waiting, and after 5 denials it stops waiting as `needs_review`. Specialists do not re-run on resume.
- **Allocation.** `plan` starts the approval graph (section 7). When the host supplies a ledger and `AllocationFacts` (debts, APRs, targets, minimum buffer), `allocate` instead runs `evaluate_allocation_candidates`: code evaluates every candidate plan in integer cents (interest saved, reserve, buffer months, violations) and ranks them; a model may add at most five shapes (bucket plus share, expanded to cents by code) and pick one, and an invalid pick falls back to the best valid plan. Only an adult or admin acts, for purpose `capital_allocation`; each run is stored as `alloc_run_<id>` evidence.
- **Reminders.** A "remind me" request becomes a draft built from the job's own evidence. With a checkpointer it needs approval by an authorized adult or admin; approval writes a local RFC 5545 `.ics` file and nothing else.

## 7. Planner and approval

`plan_allocation` is a bounded Tree-of-Thought beam search over three tiers (Safety, Tax-advantaged, Growth): 3 thoughts per node, beam of 2, so 3 + 6 + 6 = 15 evaluations (27 at worst when the critic retains discrepancy branches). A thought is a strategy (persona, share, target buckets) and never an amount; `expand_step` converts it to cents. Hard prunes (score 0): statutory cap exceeded, limit unverified, liquidity floor breached, unallocated surplus, wrong tier or bucket. Soft prune below 0.65. Score = 0.40 tax efficiency + 0.35 liquidity safety + 0.25 expected net yield. The optional critic is advisory: a gap over 0.20 retains the branch and sets `needs_review`, but it never prunes. `TemplateThoughtGenerator` is the deterministic default; `LLMThoughtGenerator` / `LLMCritic` sit behind the same protocols, validate model output with pydantic (`extra="forbid"`, no amounts) and fall back per persona to the template, recording each fallback. Plan inputs come from verified statements and recorded calculation runs (`plan_inputs`), never from model numbers, and only from accounts the asker may read.

**Approval graph** (`approval.py`): `plan` -> `approval` (`interrupt`) with a SQLite checkpoint (its own connection and file). Only an admin or adult of the household can approve (only boolean `True` counts); a wrong approver is audited as a deny and the graph keeps waiting. On resume the input hash is recomputed and a mismatch ends `stale`. A plan has `requires_approval=True` and no execution path: approving moves no money.

## 8. Verified retrieval

`retrieve_verified` requires an actor, purpose and query. It derives the accounts the actor may `RETRIEVE` (plus `PUBLIC`; dependents get none) and queries ChromaDB with a `where` filter on household, account scope and active status, so unauthorized chunks never enter the candidate set. Up to 20 candidates go through MMR (lambda 0.65) to at most 5 passages, each verified (chunk hash, expiry, tax year) as `verified`, `hash_mismatch`, `expired` or `wrong_period`, and the retrieval is audited, including instruction-like chunks (flagged, never followed). Embeddings always come from `llm.get_embeddings()`. Chunking: 500-word windows with 100 overlap; statements in 25-row groups with the header repeated.

`validate_claims` returns `supported`, `remove` or `rephrase_as_question`. A tax claim needs a verified Tier-1 passage for the same tax year; numbers must appear in the cited evidence or calculation output; citing anything not retrieved in this job removes the claim; and a cited claim must share at least half its content words with its passages (a lexical check: it catches a wrong source, not a subtle paraphrase error).

## 9. Monthly briefing

`draft_monthly_briefing` is deterministic; no model writes any section. Seven sections (data coverage, net worth and cash flow, budget and variance, material events, policy and deadlines, actions for approval, assumptions and limitations ending with a disclaimer), `draft=True`, `external_action="none"`. Items with amounts must cite evidence. Only accounts readable for `monthly_briefing` contribute (others appear as a count); unreconciled statements contribute coverage notes, never numbers. Category totals and cash flow come from the same verified set, so they agree. `cluster_risk` raises review prompts for new-merchant small-charge clusters and bursts against a 90-day baseline.

## 10. Web demo (`webapp.py`, `static/demo.html`)

`DemoService` wires one household's ledger, retrieval index, approval graph and models; `create_app` serves it with FastAPI on 127.0.0.1 with no login (the user picks a member: somal admin, sayali adult, viaan dependent; the CPA delegate is not selectable). The clock is fixed at `DEMO_AS_OF`. It adds no capability: the guardrails are the code above. The demo Supervisor runs without a checkpointer, ledger facts or reminder directory, so allocation uses the approval graph and the review and reminder paths are exercised in tests and the vertical slice, not in the page.

Endpoints: `GET /api/state`, `GET /api/samples/{name}`, `POST /api/upload` (CSV or PDF as a chosen member; 422 `rejected`, 403 `refused`, 409 `duplicate`), `POST /api/chat`, `GET /api/trace/{id}` (polled while a job runs; client IDs must start `corr_ui_` and be unused), `POST /api/approve`, `GET /api/audit` (chain verification plus recent events), `GET /api/consent`, `POST /api/consent/share`, `POST /api/consent/revoke`, `GET /api/eval`, `POST /api/eval/run` (offline gates, keyword router, about 2 s).

Panels: Ask (with 10 suggestion chips), You are, Share my accounts (consent: read and retrieve only, never upload or approve; dependents cannot be granted access), Upload a statement (plus five generated sample PDFs), Documents on file, What the agents did (per-job trace), Audit trail (Verify chain), Evaluation results (16 gates in plain words). The ledger is preloaded with 24 months of fictional statements (Oct 2024 to Sep 2026) for three accounts, Oct 2026 for the joint rental account, three months of brokerage statements, and the policy documents; reconciled statements are also indexed into `financial_history`. One lock serializes chat, upload, approve and consent. Safety: model and PDF text is written with `textContent` only (a test forbids `innerHTML`), IDs are regex-validated, messages are capped at 2,000 characters, uploads are size-capped. Without `--model`, specialists replay scripts and the router uses keywords, so only the property-tax question has a scripted answer.

## 11. Evaluation harness

`python -m household_agent.evals` runs 416 seeded synthetic scenarios (seed 0) in 12 areas through the real components and scores 16 gates (`EVAL_VERSION = "evals_0.4"`). It exits 1 if a gate fails. Oracles are independent of the code under test.

| Area | Scenarios | Gate(s) |
|---|---|---|
| routing | 55 | `routing_accuracy` >= 0.95; `schema_validity` >= 0.99; `approval_gating` == 1.0 |
| adversarial | 14 | `unsafe_actions` == 0 (every request ends `refused`) |
| authorization | 72 | `cross_member_leaks` == 0 |
| allocation | 60 | `allocation_invariants` == 1.0; `max_node_evaluations` <= 27 |
| ingestion | 30 | `statement_integrity` == 1.0 |
| calculation | 20 | `calculation_match` == 1.0 (briefing figures equal an oracle) |
| brokerage_performance | 29 | `brokerage_performance_match` == 1.0 |
| brokerage_concentration | 39 | `brokerage_concentration_match` == 1.0 (limits read from the written policy) |
| anomaly | 40 | `anomaly_recall` >= 0.90 |
| retrieval | 22 | `retrieval_recall_at_5` >= 0.90; `citation_precision` >= 0.95 |
| search_count | 27 | `policy_search_count` == 0 violations (repeats collapse, never more than 3 searches) |
| privacy | 8 | `privacy_canary_exposures` == 0 (and owners see their canary) |

Offline (keyword router, deterministic planner, hash embeddings) all 16 gates pass; the one scenario miss is `retr_005`, where hash embeddings rank the water-heater invoice below 5. With real models, add `--model`, `--planner-model`, `--embed-model` (`ollama:`, `gemini:`, `groq:`); the CLI makes one preflight call per model and exits 2 on failure, aborts early (exit 2) if more than 80% of router or critic calls fall back, and `--only-area` runs one area (the other gates are then vacuous). `--allocation-scenarios N` shrinks the costly allocation set. Live routing can be finished across sittings with `--only-area routing --resume FILE` (`eval_resume`): the checkpoint stores only real model replies, a failed call is never counted and the keyword rules never stand in, exit 3 means paused (run again) and exit 2 means the model or router prompt changed. The report prints only when every request has a model reply. Live run on `gemma-4-26b-a4b-it`: routing 55/55 (accuracy 1.0, schema_validity 1.0, no fallbacks) and all 16 gates pass (see Evaluation notes in README.md).

## 12. Vertical slice (`vertical_slice.py`)

`run_vertical_slice` exercises the design exit criteria end to end with file-backed stores so the restart is real, in 14 checks: ingest and reconcile, three-agent trace, provenance chain, authorization denial, consent revocation, unreconciled block, stale-source escalation, planning reproducibility, cap-breach hard prune, critic discrepancy review, draft briefing, interrupt and resume across a restart, no external actions, and audit-trail integrity. It runs offline with `PlaybackChatModel` scripts, or with real models via a factory.

## 13. Storage

SQLite tables `documents`, `statement_balances`, `raw_rows`, `transactions` (corrections insert `version + 1`), `calculation_runs`, `brokerage_statements`, `audit_events`, `consent_grants` and `consent_revocations` are append-only (`BEFORE UPDATE` / `BEFORE DELETE` triggers abort). `transaction_id` is `txn_` plus a row-hash prefix that includes an occurrence index, so two identical charges stay distinct. LangGraph checkpoints use a separate `checkpoints.db`; the vector index is a ChromaDB directory whose chunk metadata carries household, account scope, authority tier, tax year, validity, chunk and source hashes and an injection flag. The database is plaintext (no encryption at rest).

## 14. Guardrails at a glance

| Risk | Guardrail | Where |
|---|---|---|
| Float or rounding error | Integer cents, floats rejected | `money` |
| Model does arithmetic or invents inputs | Allowlisted calculations; tool takes only IDs this session queried | `calc_runner`, `tools` |
| Bad statement corrupts answers | `unreconciled` never queryable or calculable | `ledger`, `calc_runner` |
| Silent edits, double import | Append-only triggers; `source_hash` unique | `ledger` |
| Cross-member leak | Deny by default; all-owner consent; retrieval filter before search | `authz`, `retrieval` |
| Untraceable actions | Hash-chained audit with `verify()` | `audit` |
| Model impersonates a member | Identity bound in `ToolSession` / server env, not in schemas | `tools`, `mcp_server` |
| Money movement | Refused and audited; no execution path anywhere | `agents`, `approval` |
| Plan treated as an action | `requires_approval`; approver check; stale-input check | `approval` |
| Unsupported numbers or wrong-year tax claims | `validate_claims`; Tier-1 same-year source | `retrieval`, `agents` |
| Prompt injection in documents | Instruction-like chunks flagged and audited; text treated as data | `retrieval` |
| Runaway loops | Dispatch, time, tool-call and search budgets | `agents` |
| Result from another job | Correlation, task and agent IDs checked | `agents` |
| Flagged result accepted by the wrong person | Review interrupt with household and account authorization | `agents` |
| Model failure | Router falls back to rules; planner to templates; failures recorded | `agents`, `llm_planner` |
| Regressions | 416 scenarios, 16 gates, CLI exit code | `evals` |

## 15. Extensions in future

Not built; listed because they matter for a real deployment or a stronger evaluation.

- **Membership-inference eval.** The only design success metric not scored (`NOT_COVERED`); it needs a fine-tuned model.
- **Live evals beyond routing.** Retrieval, allocation, the model planner personas and critic have only run offline (hash embeddings, deterministic or scripted models); retrieval recall in particular is not measured with real embeddings.
- **Answer-quality evals with real models.** Ledger Analyst and Policy answers are tested only with scripted models; the eval set is template-generated, not human-labeled.
- **Indirect injection tests.** Injection is covered by unit tests and flagged chunks, but there is no eval area that plants instructions in retrieved documents or uploaded statements.
- **OCR.** Scanned statements are rejected; there is no OCR path.
- **Reviewer confirmation of `qualified` statements.** Accepting a review does not make a document usable, and there is no automatic re-ingest or repair.
- **Review expiry and idempotency.** Pending reviews never expire; tool calls have no idempotency keys; a hung specialist thread is abandoned, not killed.
- **Encryption at rest and per-household keys.** The ledger is plaintext SQLite.
- **Real calendar integration.** Reminders only write a local `.ics` file; no sending, recurring events or notification agent.
- **Real tax data.** Contribution-room values are placeholders; allocation has no brackets, employer match or return forecasts; the briefing shows no household net worth.
- **Regulatory-conflict detection.** Retrieval checks each passage's freshness, tier and tax year but does not compare passages against each other.
