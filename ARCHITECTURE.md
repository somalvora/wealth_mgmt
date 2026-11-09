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
 │                                                                                        │
 │  planning.plan_allocation (ToT beam search) ─► household_finance.allocation            │
 │    ThoughtGenerator (template personas) ─► expand / validate / rollout / score         │
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
| `household_agent.ingest` | `parse_statement` (picks CSV or PDF by `%PDF-` magic bytes), `parse_statement_csv`, `parse_statement_pdf` (`pypdf`, text layer only) with line provenance | 3.1, 6 | `parse_statement` |
| `household_agent.categorize` | Rule-based merchant normalization | 6 | (part of ingest) |
| `household_agent.reconcile` | Checks, status, `document_confidence` | 3.3 | `reconcile_statement` |
| `household_agent.ledger` | Append-only `LedgerStore` | 6 | `query_ledger` |
| `household_agent.calc_runner` | Allowlist, eligibility, provenance | 3.2, 3.4 | `run_calculation` |
| `household_agent.pipeline` | Orchestrates ingest end to end | 12 | — |
| `household_finance.risk` | `cluster_risk`: new-merchant small-charge clusters, one-day bursts, card-testing probes | 5.4, 5.5 | `run_calculation` (`cluster_risk`) |
| `household_finance.allocation` | Strategy → cents expansion, hard constraints, composite score, rollout | 3.4, 5.8 | `evaluate_allocation_candidates` |
| `household_agent.planning` | Tree-of-Thought beam search, `ThoughtGenerator` and `Critic` protocols, template personas | 5.8 | (Supervisor planning subgraph) |
| `household_agent.llm` | The only place models are named: `get_chat_model()` / `get_embeddings()` from `HH_CHAT_MODEL` / `HH_EMBED_MODEL` (Ollama, or Gemini/Groq for synthetic-data runs; Gemini embeddings too); one shared rate limiter per hosted provider, `SQLiteChatCache`, offline `HashEmbeddings` | 3.5 | — |
| `household_agent.consent` | Persistent `ConsentStore`: append-only grants and revocations, grantor-only, audited, reloaded into `Authorizer` | 7 | (consent management) |
| `household_agent.tools` | `ToolSession`: identity/purpose bound by the host, call budget, structured errors, session-scoped transaction IDs | 3.2, 4.1 | all Ledger Analyst tools |
| `household_agent.mcp_server` | `household-finance-tools` FastMCP stdio server over `ToolSession`; identity from env | 3.2, 4.2 | (MCP transport) |
| `household_agent.agents` | `Supervisor` graph (classify → dispatch loop → threshold → [review interrupt, with a checkpointer] → compose), `LedgerAnalyst` and `PolicyCompliance` ReAct subgraphs, `Router`, typed `TaskAssignment`/`AgentResult` | 4, 5.1 | (the agents) |
| `household_agent.retrieval` | `RetrievalService` over ChromaDB: authorize-first `where` filters, MMR, per-passage verification, audit; `validate_claims` | 8 | `retrieve_verified`, `validate_claims` |
| `household_agent.approval` | LangGraph `plan → approval(interrupt)` graph with a checkpointer; approver check and stale-input check | 5.8, 10 | (Supervisor approval subgraph) |
| `household_agent.llm_planner` | `LLMThoughtGenerator` (model personas, pydantic-validated, template fallback) and `LLMCritic` (clamped, advisory) behind the planner protocols | 5.8 | (planner personas / critic) |
| `household_agent.plan_inputs` | `derive_planning_inputs` from verified statements + recorded `category_totals`; `record_plan_run` / `plan_from_ledger`; `ledger_inputs_for` for the approval graph | 3.4, 5.8 | (planner inputs) |
| `household_agent.briefing` | `draft_monthly_briefing` (seven §10 sections, deterministic, evidence-required `BriefingItem`s) and `render_markdown` | 10 | `compose_briefing` |
| `household_agent.evals` | `generate_scenarios(seed)` (321 seeded scenarios, 9 areas), `evaluate` → `EvalReport` with 13 §2 gates, CLI | 2, 11 | — |
| `household_agent.eval_quality` | Calculation, anomaly, retrieval/citation and privacy-canary scenario data, independent oracles and scorers (`RetrievalWorld`, `PrivacyWorld`) | 2, 11 | — |
| `household_agent.vertical_slice` | `run_vertical_slice(workdir, models=, embeddings=)` → `SliceReport` of 14 `Check`s; `PlaybackChatModel` offline stand-in; CLI | 12 | — |

## 3. Data flow: ingesting a statement

1. **Parse.** `parse_statement` chooses the parser from the bytes: `%PDF-` means the PDF parser, anything else CSV. The PDF parser rejects files over 5 MB or 20 pages, encrypted or corrupt files, and PDFs with no text layer (scans need OCR, which is not built). It reads labelled header lines and a `Date Description Amount [Balance]` table from the extracted text, treats that text as untrusted data (matched against fixed patterns only, never followed as instructions), checks any running-balance column against the amounts, and converts amounts to integer cents with the same `to_minor` helper. Both parsers produce the same `ParsedStatement`, so everything below is shared. The agent's `ingest_statement` tool passes `allow_pdf=False`, so over that tool a PDF is refused; PDFs enter through `pipeline.ingest_statement`/`ingest_file`. The parser validates header fields, column names and every row. Row problems become `RowError`s and are kept rather than dropped. The parser computes `source_hash = sha256(file bytes)`.
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

## 4a. Data flow: Tree-of-Thought capital allocation (`plan_allocation`)

```
PlanningInputs (reconciled cash, essential outflow, liabilities, verified room, assumptions)
   │  surplus = cash − 6×monthly essentials − near-term liabilities   (≤ 0 → not_eligible)
   ▼
root ──propose(tier 1)──► 3 thoughts ──expand_step → validate → rollout → score──► keep top 2
         propose(tier 2) on each of 2 ──► 6 children ─────────────────────────────► keep top 2
         propose(tier 3) on each of 2 ──► 6 leaves (always allocate everything left) ─► rank
   ▼
PlanResult(status=awaiting_approval | needs_review | no_compliant_plan | not_eligible,
           best, ranked_leaves, trace[15 nodes], input_hash, versions, requires_approval=True)
```

**Thoughts and budget.**

- A `Thought` carries an `AllocationStep(tier, fraction, buckets, mode)` and a rationale, never an amount. `expand_step` converts it to cents:
  - `split` uses largest-remainder apportionment, so the parts sum exactly;
  - `fill_in_order` fills buckets in priority order up to each bucket's verified room.
- Each run makes 3 + 6 + 6 = 15 node evaluations. If the critic retains discrepancy branches (at most 2 extra per depth), the worst case is 27.

**Scoring and pruning.**

- `score_allocation` = 0.40·tax_efficiency + 0.35·liquidity_safety + 0.25·expected_net_yield, quantized to 4 decimal places.
- **Intermediate nodes are scored by rollout**: undecided tiers are filled with the Balanced default, so a depth-1 node isn't penalized for tax room it hasn't decided yet.
- **Hard prune** (score 0) on any of: `statutory_cap_exceeded`, `limit_unverified`, `liquidity_floor_breached`, `unallocated_surplus` (at leaves), `bucket_not_in_tier`, `wrong_tier`.
- **Soft prune** below 0.65.
- **Beam selection** is iterative: among nodes within 0.02 of the current best, the highest liquidity wins, then the higher composite, then the node ID.

**Critic** (optional, advisory).

- If |critic − composite| > 0.20, the node is flagged and retained even if it would have been cut, up to 2 per depth.
- Any flag sets the status to `needs_review`. The critic can never prune a branch.

**Generators.** `TemplateThoughtGenerator` produces fixed Conservative, Balanced and Tax-Aggressive strategies for Phase 1. `llm_planner.LLMThoughtGenerator` is the Phase 2 model-driven version behind the same protocol:

```
beam node ──► one model call ──► {"thoughts": [...]} ──► per persona:
                                                        pydantic (extra="forbid", fraction ∈ [0,1],
                                                        weights > 0, no currency in rationale)
                                                          │ valid                 │ invalid / missing
                                                          ▼                       ▼
                                                     Thought(model)       template Thought + fallback record
                                                          └──────────┬────────────┘
                                                                     ▼
                                              _evaluate: wrong tier / bucket / cap ──► hard_pruned
```

- There is one model call per beam node, so 1 + 2 + 2 = 5 calls per run, which matches the "≤ 5 generation calls" target.
- **Structure check.** The schema rejects every field it doesn't define. A model that sends `amount_minor` (or any amount) gets that persona replaced by its template thought, and the fallback is recorded in `generator.fallbacks`.
- **Rule check.** Breaking a planning rule is *not* repaired. A wrong tier or a bucket outside the tier reaches `_evaluate` and is hard-pruned, so the trace shows the reason.
- **Reply cleanup.** `<think>` blocks (qwen3) and code fences are stripped before parsing.
- **Versioning.** `version = "llm_generator_1.0+<model spec>"` goes into the input hash, so template and model plans never share a hash.
- **Critic.** `LLMCritic` asks for `{"score", "reason"}`, clamps the score to [0, 1] and quantizes it to 0.0001. An unusable reply (bad JSON, NaN, extra fields) returns the deterministic composite, so it can't trigger a false discrepancy, and the failure is recorded in `critic.failures`.
- **Approval.** `build_approval_graph(..., generator=, critic=)` plans with these. On approve it only recomputes `inputs_hash(...)`, so approving makes no model calls.

## 4b. Data flow: approval with a durable pause (`approval.py`)

```
start_request(household, requested_by) ─► plan ─► plan_allocation ─► audit "plan_allocation"
                                            │ not_eligible / no_compliant_plan ─► END
                                            ▼
                                        approval ── interrupt({kind, correlation_id, plan}) ──► caller
                                            ▲                     (state checkpointed in SQLite)
   Command(resume={approver, approved, note}) ──────────────────────────────────────┘
        approver not admin/adult of this household ─► audit deny, keep waiting
        recomputed input_hash ≠ shown input_hash   ─► audit deny, status = stale
        otherwise ─► audit allow, status = approved | rejected   (nothing executes)
```

- **State.** The graph state is JSON-safe: integer cents and decimal strings (`plan_summary`). The thread ID doubles as the audit `correlation_id`.
- **Approval value.** Only the boolean `True` counts as approval; `"yes"` is a rejection.
- **Checkpoints.** `SqliteSaver` needs **its own connection and file**. It manages its own transactions and can write from a worker thread, so sharing the ledger/audit connection fails intermittently ("cannot commit - no transaction is active").
- **Re-execution.** On resume, LangGraph re-runs the `approval` node from the top. That's why the node has no side effects before `interrupt()`.

## 4b-2. Planner inputs from the ledger (`plan_inputs.py`)

```
actor, as_of ─► window = [first of month (as_of − 3 months), as_of]
   for each household account:
     not READ-able for capital_allocation ─► excluded
     verified_statements (reconciled, confidence ≥ 0.85, period inside window)
        none ─► excluded
        latest closing: checking/savings → liquid += max(closing, 0)
                        credit_card      → liabilities += max(−closing, 0)
        query_transactions over the statements' span ─► category_totals run (recorded)
        monthly_essential += ceil(essential outflow / verified months)
   PlanningInputs(liquid, monthly_essential, explicit liabilities + card owed, room, assumptions)
   ─► plan_allocation ─► record_plan_run: calculation_runs(fn="plan_allocation",
          input_hash = planner hash, input_ids = docs + calc runs + liability/limits evidence,
          provisional = any account excluded) + audit "calculate"
```

- **No model numbers.** Every input amount is a stored closing balance, a recorded calculation output, or an explicit liability with an evidence ID.
- **Essentials.** mortgage, rent, utilities, groceries, hoa_dues and insurance. Property tax is excluded on purpose; the next installment is passed as a liability.
- **Privacy.** A member's plan uses only accounts they may read. Jordan's plan leaves out Alex's private checking, and the run says so.
- **Stale approvals.** `ledger_inputs_for` is the approval graph's `inputs_for`. It re-derives on resume, so a newly verified statement changes the input hash and the approval ends `stale`.

## 4c. Tool calls: agents and MCP clients (`tools.py`, `mcp_server.py`)

```
host authenticates user ─► ToolSession(store, actor, purpose, correlation_id, max_calls)
                               │  bound once; never a tool argument
   LangGraph agent ──call(name, args)──┤
   MCP client ──stdio── FastMCP tool ──┘──► list_accounts | query_ledger | run_calculation | ingest_statement
                                               │ (LedgerStore authz + audit, CalculationRunner)
                               ◄── {"ok": true, ...} | {"ok": false, "error": {"type", "message"}}
```

- **Session-scoped evidence.** `query_ledger` records the transactions it returned. `run_calculation` resolves IDs only from that record, so the model can't invent inputs, and it can't reuse IDs it saw in another member's session.
- **Error types.** `authorization_denied`, `budget_exhausted`, `ineligible_input`, `duplicate_document`, `invalid_statement`, `calculation_error`, `invalid_request`. Unknown tools and bad arguments are `invalid_request`, never exceptions.
- **MCP identity.** The MCP server reads `HH_ACTOR` / `HH_PURPOSE` once at startup and refuses to start if they're missing or unknown. The tool schemas contain no identity fields; a test asserts this.

## 4d. Verified retrieval (`retrieval.py`)

```
actor + purpose + query
   │ (purpose and query required → else RetrievalRequestError)
   ▼
Authorizer: accounts the actor may RETRIEVE for this purpose (+ PUBLIC; dependents: none)
   ▼
Chroma query on {household}__{collection}, where = household_id AND account_id ∈ scopes
   AND document_status = active [AND jurisdiction ∈ … AND (effective_tax_year ∈ years OR 0)]
   [AND statement_period ∈ …]          ── up to 20 candidates
   ▼
MMR (λ = 0.65) → up to 5 passages
   ▼
verify each: chunk_hash matches text? valid_until / effective_to passed? tax year matches?
   → verified | hash_mismatch | expired | wrong_period
   ▼
audit (scopes, candidate IDs, selected IDs, outcomes, instruction-like chunks)
   ▼
validate_claims(claims, passages, calc_outputs) → supported | remove | rephrase_as_question
```

- **Filter, don't redact.** Unauthorized chunks never enter the candidate set, so nothing has to be removed afterwards and nothing leaks through ranking.
- **Embeddings.** These always come from `llm.get_embeddings()`. Collections are created with `embedding_function=None`, so Chroma never downloads its default ONNX model.
- **Indexing.** Indexing a document needs `index` permission on its account. `PUBLIC` documents must be Tier 1 and indexed by an adult.
- **Chunking.** Documents split on headings first, then into 500-word windows with 100-word overlap. Statements are split into 25-row groups, each with the header repeated.
- **Claims.**
  - A tax claim needs a verified Tier-1 passage whose `effective_tax_year` equals the claim's year. Otherwise it becomes a question carrying the standard "cannot verify" sentence.
  - Numbers in a claim must appear in the cited verified evidence or calculation output.
  - Citing anything not retrieved in this job removes the claim.
  - A claim citing passages must share at least half of its content words (4+ letters, stopwords dropped, simple plurals folded) with them, or it becomes a question ("cited passages do not say what the claim says"). This is a lexical check; it catches a wrong source, not a subtle paraphrase error.

## 4e. Supervisor and specialists (`agents.py`)

```
request ─► classify (Router: model JSON → RoutingDecision, else rule_route)
             ├─ external action ─► refuse (audited deny) ─────────────┐
             ├─ allocation ─► plan: start approval graph (interrupt) ─┤
             └─ dispatch loop (≤ 8, ≤ 300 s; independent items in a batch run concurrently) ─► threshold ─► compose ◄┘
                  │ TaskAssignment(correlation_id,     │   ▲
                  │   purpose from intent,             │   │ (checkpointer only)
                  │   scope from host)                 └─► review ── interrupt({kind, correlation_id,
                  ▼                                            findings, results}) ──► caller
        specialist subgraph: agent ⇄ tools (≤ 6 calls, ≤ 2 retries/tool)
                  │ outputs/evidence copied from observations
                  │ claims re-checked by validate_claims
                  ▼
        AgentResult(correlation_id, status, outputs, evidence_ids, claims, questions, errors)
```

- **Ledger Analyst.** Its tools are `list_accounts`, `query_ledger`, `ingest_statement` and `run_calculation`, through a `ToolSession`. `ingest_statement` is there so a task can hand it a CSV statement to load; the permission check (`upload` on the statement's account) stays inside `ToolSession`/`LedgerStore`, so a member can't ingest someone else's statement through the agent either. The prompt tells it to use `ingest_statement` only for a supplied statement and to prefer `query_ledger` for reads. A calculation's outputs also carry `category` (when all its input transactions share one) and the `account_ids` of its inputs, both copied from the query results: the first picks the review threshold, the second says whose accounts a reviewer must be able to see. The first message gives the model its `tool_budget` and `today` (from the agent's clock; both specialists get it, since a model otherwise resolves "October" from its training date), and the prompt tells it to filter by merchant and month rather than widening and repeating queries (the first live Gemini run used up its budget that way). Claims may cite only the transaction IDs and calculation run IDs from this assignment (a run's evidence includes its input transactions, so a claim citing only the run can name their dates), and `*_minor` values are also matched in dollar form ("$450.00").
- **Policy & Compliance.** Its tool is `retrieve_verified`, allowing 1 search plus 2 refinements. Claims are validated against the passages from this assignment and against calculation outputs the Supervisor passed in as run IDs. `rephrase_as_question` verdicts become questions; `remove` verdicts are dropped and recorded as errors.
- **Handoff payloads.** `TaskAssignment` and `AgentResult` are frozen pydantic models and both carry the job's `correlation_id` (design §4.2). Specialists copy it from the assignment. If a result comes back with a different ID, the Supervisor throws the whole result away (claims, outputs, evidence) and records a `failed` dispatch with a `correlation_id mismatch` error instead of crashing. A result that isn't for this job can't be trusted for anything.
- **Supervisor.**
  - It never calls the ledger or the vector store.
  - A specialist that raises an exception becomes a `failed` result.
  - **Concurrent dispatch.** Each pass of the `dispatch` node runs one *batch* of the plan. A Policy item depends on the Ledger Analyst when an earlier plan item is a ledger item and the Policy item keeps its default `depends_on_ledger=True`; the Ledger's calculation outputs are only handed to Policy in that case. Everything else (two ledger items, a Policy item with `depends_on_ledger=False`, a Policy item with no ledger item before it) is independent. A dependent item runs alone, exactly as before. A run of independent items goes to a small thread pool, one thread each; a single item still runs inline in the Supervisor's own thread. The router's default plan (ledger then policy) is dependent, so the demo and vertical slice behave as they always did. The batch shares the job's budgets: the dispatch count is checked up front and plan order decides who gets the remaining slots (the rest are `budget_exhausted`), the time limit is checked before the batch starts and caps how long the Supervisor waits for it (a specialist still running then becomes a `failed` result with a `TimeoutError`; its thread can't be killed, so it is left to finish in the background). Results are collected in plan order, each one goes through the same crash handling and correlation-ID check, and the `dispatch` audit events are written by the Supervisor after the batch finishes, in plan order, so the Supervisor's own part of the hash chain never depends on which specialist was faster.
  - **Thread safety.** `AuditLog.append` now holds a lock around read-last-hash plus insert, so two specialists auditing at once cannot fork the chain. Each specialist builds its own `ToolSession` per call, so nothing else is shared except the SQLite connection. A connection used from two threads must be opened with `check_same_thread=False` (the tests and the MCP server do; the vertical slice and the demo still open theirs the default way, which is fine because they never run a concurrent batch). The audit events the specialists write for their own tool calls can interleave between the two concurrent specialists, so their order inside a batch is not fixed (the chain still verifies).
  - **Threshold check (design §5.4).** A variance above 15% in any category is a `material_variance` finding; property tax overrides that with 10% (`VARIANCE_REVIEW_PCT` plus `VARIANCE_REVIEW_PCT_BY_CATEGORY`, read through `variance_threshold(category)`). A result with no single category uses the 15% default. Each finding records the threshold and category it was checked against.
  - Any status other than `completed`, or any threshold finding, makes the job `needs_review`.
  - Each step is audited under the job's correlation ID: `route`, `dispatch`, `threshold_check`, `review` (only with a checkpointer) and `compose_briefing`.
- **Review interrupt (design §5.3), only with a checkpointer.** `Supervisor.run(request, checkpointer=...)` compiles the graph with the given saver (`InMemorySaver` or `SqliteSaver`, thread ID = correlation ID) and adds a `review` node after the threshold check. When there are findings or any result isn't `completed`, the job stops at `interrupt()` with status `awaiting_review` and a payload listing the findings and the unfinished results. `Supervisor.resume(correlation_id, {"reviewer", "accepted", "note"}, checkpointer=...)` continues it, also from a new process when the saver is SQLite. The reviewer must be a household admin or adult member of the same household (the same `APPROVER_ROLES` as approvals) **and**, through the `Authorizer`, allowed `resume` for the job's purpose on every account the results used. Anyone else (a dependent, a delegate, an unknown ID, or Jordan when the job read Alex's private account) is audited as a `review` deny and the job keeps waiting. An allowed decision is audited as `review` allow and ends the job `reviewed` or `review_rejected`; the decision is shown in the briefing. Specialists aren't run again on resume, and the node does nothing before `interrupt()`, so re-running it is safe. With no checkpointer the graph has no `review` node and behaves exactly as before (plus any `data_quality` finding in the results), which is why the offline vertical slice and its audit-stage check don't include `review`.
- **Failed reconciliation routes to human data review.** The Ledger Analyst copies, from its tool observations only, any `ingest_statement` result that is `unreconciled` or `unverified`, and any `query_ledger`/`run_calculation` error saying a document or transaction is unreconciled, into `outputs["data_quality"]` (document ID, source name, status, issue codes). The threshold check turns each entry into a `data_quality` finding, so with a checkpointer the job stops at the same `review` interrupt, and the briefing is marked `needs_review` either way. The briefing gets an `alerts` entry saying the document has a problem and that no figures were produced from it, and drops any statement whose evidence cites a flagged document (the ledger already refuses unreconciled rows, so a model claim citing them is removed earlier). **Not built:** ingestion still happens before or outside the graph unless the model calls the tool, there is no automatic re-ingest or repair, reconciliation rules are unchanged, and accepting the review does not make the document usable.
  - **Not built yet (concurrency):** the router prompt does not yet tell the model about `depends_on_ledger` (only hand-built or future plans set it), a stuck specialist's thread is abandoned rather than cancelled, there is no per-agent timeout apart from the job limit, and specialists' own tool-call audit events are not ordered within a concurrent batch.
  - **Not built yet:** pending reviews don't expire (the design's 7-day default), consent is re-checked for the reviewer only, not re-run for the original requester, tool calls have no idempotency keys (nothing re-executes today because finished nodes aren't re-run), the decision doesn't change the briefing's content beyond recording it, and the vertical slice still runs the Supervisor without a checkpointer.
- **Draft reminders (`reminders.py`, Supervisor-only).** `draft_reminder` is built (`reminders.py`, Supervisor-only): a "remind me" / "add ... to my calendar" request becomes a draft from the job's own evidence, listed under Actions for Approval. With a checkpointer the job pauses until an authorized adult or admin approves; approval writes a local RFC 5545 `.ics` file and nothing else. Real calendar integration is NOT built (no calendar connection, no sending, no recurring events, no notification agent).
  - **Flow.** `threshold`/`review` ─► `remind` (drafts and audits once) ─► `reminder_approval` (`interrupt()`; nothing with side effects runs before it, because LangGraph re-runs the node on resume) ─► `reminder_create` (writes the `.ics`) ─► `compose`. The two approval nodes exist only with a checkpointer; without one the draft is listed and nothing else happens.
  - **Rules.** The due date cannot be before the job's date; the title is at most 100 characters with no control characters or line breaks; at least one source evidence ID is required and every ID must be in the job's results. `reminder_id` is a hash of the content, so the same request gives the same ID and the same file name.
  - **Approval.** Same pattern as the review: only an adult or admin of the household who is authorized on the accounts the job used can approve. Anyone else is refused, audited as a deny, and the reminder stays a draft. Each state change (`draft_reminder`, `approve`, `create_reminder`) is audited under the job's correlation ID.
  - **The `.ics`.** Plain text built in code: CRLF line endings, text escaped, lines folded at 75 octets, UID from the reminder ID, one all-day event. It is written to `Supervisor.reminder_dir`; with no directory the reminder stays `approved` and is audited as not exported.
  - **Not built:** any real calendar (Google, Apple, Outlook), sending or emailing the file, recurring events, a notification agent, editing a draft, reminder expiry. The deadline comes from a date typed in the request; it is not yet read from a document.
- **Model binding.** `bind_tools` is used when the model supports it; otherwise the plain model is used (fake models in tests).

## 4f. Monthly briefing (`briefing.py`)

```
readable accounts (authorize READ for monthly_briefing; others → "N not shared" count only)
   │
   ├─ document_coverage (metadata, any status) ─► Data Coverage, missing/unreconciled limitations,
   │                                              confirm_statement actions, possible_duplicate events
   ├─ verified_statements ─► closing balances
   ├─ query_transactions (verified) ─► cash_flow + category_totals runs (recorded, acting_agent="briefing")
   ├─ query_transactions (+provisional) ─► provisional count, material events (unmapped ≥ $500, non-routine ≥ $2,500)
   ├─ + previous 90 days as baseline ─► cluster_risk run ─► material events small_cluster_7d / burst_24h / small_then_large
   ├─ AgentResults / findings ─► validated claims, questions, material_variance
   └─ plan_from_ledger result ─► best scenario, runner-up (different allocation), assumptions, approval pending
                                    │
                                    ▼
             MonthlyBriefing(draft=True, external_action="none", …, disclaimer last) ─► audit compose_briefing
```

No model writes any section. A `BriefingItem` with `amounts_minor` must cite evidence, and unreconciled statements contribute coverage notes but never numbers. Category totals always sum to the net cash flow, because both come from the same verified transaction set; a hypothesis test checks this.

Cluster alerts cite the `cluster_risk` run ID and every transaction in the pattern, are marked provisional when any of those rows is unverified, and are worded as a review prompt. Only new merchants (absent from the baseline) with no approved category count, so a familiar subscription never alerts.

## 4g. Evaluation harness (`evals.py`)

`generate_scenarios(seed)` is deterministic for a given seed. `evaluate(scenarios, router=, generator=, critic=, authorizer_factory=, embeddings=)` runs each scenario through the real component:

| Area | Count | Runs through | Gate |
|---|---|---|---|
| routing | 55 | `router(request)` | accuracy ≥ 0.95; schema validity ≥ 0.99 (`rules_fallback` = invalid); allocation requests reach the approval route (100%) |
| adversarial | 14 | `Supervisor.run` with the router's decision | every request `refused` (0 unsafe actions) |
| authorization | 72 | `Authorizer.authorize` (READ), with/without two-owner CPA consent | 0 cross-member leaks (allowed but labeled deny; retrieval leaks add to this) |
| allocation | 60 | `plan_allocation` | exact surplus, caps, no negatives, ≤ 27 evaluations, approval-gated, reproducible hash and tree |
| ingestion | 30 | `ingest_statement` into a fresh in-memory store | 100% match of `reconciled` / `unreconciled` |
| calculation | 20 | `ingest_statement` → `draft_monthly_briefing` | cash flow, category totals and closing balance equal an oracle built from the rows and its own category map; no other `$` amount in the Markdown (100%) |
| anomaly | 40 | September baseline + October statement → `draft_monthly_briefing` | the expected alert kind appears in material events: recall ≥ 0.90; false alerts on the 3 control types and per-transaction-only recall are reported |
| retrieval | 22 | `RetrievalService.retrieve_verified` over 13 sources (each indexed by its owner), then `validate_claims` | document recall@5 ≥ 0.90 over relevant docs the asker may see; leaks judged by an ownership oracle; citation precision ≥ 0.95 (claims tested with the right source, a wrong source and a changed digit) |
| privacy | 8 | briefing, retrieval, `ToolSession` and a recorded `Supervisor` job per observer | 0 canary exposures for 6 non-owner cases; both owners see their canary on all 4 surfaces |

Retrieval and privacy build their worlds (temp Chroma index, ledger, recorded playback models) once per run and close them afterwards. Membership inference is the only design §2 metric still listed as not covered. `python -m household_agent.evals` exits 1 if any gate fails. `--model`, `--planner-model` and `--embed-model` switch in real models: Ollama on the model machine, or `gemini:…` / `groq:…` models from any laptop (`gemini:gemini-embedding-001` for embeddings). Offline results: all 13 gates pass. The two misses are `route_054` (a keyword false positive) and `retr_005` (hash embeddings rank the water-heater invoice below 5).

## 4h. Vertical-slice exit run (`vertical_slice.py`)

`run_vertical_slice` uses file-backed stores in `workdir` (`ledger.db`, `checkpoints.db`, `chroma/`) so the restart step is real. Each check maps to a §12 exit criterion:

| Check | What it does | Passes when |
|---|---|---|
| `ingest_and_reconcile` | Each owner uploads their statements | 3 reconciled, Jordan's card qualified, September unreconciled |
| `three_agent_trace` | `Supervisor.run` on the §9 question | Both specialists ran, a calc run exists, audit shows route → 2 dispatches → compose |
| `provenance_chain` | Follows `calculation_runs.input_ids_json` → `transactions.source_document_id` → `documents.source_hash` | Every link exists, and `ingest_document` and `calculate` events share the job's correlation ID |
| `authorization_denial` | Jordan reads Alex's checking; Sam indexes a document | Both refused, the deny is audited |
| `consent_revocation_denial` | Both owners grant the CPA rental access via `ConsentStore`, then Jordan revokes | The read succeeds before and is refused after |
| `unreconciled_block` | Forces September's raw rows into `cash_flow` | `IneligibleInputError`; the September briefing has no figures |
| `stale_source_escalation` | `retrieve_verified` + `validate_claims` | Expired bill and wrong-year IRS source → `rephrase_as_question` / `CANNOT_VERIFY` |
| `planning_reproducible` | `plan_from_ledger`, re-plan, re-derive | Same input hash and tree, 15 evaluations, provisional (Jordan's card excluded) |
| `cap_breach_hard_pruned` | `OverCapPersona` puts everything in the IRA | Branches `hard_pruned` with `statutory_cap_exceeded`; a compliant best plan remains |
| `critic_discrepancy_review` | `CashLovingCritic` disagrees | `needs_review` with discrepancy node IDs; no branch removed |
| `draft_briefing` | `draft_monthly_briefing` with results, findings, plan and thread ID | Draft only, disclaimer last, amounts cite evidence, no private card |
| `interrupt_and_resume` | Supervisor allocation request → close and reopen both databases → Sam, then Jordan | Paused at `approval`; thread audit is `plan_allocation`, `approve`(deny), `approve`(allow) |
| `no_external_actions` | Transfer request; inspect the tool names | `refused`; no tool name matches pay/transfer/wire |
| `audit_trail` | `AuditLog.verify()` | Chain intact and every stage appears under the job's correlation ID |

`models(role)` selects the chat model for `router`, `ledger_analyst` and `policy_compliance`. If no factory is given, the specialists use `PlaybackChatModel` scripts. These read real IDs from earlier tool results, so the scripts never hard-code an ID, and the router uses keyword rules. A model that misbehaves makes its own checks fail, while the guardrail checks keep passing; tests cover both cases.

## 5. Storage schema (SQLite)

All tables are **append-only**: `BEFORE UPDATE` and `BEFORE DELETE` triggers raise an error.

| Table | Key | Notes |
|---|---|---|
| `documents` | `document_id` | `source_hash` UNIQUE, status, confidence, issues JSON |
| `statement_balances` | `document_id` | Account, period, opening/closing balance in cents, written at ingest in the same transaction |
| `raw_rows` | (`document_id`, `line_number`) | Exact source text for provenance |
| `transactions` | (`transaction_id`, `version`) | Corrections insert `version + 1` with `correction_reason` |
| `calculation_runs` | `run_id` | fn, params, input IDs, hashes, library version, provisional |
| `audit_events` | `seq` | `prev_hash` → `hash` chain from a genesis of `"0"*64` |
| `consent_grants` | `grant_id` | Grantor, grantee, accounts/actions/purposes JSON, effective/expiry, version |
| `consent_revocations` | `grant_id` | A revocation is a new row (`revoked_by`, `revoked_at`), never an edit |

`LedgerStore.document_coverage` reads `documents` metadata only (status, confidence, issues), never balances or rows. This lets a briefing report unreconciled or qualified statements without using their numbers.

LangGraph checkpoints are kept in a separate database (`checkpoints.db`) managed by `SqliteSaver`. The vector index lives in a ChromaDB directory. Chunk metadata includes `household_id`, `account_id` (or `PUBLIC`), `authority_tier`, `effective_tax_year` (0 = n/a), `valid_until`, `chunk_hash`, `source_hash` and `injection_suspect`.

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
| LLM invents allocation amounts | Thoughts are strategies; cents computed by `expand_step`; model thoughts with amount fields or currency in the rationale are rejected | `allocation.py`, `llm_planner.py` |
| Model persona output is malformed | Per-persona template fallback, recorded; rule-breaking thoughts hard-pruned, not repaired | `llm_planner.py`, `planning.py` |
| Plan breaks caps or reserve | Hard prune before scoring; leaves must allocate exactly the surplus | `allocation.py`, `planning.py` |
| Critic prunes a good branch | Critic is advisory; disagreement > 0.20 retains the branch for review | `planning.py` |
| Plan treated as an action | `requires_approval=True`; there is no execution path | `planning.py`, `approval.py` |
| Wrong person approves | Only admin/adult members of the same household; denials audited, graph keeps waiting | `approval.py` |
| Approving numbers nobody saw | Input hash recomputed on resume; any change → `stale` | `approval.py` |
| Consent forged or silently changed | Grantor-only grants for owned accounts; append-only tables; audited | `consent.py` |
| Approval lost on restart | `SqliteSaver` checkpoint; resume by thread ID | `approval.py` |
| Model impersonates another member | Actor/purpose bound in `ToolSession`/server env; not in tool schemas | `tools.py`, `mcp_server.py` |
| Model invents calculation inputs | `run_calculation` takes only IDs this session queried | `tools.py` |
| Runaway tool loops | Per-session `max_calls` budget → `budget_exhausted` | `tools.py` |
| Retrieval leaks another member's document | Authorizer-derived `account_id ∈ scopes` filter applied before search | `retrieval.py` |
| Stale or wrong-year tax rule cited | Expiry/tax-year verification; tax claims need a Tier-1 source for the same year | `retrieval.py` |
| Model invents numbers in a briefing | `validate_claims` removes claims whose numbers aren't in the cited evidence | `retrieval.py` |
| Prompt injection in documents | Instruction-like chunks flagged at indexing and audited at retrieval; text treated as data | `retrieval.py` |
| Index tampering | Per-chunk hash re-checked at retrieval → `hash_mismatch` | `retrieval.py` |
| Model asserts unsupported numbers | Specialist claims re-validated; outputs copied from tools | `agents.py` |
| A result from another job is mixed in | Correlation ID on both handoff payloads; a mismatched result is discarded as `failed` | `agents.py` |
| A flagged job is accepted by the wrong person | `review` interrupt: admin/adult of the household with `resume` access to every account the job used; denials audited, job keeps waiting | `agents.py` |
| Model routes to an unknown agent or plans a payment | Pydantic `Literal` allowlist, ≤ 8 dispatches, external-action flag can't be cleared by the model | `agents.py` |
| Runaway agent loops | Per-assignment tool/retry/refinement budgets, job dispatch and time limits, LangGraph recursion limit | `agents.py` |
| Briefing shows unverified numbers | Totals and balances from verified data only; provisional items labeled and excluded; amounts must cite evidence | `briefing.py` |
| Briefing leaks another member's account | Only accounts readable for `monthly_briefing`; the rest appear as a count | `briefing.py` |
| Regressions go unnoticed | Seeded scenario set; 13 gates for routing, unsafe actions, leaks, allocation invariants, integrity, calculations, anomalies, retrieval, citations and canaries; CLI exit code | `evals.py`, `eval_quality.py` |
| Fraud split into many small charges | `cluster_risk` rules over new, unmapped merchants with a 90-day baseline | `risk.py`, `briefing.py` |
| A model's citation points at the wrong source | Word-overlap check in `validate_claims`; citation-precision gate | `retrieval.py` |
| Tests depend on a local model | Models chosen only in `llm.py`; tests use fakes and `HashEmbeddings` | `llm.py`, `tests/` |

## 8. Phase 2 plan (design §4, §12 steps 6–10)

Done so far: model configuration (`llm.py`), persistent consent (`consent.py`), the durable approval interrupt (`approval.py`), the shared tool layer (`tools.py`), the stdio MCP server (`mcp_server.py`) verified retrieval (`retrieval.py`), the Supervisor with both specialists (`agents.py`), ledger-derived planner inputs with recorded plan runs (`plan_inputs.py`), the model-driven personas and critic (`llm_planner.py`), the draft monthly briefing (`briefing.py`), the eval harness with all design §2 gates except membership inference (`evals.py`, `eval_quality.py`), cluster-risk rules (`household_finance.risk`) and the §12 vertical-slice exit run (`vertical_slice.py`). Real-model runs use a hosted API (Gemini or Groq, synthetic data only) or a separate machine with Ollama. Everything here must stay testable without either.

- **Agents.** A LangGraph supervisor graph with two specialists.
  - Ledger Analyst: `parse_statement`, `reconcile_statement`, `query_ledger`, `run_calculation`, `assess_risk`.
  - Policy & Compliance: `retrieve_verified`, `validate_claims`.
  - The supervisor itself owns `draft_reminder` and `compose_briefing`.
- **Handoffs.** Typed `task_assignment` / `agent_result` handoffs.
- **Budgets.**
  - Supervisor: 8 dispatches per job and a 5-minute job limit.
  - Specialists: 6 tool calls each, with 2 retries per tool.
- **Durability.** A SQLite checkpointer, plus `interrupt()` for human approval. **Done** for allocation approval (`approval.py`) and for Supervisor review of findings and incomplete results (`Supervisor.run(..., checkpointer=...)`, `Supervisor.resume`).
- **Retrieval.** A ChromaDB index of verified documents, using 500/100 chunking and `nomic-embed-text`. Every retrieval is filtered by the authorizer.
- **MCP.** A `household-finance-tools` stdio MCP server exposes the Phase 1 functions, consumed through `langchain-mcp-adapters`.
- **Models.** Local Ollama models: `qwen2.5:7b` as the baseline, with `qwen3:8b` for comparison. Hosted models for evals on synthetic data when no local model is available: Gemini (`gemini-2.5-flash`, `gemini-embedding-001`) or Groq (`qwen/qwen3-32b`).
- **Planner (§5.8).**
  - Persona prompts emit `pydantic`-validated `AllocationStep`s, with a model critic. **Done** (`llm_planner.py`; one batched call per beam node). CrewAI is not used.
  - Evaluate each depth in one batched `evaluate_allocation_candidates` call.
  - Fetch contribution limits once per job via `retrieve_verified`.
  - Derive `PlanningInputs` from reconciled ledger balances. **Done** (`plan_inputs.py`).
  - Record each run in `calculation_runs`, then `interrupt()` for approval. **Done** (`record_plan_run`, `ledger_inputs_for`).
  - Target p95 ≤ 120 s on the local 7B model, at most 5 generation calls.
