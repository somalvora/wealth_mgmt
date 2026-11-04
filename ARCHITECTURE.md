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
| `household_agent.ingest` | `parse_statement_csv` with line provenance | 3.1, 6 | `parse_statement` |
| `household_agent.categorize` | Rule-based merchant normalization | 6 | (part of ingest) |
| `household_agent.reconcile` | Checks, status, `document_confidence` | 3.3 | `reconcile_statement` |
| `household_agent.ledger` | Append-only `LedgerStore` | 6 | `query_ledger` |
| `household_agent.calc_runner` | Allowlist, eligibility, provenance | 3.2, 3.4 | `run_calculation` |
| `household_agent.pipeline` | Orchestrates ingest end to end | 12 | — |
| `household_finance.allocation` | Strategy → cents expansion, hard constraints, composite score, rollout | 3.4, 5.8 | `evaluate_allocation_candidates` |
| `household_agent.planning` | Tree-of-Thought beam search, `ThoughtGenerator` and `Critic` protocols, template personas | 5.8 | (Supervisor planning subgraph) |
| `household_agent.llm` | The only place models are named: `get_chat_model()` / `get_embeddings()` from `HH_CHAT_MODEL` / `HH_EMBED_MODEL`; offline `HashEmbeddings` | 3.5 | — |
| `household_agent.consent` | Persistent `ConsentStore`: append-only grants and revocations, grantor-only, audited, reloaded into `Authorizer` | 7 | (consent management) |
| `household_agent.tools` | `ToolSession`: identity/purpose bound by the host, call budget, structured errors, session-scoped transaction IDs | 3.2, 4.1 | all Ledger Analyst tools |
| `household_agent.mcp_server` | `household-finance-tools` FastMCP stdio server over `ToolSession`; identity from env | 3.2, 4.2 | (MCP transport) |
| `household_agent.agents` | `Supervisor` graph (classify → dispatch loop → threshold → compose), `LedgerAnalyst` and `PolicyCompliance` ReAct subgraphs, `Router`, typed `TaskAssignment`/`AgentResult` | 4, 5.1 | (the agents) |
| `household_agent.retrieval` | `RetrievalService` over ChromaDB: authorize-first `where` filters, MMR, per-passage verification, audit; `validate_claims` | 8 | `retrieve_verified`, `validate_claims` |
| `household_agent.approval` | LangGraph `plan → approval(interrupt)` graph with a checkpointer; approver check and stale-input check | 5.8, 10 | (Supervisor approval subgraph) |
| `household_agent.llm_planner` | `LLMThoughtGenerator` (model personas, pydantic-validated, template fallback) and `LLMCritic` (clamped, advisory) behind the planner protocols | 5.8 | (planner personas / critic) |
| `household_agent.plan_inputs` | `derive_planning_inputs` from verified statements + recorded `category_totals`; `record_plan_run` / `plan_from_ledger`; `ledger_inputs_for` for the approval graph | 3.4, 5.8 | (planner inputs) |
| `household_agent.briefing` | `draft_monthly_briefing` (seven §10 sections, deterministic, evidence-required `BriefingItem`s) and `render_markdown` | 10 | `compose_briefing` |
| `household_agent.evals` | `generate_scenarios(seed)` (231 seeded scenarios, 5 areas), `evaluate` → `EvalReport` with §2 gates, CLI | 2, 11 | — |
| `household_agent.vertical_slice` | `run_vertical_slice(workdir, models=, embeddings=)` → `SliceReport` of 14 `Check`s; `PlaybackChatModel` offline stand-in; CLI | 12 | — |

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

## 4e. Supervisor and specialists (`agents.py`)

```
request ─► classify (Router: model JSON → RoutingDecision, else rule_route)
             ├─ external action ─► refuse (audited deny) ─────────────┐
             ├─ allocation ─► plan: start approval graph (interrupt) ─┤
             └─ dispatch loop (≤ 8, ≤ 300 s) ─► threshold ─► compose ◄┘
                  │ TaskAssignment(purpose from intent, scope from host)
                  ▼
        specialist subgraph: agent ⇄ tools (≤ 6 calls, ≤ 2 retries/tool)
                  │ outputs/evidence copied from observations
                  │ claims re-checked by validate_claims
                  ▼
        AgentResult(status, outputs, evidence_ids, claims, questions, errors)
```

- **Ledger Analyst.** Its tools are `list_accounts`, `query_ledger` and `run_calculation`, through a `ToolSession`. Claims may cite only the transaction IDs and calculation run IDs from this assignment, and `*_minor` values are also matched in dollar form ("$450.00").
- **Policy & Compliance.** Its tool is `retrieve_verified`, allowing 1 search plus 2 refinements. Claims are validated against the passages from this assignment and against calculation outputs the Supervisor passed in as run IDs. `rephrase_as_question` verdicts become questions; `remove` verdicts are dropped and recorded as errors.
- **Supervisor.**
  - It never calls the ledger or the vector store.
  - A specialist that raises an exception becomes a `failed` result.
  - Any status other than `completed`, or any threshold finding, makes the job `needs_review`.
  - Each step is audited under the job's correlation ID: `route`, `dispatch`, `threshold_check` and `compose_briefing`.
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
   ├─ AgentResults / findings ─► validated claims, questions, material_variance
   └─ plan_from_ledger result ─► best scenario, runner-up (different allocation), assumptions, approval pending
                                    │
                                    ▼
             MonthlyBriefing(draft=True, external_action="none", …, disclaimer last) ─► audit compose_briefing
```

No model writes any section. A `BriefingItem` with `amounts_minor` must cite evidence, and unreconciled statements contribute coverage notes but never numbers. Category totals always sum to the net cash flow, because both come from the same verified transaction set; a hypothesis test checks this.

## 4g. Evaluation harness (`evals.py`)

`generate_scenarios(seed)` is deterministic for a given seed. `evaluate(scenarios, router=, generator=, critic=, authorizer_factory=)` runs each scenario through the real component:

| Area | Count | Runs through | Gate |
|---|---|---|---|
| routing | 55 | `router(request)` | accuracy ≥ 0.95; schema validity ≥ 0.99 (`rules_fallback` = invalid); allocation requests reach the approval route (100%) |
| adversarial | 14 | `Supervisor.run` with the router's decision | every request `refused` (0 unsafe actions) |
| authorization | 72 | `Authorizer.authorize` (READ), with/without two-owner CPA consent | 0 cross-member leaks (allowed but labeled deny) |
| allocation | 60 | `plan_allocation` | exact surplus, caps, no negatives, ≤ 27 evaluations, approval-gated, reproducible hash and tree |
| ingestion | 30 | `ingest_statement` into a fresh in-memory store | 100% match of `reconciled` / `unreconciled` |

The report lists the metrics that are not covered yet: retrieval recall@5, citation precision, calculation match, anomaly recall and privacy memorization. `python -m household_agent.evals` exits 1 if any gate fails. `--model` and `--planner-model` switch in real models on the Ollama machine.

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
| Model routes to an unknown agent or plans a payment | Pydantic `Literal` allowlist, ≤ 8 dispatches, external-action flag can't be cleared by the model | `agents.py` |
| Runaway agent loops | Per-assignment tool/retry/refinement budgets, job dispatch and time limits, LangGraph recursion limit | `agents.py` |
| Briefing shows unverified numbers | Totals and balances from verified data only; provisional items labeled and excluded; amounts must cite evidence | `briefing.py` |
| Briefing leaks another member's account | Only accounts readable for `monthly_briefing`; the rest appear as a count | `briefing.py` |
| Regressions go unnoticed | Seeded scenario set; gates for routing, unsafe actions, leaks, allocation invariants and integrity; CLI exit code | `evals.py` |
| Tests depend on a local model | Models chosen only in `llm.py`; tests use fakes and `HashEmbeddings` | `llm.py`, `tests/` |

## 8. Phase 2 plan (design §4, §12 steps 6–10)

Done so far: model configuration (`llm.py`), persistent consent (`consent.py`), the durable approval interrupt (`approval.py`), the shared tool layer (`tools.py`), the stdio MCP server (`mcp_server.py`) verified retrieval (`retrieval.py`), the Supervisor with both specialists (`agents.py`), ledger-derived planner inputs with recorded plan runs (`plan_inputs.py`), the model-driven personas and critic (`llm_planner.py`), the draft monthly briefing (`briefing.py`), the eval harness skeleton (`evals.py`) and the §12 vertical-slice exit run (`vertical_slice.py`). Real-model runs and evals happen on a separate machine with Ollama. Everything here must stay testable without it.

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
- **Planner (§5.8).**
  - Persona prompts emit `pydantic`-validated `AllocationStep`s, with a model critic. **Done** (`llm_planner.py`; one batched call per beam node). CrewAI is not used.
  - Evaluate each depth in one batched `evaluate_allocation_candidates` call.
  - Fetch contribution limits once per job via `retrieve_verified`.
  - Derive `PlanningInputs` from reconciled ledger balances. **Done** (`plan_inputs.py`).
  - Record each run in `calculation_runs`, then `interrupt()` for approval. **Done** (`record_plan_run`, `ledger_inputs_for`).
  - Target p95 ≤ 120 s on the local 7B model, at most 5 generation calls.
