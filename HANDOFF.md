# Handoff: household wealth agent demo

Written at the end of a session so the next one can pick up cold. Supersedes the earlier HANDOFF.md.

## Where things stand

- `main` has everything up to #17 plus the failure-notes / `cash_flow exclude_categories` work (branch `handoff-notes-2` carries the last handoff commit).
- **Branch `history-seed` (this work, committed locally, NOT pushed, no PR yet)** adds two years of fictional history. Tests: **753 passing** (`PYTHONPATH=src .venv/bin/python -m pytest -q`; the config prints dots only, count with `--collect-only -q`).
- Verified live this session (Gemma `gemini:gemma-4-26b-a4b-it`, sayali, card statement uploaded): "what is my inflow and outflow last month" passes `exclude_categories: ["transfer"]` on its own. Result: inflow $40.15 (a real `RETURN CLINIC 562` credit), outflow $3,878.51, net -$3,838.36. The $2,206.97 card payments are excluded. No fix needed.

## What `history-seed` does

- `src/household_agent/history.py`: `history_statements()` (70 statements, Oct 2024 to Sep 2026, deterministic per seed, each account chains from the previous closing balance), `brokerage_summaries()` (8 quarterly fictional summaries, Somal-private), `statement_source()` (a parsed statement as a searchable document).
- `webapp.py`: `DemoService(history=True)` seeds the history on every start (ledger + `financial_history` vector collection). Every reconciled upload is also indexed; a statement that does not balance is not. History files are `hist_YYYY_MM_<account>.csv` and are not listed in the UI. `history=False` turns it off (tests).
- `synthetic.py`: new category rule `ACME BROKERAGE` -> `investment_transfer`.
- Seeded: 71 reconciled statements, 1,127 transactions, 103 vector chunks, 0.5 s with the default hash embeddings. A reused `--workdir` is not re-seeded, so changing `--embed-model` needs a fresh workdir.
- Docs updated: README Testing table (753, `test_history.py` row), ARCHITECTURE section 4j, VIDEO.md count.

## What the vector database holds (answered this session)

- Before: only the 9 chunks of policy documents (`policies_compliance`). Past statements were in SQLite only; `financial_history` was never filled by the app.
- Embeddings: default `hash` (256-d bag of words, not semantic). `gemini:gemini-embedding-001` works on the key (3072-d) and ranked the right lease section first for a question with little word overlap; hash did not.
- Probe scripts lived in `$CLAUDE_JOB_DIR/tmp` (gone with the job): index the policy docs, print raw cosine ranking and `retrieve_verified` per member.

## First thing next session

1. **Live Gemma test of the history (not yet seen).** The only attempt hit Gemini 503 (capacity) after the analyst read accounts and ran one calculation. Drive `DemoService(Path(tmp), model_factory=lambda role: model, embeddings=get_embeddings("gemini:gemini-embedding-001"))` with `model = get_chat_model("gemini:gemma-4-26b-a4b-it")` as somal, then `svc.chat("somal", q)`; print statements, questions, `agents[].errors` and the trace. Questions to try: "Compare my spending in October 2025 with October 2024", "Which past months looked like last month?", "Was it good to invest in stocks over the last two years?". If Gemini returns 429 or 503, tell the user and stop; do not guess.
2. **Real-data request (not started).** The user gave two real Chase card exports and asked to use them "to generate data to have reality data": `~/Downloads/Chase0396_Activity_20261116.csv` (178 rows, Nov 2026) and `~/Downloads/Chase0396_Activity_20261116 (1).csv` (312 rows, back to Oct 2025). Columns: `Transaction Date, Post Date, Description, Category, Type, Amount, Memo`; types `Sale`, `Payment`; amounts negative for spending; HTML-escaped text (`&amp;`). It is not our `hh_statement_csv_v1` format. Plan: read the files only to learn the shape (merchant mix, category shares, amount ranges, rows per month, payment sizes, return frequency), then extend the generator (`synthetic._CARD_SPEND`, `history_statements`) with fictional merchants and similar distributions. **Do not copy real merchants, descriptions or amounts into the repo, tests or docs; keep the Chase files out of git.** Ask before building a converter for the real format (an earlier decision was a generator plus fictional data).
3. Open issues found, not fixed (propose before changing):
   - "Was investing good?" routes to `allocation_planning` (separate planner), so history search and the brokerage summaries are never used for it.
   - The ledger analyst's small tool budget and `run_calculation` taking explicit transaction IDs make 24-month year-over-year questions awkward; a monthly-totals tool would help.
   - A plan can be built from September balances while October is missing (no freshness note).
4. Push `history-seed` and open a draft PR (see Pushing below). Skip pr-auto-review.

## Ideas not done (ask before doing)

- Fewer model calls per question; offering a spending-total calculation so Gemma's own arithmetic can cite a run; a bulk embedding call for seeding with a hosted embedding model (about 80 calls today); the UI test-idea list in `VIDEO.md`.

## Conventions to keep

- Every change: pytest tests (happy, negative, property), README Testing table and total, ARCHITECTURE.md, and the VIDEO.md test count.
- Maintain only `agentic_ai_course_design.md` as the design source; the capstone doc is reference.
- No local LLM on this laptop (tests use fakes); Gemma runs via the Gemini API or on the other laptop. Live checks are manual scripts, not pytest.
- Never push to `main`.
- Worktree: `/Users/somal.vora/Desktop/Agentic AI Course/project/.claude/worktrees/tool-hints`. The shell guard refuses compound commands, heredocs with computed values, `git -c` with computed values, and a bare `hash` argument; use plain separate commands, script files and the Read/Edit/Write tools.
- **Pushing:** the active `gh` account is `somal-vora_sfemu`, which cannot push to `somalvora/wealth_mgmt` (403). From a separate terminal: `setopt NO_BANG_HIST`, then `git -c credential.helper= -c 'credential.helper=!f() { echo username=somalvora; echo password=$(gh auth token --user somalvora); }; f' push -u origin history-seed`. Opening PRs also needs the `somalvora` account.

## Resume prompt

> Read HANDOFF.md in the tool-hints worktree (branch `history-seed`). First, retry the live Gemma test of the two-year history described under "First thing next session" item 1 (stop and tell me on any 429/503). Then, for the real Chase exports, learn only their shape and extend the fictional generator to match; never copy real rows into the repo. Propose design and tests before implementing the open issues.
