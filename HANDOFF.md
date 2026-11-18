# Handoff: demo UI additions and evals (Household Wealth Agent)

Updated 2026-11-18 (night). Worktree: `/Users/somal.vora/Desktop/Agentic AI Course/project/.claude/worktrees/demo-ui-additions`.
Repo `somalvora/wealth_mgmt` (PUBLIC: synthetic data only, never real balances, account numbers or PDFs).
**Merged into `main` (head 4a702fb):** #20, #21, #22, #23 (brokerage chips), #25 (real tickers, document month labels, retrieve rows
show the search text), #26 (repeat searches reuse the first result, brokerage answer leads with the result), #27 (eval gate
`policy_search_count`), #28 (router fix for route_054). Suite passes on main after both merges.
**Open:** #29 (this handoff, docs only, draft). Branch `eval-brokerage` was created off `origin/main` with no changes yet.
Titles use the placeholder `@W-XXXXXXXX : ...` (no work item exists, owner said "No WI"). Suite passes.
GitHub: the active `gh` account was the EMU one; I switched to `somalvora` only for push/PR and switched back. Plain `git push`
cannot authenticate in this shell; use `git -c "credential.helper=!gh auth git-credential" push -u origin <branch>` after
`gh auth switch --hostname github.com --user somalvora`, then switch back to `somal-vora_sfemu`.

## Goal
Make the web demo (`uv run python -m household_agent.webapp ...`, page at 127.0.0.1:8000) show most of the system from the UI,
so a video can be recorded by clicking only the suggestion chips. No upload step should be needed. Second thread: evals.

## What is on main
- **Panels** (`static/demo.html`, `webapp.py`): Share my accounts (consent, `/api/consent*`), live Audit trail (`/api/audit`),
  Evaluation results (`/api/eval`, `/api/eval/run`, offline keyword router, about 2 s). The Documents panel shows each statement
  with its month and "(latest)" on the newest of a series (`Oct 2025`, `Oct 2026 (latest)`). Trace and audit `retrieve` rows show
  the search text (`webapp.py` `_resource`; display only, the audit chain is unchanged).
- **Preloaded data:** 2 years of statements (Oct 2024 to Sep 2026) for all three accounts, plus the October 2026 joint rental
  statement, so the tax chip needs no upload. Somal's Sep 2026 checking is intentionally empty (reserved for the tampered/injection
  upload samples). Oct 2026 checking and Sayali's card (`2026_10_sayali_card.pdf`) are upload samples only.
- **Chips** (10): tax (Oct $3,200 county payment), allocate surplus, transfer refusal, groceries 2025 (expect $3,269.43), lease
  (Unit B rent $3,400 and property tax), viaan asks about Somal's Safeway spend (refused), somal asks about Sayali's card (refused until
  Sayali shares), weather (out of scope), brokerage performance, brokerage concentration.
- **Refusals** (`DemoService._private_account_refusal`): a question naming a member whose PRIVATE account the asker cannot read returns
  `denied` before any model runs; skipped for the joint rental account, money movement and out of scope. Runs under `self.lock`.
- **Brokerage** (`brokerage.py`, all fictional quantities and amounts; tickers are real widely held symbols NVDA, MSFT, AAPL, AMD,
  JNJ, XOM, JPM, VOO, VXUS, SCHD, BND chosen so a viewer can look them up, not taken from anyone's statements). Oct 2026: opens
  31,334.23, deposit 1,500.00, dividends 31.40, closes 31,045.73, market effect -1,819.90 (-5.8%). Concentration: largest NVDA 23.1%
  (over the 20% single-stock limit), top 4 = 54.9%, technology 46.2% (under 50%; September was over). `LIMITS_BPS` is the single
  source for the flags and the generated policy document. `BROKERAGE_ACCOUNT` is in `DEMO_ACCOUNTS` only, not `ACCOUNTS`.
- **Policy agent** (`agents.py` `PolicyCompliance`): an exact repeat of a search (case and spacing ignored) returns the first result,
  no new audit row, no budget use (`MAX_QUERY_REFINEMENTS = 2`). Ledger prompt: lead brokerage answers with the result, say "your
  brokerage account", never the account id.

## Evals (what exists)
- Custom, offline, deterministic harness: `python -m household_agent.evals` (`evals.py`, `eval_quality.py`), seeded synthetic
  scenarios, pass/fail gates with fixed targets and oracles independent of the code under test. No LangSmith, Langfuse or MLflow
  (`langsmith` is only a transitive dependency of `langchain-core`; no tracing env vars). The owner decided to skip LangSmith.
- On main today (#27 and #28 merged): **14 gates, 348 scenarios** (area `search_count`, 27 scenarios, gate `policy_search_count`,
  EVAL_VERSION `evals_0.3`), routing_accuracy 1.0 (keyword router), `unsafe_actions` 0, all 14 adversarial scenarios refuse,
  retrieval_recall_at_5 0.9444 (`retr_005` water_heater_invoice misses on the offline hash embeddings; expected).
- **Gaps:** (1) no eval scenarios for the brokerage tools; (2) the harness has only ever run on the offline keyword router, never a
  live model, so 0.98/1.0 is the keyword baseline, not Gemma; (3) ledger/policy answer quality is only tested with scripted models;
  (4) membership inference is listed as not covered; (5) old ideas: more adversarial scenarios, indirect injection, citation cases,
  non-regression baseline.

## Verified vs not
- Verified: full suite passes (`uv run python -m pytest tests -q`), offline eval passes all gates, groceries 2025 = $3,269.43, both
  refusals offline, the live run (screenshot 2026-11-18) showed the month labels, retrieve query text, real tickers, and the NVDA
  23.1% concentration answer with its policy citation.
- **Not verified on live Gemini after #26:** the brokerage performance answer shape (should lead with 31,334.23 to 31,045.73, -5.8%, and
  not print `acct_somal_brokerage`), a single "household concentration limits" row in the trace, and the tax, groceries, lease and
  allocation chips. The owner rehearses these; do not spend model quota without asking.
- Gemini free tier is about 20 requests/day per model; 429 and 503 show as an amber message. Answers are cached in `.cache/llm.sqlite`
  (only successful chat answers), so rehearse each chip once, then record. A prompt edit changes cache keys.
- Browser click-through still owed: the viaan chip should make the Share panel say "viaan owns no accounts".

## Standing instructions (this window)
- Do NOT dispatch a code-review agent on PRs opened in this window. This overrides the "review every new PR" default and the
  PostToolUse hook nudge to run pr-auto-review after `git push` or `gh pr create`.
- PR titles use `@W-XXXXXXXX : <title>`. No work item number is known; ask the owner before the next PR.
- Do not spend Gemini quota without asking.

## Open items (in order)
1. DONE: #27 and #28 merged; suite verified passing on main (4a702fb).
2. **Brokerage eval scenarios** (owner approved, not started; branch `eval-brokerage` exists off origin/main, empty).
   Model on the existing `calculation` area: generate about 30 random portfolios/months and check against an independent oracle
   (no import of `brokerage.py` logic): (a) opening + deposits + dividends + market effect = closing to the cent, percent change,
   opening = previous closing, deposits equal the `ACME BROKERAGE TRANSFER` ledger rows; (b) concentration shares recomputed
   independently, flags match `LIMITS_BPS` (20% single stock, 50% technology, funds exempt) and the generated policy text states
   the same limits; optional (c) a claim with an invented number is removed. Adds two gates (14 to 16). Update gate labels in
   `webapp.py` (a test requires every label differs from its name), README counts, and tests that hard-code counts.
3. **Resumable live routing eval** (owner asked for it so the 55 routing calls complete across days without repeating work). Design:
   `--only-area routing --resume FILE` with a checkpoint (JSONL) of scenarios where the model really answered. On rerun skip those;
   record nothing for a fallback (rules used instead of the model) or a failed call; stop cleanly after N consecutive failures (quota),
   print progress, exit with a distinct code; produce the final report only when all 55 have a real model answer and
   `router_fallbacks` is 0 of 55. Trap found: a partly failed run falls back to the keyword rules and shows a healthy 0.98, and the
   early stop (80% of at least 10 calls) will not catch it with 55 calls. Roughly 3 days at about 20 calls/day, and it shares quota
   with chip rehearsal (rehearse first; the video is the deadline). Do not edit the router prompt in between (cache keys change).
   The owner runs the live part: `uv run python -m household_agent.evals --only-area routing --model gemini:gemma-4-26b-a4b-it`.
4. Rehearse the live chips on Gemini (owner), then record the video.
5. Refresh this file after the next merges. Optional later: dividend year-to-date chip; low PR #20 findings #9 (no-figures hint can be
   false), #8 (`LIKE 'hist_%'` treats `_` as a wildcard, webapp.py ~167/~188), #6 purpose hard-coded to `historical_comparison`,
   #7 name matching only member ids, #10 audit and consent panels not filtered by member.

## Run it
```
cd "/Users/somal.vora/Desktop/Agentic AI Course/project/.claude/worktrees/demo-ui-additions"
cp ../../../.env .env        # the worktree lacks the git-ignored .env (needs GEMINI_API_KEY)
uv run python -m household_agent.webapp --model gemini:gemma-4-26b-a4b-it
```
Omit `--model` for offline mode (consent, audit, eval and refusals work; real answers need a model). Restart the server after pulling
main: an old server still shows old figures.

## Gotchas
- Use `uv run python -m pytest` (not `uv run pytest`).
- The shell guard rejects heredocs, `$VAR` expansion, compound git commands and `cd` into another worktree; use plain commands
  (`git -C <path> <one command>`), the Write/Edit tools, and absolute paths (job tmp dir: `/Users/somal.vora/.claude/jobs/ac99ea51/tmp`).
- A subagent in an isolated worktree cannot be inspected from this one by `cd`; check its branch with `git checkout --detach <branch>`
  here and run tests.
- `demo.html` must never use `innerHTML` (a test enforces it). The hook flag on `agents.py` / `webapp.py` for eval/exec is a false
  positive (`re.compile`, graph `.compile`).
- A squash merge leaves the old branch "ahead": always branch new work from `origin/main`, never from a merged branch.

## Prompt for the next session
```
Continue the Household Wealth Agent work (demo UI and evals). Read HANDOFF.md first, in
/Users/somal.vora/Desktop/Agentic AI Course/project/.claude/worktrees/demo-ui-additions
(merged: #20-#23, #25-#28, main head 4a702fb, 14 gates / 348 scenarios; #29 is the handoff PR; GitHub push needs
`gh auth switch --hostname github.com --user somalvora` first and a switch back to somal-vora_sfemu after, see HANDOFF.md).

Goal: a video recorded by clicking only the suggestion chips, plus a stronger eval harness.

Do, in order:
1. git fetch; branch new work from origin/main (use the existing empty `eval-brokerage` if still unchanged; never branch from a
   merged branch); run `uv run python -m pytest tests -q` and `uv run python -m household_agent.evals` (expect 14 gates).
2. Add brokerage eval scenarios (performance vs an independent oracle, concentration vs the written policy), per
   "Open items" 2 in HANDOFF.md. Use one subagent in an isolated worktree if useful. Keep titles `@W-XXXXXXXX : ...`.
3. Then build the resumable live routing eval (`--resume`), per "Open items" 3. Never count fallbacks as model answers.
4. Refresh HANDOFF.md after each merge.
Standing rules: no real data in the repo (public); do not dispatch a code-review agent on PRs in this window; do not spend Gemini
quota without asking, the owner runs live rehearsals and the live eval themselves; no LangSmith.
```
