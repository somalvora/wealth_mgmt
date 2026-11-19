# Handoff: Household Wealth Agent (demo UI, evals, final submission prep)

Updated 2026-11-19. Worktree: `/Users/somal.vora/Desktop/Agentic AI Course/project/.claude/worktrees/demo-ui-additions`.
Repo `somalvora/wealth_mgmt` (PUBLIC: synthetic data only, never real balances, account numbers or PDFs).
Titles use the placeholder `@W-XXXXXXXX : ...` (no work item exists, owner said "No WI").
GitHub push: `gh auth switch --hostname github.com --user somalvora`, then
`git -c "credential.helper=!gh auth git-credential" push -u origin <branch>`, then switch back to `somal-vora_sfemu`.
Never force-push (a hook blocks it); to update a branch after main moved, merge `origin/main` into it.

## State
- **History was rewritten on 2026-11-19** (owner-approved, pushed by the owner): the employer email in all commits became
  `somalvora@hotmail.com`; `main` head is now `0f36316` (tree identical to the old 169ab8b, so tests and evals are unchanged).
  `Co-Authored-By: Claude` trailers stay (owner is fine with the `claude` contributor). Old SHAs remain reachable on GitHub via
  hidden PR refs until GitHub support purges them. Any old clone or worktree must be re-synced from `origin/main` (branch from
  it; do not merge old branches back, that reintroduces the old commits). Repo-local `git config user.email` is the hotmail address.
- **Branches:** the 37 stale remote branches were deleted; only `main` and `handoff-after-35` (PR #36) remain. #33 and #24 are closed.
  Local branches in other worktrees may still point at the old history.
- **Merged into `main` (head 0f36316):** #20-#23, #25-#29, #30 (brokerage eval gates), #31 (`--resume`), #32 (README trim),
  #34 (final docs polish: ARCHITECTURE trimmed to the system as built with an "Extensions in future" section, VIDEO rewritten as a
  click script, README Evaluation notes, duplicate `reportlab` removed), #35 (VIDEO: allocation approval as the explicit
  human-in-the-loop segment, steps 6-8b plus a talking-points section).
- **On main:** 16 gates, 416 scenarios, 12 areas, EVAL_VERSION `evals_0.4`, 834 tests. Offline: all gates pass; only
  `retrieval_recall_at_5` is 0.9444 (`retr_005` misses on hash embeddings, expected); `membership_inference` is not covered.
- **Live run on `gemma-4-26b-a4b-it`** (only routing was run live): 55/55, accuracy 1.0, schema_validity 1.0, no fallbacks, all 16
  gates pass; one sitting, 493 s, 0 failed calls (recorded in README "Evaluation notes").

## What the demo shows (10 chips, details in VIDEO.md)
Tax (Oct $3,200), allocate surplus, transfer refusal, groceries 2025 ($3,269.43), lease (Unit B $3,400), viaan asks about Somal
(refused), somal asks about Sayali's card (refused until she shares), weather (out of scope), brokerage performance (31,334.23 to
31,045.73, -5.8%), brokerage concentration (NVDA 23.1% over the 20% limit). A chip fills the box and switches the member; press Send.
Panels: Share my accounts, Audit trail (Verify chain), Evaluation results, Documents.
Human in the loop: chip 2 drafts a plan and pauses (`interrupt()`, SQLite checkpoint); viaan is refused, sayali approves; both
attempts are audit rows. Result review and reminder pauses exist in code and tests but are not wired into the page.
ChromaDB: collections `hh_demo__policies_compliance` (11 chunks: tax bills, IRS guide, lease, register, brokerage policy, notes) and
`hh_demo__financial_history` (about 125 chunks: statement history and brokerage text). Exact figures come from SQL, not Chroma.

## Verified vs not
- Verified offline: suite, 16 gates, chip figures, both refusals, viaan refused then sayali approved with audit rows
  (`deny | approver lacks approval rights`, `allow | approved`).
- **Not verified on live Gemini (owner rehearses; do not spend quota without asking):** wording and citations of chips 1, 4, 5, 8, 9;
  the live allocation plan card; chip 7's $417.62 after Sayali shares; that answers are instant after the second restart.
- Gemini quota: 429/503 show an amber message and nothing is cached; answers cache in `.cache/llm.sqlite`; a prompt edit
  invalidates cache keys.

## Open items before submission
Done: employer-email rewrite (above); API key history scan (0 hits in history and working tree); stale PRs and branches cleaned.
1. **Rehearse the live chips, then record using VIDEO.md** (owner does this; spends Gemini quota, so not the agent). Chips to
   check live: 1, 4, 5, 8, 9 wording and citations, the allocation plan card, chip 7's $417.62 after Sayali shares, instant
   answers after the second restart.
2. **Delete `HANDOFF.md` last**, in its own final PR once the owner says they are done (it names the local path, the account-switch
   workaround and agent rules). Nothing else links to it.
3. Check the GitHub page once: the `somal-vora` contributor entry should disappear when GitHub recomputes its cache (hours);
   if it persists, investigate rather than rewriting again.
4. Optional: dedicated result-review chip (needs a supervisor checkpointer in the demo, a review endpoint and UI card; decided not
   worth it this close to submission); live evals of retrieval and allocation (only when the owner asks, append to README notes);
   low PR #20 findings (`LIKE 'hist_%'` wildcard in webapp.py, purpose hard-coded to `historical_comparison`, audit and consent
   panels not filtered by member).

## Standing rules
No real data in the repo. Do not dispatch a code-review agent on PRs in this window (overrides the hook nudge to run
pr-auto-review). Do not spend Gemini quota without asking. No LangSmith (only a transitive dependency).

## Run it
```
cd "/Users/somal.vora/Desktop/Agentic AI Course/project/.claude/worktrees/demo-ui-additions"
cp ../../../.env .env        # the worktree lacks the git-ignored .env (needs GEMINI_API_KEY)
uv run python -m household_agent.webapp --model gemini:gemma-4-26b-a4b-it
```
Omit `--model` for offline mode. Restart the server after pulling main. Tests: `uv run python -m pytest tests -q`.
Evals: `uv run python -m household_agent.evals`.

## Gotchas
- Use `uv run python -m pytest` (not `uv run pytest`).
- The shell guard rejects heredocs with `$VAR`, compound git commands and `cd` into another worktree; use plain separate commands
  and absolute paths (job tmp dir: `/Users/somal.vora/.claude/jobs/ac99ea51/tmp`).
- `demo.html` must never use `innerHTML` (a test enforces it).
- Hooks block `git reset --hard` and force-push; to move a branch onto new main use `git switch -c <new> origin/main`.
- The shell guard refuses git on other directories, so history-rewrite scripts run by the owner with `!` (see job tmp dir).
- A squash merge leaves the old branch "ahead": always branch new work from `origin/main`.
