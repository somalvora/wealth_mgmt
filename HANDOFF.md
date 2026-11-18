# Handoff: demo UI additions (Household Wealth Agent)

Updated 2026-11-18 (evening). Worktree: `/Users/somal.vora/Desktop/Agentic AI Course/project/.claude/worktrees/demo-ui-additions`.
Repo `somalvora/wealth_mgmt`. PR #20 (demo UI) and PR #21 (refusal scope, member sync) are **merged into `main`**.
**PR #22 is open** (branch `demo-chip-clarity-and-fixes`, title `@W-XXXXXXXX : ...`, no work item exists: owner said "No WI"):
PR #20 findings #2/#3/#11 fixed, plus the rent-data fix below. Suite passes. Nothing is merged from it yet.
GitHub login is set up (`somalvora` is the active account, push works). Start unrelated new work on a branch off `origin/main`.
Note: `origin/main` still has the OLD (11-17) HANDOFF.md until PR #22 merges; this file is the current one.

## Goal
Make the web demo (`uv run python -m household_agent.webapp ...`, page at 127.0.0.1:8000) show most of the system from the UI,
so a video can be recorded by clicking only the suggestion chips. No upload step should be needed.

## What is on main
- **Panels** (`static/demo.html`, `webapp.py`): Share my accounts (consent grant/revoke, `/api/consent*`), live Audit trail
  (`/api/audit` `recent`), Evaluation results (`/api/eval`, `/api/eval/run`, 13 gates, offline keyword router, about 2 s).
- **Preloaded data:** 2 years of statements (Oct 2024 to Sep 2026) for all three accounts, plus the October 2026 joint rental
  statement (`CURRENT_RENTAL`), so the tax chip needs no upload. Somal's Sep 2026 checking is intentionally empty (reserved for the
  tampered/injection upload samples). Oct 2026 checking and Sayali's card (`2026_10_sayali_card.pdf`) are upload samples only.
- **Documents panel** starts with a one-line history summary (`/api/state` `history`, from `history.history_summary()`).
- **No-figures hint** (`agents.py` `NO_LEDGER_FIGURES`): says the statements on file did not cover it, suggests an upload only if
  the question needs a statement that is not on file.
- **Explicit refusal** (`DemoService._private_account_refusal`): a question naming a member whose PRIVATE account the asker cannot
  read returns status `denied`, an alert and a `deny` audit row, before any model runs. Skipped when the question is about the
  joint rental account (keyword `_JOINT_TOPIC`: rent, lease, tenant, Unit X, property tax, mortgage) and the asker can read it.
  Names the account ("somal's private checking account", "sayali's private credit card"). For a dependent it says dependents can't
  be given access to ledger data (no "can share it") and audits that reason.
- **UI:** the member dropdown, chips and sample Upload buttons all go through `syncMember` / `pickMember`, so the role label and
  Share panel follow the selected member.
- **Chips** (`SUGGESTIONS`, 10, incl. two brokerage chips below): tax (Oct $3,200 county payment), allocate surplus, transfer refusal, groceries 2025 (expect $3,269.43,
  Somal's checking only), lease (Unit B rent and property tax), **viaan asks about Somal's Safeway spend (runs as viaan, refused)**,
  somal asks about Sayali's card (refused until Sayali shares), weather (out of scope).

## Changed in PR #22 (2026-11-18)
- **Rent now matches the lease.** The 2026 lease says Unit B rent is $3,400 (`synthetic.py` lease text) but the ledger deposits were
  $3,100, so the live lease chip answered "$3,100" (found in the owner's browser). Now rent is 3,000 before Apr 2025, 3,100 Apr to
  Dec 2025, **3,400 from Jan 2026** (`history.py` `_rental_rows`), and the Oct 2026 statement (`synthetic.py`, `sample_statements.py`)
  deposits 3,400. Oct 2025 stays 3,100. Five tests updated (+$300 shifts: rental closing 16,871.60, October inflow $11,600, net
  $3,839.63, planner cash +$300). Allocation plan with the new data: cash $91,876.30 minus $3,200 county tax minus 3-month reserve
  $13,348.41 = $75,327.89 deployable = brokerage $51,795.11 + HSA 4,000 + IRA 7,000 + 529 5,000 + buffer $7,532.78 (adds up exactly).
  Caveat for narration: it plans ALL cash above reserve (about $75k), not "this month's surplus". Somal's checking balance is as of
  Aug 31 (Sep is empty by design), the rental balance as of Oct 31.
- Refusal now runs under `self.lock`, is skipped for money-movement / out-of-scope questions (`rule_route`), and the page shows only
  the exception type (`agents.error_kind`); the raw "ledger_analyst failed: ..." line is gone from `demo.html`.
- Chip clarity (old open item 1) was **dropped** by the owner; the viaan Safeway chip is unchanged.
- **Restart the server on this branch** before rehearsing: a server started earlier still shows the old plan figures
  ($51,525.11) and the raw 429 text.

## Brokerage chips (added in PR #22, 2026-11-18)
Built from the owner's choice "A and B", option 2 (structured holdings plus deterministic calc tools). **All data is fictional**;
the owner's real PDFs were shape reference only and are not in the repo (repo is PUBLIC: never add real balances, tickers, account
numbers or PDFs).
- **Chips** (10 total now): "How did my brokerage account do last month, and how much was my deposits versus the market?" and
  "Am I too concentrated in any one stock?" (both member somal).
- **`brokerage.py`**: 11 fictional holdings (4 technology stocks, 3 other stocks, 4 funds), statements Aug/Sep/Oct 2026 for
  `acct_somal_brokerage`. Sep close matches the Q3 2026 quarterly summary and the $1,500 deposits match the checking
  `ACME BROKERAGE TRANSFER` rows, so the sources agree. October: opens 31,334.23, deposit 1,500.00, dividends 31.40, closes
  31,045.73, market effect -1,819.90 (-5.8%). Concentration (Oct): largest NVCP 23.1% (over the 20% single-stock limit), top 4 = 54.9%,
  technology 46.2% (under the 50% limit; September was over). `LIMITS_BPS` is the single source for both the flags and the generated
  policy document (`policies_compliance`, so chip B combines retrieval with a calculation).
- **Plumbing**: `brokerage_statements` table (append-only; `LedgerStore.ingest_brokerage_month` enforces owner-only upload,
  brokerage account kind, no duplicate month, opening = previous closing; `brokerage_month` reads). `calc_runner.record_run` records
  hashed, audited runs for both ledger and brokerage calcs. `ToolSession.brokerage_performance/brokerage_concentration`, MCP tools,
  LedgerAnalyst tool specs + evidence handling (claims validate against the run output). `rule_route` has a `_BROKERAGE` regex
  (not applied when money movement is requested, so the transfer chip is still refused); concentration also routes to policy.
- **Account**: `synthetic.BROKERAGE_ACCOUNT` (PRIVATE, somal) lives in `DEMO_ACCOUNTS` only (web demo), not `ACCOUNTS`, so existing
  tests and evals are untouched. `plan_inputs` skips brokerage-kind accounts (allocation plan figures unchanged, not provisional).
  Refusal wording: a brokerage question names only brokerage accounts ("somal's private brokerage account"); other questions never mention it.
- `history.py` quarterly summaries no longer claim a 70/30 holdings split (they point to the monthly statements).
- Tests: `tests/test_brokerage.py` (11), plus brokerage cases in `test_agents.py` and `test_webapp.py`; full suite passes.
- **Rehearse both chips live on Gemini** once the quota resets (not run live; no quota spent).

## Verified vs not
- Verified: full suite passes (`uv run python -m pytest tests -q`); groceries 2025 = $3,269.43; both refusals offline.
- Sayali's Safeway 2026 total $3,507.20 (Gemma's live answer) is correct: Jan to Sep 2026 on file, recomputed from the generated
  CSVs. October 2026 would add $192.20 (full year $3,699.40) but that card statement is not preloaded, so a good answer says
  "through September". Not checked: that the app's own ledger tool returns the same number.
- **Not verified on live Gemini:** the tax, groceries 2025, lease and allocation chips, and the answer after Sayali presses Share.
  The owner rehearses these once the quota resets; do not spend model quota on them without asking.
- Gemini free tier is about 20 requests/day per model. 429 (quota) and 503 (high demand) show as an amber message. Answers are
  cached in `.cache/llm.sqlite`, so rehearse each chip once, then record.
- The browser has not been clicked through since PR #21 (only a test on the page source). Click the viaan chip once and confirm
  the Share panel says "viaan owns no accounts".

## Standing instructions (this window)
- Do NOT dispatch a code-review agent on PRs opened in this window. This overrides the usual "review every new PR" default and the
  PostToolUse hook nudge to run pr-auto-review after `git push` or `gh pr create`. PR #20 already had its one review.
- PR titles use the convention `@W-XXXXXXXX : <title>`. **No work item number is known yet**; both merged PRs carry the placeholder.
  Ask the owner for the number before opening the next PR.

## Open items
1. **Chip clarity: dropped.** The owner decided not to implement it (2026-11-18); the viaan Safeway chip stays as is.
2. Rehearse the live chips on Gemini (owner does this). The daily free quota ran out on 2026-11-18 (429 RESOURCE_EXHAUSTED); wait for
   the reset and click each chip once, no rephrasing. Re-rehearse the **lease** and **allocation** chips (data changed, cache
   keys differ) and ideally the tax chip. Offline mode (no `--model`) still shows refusals, consent, audit and eval.
2b. **Rehearse the two new brokerage chips on Gemini** (done offline only; cache keys are new). Possible later extras: a dividend
   year-to-date chip, evals for the brokerage tools (the 13 gates do not cover them yet).
3. Review findings from PR #20. Fixed on branch `demo-chip-clarity-and-fixes` (2026-11-18; no work item number, placeholder title):
   #3 (refusal now runs under `self.lock`), #2 (refusal skipped when `rule_route` says money movement or out of scope; offline,
   no model call), #11 (the page gets the exception type only, via `agents.error_kind`; the duplicate "failed:" line is gone from
   demo.html; `_MODEL_BUSY` no longer matches bare 429/503; a tool name is no longer shown as the error kind). Still open, low: #9 no-figures hint can be false (unshared account, exhausted budget); #8 `LIKE 'hist_%'` treats `_` as a wildcard
     (webapp.py ~167, ~188); #6 purpose hard-coded to `historical_comparison`; #7 name matching only member ids; #10 audit and
     consent panels not filtered by member.
4. Optional: the old eval-gap ideas (more adversarial scenarios, indirect injection, citation cases, non-regression baseline).

## Run it
```
cd "/Users/somal.vora/Desktop/Agentic AI Course/project/.claude/worktrees/demo-ui-additions"
cp ../../../.env .env        # the worktree lacks the git-ignored .env (needs GEMINI_API_KEY)
uv run python -m household_agent.webapp --model gemini:gemma-4-26b-a4b-it
```
Omit `--model` for offline mode (consent, audit, eval and refusals work; real answers need a model).

## Gotchas
- Use `uv run python -m pytest` (not `uv run pytest`).
- The worktree shell guard rejects heredocs, `$VAR` expansion, `python -c` with git and compound git commands; use plain commands
  (`git -C <path> <one command>`) or the Write tool.
- `demo.html` must never use `innerHTML` (a test enforces it). The hook flag on `webapp.py` `re.compile` (eval/exec) is a false positive.
- A squash merge leaves the old branch "ahead": always branch new work from `origin/main`, never from a merged branch.

## Prompt for the next session
```
Continue the Household Wealth Agent demo-UI work. Read HANDOFF.md first, in
/Users/somal.vora/Desktop/Agentic AI Course/project/.claude/worktrees/demo-ui-additions
(PRs #20 and #21 are merged; PR #22 is open on branch demo-chip-clarity-and-fixes, so continue on that branch or branch
from it; GitHub login already works).

Goal: a video recorded by clicking only the suggestion chips, no upload step.

Do, in order:
1. git fetch, check PR #22 (merged or not), run `uv run python -m pytest tests -q` (should pass).
2. Work item number: none exists; keep the `@W-XXXXXXXX : <title>` placeholder unless I give one.
3. Brokerage chips A and B are built (see "Brokerage chips" in HANDOFF.md); I rehearse them on Gemini. Never commit real figures.
4. Remaining low PR #20 findings (#9, #8, #6, #7, #10) only if I ask.
Do not dispatch a code-review agent on PRs in this window. Do not spend Gemini quota
without asking; I will rehearse the live chips myself.
```
