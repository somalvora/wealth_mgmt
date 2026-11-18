# Handoff: demo UI additions (Household Wealth Agent)

Updated 2026-11-18. Worktree: `/Users/somal.vora/Desktop/Agentic AI Course/project/.claude/worktrees/demo-ui-additions`.
Repo `somalvora/wealth_mgmt`. PR #20 (demo UI) and PR #21 (refusal scope, member sync) are both **merged into `main`**.
GitHub login is set up (`somalvora` is the active account, push works). Start new work on a new branch off `origin/main`.

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
- **Chips** (`SUGGESTIONS`, 8): tax (Oct $3,200 county payment), allocate surplus, transfer refusal, groceries 2025 (expect $3,269.43,
  Somal's checking only), lease (Unit B rent and property tax), **viaan asks about Somal's Safeway spend (runs as viaan, refused)**,
  somal asks about Sayali's card (refused until Sayali shares), weather (out of scope).

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
2. Rehearse the live chips on Gemini (owner does this).
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
(branch off origin/main; PRs #20 and #21 are merged; GitHub login already works).

Goal: a video recorded by clicking only the suggestion chips, no upload step.

Do, in order:
1. git fetch, branch from origin/main, run `uv run python -m pytest tests -q` (should pass).
2. Ask me for the work item number (W-XXXXXXXX) before opening any PR; title format `@W-XXXXXXXX : <title>`.
3. Ask me which chip-clarity option I want (HANDOFF open item 1), then implement it.
4. Fix PR #20 findings #3 (lock), #2 (refusal before routing), #11 (raw error text), with tests.
Do not dispatch a code-review agent on PRs in this window. Do not spend Gemini quota
without asking; I will rehearse the live chips myself.
```
