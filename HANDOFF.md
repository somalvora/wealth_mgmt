# Handoff: demo UI additions (Household Wealth Agent)

Written 2026-11-17. Worktree: `/Users/somal.vora/Desktop/Agentic AI Course/project/.claude/worktrees/demo-ui-additions`,
branch `worktree-demo-ui-additions`, latest commit `e3a746e`. **Nothing is pushed and there is no PR.**

## Goal
Make the web demo (`uv run python -m household_agent.webapp ...`, page at 127.0.0.1:8000) show most of the system from the UI,
so a video can be recorded by clicking only the suggestion chips. No upload step should be needed.

## What was built (all committed locally)
- **Panels** (`static/demo.html`, `webapp.py`): Share my accounts (consent grant/revoke, `/api/consent*`), live Audit trail list
  (`/api/audit` `recent`), Evaluation results (`/api/eval`, `/api/eval/run`, 13 gates, offline keyword router, about 2 s).
- **Injection sample PDF** `2026_09_somal_checking_injection.pdf` (`sample_statements.py`): reconciles, text is flagged in the audit list.
- **Preloaded data:** 2 years of statements (Oct 2024 to Sep 2026) for all three accounts, plus the October 2026 joint rental
  statement (`CURRENT_RENTAL` in `webapp.py`), so the tax chip needs no upload. That sample PDF is hidden from the upload list.
  Somal's Sep 2026 checking is intentionally empty (reserved for the tampered/injection upload samples); Oct 2026 checking and
  Sayali's card are not preloaded.
- **Explicit refusal** (`DemoService._private_account_refusal`): a question naming a member whose accounts the asker cannot read
  returns status `denied`, an alert, and a `deny` audit row, before any model runs. Name-based only (no name means the analyst is
  simply not offered those accounts and answers "no figures").
- **Chips** (`SUGGESTIONS`, 8): tax (Oct 3,200 county payment), allocate surplus, transfer refusal, "What did I spend on groceries in
  2025?" (expected $3,269.43, Somal's checking only), lease (rent and who pays property tax), viaan asks about Somal (refused),
  somal asks about Sayali's card (refused until Sayali shares), weather (out of scope).
- Page text says history is on file and upload is optional. `VIDEO.md` and `README.md` updated. Tests pass (`uv run python -m pytest tests -q`).

## Verified vs not
- Verified: the full test suite; the grocery figures against the ledger (2025 = $3,269.43; 2026 Jan to Aug = $2,320.31, which is
  why the old 2026 chip was misleading); both refusals offline.
- **Not verified on live Gemini:** the answers for the tax, groceries 2025, lease and allocation chips, and the answer after Sayali
  presses Share. Offline mode replays canned answers, so it cannot check those.
- Gemini free tier is about 20 requests/day per model. 429 (quota) and 503 (high demand) show as an amber message. Answers are
  cached in `.cache/llm.sqlite`, so rehearse each chip once, then record.

## Open items
1. **Push and PR.** `git push` fails with 403: the only github.com login is `somal-vora_sfemu`, which cannot write to
   `somalvora/wealth_mgmt`. Fix once: `! gh auth login -h github.com -p https -w` (personal account), then
   `git remote set-url origin https://somalvora@github.com/somalvora/wealth_mgmt.git` and `gh auth setup-git -h github.com`.
   Then push the branch and open a PR. After the PR, dispatch one `code-review` agent on it.
2. Rehearse the four live chips once on Gemini; check the numbers (groceries 2025 should be $3,269.43).
3. DONE: the Documents panel now starts with a one-line history summary (`/api/state` `history`).
4. Optional: the old eval-gap ideas (more adversarial scenarios, indirect injection, citation cases, non-regression baseline).
5. DONE: the no-figures hint in `agents.py` now says the statements on file did not cover it, and only then suggests an upload.

## Run it
```
cd "/Users/somal.vora/Desktop/Agentic AI Course/project/.claude/worktrees/demo-ui-additions"
cp ../../../.env .env        # the worktree lacks the git-ignored .env (needs GEMINI_API_KEY)
uv run python -m household_agent.webapp --model gemini:gemma-4-26b-a4b-it
```
Omit `--model` for offline mode (consent, audit, eval and refusals work; real answers need a model).

## Gotchas
- Use `uv run python -m pytest` (not `uv run pytest`).
- The worktree shell guard rejects heredocs, `$VAR` expansion and complex commands; use plain commands or the Write tool.
- `demo.html` must never use `innerHTML` (a test enforces it). The hook flag on `webapp.py` `re.compile` (eval/exec) is a false positive.
