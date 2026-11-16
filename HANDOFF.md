# Handoff: household wealth agent demo

Written at the end of a session so the next one can pick up cold. Supersedes the earlier HANDOFF.md on the `handoff-notes` branch.

## Where things stand

- `main` has everything: #11 out-of-scope message, #12 VIDEO.md, #13 demo members renamed (Somal admin, Sayali adult, Viaan dependent), #14 category-totals fix plus card statement generator, #15 keyword router knows spending words, #16 a crashed analyst is not reported as a missing statement, #17 `run_calculation` params contract, the last merge: failure notes plus `cash_flow` `exclude_categories`.
- No open PRs. Tests: **736 passing**. Run with `PYTHONPATH=src .venv/bin/python -m pytest -q` (the config prints dots only; count with `--collect-only -q`).
- Verified live (Gemma `gemini:gemma-4-26b-a4b-it`, as sayali, card statement uploaded): "summarize last month's spending activity" completes with October 2026 category totals (entertainment $23.99, groceries $274.85, repairs $1,776.63, utilities $177.85, uncategorized $1,585.04; the +$2,206.97 `transfer` is left out of the summary).

## What was found and fixed this session

1. The Gemini 429 was transient (quota reset); no code change. Retries already exist: `HOSTED_MAX_RETRIES = 6` goes to the Google SDK, which retries 429 and 503. A note in the briefing means the outage outlasted them.
2. Real bug: Gemma called `run_calculation` with no `params` (bare `KeyError('start')`, three times, analyst gave up) and sometimes sent `params` as a JSON string. Fixed: `calc_runner.REQUIRED_PARAMS` drives the tool description and a clear missing-params error; `tools._params_object` reads a JSON-string `params`.
3. A provider failure left the briefing blank. Fixed: `failure_notes` in `compose_briefing` ("model service is busy or over its quota ... ask again"; other crashes show the exception type only).
4. `cash_flow` counted a card payment as income. Fixed: optional `params.exclude_categories`; the analyst prompt asks for `["transfer"]` on inflow/outflow questions.

## First thing next session

1. **Still unverified live: "what is my inflow and outflow of money last month. Basically, money which came in and money which went out."** as sayali. Three attempts hit Gemini 503, 503, then 429 (quota), so we never saw the answer or whether Gemma passes `exclude_categories: ["transfer"]` on its own. Expected for the realistic card statement: inflow $0, outflow about $3,838.36 (the sum of the five spending categories above), net about -$3,838.36. If Gemma leaves the transfer in, inflow shows $2,206.97: tighten the prompt or default to excluding transfers inside `cash_flow` for the agent.
2. Run it without the browser, driving `DemoService` directly (what the UI calls): upload `~/Downloads/2026_10_sayali_card_realistic.csv` as sayali, then `svc.chat("sayali", question)`. Scripts used last time lived in `$CLAUDE_JOB_DIR/tmp` (gone with the job); recreate: build `DemoService(Path(tmp), model_factory=lambda role: get_chat_model("gemini:gemma-4-26b-a4b-it"))`, call `upload`, then `chat`, print `statements`, `questions`, `agents[].errors`.
3. If quota is out, wait for the reset or try another model with `--model`. The briefing now says so instead of staying blank.
4. Regenerate the statement if missing: `uv run python scripts/make_card_statement.py --out ~/Downloads/2026_10_sayali_card_realistic.csv`. Uploading it as somal is refused (no consent for sayali's private card). The demo clock is 2026-11-01, so "last month" is October 2026.

## Ideas not done (ask before doing)

- Fewer model calls per question (router plus analyst each cost calls; one run can burn several of a small free quota).
- Gemma does its own arithmetic sometimes (a $3,838.36 total was dropped by the claim validator, correctly). Offering a spending-total calculation would let such a claim cite a run instead of being removed.
- A converter layer for real bank exports (the Chase CSV format is not `hh_statement_csv_v1`); we chose a generator script and fictional data instead.
- Open the UI test-idea list in `VIDEO.md` before recording the video.

## Conventions to keep

- Every change: pytest tests (happy, negative, property), README Testing table and total, ARCHITECTURE.md, and the VIDEO.md test count.
- Maintain only `agentic_ai_course_design.md` as the design source; the capstone doc is reference.
- No local LLM on this laptop (Phase 2 uses fakes); Gemma runs via the Gemini API or on the other laptop.
- Skip pr-auto-review on pushes; never push to `main`.
- Worktree: `/Users/somal.vora/Desktop/Agentic AI Course/project/.claude/worktrees/tool-hints`. The guard refuses compound shell commands, heredocs and `git -c` with computed values; use plain, separate commands and the Read/Edit/Write tools.
- **Pushing:** the active `gh` account is `somal-vora_sfemu`, which cannot push to `somalvora/wealth_mgmt` (403). Push from a separate terminal: `setopt NO_BANG_HIST`, then `git -c credential.helper= -c 'credential.helper=!f() { echo username=somalvora; echo password=$(gh auth token --user somalvora); }; f' push -u origin <branch>`. Opening PRs also needs the `somalvora` account.
