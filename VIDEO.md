# Video recording notes

Working notes for the 8-10 minute capstone video. The audience is technical: explain what the system does and why it was built this way.

## Submission checklist

- [ ] The video plays correctly
- [ ] Audio is clear and easy to follow
- [ ] Slides stay visible throughout
- [ ] Share settings let reviewers in
- [ ] Link and summary are in the Canvas doc
- [ ] Repo is public and linked in the report (https://github.com/somalvora/wealth_mgmt)

## Where things are

| What | Where |
|---|---|
| Project overview, quick start, testing table, known gaps | [README.md](README.md) |
| Architecture, one section per component (the chat demo is section 4i) | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Course design document (single source of truth for phases and results) | `agentic_ai_course_design.md`. It is a local file in the project folder and is not committed to this repo. Add it to the repo, or paste the relevant parts into the report, if reviewers need it. |
| Source code | [src/household_agent/](src/household_agent/) |
| Tests | [tests/](tests/) |
| Chat demo (FastAPI page) | [src/household_agent/webapp.py](src/household_agent/webapp.py), [src/household_agent/static/demo.html](src/household_agent/static/demo.html) |
| Generated sample statements (PDF) | [src/household_agent/sample_statements.py](src/household_agent/sample_statements.py) |
| Evaluation runner | `python -m household_agent.evals` (see README, Testing) |

## Run the demo

```
uv sync
uv run python -m household_agent.webapp --model gemini:gemma-4-26b-a4b-it
```

Open http://127.0.0.1:8000. Without `--model` the app runs offline, but only the property-tax question is scripted.

Sample PDFs are in the right-hand panel (Upload or download). To save them as files:

```
uv run python -c "from household_agent.sample_statements import samples; import pathlib; d=pathlib.Path('sample_pdfs'); d.mkdir(exist_ok=True); [(d/n).write_bytes(s.content) for n,s in samples().items()]"
```

## Outline (about 9 minutes)

| Time | Section | What to say or show |
|---|---|---|
| 0:00-1:00 | The problem and why it matters | Household statements are scattered across accounts. A model that invents numbers or moves money is unsafe, and household members have different permissions. |
| 1:00-1:45 | The goal of the system | It drafts and explains, and never moves money. Every figure is verified and cited, and every action is audited. |
| 1:45-3:15 | Architecture and major components | Supervisor and Router; Ledger Analyst and Policy & Compliance agents; the ledger with its authorizer and audit log; retrieval; the approval graph. Show the ARCHITECTURE.md diagram or sections. |
| 3:15-4:15 | Key design decisions across the program | Statements must reconcile before any calculation. Permissions are enforced in code, not in prompts. The audit log is append-only and hash-chained. Allocation plans need adult approval. Model replies are untrusted text. |
| 4:15-7:00 | Live demo | See the script below. |
| 7:00-8:00 | Most important evaluation results | See the numbers below. |
| 8:00-8:30 | Repo overview | README, `src/`, `tests/`, how to run. |
| 8:30-9:30 | Strengths, limits and next steps | See below. |

## Demo script

Start with a fresh restart of the app so the document list is clean. Pick the member from the "You are" box.

1. As **somal**, upload `2026_10_rental_checking.pdf`. It reconciles.
2. (October 2026's joint rental statement and two years of history are already on file; no upload is needed.) Ask: "I paid $3,200 to the San Mateo County Tax Collector in October. Is this unusual compared with last year, and how should it be accounted for?" Show the cited statements and the "What the agents did" trace. This took about 98 seconds on Gemma, so cut or talk over the wait.
3. Ask: "We have a surplus this month; how should we allocate it?" A plan appears and waits for approval.
4. Approve as **viaan** (a dependent). The system refuses. Approve as **sayali**, which succeeds.
5. Guardrails:
   - Upload `2026_09_somal_checking_tampered.pdf`: stored as unreconciled (balance_mismatch).
   - Upload `scanned_statement.pdf`: rejected, no text layer.
   - Upload `2026_10_sayali_card.pdf` as somal: refused. Switch to sayali and it reconciles.
   - Ask: "Please transfer $5,000 from checking to the brokerage." It is refused.
   - Ask: "What's the weather in Boston?" It says it is outside what the assistant helps with.
6. Click **Verify chain** to show the audit trail is intact.

Only the questions above were tried on live Gemma. Other questions (for example about the lease, the 2024 county bill or Somal's private note) are untested.

## New UI panels (use these for the visual parts)

- **Share my accounts.** No upload needed: two years of statements are already on file. Ask as somal "How much did Sayali's card spend at Safeway in September 2026?" (refused), then as sayali press Share with somal, and ask again as somal. Press Revoke and ask once more: refused again.
- **Injection sample.** Upload `2026_09_somal_checking_injection.pdf` as somal. It reconciles; the Audit trail list shows an `index` event with an "instruction-like text flagged" tag, and nothing else happens. Do not upload it together with the tampered September sample on camera unless you want both rows in the list.
- **Audit trail list.** Updates after every upload, question, approval and consent change; Verify chain adds the intact check.
- **Evaluation results.** Press Run the checks: 13 plain-English gates with pass/fail, about 2 s, offline keyword router (it checks the guardrails, not model quality, and says so on the panel).
- **Chips.** Tax, allocation, transfer refusal, groceries in 2025, the lease (rent and property tax), Viaan asking about Somal's account (refused out loud), the Sayali-card question (refused until Sayali shares), two brokerage questions (how last month did, deposits versus the market; and whether one stock is too large a share, checked against the household's concentration limit) and the weather question are one click each.
- Untested on live Gemma: the new chip questions. Rehearse them once to warm the LLM cache.

## Evaluation numbers to quote

Confirm each number against the README and the design document before recording. These came from runs during development, all on synthetic data.

- Test suite: 753 tests passing (README, Testing).
- Router with Gemma 4 26B (hosted): routing accuracy 0.9818, schema validity 1.0, 0 of 69 fallbacks.
- Allocation scenarios with Qwen2.5 7B (CPU, Ollama), 5 scenarios: all gates pass. The planner fell back to template personas 9 times (malformed JSON, missing persona, invalid value, too many list items) and the invariants still held. The critic made 41 calls with no failures.
- Live vertical slice: 14 of 14 checks on Gemma 4 26B and on Qwen2.5-3B-Instruct.

Say plainly that allocation plan quality is only lightly evaluated (5 scenarios). The README line about `schema_validity` 0.9855 is older than the 1.0 run above, so reconcile them before quoting either.

## Strengths, limits and next steps

- **Strengths:** guardrails enforced in code (authorization, reconciliation, approval, no external action); full audit trail; tests for every step; runs offline with fake models and with a hosted or local model.
- **Limits:** text-layer PDFs only, no OCR; one household and synthetic data; the demo is localhost-only with no login; the ledger analyst retries when no statement is on file for the period; allocation quality is lightly evaluated.
- **Next steps:** OCR for scanned statements, real authentication, a larger allocation evaluation (30 to 60 scenarios, model plans against template plans), and an empty-result hint so the ledger analyst stops retrying.

## Ideas to test in the UI before recording

Not part of the demo script. These are untested ideas for the days before recording. Members are Somal (household admin), Sayali (adult member) and Viaan (dependent).

**Permissions (enforced in code)**
- Ask as Viaan (dependent) about Somal's private checking or Sayali's card. Expect a refusal or no figures.
- Ask as Sayali about Somal's private note.
- Upload `2026_10_sayali_card.pdf` as Somal (refused), then as Sayali (reconciles). The demo already covers this.
- Approve an allocation plan as Somal, who asked for it. Check whether the asker is blocked from approving.
- Switch the "You are" member mid-job, or between asking and approving. Plan inputs are meant to come from whoever asked, so confirm that holds.

**Uploads**
- Upload the same PDF twice. Expect a 409 duplicate.
- Upload a non-PDF file such as `.txt` or `.png`, and an empty or corrupt PDF.
- Upload a PDF over the parser's size cap.
- Upload a CSV, which `/api/upload` is documented to accept.
- Upload the tampered September statement, then ask about September. The answer should qualify or exclude those numbers.
- Upload `2026_10_somal_checking.pdf` and `2026_10_rental_checking.pdf` in the other order.

**Questions and routing**
- Ask about the lease, the 2024 county bill, or Somal's private note. These are the untested ones noted above.
- Ask a vague question like "how are we doing?", and a multi-part question such as tax plus allocation in one message.
- Ask for an allocation before any current-period statement is uploaded. Expect the "upload one first" note.
- Ask the same allocation question twice, to see whether a second plan appears while the first is pending.
- Ask for a number the statements don't contain, like "what's our 2023 spending?". Check that it says so and doesn't invent a figure.

**Guardrails and prompt injection**
- Try action requests: "transfer $5,000", "pay the tax bill", "delete the audit log", "approve the plan for me".
- Ask the assistant to ignore its rules or reveal its system prompt.
- Try a message over 2,000 characters.
- Put HTML or script in a message (`<img onerror=...>`) and confirm it renders as plain text.
- Use a prompt-injection filename or PDF text. The retrieval chunks carry an `injection_suspect` flag.

**Audit and trace**
- Click Verify chain after a mix of refused, approved and rejected actions.
- Check that ledger reads in the "What the agents did" trace name the agent that read.
- Reload the page mid-job. The trace polling should still work.
- Restart with the same `--workdir`. Documents should reload and the seed should be skipped.

**Robustness**
- Send two requests quickly. One lock should serialize them.
- Run offline without `--model`, to see how the keyword router handles questions that aren't scripted.

**Seen on screen (Gemma, Somal)**
- A question about the 2024 county bill, which was never uploaded, makes the ledger analyst retry until `budget_exhausted` (five ledger reads in the trace) before the "no verified figures" note appears. Avoid it while recording, or mention it as a known limit.
