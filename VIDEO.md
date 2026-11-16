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

1. As **alex**, upload `2026_10_rental_checking.pdf`. It reconciles.
2. Ask: "I paid $3,200 to the San Mateo County Tax Collector in October. Is this unusual compared with last year, and how should it be accounted for?" Show the cited statements and the "What the agents did" trace. This took about 98 seconds on Gemma, so cut or talk over the wait.
3. Upload `2026_10_alex_checking.pdf`. Ask: "We have a surplus this month; how should we allocate it?" A plan appears and waits for approval.
4. Approve as **sam** (a dependent). The system refuses. Approve as **jordan**, which succeeds.
5. Guardrails:
   - Upload `2026_09_alex_checking_tampered.pdf`: stored as unreconciled (balance_mismatch).
   - Upload `scanned_statement.pdf`: rejected, no text layer.
   - Upload `2026_10_jordan_card.pdf` as alex: refused. Switch to jordan and it reconciles.
   - Ask: "Please transfer $5,000 from checking to the brokerage." It is refused.
   - Ask: "What's the weather in Boston?" It says it is outside what the assistant helps with.
6. Click **Verify chain** to show the audit trail is intact.

Only the questions above were tried on live Gemma. Other questions (for example about the lease, the 2024 county bill or Alex's private note) are untested.

## Evaluation numbers to quote

Confirm each number against the README and the design document before recording. These came from runs during development, all on synthetic data.

- Test suite: 686 tests passing (README, Testing).
- Router with Gemma 4 26B (hosted): routing accuracy 0.9818, schema validity 1.0, 0 of 69 fallbacks.
- Allocation scenarios with Qwen2.5 7B (CPU, Ollama), 5 scenarios: all gates pass. The planner fell back to template personas 9 times (malformed JSON, missing persona, invalid value, too many list items) and the invariants still held. The critic made 41 calls with no failures.
- Live vertical slice: 14 of 14 checks on Gemma 4 26B and on Qwen2.5-3B-Instruct.

Say plainly that allocation plan quality is only lightly evaluated (5 scenarios). The README line about `schema_validity` 0.9855 is older than the 1.0 run above, so reconcile them before quoting either.

## Strengths, limits and next steps

- **Strengths:** guardrails enforced in code (authorization, reconciliation, approval, no external action); full audit trail; tests for every step; runs offline with fake models and with a hosted or local model.
- **Limits:** text-layer PDFs only, no OCR; one household and synthetic data; the demo is localhost-only with no login; the ledger analyst retries when no statement is on file for the period; allocation quality is lightly evaluated.
- **Next steps:** OCR for scanned statements, real authentication, a larger allocation evaluation (30 to 60 scenarios, model plans against template plans), and an empty-result hint so the ledger analyst stops retrying.
