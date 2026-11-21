# Video recording notes

Working notes for the 8-10 minute capstone video. The audience is technical: explain what the system does and why it was built this way. The live demo is recorded by clicking the suggestion chips in the web page, with a few clicks in the side panels.

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
| Project overview, quick start, known gaps | [README.md](README.md) |
| Architecture, one section per component (the chat demo is section 4i) | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Run log of the live routing eval | [README.md](README.md) (Evaluation notes) |
| Course design document | `agentic_ai_course_design.md`. Local file, not committed. Paste the relevant parts into the report if reviewers need them. |
| Source, tests | [src/household_agent/](src/household_agent/), [tests/](tests/) |
| Chat demo | [webapp.py](src/household_agent/webapp.py), [static/demo.html](src/household_agent/static/demo.html) |

## Pre-flight (the day before, and again just before recording)

1. Pull main, then **restart the server**. An old server still shows old figures.
2. Run from the folder that holds `.env` (the app reads `.env` from the working directory) with `GEMINI_API_KEY` set. Then:
   ```
   uv run python -m household_agent.webapp --model gemini:gemma-4-26b-a4b-it
   ```
   Open http://127.0.0.1:8000. The header tag must read `model: gemini:gemma-4-26b-a4b-it`. Offline (no `--model`) the page works but most chips have no real answer.
3. **Rehearse every chip once with the model.** A live answer can take about 100 s; answers are cached in `.cache/llm.sqlite`, so the recorded take is then instant. Any edit to a prompt or a model change invalidates the cache, so do not pull or edit after rehearsing.
4. **Restart the server once more before the take.** Each start uses a fresh temporary working folder, so the Documents list, the audit trail, pending plans and Share grants start clean, while the answer cache on disk survives. Click one rehearsed chip to confirm it answers instantly (not checkable offline; see "Not verified offline").
5. Reload the page before each take to clear the chat log. Browser zoom around 110 percent so the panels are readable.
6. Quota: if a chip shows an amber "The model service is busy or over its quota ..." message (a 429 or 503 from the model), nothing was cached. Wait a minute and click Send again. Failed calls are never cached, so a retry costs nothing extra. Do not rehearse everything in one burst; the shared limiter allows about 10 requests per minute by default (`HH_REQUESTS_PER_MINUTE`).

## How the chips work

A chip does two things: it switches "You are" to the member named below, and it puts the question in the box. **It does not send it.** Press Send (or Enter) after each chip. The chip text on screen is cut at 70 characters; hover shows the full text.

Chip order on the page, left to right, wrapping:

| # | Chip starts with | As |
|---|---|---|
| 1 | "I paid $3,200 to the San Mateo County Tax Collector in October. Is ..." | somal |
| 2 | "We have a surplus this month; how should we allocate it?" | somal |
| 3 | "Please transfer $5,000 from checking to the brokerage." | somal |
| 4 | "What did I spend on groceries in 2025?" | somal |
| 5 | "What is the monthly rent for Unit B, and who pays the property tax?" | somal |
| 6 | "How much did Somal spend at Safeway in September 2026?" | viaan |
| 7 | "How much did Sayali's card spend at Safeway in September 2026?" | somal |
| 8 | "How did my brokerage account do last month, and how much was my dep..." | somal |
| 9 | "Am I too concentrated in any one stock?" | somal |
| 10 | "What's the weather in Boston?" | somal |

## Outline (about 9 minutes)

| Time | Section | What to say or show |
|---|---|---|
| 0:00-1:00 | The problem and why it matters | Household statements are scattered across accounts. A model that invents numbers or moves money is unsafe, and household members have different permissions. |
| 1:00-1:45 | The goal of the system | It drafts and explains, and never moves money. Every figure is verified and cited, and every action is audited. |
| 1:45-3:15 | Architecture and major components | Supervisor and Router; Ledger Analyst and Policy & Compliance agents; the ledger with its authorizer and audit log; retrieval; the approval graph. Show the ARCHITECTURE.md diagram or sections. |
| 3:15-4:15 | Key design decisions across the program | Statements must reconcile before any calculation. Permissions are enforced in code, not in prompts. The audit log is append-only and hash-chained. Allocation plans need adult approval (the human-in-the-loop moment, shown live in demo steps 6-8b). Model replies are untrusted text. |
| 4:15-7:15 | Live demo | The script below. |
| 7:15-8:00 | Evaluation results | The Evaluation results panel and the numbers below. |
| 8:00-8:30 | Repo overview | README, `src/`, `tests/`, how to run. |
| 8:30-9:30 | Strengths, limits and next steps | See the end of this file. |

## Intro slides talk track (about 2.5 minutes, before the live demo)

Seven slides: title, why, agents, architecture, guardrails, tests, "Demo". Say it in your own words; these are prompts.

| Slide | Say (about) |
|---|---|
| 1. Household Wealth Agent | "A family asks something like 'why did our property tax go up?' and gets an answer where every number points back to a statement line. It runs locally, it is multi-agent, and all the data is synthetic." (15 s) |
| 2. Why this project | "Money is high stakes: wrong numbers and leaked data both hurt. Plain language models get arithmetic wrong, invent sources and do not know who may see which account. So the model chooses and explains, and plain code does the math and the permission checks." (25 s) |
| 3. The agents | "A router picks the request type and flags money movement; keyword rules back it up. A supervisor, which is code, dispatches the work and enforces budgets: at most 8 tasks and 300 seconds. The Ledger Analyst pulls verified transactions and runs calculations. The Policy and Compliance agent retrieves the tax bills, lease and IRS guide with citations. For 'what should we do with our surplus' a planner builds the plan in three steps, Safety, then Tax-advantaged, then Growth. At each step the same model suggests three strategies, conservative, balanced and tax-aggressive, with no dollar amounts. Code turns them into numbers, scores them and keeps the best two, and a critic advises but never prunes. The models never produce the numbers." (30 s) |
| 4. Architecture | "Green boxes are models, gray is plain code, gold is a person. Most of the diagram is gray. Statements are reconciled to zero cents, meaning opening balance plus transactions must equal the closing balance to the exact cent, before anything uses them. Every claim is re-validated against evidence before it reaches the answer. Allocation plans wait for an adult or admin to approve, and nothing executes." (25 s) |
| 5. Guardrails | "Access is decided in code before any model touches account data. Money movement is refused and audited. Owners see their own accounts; anyone else needs every owner's consent. Statements with material reconciliation errors are blocked from calculations. Every allow and deny goes into a hash-chained audit log you can verify in the UI. In the demo, Viaan asking about Somal's account is refused, and that refusal is in the audit trail." (25 s) |
| 6. How I tested it | "420 seeded synthetic scenarios across 12 areas, scored by 16 pass/fail gates. All pass offline. On a real model, routing was 55 of 55, the allocation invariants scored 1.0 over 6 scenarios (some planner thoughts fell back to templates), and retrieval recall at 5 was 1.0 with real embeddings. Negative controls check that an invented citation is caught, and there are 800+ unit and property tests." (25 s) |
| 7. Demo | "I will click only the suggestion chips, including one where the plan pauses and an adult or admin has to approve it." (10 s) |

If asked about the planner: it is one model asked, once per branch of the plan, for three options, not three agents. The three "viewpoints" are conservative, balanced and tax-aggressive; the three steps (Safety, Tax-advantaged, Growth) are the stages of the plan. The model only suggests strategies (shares and buckets); a template generator is the fallback per viewpoint, and the critic is advisory. Live allocation runs fell back to template thoughts after model API errors (18 in the 6-scenario run) and 1 of 42 critic calls failed; the invariants still held because code enforces them. Offline recall at 5 is 0.9444 (hash embeddings, `retr_005`); with a real embedding model it is 1.0.

## Demo script (about 3 minutes with warm cache)

Start on the page with "You are" = somal. Two years of statements (Oct 2024 to Sep 2026) and the October 2026 joint rental statement are already on file, so no upload is needed. Say that once: the Documents panel shows them, with "Oct 2026 (latest)" on the newest.

| Step | Click, then Send | Say / show | Expect on screen |
|---|---|---|---|
| 1 | Chip 1 (tax) | Ledger Analyst reads exact figures, Policy agent searches the county bill and tax rules. Open "What the agents did": each read and search is a row with the agent's name. | Cited statements with an evidence tag. The October $3,200 county payment compared with the earlier ones in the ledger (Oct 2025 was $2,750.00, Oct 2024 $2,600.00). Policy sources cited. A "needs review" note about an unverifiable tax rule is possible (the offline script shows one). Exact wording varies with the model. |
| 2 | Chip 4 (groceries 2025) | A number comes from a recorded calculation, not from the model. | "$3,269.43", Somal's checking, 2025, with a citation. |
| 3 | Chip 5 (lease) | Policy agent answers from the lease document, not the ledger. | Unit B rent "$3,400.00" per month, and that the landlord pays the property tax directly to the county. Do not compare with the ledger's $3,100.00 rent deposits on camera; they are different sources. |
| 4 | Chip 8 (brokerage performance) | A second calculation type, with a deposits-versus-market split. | October 2026: opened $31,334.23, closed $31,045.73. Deposit $1,500.00, dividends $31.40, market effect -$1,819.90, which is -5.8 percent of the opening value. It should say "your brokerage account", never print the account id. |
| 5 | Chip 9 (concentration) | The answer checks the household's own written limit. | NVDA is 23.1 percent, over the 20 percent single-stock limit, with the policy cited. Top four holdings 54.9 percent. Technology 46.2 percent, under the 50 percent limit. |
| 6 | **Human in the loop, part 1.** Chip 2 (allocate surplus) | Say: "The system drafts a plan and then stops. It cannot approve its own plan, and nothing is executed." | "I drafted an allocation plan. It needs a household adult to approve it." A plan card with buckets (liquidity buffer, IRA, HSA, 529, brokerage) and Status: awaiting_approval, plus the Approve and Reject buttons. Dollar amounts depend on the planner; do not quote them. Open "What the agents did" to show the plan step. |
| 7 | **Part 2: the wrong person.** In the plan card, pick **viaan** and press Approve | Say: "Viaan is a dependent. The check is in code, not in the prompt: a dependent is refused and the plan keeps waiting." | "Refused: only a household admin or adult member may approve". Status stays awaiting_approval, the buttons stay active. |
| 8 | **Part 3: the right person.** Same card, pick **sayali**, Approve | Say: "Sayali is a household adult, so she can approve. A person made the decision, not the model." Then add: "Approval moves no money. It only records the decision; there is no execution path anywhere." | "Status: approved"; the buttons and the picker go grey. |
| 8b | **Part 4: the evidence.** Open the Audit trail panel (it refreshed itself) | Say: "Both attempts are on the record." Point at the two approve rows, then the Verify chain in step 16 covers them. | An `approve` row with decision `deny` and the reason "approver lacks approval rights" for viaan, then an `approve` row with decision `allow` and reason "approved" for sayali. |
| 9 | Chip 3 (transfer $5,000) | No external action, ever. | "Refused: the system only drafts and never moves money." |
| 10 | Chip 6 (viaan asks about Somal) | Refused before any model runs. | Page switches to viaan. "Refused: somal's private checking account is not available to viaan ... Dependents can't be given access to ledger data." Share panel says viaan owns no accounts. |
| 11 | Chip 7 (Somal asks about Sayali's card) | Consent is per account and per owner. | Page switches to somal. "Refused: sayali's private credit card is not shared with somal ... sayali can share it from Share my accounts". |
| 12 | Pick **sayali** in "You are". In Share my accounts choose `acct_sayali_card (private)` with somal, press Share (read only). | Read-only, purposes limited to spending, tax and briefing questions; can be revoked. Do not pick the joint rental account in the list; it is the first option. | "Shared. They can now ask about this account." and a "shared" row. |
| 13 | Chip 7 again, Send | Same question, now allowed. | Sayali's card at Safeway in September 2026: $417.62 (from the ledger; not seen live). |
| 14 | Pick **sayali**, press Revoke on the row; then chip 7 again | Revocation takes effect on the next question. | "Revoked. The next question is refused again." then the refusal from step 11. Skip this step if short on time. |
| 15 | Chip 10 (weather) | Scope check. | "That is outside what I can help with. I only answer questions about this household's money ..." |
| 16 | Audit trail panel: press **Verify chain** | Every read, refusal, approval and consent change above is a row. The chain is hash-linked. | "intact: N chained events". N depends on how many steps ran, so read it off the screen. |

Must-have if time is short: steps 1, 2, 4, 5, 6-8, 9-11, 16. Cut first: 3, 14, 15.

### Optional: bad uploads (about 45 seconds)

The Upload card lists five generated samples with an "expect" line; its Upload button also switches "You are" to the right member. Pick any of these:

- `2026_09_somal_checking_tampered.pdf`: stored as unreconciled and blocked from every calculation (balance_mismatch).
- `scanned_statement.pdf`: rejected, no text layer (OCR is not built).
- `2026_10_sayali_card.pdf` (uploaded as somal): refused, Somal may not write to Sayali's private account.
- `2026_09_somal_checking_injection.pdf`: reconciles; the Audit trail shows an "instruction-like text flagged" tag and nothing else happens.

Upload a sample only once per take (a repeat is a 409 duplicate). Uploads do not need the model.

## Human in the loop (say it clearly, about 30 seconds around steps 6-8b)

Name the idea out loud when you click chip 2: "This is the human-in-the-loop step." Then make these points, in this order:

1. **The model proposes, a person decides.** The planner only drafts. The graph pauses at an interrupt and the plan stays `awaiting_approval`; nothing continues until a person responds.
2. **Who may decide is enforced in code.** Only a household admin or adult can approve. A dependent is refused, the refusal is audited, and the plan keeps waiting.
3. **The decision is checked at the moment of approval.** The inputs are re-hashed when someone approves; if the ledger changed since the draft, the result is `stale`, not approved.
4. **Approval is not an action.** There is no code path that moves money, even after approval. Chip 3 (transfer $5,000) is refused outright.
5. **Everything is on the record.** Denied and allowed approvals are rows in the audit trail, and the paused plan survives a restart because its state is in a SQLite checkpoint.

One closing sentence, since only this pause is clickable in the page: "The same pause-and-resume mechanism also guards flagged results and reminder drafts; those are covered by tests and the `approval_gating` gate, not by this page." Press Reject on a fresh plan only if you have time left: it ends as `rejected`, also audited.

## Evaluation results (about 45 seconds)

Press **Run the checks** in the Evaluation results panel. It runs in about 1-2 seconds with no model and shows "16 of 16 checks pass · 420 scenarios", each gate in plain words with its target. Say what it is: the same offline scenario set the project is graded on (12 areas: routing, adversarial, authorization, allocation, ingestion, calculation, anomaly, retrieval, search count, privacy, brokerage performance, brokerage concentration), run with the keyword router. It checks the guardrails, not model quality, and the panel says so. Point at: unsafe actions 0, private data leaked 0, planted secrets leaked 0, calculation and brokerage figures match an independent calculation 1.0. The panel lists `membership_inference` as not covered.

Numbers to quote (re-checked 2026-11-18 offline):

- 834 tests pass (`uv run python -m pytest tests -q`).
- Offline eval: all 16 gates pass over 420 scenarios. One known scenario miss, `retr_005`, because hash embeddings rank the water-heater invoice below 5; retrieval recall at 5 is 0.9444 against a 0.90 target.
- Live router: `gemma-4-26b-a4b-it` routed 55 of 55 held-out requests correctly (accuracy 1.0, schema validity 1.0, no fallbacks). Source: Evaluation notes in [README.md](README.md).
- Everything is synthetic data.

Say plainly that only the router has been run on a live model. The allocation planner and critic personas, retrieval with real embeddings and answer quality are evaluated offline or with scripted models only (README, Known gaps).

## Strengths, limits and next steps

- **Strengths:** guardrails enforced in code (authorization, reconciliation, approval, no external action); full hash-chained audit trail; consent that can be granted and revoked; every figure from a recorded calculation; runs offline with fake models and with a hosted or local model.
- **Limits:** text-layer PDFs only, no OCR; one household and synthetic data; localhost only with no login; the SQLite file is not encrypted; allocation quality and answer quality are not evaluated on real models; a question about a statement that was never uploaded (for example the 2024 county bill as a ledger figure) makes the ledger analyst retry until its budget runs out before it says "no verified figures". Avoid it on camera or mention it as a known limit.
- **Next steps:** OCR for scanned statements, real authentication and encryption at rest, live evaluation of allocation and retrieval, an empty-result hint so the ledger analyst stops retrying.

## Not verified offline (needs the owner's live rehearsal)

Checked offline against the code and data: the chip list, wording and order; groceries 2025 = $3,269.43 (Somal's checking, ledger sum); lease rent $3,400.00 and the landlord-pays-tax text; county payments ($3,200.00 Oct 2026, $2,750.00 Oct 2025, $2,600.00 Oct 2024 ledger rows); brokerage and concentration figures; both refusal texts; the transfer and weather replies; the plan card, the viaan refusal and the sayali approval; the Share flow; the eval counts; the 834 passing tests.

Needs a live run, because the answer comes from the model:

- The wording and citations of chips 1, 4, 5, 8 and 9 on Gemma. In particular that chip 8 leads with the 31,334.23 to 31,045.73 result and does not print the account id, and that chip 1's comparison mentions the earlier county payments.
- Chip 7's $417.62 after Sayali shares (the ledger holds it; the model has not been seen stating it).
- That the allocation chip, planner and approval work with the live model (offline the planner draft appears).
- That answers are still instant after the second restart in pre-flight step 4.
- How long each uncached chip takes today (earlier runs: about 100 seconds on Gemma).
