# Eval baselines

Named snapshots of eval runs. Each one is the evidence behind a claim in README "Evaluation notes" or ARCHITECTURE section 11, so a later change to a model, prompt or scorer can be compared against a known result instead of a remembered one. All data is synthetic.

## Files

| File | What it is |
|---|---|
| `2026-11-19-offline-evals_0.5.json` | Offline run, all 16 gates, written by `uv run python -m household_agent.evals --out <file>`. |
| `2026-11-19-live-routing-gemma-4-26b-a4b-it.json` | Live routing replies from `gemini:gemma-4-26b-a4b-it` (55/55, recorded 2026-11-18), re-scored on `evals_0.4` from the saved checkpoint with 0 new model calls. Only the routing gates are meaningful; the other areas ran offline in the same report. |
| `2026-11-19-live-retrieval-nomic-embed-text.json` | Retrieval area (26 scenarios incl. the 4 negative controls) with real local embeddings, `--only-area retrieval --embed-model ollama:nomic-embed-text`, 1.5 s, no hosted-model quota. Only the retrieval gates are meaningful. |

Compare a new run with the saved one by diffing the `gates` and `metrics` blocks.

## What a snapshot pins

Every JSON report carries a `run` block with everything the numbers depend on:

| Field | Meaning |
|---|---|
| `version` | Scenario and scorer version (`EVAL_VERSION`, now `evals_0.5`; the live routing and allocation snapshots below were scored under `evals_0.4`, which has the same scenarios in those areas). A new version means the numbers are not comparable. |
| `seed` | Scenario generator seed (default 0). |
| `as_of` | The fixed demo clock (`DEMO_AS_OF`), so dates never drift. |
| `router_model`, `planner_model`, `embed_model` | `keyword rules` / `template` / `hash` offline, or the `--model`, `--planner-model`, `--embed-model` specs of a live run. |
| `only_area`, `allocation_scenarios` | Set when a run covered fewer scenarios; the other gates are then vacuous. |

A live routing run also records `live_routing` (checkpoint file, model, reply count, unusable replies, failed calls that were never counted).

## Decisions these snapshots back

- **All 16 gates pass offline** with the keyword router, the template planner and hash embeddings. The only scenario miss is `retr_005` (hash embeddings rank the water-heater invoice below 5), and `retrieval_recall_at_5` is 0.9444 against a 0.90 gate.
- **Router model: `gemini:gemma-4-26b-a4b-it`.** Live routing run (`--only-area routing --resume`): 55 of 55 replies, 0 failed calls, one sitting of 493 s. `routing_accuracy` 1.0000, `schema_validity` 1.0000, no router fallbacks, all 16 gates pass. The checkpoint file stays in git-ignored `.cache/`; the live-routing JSON above is its re-scored report. Retrieval and allocation have not been run on a live model.
- **Embeddings: `ollama:nomic-embed-text`.** `retrieval_recall_at_5` 1.0000 (hash embeddings: 0.9444), `citation_precision` 1.0000, `negative_control_pass` 1.0, `fabricated_citations_missed` 0, no scenario failures. `retr_005`, the one offline miss, passes, so that miss came from the hash embeddings and not from retrieval logic or the labels.

## Adding a snapshot

1. Run the eval with the flags you want to record and `--out evals/baselines/<YYYY-MM-DD>-<what>.json`.
2. Check the `run` block shows the models and seed you meant.
3. Add a row to Files and, if the run backs a decision, a line under Decisions.
