# Eval baselines

Named snapshots of eval runs. Each one is the evidence behind a claim in README "Evaluation notes" or ARCHITECTURE section 11, so a later change to a model, prompt or scorer can be compared against a known result instead of a remembered one. All data is synthetic.

## Files

| File | What it is |
|---|---|
| `2026-11-19-offline-evals_0.4.json` | Offline run, all 16 gates, written by `uv run python -m household_agent.evals --out <file>`. |

Compare a new run with the saved one by diffing the `gates` and `metrics` blocks.

## What a snapshot pins

Every JSON report carries a `run` block with everything the numbers depend on:

| Field | Meaning |
|---|---|
| `version` | Scenario and scorer version (`EVAL_VERSION`, now `evals_0.4`). A new version means the numbers are not comparable. |
| `seed` | Scenario generator seed (default 0). |
| `as_of` | The fixed demo clock (`DEMO_AS_OF`), so dates never drift. |
| `router_model`, `planner_model`, `embed_model` | `keyword rules` / `template` / `hash` offline, or the `--model`, `--planner-model`, `--embed-model` specs of a live run. |
| `only_area`, `allocation_scenarios` | Set when a run covered fewer scenarios; the other gates are then vacuous. |

A live routing run also records `live_routing` (checkpoint file, model, reply count, unusable replies, failed calls that were never counted).

## Decisions these snapshots back

- **All 16 gates pass offline** with the keyword router, the template planner and hash embeddings. The only scenario miss is `retr_005` (hash embeddings rank the water-heater invoice below 5), and `retrieval_recall_at_5` is 0.9444 against a 0.90 gate.
- **Router model: `gemini:gemma-4-26b-a4b-it`.** Live routing run (`--only-area routing --resume`): 55 of 55 replies, 0 failed calls, one sitting of 493 s. `routing_accuracy` 1.0000, `schema_validity` 1.0000, no router fallbacks, all 16 gates pass. This run's checkpoint file lived in git-ignored `.cache/`, so the figures are recorded here and in README "Evaluation notes" rather than as a JSON snapshot.

## Adding a snapshot

1. Run the eval with the flags you want to record and `--out evals/baselines/<YYYY-MM-DD>-<what>.json`.
2. Check the `run` block shows the models and seed you meant.
3. Add a row to Files and, if the run backs a decision, a line under Decisions.
