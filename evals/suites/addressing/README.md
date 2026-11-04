# Addressing classifier evaluation

Compare classifier prompts using frozen regex candidates and source context.
Preparation and scoring are offline. Only the explicit `run` command calls the
paid Decisions API; it reads exported model inputs, never expert labels.

Private datasets live under ignored `evals/datasets/addressing/`. Read their
README for provenance, coverage, annotation policy and split limitations.
Generated inputs, predictions and reports belong under `evals/runs/`.

## Dataset contract

- `inputs.jsonl`: exactly `case_id` and `input`, where `input` is the production
  classifier payload, including the current message and recent conversation.
- `cases.jsonl`: the same IDs, `partition` (`calibration` or `test`),
  `metric_eligible`, and evaluator-only `gold.should_reply` boolean with a
  rationale and confidence. Runtime decisions are separate observations.
- `manifest.json`: `artifacts` maps relative artifact paths to SHA-256 hashes.
  Loading a dataset verifies these hashes and matching unique case/input IDs.

The default `core` panel includes only `metric_eligible` cases. `--panel all`
also includes supplemental cases, such as ambiguous annotations or direct
mention/reply routes that bypass the classifier. Report that panel separately.

## Prepare inputs

```sh
uv run python -m evals.suites.addressing validate evals/datasets/addressing/20261103-v1
uv run python -m evals.suites.addressing export evals/datasets/addressing/20261103-v1 \
  --partition calibration --out evals/runs/addressing/prompt-a/calibration-inputs
uv run python -m evals.suites.addressing export evals/datasets/addressing/20261103-v1 \
  --partition test --out evals/runs/addressing/prompt-a/test-inputs
```

Output directories must be new. Preparation never overwrites a dataset or run.

## Collect probabilities

Run each prompt on the same exported calibration inputs. Supply its addressing
workflow as a UTF-8 text file; shared provenance, timezone and message-format
instructions stay equal across variants. Omitting `--prompt` uses the current
production addressing prompt. `OPENAI_API_KEY` is read from the environment or
root `.env`; model selection defaults to `OPENAI_DECISION_MODEL`.

```sh
uv run python -m evals.suites.addressing run \
  --inputs evals/runs/addressing/prompt-a/calibration-inputs/inputs.jsonl \
  --prompt evals/datasets/addressing/20261103-v1/prompts/recall-first.txt \
  --out evals/runs/addressing/prompt-a/calibration-run
```

Each classification uses a fresh AG2 stream and the production decision model
configuration: ten-second provider timeout, no retries, twelve-second outer
deadline. The run preserves the raw Decisions predicate `probability` from
response metadata, before AG2 converts it to a boolean at threshold 0.5. This
is a provider probability, not a self-reported confidence or a thresholded label.

`predictions.jsonl` contains `case_id`, `probability`, resolved model, usage and
duration. API failures have a null probability and redacted exception type.
`run.json` records model, timezone, prompt/input hashes and completeness.
No failed result is silently replaced with zero or removed from scoring.

## Select and verify a threshold

ROC AUC measures ranking across thresholds. Select an operating threshold on
calibration data according to the desired recall/false-positive tradeoff, then
evaluate the fixed prompt and threshold on the test partition.

```sh
uv run python -m evals.suites.addressing score evals/datasets/addressing/20261103-v1 \
  --partition calibration \
  --predictions evals/runs/addressing/prompt-a/calibration-run/predictions.jsonl \
  --min-recall 0.98 --out evals/runs/addressing/prompt-a/calibration-metrics
```

The scorer emits ROC AUC, a threshold sweep, confusion counts, recall,
precision, FPR, specificity, accuracy and F1. Equal probabilities move together;
the decision rule is `probability >= threshold`, matching AG2. AUC is null when
the selected partition contains only one class. Missing, duplicate, nonfinite,
out-of-range, boolean or failed probability results reject the score operation.

On `calibration` only, `recommended_threshold` minimizes false positives among
observed thresholds meeting `--min-recall` (default 0.98), choosing the largest
threshold when tied. Test scoring never recommends a threshold. Run the fixed
prompt on test inputs, then pass the chosen numeric value with `--threshold`.
Repeated test inspection turns that partition into development data.

The core offline boundaries are `export_inputs()` and `score_predictions()` in
`benchmark.py`, covered with synthetic fixtures and deliberate fault checks.
Routine tests make no Telegram, PostgreSQL or model requests.
