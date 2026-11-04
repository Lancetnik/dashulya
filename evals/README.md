# Evaluations

Version reusable runners, input contracts and scoring rules under `suites/`.
The [memory suite](suites/memory/README.md) adapts source-backed chat questions to
AG2 Evals, with offline validation/export, production memory tools, trace checks
and a provisional semantic judge. Its `run` command is explicitly separate from
preparation and development checks.

The [addressing suite](suites/addressing/README.md) exports frozen regex candidates
without expert labels, collects Decisions probabilities for prompt variants,
and computes ROC AUC and threshold metrics. Its local dataset includes original
messages, reconstructed production context and expert addressing decisions.

`datasets/` is entirely local and ignored, including dataset descriptions,
manifests, preparation scripts, cases and annotations. `runs/` contains generated
indices, exports and results and is also ignored. A fresh clone contains neither;
obtain the required data separately and read its local coverage/provenance notes
before evaluation. Never treat observed bot replies as verified expected answers.

Keep run outputs separate from the running agent's `var/` state. New evaluation
runners must have automated tests using temporary synthetic fixtures, without
private datasets, Telegram requests or paid model calls.
