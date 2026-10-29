# Evaluations

Version reusable runners, input contracts and scoring rules under `suites/`.
There are currently no versioned evaluation suites.

`datasets/` is entirely local and ignored, including dataset descriptions,
manifests, preparation scripts, cases and annotations. `runs/` contains generated
indices, exports and results and is also ignored. A fresh clone contains neither;
obtain the required data separately and read its local coverage/provenance notes
before evaluation. Never treat observed bot replies as verified expected answers.

Keep run outputs separate from the running agent's `var/` state. New evaluation
runners must have automated tests using temporary synthetic fixtures, without
private datasets, Telegram requests or paid model calls.
