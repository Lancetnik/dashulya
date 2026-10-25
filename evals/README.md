# Evaluations

Version reusable runners, input contracts and scoring rules under `suites/`.
The [OpenViking suite](suites/openviking/README.md) compares semantic recall with
the offline FTS5 baseline. Run it from the repository root.

`datasets/` is entirely local and ignored, including dataset descriptions,
manifests, preparation scripts, cases and annotations. `runs/` contains generated
indices, exports and results and is also ignored. A fresh clone contains neither;
obtain the required data separately and read its local coverage/provenance notes
before evaluation. Never treat observed bot replies as verified expected answers.

Keep run outputs separate from the running agent's `var/` state. Automated
tests in `tests/` exercise evaluation code using temporary synthetic fixtures,
without private datasets, Telegram requests or paid model calls.
