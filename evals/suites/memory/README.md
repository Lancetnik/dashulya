# Chat-memory implementation comparison

This suite compares isolated experimental memory implementations against
source-grounded questions. It does not select a replacement for the running
bot's backend. [ADR 0001](../../../docs/adr/0001-openviking-memory.md) continues
to govern production; the experiment addresses the replacement behavior in
[PRD #3](https://github.com/Lancetnik/dashulya/issues/3).

## Variants

| Variant | Implementation |
| --- | --- |
| `raw_lexical` | SQLite FTS5 over original messages and reply context; no model calls |
| `raw_hybrid` | The same original-message blocks with BM25 and cached dense embeddings combined by reciprocal rank fusion |
| `claims` | Automatic qualified-claim extraction, exact source quotes, correction/change/uncertainty reconciliation and lexical knowledge recall |
| `hindsight` | Pinned OSS Hindsight with LLM-extracted native world/experience facts and canonical source reconstruction; observations disabled |
| `hindsight_chunks` | The same OSS engine's native raw-chunk mode: embeddings, BM25 and temporal/semantic links; no LLM extraction or observations |

Every variant implements the same interface in [types.py](types.py). Preparation
receives only original messages, not benchmark questions, gold or selected
evidence. Search receives a question and trusted chat/time scope. Memory hits
identify their originating chat and original message IDs. The common harness
rejects foreign, future or missing sources, including collisions between local
message IDs in different chats.

The same [AG2 answerer](answerer.py), model, tool budget and history operations
apply to every variant. `search_memory` recalls knowledge and original evidence;
`chat_history` reads originals, resolves authors, paginates literal search and
counts the complete matching set. No Telegram or external-web tool is exposed.
Original content is untrusted evidence. Bot messages remain labeled and cannot
independently support factual claims.

The local AG2-Assistant checkout was consulted for construction of AG2 agents,
Responses configuration, typed outputs and stream usage. No code or runtime
state in that project was changed.

## Container-only execution

Run commands from the repository root. All infrastructure, extra packages and
model operations run in Docker. The evaluator image contains the existing AG2
dependencies, NumPy and the development checks. Its mounts expose source code,
the original-message file, input-only questions and experimental outputs. Gold
is added only to the grading command. Production memory paths and containers
are separate.

```sh
docker compose --project-directory . \
  -f deploy/memory-eval.compose.yaml build evaluator
```

The local dataset must exist separately under `evals/datasets/ostrikov/`; it is
ignored and absent from fresh clones. Before using it, read its local README
and the v2 annotation contract. The default corpus contains all 28,675 saved
group messages. The 49 authored questions use the completed snapshot; the one
historical question requires an earlier prefix. Preparation creates separate
indices/banks for those prefixes. Never reuse knowledge extracted from the full
snapshot for the earlier question.

An entirely local baseline needs no external model access:

```sh
docker compose --project-directory . -f deploy/memory-eval.compose.yaml \
  run --rm evaluator python -m evals.suites.memory.runner prepare \
  --variant raw_lexical
docker compose --project-directory . -f deploy/memory-eval.compose.yaml \
  run --rm evaluator python -m evals.suites.memory.runner recall \
  --variant raw_lexical --run-id raw-lexical-local-v1
```

Other preparation modes, generated answers and grading transmit source excerpts
to the configured model provider. Use them only after explicit authorization
for that corpus and destination. Credentials come from environment settings;
do not print `.env`, Compose's expanded configuration or provider error bodies.
The model defaults to the configured `OPENAI_MODEL`; override it with
`MEMORY_EVAL_MODEL`. `MEMORY_EVAL_JUDGE_MODEL` selects the grader and
`MEMORY_EVAL_EMBEDDING_MODEL` selects embeddings.

After authorization, prepare and run each candidate, using a distinct run ID:

```sh
docker compose --project-directory . -f deploy/memory-eval.compose.yaml \
  run --rm evaluator python -m evals.suites.memory.runner prepare \
  --variant raw_hybrid
docker compose --project-directory . -f deploy/memory-eval.compose.yaml \
  run --rm evaluator python -m evals.suites.memory.runner run \
  --variant raw_hybrid --run-id raw-hybrid-v1
```

Substitute `claims` and a new run ID for the accumulated-knowledge candidate.
Preparation caches completed work and resumes after interruption. Run manifests
record source/question/code hashes, AG2 and model versions; a changed answerer
or input requires a new run ID. A run may select a few IDs with `--only` for
diagnosis, but the final report requires every question and includes failures.

Hindsight uses an additional isolated PostgreSQL/pgvector deployment, without
host ports or host database installation:

```sh
docker compose --project-directory . \
  -f deploy/memory-eval.compose.yaml \
  -f deploy/memory-hindsight.compose.yaml up -d --wait hindsight-db hindsight
docker compose --project-directory . \
  -f deploy/memory-eval.compose.yaml \
  -f deploy/memory-hindsight.compose.yaml \
  run --rm evaluator python -m evals.suites.memory.runner prepare \
  --variant hindsight_chunks
```

The Compose file pins the official Hindsight image digest. Its adapter records
deterministic document/operation IDs and acknowledged completion before a bank
can be used. A source document's message-ID set is provenance for a passage,
not proof that every message in it semantically supports a returned fact; the
answerer and grader inspect canonical originals.

## Grading and the 70% target

Gold is never mounted for answering. Add it only for evaluation:

```sh
docker compose --project-directory . -f deploy/memory-eval.compose.yaml \
  run --rm \
  -v "$PWD/evals/datasets/ostrikov/v2/cases.jsonl:/gold/cases.jsonl:ro" \
  evaluator python -m evals.suites.memory.runner grade \
  --variant raw_hybrid --run-id raw-hybrid-v1
docker compose --project-directory . -f deploy/memory-eval.compose.yaml \
  run --rm \
  -v "$PWD/evals/datasets/ostrikov/v2/cases.jsonl:/gold/cases.jsonl:ro" \
  evaluator python -m evals.suites.memory.runner report \
  --variant raw_hybrid --run-id raw-hybrid-v1
```

[grading.py](grading.py) combines deterministic checks with a separate semantic
model judgment. A whole-answer pass requires every annotated criterion,
supporting original citations, correct scope/time, no material unsupported
claim or prohibited conclusion, and appropriate answerability. Citation IDs
must have been exposed to the answering model. Counts are independently
recomputed across the complete corpus predicate, including empty messages;
the model's structured count must match. An abstention on an answerable case
fails. Other genuinely supporting originals may be accepted because the gold
source annotations are not exhaustive.

`summary.json` reports complete-answer accuracy over all 50 cases and separately
over the 48 answerable cases, per-category results, missing cases, latency and
tool usage. The user-requested experiment target is at least **70% on both
complete sets**. The PRD's original 90% acceptance target remains a distinct
product target. A partial run or a retrieval-only score does not meet this goal.

Results remain development-set, model-graded evidence, not a held-out quality
claim or human certification. Save predictions, tool traces, criterion reasons,
usage and manifests under ignored `evals/runs/memory/`. Inspect disputed grades
against originals rather than silently changing expected answers or criteria.

## Current local evidence

The first offline lexical run found all labeled support in its initial top-eight
reply-aware hits for **34/46 factual retrieval cases (73.9%)**, and some labeled
source for 40/46. It covered 95/121 individual factual criteria. The denominator
excludes the two exact-count and two unavailable-answer cases, which require
different checks. These are source-retrieval diagnostics, **not generated-answer
accuracy**. The local artifact is
`evals/runs/memory/runs/raw-lexical-local-v1/recall-metrics.json`.

The user authorized GPT-6-Luna API runs on the saved corpus. The first completed
hybrid original-message run produced **32/50 complete answers (64%)**, including
30/48 answerable cases (62.5%). Both exact counts were correct but their initial
semantic grades incorrectly rejected the snapshot-coverage qualification:
the grader had not received trusted coverage metadata. Those original grades,
predictions and the summary are preserved separately before evaluation fixes.
This baseline does not meet the requested target.

The second shared answerer adds bounded neighboring originals, distinguishes
navigation context from reply evidence, and emphasizes exact names, qualifications
and complete answers. Every candidate completed all 50 questions without execution
errors. Latest complete model-grader passes for the current runner are:

| Candidate | All 50 | Answerable 48 |
| --- | --- | --- |
| `raw_hybrid` | 41/50 (82%) | 39/48 (81.25%) |
| `raw_lexical` | 35/50 (70%) | 33/48 (68.75%) |
| `claims` | 35/50 (70%) | 33/48 (68.75%) |
| `hindsight_chunks` | 33/50 (66%) | 31/48 (64.58%) |

Only hybrid original-message retrieval meets the numerical target on both
complete sets. The same saved hybrid predictions received 38/50 and 41/50 on two
full grader passes; lexical predictions received 33/50 and 35/50. Earlier grades
are preserved. These differences are grader variation, not changes to answers
or a per-case selection of favorable grades. Independent, blind source audits
inspect every mandatory facet separately; their findings and disagreements
remain distinct from model grades.

The full local [comparison](../../runs/memory/comparison.md) and
[machine-readable results](../../runs/memory/comparison.json) include run paths,
latencies, source audits, usage and limitations. They are ignored generated
artifacts, absent from fresh clones. Maintained claims use lexical recall in
this experiment, so its score does not isolate extraction from retrieval strategy.
Native Hindsight chunks use 1800-character fragments without LLM extraction or
observations. Its separate verbose-fact pilot covered only 4066 messages;
consolidation was cancelled and no full extracted-fact accuracy is claimed.

All native Hindsight source documents were verified against reconstructed
originals. The full raw-chunk prefix took 191.57 seconds to prepare; both native
preparation and native recall recorded zero LLM calls. Embeddings, common answer
generation and grading still use the authorized OpenAI APIs. Experiment services
were stopped and removed after completion; their PostgreSQL files, indices and
reports persist under `evals/runs/memory/`. Ollama and its model weights were
also removed. A production backend decision remains separate from this benchmark.

## Routine checks

```sh
docker run --rm -v "$PWD:/app" -w /app \
  -e PYTHONPATH=/app/src:/app -e UV_CACHE_DIR=/tmp/uv-cache \
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory \
  -e GIT_CONFIG_VALUE_0=/app dashulya-memory-eval:local just check
```

Routine tests use synthetic messages, injected embedding/extraction functions,
AG2 scripted configuration and mocked HTTP. They require no private corpus,
Telegram access or paid model calls. Run the complete checks after changes;
preserve their output alongside the evaluation artifacts. The final Docker run
passed 265 tests, mypy on 85 source files, lint, hygiene and all nine import contracts.
