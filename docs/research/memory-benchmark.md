# Memory benchmark: protocol and provisional evidence

This report records a development-set comparison of chat-memory implementations.
The matrix, generation repeats and production-profile verification are still in
progress. The provisional scores below do not select a final winner or establish
the product specification's 90% acceptance target.

The reusable runner is under [`evals/suites/memory/`](../../evals/suites/memory/README.md).
Private source messages, questions, annotations, predictions and detailed traces
remain outside version control. This report contains aggregate evidence only.

## Data and isolation

The immutable source snapshot contains 28,675 saved group messages. The benchmark
has 50 questions, 48 answerable cases and 125 annotated requirements.
It covers people, projects, reported tool experience, resources, discussion
outcomes, corrections, unresolved disagreement, exact counts and abstention.
Forty-nine questions were authored from evidence; one preserves a historical
request. This is not a sample of naturally occurring question frequencies.

Each question has a trusted chat identity and exclusive `(send time, message ID)`
cutoff. Separate full and earlier-prefix indices contain 28,675 and 4,066 original
messages respectively. Derived knowledge from the full snapshot is never reused
for the earlier question. Saved exports have coverage gaps and no edit-version
history; they do not establish that every saved text existed at its original
send-time cutoff. The single-chat snapshot cannot establish multi-chat access,
edit/deletion propagation or operational recovery correctness.

Preparation receives the complete eligible original prefix, with no questions,
gold answers or gold-selected source subset. Answering and reranking receive
questions, trusted scope and retrieved evidence; gold is mounted only for grading.
Indices and engine workspaces are isolated from the running bot. Corpus transmission
to the configured API was explicitly authorized; credentials are excluded from
reports and provider error bodies are not published.

## Frozen comparison

Protocol v3 freezes source, queries, annotations, code and configuration before
comparing the matrix. A changed answerer or input requires a new run identity.
Manifests retain package/model versions, preparation fingerprints, timings,
tool traces, token counters and original-source exposure. The model is
`gpt-6-luna`; dense vectors use `text-embedding-3-small` at 1,536 dimensions.
The common answerer uses AG2 1.2.1 and the Responses API. R reuses a
content-addressed vector cache, with SQLite FTS5/BM25 over the same reply-aware
original-message blocks; cache reuse does not admit later-prefix messages.

| Frozen input | SHA-256 |
| --- | --- |
| Source corpus | `597de61b42f982aab85c175952bac422de7d8891f3046b9fee417ef00bdacb0c` |
| Questions | `90186f328d2c21f6967bcc7586c4ce03f5e61b7d4a21fd601a821f08af57cb77` |
| Gold annotations | `dca56d0b1ac45841b555a696a8c8407e67c556d9ecc15417a561880714189cbc` |
| Comparison code | `185958b660689bd846544126eba5408d0b2cb7c97cadab610b6b89b23d150593` |
| Snapshot manifest | `08511f606b4bfcc652b16dd93721d2f4fda63740736c10e462adc108c5d15736` |

These hashes identify a comparison, not a claim that stochastic provider outputs
can be reproduced byte for byte. The production candidate has its own code and
run manifest and requires a separate complete run.

## Finite candidate space

The core matrix tests all 15 nonempty subsets of four components, both with and
without native Decisions reranking: 30 profiles. Ten additional controls make
40 profiles in total.

| Symbol | Component |
| --- | --- |
| R | Original-message lexical search plus cached dense embeddings, fused by reciprocal rank fusion |
| C | Automatically extracted, qualified claims with original quotes and lexical/dense recall |
| O | OpenViking 0.5.2 using the current historical export, native summaries and vector search |
| H | Hindsight 1.1.0 native **concise** world/experience fact extraction and recall; observations excluded |

The controls are raw lexical search with/without Decisions; lexical claims;
R plus lexical claims; native Hindsight chunks with/without Decisions;
R plus native Hindsight chunks with/without Decisions; and native Hindsight
**verbose** facts with/without Decisions. Chunk mode is not extracted knowledge
or consolidated memory.

Components supply bounded candidate pools. Combinations merge candidates by
validated original-source sets and use equal-weight reciprocal rank fusion
with constant 60, then apply source/conversation diversity. Candidate pool size
is at most 100; the usual final recall limit is eight, while the native
OpenViking-only profile preserves its five-hit production limit. Source IDs are
passage provenance, not automatic proof that every linked original supports
every returned claim.

The Decisions ablation calls OpenAI's native
[Decisions API](https://developers.openai.com/api/docs/guides/decisions), not a
chat-completion prompt that asks for a ranked list. The adapter submits bounded
candidate relevance-score questions with opaque IDs, validates the returned
schema and scores, then sorts deterministically. Retrieved text remains
untrusted evidence and cannot change trusted chat/time filters or source IDs.
The [native API schema](https://developers.openai.com/api/reference/resources/decisions/methods/create)
and [model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna)
describe the supported interface; the native client uses
[OpenAI Python 3.37.0](https://github.com/openai/openai-python/tree/v3.37.0/src/openai/resources/decisions).
Local manifests pin the SDK implementation.

## Answering and scoring

Every matrix profile uses the same common AG2 answerer, prompt, model, 12-call
tool budget and canonical-history interface. `search_memory` recalls evidence;
`chat_history` reads originals, resolves authors, paginates literal search and
counts the full matching set. Counts do not rely on retrieval top-k. No external
web or live Telegram history tool is exposed.

The harness verifies exact originating chat and cutoff before expanding any
source IDs, reconstructs canonical originals and reply context, and records what
the answerer actually saw. Bot responses may provide context but cannot independently
establish factual claims. Reranking does not bypass these checks.

Each saved set of 50 answers receives two independent grading passes using the
same judge model. These are repeated model judgments, not two different judge
models or human review. A whole-answer pass requires all mandatory content,
correct attribution and qualification, supporting original citations, correct
scope/time and no material unsupported addition. An abstention on an answerable
case fails. Exact count predicates are independently recomputed across the
complete eligible corpus, including empty records where the predicate requires
it. Alternative genuinely supporting originals can be accepted; gold relevance
annotations are not exhaustive.

Report complete-answer accuracy over all 50 cases and separately over the 48
answerable cases, plus criterion-level results, categories, latency, tool use
and token usage. Missing cases and execution failures remain in the denominator.
The experiment's 70% target applies to both complete denominators; it is distinct
from the product specification's 90% target. A retrieval-only or partial-run
score does not meet an answer-accuracy target.

Finalist selection and two additional answer-generation repeats are pending.
Two grades of one fixed answer set measure judging variation, not generation
variation. Final selection must also consider original-source audits, cost and
latency, followed by a complete run through the actual production candidate's
agent/provider/dialogue/tool path. The common matrix answerer and the actual
production profile must be reported separately.

## Provisional results

The following snapshot is provisional. Each number is whole-answer passes out
of all 50 cases, shown as judge pass 1 / judge pass 2. “Pending” means no complete
pair of grades is available; it is not a zero or an inferred OSS quality score.

| Core components | Without Decisions | With Decisions |
| --- | --- | --- |
| R | 41 / 40 | 39 / 36 |
| C | 37 / 38 | 39 / 36 |
| O | Pending | Pending |
| H | 36 / 37 | 38 / 39 |
| R+C | 35 / 37 | 40 / 40 |
| R+O | Pending | Pending |
| R+H | 40 / 40 | 30 / 34 |
| C+O | Pending | Pending |
| C+H | 37 / 40 | 38 / 39 |
| O+H | Pending | Pending |
| R+C+O | Pending | Pending |
| R+C+H | 37 / 36 | 41 / 42 |
| R+O+H | Pending | Pending |
| C+O+H | Pending | Pending |
| R+C+O+H | Pending | Pending |

| Additional control | Without Decisions | With Decisions |
| --- | --- | --- |
| Raw lexical | 41 / 38 | 34 / 31 |
| Lexical claims | 33 / 30 | Not scheduled |
| R + lexical claims | 39 / 37 | Not scheduled |
| Hindsight chunks | 27 / 29 | Pending |
| R + Hindsight chunks | Pending | Pending |
| Hindsight verbose facts | Pending | Pending |

For R, the corresponding answerable-case results are 39/48 and 38/48. Independent
agent audits inspected all 50 cases and all 125 requirements for R and R+C with
Decisions, checking canonical original support and exposure rather than treating
the judges as ground truth. They found respectively 45 supported passes, one
borderline case and four failures; and 43 supported passes, four borderline cases
and three failures. These audits are **not human review**, and they do not replace
or silently overwrite the saved model grades. Selection and generation repeats
remain unfinished.

## Native-engine preparation evidence and limits

Hindsight source audits verified exact canonical message-ID sets, document bytes,
metadata and source hashes for 4,096 documents across four fact banks
(concise/verbose × two time prefixes). This proves imported-source coverage,
not complete fact extraction: some documents produce no facts. Existing pilot
observations are excluded from recall. The separate native chunk experiment
uses chunk size 1,800 and no extraction LLM calls; its quality must not be presented
as consolidated-knowledge quality.

OpenViking uses the official 0.5.2 image and the production model, parser,
historical resource representation, summary prompts and vector backend. Isolated
preparation raises VLM concurrency from four to 128; answering must restore four.
Native recursive `ls` silently stopped at depth three, so experiment reconciliation
uses paginated native `tree` with sufficient depth to verify every original leaf.
The earlier prefix has 4,625 exported resources; the full prefix has 34,636.
An import completion flag or URI count alone does not establish healthy generated
summaries; generated abstracts/overviews also require content and failure-marker
audits before answers run.

The full import exposed a measured upstream completion-wait bottleneck:
[semantic completion](https://github.com/volcengine/OpenViking/blob/7f3ae1af822d936c1d0b070537772b0f1d59122f/openviking/storage/queuefs/semantic_work.py#L68)
waits on all embedding work IDs, with
[50ms polling](https://github.com/volcengine/OpenViking/blob/7f3ae1af822d936c1d0b070537772b0f1d59122f/openviking/service/task_tracker.py#L88), while
[the task store deep-copies records under a shared lock](https://github.com/volcengine/OpenViking/blob/7f3ae1af822d936c1d0b070537772b0f1d59122f/openviking/service/task_store.py#L683).
A native synthetic 117,000-record test took 1.48s per copy. A nonblocking 637-sample
live profile attributed 81.5% of samples to this copy path. Embedding drain slowed
to about two records/s after summaries finished. Paired copied-SQLite tests did
not support database auto-vacuum as the cause.

A proposed preparation-only workaround increases the existing waiter's polling
interval, preserving source bytes, summaries, model settings, vectors and recall.
Its application, resulting throughput and restoration must be recorded before
claiming a completed O result. Such a run must be labeled as the native retrieval
baseline with a **poll-only preparation workaround**, not an unmodified end-to-end
native timing measurement. Production services have not been modified by these
experiments.
