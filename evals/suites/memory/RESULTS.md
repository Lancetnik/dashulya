# Memory benchmark: comparison and production validation

This suite record preserves the comparison protocol and aggregate development-set
evidence for chat-memory implementations.
The owner selected original-message hybrid recall: BM25 plus embeddings.
The comparison, confirmation generations and actual application runs below are
separate evidence. The actual application averages 52% in the final full run,
below both the experiment's 70% and product specification's 90% targets. The
implementation is reviewable, with deployment pending further quality work.

The reusable runner is in [this suite](README.md).
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
`gpt-6-luna`; embeddings use `text-embedding-3-small`. R and C use 512
dimensions; native engines preserve their recorded preparation configuration.
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
Local manifests pin the SDK implementation. A separate audit of 172 native
reranking calls across R, C and R+C recorded successful native responses with
no fallback or refusal: 8,888,298 input tokens and zero reported output tokens.
Those native counters are separate from AG2 answer usage; they do not imply
zero cost or include preparation, answer generation or judging.

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

Two additional answer generations were completed for R and R+C+H with
Decisions. Two grades of one fixed answer set measure judging variation, not
generation variation. Every repeat preserves the frozen inputs and reports all
50 cases; no per-case best answer or grade is selected. The common matrix
answerer and actual production profile are reported separately.

## Frozen comparison results

Every one of the 40 profiles completed all 50 cases and two grading passes.
Numbers below are whole-answer passes out of 50, shown as judge 1 / judge 2.

| Core components | Without Decisions | With Decisions |
| --- | --- | --- |
| R | 41 / 40 | 39 / 36 |
| C | 37 / 38 | 39 / 36 |
| O | 35 / 32 | 34 / 37 |
| H | 36 / 37 | 38 / 39 |
| R+C | 35 / 37 | 40 / 40 |
| R+O | 36 / 37 | 40 / 39 |
| R+H | 40 / 40 | 30 / 34 |
| C+O | 34 / 33 | 39 / 40 |
| C+H | 37 / 40 | 38 / 39 |
| O+H | 39 / 36 | 33 / 38 |
| R+C+O | 38 / 38 | 38 / 39 |
| R+C+H | 37 / 36 | 41 / 42 |
| R+O+H | 36 / 37 | 40 / 40 |
| C+O+H | 38 / 39 | 36 / 39 |
| R+C+O+H | 34 / 34 | 36 / 38 |

| Additional control | Without Decisions | With Decisions |
| --- | --- | --- |
| Raw lexical | 41 / 38 | 34 / 31 |
| Lexical claims | 33 / 30 | Not scheduled |
| R + lexical claims | 39 / 37 | Not scheduled |
| Hindsight chunks | 27 / 29 | 28 / 31 |
| R + Hindsight chunks | 36 / 39 | 37 / 36 |
| Hindsight verbose facts | 33 / 36 | 37 / 37 |

For the initial R generation, the answerable-case results are 39/48 and 38/48. Independent
agent audits inspected all 50 cases and all 125 requirements for R and R+C with
Decisions, checking canonical original support and exposure rather than treating
the judges as ground truth. They found respectively 45 supported passes, one
borderline case and four failures; and 43 supported passes, four borderline cases
and three failures. These audits are **not human review**, and they do not replace
or silently overwrite the saved model grades. The R+C+H+Decisions audit
separately found 41 supported passes, three borderline cases and six failures.
All three audits inspected source attribution, exposure, cutoff and scope.

## Confirmation generations and selection

| Component profile | Generation 1 | Generation 2 | Generation 3 | Mean of all six grades |
| --- | --- | --- | --- | --- |
| R | 41 / 40 | 37 / 38 | 38 / 37 | 231/300 (77.0%) |
| R+C+H+Decisions | 41 / 42 | 39 / 37 | 37 / 39 | 235/300 (78.3%) |

The difference is less than one question per generation and does not establish
statistical superiority. The owner explicitly selected R for the application:
local canonical originals, content-addressed vectors, fewer services and direct
handling of edits. This choice is recorded in [ADR 0004](../../../docs/adr/0004-select-source-hybrid-memory.md).
The production candidate requires its own validation; the 77% number belongs to
the common comparison answerer, not the deployed bot.

## Actual application validation

The replacement application was evaluated through the real `AgentProvider`,
`Dialogue`, `HybridChatMemory`, `search_memory` and `chat_history`, using the
production persona, model configuration, original request and observed reply
context. No evaluation answer prompt was substituted. Trusted experiment-local
chat/bot IDs are explicitly synthetic. The historical request uses only its
eligible prefix. Telegram delivery and the access/addressing classifier are not
invoked by this offline answer evaluation.

The current historical OpenViking application path was also run on all 50 cases
with its original factory/prompt/tool path, separately from the common O profile.
The only evaluation normalizer extracts count/abstention metadata from the already
saved answer; it sees no sources or gold and requires verbatim answer witnesses.
It cannot change reply text, citations or factual content.

| Actual application profile | Judge 1 | Judge 2 | Execution failures / 50 |
| --- | --- | --- | --- |
| Previous OpenViking production path | 15/50 | 13/50 | 0 |
| Hybrid integration v1, earlier prompt | 11/50 | 11/50 | 0 |
| Hybrid integration v2, fuller grounding instructions | 25/50 | 23/50 | 0 |
| Hybrid integration v3, medium reasoning | 31/50 | 28/50 | 0 |
| Hybrid integration v4, initial recall | 27/50 | Not run | 1 |
| **Selected implementation v5, reviewed timeout/date-boundary fixes** | **25/50** | **27/50** | **0** |

V5's mean is **52%**, with answerable-case passes 23/48 and 26/48. Median answer
time was 19.71s and p95 was 57.50s; its 50 answers made 101 memory searches
(including initial recall) and 39 direct-history calls. These are one generation
with two judge passes. They do not establish generation-level stability.
The common R answerer's 77% must not be presented as this application's score.
The actual application **does not meet the experiment's 70% target**, nor the
PRD's 90% target. Deployment and acceptance remain pending quality work.

V4 preserved a stale `automatic_initial_search=false` manifest field despite
observed recall in all 50 traces; v5 fixes this field without rewriting earlier
artifacts. Its one timed-out answer remains a failure in the denominator. The
old report crashed on that failure's missing timing field; 27/50 is the aggregate
of 49 saved semantic grades plus that execution failure, not a complete official
report or a successful-run subset. The final evaluator records failure timing
and reports absent legacy timings explicitly; it does not replace failed answers.
The v5 source/answering/grading snapshot is preserved before this report-only fix.

A separate grade-blind agent audit read all v4 raw replies and all 125 criteria:
30 passes, four borderline cases and 16 failures; criterion results were 102
supported, five borderline and 18 failed. It checked 6,715 exposed original
objects with no canonical-text, scope, cutoff or citation-exposure mismatch,
and independently verified both exact counts. This is an agent audit, not human
certification. The principal application gap is answer completeness and grounding:
omitted qualifications, guessed model/unit expansions and reversed participant
roles. It is not evidence that retrieved original text or chat authority may be
relaxed. All saved intermediate runs remain available; no per-case result or
favorable generation was selected for the final score.

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

With explicit user approval, the active preparation waiter's polling interval
was changed from 50ms to 10s. Source bytes, summary content, model settings,
vectors and recall were unchanged. Embedding drain increased from about 1.8 to
24 records/s; all 132,584 native embedding jobs completed without native
requeues/errors. Native preparation telemetry across the full and earlier
prefixes records 72,702,271 LLM input tokens, 38,163,979 LLM output tokens and
49,532,959 embedding tokens. These imports are shared by all O profiles; do not
charge them once per profile or conflate native tokens with AG2 answer/judge
usage. Monetary cost was not inferred from incomplete cross-provider counters.
Both prefix imports completed and all canonical leaves were
reconciled. Audits parsed all 41,342 generated abstracts and 41,342 overviews
without empty content, decoding errors or native fallback markers. Eight
overviews contained unresolved sample-navigation links; these are not used as
canonical citations. Answering restored concurrency four and restarted the
isolated service, removing the waiter-local intervention.

O results are a native retrieval baseline with a **poll-only preparation
workaround**, not an unmodified end-to-end native timing measurement. Full-import
telemetry reports 30,989 processing seconds but 8,681 monotonic duration seconds
across a host sleep gap; these definitions must not be conflated. The running
production services were not modified by these experiments.
