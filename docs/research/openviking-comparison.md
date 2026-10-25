# OpenViking comparison: pilot

## Outcome

Completed on October 25, 2026, after the user authorized OpenAI processing.
OpenViking found labeled supporting messages in 14 of 18 positive queries,
compared with 10 of 18 for FTS5. It improved paraphrase retrieval while returning
more source text and taking longer per search. This small development pilot
alone does not establish that it should replace FTS5 across the full corpus.

The server now runs in Docker Compose. The standalone Python server environment,
its cached package and temporary source checkout were removed; the lightweight
SDK remains a project dependency. After this comparison, the agent was changed
to require OpenViking; FTS5 is retained only as an offline baseline.

## Corpus and configuration

- Corpus: 379 original messages, 81 discussion/window extracts and one maintained
  summary, copied from the existing index without modifying it.
- Selection: last 300 messages plus existing message IDs in `[15060, 15140]`.
- Pilot index SHA-256:
  `97cb2f47f9cb9959c7a0889333bf957a69c72882589a63d63f1e5ef4aefb70db`.
- Server: OpenViking 0.5.2; official Python SDK: 0.1.16.
- Runtime for the final comparison: Docker Compose, official image
  `ghcr.io/volcengine/openviking:v0.5.2`, host port `127.0.0.1:1933`.
- Embedding: OpenAI `text-embedding-3-small`, 1,536 dimensions.
- Summaries: `gpt-5.6-luna`, selected by `OPENAI_MODEL` in this run.
- Retrieval: `find`, QUICK, all mapped L0/L1/L2 levels, no reranker.
- Both variants: up to 100 candidates, three returned conversations, identical
  SQLite evidence expansion, exact original-author/time filtering.

There are 21 manually labeled queries: eight keyword queries, eight paired
Russian paraphrases, two positive filters, two impossible filters and one absent
topic. They were written before any OpenViking retrieval run. Positive labels
are example source IDs, not exhaustive relevance judgments.

## Retrieval results

Measured locally with three sequential calls per query:

| Measure | FTS5 | OpenViking |
| --- | --- | --- |
| Keyword support hit@3 | 8/8 | 7/8 |
| Paraphrase support hit@3 | 0/8 | 5/8 |
| Positive filter support hit@3 | 2/2 | 2/2 |
| Impossible filters return empty | 2/2 | 2/2 |
| Absent topic returns empty | 1/1 | 0/1 |
| Overall positive support hit@3 | 10/18 (55.6%) | 14/18 (77.8%) |
| Overall positive evidence hit@3, including context | 10/18 (55.6%) | 16/18 (88.9%) |
| Mean reciprocal rank@3 | 0.528 | 0.639 |
| Median complete retrieval call | 0.71 ms | 294 ms |
| p95 complete retrieval call | 1.55 ms | 358 ms |

The keyword baseline requires every query token as a prefix. Its failure on
natural-language paraphrases is expected under that contract. The same agent
could improve it by rewriting questions into keywords; that behavior is outside
this retrieval-only experiment. Both adapters use the project's current async
memory interface. Timings include SQLite evidence reconstruction and, for
OpenViking, a real query embedding and the local container request. Quality
metrics use the first recorded result; the other calls measure latency. No
answer generation, follow-up memory expansion, reranking or relevance-threshold
tuning was performed.

OpenViking missed supporting records for `framework-keywords`,
`framework-paraphrase`, `ag2-repo-paraphrase` and `subscription-paraphrase`.
The first two were recovered in separately labeled context. Its absent-topic
query still returned three nearby candidates: this measures retrieval, not
whether the reply agent would invent an answer. Empty author/time filters
returned no results through both adapters.

The output budget was not equal: across the 21 recorded first results FTS5
returned 12 supporting excerpts and 37 context excerpts; OpenViking returned
97 and 143 respectively. OpenViking often selected multi-source discussion
records. Its additional recall therefore comes with more input for an answering
model. Compare a fixed source/token budget before drawing an efficiency claim.

Final raw results: `data/experiments/openviking/comparison.json`. The earlier
sync-interface run is retained in `comparison-sync.json`; its quality counts
were the same, but the table above uses the final async implementation.
Local corpus, mapping and Markdown resources live in
`data/experiments/openviking/export/`.

## Ingestion, citations and persistence

The initial import took 1,663 seconds, approximately 27 minutes 43 seconds.
Its 461 Markdown records became 461 original leaves and 580 directories:
OpenViking wraps each Markdown file in a directory before generating summaries.
Together with the root, the semantic queue processed 1,042 nodes. This import
layout is expensive; the elapsed time is not a general OpenViking throughput
benchmark. All 461 canonical memory records have a mapped original leaf.

The server reported this usage for the indexing session:

| Model | Calls | Input tokens | Output tokens |
| --- | --- | --- | --- |
| `gpt-5.6-luna` | 1,043 | 918,933 | 624,014 |
| `text-embedding-3-small` | 1,639 | 980,832 | 0 |

These are server counters, including startup context processing; billing was
not measured. Each complete comparison run made 57 query-embedding calls and
no summary-generation or reranker calls. Impossible filters made no embedding
request. The initial client's 20-minute wait timed out while its durable server
task continued; the final task completed successfully without another import.
The importer now allows a one-hour wait.

An independent SQLite audit checked 289 recorded source/context excerpts.
Message IDs, URLs, authors, timestamps, reply targets and exact text slices
matched the local originals. No supporting-source author/time filter violations
were found. The maintained index and source snapshot still match their recorded
checksums. This verifies citation integrity, not the semantic truth of generated
summaries or complete relevance judgments.

Compose startup and authentication were tested against an isolated workspace;
an incorrect key was rejected. The real container was then recreated with the
bind-mounted corpus: all 1,041 mapped resource paths and the server's 1,638
vectors survived. The container is healthy. The original native server was
stopped before opening the workspace in Docker.

Audit and operational evidence remain local in `audit.json`,
`import-coverage.json`, `ingestion-task.json`, `models-after-ingestion.json`,
`compose-smoke.json`, `compose-persistence.json` and `native-cleanup.json` under
`data/experiments/openviking/`.

## Reproduction and next decisions

Follow the [experiment guide](../../experiments/openviking/README.md) to import
and run the comparison. The importer checks mapped leaf coverage before the
backend can be selected. Model calls use the local server's configured provider;
data storage remains local.

After the pilot, use a larger held-out question set to assess answer quality,
source recall at a fixed output budget, absent-topic handling, the conversation
diversity rule and performance on the full corpus. Evaluate a cheaper import
layout before processing all 17,000 messages. The full corpus has not been
ingested into OpenViking in this experiment.
