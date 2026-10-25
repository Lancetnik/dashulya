# OpenViking as a memory backend

Reviewed on 25 October 2026 against OpenViking commit
[`4ab22a4923452362b60aef2826c2083ce36168d2`](https://github.com/volcengine/OpenViking/tree/4ab22a4923452362b60aef2826c2083ce36168d2).
This is a source and documentation review: no server, model, ingestion job, or
quality benchmark was run. Dashulya's requirements are recorded in
[the memory design](../design/chat-memory.md).

## Fit

OpenViking is a plausible backend for derived context, embeddings, and retrieval.
Its directory summaries and original resources fit our idea of searching memory
and resolving it to original evidence through one `search_memory` tool. It does
not supply Dashulya's Telegram source graph or exact message filters as a drop-in
replacement. Keep SQLite as the canonical message archive and evaluate OpenViking
as a separate service before changing the production retrieval path.

## What retrieval actually does

The current implementation performs **global vector retrieval**, despite the
`HierarchicalRetriever` class name. It combines target URI scope, access scope,
metadata filters, and an optional level restriction before retrieving candidates.
Without a level restriction, L0, L1, and L2 can all enter the candidate set directly;
original leaves do not have to pass a parent-summary gate. Current documentation
explicitly says the former directory traversal and parent-child score propagation
were removed. Do not infer the current algorithm from older hierarchical-search
descriptions. [Retriever source](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/openviking/retrieve/hierarchical_retriever.py#L109),
[retrieval documentation](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/docs/en/concepts/07-retrieval.md).

| API | Current behavior |
| --- | --- |
| `find()` | One supplied query in QUICK mode; no session intent analysis or reranking. |
| `search()` | Can plan multiple queries using the supplied query and optional session context; uses THINKING mode when a reranker is available. |

The `limit` applies per planned query, so a combined `search()` result can exceed
it. The retriever removes duplicate URIs within one recall, keeping the highest
vector score. This does not establish cross-query deduplication, content-based
deduplication, or consolidation of a summary and its source messages. Our adapter
would still normalize results by canonical message identity and preserve distinct
messages with identical text. [API semantics](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/docs/en/concepts/07-retrieval.md),
[candidate handling](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/openviking/retrieve/hierarchical_retriever.py#L209).

With reranking enabled, recall requests twice the requested limit and reranks
candidate **abstracts**, rather than automatically reading every complete source
file. The implementation can fall back to vector scores if reranking fails or
returns unusable results. Reranking therefore cannot recover evidence absent from
the initial candidates. [Reranking implementation](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/openviking/retrieve/hierarchical_retriever.py#L251).

Configured providers include VikingDB, Cohere, LiteLLM, Jev, and an `openai`
provider. The latter is an adapter for third-party OpenAI-compatible rerank
services, including DashScope: it posts a query/documents request to a configured
endpoint. It is **not an OpenAI Decisions API integration**, and the inspected
source contains no native Decisions adapter. Using Decisions would require a
separate adapter or an application-level reranking step; the provider name does
not imply an official OpenAI rerank endpoint.
[Provider configuration](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/openviking_cli/utils/config/rerank_config.py),
[specific adapter](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/openviking/models/rerank/openai_rerank.py#L1).

## Summaries and original evidence

L0 `.abstract.md` and L1 `.overview.md` are directory sidecars; L2 contains detailed
resources. They are not automatically one summary per Telegram message. Directory
summary generation samples direct children when there are more than 32 by default,
and parent refreshes can lag changes under the documented freshness policy. Use
bounded thread or time-window directories and test summary coverage. Index original
message resources as well, so L2 remains directly eligible for retrieval.
[Context layers and generation](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/docs/en/concepts/03-context-layers.md).

Unknown fields in sidecar metadata are silently dropped. Store Telegram IDs,
original timestamps, speakers, reply relationships, and source URLs in resource
bodies and an explicit external mapping, rather than assuming custom sidecar
frontmatter survives. `include_provenance` exposes retrieval queries, searched
directories, and matched URIs; it does not create a Telegram message citation graph.
[Sidecar metadata](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/openviking/storage/abstract_overview.py#L95),
[result provenance](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/openviking_cli/retrieve/types.py#L346).

Session `commit()` archives messages to `messages.jsonl` synchronously, then runs
summary generation and memory extraction in the background. Extracted memories can
be edited, merged, or deleted; extraction provenance records the session/archive
and extraction run. Neither this archive operation nor memory extraction proves
that every original Telegram message is independently searchable. Do not use a
whole multi-speaker group as one owner's personal-preference session: import it as
resources, retain speaker identity, and keep canonical evidence outside mutable
derived memory. [Session lifecycle](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/docs/en/concepts/08-session.md),
[memory operation provenance](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/openviking/session/memory/dataclass.py#L158).

## Exact chat, author, and date constraints

The HTTP API supports scoped URIs, generic filter expressions, tags, and time
filters over `created_at` or `updated_at`. The standard collection schema has no
Telegram author ID or original `sent_at` field. Context timestamps default to the
processing time unless explicitly supplied, so a date filter must not silently
interpret them as message send time.
[Search request/filter model](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/openviking/server/routers/search.py#L67),
[collection schema](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/openviking/storage/collection_schemas.py#L80),
[timestamp defaults](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/openviking/core/context.py#L88).

For Dashulya, scope chats by resource subtree and implement an explicit importer
mapping for authors and original times. Verify whether that representation permits
the required constraints before candidate selection. Checking SQLite after a
top-k search preserves answer correctness, but can lose recall if unrelated
authors or dates occupied the candidate slots. Preserve the existing exact
retrieval path inside the unified tool until this issue is demonstrated resolved.

## Proposed experiment

Keep the public AG2 interface as one `search_memory` tool. Its backend can call
OpenViking `find`/`search`, read selected resources, and expand original evidence
from SQLite; no additional AG2 tool per context layer is necessary.

Use an owner-scoped resource tree such as:

```text
viking://user/<owner>/resources/telegram/fastnewsdev_chat/
  <thread-or-window>/<original-message>.md
```

Each imported original should retain a deterministic mapping to `(chat_id,
message_id)`, original sender/time, text, and Telegram URL. Derived summaries must
resolve to these originals. Keep experiment artifacts under ignored
`data/experiments/openviking/`, leaving the imported snapshot and maintained memory
index intact. This is a proposed design, not an implemented integration.

Compare the same Russian queries and known source messages against current FTS5:
source-message recall, exact author/time filtering, citation fidelity, facts lost
in summaries, cross-query duplicates, incremental update/deletion behavior,
latency, and indexing/model usage. No retrieval-quality or performance conclusion
is justified by this source review alone.

## Deployment and licensing

At the reviewed commit the server requires Python 3.10+, advertises classifiers
through 3.14, and has native build requirements including CMake and Maturin. The
separate HTTP Python SDK requires Python 3.8+ and only `httpx` as a runtime
dependency. A separate server plus HTTP SDK is the clearest experiment boundary
for Dashulya's Python 3.12–3.14 environment; declared compatibility still needs
verification on our platform. Server setup requires embedding and VLM
configuration; reranking is optional. This review establishes no minimum RAM,
throughput, provider cost, or local-model performance.
[Server packaging](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/pyproject.toml),
[HTTP SDK packaging](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/sdk/python/pyproject.toml),
[configuration](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/docs/en/guides/01-configuration.md).

The inspected `LICENSE` and server package metadata specify **AGPL-3.0** at this
commit. [License](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/LICENSE),
[package metadata](https://github.com/volcengine/OpenViking/blob/4ab22a4923452362b60aef2826c2083ce36168d2/pyproject.toml#L19).
