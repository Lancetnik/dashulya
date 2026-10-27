# Select hybrid recall over canonical original messages

Storage and execution details are superseded by [ADR 0005](0005-unify-memory-in-postgresql.md).
The hybrid ranking and source-evidence decision remains in effect.

Use BM25 and dense embeddings of original messages with up to two reply parents,
combine rankings with reciprocal rank fusion, and return source-diverse evidence.
Persist originals and content-addressed vectors locally. The application starts
without an OpenViking service; its existing adapter and workspace remain available
for offline maintenance. This supersedes the backend selection in
[ADR 0001](0001-openviking-memory.md); [ADR 0003](0003-bind-chat-memory-access.md)
governs access independently of ranking.

Original messages remain canonical evidence. This keeps attribution and citations
verifiable and lets a live correction replace stale imported text before query
filters run. It also avoids a separate generated-knowledge service at startup.

Use `text-embedding-3-small` with 512 dimensions and reply-aware blocks. Both
lexical and dense rankings consider at most 160 candidates; RRF uses 60 and
recall returns at most eight hits. Canonical authorization and chat selection
precede ranking. Returned citations and context come from current local
originals, never generated URLs. Exact scans and counts use `chat_history`.

Edits change block content keys, including dependent reply-parent blocks.
Unindexed revisions remain readable and lexically searchable; coverage exposes
pending blocks while the durable worker retries embeddings. This is the fixed
hybrid pipeline's explicit availability behavior. It does not silently select a
different engine. Index construction and CPU ranking run outside the event loop,
and cancellation retains resource ownership until SQLite work finishes.

This decision covers source-based recall. Maintained fact writing/correction,
explicit forgetting and source-deletion propagation, and complete chat migration
continuity remain work under [PRD #3](https://github.com/Lancetnik/dashulya/issues/3).
No imported snapshot, maintained legacy record or native workspace is deleted by
switching the application backend.
