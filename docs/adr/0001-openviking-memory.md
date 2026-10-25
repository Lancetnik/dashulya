# Use OpenViking for agent memory with canonical SQLite evidence

Accepted on October 25, 2026. Use OpenViking as the sole agent memory backend,
accessed through its official Python SDK and run locally in Docker Compose.
Keep original messages and citation metadata in SQLite, reconstruct supporting
evidence locally, and expose one `search_memory` tool across originals and
derived records. Retain FTS5 for offline comparison. Startup requires a healthy
memory service and a matching imported corpus; there is no silent retrieval
fallback. Live messages use durable local persistence before asynchronous
OpenViking indexing.

The development comparison favored semantic recall: on 18 positive queries
over 379 original messages and 82 derived records, OpenViking found labeled
support in 14 queries versus 10 for FTS5. The result supports trying semantic
memory; it does not establish general superiority. The cases were manually
selected, output text budgets differed, and the full corpus was not evaluated.
The initial Markdown import took approximately 27 minutes 43 seconds because
the importer generated summaries for many wrapped directories. This cost is
why live ingestion uses bounded resource batches instead of repeating imports.

The decision trades a second local process, model calls and persisted vector
state for semantic recall and generated summaries. SQLite remains authoritative
for identities, original timestamps, authors, text and Telegram links; generated
summaries and URLs are not sufficient citation evidence. OpenViking is a resource
index here, not personal-preference extraction. The decision records the current
prototype and may be superseded by implementation of the
[replacement memory specification](https://github.com/Lancetnik/dashulya/issues/3).

The inspected implementation used OpenViking 0.5.2 and SDK 0.1.16. Upstream:
[repository](https://github.com/volcengine/OpenViking),
[content operations](https://docs.openviking.ai/en/api/12-content).
