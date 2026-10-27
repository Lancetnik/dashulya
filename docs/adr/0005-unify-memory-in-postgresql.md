# Unify chat memory in PostgreSQL with pgvector

Use one PostgreSQL database for imported and received originals, the group
registry, durable indexing revisions, lexical postings and embeddings. Both
memory tools and immediate reply context read the same canonical message rows.
This supersedes the SQLite storage/execution details in
[ADR 0004](0004-select-source-hybrid-memory.md), preserving its hybrid pipeline
and [ADR 0003](0003-bind-chat-memory-access.md)'s trusted access scope.

Separate runtime SQLite stores made archive coverage differ from immediate reply
context and required application-side vector scans. A shared async Psycopg pool
now owns connections; transactional upserts preserve chat/message identity,
live-edit precedence and revision-aware acknowledgements. Original messages
remain the evidence authority; search data is derived and content-addressed.

Store normalized `text-embedding-3-small` vectors in `vector(512)` and rank with
pgvector's native negative inner product. This preserves cosine-equivalent
ordering and the former zero-vector behavior. Start with exact SQL search:
authorization, trusted send-time cutoff and optional filters determine eligible
block keys before ordering and `LIMIT`. Add ANN only after measuring a need and
checking recall under those filters; pgvector documents possible underfilled
results from approximate scans with post-filtering.

Retain BM25 in SQL over token postings, with prefix matching and corpus statistics
computed only over the authorized snapshot. Keep the existing Russian query
heuristics, BM25 constants, bounded candidates, RRF and source diversity.
PostgreSQL `ts_rank` is not an equivalent replacement for FTS5 BM25. Unicode 6.1
tables derived from SQLite preserve document tokenization and case folding,
including unassigned characters and combining marks. SQLite FTS5 reference tests
and the local corpus comparison check compatibility. SQLite is an offline
oracle/input only.

Migration reads legacy SQLite files without modifying them, merges originals by
chat/message identity, copies cached vectors without provider calls, and checks
every canonical message against the source. Legacy fact/summary records and
source relationships are preserved separately; this does not add fact maintenance
to the agent. Stop the old writer for the final migration pass. Preserve its
files and configuration; reverting after PostgreSQL receives new messages also
requires transferring those new originals back to the old store.

Compose runs PostgreSQL 16.12 with pgvector 0.8.1 at a pinned multi-platform image
digest, persists its data on the host, and exposes it only to the internal
network. CI tests use disposable schemas in an isolated pgvector server.

References: [Psycopg async pools](https://www.psycopg.org/psycopg3/docs/advanced/pool.html),
[pgvector exact search and filtering](https://github.com/pgvector/pgvector/blob/v0.8.1/README.md),
[FTS5 BM25](https://sqlite.org/fts5.html#the_bm25_function).
