# Chat memory

The agent requires OpenViking. It exposes one read tool, `search_memory`, across
original messages, discussion extracts, summaries and facts. Every derived
record links to original messages, which remain directly retrievable. SQLite
preserves canonical evidence; it is not an alternate agent retrieval backend.

```mermaid
flowchart LR
    Query[Query and original-message filters] --> Index[OpenViking semantic recall]
    Index --> Layers[L0 abstracts / L1 overviews / L2 originals]
    Layers --> Map[Verified URI to memory ID map]
    Map --> SQLite[Canonical messages and source links]
    SQLite --> Evidence[Source excerpts / reply context / Telegram links]
```

## Local data and coverage

The immutable source snapshot is `data/fastnewsdev/snapshot.sqlite3`. The full
maintained evidence index, `data/memory.sqlite3`, contains 17,000 messages and
one manually reviewed example summary citing message 15100. Creating an index
does not automatically recreate maintained summaries.

The current OpenViking import is a separate pilot: 379 original messages, 81
discussion/window extracts and one maintained summary. Its matching SQLite
index is `data/experiments/openviking/pilot.sqlite3`; its completed manifest is
`data/experiments/openviking/export/manifest.json`. The full corpus has not been
imported into OpenViking. A working clone needs its own local dataset and import;
these files are excluded from Git.

| Table | Role |
| --- | --- |
| `messages` | Original text, sender, timestamp, reply target, conversation key and Telegram URL |
| `memories` | Records of kind `message`, `discussion`, `window`, `summary` or `fact` |
| `sources` | Links from records to original message IDs |
| `metadata` | Source provenance and historical coverage |
| `memory_fts` | Lexical index used by offline inspection and comparison commands |

Discussion extracts concatenate up to eight messages. Grouping uses a channel
post thread or reply chain, with 15-minute windows for unrelated messages. These
are navigation aids, not guaranteed topic boundaries. OpenViking additionally
generates summaries during import. Read the [import and retrieval guide](../../experiments/openviking/README.md)
before changing resource export, path reconciliation or semantic recall.

## Search contract

The domain exposes an async `MemoryBackend.search` interface. The agent's only
implementation is OpenViking:

```python
await memory.search(
    query="Who considered reusing Osinara for a bot with custom skills?",
    author_id=None,
    after=None,
    before=None,
    limit=3,
)
await memory.search(memory_id="message:15100")
```

Provide a semantic query or a returned memory ID. Query recall uses OpenViking's
`AsyncHTTPClient.find`, without a reranker or session query planning. All mapped
L0/L1/L2 records can enter recall, including original messages absent from
summaries. Direct memory-ID expansion reads canonical local evidence.

Author IDs and inclusive timezone-aware ISO bounds constrain original sources
before recall. A multi-source record qualifies only if every source satisfies
the filters. OpenViking's import timestamps do not replace Telegram send times.
The adapter checks returned URIs against the verified manifest before expanding
the corresponding SQLite record.

Each hit contains its kind, ID, excerpt, supporting messages, up to four related
context messages and discussion IDs that the same tool can expand. Context
includes reply parents up to two edges away and neighboring messages. It may
contain other authors and dates and is labeled separately. Original message
excerpts are capped at 1,200 characters with `truncated` and `excerpt_start`.
Telegram URLs always come from canonical messages.

Both the agent and offline baseline consider at most 100 ranked candidates and
return up to five hits, keeping one per conversation. This provides diversity,
not text deduplication; distinct useful messages within a conversation can be
suppressed. Top-k recall does not establish completeness or prove absence.
Questions about all activity over a period need a chronological scan or an
aggregation rather than this bounded retrieval.

## Startup and storage

`MEMORY_PATH` and `OPENVIKING_MANIFEST` select a matching local evidence index and
completed import. The manifest records the index SHA-256 and actual parser paths,
including wrapped Markdown files and split chunks. Every exported record must
have an original leaf. Startup rejects missing/mismatched files, an incomplete
import, an unavailable server or a missing remote corpus. There is no lexical
fallback or startup mode without memory.

`docker compose up -d --build --wait` starts both services. The agent waits for
OpenViking readiness, validates memory and its Telegram identity, and then polls.
The agent uses `http://openviking:1933`; host import tools use the published
loopback port. Evidence and the manifest are mounted read-only. The OpenViking
workspace, group registry and logs persist separately on the host.

Restart the agent after changing evidence or the manifest. Import into a fresh
namespace after index updates. Snapshot refreshes, live message synchronization
and deletion propagation are not automatic. Historical content is untrusted
source data; memory access does not imply live Telegram access. The current
single-owner, single-group corpus is distinct from the agent's in-memory
conversation history and observed-group registry.

## Offline maintenance and comparison

SQLite indexing and lexical inspection remain available without model calls:

```sh
uv run python -m dashulya.memory --index data/memory-next.sqlite3 build \
  data/fastnewsdev/snapshot.sqlite3 --chat fastnewsdev_chat
uv run python -m dashulya.memory search "rtk pytest"
uv run python -m dashulya.memory expand "message:15100"
```

Builders refuse existing destinations. Preserve maintained records when
creating another index. Summary/fact writes require one to eight existing source
IDs; this enforces provenance integrity, not the semantic truth of the text:

```sh
uv run python -m dashulya.memory remember \
  "A participant considered reusing Osinara for a bot with custom writing-style skills instead of building a bot from scratch." \
  --sources 15100
```

The agent has no write tool. Re-export and import after maintenance before using
that changed index with OpenViking. Follow the [experiment guide](../../experiments/openviking/README.md)
for import and offline comparisons. The [pilot report](../research/openviking-comparison.md)
records measured retrieval quality, context budget, latency and indexing cost;
the [source review](../research/openviking.md) records upstream behavior and limits.
