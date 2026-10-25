# OpenViking memory experiment

Dashulya requires the official `openviking-sdk==0.1.16` against a separate
OpenViking 0.5.2 server. The assistant still has one `search_memory` tool.
OpenViking is the only agent memory backend. FTS5 remains an offline comparison baseline.

## Retrieval and evidence

The exporter copies every indexed message, discussion/window extract, maintained
summary and fact into Markdown resources. Discussion directories have at most
eight original messages and their extract. Singleton records get their own
directory. OpenViking generates directory abstracts and overviews; all original
message files remain candidates too. Generated directory summaries map to their
canonical discussion record when reconstructing evidence.
OpenViking's Markdown parser can wrap a file in its own directory or split it
into chunks. After ingestion the SDK lists actual resource paths and reconciles
them with the export, requiring a mapped original leaf for every exported record.
Both file-directory summaries and their chunks retain that record's source IDs.

`AsyncHTTPClient.find` searches L0/L1/L2 together, with at most 100 candidates.
This experiment uses QUICK vector retrieval, without session query planning or a
reranker. It does not use OpenAI Decisions for ranking. It is not a mandatory
parent-first tree traversal.

A manifest maps exact resource URIs to local memory IDs. Before vector recall,
the adapter computes eligible records from original Telegram authors and
timestamps and sends a URI equality filter. Every source of a derived record
must satisfy the filters. Returned URIs are checked against the manifest and
expanded through SQLite, including reply context and original Telegram links.
The adapter does not use model-generated URLs or attribution as citation data.
Directory summaries above the mapped discussion level are excluded from recall.

Both backends keep the existing one-result-per-conversation rule. This provides
diversity, not text deduplication, and can suppress distinct relevant messages
within the same conversation. Both use the same source/context expansion.
Context may contain other authors and dates; source filters apply to support.

The manifest records the SHA-256 of the exact SQLite index. Startup rejects a
different index, including a changed maintained index. Export and ingest a new
version after updates. Imports are explicit; live synchronization and deletion
propagation are not implemented.

## Run the experiment

Run from the project root. The model key and `OPENAI_MODEL` come from `.env`.
The included server configuration uses OpenAI `text-embedding-3-small` with 1,536
dimensions and `OPENAI_MODEL` for summaries. Both ingestion and semantic queries
use model APIs; the configured provider processes message text during ingestion.
Storage, exports and results live under ignored `data/experiments/openviking/`.

Start the server with Docker Compose:

```sh
uv sync --locked
docker compose up -d --wait openviking
```

Set `OPENVIKING_API_KEY` in `.env` before the first start; generate a local
service key with `openssl rand -hex 32`. The existing local setup already has a
generated key. Compose passes only the model credentials and this service key
to the container, rather than the bot's entire environment.

The [Compose service](../../docker-compose.yaml) uses the official `v0.5.2` image,
publishes `127.0.0.1:1933` and checks `/ready`. Both services are started manually
for local development, with automatic restarts disabled.
It mounts `data/experiments/openviking/workspace/` as persistent storage and the
[container config](ov.docker.conf) read-only. The `openviking` service runs the memory server;
The agent runs in a second container through its SDK. No standalone server
installation or global configuration is needed. `docker compose up -d --build --wait`
starts both services after the import is prepared.

Inside the container the server binds to `0.0.0.0` for Docker port forwarding.
It uses OpenViking's `trusted` service authentication: requests carry the shared
key and the SDK's explicit `default` account/user identity. The published host
port stays on loopback. This is a local, single-owner experiment. The setup
follows the upstream [container deployment guide](https://docs.openviking.ai/en/guides/03-deployment).

Inspect and stop it with:

```sh
docker compose ps
docker compose logs -f openviking
docker compose down
```

Stopping or recreating the container preserves the bind-mounted workspace.
Use `OPENVIKING_PORT` to change the published host port and update
`OPENVIKING_URL` to match. Compose uses `ov.docker.conf`.

In another terminal, create a fresh pilot and export it:

```sh
uv run python -m dashulya.memory.experiment sample
uv run python -m dashulya.memory.experiment export \
  --root-uri viking://resources/dashulya-fastnewsdev-pilot
```

The pilot copies the last 300 stored messages plus IDs within ±40 of 15100.
It retains derived records only when all their sources are included, rebuilds
its FTS index and records its parent index hash. Neither the full index nor the
original snapshot changes. Existing pilot/export paths are never overwritten.

Import with the SDK, wait for semantic processing, then compare:

```sh
uv run --env-file .env python -m dashulya.memory.experiment ingest
uv run --env-file .env python -m dashulya.memory.experiment compare
```

The SDK handles directory archiving, temporary uploads and actual path discovery. Ingestion details
are saved in `export/ingestion.json`; full benchmark results and excerpts are
saved in `comparison.json`. The committed [cases](cases.json) contain only test
queries and labeled message IDs. The default benchmark runs each query three
times and measures complete tool calls, including local evidence expansion.

To index the full maintained corpus, use a fresh export directory and namespace:

```sh
uv run python -m dashulya.memory.experiment \
  --index data/memory.sqlite3 \
  --export data/experiments/openviking/full-export \
  export --root-uri viking://resources/dashulya-fastnewsdev-full
uv run --env-file .env python -m dashulya.memory.experiment \
  --export data/experiments/openviking/full-export ingest
```

## Configure the agent memory

For the pilot, set these variables and restart the bot:

```dotenv
MEMORY_PATH=data/experiments/openviking/pilot.sqlite3
OPENVIKING_URL=http://127.0.0.1:1933
OPENVIKING_MANIFEST=data/experiments/openviking/export/manifest.json
```

Set `OPENVIKING_API_KEY` to the Compose service key. Agent startup requires
both the matching local index and manifest and a reachable imported corpus.
Server failures produce an explicit error; the agent has no alternate backend.
For the full export, select `data/memory.sqlite3` and its matching manifest.

These paths identify host files. Compose mounts them read-only at fixed paths
in the agent container and overrides its server URL with `http://openviking:1933`.
The host URL above remains available to offline import and comparison tools.

## Evaluation scope

Read [the comparison note](../../docs/research/openviking-comparison.md) for the
current run status and measurements. These cases are a small, manually labeled
development set, not a held-out benchmark. Labels identify relevant examples,
not every relevant message. Support hit@3 and source recall use only `sources`;
evidence hit@3 additionally counts `context`. Negative filter cases and an absent
topic expose different failure modes. A semantic top-k result alone does not
establish that a question has an answer in the corpus.

This benchmark compares retrieval under the same input queries. It does not
compare complete generated answers or an agent that rewrites questions into
several keyword searches. No quality or latency conclusion applies to the full
17,000-message corpus without a separate run.
