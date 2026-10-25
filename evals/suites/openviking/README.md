# OpenViking retrieval evaluation

This suite compares OpenViking semantic recall with the offline FTS5 baseline
through the same asynchronous memory interface. It measures retrieval, not
complete generated answers or an agent's query-rewriting strategy. The
[ADR](../../../docs/adr/0001-openviking-memory.md) records the original decision;
the [memory design](../../../docs/design/chat-memory.md) describes production behavior.

Run commands from the repository root. Runtime service setup is described in
[the project README](../../../README.md#getting-started); Compose reads
[`deploy/openviking.conf`](../../../deploy/openviking.conf). The lightweight SDK
runs in the development environment; the OpenViking server runs in Docker.

## Local inputs and outputs

The entire `evals/datasets/` directory is excluded from Git. Obtain a corpus and
its cases separately, and read the dataset's local provenance and coverage notes.
Current defaults use:

- Source evidence: `evals/datasets/fastnewsdev/memory.sqlite3`.
- Cases: `evals/datasets/fastnewsdev/openviking-cases.json`.
- Isolated pilot: `evals/runs/openviking/pilot.sqlite3`.
- Export and mapping: `evals/runs/openviking/export/`.
- Measurements: `evals/runs/openviking/comparison.json`.

`evals/runs/` is also ignored. Corpus descriptions, preparation scripts,
annotations, Markdown exports and results stay local. Automated tests construct
temporary synthetic fixtures and require none of these inputs.

## Prepare and import

Install development dependencies and start only the memory service:

```sh
uv sync --locked
docker compose up -d --wait openviking
```

The server uses the credentials and models configured in `.env`. Ingestion
sends message text to the configured provider, and semantic queries make query
embedding calls. Evaluation does not need the Telegram bot to run.

Create a pilot at a fresh destination and export it into a fresh Viking namespace:

```sh
uv run python -m evals.suites.openviking.runner sample
uv run python -m evals.suites.openviking.runner export \
  --root-uri viking://resources/dashulya-fastnewsdev-pilot
uv run --env-file .env python -m evals.suites.openviking.runner ingest
```

The existing development pilot selects the last 300 stored messages and IDs
within 40 of message 15100. Derived records are retained only when all their
sources are selected. The source index and its maintained summaries remain
unchanged. Existing pilot and export destinations are never overwritten.
Use `sample --last` and `sample --around` to choose a different sample.

The SDK uploads resources and waits for processing. After ingestion, actual
resource paths are reconciled with exported records, including parser wrappers
and split chunks. Every exported record must have a mapped original leaf.
`export/manifest.json` records this coverage and the evidence index SHA-256;
`export/ingestion.json` records ingestion results.

Override `--index` and `--export` before the subcommand to use other inputs:

```sh
uv run python -m evals.suites.openviking.runner \
  --index evals/datasets/fastnewsdev/memory.sqlite3 \
  --export evals/runs/openviking/full-export \
  export --root-uri viking://resources/dashulya-fastnewsdev-full
```

Use a separate namespace for another import. The server workspace is persistent
application state under `var/openviking/`; exports and measured run outputs stay
under `evals/runs/`. Starting this suite does not replace the active agent's
SQLite evidence or manifest.

## Case contract and scoring

Cases form a JSON array. Each object contains `id`, `category`, `query` and
`expected` (labeled supporting message IDs). Optional `author_id`, `after` and
`before` constrain original sources. Empty `expected` identifies a negative case.
Local datasets own actual questions, labels and annotations.

```sh
uv run --env-file .env python -m evals.suites.openviking.runner compare
```

`compare --cases`, `--output` and `--repeats` override local cases, result path
and timing repetitions. The default is three repetitions. Both backends use
at most 100 candidates, return three hits and retain one per conversation.
Source filters apply before recall and during canonical evidence expansion;
context may contain other authors or dates.

Support hit@3, source recall and reciprocal rank count only `sources`.
Evidence hit@3 additionally counts `context`. Negative-empty rate checks whether
negative cases return no hits. Latency covers complete tool calls, including
local evidence reconstruction. Outputs include source excerpts and are private.

A bounded search is not proof of absence. Development labels can identify useful
examples without labeling every relevant message. Compare fixed source/token
budgets before making an efficiency claim; a backend returning more source text
can improve recall while increasing the answering model's input cost. Use held-out
cases for broader quality claims and keep historical replay free of later messages.

## Activate an imported corpus

The agent defaults to separate files under `var/agent/`. After a new import,
stop the agent, explicitly select the evidence and matching manifest, then start it:

```sh
docker compose stop agent
mkdir -p var/agent/import
cp evals/runs/openviking/pilot.sqlite3 var/agent/evidence.sqlite3
cp evals/runs/openviking/export/manifest.json var/agent/import/manifest.json
docker compose up -d --build --wait
```

The evidence hash must match the manifest and the remote corpus must exist.
Choose an alternate import by setting `MEMORY_PATH` and `OPENVIKING_MANIFEST`
in `.env`. Compose mounts these host files read-only and supplies the internal
server URL. Host tools use the published loopback URL instead.
