# Repository layout

Keep project design, evaluation infrastructure and application state separate.

| Path | Purpose | Versioned |
| --- | --- | --- |
| `docs/design/` | Current implementation and architecture | Yes |
| `docs/adr/` | Accepted decisions, rationale and trade-offs | Yes |
| `deploy/` | Service configuration consumed by Compose | Yes, with environment references for secrets |
| `evals/suites/` | Evaluation runners, case contracts and scoring rules | Yes |
| `evals/datasets/` | Entire local datasets, including descriptions, manifests, cases and annotations | No |
| `evals/runs/` | Generated exports, isolated indices and evaluation results | No |
| `var/postgres/` | Active PostgreSQL originals, registry, indexing queue and search data | No |
| `var/agent/` | Preserved pre-migration SQLite files | No |
| `var/backups/` | Migration snapshots and database backups | No |
| `var/openviking/` | Persistent OpenViking workspace | No |
| `var/logs/` | Application logs | No |
| `.scratch/` | Temporary research, interview notes and working drafts | No |
| `.github/agents/` | Contributor-agent workflow conventions | Yes |

PRDs and task specifications live in GitHub Issues. The current chat-memory
replacement specification is [issue #3](https://github.com/Lancetnik/dashulya/issues/3);
it describes requested behavior, not the current implementation. Keep research
working files local; record accepted architectural decisions in ADRs and then
discard the working files when they are no longer useful.

Datasets are excluded as complete directories, including preparation scripts
specific to a private corpus. Reusable evaluation infrastructure belongs in
`evals/suites/` and must run its automated tests without local datasets or model
calls. Each local dataset should carry its own provenance and coverage notes.

Treat an evaluation index as an input to a particular run. To ingest its original
messages into the running database, use the explicit migration command and verify
the numeric chat binding. Preserve the source snapshot and reuse cached vectors;
see [the migration procedure](chat-memory.md#sqlite-migration). Evaluation runs
must never replace application state. Container recreation preserves `var/`;
deleting it discards application state.
