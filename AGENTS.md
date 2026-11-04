# Working on Dashulya

Dashulya is a Telegram assistant for group conversations. Private access, group
activation and other commands belong to one configured owner; chat administrators
may also use `/stop` in their own group. Groups activated by the owner allow all
human participants to address her. Use Russian for bot service messages. Preserve imported
message text in its original language.

## Setup and checks

Run commands from the project root. Follow [README.md](README.md#getting-started)
for installation, environment configuration and local startup.

The local Compose setup bind-mounts `./src` at `/app/src`. For Python-only
changes, reload the agent with `docker compose restart agent` without rebuilding
the image. Rebuild when dependencies or the Dockerfile change. Before reloading,
check the running container's source mount: an existing Compose override may
point it at an older release snapshot. Use the current working tree for the
local stand and verify container health after reloading.

Use the [justfile](justfile) recipes: `just check` for Python changes and
`just test-local` for PostgreSQL-independent iteration. The full suite requires
an isolated `TEST_DATABASE_URL`; see [Development](README.md#development).
Keep routine tests independent of Telegram and paid model APIs. Before adding
or strengthening behavioral tests, read [TDD](.agents/skills/tdd/SKILL.md), including
its targeted mutation requirements. For documentation-only changes, verify links
and run `git diff --check`.

## Architecture

| Module | Responsibility |
| --- | --- |
| `src/dashulya/main/` | Startup, environment configuration, container, providers, and integration setup |
| `src/dashulya/bot/` | aiogram routing, command/dialogue handlers, and middleware |
| `src/dashulya/ai/` | Agent factories and model policy, tools, prompts, dialogue, and addressing |
| `src/dashulya/domain/` | Domain models, group lifecycle rules, and persistence/search interfaces |
| `src/dashulya/memory/` | Indexing, unified search, and evidence reconstruction over PostgreSQL |
| `src/dashulya/storage/` | Known-group persistence |
| `src/dashulya/database/` | Shared PostgreSQL pool, schema and migrations |

When reviewing code, read [CODING_STANDARDS.md](CODING_STANDARDS.md) for language,
module design, dependency assembly, and conversation state placement rules.

## Read before changing

- **Access, commands, addressing, private input, delivery or group lifecycle:** read
  [bot behavior](docs/design/bot-behavior.md).
- **Reply context, session lifetime, time/author formatting or media:** read
  [ADR 0002](docs/adr/0002-rebuild-group-reply-context.md).
- **Dependency assembly or framework injection:** read
  [the DI design](docs/design/dependency-injection.md).
- **Indexing, retrieval, evidence expansion or memory tools:** read
  [the memory design](docs/design/chat-memory.md) and
  [the access ADR](docs/adr/0003-bind-chat-memory-access.md).
- **Capture, edits, retries or indexing workers:** read
  [incoming-message rules](docs/design/chat-memory.md#incoming-group-messages).
- **Lexical scoring, snapshot indexes or their cache:** read
  [the ParadeDB decision](docs/adr/0008-use-chat-isolated-paradedb.md).
- **Bulk-loading originals or changing canonical storage:** read
  [the storage ADR](docs/adr/0005-unify-memory-in-postgresql.md).
- **Model-generated history SQL:** read
  [ADR 0006](docs/adr/0006-scope-model-sql-with-postgresql-views.md).
- **File locations or storage paths:** read
  [the repository layout](docs/design/repository-layout.md).
- **Reference-bot comparisons:** read
  [reference bots](docs/design/chat-memory.md#reference-bots) before borrowing designs.

## Memory and local data

Before querying, importing, refreshing, or evaluating a local Telegram corpus,
read its README under ignored `evals/datasets/` for provenance and coverage.
Preserve source snapshots and application originals. Existing indices may hold
maintained knowledge; rebuild into a fresh path under ignored `evals/runs/`.
Application data in `var/` stays separate from disposable test/evaluation schemas.
Runtime, maintenance and storage tests use PostgreSQL; see
[legacy-storage retirement](docs/adr/0007-retire-legacy-storage.md).

## Workflow references

- **Codebase exploration:** read [the domain doc rules](.github/agents/domain.md)
  and [GLOSSARY.md](GLOSSARY.md).
- **GitHub Issues and specs:** read [tracker conventions](.github/agents/issue-tracker.md)
  before tracker operations.
- **Triage:** read [the label mapping](.github/agents/triage-labels.md).
