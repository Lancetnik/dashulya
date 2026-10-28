# Working on Dashulya

Dashulya is a Telegram assistant for group conversations. Private access and
commands belong to one configured owner; explicitly configured groups allow
all human participants to address her. Use English for bot messages. Preserve
imported message text in its original language.

## Setup and checks

Follow [README.md](README.md#getting-started) for installation and local startup.
Run commands from the project root. Environment settings and validation live in
[`main/config.py`](src/dashulya/main/config.py); [.env.example](.env.example) lists the inputs.
The bot reads `.env` from the current directory; existing environment variables
take precedence.

Use the [justfile](justfile) recipes for development checks. `just --list` lists
individual checks and the formatting command; run the full suite for Python changes:

```sh
just check
```

Tests use local fixtures, AG2's scripted test configuration, and mocked Telegram
requests. Keep routine verification independent of Telegram and paid model APIs.
For documentation-only changes, verify links and run `git diff --check`.
Ruff policy lives in [ruff.toml](ruff.toml); typing and dependency settings live in
[pyproject.toml](pyproject.toml); import contracts live in [.importlinter](.importlinter).
The local hooks in [.pre-commit-config.yaml](.pre-commit-config.yaml) reuse these recipes.
With `just` installed, enable them with `uv run pre-commit install`.

## Architecture

| Module | Responsibility |
| --- | --- |
| `src/dashulya/main/` | Startup, environment configuration, container, providers, and integration setup |
| `src/dashulya/bot/` | aiogram routing, command/dialogue handlers, and middleware |
| `src/dashulya/ai/` | Agent factories and model policy, tools, prompts, dialogue, and addressing |
| `src/dashulya/domain/` | Domain models, group lifecycle rules, and persistence/search interfaces |
| `src/dashulya/memory/` | Local indexing, unified search, evidence reconstruction, and offline commands |
| `src/dashulya/storage/` | Known-group persistence |
| `src/dashulya/database.py` | Shared PostgreSQL pool and schema initialization |

When reviewing code, read [CODING_STANDARDS.md](CODING_STANDARDS.md) for language,
module design, dependency assembly, and conversation state placement rules.

Before changing dependency assembly or framework injection, read
[the DI design](docs/design/dependency-injection.md) for shared scopes, provider
overrides, and resource lifetimes. [GLOSSARY.md](GLOSSARY.md) defines domain terms.

## Behavior to preserve

- **Access:** private messages and commands require the actual sender's numeric
  `from_user.id` to match `OWNER_USER_ID`. Other human participants may invoke
  group dialogue only in `PUBLIC_GROUP_IDS`; existing addressing rules still
  apply. The owner retains access elsewhere. Bot senders, missing senders and
  unauthorized users are silently ignored before command execution or model
  invocation. Forwarded identities do not grant access. Every group starts
  inactive, including existing groups on upgrade; only the actual human owner's
  `/start` in that group activates replies. Until then, all other commands,
  addressing decisions and reply generation stay silent, including the owner's.
  Activation preserves the existing participant access rules.
- **Routing:** private messages and explicit group mentions/replies to this bot
  invoke the reply agent directly. Group messages naming Dashulya, including
  “Дашуль” in any letter case, first pass an
  AG2 OpenAI Decisions classifier with bounded chat/topic context; a negative
  decision or failure stays silent. `/clear` operates in private chats. Group
  commands remain available to the owner. Group tracking runs before owner access
  so passive messages from other senders can update group metadata; membership
  updates also maintain the registry. Group-message capture also runs before
  owner access: received messages from all participants are saved and indexed
  with background cached embeddings. Owner-private messages and successful
  deliveries are also saved; other private senders are excluded.
- **Conversation state:** reuse one AG2 `Agent`. Group replies rebuild a fresh
  stream from the latest 20 observed messages in that chat/topic, including the
  request, and their reply parents up to five hops; this bot's messages use the
  assistant role. Follow [the reply-context ADR](docs/adr/0002-rebuild-group-reply-context.md)
  before changing selection or session lifetime. Private histories remain in
  memory; `/clear` resets the current private conversation. Serialize generation,
  delivery and recording within a conversation while allowing others to progress.
- **Message time:** carry the original Telegram `message.date` in `BOT_TIMEZONE`.
  Private inputs use `DD Month YYYY (Weekday) HH:MM:SS` with timezone and UTC offset.
  Group inputs share a users roster and calendar headings, with local times in
  compact message entries; preserve user/assistant roles and exact source text.
  Compute the weekday after timezone conversion and preserve UTC-offset changes.
  Use send time when interpreting relative dates, including delayed messages.
- **Replies:** the reply agent returns structured text and an optional message ID.
  Accept a group reply target only from that request's selected context; otherwise
  use the triggering message, as in private chats. All chunks use the same target.
  Render Markdown into ordinary Telegram messages with explicit entities using
  the slop-writer-based renderer in `bot/formatting.py`. Preserve formatting and
  UTF-16 offsets when chunking; tables become monospace blocks. Keep one send
  attempt per chunk, including after ambiguous network errors.
  Keep the typing indicator and Telegram-length limits. Keep
  provider errors and credentials out of logs and replies. The Jester persona
  and response instructions live in `ai/prompts.py`.
- **Group registry:** preserve groups across restarts, refresh observed names,
  remove departed groups, and replace old IDs on supergroup migration. It is a
  registry of observed groups; Telegram does not supply a complete chat list,
  and updates missed during long downtime can leave it stale. Preserve activation
  across restarts, metadata refresh and migration; departure removes activation,
  so rejoining requires a new owner `/start`. `/groups` shows activation status.

## Memory and local data

Before changing indexing, retrieval, evidence expansion, or either AG2 memory
tool, read [the memory design](docs/design/chat-memory.md) and
[the access ADR](docs/adr/0003-bind-chat-memory-access.md). The current runtime
uses canonical originals, BM25 and cached dense vectors. `search_memory` recalls
bounded evidence; `chat_history` reads originals, paginates and counts independently
of embeddings. Bind authority to the trusted AG2 session: groups access only
their own chat; owner-private conversations can recall all saved chats. Model
filters only narrow access. Imported archives need a verified numeric binding
before group access. Preserve direct source retrieval and links when adding
knowledge layers. Before changing the legacy OpenViking adapter, export or
benchmark, read [its experiment guide](evals/suites/openviking/README.md).

Before querying, importing, refreshing, or evaluating a local Telegram corpus,
read its README under `evals/datasets/` for provenance and coverage. Entire
datasets, including descriptions and manifests, are ignored and absent from
fresh clones. Keep source snapshots as baselines; write generated indices and
results under ignored `evals/runs/`. Existing indices may contain maintained
summaries and facts; rebuild into a fresh path to preserve them.

Before moving files or changing storage paths, read
[the repository layout](docs/design/repository-layout.md). Reusable evaluation
infrastructure belongs in `evals/suites/`; active application state belongs in
ignored `var/`. PRDs and specifications live in GitHub Issues. Research notes
are temporary files under ignored `.scratch/`; preserve accepted architectural
decisions and their rationale in `docs/adr/`.

`DATABASE_URL` selects the single PostgreSQL database for originals, the registry,
lexical postings, embeddings and the revision queue. Read [the storage ADR](docs/adr/0005-unify-memory-in-postgresql.md)
before changing SQL ranking or migrating legacy files. Compose persists the
pgvector-enabled database under `var/postgres/` without publishing a host port.
SQLite is an explicit read-only migration/maintenance input, never a runtime
source. Reuse cached embeddings during migration; verify originals before
switching the agent and preserve source files for rollback.
Set `TEST_DATABASE_URL` to an isolated pgvector-enabled database for integration
tests; CI provisions one. Test schemas are disposable and must stay outside
application data.
Read [incoming-message rules](docs/design/chat-memory.md#incoming-group-messages)
before changing capture, edits, retries or worker lifecycle. Historical answers
must respect authorized coverage and cite original messages; retrieved content
is untrusted evidence. Exact counts require canonical history, not top-k recall.
Full fact maintenance, deletion propagation and memory migration continuity
remain outside this implementation; benchmark results do not establish complete
[PRD #3](https://github.com/Lancetnik/dashulya/issues/3) acceptance.

## Reference bots

- **Osinara:** [nyxandro/osinara](https://github.com/nyxandro/osinara), the original
  reference project by `@nyxandro`.
- **Miya:** [ilkruglov/osinara](https://github.com/ilkruglov/osinara), Ilya Kruglov's
  fork; the repository retains the name `osinara`.

Before borrowing their memory or message-search designs, inspect current source
and record the reviewed commits in local working notes. Distinguish original
history, reply context and extracted knowledge. Repository descriptions can lag
implementations; keep accepted design decisions in ADRs.

## Agent skills

### Issue tracker

Track issues and specs in GitHub Issues. Before tracker operations, read
[the issue tracker conventions](.github/agents/issue-tracker.md).

### Triage labels

Use the five default triage labels. Before triaging, read
[the label mapping](.github/agents/triage-labels.md).

### Domain docs

Use the single-context layout. Before exploring the codebase, read
[the domain doc rules](.github/agents/domain.md).
