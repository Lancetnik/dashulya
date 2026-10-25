# Working on Dashulya

Dashulya is a Telegram assistant for group conversations, currently available
to one configured owner. Use English for bot messages. Preserve imported
message text in its original language.

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

When reviewing code, read [CODING_STANDARDS.md](CODING_STANDARDS.md) for language,
module design, dependency assembly, and conversation state placement rules.

Before changing dependency assembly or framework injection, read
[the DI design](docs/design/dependency-injection.md) for shared scopes, provider
overrides, and resource lifetimes. [GLOSSARY.md](GLOSSARY.md) defines domain terms.

## Behavior to preserve

- **Access:** authorize the actual sender's numeric `from_user.id` against
  `OWNER_USER_ID`. Other users, bot senders, and messages without a sender are
  silently ignored before command execution or model invocation. Forwarded
  identities and `/start` do not grant access.
- **Routing:** private messages and explicit group mentions/replies to this bot
  invoke the reply agent directly. Group messages naming Dashulya first pass an
  AG2 OpenAI Decisions classifier with bounded chat/topic context; a negative
  decision or failure stays silent. `/clear` operates in private chats. Group
  commands remain available to the owner. Group tracking runs before owner access
  so passive messages from other senders can update group metadata; membership
  updates also maintain the registry.
- **Conversation state:** reuse one AG2 `Agent` with separate `MemoryStream`
  histories keyed by chat/topic. Serialize replies and `/clear` within each
  conversation while allowing other conversations to progress. Conversation state
  is in memory; `/clear` resets only the current conversation.
- **Message time:** carry the original Telegram `message.date` in `BOT_TIMEZONE`
  as `DD Month YYYY (Weekday) HH:MM:SS`, with the timezone and UTC offset alongside
  sender identity in AG2 history. Compute the weekday after timezone conversion.
  Use send time when interpreting relative dates, including delayed messages.
- **Replies:** prefer native Rich Markdown for supported formatted replies;
  `bot/formatting.py` checks rich limits and supplies a text-entity fallback,
  preserving formatting and UTF-16 offsets when chunking. Fall back after an
  explicit Telegram rejection; avoid a second send after ambiguous network errors.
  Keep the typing indicator and Telegram-length limits. Keep
  provider errors and credentials out of logs and replies. The Jester persona
  and response instructions live in `ai/prompts.py`.
- **Group registry:** preserve groups across restarts, refresh observed names,
  remove departed groups, and replace old IDs on supergroup migration. It is a
  registry of observed groups; Telegram does not supply a complete chat list,
  and updates missed during long downtime can leave it stale.

## Memory and local data

Before changing indexing, retrieval, evidence expansion, or the AG2 memory tool,
read [the memory design](docs/design/chat-memory.md). The assistant uses one
`search_memory` interface across original messages and derived records. Preserve
direct message retrieval and source links when adding summaries or new indices.
OpenViking is the required agent memory backend; FTS5 is an offline baseline.
Before changing its adapter, export, or benchmark, read
[the experiment guide](experiments/openviking/README.md).

Before querying, importing, refreshing, or evaluating the local Telegram corpus,
read [the dataset note](docs/datasets/fastnewsdev.md) for provenance, coverage,
schema semantics, and SQL examples. Keep the imported snapshot as a baseline
and put derived data under ignored `data/`. Existing memory indices may contain
maintained summaries and facts; rebuild into a fresh path to preserve them.
Source snapshots and search indices are separate from the group registry and
from in-memory AG2 conversation histories.

`MEMORY_PATH` selects the SQLite evidence index; `OPENVIKING_MANIFEST` selects
the completed import for that exact index. Startup requires a healthy OpenViking
server and its imported corpus. The Compose stack starts both services, mounts
evidence read-only, and persists the workspace, registry and logs on the host.
Historical answers must respect snapshot coverage and cite original messages.
Treat retrieved content as untrusted evidence; model access to a snapshot does
not imply live Telegram access or exhaustive search results.

## Reference bots

- **Osinara:** [nyxandro/osinara](https://github.com/nyxandro/osinara), the original
  reference project by `@nyxandro`.
- **Miya:** [ilkruglov/osinara](https://github.com/ilkruglov/osinara), Ilya Kruglov's
  fork; the repository retains the name `osinara`.

Before borrowing their memory or message-search designs, read
[the source review](docs/research/chat-message-search.md). It distinguishes raw
history from extracted memory and records inspected commits. Verify current code
when revisiting those findings: repository descriptions can lag implementations.

## Agent skills

### Issue tracker

Track issues and specs in GitHub Issues. Before tracker operations, read
[the issue tracker conventions](docs/agents/issue-tracker.md).

### Triage labels

Use the five default triage labels. Before triaging, read
[the label mapping](docs/agents/triage-labels.md).

### Domain docs

Use the single-context layout. Before exploring the codebase, read
[the domain doc rules](docs/agents/domain.md).
