# Working on Dashulya

Dashulya is a Telegram assistant for group conversations, currently available
to one configured owner with AI replies in private chats. Use English for project
documentation, code comments, metadata, and bot messages. Preserve imported
message text in its original language.

## Setup and checks

Follow [README.md](README.md#getting-started) for installation and local startup.
Run commands from the project root. Environment settings and validation live in
[`config.py`](src/dashulya/config.py); [.env.example](.env.example) lists the inputs.
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
Ruff policy lives in [ruff.toml](ruff.toml); typing, dependency, and import
contracts live in [pyproject.toml](pyproject.toml).
The local hooks in [.pre-commit-config.yaml](.pre-commit-config.yaml) reuse these recipes.
With `just` installed, enable them with `uv run pre-commit install`.

## Architecture

| Module | Responsibility |
| --- | --- |
| `src/dashulya/main.py` | Dependency assembly, startup, and cleanup |
| `src/dashulya/bot/` | aiogram routing, command/dialogue handlers, and middleware |
| `src/dashulya/ai/` | AG2 agent construction, prompts, conversation execution, and tool adapters |
| `src/dashulya/memory/` | Local indexing, unified search, evidence reconstruction, and offline commands |
| `src/dashulya/storage/` | Known-group persistence |
| `src/dashulya/config.py` | Environment configuration |

Keep dependency direction aligned with the Import Linter contracts. The AI module
is independent of aiogram; memory and storage are independent of aiogram and AG2.
Bot runtime imports stay independent of offline indexing and command execution.
The exhaustive contracts require new top-level modules and modules inside `bot/`
to be assigned a place in the hierarchy.

## Behavior to preserve

- **Access:** authorize the actual sender's numeric `from_user.id` against
  `OWNER_USER_ID`. Other users, bot senders, and messages without a sender are
  silently ignored before command execution or model invocation. Forwarded
  identities and `/start` do not grant access.
- **Routing:** AI replies and `/clear` operate in private chats. Group mentions
  and replies do not invoke the model. Group commands remain available to the
  owner. Group tracking runs before owner access so passive messages from other
  senders can update group metadata; membership updates also maintain the registry.
- **Conversation state:** reuse one AG2 `Agent` with separate `MemoryStream`
  histories keyed by chat/topic. Keep history and turn locking together in
  `ai/dialogue.py`. Serialize replies and `/clear` within each conversation while
  allowing other conversations to progress. Conversation state is in memory;
  `/clear` resets only the current conversation.
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
The current retrieval is lexical FTS5; semantic embeddings and automatic summary
generation remain future experiments.

Before querying, importing, refreshing, or evaluating the local Telegram corpus,
read [the dataset note](docs/datasets/fastnewsdev.md) for provenance, coverage,
schema semantics, and SQL examples. Keep the imported snapshot as a baseline
and put derived data under ignored `data/`. Existing memory indices may contain
maintained summaries and facts; rebuild into a fresh path to preserve them.
Source snapshots and search indices are separate from the group registry and
from in-memory AG2 conversation histories.

`MEMORY_PATH` selects the optional local index at startup. Without an index,
ordinary conversation remains available with the no-history-access prompt.
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
