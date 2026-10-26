<div align="center">
  <a href="https://t.me/dashulya_ai">
    <img src="assets/avatar-circle.png" width="160" height="160" alt="Dashulya's circular avatar" />
  </a>
  <p>
    <strong>Dashulya</strong> &nbsp; <a href="https://t.me/dashulya_ai">@dashulya_ai</a><br />
    Спрашивай, котик💋
  </p>
  <p>
    <sub>Also <a href="https://t.me/dashulyai_bot">@dashulyai_bot</a> · A Telegram assistant with memory and a sense of humor · Owner-only prototype</sub>
  </p>
</div>

---

## The idea

A group chat is full of useful things: recommendations, links, decisions, and
arguments worth coming back to. Dashulya is an experiment in giving that chat
an assistant who can find the relevant conversation, explain its context, and
point to the messages behind an answer.

She is meant for everyday conversations across all kinds of topics. The
personality is part of the concept: warm irony, a little cheekiness, and useful
answers when the question is serious.

The name grew out of a meme in the [FastNews chat](https://t.me/fastnewsdev_chat)
and [Ostrikov's chat](https://t.me/aostrikov_agents_chat)

> Hot robots in the chat 💋

## What she can do today

- **Talk in private.** Keep the current conversation in context, with a fresh
  start whenever you send `/clear`.
- **Find earlier discussions.** Recall saved originals by keywords and meaning,
  with original Telegram links and surrounding replies.
- **Save received messages.** Persist group messages from all participants and
  the owner's private messages, including edits and successfully delivered replies.
- **Inspect original history.** Read full sources, page through saved messages,
  resolve sender identities and count literal matches independently of embeddings.
- **Keep track of groups.** List groups the bot has observed, preserving the
  registry across restarts.

Try questions such as:

> How did designers suggest moving HTML mockups into Figma?
>
> Who was considering reusing Osinara for a bot with custom skills?

Memory uses canonical SQLite originals, BM25 keyword search and cached dense
embeddings over messages with their reply parents. An imported snapshot is
optional. Groups can recall only their own chat, across topics; the owner's
private conversation can recall all saved chats. Imported archives need an
explicit numeric chat binding before a group can access them.

Embedding preparation runs in the background after an initial startup attempt;
completed vectors and pending message revisions survive restarts. Original
history reads remain available during embedding outages. A semantic-query
failure is reported through the tool rather than silently claiming a complete
search. Historical backfill, maintained-fact writes/corrections, deletion
synchronization, full memory continuity across group migrations and web search
are not implemented. The [comparison record](evals/suites/memory/RESULTS.md)
separates engine comparisons from actual application validation;
[ADR 0004](docs/adr/0004-select-source-hybrid-memory.md) records the selected variant.

## Commands

| Command | What happens |
| --- | --- |
| `/start` | Meet Dashulya |
| `/help` | See the available commands and how to use them |
| `/groups` | List the groups she knows about |
| `/clear` | Reset the current private conversation |

The current build responds only to the user configured as its owner. Private
messages, an explicit @mention of the bot, and replies to its messages start an
AI conversation directly. Group commands remain available to the owner.

In groups, messages containing “Dashulya”, “Дашуля” or “Дашуль” in any letter case
(including Russian case endings) first go through a separate AG2 decision agent using
[OpenAI Decisions](https://docs.ag2.ai/docs/user-guide/decision_models/).
It distinguishes an invitation to answer from a discussion about the bot. The
classifier uses `gpt-6-luna` with the same `OPENAI_API_KEY`; `OPENAI_MODEL` selects
the agent that writes replies. It receives the current message, its reply target,
and the observed group context described below. Other group chatter does not
start a model turn. A negative decision or classifier failure sends
nothing, including no typing indicator. Explicit mentions and replies bypass
the classifier, even when it is unavailable.

Addressing logs include `should_reply` and `addressing_reason`. `name_not_found`
means the name filter skipped the model; `classifier_rejected` means the model
returned false; `classifier_no_answer` means it returned no decision.
`context_error` and `classifier_error` identify redacted failures at those stages.
Successful decisions distinguish `explicit_mention`, `reply_to_bot` and
`classifier_accepted`.

Each group reply uses a fresh model stream with the latest 20 observed messages
in that chat/topic, including the current request, plus their reply parents up
to five hops. Messages appear once in chronological message order; this bot's
previously delivered replies use the assistant role. Context reads local originals
immediately, without waiting for semantic indexing. Missing parents and media
contents cannot be reconstructed. The selection has at most 120 messages, with
no additional token cap. See [the decision](docs/adr/0002-rebuild-group-reply-context.md).

The agent returns structured reply text and an optional target message ID. It
can reply to another message in the supplied group context; null or an ID outside
that selection defaults to the triggering message. Every chunk uses the same
target. Only the reply text is sent to Telegram.
Requests to answer or greet another participant direct the agent to choose that
participant's relevant message as the reply target, including channel-authored
messages identified by the requester. Target-selection logs retain the requested
ID, effective ID, selection reason and available context IDs; they omit reply text.

Private conversation history lasts until a bot restart or `/clear`. The
known-group registry, saved originals and embedding cache persist between restarts.
`/clear` resets private reply context; it does not erase saved chat memory.

Formatted replies use Telegram Rich Messages for Markdown emphasis, source
links, headings, lists, tables, quotes, code, and collapsible `<details>` sections.
Plain answers use ordinary messages. Replies outside the native rich limits, or
explicitly rejected by Telegram, fall back to text entities with formatting
preserved across message chunks. Network failures do not trigger a second send.
The transport follows the [Osinara rich-message pattern](https://github.com/nyxandro/osinara/blob/4ca8eff77e0bf403f998aebefedd2deae2e60ba8/agent/lib/telegram-rich-messages.ts).

Messages reach the AI with their original Telegram send time in local time,
for example `25 October 2026 (Sunday) 12:16:50`. The timezone and UTC offset are
included separately; the application computes the weekday. `BOT_TIMEZONE`
defaults to `Europe/Moscow` and accepts IANA timezone names. Log timestamps
remain in UTC.

## Getting started

You need Docker with Compose, a Telegram bot token and an OpenAI API key.
Run commands from the project root. The agent can start with an empty local
journal; an imported corpus and OpenViking service are optional.

### 1. Configure the application

```sh
cp .env.example .env
```

Fill in these values:

```dotenv
BOT_TOKEN=your_bot_token
OWNER_USER_ID=your_numeric_telegram_user_id
OPENAI_API_KEY=your_openai_api_key
```

Create a bot through [@BotFather](https://t.me/BotFather). Use your numeric
Telegram user ID for `OWNER_USER_ID`. The existing local setup already has
credentials. Your `.env` stays out of Git and the Docker build context. The reply
and addressing models default to `gpt-6-luna`; model, timezone and local storage
settings are listed in [.env.example](.env.example).

### 2. Choose optional imported evidence

Skip this step to start with newly received messages. To retain an existing
SQLite evidence index, place it at `var/agent/evidence.sqlite3`. Compose mounts
`MEMORY_DIRECTORY` (default `./var/agent`) read-only at `/memory` and sets the
container's `MEMORY_PATH` to `/memory/evidence.sqlite3`. No completed OpenViking
import or manifest is required. A malformed existing index still fails startup.

For a custom host `MEMORY_PATH` from an earlier deployment, set
`MEMORY_DIRECTORY` to that file's parent directory and ensure the index is named
`evidence.sqlite3` there; preserve the original if making a copy. Compose no
longer uses the host `MEMORY_PATH` as a single-file mount. Direct Python runs
continue to use `MEMORY_PATH` normally.

Set `MEMORY_IMPORT_CHAT_ID` only after verifying the archive's actual numeric
Telegram chat ID. Without this trusted binding, the archive is available only
in the owner's private conversation; a matching public username does not grant
access to a group. See the [memory guide](docs/design/chat-memory.md).

### 3. Start the agent

```sh
docker compose up -d --build --wait
```

Startup initializes local originals and the vector cache, attempts pending
embedding preparation, verifies the Telegram bot identity and starts polling.
Embedding errors leave canonical history available and indexing queued for retry.
Docker's agent health check reflects completed startup, rather than complete
embedding coverage.

Open the bot in Telegram and send `/start`, then a message. Inspect and stop the
stack with:

```sh
docker compose ps
docker compose logs -f agent
docker compose down
```

The [Compose file](docker-compose.yaml) persists the group registry, live
originals and vectors under `var/agent/`, and logs under `var/logs/`. Imported
SQLite evidence is read-only. These local files survive container recreation
and are excluded from Git. Credentials, chat history, logs and developer tools
are excluded from the agent image.

The optional `openviking` Compose profile preserves legacy maintenance access
and existing `var/openviking/` state. Set a nonempty `OPENVIKING_API_KEY` before
starting it explicitly:

```sh
docker compose --profile openviking up -d --wait openviking
```

Its port stays bound to loopback (`http://127.0.0.1:1933` by default). The default
agent does not depend on this service or its key. The optional service refuses
to start without its key. Follow the [legacy import guide](evals/suites/openviking/README.md)
for maintenance of an existing OpenViking corpus.

Compose bind-mounts `./src` read-only at `/app/src`. `PYTHONPATH` makes these host
sources take precedence over the installed package in the image. After editing
Python files, reload the agent with `docker compose restart agent`; Python loads
the updated modules when its process starts. Rebuild the image when dependencies
change. The source mount adds no files from the rest of the repository.

For recording ordinary group messages and name-based addressing, make the bot
an administrator, or
[disable privacy mode](https://core.telegram.org/bots/features#privacy-mode)
through @BotFather and re-add the bot. Telegram does not deliver other bots'
messages to it. All received group participants' messages are saved, alongside
owner-private messages and delivered replies; commands and replies remain
restricted to the owner. Text, captions and basic attachment metadata are
retained; attachment contents are not downloaded. Coverage and provenance belong
to each local dataset under `evals/datasets/`. The [original memory ADR](docs/adr/0001-openviking-memory.md)
records an 18-query historical pilot, not a full-corpus quality result or the
final decision for the current comparison.

## Development

GitHub Actions runs the development checks on pushes to `main` and pull requests.
Tests and package builds run on Python 3.12, 3.13, and 3.14 without live Telegram
or model API calls. Run `just check` locally before publishing changes.

Architecture, development commands, and reference bot repositories live in
[AGENTS.md](AGENTS.md). The [repository layout](docs/design/repository-layout.md)
separates design documents, evaluation infrastructure and local application state.
PRDs live in GitHub Issues; datasets, evaluation results and research drafts are
local files excluded from Git. See [evals/](evals/README.md) for evaluation setup.

Dishka constructs the shared dependency graph for aiogram and AG2. The
[dependency injection design](docs/design/dependency-injection.md) describes
module responsibilities, scopes, resource cleanup, and provider overrides in
tests. Domain terms are defined in [GLOSSARY.md](GLOSSARY.md).
