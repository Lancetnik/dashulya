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
- **Find earlier discussions.** Search an imported chat-history snapshot and
  return original messages with Telegram links and surrounding replies.
- **Remember new group messages.** Save received messages and edits locally,
  then index them in OpenViking for recall through the same memory tool.
- **Follow memories back to their sources.** Search original messages,
  discussion extracts, and source-backed summaries or facts through one tool.
- **Keep track of groups.** List groups the bot has observed, preserving the
  registry across restarts.

Try questions such as:

> How did designers suggest moving HTML mockups into Figma?
>
> Who was considering reusing Osinara for a bot with custom skills?

Memory combines an imported snapshot with new group messages received while the
bot is running. OpenViking provides semantic retrieval and generated summaries,
with original messages preserved locally as evidence. Pending messages are
indexed in the background; retries survive container restarts. Historical
backfill, deletion synchronization and web search are not implemented.

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

In groups, messages containing “Dashulya” or “Дашуля” (including Russian case
endings) first go through a separate AG2 decision agent using
[OpenAI Decisions](https://docs.ag2.ai/docs/user-guide/decision_models/).
It distinguishes an invitation to answer from a discussion about the bot. The
classifier uses `gpt-6-luna` with the same `OPENAI_API_KEY`; `OPENAI_MODEL` selects
the agent that writes replies. It receives the current message, its reply target,
and the observed group context described below. Other group chatter does not
start a model turn. A negative decision or classifier failure sends
nothing, including no typing indicator. Explicit mentions and replies bypass
the classifier, even when it is unavailable.

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

Private conversation history lasts until a bot restart or `/clear`. The
known-group registry and group-message journal persist between restarts.

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

You need Docker with Compose, a Telegram bot token, an OpenAI API key, and an
imported memory corpus. Run commands from the project root. OpenViking is the
agent's required memory backend.

### 1. Configure the application

```sh
cp .env.example .env
```

Fill in these values:

```dotenv
BOT_TOKEN=your_bot_token
OWNER_USER_ID=your_numeric_telegram_user_id
OPENAI_API_KEY=your_openai_api_key
OPENVIKING_API_KEY=your_local_service_key
```

Create a bot through [@BotFather](https://t.me/BotFather). Use your numeric
Telegram user ID for `OWNER_USER_ID`. Generate the local service key with
`openssl rand -hex 32`. The existing local setup already has credentials.
Your `.env` stays out of Git and the Docker build context. Model, timezone and
host-side memory paths are listed in [.env.example](.env.example).

### 2. Prepare the required memory

The current local setup has a completed OpenViking import of 379 messages.
`MEMORY_PATH` and `OPENVIKING_MANIFEST` must reference its matching SQLite
index and import manifest. Exports and indices are local files, so a fresh
clone needs its own corpus before the agent can start.

To prepare a new corpus, follow the [memory guide](docs/design/chat-memory.md)
and [OpenViking import guide](evals/suites/openviking/README.md). Offline import
commands use Python 3.12–3.14 and [uv](https://docs.astral.sh/uv/). Start only the
memory service while importing:

```sh
docker compose up -d --wait openviking
```

The agent requires a verified import and a reachable OpenViking service.
Startup fails explicitly when either is missing. FTS5 is available only as an
offline comparison baseline.

### 3. Start both services

```sh
docker compose up -d --build --wait
```

The agent starts after OpenViking becomes healthy, verifies the imported corpus
and its Telegram identity, then starts polling. Docker's agent health check
reflects this startup readiness. The agent reaches OpenViking at
`http://openviking:1933`; host import tools use `http://127.0.0.1:1933` by default.

Open the bot in Telegram and send `/start`, then a message. Inspect and stop the
stack with:

```sh
docker compose ps
docker compose logs -f agent
docker compose down
```

The [Compose file](docker-compose.yaml) mounts the evidence database and manifest
read-only. Active evidence, the group registry and live messages persist under
`var/agent/`; OpenViking uses `var/openviking/`, and logs use `var/logs/`. These
local files survive container recreation and are excluded from Git. Credentials,
chat history, logs and developer tools are excluded from the agent image.

For recording ordinary group messages and name-based addressing, make the bot
an administrator, or
[disable privacy mode](https://core.telegram.org/bots/features#privacy-mode)
through @BotFather and re-add the bot. Telegram does not deliver other bots'
messages to it. All received group participants' messages are indexed; replies and commands
remain restricted to the owner. Private conversations are not added to chat
memory. Text, captions and basic attachment metadata are retained; attachment
contents are not downloaded. Coverage and provenance belong to each local
dataset under `evals/datasets/`. The [memory ADR](docs/adr/0001-openviking-memory.md)
records the pilot's decision and limitations.

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
