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
- **Understand pictures.** Send a photo or a JPEG, PNG or WebP file, with an
  optional question in its caption, or reply to a picture with your question.
- **Understand stickers.** Static stickers use their original image; animated
  and video stickers use Telegram's static thumbnail, with motion unavailable.
- **Find earlier discussions.** Recall saved originals by keywords and meaning,
  with original Telegram links and surrounding replies.
- **Save received messages.** Persist group messages from all participants and
  the owner's private messages, including edits and successfully delivered replies.
- **Inspect original history.** Read full sources, page through saved messages,
  resolve sender identities and count literal matches independently of embeddings.
- **Search the web.** With `EXA_API_KEY` configured, find current public information
  through Exa and read source pages, with links in the answer.
- **Keep track of groups.** List groups the bot has observed, preserving the
  registry across restarts.

Try questions such as:

> How did designers suggest moving HTML mockups into Figma?
>
> Who was considering reusing Osinara for a bot with custom skills?

Memory uses canonical PostgreSQL originals, BM25 keyword search and pgvector dense
embeddings over messages with their reply parents. An imported snapshot is
optional. Groups can recall only their own chat, across topics; the owner's
private conversation can recall all saved chats. Imported archives need an
explicit numeric chat binding before a group can access them.

Embedding preparation runs in the background after an initial startup attempt;
completed vectors and pending message revisions survive restarts. Original
history reads remain available during embedding outages. A semantic-query
failure is reported through the tool rather than silently claiming a complete
search. Historical backfill, maintained-fact writes/corrections, deletion
synchronization, full memory continuity across group migrations
are not implemented. [ADR 0005](docs/adr/0005-unify-memory-in-postgresql.md)
records the storage decision.

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
in the whole ordinary group, or the current real forum topic, including the
current request, plus their reply parents up
to five hops. Messages appear once in chronological message order; this bot's
previously delivered replies use the assistant role. Context reads local originals
immediately, without waiting for semantic indexing. Missing parents and earlier media
contents cannot be reconstructed. The current picture and its directly replied-to
picture can be supplied as native image inputs when their messages are in this
selection. The selection has at most 120 messages, with
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

Pictures are downloaded into RAM only after owner authorization and group addressing.
The original photo or image-file bytes go directly to the reply model through AG2's
native image input; no separate description model or image files are created.
The combined image limit is 20 MB (20,000,000 bytes), checked before download and
while streaming, even when Telegram omits or understates the file size. Downloads
are skipped when metadata already shows the image will not fit; the agent instead
receives a source-labelled note that the image is too large. If an unknown or
understated size exceeds the limit during streaming, the stream stops and the
same note is supplied while other available text and images remain usable. Downloads
have a 30-second deadline. File signatures determine JPEG, PNG or WebP MIME types;
Animation frame extraction, video/audio decoding and document parsing are not
supported by this image path.
The reply model must support image input; the default `gpt-5.6-luna` does.
Private history keeps recent images within the same 20 MB budget, replacing the
oldest images with explicit placeholders as needed. Text and replies remain.
Group images are available only for that request and are not added to searchable
chat memory. Group captions follow the same mention/name/reply rules as text.

Sticker messages follow the same access and addressing rules, including a sticker
sent in reply to the bot and a question replying to another participant's sticker.
Regular, mask and custom-emoji sticker objects use the same format handling:
static stickers supply their image bytes, while animated TGS and video WEBM
stickers supply only the JPEG/WebP thumbnail provided by Telegram. The agent is
explicitly told when it sees a thumbnail and must not infer animation or motion.
Missing thumbnails produce an unavailable-image note without downloading the
animation. Stickers and previews share the image byte budget and RAM-only path.
Inline custom emoji inside text are not fetched as separate sticker images.

GIF and silent MP4 messages delivered as Telegram animations use their static
thumbnail through the same path, including captions and direct-reply questions.
GIF files sent as documents use their thumbnail when Telegram provides one.
The original animation is not downloaded; missing or oversized previews produce
an omission note for the agent. Preview inputs are labelled as animations with
motion unavailable and share the existing 20 MB budget.

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
Run commands from the project root. Compose starts PostgreSQL with pgvector and
the agent. The database can start empty; importing an existing corpus is optional.

### 1. Configure the application

```sh
cp .env.example .env
```

Fill in these values:

```dotenv
BOT_TOKEN=your_bot_token
OWNER_USER_ID=your_numeric_telegram_user_id
OPENAI_API_KEY=your_openai_api_key
POSTGRES_PASSWORD=generated_random_hex_password
```

Create a bot through [@BotFather](https://t.me/BotFather). Use your numeric
Telegram user ID for `OWNER_USER_ID`. The existing local setup already has
credentials. Your `.env` stays out of Git and the Docker build context. The reply
and addressing models default to `gpt-6-luna`; model, timezone and local storage
settings are listed in [.env.example](.env.example).

Set `EXA_API_KEY` to enable the AG2 [Exa tools](https://docs.ag2.ai/docs/user-guide/extensions/tools/search/exa/).
The reply agent receives `exa_search` (five results with bounded text excerpts) and
`exa_get_contents` (up to three public URLs per call). Without the key, those tools
are absent and conversations and chat memory remain available. The addressing
classifier never receives web tools. Search and page contents are untrusted evidence;
answers should cite their original URLs. Web-tool failures are reported without
provider payloads, and each call has a 30-second deadline.

Generate `POSTGRES_PASSWORD` with `openssl rand -hex 32`. Keep it stable for an
existing database. Compose supplies the internal `DATABASE_URL`; direct Python
and CLI runs need a URL for a reachable PostgreSQL server with pgvector installed.

### 2. Import optional existing data

Skip this step for a new deployment. To migrate an existing SQLite deployment,
stop its agent, preserve the old files, and follow the [migration procedure](docs/design/chat-memory.md#sqlite-migration).
It copies imported and received originals, the group registry, cached embeddings
and legacy knowledge into PostgreSQL, verifies canonical messages and never
requests new embeddings. An archive needs a verified numeric Telegram chat ID
before it can be exposed to that group; an unbound archive remains private-only.

### 3. Start the agent

```sh
docker compose up -d --build --wait
```

Startup initializes the PostgreSQL schema and vector cache, attempts pending
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

The [Compose file](docker-compose.yaml) persists originals, the registry, lexical
postings and embeddings in `var/postgres/`, and logs in `var/logs/`. PostgreSQL
has no published host port. These directories survive container recreation and
are excluded from Git. Back up the running database with `pg_dump`; copy its data
directory only while PostgreSQL is stopped. Preserve pre-migration SQLite files
for rollback. The runtime uses PostgreSQL exclusively; legacy SQLite/OpenViking
maintenance commands remain available offline.

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
retained; passive capture does not download attachment contents. Authorized
picture requests use the RAM-only image path described above. Coverage and provenance belong
to each local dataset under `evals/datasets/`. The [original memory ADR](docs/adr/0001-openviking-memory.md)
records an 18-query historical pilot, not a full-corpus quality result or the
final decision for the current comparison.

## Development

GitHub Actions runs the development checks on pushes to `main` and pull requests.
Tests and package builds run on Python 3.12, 3.13, and 3.14 without live Telegram
or model API calls. PostgreSQL integration tests use a disposable schema per test.
Set `TEST_DATABASE_URL` to an isolated pgvector-enabled test database before
running `just check`; CI provisions it automatically. Without this variable,
PostgreSQL integration tests are skipped. Never point it at application data.

Architecture, development commands, and reference bot repositories live in
[AGENTS.md](AGENTS.md). The [repository layout](docs/design/repository-layout.md)
separates design documents, evaluation infrastructure and local application state.
PRDs live in GitHub Issues; datasets, evaluation results and research drafts are
local files excluded from Git. See [evals/](evals/README.md) for evaluation setup.

Dishka constructs the shared dependency graph for aiogram and AG2. The
[dependency injection design](docs/design/dependency-injection.md) describes
module responsibilities, scopes, resource cleanup, and provider overrides in
tests. Domain terms are defined in [GLOSSARY.md](GLOSSARY.md).
