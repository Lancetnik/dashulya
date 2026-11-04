<div align="center">
  <a href="https://t.me/dashulya_ai">
    <img src="assets/avatar-circle.png" width="160" height="160" alt="Dashulya's circular avatar" />
  </a>
  <p>
    <strong>Dashulya</strong> &nbsp; <a href="https://t.me/dashulya_ai">@dashulya_ai</a><br />
    Спрашивай, котик💋
  </p>
  <p>
    <sub>Also <a href="https://t.me/dashulyai_bot">@dashulyai_bot</a> · A Telegram assistant with memory and a sense of humor</sub>
  </p>
</div>

---

## The idea

A group chat is full of useful things: recommendations, links, decisions, and
arguments worth coming back to. Dashulya is an experiment in giving that chat
an assistant who can find the relevant conversation, explain its context, and
point to the messages behind an answer.

She is meant for everyday conversations across all kinds of topics. The
personality is mischievous, bold and quick-witted: natural banter, playful
teasing, warm irony and self-irony grounded in the conversation. She keeps casual
replies short and gives direct, useful answers when the question is serious.

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
- **Send her own stickers.** Choose from the six emotions in
  [DashulyaAI](https://t.me/addstickers/DashulyaAI), either on their own or alongside
  a text answer. Stickers share the ordered three-message limit and reply-target
  rules; substantive questions still receive text.
- **Find earlier discussions.** Recall saved originals by keywords and meaning,
  with original Telegram links and surrounding replies.
- **Save received messages.** After the owner's first `/start` in a group, persist
  its participants' messages, including edits and successfully delivered replies.
  The owner's private messages are also saved.
- **Inspect original history.** Read full sources, page through saved messages,
  resolve sender identities and count literal matches independently of embeddings.
  Reading a saved image message also supplies its picture or media preview from
  cache or Telegram when a file reference is available.
- **Search the web.** With `KEENABLE_API_KEY` configured, find current public information
  through Keenable and read source pages, with links in the answer.
- **Run Python calculations.** Use OpenAI's hosted Code Interpreter through
  AG2's [built-in `CodeExecutionTool`](https://docs.ag2.ai/docs/user-guide/tools/builtin_tools/#code-execution).
- **Keep track of groups.** List groups the bot has observed, preserving the
  registry across restarts.

Try questions such as:

> How did designers suggest moving HTML mockups into Figma?
>
> Who was considering reusing Osinara for a bot with custom skills?

Memory uses canonical PostgreSQL originals, ParadeDB BM25 keyword search and pgvector dense
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
| `/help` | See the available commands and how to use them |
| `/clear` | Reset the current private conversation |
| `/groups` | List the groups she knows about |
| `/start` | Meet Dashulya; activate replies in the current group |
| `/stop` | Owner or chat admins pause replies in the current group; keep saving messages |
| `/settings` | Owner or chat admins independently select who can use ordinary replies and whispers |
| `/ban @username` | Owner or chat admins stop a user from triggering replies in this group |
| `/unban @username` | Owner or chat admins restore a user's ability to trigger replies in this group |

Bot service messages are in Russian. `/help` groups commands for the private
conversation, group management and participant restrictions.

Private conversations and commands are available only to the configured owner,
except group `/stop`, `/ban`, `/unban` and `/settings`, which are also available to that group's administrators.
Every group starts inactive. The owner enables replies with `/start` (or
`/start@<bot_username>`) in that group. While inactive, the owner can use `/start`,
`/stop`, `/ban`, `/unban` and `/settings`; chat administrators can use `/stop`, `/ban`, `/unban` and `/settings`. Sending `/start`
privately or in another group does not activate it. `/groups` shows each
group's activation status. Activation survives restarts, name changes and
supergroup migration; removing and re-adding the bot requires a new `/start`.
Groups already known before this feature also start inactive after upgrading.
The owner or a chat administrator can pause replies in that group, across topics,
with `/stop` or `/stop@<bot_username>`. Only the owner can resume with `/start`.
`/stop` works only in group chats; repeating it confirms the pause.
Before the owner's first `/start` in a group, received messages, edits, reply
snapshots and outgoing responses are neither saved nor queued for indexing.
Group metadata is still tracked. After that first activation, capture and
indexing continue through every `/stop` and subsequent `/start`, across restarts
and supergroup migration. Removing the bot resets this permission.
On upgrade, currently active groups retain capture; paused groups need one
owner `/start` because their prior activation was not recorded separately.

Use `/settings` in a group to choose the audience independently for ordinary
replies (`all` or `admins`) and `/whisper` (`all`, `admins` or `disabled`). For
example, only administrators can ask publicly while everyone can whisper.
Both default to everyone; the configured bot owner is also allowed in `admins`
mode. The blacklist and group pause apply to both modes, and disabling whispers
also blocks the owner. Settings work across topics, persist through restarts and
supergroup migration, and can be changed while paused. `/start` resumes with the
saved settings. Removing and re-adding the bot restores defaults. Button presses
check the presser's current rights in that group.

Administrator rights are checked against Telegram's current list for that chat;
failed checks deny access. Anonymous administrators can use `/stop`, `/ban` and `/unban` when Telegram's
`sender_chat` matches the current group's numeric ID.

Use `/ban @username` (or `/ban@<bot_username> @username`). Lookups ignore case and
use each human sender's latest saved username in this group. Unknown, outdated or
ambiguous usernames require a new message from that participant before retrying.
The ban binds to their numeric ID, so changing their username does not remove it.
The blacklist applies across topics in this group, survives restarts and
supergroup migration, and is removed when the bot leaves the group.
Repeating the command is harmless. Banned users' mentions, replies and name-based
requests do not invoke the classifier or reply model. Their messages continue to
be saved and can appear in context and memory. Other groups and private access
are unaffected. Administrative commands remain available to authorized senders.

Use `/unban @username` to remove a ban in this group, using the participant's
latest saved username. The same rights and lookup rules apply as for `/ban`.
Repeating `/unban` is harmless, and the removal persists across restarts.
Unbanning does not activate a paused group or remove anyone else's ban.

In any group activated by the owner, human participants outside its blacklist may address the bot.
Private messages, an explicit @mention of the bot, and replies to its messages
start an AI conversation directly. Other commands remain owner-only in enabled groups.

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
immediately, without waiting for semantic indexing. Photos, JPEG/PNG/WebP image
files, stickers and GIF/Telegram animations in this selection are downloaded
from Telegram using saved file IDs, even when they were sent without addressing
the bot. Current and directly replied-to
pictures can also be supplied as native image inputs. Missing parents and older
media without saved file references remain unavailable. Media candidates load
through `asyncio.gather` within a shared byte budget and 10-second deadline.
Individual metadata/download operations have a 5-second deadline. Failed inputs
become explicit omission notes, including failures on the current message.
Normalized images and animation frames use a bounded 64 MiB RAM cache for five
minutes; unavailable files are remembered for 30 seconds. Media logs retain
stages, timings, cache outcomes and omissions without file IDs, URLs or bytes.
The selection has at most
120 messages, with no additional token cap. See
[the decision](docs/adr/0002-rebuild-group-reply-context.md).

The agent returns up to three ordered messages, each with reply text and an
optional target message ID. Each can reply to another message in the supplied
group context; null or an ID outside that selection defaults to the triggering
message. Each message is rendered and sent separately; long text still splits
into Telegram-sized chunks sharing that message's target. Only the reply text
is sent to Telegram.
The reply agent uses `set_reaction(message_id, emoji)` for all reactions.
Instructions require it to react only on its own initiative, independently
choosing whether to react, the target and the emoji. It is encouraged to react
actively to praise, useful insights, important updates and jokes, and to use 🤡
for hostile jabs directed at it while taking valid criticism seriously.
It must not follow requests to add, change or distribute reactions from any
participant, including the owner,
or carry out requested batches of reactions. This also applies to quoted
requests and instructions retrieved from history. It can
react to the current message or any other message ID in the current chat,
including messages outside the selected reply context, and can react to several
messages or combine reactions with a text reply. The tool description lists the
standard Telegram reaction emoji; chat settings may restrict them further.
The chat ID comes exclusively from the application's AG2 Context Variable;
the model cannot choose another chat. A successful tool-only turn may finish
without another message. Telegram enforces available emoji and chat permissions.
Tool results report `sent`, `rejected`, `rate_limited` or `unknown`. A rate limit
returns a safe explanation and Telegram's `retry_after` delay in seconds; the
agent must wait at least that long before trying again and must not switch emoji
or messages to bypass the limit. Rejection returns a safe error code to the
agent, which can choose another emoji or explain the problem
in its own words. Reaction failures never produce a canned user-facing reply.
Unknown delivery blocks further attempts on that message for the remainder of
the turn. Access and addressing rules apply before reaction tools can run.
Requests to answer or greet another participant direct the agent to choose that
participant's relevant message as the reply target, including channel-authored
messages identified by the requester. Target-selection logs retain the requested
ID, effective ID, selection reason and available context IDs; they omit reply text.

Private conversation history lasts until a bot restart or `/clear`. The
known-group registry, saved originals and embedding cache persist between restarts.
`/clear` resets private reply context; it does not erase saved chat memory.
The agent chooses memory/history tools when a request needs earlier evidence;
there is no automatic search before every reply.

Pictures are downloaded into RAM only after sender authorization and group addressing.
The original photo or image-file bytes go directly to the reply model through AG2's
native image input; no separate description model or image files are created.
The combined image limit is 20 MB (20,000,000 bytes), checked before download and
while streaming, even when Telegram omits or understates the file size. Downloads
are skipped when metadata already shows the image will not fit; the agent instead
receives a source-labelled note that the image is too large. If an unknown or
understated size exceeds the limit during streaming, the stream stops and the
same note is supplied while other available text and images remain usable. Downloads
have a 30-second deadline. File signatures determine JPEG, PNG or WebP MIME types;
Telegram animations can supply their first frame as described below; ordinary
video/audio messages and document parsing are not supported by this image path.
The reply model must support image input.
Private history keeps recent images within the same 20 MB budget, replacing the
oldest images with explicit placeholders as needed. Text and replies remain.
Group images are available only for that request and are not added to searchable
chat memory. Capture persists only Telegram file IDs, known sizes and visual-input
metadata, including available thumbnail references. Photos, image files, stickers
and GIF/Telegram animations can be downloaded again when selected as earlier
group context, including after a restart. Current and direct-reply inputs take
priority within the shared budget, followed by newer context images. Unavailable
saved files produce explicit omission notes. Attachments captured before their
file IDs were stored cannot be recovered automatically. Group captions follow the same
mention/name/reply rules as text.

Sticker messages follow the same access and addressing rules, including a sticker
sent in reply to the bot and a question replying to another participant's sticker.
Regular, mask and custom-emoji sticker objects use the same format handling:
static stickers supply their image bytes, while animated TGS and video WEBM
stickers supply only the JPEG/WebP thumbnail provided by Telegram. The agent is
explicitly told when it sees a thumbnail and must not infer animation or motion.
Missing thumbnails produce an unavailable-image note without downloading the
animation. Stickers and previews share the image byte budget and RAM-only path.
These rules also apply to earlier stickers selected in group reply context.
Inline custom emoji inside text are not fetched as separate sticker images.

GIF and silent MP4 messages delivered as Telegram animations, including GIF files
sent as documents, download the original and extract only its first frame in RAM.
No later frames are decoded or sent to the model. The JPEG is resized to at most
768 pixels on its longest edge and shares the existing 20 MB image budget.
If the original is too large or cannot be decoded, Telegram's thumbnail is used
when available; otherwise the agent receives an omission note. First frames and
fallback thumbnails are labelled explicitly, with motion unavailable.
For stickers, reaction GIFs and memes, the agent interprets the conveyed emotion
or reaction in conversation; explicit captions and questions take priority.

Replies render Markdown directly into ordinary Telegram messages with explicit
formatting entities. The renderer is adapted from
[slop-writer 0.5.1](https://github.com/Lancetnik/slop-writer/blob/e80d5b871cfa4816e9db999d55e30c812364cc6d/src/slop_writer/markdown.py)
under [Apache-2.0](src/dashulya/bot/slop-writer.LICENSE). It supports emphasis,
source links, headings, lists, quotes, code, `^^underline^^` and `||spoilers||`;
tables become aligned monospace blocks. HTML tags remain literal text.
Formatting and UTF-16 offsets survive message chunking. Failed sends are not retried
through a second transport, avoiding duplicate replies after network errors.

Messages reach the AI with their original Telegram send time in local time,
for example `25 October 2026 (Sunday) 12:16:50`. Both agents' system prompts
declare the timezone; message context contains local dates and times without
repeating that declaration. Repeated local hours include a numeric UTC offset
to distinguish their occurrences. The application computes the weekday. `BOT_TIMEZONE`
defaults to `Europe/Moscow` and accepts IANA timezone names. Log timestamps
remain in UTC.
Logs correlate model and tool calls with the triggering chat/message. Function tools
have start/end events with their name, call ID, duration and outcome; model events
include timings, input/output/reasoning token counts when supplied, and input
image counts. Successful model-call events also include the raw `agent_output`,
`message_count` from its JSON `messages` list (null when unavailable), and
`tool_call_count`. Output is logged before schema validation, including invalid
JSON or an oversized message list; model-call success does not imply valid output.
`AI reply validation failed` records validation error types and field paths.
Schema rejection does not automatically request another model response.
An AG2 observer logs provider-hosted tools, including Code Interpreter, as
`AI builtin tool call observed` and `AI builtin tool result observed`, with
`tool_name`, `tool_call_id`, chat/message IDs and `tool_status` when supplied.
These events arrive with the provider response and do not measure execution
duration. Code and tool results stay excluded. `tool_call_count` counts only
function tools; builtin calls have their own observer records.
Media loading and Telegram delivery have separate timings. Generated and
delivered replies record their message count; delivery also records the number
of confirmed Telegram chunks. `AI reply delivery failed` distinguishes `render`
and `send` stages and retains the confirmed chunk count without retrying.
Confirmed reactions log `Reaction delivered` with chat/message IDs, normalized
emoji, duration and `outcome=sent`; rejected or unconfirmed attempts retain their
separate outcomes. Rate limits log `Reaction rate limited` with `retry_after`
in seconds and `outcome=rate_limited`. Typing errors produce a redacted warning.
The typing task is cancelled and awaited when generation ends, so it cannot
delay reply delivery.
Agent output text is included with credential redaction. Input conversation text,
image bytes, tool results and provider error bodies remain excluded. History SQL
is logged explicitly: `chat_history` tool events include the `sql` list and call
ID; `History SQL query started/completed` events add each query's index, batch ID,
trusted chat/message IDs, duration, outcome, returned row count and truncation.
Credential redaction applies to SQL too; other tool arguments stay excluded.

## Getting started

You need Docker with Compose, a Telegram bot token and an OpenAI API key.
Run commands from the project root. Compose starts PostgreSQL with pgvector,
ParadeDB `pg_search` 0.26.4 and
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
POSTGRES_APP_PASSWORD=different_generated_random_hex_password
```

Create a bot through [@BotFather](https://t.me/BotFather). Use your numeric
Telegram user ID for `OWNER_USER_ID`. The existing local setup already has
credentials. Your `.env` stays out of Git and the Docker build context. The reply
and addressing models default to `gpt-6-luna`; model, timezone and local storage
settings are listed in [.env.example](.env.example).

Set `KEENABLE_API_KEY` to enable [Keenable MCP tools](https://docs.keenable.ai/mcp-server).
The reply agent receives `search_web_pages` (five results with bounded snippets) and
`fetch_page_content` (one public URL per call, up to 12000 characters). Page reads
use the indexed copy by default; `live=true` fetches current or unindexed pages.
AG2's `MCPToolkit` connects to `https://api.keenable.ai/mcp`, authenticated with
`X-API-Key`, and discovers the two allowed tools on the first reply. Discovery is cached;
if it fails, the bot can still reply without web tools and discovery can recover
on a later turn.
Without the key, those tools are absent and conversations and chat memory
remain available. The addressing classifier never receives web tools. Search and page contents are untrusted evidence;
answers should cite their original URLs. Web-tool failures are reported without
provider payloads, and each call has a 30-second deadline. MCP tool rejections
log `mcp_is_error`, a safe `error_category`, and recognized error codes or HTTP
statuses when present. Unrecognized responses retain the `unknown` category;
raw provider messages and query arguments are omitted.

Generate each database password separately with `openssl rand -hex 32`. Keep
`POSTGRES_PASSWORD` stable for an existing database. `POSTGRES_APP_PASSWORD`
authenticates the nonadministrative runtime login `dashulya_app`; the migration
service creates it and sets its password. Compose supplies the internal `DATABASE_URL`; direct Python
and CLI runs need a URL for a reachable PostgreSQL server with pgvector and
`pg_search` 0.26.4 installed.

For a PostgreSQL server outside Compose, first create the target database using
an administrator and install the pgvector and `pg_search` 0.26.4 extension packages
on the server. Then run:

```sh
# MIGRATION_DATABASE_URL uses the database administrator; DATABASE_URL uses dashulya_app.
uv run python -m dashulya.main.migrations
```

Set `MIGRATION_DATABASE_URL` and `POSTGRES_APP_PASSWORD` in `.env` before this
command. Repeating it checks applied migration checksums and updates the runtime
password without rerunning migrations. Do not edit an applied SQL file: add a new
numbered migration. Runtime startup refuses a missing or outdated schema.
The target database is dedicated to Dashulya: the role migration revokes PUBLIC
CREATE on its application schema and PUBLIC execution of `pg_catalog.set_config`.
The reader role is NOLOGIN and entered with SET LOCAL ROLE; it needs no password.
It can SELECT the scoped temporary history view and cannot read base tables.
See [the SQL access decision](docs/adr/0006-scope-model-sql-with-postgresql-views.md).

### 2. Start the agent

```sh
docker compose up -d --build --wait
```

Compose first runs the `migrate` service under the PostgreSQL administrator.
It applies the numbered SQL files in [database/migrations](src/dashulya/database/migrations),
records checksums, adopts an existing unversioned schema and provisions runtime
roles. The bot then verifies the migration journal, opens the vector cache, attempts pending
embedding preparation, verifies the Telegram bot identity and starts polling.
Embedding errors leave canonical history available and indexing queued for retry.
Docker's agent health check reflects completed startup, rather than complete
embedding coverage.

The [database image](Dockerfile.postgres) adds ParadeDB to PostgreSQL 16 and
preserves the existing pgvector 0.8.7 binaries and extension files. Migrations
preserve originals and embeddings; the follow-up cleanup migration removes the
retired SQL posting tables.
Lexical indexes are built lazily or by background preparation for each chat's
exact authorized snapshot, with eight idle snapshots retained on disk. A new
snapshot has an index-building cost; repeated searches reuse it. See
[the search decision](docs/adr/0008-use-chat-isolated-paradedb.md).

The database base image moves from Debian 12 to Debian 13. For an existing stand,
stop the agent and any other writers, retain a full database backup, then start
only PostgreSQL with `docker compose up -d --build --no-deps --wait postgres`.
Check `pg_database_collation_actual_version(oid)` against
`pg_database.datcollversion`; for a changed libc collation, rebuild that database's
indexes with `REINDEX DATABASE` and then `ALTER DATABASE ... REFRESH COLLATION VERSION`.
After this maintenance, use `docker compose up -d --wait` to run migrations and
restart the agent. Keep the previous image and backup until validation completes.

Open the bot in Telegram and send `/start`, then a message. Inspect and stop the
stack with:

```sh
docker compose ps
docker compose logs -f agent
docker compose down
```

The [Compose file](docker-compose.yaml) persists originals, the registry, native
search snapshots and embeddings in `var/postgres/`, and logs in `var/logs/`. PostgreSQL
has no published host port. These directories survive container recreation and
are excluded from Git. Back up the running database with `pg_dump`; copy its data
directory only while PostgreSQL is stopped. Runtime, maintenance and storage
tests use PostgreSQL; legacy storage adapters and migration commands have been
retired. Existing originals, cached embeddings and preserved source snapshots
remain intact.

Compose bind-mounts `./src` read-only at `/app/src`. `PYTHONPATH` makes these host
sources take precedence over the installed package in the image. After editing
Python files, reload the agent with `docker compose restart agent`; Python loads
the updated modules when its process starts. Rebuild the image when dependencies
change. The source mount adds no files from the rest of the repository.

For recording ordinary group messages and name-based addressing, make the bot
an administrator, or
[disable privacy mode](https://core.telegram.org/bots/features#privacy-mode)
through @BotFather and re-add the bot. Telegram does not deliver other bots'
messages to it. After the owner's first `/start`, received group participants'
messages are saved, alongside
owner-private messages and delivered replies; commands and replies remain
restricted to the owner. Text, captions and basic attachment metadata are
retained; passive capture does not download attachment contents. Authorized
picture requests use the RAM-only image path described above. Coverage and provenance belong
to each local dataset under `evals/datasets/`. The [original memory ADR](docs/adr/0001-openviking-memory.md)
records an 18-query historical pilot, not a full-corpus quality result or the
final decision for the current comparison.

## Development

Binary images and videos under `assets/` use [Git LFS](https://git-lfs.com/).
Install the Git LFS client before cloning. In an existing checkout, run:

```sh
git lfs install --local
git lfs pull
```

The tracked formats are listed in [.gitattributes](.gitattributes). Add asset
files with `git add` as usual; Git stores LFS pointers and uploads their content
on push. Keep scripts, prompts and metadata in ordinary Git. Full-size sticker
generations and ZIP bundles remain local, as listed in [.gitignore](.gitignore).

Run `just preview` and open <http://localhost:8765> to inspect saved group
messages alongside their selected reply context and AG2 events. The local
[context preview](preview/README.md) reads the existing Compose PostgreSQL
database in read-only mode, without Telegram or model calls.

GitHub Actions runs the development checks on pushes to `main` and pull requests.
Tests and package builds run on Python 3.12, 3.13, and 3.14 without live Telegram
or model API calls. PostgreSQL integration tests use a disposable schema per test;
the bootstrap check also creates and drops a separate database from `template0`
to verify extension installation. The test administrator needs CREATE DATABASE
and CREATE ROLE privileges.
Set `TEST_DATABASE_URL` to an isolated PostgreSQL database with pgvector and
`pg_search` 0.26.4 before running `just test` or `just check`; CI provisions it
automatically. The full
suite fails immediately if this variable is missing. Never point it at
application data.

Run `just test-local` for the PostgreSQL-independent tests under `tests/local/`.
These include focused model, delivery, formatting and tokenizer checks.
`tests/integration/` exercises real PostgreSQL storage and dispatcher scenarios,
including group access, activation and delivered-message capture. Both sets run
with `just test`; each test belongs to one set.

Both commands use pytest with pytest-cov and report line and branch coverage for
`src/dashulya/`, including missing lines. The suite uses pytest assertions and
fixtures, with pytest-asyncio running async tests in a separate event loop per
test. Coverage records each test's context; HTML reports show which tests executed
a line. The `ctrace` core keeps these contexts available on Python 3.14,
whose default `sysmon` core does not support them. No minimum coverage threshold
is enforced. For an HTML report, run
`uv run --locked --no-dev --group test python -m pytest --cov-report=html` with
`TEST_DATABASE_URL` set, then open `htmlcov/index.html`. Coverage data and reports
are ignored by Git.

The test dependency group includes [dirty-equals](https://dirty-equals.helpmanual.io/)
for nested expectations and [Hypothesis](https://hypothesis.readthedocs.io/)
for generated cases such as Unicode chunking. Write Hypothesis tests as ordinary
pytest functions so repeated in-process runs keep the same executor. Keep original text, identities,
scope and collection sizes exact. Use partial dictionary matchers only for fields
outside the behavior being tested; constrain variable values such as durations.

Run `just mutate` for mutation testing with
[mutmut](https://mutmut.readthedocs.io/en/latest/). It mutates `src/dashulya/` and
uses both test sets, so `TEST_DATABASE_URL` must point to an isolated test
database. To focus a run, pass a quoted mutant pattern, for example
`just mutate 'dashulya.ai.addressing.*'`. Use `just mutmut results` to list
stored outcomes and `just mutmut browse` to inspect mutants interactively.
Results and generated source copies stay in the ignored `mutants/` directory;
later runs reuse them. Coverage collection is disabled during mutation runs.
Mutation testing is separate from `just check` and CI. Run it on Linux or macOS;
Windows requires WSL because mutmut uses fork.

Coverage overlap identifies candidates for consolidation, not redundant tests.
Inspect their assertions and surviving mutants, then compare the same mutations
after a change. Distinguish behavioral gaps from equivalent mutations and timing
failures; a selected mutation score does not establish complete test coverage.

Architecture and contributor instructions live in [AGENTS.md](AGENTS.md).
See [reference bots](docs/design/chat-memory.md#reference-bots) for source comparisons.
The [repository layout](docs/design/repository-layout.md)
separates design documents, evaluation infrastructure and local application state.
PRDs live in GitHub Issues; datasets, evaluation results and research drafts are
local files excluded from Git. See [evals/](evals/README.md) for evaluation setup.

Dishka constructs the shared dependency graph for aiogram and AG2. The
[dependency injection design](docs/design/dependency-injection.md) describes
module responsibilities, scopes, resource cleanup, and provider overrides in
tests. Domain terms are defined in [GLOSSARY.md](GLOSSARY.md).

## License

Dashulya is licensed under the [MIT License](LICENSE).
The adapted slop-writer renderer retains its [Apache-2.0 license](src/dashulya/bot/slop-writer.LICENSE).
