<div align="center">
  <a href="https://t.me/dashulya_ai">
    <img src="assets/avatar.png" width="190" height="190" alt="Dashulya: a platinum-haired cyber assistant in a pink coat, framed by a pink ring" />
  </a>
  <h1>Dashulya</h1>
  <p>
    <a href="https://t.me/dashulya_ai"><strong>@dashulya_ai</strong></a><br />
    <sub>also <a href="https://t.me/dashulyai_bot">@dashulyai_bot</a></sub>
  </p>
  <p>
    <strong>Спрашивай, котик💋</strong><br />
    <em>Ask away, kitten.</em>
  </p>
  <p>A Telegram assistant with memory and a sense of humor.</p>
  <p><a href="https://t.me/dashulya_ai"><strong>Open in Telegram ↗</strong></a></p>
  <p><sub>Early prototype · Owner-only access</sub></p>
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
and [Ostrikov's chat](https://t.me/aostrikov_agents_chat). The platinum bob,
mechanical neck, pink coat, and cheek heart became her Telegram face.

> Hot robots in the chat 💋

## What she can do today

- **Talk in private.** Keep the current conversation in context, with a fresh
  start whenever you send `/clear`.
- **Find earlier discussions.** Search an imported chat-history snapshot and
  return original messages with Telegram links and surrounding replies.
- **Follow memories back to their sources.** Search original messages,
  discussion extracts, and source-backed summaries or facts through one tool.
- **Keep track of groups.** List groups the bot has observed, preserving the
  registry across restarts.

With a history index connected, try questions such as:

> Where did we discuss RTK and pytest?
>
> Who was considering reusing Osinara for a bot with custom skills?

Memory currently comes from a local imported snapshot. Live history sync,
automatic summary generation, semantic search, and web search are not
implemented yet.

## Commands

| Command | What happens |
| --- | --- |
| `/start` | Meet Dashulya |
| `/help` | See the available commands and how to use them |
| `/groups` | List the groups she knows about |
| `/clear` | Reset the current private conversation |

The current build responds only to the user configured as its owner. AI replies
are available in private chats; group commands are available to the owner.
Group mentions and replies do not start an AI conversation.

Private conversation context lasts until `/clear` or a bot restart. The known-group
registry and the local history index persist between restarts.

Replies render Markdown emphasis, clickable source links, headings, lists,
quotes, and code through Telegram text entities. Long replies keep formatting
when split into messages; literal HTML and incomplete Markdown stay readable.

Messages reach the AI with their original Telegram send time in local time,
for example `25 October 2026 (Sunday) 12:16:50`. The timezone and UTC offset are
included separately; the application computes the weekday. `BOT_TIMEZONE`
defaults to `Europe/Moscow` and accepts IANA timezone names. Log timestamps
remain in UTC.

## Getting started

You need Python 3.12–3.14, [uv](https://docs.astral.sh/uv/), a Telegram bot token,
and an OpenAI API key. Run these commands from the project root.

### 1. Install dependencies

```sh
uv sync --locked
```

### 2. Configure your bot

```sh
cp .env.example .env
```

Fill in the required values in `.env`:

```dotenv
BOT_TOKEN=your_bot_token
OWNER_USER_ID=your_numeric_telegram_user_id
OPENAI_API_KEY=your_openai_api_key
```

Create a bot and get its token from [@BotFather](https://t.me/BotFather).
`OWNER_USER_ID` is your numeric Telegram user ID, rather than your username.
Optional model, storage, memory, and logging settings are listed in
[.env.example](.env.example). Your `.env` stays out of Git.

### 3. Start chatting

```sh
uv run dashulya
```

Open the bot you configured in Telegram and send `/start`, then a message.
Press `Ctrl+C` in the terminal to stop it.

### Optional: connect chat history

Ordinary AI conversation works without a history index. To enable historical
search, point `MEMORY_PATH` at a local index and restart the bot. The default
location is `data/memory.sqlite3`.

History exports and indices stay local and are not included in a fresh clone.
See the [memory setup guide](docs/design/chat-memory.md#commands) for building
an index, and the [dataset note](docs/datasets/fastnewsdev.md) for the existing
FastNews snapshot and its coverage.

## Development

GitHub Actions runs the development checks on pushes to `main` and pull requests.
Tests and package builds run on Python 3.12, 3.13, and 3.14 without live Telegram
or model API calls. Run `just check` locally before publishing changes.

Architecture, development commands, and reference bot repositories live in
[AGENTS.md](AGENTS.md).
