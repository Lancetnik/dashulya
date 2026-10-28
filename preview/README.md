# Reply context preview

A local prototype for exploring which saved group messages enter one request's
reply context, and how those messages become AG2 events. It uses the real
`Dialogue.select_context()` and `MessageFormatter.events()` implementations.

From the project root, with the existing Compose PostgreSQL service running:

```sh
just preview
```

Open <http://localhost:8765>. The chat with the most saved messages opens
automatically. Click any message on the left; the right pane displays the
actual formatted text of each AG2 event. Selection
colors distinguish the current request, recent messages and added reply parents.
Use **Load older messages** to walk back through history. Reply references and
event message IDs navigate to their saved originals. The **JSON** tab shows
the same events as a copyable JSON array. Both panes fill the window, with no
global header or filter toolbar.

Each event shows a local day and author declaration where first applicable,
then a time/JSON metadata line and the exact original text. This bot has no
author declaration because its assistant role identifies it. Only the last
record includes `Interlocutor user id`. Later author aliases update the
declaration where observed, preserving earlier records. Truncated-chain
navigation appears in the boundary message's JSON.

The compact **Recent ↑** and **Hops** inputs adjust selection immediately for
the selected message. Recent accepts any positive message count, including
the selected message; hops accepts any nonnegative number of reply-parent edges
from each recent message. Neither has an upper limit. Zero hops disables parent
expansion. Defaults are 20 and 5, matching the running
bot. These controls change only the preview request, not the bot's settings.
Selection colors, counts, truncated-chain notes and both event views update
together.

The context total and each event card show character counts of the formatted
text, including spaces, newlines, JSON punctuation and shared metadata. Counts
use Unicode code points, so supplementary characters such as emoji are not
counted twice. The total is the sum of all event text counts; serialized event
envelopes, global prompts, tools and image bytes are not included.

Optional URL parameters select another conversation, for example
`/?chat_id=-1001234567890&topic_id=1`. Omit `topic_id` for ordinary groups,
including legacy reply threads. Set it only when simulating a forum topic;
General is topic 1. The database does
not retain Telegram's `is_forum` flag, so the preview does not guess it from old
thread IDs. Only numerically bound group collections are shown.

The preview reads current saved originals, not historical edit versions. It
shows the message context and AG2 text events, excluding global agent prompts,
tool definitions/results and native image bytes. Media remain labeled
placeholders; there are no Telegram downloads or model calls. Bot messages can
also be selected for inspection, although they do not trigger bot replies in
the real Telegram router. This is a local inspection tool, not a public app.

Database connections enforce PostgreSQL read-only mode. The preview does not
initialize/migrate the schema, write messages, index data or alter bot state.
Compose publishes only the preview on loopback; PostgreSQL keeps its existing
internal port. Stop only the preview with:

```sh
docker compose -f docker-compose.yaml -f preview/compose.yaml stop preview
```

For an already host-accessible database, use the existing Python environment:

```sh
uv run --locked python preview/server.py
```

This reads `.env` from the current directory, with environment variables taking
precedence. `DATABASE_URL` and `BOT_TIMEZONE` configure the connection and dates.
`BOT_TOKEN` supplies only the numeric bot ID, without any Telegram API call;
`OWNER_USER_ID` and `OWNER_AUTHOR_IDS` preserve configured author bindings.
Use `--port` to change the default 8765, or `--host` to change the default
loopback listener. No frontend dependencies or build step are needed.
