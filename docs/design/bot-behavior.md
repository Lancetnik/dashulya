# Bot behavior

## Access and activation

Authorize private messages, `/start` and other owner commands using the actual
sender's numeric `from_user.id == OWNER_USER_ID`. Forwarded identities and
authorship bindings grant no access. Group `/stop`, `/ban`, `/unban` and `/settings` are also available to the current chat's creator
and administrators, checked with `getChatAdministrators` for that chat on each
request. A failed or timed-out lookup denies access and logs a redacted failure;
the configured owner's commands do not depend on this lookup.

Anonymous administrators may use `/stop`, `/ban`, `/unban` and `/settings` only when Telegram's authenticated
`sender_chat.id` equals the current group's numeric ID. A forwarded identity or
another sender chat grants no authority. Other bot senders, missing senders and
unauthorized users are silently ignored before command execution or model
invocation.

After the owner activates a group, human participants outside its blacklist may
invoke group dialogue there, subject to its reply audiences. Addressing rules still apply. Other group commands remain
owner-only.

The owner and current chat administrators may use `/ban @username` in groups,
including inactive groups, with the same authority checks as `/stop`. Accept
exactly one `@username` using ASCII letters, digits and underscores, up to 32
characters after `@`; invalid arguments receive a Russian usage message.
Resolve usernames without case sensitivity from each human sender's latest saved
identity in this group. Ignore bot and channel identities; unknown, outdated or
ambiguous names receive a Russian explanation asking for a new message from
that participant. Store the ban by numeric user ID so a username change does not
remove it. Repeated bans are idempotent.
`/unban @username` removes the resolved sender's ban in the current group, using
the same authority, argument validation and latest-username lookup. Repeating it
confirms the removal even when no ban exists. The removal persists across
restarts, leaves other users and groups unchanged, and does not activate a paused
group. Both commands accept the bot-qualified form and work while inactive.
The blacklist is scoped to the group across topics and persists in PostgreSQL.
Check it before addressing, image loading or reply generation for all human
senders, including the owner. Authorized administrative commands remain available.
Capture and saved history remain unchanged; banned users' originals can still
appear in context and memory. Private access and other groups are unaffected.
Preserve bans across metadata updates and supergroup migration; remove them when
the bot departs, along with the group registry entry.

Every group starts inactive, including existing groups on upgrade. Only the
actual human owner's `/start` in that group activates replies. The bot owner or a
chat administrator can use `/stop` to pause replies throughout the current group,
across topics. The command confirms the pause even if the group is already
inactive. `/stop` is group-only; `/start`
resumes group dialogue. Both activation and deactivation persist across restarts.
While inactive, the owner may use `/start`, `/stop`, `/ban`, `/unban` and `/settings`, and chat administrators
may use `/stop`, `/ban`, `/unban` and `/settings`; other commands, addressing decisions and reply generation remain
silent, including the owner's. Activation grants group dialogue access to human
participants outside the blacklist.

The owner's first `/start` also permanently enables message capture for that
group. Before it, incoming messages, edits, reply snapshots and outgoing replies
are not stored or queued for indexing. Later `/stop` and `/start` change replies
only. Capture permission survives restart, metadata refresh and supergroup
migration; departure removes it. On upgrade, only currently active groups can
inherit capture permission; previously paused groups need one owner `/start`.

### Per-group reply audiences

`/settings` opens an inline panel in the current group, including while paused.
The owner and current group administrators may open it and change its settings.
Anonymous administrators may open the panel using the same authenticated
`sender_chat` rule as `/stop`; button presses use the actual human presser's
current rights. Every press checks authority again in the panel's group, without
caching administrators. Failed lookups deny the change. No private-chat or
activation authority is granted.

Ordinary replies have `reply_audience=all|admins`; whispers independently have
`whisper_audience=all|admins|disabled`. Both default to `all`, including for
existing groups on upgrade. `admins` includes the group creator, administrators
and the configured human bot owner. `disabled` blocks whispers even for the
owner. Activation and the blacklist take precedence over either audience;
authorized administrative commands remain available. Apply these checks before
addressing classification, image downloads or generation. Denied requests stay
silent. Ordinary addressing rules still apply to allowed public requests.

Both settings cover all topics of one group and leave other groups and private
access unchanged. `/stop` pauses both kinds of reply; `/start` resumes with the
saved audiences. Metadata refresh, restart and supergroup migration preserve
the settings; departure removes them, so rejoining restores the default audiences
and requires owner activation. `/groups` shows nondefault audience restrictions.
Capture and shared memory continue to include public originals from participants
who cannot invoke replies; hidden questions remain excluded from shared history.

## Routing and capture

Authorized private messages and explicit group mentions/replies to this bot
invoke the reply agent directly. Group messages naming Dashulya, including
“Дашуль” in any letter case, first pass an AG2 OpenAI Decisions classifier with
bounded chat/topic context. A negative decision or classifier failure stays silent.
The classifier favors replying when intent is uncertain. A direct call by name,
question, request or social remark counts even when the request is infeasible or
inappropriate; the reply agent handles those limitations. Clearly third-person
discussion, quoted examples and explicit requests not to answer the current
message stay silent. A real direct address takes precedence over surrounding
discussion about the bot.
`/clear` resets the current private conversation.

Group tracking runs before sender access checks, including before activation.
Message capture also precedes sender access, but requires the group's first
owner activation and continues while replies are paused afterward. Passive
messages update group metadata; membership updates also maintain the registry.
In approved groups, save received messages from all participants and successful
deliveries. Save owner-private messages; exclude other private senders.
Index saved messages with background cached embeddings. Capture,
edits and worker recovery follow the
[incoming-message rules](chat-memory.md#incoming-group-messages).

## Conversation context and delivery

### Private whispers inside groups

Whispers use the ordinary `AI request started`, `AI reply generated`,
`AI reply delivered` and failure events, with confirmed chunk counts and timings.
Their lifecycle and model/tool traces share the Telegram update ID, sender,
chat/topic, ephemeral message ID and `is_whisper=True`. Ephemeral IDs remain
separate from canonical original-message IDs and memory cutoffs. Raw model
outputs and history SQL follow the ordinary operational logging policy;
exclusion from shared memory does not suppress those logs. Skipped requests
record their reason, including expired reply windows and missing permissions.

At startup, register `/whisper` with `BotCommand.is_ephemeral=True` in the
all-group command scope. Human participants outside the group's blacklist may
use it while the group is active and its whisper audience allows them; it does not grant private-chat or owner-command
access. Route every ephemeral group update, including unsupported commands,
through hidden delivery before ordinary handlers. Unsupported commands and empty
questions receive private usage guidance without performing administrative work.

Each whisper starts an independent stream seeded with up to 20 public originals
and their bounded reply ancestors from the same group/topic, strictly before
the request's send time. Exclude originals edited at or after that cutoff.
Whispers have no canonical public message ID: Telegram uses `message_id=0` and
a separate reusable `ephemeral_message_id`. Memory tools retain the original
group scope and send-time cutoff, including for the owner; hidden delivery does
not acquire the owner's federated private-memory access. Questions and responses,
including reply snapshots of public `/whisper` commands, never enter shared
capture, indexing or public context. No private history is retained between turns.
Do not send a public typing indicator or allow the reaction tool during a whisper.

Deliver text chunks and stickers with `EphemeralMessageParameters.receiver_user_id`
bound to the actual sender, ignoring model-selected public reply targets. Check
the bot's own membership for each request. An administrator sends independently
of the incoming hidden message so longer generation remains deliverable. Other
bots reply using `ReplyParameters.ephemeral_message_id` within Telegram's
15-second window, reserving time for hidden timeout guidance. Already expired
requests remain silent; cancellation stops generation and delivery. Delivery
errors never trigger public fallback or automatic retries. Ephemeral delivery is
not guaranteed, particularly for offline users, and messages can disappear when
the Telegram client restarts.

If a client sends `/whisper` as a public command, an administrator bot attempts
to delete it and answers privately; deletion is best effort and cannot undo
prior exposure. Without administrator status and an incoming ephemeral ID, there
is no eligible hidden reply. Send only public setup guidance, without generating
or quoting a private answer. Select the registered hidden
command in a supporting Telegram client for an initially private question.

### Ordinary replies

Reuse one AG2 `Agent`. Follow
[ADR 0002](../adr/0002-rebuild-group-reply-context.md) for group context selection,
roles, incremental author/date declarations, media and ordered reply targets.
Private histories remain in memory until restart or `/clear`. Serialize private
generation, delivery and recording within a conversation. Group requests generate
concurrently in separate streams; serialize only sending and recording of each
complete reply batch within its chat/topic. Ready group replies may overtake slow
generations, while messages from different batches cannot interleave.

Carry original Telegram `message.date` in `BOT_TIMEZONE`. Both agents' system
prompts declare the IANA timezone; user context omits that declaration. Private
inputs use `DD Month YYYY (Weekday) HH:MM:SS`, adding a numeric UTC offset during
repeated local hours. Compute the weekday after conversion and preserve offset
changes. Interpret relative dates using send time, including delayed messages.

Render Markdown using the slop-writer-based renderer in
[`bot/formatting.py`](../../src/dashulya/bot/formatting.py). Preserve formatting
and UTF-16 offsets when chunking; tables become monospace blocks. All replies
use ordinary Telegram messages with explicit entities. The application marks
each reply with more than 600 rendered Unicode characters, or any code block or
table, with an `expandable_blockquote` spanning its body. This policy does not
depend on model markup. Keep the first prose paragraph outside that span only
when it has at most 240 characters and two lines and other paragraphs follow.
Headings, lists and quotes stay inside. Every continuation chunk is marked for
collapse, including short final chunks. Telegram controls the visible preview
and expansion arrow; do not add a separate disclosure title or padding.

Preserve text, links, authored emphasis, inline code and preformatted languages
inside the collapsible span. Telegram cannot nest quotations, so render inner
quoted passages as italic spans instead. Preserve all UTF-16 offsets and the full
rendered text in recorded history. Use the ordinary message renderer to avoid
the heavier default font observed for Rich Messages in the user's client.
Keep the typing indicator and Telegram-length limits. Attempt each chunk once, including
after ambiguous network errors. Keep provider errors and credentials out of
logs and replies. The Jester persona and response instructions live in
[`ai/prompts.py`](../../src/dashulya/ai/prompts.py).

## Group registry

Preserve observed groups across restarts, refresh names, remove departed groups
and replace old IDs on supergroup migration. Telegram supplies no complete chat
list; missed updates during downtime can leave the registry stale.

Treat both absent `my_chat_member` status and a service message whose
`left_chat_member.id` is this bot's ID as departure. The service message must
remove the group rather than re-register it, regardless of which event arrives
first or who removed the bot. Another participant's departure preserves this
bot's group membership and activation.

Preserve activation across restarts, metadata refresh and migration. Departure
removes activation, so rejoining requires a new owner `/start`. `/groups` shows
activation status. Registry migration does not establish full chat-memory
migration continuity; see [the memory design](chat-memory.md).
