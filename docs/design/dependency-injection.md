# Dependency injection

The top-level `main` package owns environment configuration, assembly, every
provider, and framework integration setup. Telegram handlers and AG2 tools use
standard Dishka decorators and dependency annotations directly. Domain, memory,
storage and database infrastructure stay independent of DI. Lower modules never import `main`; the
launch entrypoint and tests may do so.

## Module responsibilities

```text
main/
  config.py              Environment loading and validation
  container.py           Provider registration and shared scope configuration
  runtime.py             Startup, polling, logging, and container cleanup
  providers/
    ai.py                Connect AI factories to configuration and DI
    database.py          Shared PostgreSQL pool lifetime
    domain.py            Group repository and domain registration
    memory.py            Select and construct the memory backend
    telegram.py          Bot session, middleware, and dispatcher
ai/
  agents/
    models.py            Reply/classification model policy
    factory.py           Agent identities, prompts, schemas, and construction
  tools/memory.py        Bounded recall over the async backend interface
  tools/history.py       Canonical original reads, counts and pagination
  embeddings.py          Embedding provider/model policy
  message_format.py      Incremental author/date declarations and message records
  dialogue.py            Reply context, conversation history, turn locking, and reset
  addressing.py          Direct-address recognition and classification
  prompts.py             Persona and response instructions
domain/
  groups.py              Group, GroupRepository, and GroupRegistry
  messages.py            GroupMessage, AddressingDecision, ReplyBatch, Reply and ConversationKey
  memory.py              MemoryAccess, canonical messages and memory interfaces
```

`domain` has no application or infrastructure imports. `GroupRegistry` owns
observed group migrations, membership changes, persistent activation, blacklists,
and bounded group-list output.
`storage.groups.GroupStore` implements the asynchronous `GroupRepository` using
PostgreSQL. `DatabaseProvider` owns the single application-scoped async pool,
verifies its migration journal and closes it after dependent resources are finalized.
The separate `main.migrations` entry point applies versioned SQL under an
administrative connection; the runtime login performs no schema setup.
The domain provider aliases the registry implementation to its interface.

`MessageStore` implements `MessageRecorder` and `ChatHistory`. Imported and
received originals share its canonical table, read by immediate reply context,
`chat_history` and `search_memory`. All storage operations use PostgreSQL.
[ADR 0005](../adr/0005-unify-memory-in-postgresql.md)
records the storage decision. `ChatMemoryService` binds trusted `MemoryAccess`
to separate recall and canonical readers for each AG2 session.

`MemoryProvider` shares the store and persistent `EmbeddingCache`, and owns the
OpenAI embedding client. `HybridChatMemory` owns an incremental `SearchIndex`,
its background `IndexingWorker`, and scope-bound `ScopedRecall` readers. The
worker consumes database revisions and updates only changed reply blocks.
Persistent per-chat ParadeDB and HNSW indexes select bounded IDs in PostgreSQL;
`IndexedSearch` fuses rankings and source identities before `ScopedRecall` loads
original bodies and supporting context. A bounded SQL delta keeps pending edits
searchable without exporting their texts to Python. Canonical history remains
independent of indexing. `prepare_cached()` supports offline snapshots without
creating historical embeddings. Startup prepares pending revisions before
readiness; finalization cancels the worker and drains active searches before
closing shared clients and the pool. Provider failures retain queued work.

`AgentProvider` supplies the same store and an application-scoped `MessageFormatter`
to `Dialogue` and the memory tools. It also injects `ContextImages`, an alias of
`TelegramImages`, into `Dialogue` once. `ReplyDelivery` converts each authorized
Telegram message into a typed `ReplyRequest`, including already downloaded current
and direct-reply images. `Dialogue` selects fresh group context using the request's
chat identity and resolves saved images through `ContextImages`; callers supply
neither context dictionaries nor per-turn loading callbacks. Private requests
retain their in-memory stream. History and turn locks stay together in `Dialogue`.
The formatter receives `BOT_TIMEZONE`, the actual Telegram bot ID and
`OWNER_USER_ID`; it owns no chat authority or mutable conversation state.
The owner ID supplies provenance labels for actual numeric senders, independently
of authorship bindings. The addressing classifier receives the same owner ID.
The provider adapts optional `OWNER_AUTHOR_IDS` into explicit sender-to-user
authorship bindings for the formatter. These affect presentation only, not
sender authorization or memory access. No binding is inferred from names.
Each rendering receives the current request's sender ID as an argument for the
final `Interlocutor user id` marker. Tools receive the `user_id` supplied by `Dialogue` through an AG2
`Variable` annotation; this parameter is hidden from the model-facing schema.
Per-turn identity never mutates the shared formatter.
The same `BOT_TIMEZONE` setting reaches both agent factories as a system-prompt
fragment. Context and tool text contain local dates and times without a timezone
declaration; numeric offsets disambiguate repeated local hours.
Incoming/outgoing capture share conversion in `bot/messages.py`. Group messages
from all participants, owner-private messages and successful deliveries use
actual Telegram IDs; other private senders are excluded. `TelegramProvider`
attaches outgoing capture to the shared bot session.
Both capture middleware instances receive `GroupRegistry` and require the
group's persisted `capture_enabled` permission before processing originals or
responses. The first owner `/start` enables it atomically with replies; `/stop`
leaves it set. Group observations and migration preserve it; departure removes it.
It supplies `AccessMiddleware` with the owner ID and `GroupRegistry`.
Private messages, activation and other commands stay owner-only. `/stop`, `/ban`, `/unban` and `/settings`
also admit the current chat's administrators, verified through the event's Telegram
bot using `getChatAdministrators`. Failed lookups deny access; named administrator
rights are not cached. Telegram's authenticated `sender_chat` for the current
group allows anonymous administrators to pause replies and manage bans without
granting `/start`. Human participants in any activated group may proceed to normal
addressing unless banned; bot/missing senders and
unauthorized private chats remain silent. Capture and group tracking still
precede this gate. Activation grants dialogue access within the group without
expanding memory authority.
The same gate checks group activation through `GroupRegistry`; inactive groups
admit the human owner's `/start` and owner/admin `/stop`, `/ban`, `/unban` and `/settings`. `/start` and
`/stop` persist the current group's activation or deactivation in PostgreSQL
before confirming it.
Ordinary observations preserve activation, migration
transfers it, and departure removes it with the registry entry.

`/ban @username` and `/unban @username` resolve a unique human sender from their latest canonical
message metadata in the current chat through `GroupRegistry` and `GroupStore`.
The blacklist stores numeric sender IDs separately from originals and is checked
before addressing and generation. Administrative commands retain their access.
The store merges bans during group migration and cascades their deletion on
departure. Capture continues for banned senders; username lookups and bans do not
change saved originals or memory authority.
Unbanning deletes only the current group's row for the resolved numeric user ID;
both commands share argument validation and lookup through one handler.

AI factories accept ordinary arguments; tools declare injected dependencies
with `FromDishka` and the AG2 integration's `@inject`. Model choices, timeouts,
retry policy, agent identities, prompts, and response schemas live in `ai`.
Providers adapt validated application settings to those inputs, attach DI
middleware, and supply dependencies. `OPENAI_MODEL` selects the reply model;
`OPENAI_DECISION_MODEL` selects the addressing model.
The reply prompts use Markdown headings and semantic XML sections for persona,
reaction initiative, input provenance, reply workflow, delivery rules, shared
input/citation semantics, historical evidence policy and web policy. Both agents
receive shared provenance and message-format system-prompt fragments; context and memory
blocks contain data with incremental author/date declarations and time/JSON
headers followed by original text. Tool
docstrings own call parameters, result meanings and recovery contracts; these
are not repeated in the shared prompts. The addressing classifier has its own
intent rules alongside those shared fragments. User inputs and tool evidence use
`external_input` envelopes with literal source text; native assistant records
remain unwrapped. Source labels describe origin, never extra permissions.
The reply-agent factory also attaches `ReplyTrace` middleware for correlated
model/tool timings, outcomes and token counts, and an AG2 observer for builtin
tool call/result events. The observer logs their names, call IDs, request IDs and
provider status without code or result bodies; it infers no execution duration
from events delivered with the provider response. The model's `tool_call_count`
counts function tools only. Both read trusted `MemoryAccess` through an AG2
`Inject` annotation. The middleware records raw model output before validation,
its message count when readable, and the number of function tool calls. Input conversation
bodies and tool results stay excluded; credential redaction also covers
model output and explicitly logged history SQL. Other tool arguments stay excluded.
History-call events include the SQL list and call ID. The SQL adapter logs
individual query indexes, batch IDs, trusted request IDs, timings, outcomes and
returned row counts without selected data. Tool failures include full structured
exception chains and stack frames. Invalid arguments use warning level; provider, database
and unexpected failures use error level. Successful and cancelled calls use info
level. Before a memory-tool error reaches
model input, the middleware replaces AG2's raw traceback with a plain string:
fixed input-validation reasons or field/type details, otherwise an error type
and safe recovery guidance. The original exception remains on the internal error
event for framework error handling; only the string result is sent to the model.
`Dialogue` binds tool authority without running
an automatic recall; the agent chooses which tools to call.

The reply agent has an AG2 `ResponseSchema[ReplyBatch]` for up to three ordered
`messages`, each a `Reply` with `text`, nullable `sticker` emotion and nullable
`reply_to_message_id`.
`Dialogue` validates each parsed target against the same group-context snapshot
that seeded the stream; private replies retain the default target. The result
crosses into `bot` as the domain's `ReplyBatch`, with no AG2 objects. Delivery
holds the conversation lock through all messages and their chunks; successful
chunks are recorded individually, and a send failure stops the remaining batch.
Schema validation failures log error types and field paths without input values
or a model retry. Generated/delivered events record message counts; delivery
also records confirmed Telegram chunks and distinguishes rendering from send
failures.

Both model-facing memory tools omit the `chat` selector and keep their scope
bound through `MemoryAccess`. A single-row result within a `chat_history(sql=[...])`
batch can resolve the selected original's private file references through `HistoryReader.original()`, which
reuses the scoped canonical reader and cutoff. It returns the original's image
or preview as native AG2 tool-result media through the shared `ContextImages`
adapter. `Dialogue` supplies a fresh `HistoryImageTurn` dependency for every ask;
it accounts for retained/current images and serializes subsequent image reads
within the remaining budget. Private stream eviction also handles tool-result
images. These reads grant neither cross-chat authority nor new reply targets.

All reactions, including those on the triggering message, use the injected
`ai/tools/reactions.py` tool. The application-scoped `TelegramReactions`
implements the domain's `MessageReactions` interface for that tool.
`set_reaction` accepts only message ID and emoji, with no model-facing chat
override or reply-context target restriction. `Dialogue`
passes a fresh `ReactionTurn` through the `reaction_turn` AG2 dependency for every
ask. `get_turn_reactions` combines its `Inject` annotation, the `chat_id`
`Variable` and the application-scoped `MessageReactions` from Dishka. The tool
receives that bound `TurnReactions` through `Depends` and delegates to its `set()`
method. Invalid or missing trusted inputs produce the same rejected result
without calling Telegram. `ReactionTurn.set()` owns per-turn attempts and
unconfirmed deliveries and serializes parallel reaction calls, allowing a
tool-only turn to finish silently and suppressing retries of any unconfirmed
target. The factory opens a Dishka tool request to resolve the
application-scoped adapter; the returned object holds no request-scoped resource.
No mutable chat identity lives on the shared agent or service.
Known Telegram rejection descriptions become specific `reason` codes and fixed
English `message` explanations with recovery guidance. Invalid reactions, reaction
limits, missing messages/chats, membership and permission failures stay distinct;
`REACTION_INVALID` does not by itself distinguish an unsupported emoji from a
chat restriction. Unknown descriptions retain an explicit unrecognized-error
result. Raw descriptions never enter model history or logs; cancellation
propagates. The tool description lists supported emoji and guides recovery after
rejection. The agent can correct the call or explain the
failure itself; delivery adds no canned reaction error response. Reactions
create no synthetic delivered text messages in chat history.

When `KEENABLE_API_KEY` is configured, `AgentProvider` registers the
`KeenableToolkit` assembled by `ai/tools/web.py`. It uses AG2's `MCPToolkit`
with `MCPServerConfig` to connect to `https://api.keenable.ai/mcp` over Streamable
HTTP, authenticating with `X-API-Key` and exposing only `search_web_pages` and
`fetch_page_content`. AG2 discovers and caches the remote tool schemas and owns
MCP session and HTTP client cleanup; the addressing agent has no web tools.
Discovery has a 30-second deadline. A discovery failure logs safe diagnostics
and omits web tools from that reply; a later turn can discover them again.
Calls explicitly request `output_format=json` instead of the server's default
formatted text. All model-supplied arguments otherwise pass through to Keenable,
which owns input validation; no local parameter allowlist or validation rules
duplicate its discovered schemas. When omitted, request defaults select five
search results, 4000-character snippets and 12000-character page content. Explicit
model options take precedence. Page reading defaults to indexed content, with
explicit `live=true` for current or unindexed pages. Regardless of provider
options, model-facing results retain at most five bounded snippets or 12000
characters of page content. The middleware bounds content and call duration,
wraps evidence as untrusted web input, sanitizes provider failures
before they enter AG2 history, and propagates cancellation. AG2 exposes MCP
rejections as string errors; diagnostic extraction retains only recognized
code/status hints, with unknown formats left unclassified. Raw provider errors
never enter logs or model history. The key is excluded from settings repr and
included in log redaction.

`ReplyDelivery` remains in `bot`: it owns Telegram typing, redacted failure
handling, and entity-based Markdown delivery through the slop-writer renderer.
Each `Reply` contains either text or one of six `StickerEmotion` names, with the
same optional target. A sticker occupies one slot in the three-message batch;
text and sticker content cannot share a slot. `Dialogue` preserves the emotion
while validating its target against the selected context.
The application-scoped `TelegramStickers` resolves those names from the published
`DashulyaAI` pack by its emoji, independent of pack order, with a five-minute
cache and a shared lookup lock. Lookup is lazy, has a ten-second timeout and
rejects missing or ambiguous emoji. Telegram file IDs stay in `bot`, outside the
model schema. Delivery sends ordinary text chunks or stickers once per attempt,
and stops the batch on lookup or send failure without an automatic resend.
The sent-message middleware records confirmed sticker messages through the same
canonical capture path, including their actual IDs, times, topics, reply targets
and image references. Domain messages carry no aiogram objects.
`bot/chat_actions.py` scopes a best-effort typing task to each reply, cancels and
joins it on exit, and logs failures without propagating notification errors into
generation. `TelegramReactions` logs each confirmed tool delivery and its timing;
rejected or unconfirmed attempts remain distinguishable.
Its application-scoped `TelegramImages` dependency downloads authorized private
current/direct-reply images into bounded RAM, verifies supported signatures and
returns domain `ImageAttachment` or `ImageOmission` values. The Telegram provider supplies the
download budget from `main/config.py`; the agent provider supplies the same
budget to `Dialogue` for bounded private image history. `Dialogue` turns these
values into AG2 native image inputs and restricts group source IDs to the
selected reply context. For groups, `ReplyDelivery` defers all image loading to
`Dialogue`'s injected `ContextImages`: current, direct-parent and historical
references are loaded together from the exact selected originals. The request's
message ID supplies download priority and log correlation, never authority.
`TelegramImages` owns per-call byte reservations, gathered tasks and deadlines,
admitting ready inputs without waiting for other metadata calls. Completed
lower-priority inputs can be omitted to admit a later-ready higher-priority input;
the batch resolves final inputs in original priority order after gathering tasks.
The loader also owns shared bounded `cachetools` TTL caches of normalized images and short-lived
failures. Cache access stays on the event loop; decoding runs in a worker thread
without touching shared cache state. Cancellation joins the batch's tasks.
No conversation state is added to the shared image adapter. Binary media never
enters durable message capture; only Telegram original/thumbnail file references,
sizes and the visual-input kind are persisted. Current and saved inputs share the
same normalization and loading functions, including preview and first-frame
fallbacks. Existing photo references need no schema or data migration.

## Shared graph and scopes

`main/container.py` builds one graph with `AG2Scope`. Application dependencies
are cached in `APP`: settings, model configurations, both agents, `Dialogue`,
`GroupAddressing`, `ChatMemoryService`, the database pool, vector cache/client, `GroupStore`,
`GroupRegistry`, `ReplyDelivery`, the Telegram bot, and the dispatcher. Application resolution
uses an async lock to keep lazy construction unique across concurrent events.

The built-in aiogram integration opens a new `REQUEST` container for each event.
The Telegram provider declares `TelegramObject` and `AiogramMiddlewareData`
context dependencies with `AG2Scope.REQUEST`; the optional built-in
`AiogramProvider` uses Dishka's standard `Scope` enumeration and therefore cannot
be registered unchanged in this graph. Telegram handlers use
`dishka.integrations.aiogram.inject` and `FromDishka` explicitly. Polling and
direct `Dispatcher.feed_update` calls use the same graph without an extra
startup injection step. Middleware receives its dependencies through
constructors. Group tracking and live-message capture run before owner access.
Edited-message updates go directly to capture without dialogue execution.

Both AG2 agents carry `DishkaAsyncMiddleware`. Each `Agent.ask()` opens a
`SESSION`; injected tools open their own `REQUEST` beneath that session.
These scopes are separate from the Telegram event. Agents retain the application
container rather than the event container, so an AG2 turn opens its session
without inheriting a Telegram request's lifetime. `Agent[ReplyBatch]` and `Agent[bool]`
distinguish the reply and addressing roles.

The reply agent always registers `search_memory` and `chat_history` with `@tool`
and standard `dishka_ag2.inject`. Their `FromDishka[MemoryBackend]` and
`FromDishka[HistoryReader]` and `FromDishka[MessageFormatter]` parameters stay outside
the model-facing schema.
`Dialogue.reply()` passes `MemoryAccess` as the trusted `memory_access` AG2
dependency using the conversation's numeric chat identity and original request
cutoff. The session provider resolves it through `Inject`, validates that
authority, then binds both readers in `AG2Scope.SESSION`. Model-provided filters only narrow access. Groups recall
their own chat; the owner's private conversation can federate saved chats.

AG2 does not resolve annotations for Dishka providers or middleware hooks. These
boundaries pass the context to FastDepends to resolve `Inject` annotations without
reading context dictionaries. Resolution disables coercion so malformed trusted
values still fail the existing validation.

An imported archive needs an explicit numeric binding to become group-visible.
See [ADR 0003](../adr/0003-bind-chat-memory-access.md).

The shared application agent owns neither a Telegram event container nor an
unscoped model-facing memory reader. A semantic provider error does not disable
canonical history. OpenViking, a completed import and an existing vector corpus
are not prerequisites for agent construction.

`Dialogue` owns chat/topic isolation, original send-time formatting, reply
deadlines, and serialization with `/clear`. Private streams and conversation
locks stay together across Telegram events; group streams are rebuilt per
request under [the reply-context ADR](../adr/0002-rebuild-group-reply-context.md).
Private delivery holds a conversation's delivery lock through generation,
Telegram sending and local recording; the model lock also protects direct private
dialogue calls and private history reads. `/clear` acquires both in that order.
Group streams generate concurrently without a model lock. Their delivery boundary
acquires the chat/topic delivery lock only for sending and recording a complete
ready reply batch, so a slow generation cannot queue other participants' requests.
These are not request dependencies. AG2's explicit conversation scope is not
needed here.

`main.run` closes the container in `finally`, including polling failures,
cancellation, and partial startup failures. The bot session and shared database pool
are finalized by their providers. Memory resources live in the application
scope; conversation authority and bound readers live in the AG2 session scope.

## Tests and changes

`create_container(settings, *overrides)` accepts explicit Dishka providers.
Tests reuse the production graph and replace model configurations, the Telegram
bot, dialogue behavior, or the memory backend. `AgentModels` can hold AG2
scripted configurations as well as real provider configurations. Focused tests
may call AI factories and domain modules without a container; injected tools
are exercised through the real AG2 integration.

`tests/local/test_dependencies.py` exercises shared agents, model settings,
concurrent AG2 turns, per-tool finalization, cancellation, and partial startup
failure without PostgreSQL. `tests/integration/test_dependencies.py` exercises
Telegram event scopes, production startup, readiness, and resource cleanup with
the real database.
Dispatcher tests combine injection with access, tracking, routing, and delivery.
Memory tests execute injected tools through real AG2 with scripted or
mock-transport model responses. They check scope isolation, edit shadowing,
pagination/counts, embedding failures and cancellation-safe resource release.
They also cancel a real PostgreSQL query and verify pool reuse. Search,
expansion, and comparison commands await the same backend interface, with one
event loop started at the CLI entrypoint. No live Telegram or paid model
requests are needed.

When adding a dependency, register its lifetime in the relevant provider and
test behavior through the consuming module's interface. The exhaustive
hierarchies in [`.importlinter`](../../.importlinter) cover the root package,
`main`, `ai`, and `bot`. Separate contracts keep configuration inside `main`
and domain/infrastructure modules independent of DI.

## Container startup

Compose waits for its pgvector-enabled PostgreSQL service to become healthy.
Compose first runs the administrative migration service.
The runtime resolves `ChatMemoryService`, verifying the shared schema and
attempting pending preparation before accessing Telegram. With
`HEALTHCHECK_PATH` configured, startup removes a stale marker, verifies Telegram
identity and writes the current PID before polling. Shutdown removes the marker
and closes the container. Readiness verifies startup, not embedding coverage.

Group records also persist independent ordinary-reply and whisper audiences.
`AccessMiddleware` applies the selected audience before addressing or generation,
with group activation and the blacklist taking precedence. The same middleware
authorizes settings callbacks using the actual presser and panel group;
`bot/handlers/settings.py` owns panel rendering and typed callback routing.
Audience changes update only their own column. Migration transfers both values
from the old group when present; ordinary observations preserve the target values.
