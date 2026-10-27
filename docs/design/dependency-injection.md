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
  message_format.py      Shared roster, calendar and message rows for context/tools
  dialogue.py            Reply context, conversation history, turn locking, and reset
  addressing.py          Direct-address recognition and classification
  prompts.py             Persona and response instructions
domain/
  groups.py              Group, GroupRepository, and GroupRegistry
  messages.py            GroupMessage, AddressingDecision, Reply and ConversationKey
  memory.py              MemoryAccess, canonical messages and memory interfaces
```

`domain` has no application or infrastructure imports. `GroupRegistry` owns
observed group migrations, membership changes, and bounded group-list output.
`storage.groups.GroupStore` implements the asynchronous `GroupRepository` using
PostgreSQL. `DatabaseProvider` owns the single application-scoped async pool,
initializes its schema and closes it after dependent resources are finalized.
The domain provider aliases the registry implementation to its interface.

`MessageStore` implements `MessageRecorder` and `ChatHistory`. Imported and
received originals share its canonical table, read by immediate reply context,
`chat_history` and `search_memory`. Offline migration is explicit; runtime startup
never opens a SQLite archive. [ADR 0005](../adr/0005-unify-memory-in-postgresql.md)
records the storage decision. `ChatMemoryService` binds trusted `MemoryAccess`
to separate recall and canonical readers for each AG2 session.

`MemoryProvider` shares the store and persistent `EmbeddingCache`, and owns the
OpenAI embedding client. `HybridChatMemory` owns background preparation and
bounded metadata engines. Startup attempts preparation; provider failures retain
queued work. Finalization cancels/joins the worker and waits for active retrieval
leases before the client and database pool close. Completed batches and pending
revisions survive restarts. SQL operations use async pooled connections; only
metadata construction/fingerprinting runs in CPU threads. Ranking uses native
pgvector and SQL BM25, then bounded RRF in Python. Cancelled SQL returns its
connection safely to the pool.

`AgentProvider` supplies the same store and an application-scoped `MessageFormatter`
to `Dialogue` and the memory tools. The formatter receives `BOT_TIMEZONE` and the
actual Telegram bot ID; it owns no chat authority or mutable conversation state.
The provider adapts optional `OWNER_AUTHOR_IDS` into explicit sender-to-user
authorship bindings for the formatter. These affect presentation only, not
sender authorization or memory access. No binding is inferred from names.
Each rendering receives the current request's sender ID as an argument to mark
`interlocutor`. Tools receive the session's context through AG2's `Context` annotation
and read the `user_id` variable supplied by `Dialogue`; this is hidden from the
model-facing schema. Per-turn identity never mutates the shared formatter.
Incoming/outgoing capture share conversion in `bot/messages.py`. Group messages
from all participants, owner-private messages and successful deliveries use
actual Telegram IDs; other private senders are excluded. `TelegramProvider`
attaches outgoing capture to the shared bot session.

AI factories accept ordinary arguments; tools declare injected dependencies
with `FromDishka` and the AG2 integration's `@inject`. Model choices, timeouts,
retry policy, agent identities, prompts, and response schemas live in `ai`.
Providers adapt validated application settings to those inputs, attach DI
middleware, and supply dependencies. `OPENAI_MODEL` selects the reply model;
`OPENAI_DECISION_MODEL` selects the addressing model.
The reply-agent factory also attaches `ReplyTrace` middleware for correlated
model/tool timings, outcomes and token counts. It reads trusted `MemoryAccess`
from each turn's context and records metadata without arguments, results or
conversation bodies. `Dialogue` binds tool authority without running an automatic
recall; the agent chooses which tools to call.

The reply agent has an AG2 `ResponseSchema[Reply]` for `text`, nullable
`reply_to_message_id` and nullable `reaction`. `Dialogue` validates the parsed target against the same
group-context snapshot that seeded the stream; private replies retain the default
target. The result crosses into `bot` as the domain's `Reply`, with no AG2 objects.

An empty text with a reaction emoji requests a reaction on the triggering message.
The application-scoped `TelegramReactions` implements the domain's
`MessageReactions` interface for both final-response delivery and the injected
`ai/tools/reactions.py` tool. `set_reaction` accepts only message ID and emoji;
it reads chat identity from the trusted AG2 `chat_id` Context Variable, with no
model-facing chat override or reply-context target restriction. `Dialogue`
passes a fresh `ReactionTurn` through AG2 dependencies for every ask. It owns
per-turn delivery outcomes and serializes parallel reaction calls, allowing a
successful tool-only turn to finish silently and suppressing retries of any
unconfirmed target. No mutable chat identity lives on the shared agent or service.
Telegram errors become fixed result codes and redacted logs; cancellation
propagates. Reactions create no synthetic delivered text messages in chat history.

When `EXA_API_KEY` is configured, `AgentProvider` also registers the search and
content-reading subset of AG2 `ExaToolkit`, assembled by `ai/tools/web.py`. The
addressing agent has no web tools. The toolkit creates an async SDK client for
each call and closes it in `finally`; no client is retained by the application
container. Tool middleware bounds requests, content and duration, sanitizes provider
failures before they enter AG2 history, and propagates cancellation. The key is
excluded from settings repr and included in log redaction.

`ReplyDelivery` remains in `bot`: it owns Telegram typing, redacted failure
handling, and entity-based Markdown delivery through the slop-writer renderer.
It sends ordinary Telegram messages once per chunk. Domain messages carry no
aiogram objects.
`bot/chat_actions.py` scopes a best-effort typing task to each reply, cancels and
joins it on exit, and logs failures without propagating notification errors into
generation. `TelegramReactions` logs each confirmed delivery and its timing for
both structured reaction replies and tool calls; rejected or unconfirmed attempts
remain distinguishable.
Its application-scoped `TelegramImages` dependency downloads authorized current
and direct-reply images into bounded RAM, verifies supported signatures and
returns domain `ImageAttachment` values. The Telegram provider supplies the
download budget from `main/config.py`; the agent provider supplies the same
budget to `Dialogue` for bounded private image history. `Dialogue` turns these
values into AG2 native image inputs and restricts group source IDs to the
selected reply context. Binary media never enters durable message capture.

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
without inheriting a Telegram request's lifetime. `Agent[Reply]` and `Agent[bool]`
distinguish the reply and addressing roles.

The reply agent always registers `search_memory` and `chat_history` with `@tool`
and standard `dishka_ag2.inject`. Their `FromDishka[MemoryBackend]` and
`FromDishka[HistoryReader]` and `FromDishka[MessageFormatter]` parameters stay outside
the model-facing schema.
`Dialogue.reply()` passes `MemoryAccess` through trusted AG2 dependencies using
the conversation's numeric chat identity and original request cutoff. The
session provider validates that authority, then binds both readers in
`AG2Scope.SESSION`. Model-provided filters only narrow access. Groups recall
their own chat; the owner's private conversation can federate saved chats.
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
The delivery boundary holds a conversation's delivery lock through generation,
Telegram sending and local recording; the model lock also protects direct
dialogue calls and private history reads. `/clear` acquires both in that order.
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

`tests/test_dependencies.py` exercises framework scope entry, concurrent AG2
turns, per-tool finalization, cancellation, production startup, and cleanup.
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
The runtime resolves `ChatMemoryService`, initializing the shared schema and
attempting pending preparation before accessing Telegram. With
`HEALTHCHECK_PATH` configured, startup removes a stale marker, verifies Telegram
identity and writes the current PID before polling. Shutdown removes the marker
and closes the container. Readiness verifies startup, not embedding coverage.
