# Dependency injection

The top-level `main` package owns environment configuration, assembly, every
provider, and framework integration setup. Telegram handlers and AG2 tools use
standard Dishka decorators and dependency annotations directly. Domain, memory,
and storage stay independent of DI. Lower modules never import `main`; the
launch entrypoint and tests may do so.

## Module responsibilities

```text
main/
  config.py              Environment loading and validation
  container.py           Provider registration and shared scope configuration
  runtime.py             Startup, polling, logging, and container cleanup
  providers/
    ai.py                Connect AI factories to configuration and DI
    domain.py            Group repository lifetime and domain registration
    memory.py            Select and construct the memory backend
    telegram.py          Bot session, middleware, and dispatcher
ai/
  agents/
    models.py            Reply/classification model policy
    factory.py           Agent identities, prompts, schemas, and construction
  tools/memory.py        Bounded recall over the async backend interface
  tools/history.py       Canonical original reads, counts and pagination
  embeddings.py          Embedding provider/model policy
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
`storage.groups.GroupStore` implements `GroupRepository` using SQLite. The
provider aliases that implementation to the domain interface and owns cleanup.
`ChatMemoryService` binds application resources to a trusted `MemoryAccess` and
returns separate `MemoryBackend` recall and `HistoryReader` canonical readers.
The selected implementation is `HybridChatMemory`, as recorded in
[ADR 0004](../adr/0004-select-source-hybrid-memory.md).
Imported evidence metadata is loaded with `await MemorySearch.open(path)`
when a local index exists. An absent optional index is allowed; an existing
invalid index or a numeric import binding without evidence fails startup.

`LiveMemory` implements `MessageRecorder` with asynchronous SQLite writes and a
persisted revision queue. The application-scoped `MemoryProvider` initializes
it and shares the same instance between capture, canonical history and recall.
The provider owns the OpenAI embedding client and persistent `EmbeddingCache`;
`HybridChatMemory` owns background preparation and bounded cached retrieval
engines. Startup attempts preparation, retaining queued work after provider
failure. The service cancels and joins its worker, waits for active retrieval
leases and closes cached SQLite engines before the provider closes its client.
Completed vector batches and pending revisions survive restarts. Capture waits
for local persistence, while preparation owns provider calls and retries.

The same journal implements `ChatHistory` for recent-message and bounded
reply-parent selection independently of embeddings. `AgentProvider` supplies
it and the Telegram bot ID to `Dialogue`. Input and outgoing session middleware
share conversion in `bot/messages.py`. Group messages from all participants,
owner-private messages and successful deliveries are captured with actual
Telegram-assigned IDs; other private senders are excluded. `TelegramProvider`
attaches outgoing capture to the shared bot session.

Canonical history uses asynchronous SQLite reads. Hybrid ranking and construction
run bounded CPU/SQLite work in worker threads; cancellation drains started work
before releasing its resource lease. Embedding provider calls stay on the owning
event loop. The injected tools await their readers directly and own no thread
scheduling. Offline index maintenance retains synchronous SQLite access.

AI factories accept ordinary arguments; tools declare injected dependencies
with `FromDishka` and the AG2 integration's `@inject`. Model choices, timeouts,
retry policy, agent identities, prompts, and response schemas live in `ai`.
Providers adapt validated application settings to those inputs, attach DI
middleware, and supply dependencies. `OPENAI_MODEL` selects the reply model;
`OPENAI_DECISION_MODEL` selects the addressing model.

The reply agent has an AG2 `ResponseSchema[Reply]` for `text` and nullable
`reply_to_message_id`. `Dialogue` validates the parsed target against the same
group-context snapshot that seeded the stream; private replies retain the default
target. The result crosses into `bot` as the domain's `Reply`, with no AG2 objects.

`ReplyDelivery` remains in `bot`: it owns Telegram typing, redacted failure
handling, and rich/text delivery. Only an explicit Telegram rejection permits
fallback. Domain messages carry no aiogram objects.

## Shared graph and scopes

`main/container.py` builds one graph with `AG2Scope`. Application dependencies
are cached in `APP`: settings, model configurations, both agents, `Dialogue`,
`GroupAddressing`, `ChatMemoryService`, the vector cache/client, `GroupStore`,
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
`FromDishka[HistoryReader]` parameters stay outside the model-facing schema.
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
cancellation, and partial startup failures. The bot session and group database
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
They also check that a database lock leaves the event loop available. Search,
expansion, and comparison commands await the same backend interface, with one
event loop started at the CLI entrypoint. No live Telegram or paid model
requests are needed.

When adding a dependency, register its lifetime in the relevant provider and
test behavior through the consuming module's interface. The exhaustive
hierarchies in [`.importlinter`](../../.importlinter) cover the root package,
`main`, `ai`, and `bot`. Separate contracts keep configuration inside `main`
and domain/infrastructure modules independent of DI.

## Container startup

Compose starts the agent without a memory-service dependency. The runtime
resolves `ChatMemoryService`, initializing local state and attempting pending
preparation before accessing Telegram. With `HEALTHCHECK_PATH`
configured by the image, startup removes any stale readiness marker, verifies
the Telegram bot identity, and writes the current PID before polling. Shutdown
removes the marker and closes the container. The health check therefore verifies
completed startup, not just the existence of a Python process.

The optional `openviking` Compose profile is for legacy maintenance. Its service
key is checked inside its entrypoint so an inactive profile does not require the
key during Compose interpolation. Existing workspace state stays mounted and
the published port remains bound to loopback.
