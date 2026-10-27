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
  tools/memory.py        Memory tool behavior over the async backend interface
  dialogue.py            Reply context, conversation history, turn locking, and reset
  addressing.py          Direct-address recognition and classification
  prompts.py             Persona and response instructions
domain/
  groups.py              Group, GroupRepository, and GroupRegistry
  messages.py            GroupMessage, AddressingDecision, Reply and ConversationKey
  memory.py              MemoryBackend, MemoryMessage, MessageRecorder and ChatHistory
```

`domain` has no application or infrastructure imports. `GroupRegistry` owns
observed group migrations, membership changes, and bounded group-list output.
`storage.groups.GroupStore` implements `GroupRepository` using SQLite. The
provider aliases that implementation to the domain interface and owns cleanup.
Memory implementations satisfy the domain's async `MemoryBackend.search`
interface. FTS5 and evidence reconstruction use read-only `aiosqlite`
connections; OpenViking recall uses `AsyncHTTPClient`. Backend initialization
loads SQLite metadata with `await MemorySearch.open(path)`. The tool awaits
the backend directly and owns no thread scheduling. Indexing and reviewed
maintenance writes retain synchronous SQLite access in the offline commands.

`LiveMemory` implements `MessageRecorder` with an asynchronous SQLite write and
a persisted revision queue. The application-scoped `MemoryProvider` initializes
it, supplies the same instance to capture and recall, and starts `MemoryIngestion`
after OpenViking readiness checks. Its async generator cancels and joins the
worker at shutdown; pending rows survive. Capture waits only for local
persistence, while the worker owns remote indexing and retries.
The same instance implements `ChatHistory` for direct recent-message and bounded
reply-parent selection, independent of indexing readiness. `AgentProvider`
supplies it and the Telegram bot ID to `Dialogue`. Telegram input and outgoing
session middleware share the message conversion in `bot/messages.py`; successful
group deliveries, including command responses, are recorded with Telegram-assigned
IDs. `TelegramProvider` attaches outgoing capture to the shared bot session.

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
`GroupAddressing`, the OpenViking memory backend, `GroupStore`, `GroupRegistry`,
`ReplyDelivery`, the Telegram bot, and the dispatcher. Application resolution
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

`ai/tools/memory.py` registers the memory tool with `@tool` and the standard
`dishka_ag2.inject` decorator. Its `FromDishka[MemoryBackend]` parameter is
resolved when AG2 invokes the tool and is excluded from the model-facing schema.
The reply agent always has the memory tool. Its provider requires the memory
backend, which validates the local evidence index, completed import, server
health and remote corpus before agent construction. No runtime FTS5 or
no-memory mode is registered.

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
are finalized by their providers. The OpenViking memory backend is shared across
Telegram and AG2 scopes.

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
Memory tests execute the injected tool through real AG2 with scripted or
mock-transport model responses. They also check that a database lock leaves the
event loop available and that cancelled HTTP recall closes its client. Search,
expansion, and comparison commands await the same backend interface, with one
event loop started at the CLI entrypoint. No live Telegram or paid model
requests are needed.

When adding a dependency, register its lifetime in the relevant provider and
test behavior through the consuming module's interface. The exhaustive
hierarchies in [`.importlinter`](../../.importlinter) cover the root package,
`main`, `ai`, and `bot`. Separate contracts keep configuration inside `main`
and domain/infrastructure modules independent of DI.

## Container startup

Compose starts the agent after OpenViking is healthy. The runtime resolves the
required memory backend before accessing Telegram. With `HEALTHCHECK_PATH`
configured by the image, startup removes any stale readiness marker, verifies
the Telegram bot identity, and writes the current PID before polling. Shutdown
removes the marker and closes the container. The health check therefore verifies
completed startup, not just the existence of a Python process.
