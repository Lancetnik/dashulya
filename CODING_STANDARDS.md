# Coding Standards

Use these rules when reviewing changes. The module map in
[AGENTS.md](AGENTS.md#architecture) describes each module's responsibility.

## Language

Use English for project documentation, code comments, and metadata.

## Module design

Keep domain models, lifecycle rules, and persistence/search interfaces in
`domain`; storage implements the interfaces declared by the domain.

[Import Linter contracts](.importlinter) are authoritative for dependency
direction and framework independence. They cover top-level modules and the
contents of `main/`, `ai/`, and `bot/`, including the boundary between runtime
code and offline indexing or command execution. Use the automated checks for
import violations; review whether responsibilities belong in the chosen module.

## Dependency assembly

Container construction and providers live in `main`; handlers in `bot` and tools
in `ai` use the standard Dishka `@inject` and `FromDishka` integration directly.
`main/providers/` groups registration by responsibility. Configuration stays in
`main/config.py` and is adapted to narrow constructor inputs by providers.

Review scopes, provider overrides, and resource lifetimes against
[the DI design](docs/design/dependency-injection.md).

## Conversation state

Keep history and turn locking together in `ai/dialogue.py` so replies and
clearing history share the same conversation state boundary.
