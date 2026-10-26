# Bind chat memory access to the trusted conversation

The [replacement specification](https://github.com/Lancetnik/dashulya/issues/3)
requires each group to recall only its own chat, across topics, while the owner's
private conversation can recall every saved chat, including private records. Bind
that authority to each AG2 session using the actual numeric Telegram chat identity;
model-facing filters only narrow it. A shared application agent must never hold an
unscoped reader that can expose another conversation's sources.

Expose independent tools for bounded memory recall and canonical history reads,
counts and pagination. Semantic-index outages do not remove original history.
Imported archives without a verified numeric binding are available only privately:
a public username is insufficient to merge identities with a live group. Resolve
latest live originals before filtering an overlapping archive so an edit cannot
uncover obsolete snapshot text. Prior edit versions remain unavailable.

This replaces the unscoped access behavior described with the original
[OpenViking prototype](0001-openviking-memory.md), independently of the retrieval
engine. Owner authorization and per-topic
reply-context selection retain their existing responsibilities.
