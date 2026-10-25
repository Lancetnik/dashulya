---
status: accepted
---

# Rebuild group reply context from observed messages

Group requests need the conversation between participants, including exchanges
that did not involve Dashulya. Each addressed group message therefore starts a
fresh AG2 stream, populated from the local journal rather than accumulated model
turns. This keeps context selection reproducible and avoids repeatedly
summarizing unrelated discussions together. Private conversations retain their
existing in-memory history and `/clear` behavior.

Select the latest 20 observed messages in the same chat/topic up to and including
the triggering message. For each selected message, follow at most five reply-parent
edges within that chat/topic. Deduplicate by message identity and order the union
by Telegram message ID. Supply the triggering message once as the current user
request; earlier messages authored by this bot's numeric Telegram ID use the
`assistant` role, and other senders use `user`. Preserve author identity, original
send time, message IDs and reply references. Historical text is untrusted context.

The reply agent returns structured `text` and nullable `reply_to_message_id`.
It may choose the current request or another message actually included in this
reply context. Validate the ID against that exact selection before Telegram
delivery; null or an ID outside the selection defaults to the triggering message.
Private replies also default to the triggering message. Use the chosen target
for rich delivery and every text chunk, and record that actual reply relationship.

Read canonical local messages, including edits and messages awaiting semantic
indexing. Persist successfully delivered bot responses with their actual Telegram
IDs and reply targets, including individual chunks. An embedded reply snapshot
can fill a missing parent but cannot replace an already recorded original.
Unavailable parents end that path; attachment types are placeholders rather than
interpreted media. Serialize generation, Telegram delivery and recording within
each conversation so a subsequent turn cannot overtake an unfinished delivery.

The selection contains at most 120 unique messages, including the request; depth
and count do not impose a token limit. Newer messages are excluded from a request's
context even if they arrive during processing. Tool activity stays within the
current stream, while durable chat memory remains separately searchable.

We defer accumulated sessions with automatic compaction. Revisit that alternative
if the five-hop window loses necessary continuity or measured context costs justify
summaries. Compaction would need its own source-retention and attribution policy;
it would not by itself separate simultaneous discussions.
