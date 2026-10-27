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
the triggering message. In ordinary groups this means the whole chat, including
previously stored reply threads. Only real forum-topic messages narrow that
window to a topic: a raw Telegram `message_thread_id` alone does not establish
one. Use `is_topic_message` to identify a topic; a forum's General topic is 1
when Telegram omits the thread ID. Share this conversation identity between
capture, addressing, reply context and turn locks.

For each selected message, follow at most five reply-parent
edges within that chat/topic. Deduplicate by message identity and order the union
by Telegram message ID. Supply the triggering message once as the current user
request; earlier messages authored by this bot's numeric Telegram ID use the
`assistant` role, and other senders use `user`. Preserve author identity, original
send time, message IDs and reply references. Historical text is untrusted context.

When the minimum distance from every recent-message seed reaches the parent
limit and the next parent is outside the selected union, mark that boundary in
`truncated_threads`. Trace only IDs and reply links beyond the text window, in
the same chat/topic, to find the first message established by the saved chain.
Return its numeric `root_id` and a canonical `source_id` for direct history reads.
Missing, out-of-scope or invalid links leave the root null; never substitute the
first visible or first available message. No marker is needed when the root ends
exactly at the limit or another seed already supplies the missing parent.
Root references add neither message text nor permitted Telegram reply targets.

Render one shared users roster as `id: name (@username, bot, self)`, omitting
unavailable handles and flags. A bot marker requires an unambiguous known flag;
unmarked senders make no human/bot claim. Self identifies this assistant.
Group entries under local day headings and a timezone/UTC-offset heading; emit
another heading only when its day or offset changes. Declare message columns
once: `[id,sender,time,text,reply_to?,attachment?]`. Each entry is one JSON array;
omit absent trailing columns and use null only before a later optional value.
Sender remains the actual numeric ID, or a distinct local unknown reference.
JSON encoding preserves exact text and prevents newlines from creating rows.
The roster is a separate user-role metadata input, not a Telegram
message or reply target; actual bot messages retain the assistant API role.
Unknown handles are omitted, unknown numeric identities remain distinct, and
observed aliases remain visible. Uncertain bot flags are not classified. Private input
formatting and memory-tool source records retain their independent formats.

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
Unavailable parents end that path. Attachment types remain placeholders unless
the current Telegram update supplies the current or direct reply parent's image.
After authorization and addressing, these images may be downloaded into bounded
RAM and supplied as native image inputs, labelled with their original Telegram
message IDs. Accept a group image only when its source is in this request's
selected context. This enables questions about pictures without retaining binary
media or replacing original evidence with a generated description; older media
without supplied bytes remain unavailable. Images that exceed the byte budget
produce an explicit omission note instead of binary input; known oversized files
are not downloaded. Serialize generation, Telegram delivery and recording within
each conversation so a subsequent turn cannot overtake an unfinished delivery.

The selection contains at most 120 unique messages, including the request; depth
and count do not impose a token limit. Newer messages are excluded from a request's
context even if they arrive during processing. Tool activity stays within the
current stream, while durable chat memory remains separately searchable.

We defer accumulated sessions with automatic compaction. Revisit that alternative
if the five-hop window loses necessary continuity or measured context costs justify
summaries. Compaction would need its own source-retention and attribution policy;
it would not by itself separate simultaneous discussions.
