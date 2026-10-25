# FastNews history dataset

A local SQLite snapshot of the existing `slop-writer` database for the
`@fastnewsdev` channel and its linked discussion group, `@fastnewsdev_chat`.
The snapshot was copied on October 25, 2026; it is independent of the source
database and does not require Telegram credentials or a running bot.

## Local files

| Path | Contents |
| --- | --- |
| `data/fastnewsdev/snapshot.sqlite3` | Full database snapshot, including its full-text indexes |
| `data/fastnewsdev/manifest.json` | Source provenance, copy time, SHA-256 checksums, row counts, and date ranges |
| `data/fastnewsdev/latest-group-scan.json` | Result of the latest Telegram group scan |
| `data/fastnewsdev/latest-post-refresh.json` | Result of refreshing the newly referenced channel post |
| `data/fastnewsdev/archive/before-refresh-2026-10-25/` | Original snapshot and manifest before the first live refresh |

`data/` is excluded from Git. These files exist on this machine; cloning the
repository alone does not provide the dataset. Original message text retains
its original language.

The source was `~/Desktop/media/.tg-analytic/fastnewsdev.db`. Schema semantics
were checked against `~/Desktop/media/.agents/skills/slop-writer/SKILL.md` and
its `references/analysis.md` and `references/schema.md`. The source database
was opened read-only and copied with SQLite's backup API. Integrity checks and
table counts passed, and the source file's checksum remained unchanged.

The existing scrape was refreshed from Telegram on October 25, 2026, using
`slop-writer` 0.5.1. `scan_linked_group` walked forward from group message ID
18907 without a limit, adding 216 group message rows, including one thread root.
`refresh_posts` also fetched channel post 419 to fill in the new thread's context.
Historical completeness is not guaranteed. The latest recorded group scan is
October 25, 2026, at 08:24 UTC; the latest saved message is from 08:20:14 UTC
(11:20:14 Moscow time). Telegram sessions, credentials, and external media files were
not copied. Attachment metadata is preserved; `photo_path` values may still
point to historical locations outside this project.

## Coverage

| Data | Count or range |
| --- | --- |
| Group message rows | 17,000 |
| Group messages excluding thread roots | 16,641 |
| Automatic thread roots | 359 |
| Message rows with nonempty text | 16,530 |
| Distinct sender IDs, including channel identities and bots | 402 |
| Group message dates, UTC | November 29, 2024 – October 25, 2026 |
| Channel posts | 319 |
| Channel post dates, UTC | November 27, 2024 – October 25, 2026 |
| Group join/leave event rows | 386 |
| Group metric snapshots | 150 |
| Post metric snapshots | 2,032 |

The manifest contains counts for every application table. The full source
database also retains attachment metadata, public shares, and subscriber
statistics and audit records.

## Tables for experiments

| Table | Useful fields and meaning |
| --- | --- |
| `group_messages` | `id`, `date`, `text`, `user_id`, `user_name`, `user_username`, `author`, `reply_to_msg_id`, `thread_post_id`, `is_thread_root`, `reactions`, `media_type` |
| `posts` | Channel posts: `id`, `link`, `date`, `text`, `edit_date`, `tags`, `grouped_id`, `forwarder_from_channel` |
| `group_events` | Observed membership events, keyed by `(id, user_id)` |
| `group_metrics` | Group identity and member count snapshots |
| `post_metrics` | Append-only engagement snapshots, joined to `posts` by `post_id` |
| `gm_fts`, `posts_fts` | FTS5 text indexes; `rowid` matches the corresponding message or post `id` |

Message IDs belong to their chat. `group_messages.reply_to_msg_id` points to
another group message; `thread_post_id` points to a channel post in `posts`.
Use a left join when adding post context: a referenced post may be absent from
the scrape. There is no per-row chat ID: this database contains one channel
and its linked group. The running bot's group registry remains separate in
`data/dashulya.sqlite3`.

Exclude `is_thread_root = 1` when counting participant messages. Identify
senders by `user_id`; `author` is a generated display label, and usernames or
display names can change. Sender IDs also include bots and the channel itself;
the channel identity can represent both automatic posts and human replies.

Dates are ISO-8601 strings with UTC offsets. Use `datetime(date)` for time
comparisons rather than comparing the raw column with SQLite's space-separated
`datetime()` output. For the latest post metrics, choose `MAX(id)` per
`post_id`, since scrape timestamps can tie. The FTS indexes tokenize Unicode
text without stemming; prefix searches can cover inflected forms.

## Read-only queries

Run from the project root:

```sh
sqlite3 -readonly data/fastnewsdev/snapshot.sqlite3
```

Inspect the schema and recent participant messages:

```sql
.tables
.schema group_messages

SELECT id, date, author, text, reply_to_msg_id, thread_post_id
FROM group_messages
WHERE is_thread_root = 0
ORDER BY datetime(date) DESC, id DESC
LIMIT 20;
```

Search group messages using the existing full-text index:

```sql
SELECT gm.id, gm.date, gm.author, gm.text, bm25(gm_fts) AS rank
FROM gm_fts
JOIN group_messages AS gm ON gm.id = gm_fts.rowid
WHERE gm_fts MATCH 'agent* OR llm*'
  AND gm.is_thread_root = 0
ORDER BY rank, gm.id
LIMIT 20;
```

Read the dataset from Python:

```python
import sqlite3
from contextlib import closing
from pathlib import Path

path = Path("data/fastnewsdev/snapshot.sqlite3").resolve()
with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT id, text FROM group_messages WHERE is_thread_root = 0 LIMIT 10"
    ).fetchall()
```

Keep `snapshot.sqlite3` as the baseline. Experiments that modify data can work
on their own copy under `data/experiments/`; derived data should also stay under
`data/`. The snapshot does not update when the source database changes.
