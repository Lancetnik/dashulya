# Chat message search: Osinara, Miya, and established product patterns

Researched on October 25, 2026. This is a static source review, not an audit of
the deployed bots or a benchmark of their retrieval quality.

| Project | Reviewed branch | Pinned commit |
| --- | --- | --- |
| [Osinara](https://github.com/nyxandro/osinara) | `develop`, the default branch at inspection | `4ca8eff77e0bf403f998aebefedd2deae2e60ba8` |
| [Miya](https://github.com/ilkruglov/osinara) | `main` | `8adda8f5de1a910c690927afcd7ca6154af9ecd4` |

## Main finding

Both repositories separate **recent conversation context**, a **retained raw
group journal**, and **extracted long-term memory**. Their sophisticated hybrid
retrieval operates on saved memory records. Searching the raw group journal
is a different tool and currently uses a case-insensitive literal substring,
not semantic search. This distinction follows the actual tool definitions and
SQL, rather than the repositories' general feature descriptions.
[Osinara history tool][o-history-tool], [Miya history tool][m-history-tool],
[Osinara memory retrieval][o-memory-sql], [Miya memory retrieval][m-memory-sql].

## How the group journal works

### Ingestion and coverage

Osinara checks that a group is registered and that the incoming message is
admitted by the group's policy, then records it before deciding whether the
message should start an agent turn. Consequently the journal can include
passive messages, not just messages addressed to the bot. It is still limited
to messages that reached and passed the ingress policy; it should not be
called a complete Telegram archive. The record path stores sender identity,
text, timestamps, topic and reply metadata, allocates an internal sequence,
deduplicates Telegram message IDs, and prunes retained history.
[Ingress][o-ingress], [record implementation][o-journal].

The inspected active ingestion and search paths do not fetch old messages
from Telegram. No general chat-history backfill was found in the inspected
tools or scripts. This is a source-review observation, not proof about every
possible external deployment or operational import. A migration/import of
previously stored application history would also be different from fetching
Telegram history.
[Ingress][o-ingress], [history application][o-history-app].

Both configurations target the latest **10,000 journal records per group**.
This is not a hard absolute row cap: records referenced as memory evidence
are protected from pruning; Osinara additionally protects messages pending
memory review. The shared journal-context configuration has a budget of
**100 messages and 12,000 characters**, with recent/incremental selection
logic rather than a promise to include the last 100 messages on every turn.
Miya also configures a smaller recent window of 15 messages and 3,000 characters.
These are different limits: storage retention does
not mean all retained messages are placed in every LLM request.
[Osinara configuration][o-config], [Miya configuration][m-config],
[Osinara pruning][o-pruning], [Miya pruning][m-pruning].

### Explicit raw-history search

The call chain in both projects is:

```text
LLM calls list_group_history
  -> authorization supplies the current registered group
  -> searchTelegramGroupHistory builds bounded filters
  -> telegramGroupJournalRepository.search executes PostgreSQL SQL
  -> matching entries and a pagination cursor return to the LLM
```

The tool does not let the model choose an arbitrary group ID. The application
derives it from verified authorization and searches all forum topics of that
group. Attachment visibility is separately constrained by the current
group's capabilities.
[Osinara authorization and projection][o-history-app],
[Miya authorization and projection][m-history-app].

| Input or output | Actual behavior |
| --- | --- |
| `query` | Literal case-insensitive substring in `content_text` |
| `participant` | Tool asks for an exact username without `@`; SQL accepts exact `actor_id` or exact `sender_username` |
| `from`, `to` | Inclusive message timestamp bounds |
| `sequenceFrom`, `sequenceTo` | Inclusive internal sequence bounds |
| `beforeSequence` | Exclusive sequence bound for reading older results |
| Combined filters | Logical AND |
| `limit` | Default 25, maximum 100, after filtering |
| Ordering | SQL selects newest matches; the returned page is reversed into chronological order |
| Pagination | `nextBeforeSequence` gives the next older page |
| Addressing | Model-visible `sequence` and `replyToSequence`, not native Telegram message IDs |

The decisive predicate is:

```sql
position(lower($9) in lower(coalesce(message.content_text, ''))) > 0
```

That path does not apply stemming, typo correction, synonyms, BM25, vector
similarity, or a semantic reranker. A request such as “what did we decide
about replacing Redis?” needs the LLM to choose useful literal searches and
possibly repeat with alternative terms. A paraphrase alone is not guaranteed
to match stored wording.
[Osinara SQL][o-history-sql], [Miya SQL][m-history-sql],
[tool descriptions][o-history-tool].

### Reading enough context

The explicit search returns the matching entries and their reply sequence
references. It does **not** automatically expand every result into its full
discussion. The model can make another bounded history call for the relevant
sequence range. Separately, the automatic recent-context repository follows
up to two reply-parent edges, so its behavior should not be attributed to the
explicit search SQL.
[History projection][o-history-app], [automatic recent context][o-recent],
[explicit search SQL][o-history-sql].

## Separate long-term memory retrieval

Saved memory is a projection of selected facts, preferences, episodes,
decisions, and other useful records, with source evidence. For example,
Osinara's passive review prompt selects facts from a bounded batch and writes
them with exact source sequences. This can support remembering a decision,
but does not preserve every original message or guarantee verbatim quotation.
[Memory-review prompt][o-review], [memory-search tool][o-memory-tool].

Both memory retrievers combine PostgreSQL `simple` full-text search, Russian
morphological full-text search, and cosine vector retrieval. Their SQL applies
authorization before producing candidates and combines branch ranks with
reciprocal rank fusion, using an offset of 60. Retrieval thresholds, retention
or freshness, and other project-specific adjustments also influence selection.
This is a richer pipeline than raw-history substring matching.
[Osinara memory SQL][o-memory-sql], [Miya memory SQL][m-memory-sql],
[Osinara memory configuration][o-memory-config], [Miya memory configuration][m-memory-config].

| Current pinned implementation | Osinara | Miya |
| --- | --- | --- |
| Memory embedding model | `intfloat/multilingual-e5-small` | `sergeyzh/BERTA` |
| Dimensions | 384 | 768 |
| Explicit semantic similarity gate | 0.78 | 0.20; optional weak-match search admits 0.12 |
| Neural reranker | Not part of the inspected active retrieval pipeline | Explicitly removed according to current configuration |

The thresholds belong to different models and calibration regimes; their
numeric values are not comparable measures of quality. Miya's current source
states that BERTA replaced E5 and the mmarco reranker on October 2, 2026,
following local evaluation and CPU timeout issues. Its repository About text
still advertises “FTS + E5 + reranker”; use the pinned source for the active
implementation. The comments' evaluation results are author-reported, not
independently reproduced in this review.
[Osinara configuration][o-memory-config], [Miya configuration][m-memory-config],
[Miya repository About](https://github.com/ilkruglov/osinara).

## How other products approach the problem

**Slack:** ordinary search exposes phrase, prefix, channel, author, date, and
thread filters. Its documented real-time search API distinguishes keyword and
semantic retrieval: semantic retrieval is available for eligible natural-language
queries and relevance sorting, while timestamp sorting switches retrieval back
to keyword search. Its guide also documents expanding search hits with
`conversations.replies` or surrounding `conversations.history`. AI answers
operate on messages and files the user may access; the public documentation
does not establish the internal embedding model or ranking implementation.
[Search UI](https://slack.com/help/articles/202528808-Search-in-Slack),
[real-time search API and context pattern](https://docs.slack.dev/apis/web-api/real-time-search-api/),
[AI search permissions](https://slack.com/intl/en-us/help/articles/31739993134867-Search-with-AI-in-Slack).

**Discord:** its engineering account describes Elasticsearch, queued and bulk
indexing, and routing guild/DM search data across smaller clusters or cells.
This demonstrates that scalable message search still relies on an indexed
lexical retrieval system; the article does not establish a semantic search or
answer-generation pipeline.
[Discord engineering](https://discord.com/blog/how-discord-indexes-trillions-of-messages).

**Zulip:** built-in search uses PostgreSQL full-text search by default, with
index updates performed after message delivery. Its docs describe English
stop words and an experimental PGroonga multilingual alternative; do not infer
Russian morphology support from the default configuration. The UI supports
topic/phrase/word search and navigation into the surrounding conversation.
[Full-text implementation](https://zulip.readthedocs.io/en/latest/subsystems/full-text-search.html),
[message-search UI](https://zulip.com/help/search-for-messages).

**Hybrid retrieval as a documented design pattern:** Azure AI Search runs
lexical BM25 and vector searches in parallel, fuses the rankings with RRF,
and can run a semantic reranker afterward. This illustrates one established
retrieval architecture; it is not evidence that Slack or Discord use that
exact pipeline.
[Hybrid overview](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview),
[fusion and ranking](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking).

## Recommendation for Dashulya

The following is a proposed experiment plan, not an implemented feature or a
claim that one architecture wins on this dataset.

Our existing snapshot has 17,000 group message records, a SQLite FTS5 index
using `unicode61`, 10,717 messages with reply references, and 5,652 messages
linked to channel posts. The tokenizer is not a configured Russian stemmer.
These are local database observations; the dataset and reproducible query
examples are documented in [the dataset note](../datasets/fastnewsdev.md).

1. **Establish a lexical baseline.** Keep the raw messages as the source of
   truth. Search FTS5 with author, date, and thread filters, rank by BM25, and
   return stable message IDs with links to source messages. Compare whole-word
   and prefix queries, especially for Russian inflections, names, URLs, and
   tool names.
2. **Expand retrieved messages into conversation context.** Fetch reply
   ancestors, relevant descendants, and a bounded neighboring window. Preserve
   authors, timestamps, and thread membership; a one-line answer such as
   “yes, that is the repository” cannot be understood without its parent.
   Let the answering model cite the exact message evidence.
3. **Add semantics only against measured failures.** Build a small set of real
   Russian questions with known relevant messages and threads. Measure
   Recall@k, MRR, latency, and whether final answers are supported by citations.
   Then compare embeddings of individual messages against bounded conversation
   windows, combine lexical and semantic rankings with RRF, and test whether a
   reranker improves enough to justify its cost.

Keep extracted long-term memory as a separate derived view for useful facts
and decisions. It can help answer a question, but should not replace the
message history used for evidence, attribution, and exact quotation. Apply
chat-access filters before retrieval and context expansion, not only while
formatting the final answer. These design choices are informed by the
repositories' explicit scope enforcement, Slack's search/context pattern,
and the documented hybrid architecture above.

[o-history-tool]: https://github.com/nyxandro/osinara/blob/4ca8eff77e0bf403f998aebefedd2deae2e60ba8/agent/lib/tools/list_group_history.ts
[m-history-tool]: https://github.com/ilkruglov/osinara/blob/8adda8f5de1a910c690927afcd7ca6154af9ecd4/agent/lib/tools/list_group_history.ts
[o-history-app]: https://github.com/nyxandro/osinara/blob/4ca8eff77e0bf403f998aebefedd2deae2e60ba8/agent/lib/telegram-group-history.ts
[m-history-app]: https://github.com/ilkruglov/osinara/blob/8adda8f5de1a910c690927afcd7ca6154af9ecd4/agent/lib/telegram-group-history.ts
[o-history-sql]: https://github.com/nyxandro/osinara/blob/4ca8eff77e0bf403f998aebefedd2deae2e60ba8/agent/lib/telegram-group-journal-repository.ts#L365-L389
[m-history-sql]: https://github.com/ilkruglov/osinara/blob/8adda8f5de1a910c690927afcd7ca6154af9ecd4/agent/lib/telegram-group-journal-repository.ts#L389-L413
[o-journal]: https://github.com/nyxandro/osinara/blob/4ca8eff77e0bf403f998aebefedd2deae2e60ba8/agent/lib/telegram-group-journal-repository.ts
[o-recent]: https://github.com/nyxandro/osinara/blob/4ca8eff77e0bf403f998aebefedd2deae2e60ba8/agent/lib/telegram-group-journal-repository.ts#L168-L200
[o-ingress]: https://github.com/nyxandro/osinara/blob/4ca8eff77e0bf403f998aebefedd2deae2e60ba8/agent/lib/telegram-on-message.ts#L108-L153
[o-config]: https://github.com/nyxandro/osinara/blob/4ca8eff77e0bf403f998aebefedd2deae2e60ba8/agent/config.ts#L51-L53
[m-config]: https://github.com/ilkruglov/osinara/blob/8adda8f5de1a910c690927afcd7ca6154af9ecd4/agent/config.ts#L81-L89
[o-pruning]: https://github.com/nyxandro/osinara/blob/4ca8eff77e0bf403f998aebefedd2deae2e60ba8/agent/lib/telegram-group-message-storage.ts#L135-L160
[m-pruning]: https://github.com/ilkruglov/osinara/blob/8adda8f5de1a910c690927afcd7ca6154af9ecd4/agent/lib/telegram-group-message-storage.ts#L121-L145
[o-review]: https://github.com/nyxandro/osinara/blob/4ca8eff77e0bf403f998aebefedd2deae2e60ba8/agent/lib/memory-review/memory-review-prompt.ts#L25-L37
[o-memory-tool]: https://github.com/nyxandro/osinara/blob/4ca8eff77e0bf403f998aebefedd2deae2e60ba8/agent/lib/tools/search_memories.ts
[o-memory-sql]: https://github.com/nyxandro/osinara/blob/4ca8eff77e0bf403f998aebefedd2deae2e60ba8/agent/lib/memory-retrieval-repository.ts
[m-memory-sql]: https://github.com/ilkruglov/osinara/blob/8adda8f5de1a910c690927afcd7ca6154af9ecd4/agent/lib/memory-retrieval-repository.ts
[o-memory-config]: https://github.com/nyxandro/osinara/blob/4ca8eff77e0bf403f998aebefedd2deae2e60ba8/agent/lib/memory-config.ts
[m-memory-config]: https://github.com/ilkruglov/osinara/blob/8adda8f5de1a910c690927afcd7ca6154af9ecd4/agent/lib/memory-config.ts#L71-L121
