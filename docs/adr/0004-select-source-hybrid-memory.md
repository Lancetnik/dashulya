# Select hybrid recall over canonical original messages

Accepted on October 26, 2026, following the owner's explicit choice of the
simpler source-based variant. Use BM25 and dense embeddings of original messages
with up to two reply parents, combine rankings with reciprocal rank fusion, and
return source-diverse evidence. Persist originals and content-addressed vectors
locally; OpenViking and Hindsight remain evaluation/maintenance options rather
than required application services. This supersedes the backend selection in
[ADR 0001](0001-openviking-memory.md); [ADR 0003](0003-bind-chat-memory-access.md)
governs access independently of ranking.

In the frozen 50-question development comparison, three answer generations with
two grading passes each averaged 77% for raw hybrid recall and 78.3% for raw
hybrid plus extracted claims, Hindsight facts and an OpenAI Decisions reranker.
The difference is less than one question per generation. Independent agent
source audits of the initial answers conservatively passed 45/50 and 41/50,
respectively; these are separate agent reviews, not human certification or the
same metric as model grading. The owner chose the smaller operational footprint
and simpler revision handling. The [evaluation record](../../evals/suites/memory/RESULTS.md)
distinguishes this engine comparison from actual production-path validation.
The actual selected application averaged 52% across two grades of one full run,
so the implementation is not accepted for deployment on answer quality. The
engine choice remains the owner's selection; completion/grounding work is required
before claiming the experiment's 70% target or the PRD's 90%.

Use `text-embedding-3-small` with 512 dimensions and reply-aware blocks. Both
lexical and dense rankings consider at most 160 candidates; RRF uses 60 and
recall returns at most eight hits. Canonical authorization and chat selection
precede ranking. Returned citations and context come from current local
originals, never generated URLs. Exact scans and counts use `chat_history`.

Edits change block content keys, including dependent reply-parent blocks.
Unindexed revisions remain readable and lexically searchable; coverage exposes
pending blocks while the durable worker retries embeddings. This is the fixed
hybrid pipeline's explicit availability behavior. It does not silently select a
different engine. Index construction and CPU ranking run outside the event loop,
and cancellation retains resource ownership until SQLite work finishes.

This decision covers source-based recall. Maintained fact writing/correction,
explicit forgetting and source-deletion propagation, and complete chat migration
continuity remain work under [PRD #3](https://github.com/Lancetnik/dashulya/issues/3).
No imported snapshot, maintained legacy record or native workspace is deleted by
switching the application backend.
