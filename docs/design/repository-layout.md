# Repository layout

Keep project design, evaluation infrastructure and application state separate.

| Path | Purpose | Versioned |
| --- | --- | --- |
| `docs/design/` | Current implementation and architecture | Yes |
| `docs/adr/` | Accepted decisions, rationale and trade-offs | Yes |
| `deploy/` | Service configuration consumed by Compose | Yes, with environment references for secrets |
| `evals/suites/` | Evaluation runners, case contracts, scoring rules and aggregate comparison records | Yes |
| `evals/datasets/` | Entire local datasets, including descriptions, manifests, cases and annotations | No |
| `evals/runs/` | Generated exports, isolated indices and evaluation results | No |
| `var/agent/` | Optional imported evidence, group registry, live originals, revision queue and vector cache | No |
| `var/openviking/` | Optional legacy OpenViking workspace | No |
| `var/logs/` | Application logs | No |
| `.scratch/` | Temporary research, interview notes and working drafts | No |
| `.github/agents/` | Contributor-agent workflow conventions | Yes |

PRDs and task specifications live in GitHub Issues. The current chat-memory
replacement specification is [issue #3](https://github.com/Lancetnik/dashulya/issues/3);
it describes requested behavior, not the current implementation. Keep research
working files local; record accepted architectural decisions in ADRs and then
discard the working files when they are no longer useful.

Datasets are excluded as complete directories, including preparation scripts
specific to a private corpus. Reusable evaluation infrastructure belongs in
`evals/suites/` and must run its automated tests without local datasets or model
calls. Each local dataset should carry its own provenance and coverage notes.

Treat an evaluation index as an input to a particular run. To use an imported
index for the running agent, explicitly copy its evidence to
`var/agent/evidence.sqlite3` (or a separately configured `MEMORY_DIRECTORY`),
verify any numeric `MEMORY_IMPORT_CHAT_ID` binding, then restart the agent.
The runtime reads original messages from this optional index and does not
require a completed OpenViking import or manifest. Keep legacy import manifests
with their matching evidence when maintaining an optional OpenViking corpus.

Compose mounts `MEMORY_DIRECTORY` read-only at `/memory`, and host `var/agent/`
read/write at `/var/lib/dashulya` for live originals, durable revisions, vectors
and the registry. This separates evaluation artifacts from active state:
rerunning an evaluation must not replace the agent's files. Container recreation
preserves `var/`; deleting it discards application state.

Reusable methodology and aggregate comparison evidence live with their suite,
for example `evals/suites/memory/RESULTS.md`. Keep case text, gold annotations,
predictions and detailed traces under ignored dataset/run paths. Accepted
backend rationale belongs in an ADR; provisional aggregate scores alone do not
constitute a final engine decision or complete product acceptance.
