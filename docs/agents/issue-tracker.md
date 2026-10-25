# Issue tracker: GitHub

Issues and specs live in GitHub Issues for `Lancetnik/dashulya`.
Use the `gh` CLI from the project root.

## Conventions

- Create: `gh issue create --title "..." --body-file <file>`.
- Read: `gh issue view <number> --json number,title,body,labels,comments`.
- List: `gh issue list --state open --json number,title,body,labels`.
  Apply label and state filters as needed.
- Comment: `gh issue comment <number> --body-file <file>`.
- Update a body: `gh issue edit <number> --body-file <file>`.
- Apply or remove labels: `gh issue edit <number> --add-label "..."`
  or `--remove-label "..."`.
- Close: `gh issue close <number> --comment "..."`.

For multiline bodies, write the exact Markdown to a temporary file
and pass it with `--body-file`.

“Publish to the issue tracker” means create a GitHub issue.
“Fetch the relevant ticket” means read the referenced GitHub issue.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## Wayfinding operations

- Map: one issue labelled `wayfinder:map`, containing Notes,
  Decisions-so-far, and Fog.
- Child ticket: link it to the map as a GitHub sub-issue.
  Use `wayfinder:<type>` for research, prototype, grilling, or task.
  If sub-issues are unavailable, use a task list in the map and
  `Part of #<map>` at the top of each child body.
- Blocking: use native GitHub issue dependencies. If unavailable,
  put `Blocked by: #<number>, #<number>` at the top of the child body.
  A ticket is unblocked when every blocker is closed.
- Frontier: inspect the map's open children; select the first in map
  order that has no open blockers and no assignee.
- Claim: `gh issue edit <number> --add-assignee @me` before work.
- Resolve: comment with the answer, close the ticket, and append
  a gist and link to the map's Decisions-so-far.

Check installed CLI support before using sub-issue or dependency
commands; use `gh api` when the CLI lacks a dedicated command.
