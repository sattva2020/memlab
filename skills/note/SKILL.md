---
name: note
description: >-
  File this session's decisions, postmortems and findings as docs/notes via memlab add_note —
  drafts first, nothing is written until you pick.
disable-model-invocation: true
---

# memlab note

Answer in the user's language. In a git worktree (`git rev-parse --show-toplevel` differs from the main checkout),
pass `root` = that toplevel to every memlab tool call, so the notes land in this branch.

1. Go through this session and pick conclusions worth keeping: decisions (what + why),
   postmortems (root cause + fix), non-obvious findings. Skip what the code or git log already says.
2. For each, check memlab's `search_decisions` tool for an earlier record it replaces.
3. Show a numbered list of drafts: `type` (decision|note|postmortem), one-line summary, tags,
   and `supersedes` when step 2 found one. Wait for the user to choose. Nothing is written
   without their pick.
4. For each chosen draft call memlab's `add_note` tool (summary, body with context, alternatives,
   file paths; type; tags; supersedes; root in a worktree).
5. Report the created `docs/notes/...` paths. Do not commit — the note goes in with the change
   it belongs to.
