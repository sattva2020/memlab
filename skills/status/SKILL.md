---
name: status
description: >-
  memlab overview and health for this session: which root is indexed (main checkout or this
  worktree), ready / failed / hung, errors, notes at risk, a live search check, and the command list.
disable-model-invocation: true
---

# memlab status

Answer in the user's language. In a git worktree (`git rev-parse --show-toplevel` differs from the
main checkout), pass `root` = that toplevel to every memlab tool call.

1. Run (Bash, from the session cwd):
   `CLAUDE_PLUGIN_DATA="${CLAUDE_PLUGIN_DATA}" python "${CLAUDE_PLUGIN_ROOT}/skills/status/status.py"`
   It reads memlab's call log only and prints the last server session for the cwd and, in a
   worktree, for the main checkout, then the notes at risk.
2. Live check: call `mcp__plugin_memlab_memlab__search_decisions` (or `mcp__memlab__search_decisions`)
   with a query about the current project, `k: 1` (with `root` in a worktree). 0 documents is a valid
   answer (no decision records); a tool error means the server is down.
3. Report in 3–5 lines: which root is indexed, ready/failed/hung, chunks, errors, live check result,
   and every line under "notes at risk" with what to do: uncommitted notes in a worktree → commit them
   with their change; main checkout behind origin → pull main. If the server's root ≠ cwd, say that
   tools need `root` = the worktree (the first such call builds the worktree's index).
4. End with the commands: `/memlab:recall <topic>` (prior decisions + code, with sources),
   `/memlab:note` (file this session's decisions as docs/notes).

The "last start" is the newest server for that root — with several sessions open on one repo it
may be another session's server.
