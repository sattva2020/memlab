---
name: recall
description: >-
  Prior decisions and the relevant code on a topic, from memlab: a short summary with the
  decision files and path:line links.
disable-model-invocation: true
argument-hint: "<topic>"
---

# memlab recall: $ARGUMENTS

Answer in the user's language. In a git worktree (`git rev-parse --show-toplevel` differs from the main checkout),
pass `root` = that toplevel to every memlab tool call.

1. In parallel: memlab's `search_decisions` tool (query = topic, k 5) and
   memlab's `search_code` tool (query = topic, default budget). No topic → ask for one.
2. Write a short summary: **Decided** (decisions, each with its file; say when one is marked
   ⟨superseded by …⟩) and **In the code** (`path:line` links). Only claims backed by a returned
   chunk; say when nothing was found.
3. Before stating a code fact as current, read that source (the index is from server start;
   ⟨stale⟩ / ⟨deleted⟩ marks say the file changed since).
