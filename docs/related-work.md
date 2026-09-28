# Related work and ideas to test (survey 2026-09-27)

Three surveys: code-context tools, agent-memory systems, code-localization research.
Sources were read from repositories (GitHub API/raw) and arXiv; licenses from GitHub `spdx_id`.
Ideas only — no code is copied from non-permissive projects.

## What the field does (short)

| System | License | Ranking / routing | Relevant trick |
|---|---|---|---|
| Aider repo-map | Apache-2.0 | PageRank on file graph from tree-sitter tags, personalization to chat/mentioned files | edge weight `mul·sqrt(refs)`; ×10 mentioned identifiers, ×10 long snake/camel names, ×0.1 `_private`, ×0.1 names defined in >5 files; binary search to fit tokens |
| graphify | Apache-2.0 | lexical seeds → BFS depth 3 | hubs (deg ≥ max(50,p99)) not expanded; gap/label/per-term seeds — **tested here as H5/H6, both met** |
| RepoGraph | Apache-2.0 | k-hop ego graph on def/ref lines | drop builtins/stdlib/third-party names; k=1 beats k=2 |
| LocAgent | Apache-2.0 | LLM traverses dir/file/class/function graph | file Acc@5 on SWE-bench Lite: BM25 61.7, CodeRankEmbed 84.7, LocAgent 92.7–94.2 |
| Continue.dev | Apache-2.0 | FTS5 trigram + embeddings + recent files, quotas or rerank | AST chunks ≤384 tok with folded bodies; path weight ×10 |
| zoekt | Apache-2.0 | BM25 k=1.2, b=0.75 | match in filename/symbol definition TF += 5; test/vendor/generated TF ÷5 |
| probe | Apache-2.0 | BM25 + coverage boost | `1 + cov^1.5·2`; node-type multipliers (type 1.8 … comment 0.5); merge blocks ≤5 lines apart |
| Moatless | MIT | embedding spans to 8k tokens | exact-match filter; generated files skipped |
| cAST / astchunk | MIT | split-then-merge AST chunking | RepoEval Recall@5 +4.3 |
| Graphiti/Zep | Apache-2.0 | separate lists per memory type, two MCP tools | bi-temporal edges (`valid_at/invalid_at`), supersede without delete |
| Letta | Apache-2.0 | agent chooses tools | plain file tools 74.0% LoCoMo vs Mem0g 68.5% |
| Mem0 (OSS) | Apache-2.0 | one fused ranking + entity boost | additive memories that record transitions |
| HippoRAG 2 | MIT | PPR, passages seeded at 0.05 | falls back to dense retrieval when no fact survives the filter |
| LightRAG / GraphRAG | MIT | caller picks mode; fixed token splits | fixed splits = what failed here (H3/H4/H7) |

Serena is GPL-3.0 (app) — reference only. SweRankEmbed and jina-code-embeddings are CC-BY-NC — internal experiments only.

## Evaluation practice to adopt

1. **Leakage slices**: strip conventional-commit scope from the query (`fix(pose): …`) and report
   a zero-overlap slice (no query token in gold paths). Our cases keep the scope today.
2. **Snapshot**: we index HEAD (project-memory framing: "where does X live now"). Report
   separately the gold files created by the commit itself; for localization framing index `commit^`.
3. Strict Acc@k (all gold in top-k) next to recall@k; slice by |gold| = 1 / 2–4.
4. Dedupe near-identical subjects (Jaccard) and drop reverts/cherry-picks.
5. Tune on older commits, report on newer (time split); macro-average across repos.

## Candidate hypotheses, prioritized

| # | Idea | Source | Why here | Cost |
|---|---|---|---|---|
| 1 | **Oracle gap for two tools** (best of code-only vs decision-only per case) | Letta, Graphiti MCP | decides whether "two tools for the agent" is worth building | minutes, existing results |
| 2 | **Past-commits channel**: BM25 over prior commit subjects → files they changed, time-masked, fused by RRF | arXiv 2502.07067 | Russian query ↔ Russian history, bypasses RU→EN gap | small |
| 3 | **Soft edge weights** instead of the hard df cap: `mul·sqrt(refs)`, ×10 query identifiers, ×0.1 names defined in >5 places | Aider repomap.py | smoother hub control than cap | small |
| 4 | **BM25F**: path/definition match boost ×5; test/generated ÷5; docs in low priority (our extension) | zoekt, Continue | docs flooding at the seed | small |
| 5 | **Code embedder**: Qwen3-Embedding-0.6B (Apache-2.0, RU, MTEB-Code 75.4) vs CodeRankEmbed (MIT, 137M, EN) + RU→EN query translation | arXiv 2506.05176, CoRNStack | the dense channel is weak on RU→EN | needs CUDA torch for reasonable time |
| 6 | **tree-sitter tags** for defs/refs, drop builtins/third-party names | Aider queries, RepoGraph | regex refs are noisy | medium |
| 7 | **Decisions attached to code** (1-hop from top-k code files via co-change/path mentions) | Claude Code `paths:`, OpenHands triggers, HippoRAG 2 | LLM-free alternative to fixed splits | medium |
| 8 | **Supersede for decisions** (Jaccard of touched files + explicit "supersedes", keep history) | Graphiti, Mem0 additive | keeps decisions current | separate case set |
