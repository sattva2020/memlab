# memlab — research log

Every hypothesis below was committed with its success criteria before its run; results are
reported as they came out, including the ones that failed. Evaluation sets come from four
private repositories and are not published (see the README).

## Question

At a matched token budget, does any structured memory (graph propagation, typed
edges, consolidation) beat a well-tuned flat hybrid retriever on real projects?

The null hypothesis is kept on purpose: several recent studies find that graph
retrieval often underperforms plain RAG (arXiv:2506.05690, arXiv:2608.28978), and
that the winner depends on the corpus (arXiv:2608.05153).

## Hypotheses

| | Hypothesis | Grounding |
|---|---|---|
| H0 | Flat hybrid retrieval (BM25 + dense, RRF) is not worse than any graph scheme | arXiv:2608.28978, arXiv:2506.05690 |
| H1 | Lexical anchors + local graph expansion beat global Personalized PageRank for code | LARGER, arXiv:2605.16352 |
| H2 | Query-dependent edge weights / degree normalization remove hub leakage of PPR | CatRAG arXiv:2602.01965; GAAMA arXiv:2603.27910 |
| H3 | Decisions and notes should be kept as raw episodes and retrieved on their own channel; consolidation only explicitly | arXiv:2605.12978 |

## Method

- **Corpus**: git-tracked text files of a project at a pinned commit, cut into chunks
  (code at top-level declarations, markdown at headings, other text in windows).
- **Cases**: a query plus gold items `{path, symbol?}` — what an agent must see.
- **Budget**: every method gets the same number of tokens; ranked chunks are taken in
  order while they fit.
- **Metrics**: recall at file level and at symbol level; paired bootstrap 95% CI of
  the difference between methods.
- **Robustness** (per arXiv:2608.05153): at least two corpora and two embedders
  before any claim.

## Stages

1. Harness + flat baselines (BM25, dense, hybrid). ✅
2. Hypotheses H1–H6 on two development corpora, git-mined cases. ✅
3. Assemble the system from what wins; held-out test on a third corpus. ✅
4. Next: H7 (code vs docs in the seed), second embedder, larger held-out set, public report.

## Results

### Stage 1 — flat baselines, aifc360 (48 cases, 13,702 chunks, commit `47d4ba10`)

Recall at a matched token budget; embedder `paraphrase-multilingual-MiniLM-L12-v2` (max 128 tokens).

| Budget | Level | BM25 | Dense | Hybrid (RRF) |
|---|---|---|---|---|
| 4k | file | 0.52 | 0.57 | **0.64** |
| 4k | symbol | **0.30** | 0.20 | 0.28 |
| 16k | file | 0.65 | 0.74 | **0.82** |
| 16k | symbol | 0.48 | 0.35 | **0.54** |
| 38k | file | 0.70 | 0.85 | **0.87** |
| 38k | symbol | 0.55 | 0.50 | **0.63** |

Hybrid is significantly ahead of BM25 at file level at every budget (paired bootstrap,
95% CI excludes 0) and of dense at symbol level from 4k up. Full numbers:
`results/aifc360-baselines.json`.

Reference point (not the same unit, indicative only): a graph-memory system with
Personalized PageRank measured on the same 48 cases delivered ~0.91 at file level and
~0.54 at symbol level in a pack of comparable size (~38k tokens).

### Stage 2 / H1 — graph on top of the hybrid seed, aifc360

Graph built from the sources only: 3,157 file edges (Dart/TS imports, markdown links and
repo paths), 14,072 symbol-reference edges (a chunk mentions a name defined in another
file; names defined in >3 places or mentioned in >200 chunks are skipped), plus
same-file siblings. Two rankers on the hybrid seed: anchored local expansion
(LARGER-style) and global Personalized PageRank. Difference to hybrid with 95% CI.

| Budget | Level | Hybrid | expand all 10:3 *(a priori)* | expand refs 10:3 | PPR refs |
|---|---|---|---|---|---|
| 4k | file | 0.64 | 0.55 (−0.08, n.s.) | 0.65 (n.s.) | 0.62 (n.s.) |
| 4k | symbol | 0.28 | 0.32 (n.s.) | **0.37** (+0.09) | **0.38** (+0.09) |
| 16k | file | 0.82 | 0.81 (n.s.) | 0.85 (n.s.) | 0.87 (n.s.) |
| 16k | symbol | 0.54 | **0.60** (+0.06) | **0.65** (+0.11) | **0.66** (+0.12) |
| 38k | file | 0.87 | 0.87 (n.s.) | **0.90** (+0.03) | 0.90 (n.s.) |
| 38k | symbol | 0.63 | **0.68** (+0.06) | **0.71** (+0.08) | **0.74** (+0.11) |

Bold = 95% CI excludes 0. Full grid (edge-type ablation, anchors 5/10/20,
per-anchor 1/3/5): `results/aifc360-h1.json`.

Findings (one corpus, 48 cases — to be confirmed on a second corpus):

- **H0 is rejected at symbol level.** Structure adds +6..+12 points of symbol recall at
  16k–38k tokens with CIs clear of zero; at file level it is neutral.
- **The useful structure is symbol references.** Imports add little; same-file siblings
  help symbol recall but crowd other files out at small budgets (file −0.19 at 4k).
- **H1 is not supported here.** Global PPR over the reference graph is at least as good
  as anchored local expansion. Likely reason: this graph has no mega-hubs by
  construction (the df cap on reference edges is itself a degree control, cf. H2).
- Multiple comparisons: 10 variants × 6 cells. The a-priori setting (expand all 10:3)
  is the honest claim; the best variants (refs-only) are post-hoc and need the second
  corpus to be confirmed.

### Stage 2 / H2 — hub control

Same seed, global PPR over symbol references. The a-priori graph skips names mentioned
in more than 200 chunks (a frequency cap); the ablation removes that cap (27,444 instead
of 14,072 reference edges) and/or divides PPR mass by degree^0.5.

| Budget | Level | Hybrid | PPR capped | PPR no cap | no cap + degree norm |
|---|---|---|---|---|---|
| 4k | file | 0.64 | 0.62 | **0.46** (−0.17) | 0.56 |
| 4k | symbol | 0.28 | **0.38** (+0.09) | 0.23 | **0.21** (−0.08) |
| 16k | symbol | 0.54 | **0.66** (+0.12) | 0.57 | **0.61** (+0.07) |
| 38k | symbol | 0.63 | **0.74** (+0.11) | **0.74** (+0.11) | **0.76** (+0.13) |

- **Hub leakage is real and budget-dependent.** Without the cap, frequently referenced
  names become hubs and PPR drops below the flat baseline at 4k (file −0.17, CI clear of
  zero); at 38k the budget is large enough to absorb it.
- **Pruning hub edges beats degree normalization.** Degree^0.5 normalization recovers
  part of the loss on the uncapped graph but hurts the capped one at small budgets.
  The cheap frequency cap is the best hub control tried.

### Stage 2 / H3 — a budget channel for decisions

26 additional cases whose answer is a decision document (session notes, ADRs,
postmortems; `datasets/aifc360/decisions.json`). `channel:s:base` reserves share *s* of
the budget for the best decision chunks by the hybrid, the rest comes from *base*.

| Budget | Hybrid | PPR refs | channel 0.15 over PPR | channel 0.3 over hybrid |
|---|---|---|---|---|
| 4k | 0.50 | **0.33** (−0.17) | 0.60 (n.s.) | **0.71** (+0.21) |
| 16k | 0.54 | 0.50 | **0.75** (+0.21) | **0.79** (+0.25) |
| 38k | 0.75 | **0.56** (−0.19) | **0.83** (+0.08) | **0.90** (+0.15) |

Harm check on the 48 code cases: `channel 0.15 over PPR` keeps the code gains of PPR
(symbol 0.37 / 0.66 / 0.74); `channel 0.3 over hybrid` costs code up to −0.03 (n.s.).

- **H3 is supported.** Decisions lose to code in a shared ranking, and the graph makes
  it worse (it pulls toward code). A reserved slice of the budget fixes this without
  hurting code retrieval.
- **Best compromise so far:** PPR over capped symbol references + a 15% decision channel.
- Caveats: 26 decision cases written by the same agent; shares 0.15/0.3 were the only
  two tried.

### Stage 2b — confirmation run (pre-registered 2026-09-27, before any result)

Cases mined from git history (`memlab cases-from-git`): query = commit subject,
gold = source files the commit changed (tests/generated excluded, 1–4 files). Nobody
chose wording or answers. 80 cases per corpus:
`datasets/aifc360/git-cases.json` (commit `47d4ba10`), `datasets/cryonick/git-cases.json`
(Cryonick CRM, commit `b2ed2230`, TypeScript: NestJS + Next.js).

Methods, fixed in advance: `hybrid` (baseline), `ppr:refs` (capped refs — the stage 2
winner), `ppr:refs+imports+siblings`, `expand:refs+imports+siblings:10:3` (H1 a-priori),
`ppr:refs:df=1e9` (H2 no cap), `channel:0.15:ppr:refs` (proposed system).
Budgets 4k / 16k / 38k, file level (gold is file-level).

Confirmation criteria:
- **C1** `ppr:refs` beats `hybrid` with 95% CI above 0 at ≥1 budget on **both** corpora,
  and is not significantly worse at any budget.
- **C2** `ppr:refs:df=1e9` is below `ppr:refs` at 4k on both corpora (hub leakage).
- **C3** `channel:0.15:ppr:refs` is not significantly worse than `ppr:refs` on code cases.

#### Result (run 2026-09-27, `results/aifc360-confirm.json`, `results/cryonick-confirm.json`)

File recall, 80 git-mined cases per corpus; bold = 95% CI vs hybrid excludes 0.

| Corpus | Budget | Hybrid | PPR refs | PPR refs no cap | PPR refs + 15% decisions |
|---|---|---|---|---|---|
| aifc360 | 4k | 0.71 | 0.68 | **0.44** | 0.65 |
| aifc360 | 16k | 0.87 | 0.85 | **0.82** | **0.82** |
| aifc360 | 38k | 0.93 | 0.91 | 0.90 | 0.90 |
| cryonick | 4k | 0.46 | **0.55** | 0.48 | **0.54** |
| cryonick | 16k | 0.71 | 0.73 | 0.70 | 0.69 |
| cryonick | 38k | 0.83 | 0.87 (CI −0.00) | 0.82 | 0.86 |

Channel vs PPR refs (`--ref ppr:refs`): aifc360 −0.03 / **−0.03** / −0.01, cryonick −0.01 / **−0.04** / −0.01.

- **C1 — not confirmed.** PPR refs beats hybrid on Cryonick (+0.09 at 4k, never worse)
  but not on aifc360 (−0.03…−0.01, n.s.). The effect is corpus-dependent: the graph helps
  where the flat seed is weak (Cryonick hybrid 0.46 at 4k) and adds nothing where it is
  already strong (aifc360 0.71), consistent with arXiv:2608.05153.
- **C2 — confirmed on both corpora.** Removing the hub cap lowers PPR at 4k
  (0.68 → 0.44 and 0.55 → 0.48).
- **C3 — not confirmed.** A fixed 15% decision channel costs code tasks 3–4 points at
  16k on both corpora (CI just below 0). The decision gain (stage 2, +0.21 at 16k) was
  measured on self-written cases; a fixed quota is a real trade-off, not free.
- Not significantly worse anywhere: PPR over capped symbol references. It is the safe
  default; the decision channel needs a per-query gate rather than a fixed share.

### Stage 2c — H4: a per-query gate for the decision channel (pre-registered 2026-09-27)

Hypothesis H4: reserving budget for decisions only when the query itself pulls toward
them keeps the decision gain without the code cost of a fixed quota.

- Gate, fixed in advance: `gate:10:0.3:ppr:refs` — if the hybrid seed's top-10 holds at
  least one decision chunk, 30% of the budget goes to the decision channel; otherwise the
  query is served by `ppr:refs` unchanged. No LLM, no keyword list.
- Decision cases mined from git (`cases-from-git --decisions`): a commit that added an
  ADR or postmortem; query = its subject, gold = the added document(s); bookkeeping
  commits (scope amg/release, >2 docs) skipped. Cryonick 80 cases, aifc360 20 cases.
- Code cases: the stage 2b git cases.
- Reference methods: `ppr:refs`, `channel:0.15:ppr:refs`, `channel:0.3:ppr:refs`.

Criteria (compared with `--ref ppr:refs`):
- **D1** on decision cases the gate beats `ppr:refs` with 95% CI above 0 at ≥2 of 3
  budgets on both corpora;
- **D2** on code cases the gate is not significantly worse than `ppr:refs` at any budget
  on either corpus.

#### Result (run 2026-09-27, `results/*-h4-*.json`)

File recall vs `ppr:refs` (95% CI). Gate activation: Cryonick 71/80 decision and 31/80
code queries; aifc360 12/20 decision and 18/80 code queries.

| Corpus / cases | Budget | ppr:refs | gate | fixed 15% | fixed 30% |
|---|---|---|---|---|---|
| cryonick / decisions | 4k | 0.67 | **+0.156** | **+0.21** | **+0.23** |
| cryonick / decisions | 16k | 0.86 | +0.031 [0.000, 0.075] | **+0.08** | **+0.09** |
| cryonick / decisions | 38k | 0.91 | +0.037 [0.000, 0.087] | **+0.05** | **+0.06** |
| aifc360 / decisions | 4k | 0.45 | +0.150 [0.000, 0.300] | **+0.55** | **+0.55** |
| aifc360 / decisions | 16k | 0.62 | +0.050 [0.000, 0.150] | **+0.38** | **+0.38** |
| aifc360 / decisions | 38k | 0.82 | 0.000 | **+0.17** | **+0.17** |
| cryonick / code | 4k | 0.55 | −0.03 | −0.01 | **−0.08** |
| cryonick / code | 16k | 0.73 | **−0.05** | **−0.04** | **−0.07** |
| cryonick / code | 38k | 0.87 | −0.02 | −0.01 | **−0.04** |
| aifc360 / code | 4k–38k | 0.68–0.91 | −0.01…0.00 | −0.03…−0.01 | −0.04…−0.02 |

- **D1 — not met.** The gate beats PPR with CI above 0 only at 4k on Cryonick; elsewhere
  the lower bound is exactly 0 (it never hurts decision cases, but often does not fire).
- **D2 — not met.** On Cryonick the gate fires on 39% of code tasks (133 postmortems sit
  close to the code they describe) and costs −0.05 at 16k.
- **H4 is not supported.** A seed-based gate is too loose where decisions are many and
  too strict where they are few. The decision/code trade-off remains; a fixed 15% channel
  is the better of the tried policies (large decision gain, −0.01…−0.04 on code).

### Stage 2d — ideas from graphify, and graphify as an external baseline (pre-registered 2026-09-27)

graphify (Apache-2.0, github.com/Graphify-Labs/graphify, v0.9.69) builds a tree-sitter
graph and answers by BFS from lexically picked seeds. Two of its rules become hypotheses:

- **H5 — hubs as non-transit nodes.** In `_bfs`, nodes with degree ≥ max(50, p99) are
  reachable but not expanded. As PPR (`transit=1`): hubs keep incoming mass, their outflow
  returns to the teleport vector.
- **H6 — seed selection.** `_pick_seeds`: drop seeds below 20% of the top lexical score,
  one seed per symbol label, at least one seed per matched query term. Here (`seeds=gfy`):
  hybrid top-50 filtered by BM25 ≥ 0.2·max (dense top-5 kept), deduped by symbol, plus the
  best BM25 chunk for every query token.

Criteria (git code cases, both corpora; budgets 4k/16k/38k and files@5/10/20):
- **E1 (H5)** `ppr:refs:df=1e9&transit=1` beats `ppr:refs:df=1e9` with CI above 0 at 4k on
  both corpora, and `ppr:refs:transit=1` is not significantly worse than `ppr:refs` anywhere.
- **E2 (H6)** `ppr:refs:seeds=gfy` beats `ppr:refs` with CI above 0 in ≥1 cell on ≥1
  corpus and is not significantly worse in any cell on either corpus.

graphify baseline (descriptive, no hypothesis): `tools/graphify_adapter.py` builds
graphify's graph over the same files and records the files its `query` answer surfaces,
the answer cut to the same token budget; `files@k` uses its first-appearance order.
Caveat stated up front: graphify returns one-line node pointers, not code bodies, so at
an equal budget it lists more files — `files@k` is the fair comparison.

#### Result (run 2026-09-27, `results/*-h5h6.json`, `results/*-h5-nocap.json`, `results/ext/`)

Git code cases, file recall; difference to `ppr:refs`, bold = 95% CI excludes 0.

| Corpus | Metric | hybrid | ppr:refs | + transit (H5) | + gfy seeds (H6) | graphify 0.9.69 |
|---|---|---|---|---|---|---|
| aifc360 | 4k | 0.71 | 0.68 | 0.70 | 0.68 | **0.47** |
| aifc360 | 16k | 0.87 | 0.85 | **0.87** | 0.84 | **0.61** |
| aifc360 | 38k | 0.93 | 0.91 | 0.92 | 0.90 | **0.62** |
| aifc360 | files@5 | **0.55** | 0.40 | **0.59** | 0.44 | **0.26** |
| aifc360 | files@10 | 0.70 | 0.66 | **0.77** | **0.69** | **0.38** |
| aifc360 | files@20 | 0.79 | 0.81 | 0.83 | 0.81 | **0.41** |
| cryonick | 4k | **0.46** | 0.55 | 0.53 | 0.57 | **0.32** |
| cryonick | 16k | 0.71 | 0.73 | 0.73 | **0.79** | **0.51** |
| cryonick | 38k | 0.83 | 0.87 | 0.87 | 0.88 | **0.56** |
| cryonick | files@5 | 0.36 | 0.38 | **0.43** | **0.45** | **0.09** |
| cryonick | files@10 | 0.49 | 0.54 | 0.58 | 0.57 | **0.12** |
| cryonick | files@20 | 0.66 | 0.67 | 0.67 | **0.71** | **0.23** |

No-cap pair (vs `ppr:refs:df=1e9`): transit adds +0.249 / +0.046 at 4k and
+0.397 / +0.245 at files@5 (aifc360 / cryonick), all CIs above 0.

- **E1 (H5) — met.** Transit blocking rescues the uncapped graph on both corpora and never
  significantly hurts the capped one (Cryonick 4k: −0.023, CI [−0.056, 0.0000] — upper
  bound exactly 0 at 4 decimals, so not significant; same boundary rule as D1).
  The files@k view exposes why: plain PPR ranks hub chunks first (files@5 0.40 on aifc360,
  below the flat hybrid's 0.55); blocking transit through hubs fixes the head of the list.
- **E2 (H6) — met.** graphify-style seeds help on Cryonick (+0.067 at 16k, +0.073 at
  files@5) and aifc360 files@10 (+0.030); not significantly worse anywhere (aifc360 38k:
  −0.010, CI [−0.027, 0.0000] — boundary, not significant).
- **graphify as shipped is far behind** every memlab method on both corpora (files@5 0.26 /
  0.09). Its `query` seeds lexically on node labels without embeddings; its published
  LOCOMO numbers use a separate embedding-based adapter, not measured here.
- Both hypotheses were confirmed on the same cases used to design them; a combined
  H5+H6 system must be tested on held-out cases before any claim.

### Stage 3 — held-out test of the assembled system (pre-registered 2026-09-27)

Candidate, fixed before any held-out result: **`ppr:refs:transit=1&seeds=gfy`** — hybrid
seed (BM25 + dense, RRF) → graphify-style seed filter → Personalized PageRank over
symbol-reference edges (names mentioned in ≤200 chunks, defined in ≤3) with hubs
(degree ≥ max(50, p99)) blocked as transit.

Held-out corpus: **HealBot** (`projects/healbot.toml`, commit `45691b1`, TS/JS: Node bot +
payments services, React site) — never used to design any method. Cases mined with the
unchanged stage 2b rule (feat/fix/refactor/perf, 1–4 source files): **25 cases**, stated in
advance as low power (differences below ~0.10 are unlikely to be detectable). HealBot
has 4 decision documents, so only code retrieval is tested.

Criteria (file recall at 4k/16k/38k and files@5/10/20):
- **F1** the candidate is not significantly worse than `hybrid` in any cell and beats it
  with CI above 0 in at least one cell;
- **F2** the candidate is not significantly worse than `ppr:refs` in any cell.

Also reported (descriptive): graphify 0.9.69 on HealBot; the candidate on the two
development corpora (it was never run as a combination before).

#### Result (run 2026-09-27, `results/healbot-heldout.json`, `results/*-candidate.json`)

Candidate `ppr:refs:transit=1&seeds=gfy`, file recall; bold = 95% CI excludes 0.

| Corpus | vs | 4k | 16k | 38k | files@5 | files@10 | files@20 |
|---|---|---|---|---|---|---|---|
| **HealBot (held-out, n=25)** | hybrid | +0.13 | +0.07 | +0.03 | +0.05 | **+0.16** | +0.06 |
| **HealBot (held-out, n=25)** | ppr:refs | **+0.17** | +0.04 | +0.07 | +0.04 | **+0.25** | +0.01 |
| aifc360 (dev, n=80) | hybrid | +0.02 | +0.01 | −0.02 | +0.03 | +0.06 | **+0.04** |
| cryonick (dev, n=80) | hybrid | **+0.12** | **+0.08** | **+0.04** | **+0.09** | **+0.11** | +0.05 |

- **F1 — met** (held-out): never significantly worse than hybrid; better at files@10
  (+0.157, CI [+0.003, +0.300] — a narrow margin at n=25).
- **F2 — met** (held-out): never significantly worse than `ppr:refs`; better at 4k (+0.17)
  and files@10 (+0.25). The H5/H6 fixes replicate on a corpus that played no part in design.
- Across all three corpora the candidate is never significantly worse than the flat hybrid
  in any of 18 cells, and significantly better in 7.
- **graphify on HealBot is strong** (4k 0.71, files@5 0.36 vs candidate 0.43 / 0.21), the
  reverse of the two larger corpora. Diagnosis (descriptive): 68 of the hybrid's 125 top-5
  files on HealBot are documentation (`docs/`, `.ai-factory/`, root READMEs) while every
  gold file is code; graphify seeds on code symbol labels and is not distracted by prose.
  This becomes **H7** (separate code and documentation in the seed), to be tested on a new
  held-out corpus — HealBot is no longer held-out for it.

### Stage 4 — H7 (code vs docs) on held-out WhaleCast, and a second embedder (pre-registered 2026-09-27)

**H7.** On HealBot most of the hybrid's top files were documentation while the tasks
needed code. Dropping docs would win on git cases by construction (their gold is code), so
H7 keeps docs and only stops them from flooding: code chunks and non-code chunks are ranked
separately and merged 3:1 (`mix:3:1:<base>`), each side in its own order.
Candidate base: `ppr:refs:transit=1&seeds=gfy`.

Held-out: **WhaleCast** (`projects/whalecast.toml`, commit `3821d81`, TS indexer; 1,782 doc
chunks vs 755 code chunks), 22 git cases (unchanged rule). HealBot inspired H7, so it now
counts as development data.

- **G1** (WhaleCast, code): the mixed candidate beats the candidate with CI above 0 in ≥1
  cell and is not significantly worse in any cell.
- **G2** (doc protection, dev decision sets: aifc360 git-decisions 20, cryonick
  git-decisions 80, aifc360 decisions.json 26): the mixed candidate is not significantly
  worse than the candidate at 16k and 38k (top-k cells may drop by design).
- **G3** (dev code sets aifc360, cryonick, healbot): not significantly worse anywhere.

#### H7 result (run 2026-09-27, `results/*-h7*.json`)

`mix:3:1` vs the candidate; bold = 95% CI excludes 0.

| Set | 4k | 16k | 38k | files@5 | files@10 | files@20 |
|---|---|---|---|---|---|---|
| **WhaleCast code (held-out, 22)** | +0.01 | +0.03 | 0 | +0.01 | +0.03 | +0.01 |
| aifc360 git-decisions (20) | +0.05 | +0.03 | 0 | +0.05 | +0.08 | +0.08 |
| cryonick git-decisions (80) | **−0.09** | **−0.05** | 0 | −0.05 | **−0.13** | **−0.06** |
| aifc360 decisions (26) | 0 | −0.10 | 0 | **−0.17** | −0.06 | −0.08 |
| aifc360 code (80) | −0.01 | −0.00 | 0 | −0.01 | −0.02 | +0.00 |
| cryonick code (80) | +0.03 | +0.01 | 0 | +0.02 | +0.04 | **+0.04** |
| healbot code (25, dev) | 0 | **+0.14** | +0.01 | −0.01 | −0.04 | **+0.12** |

- **G1 — not met.** On held-out WhaleCast the interleave trends up (+0.01…+0.03) but no
  cell clears 0 at n=22.
- **G2 — not met.** It costs Cryonick's decision retrieval 5 points at 16k (CI below 0).
- **G3 — met.** Never significantly worse on code sets.
- **H7 is not supported as a fixed ratio** — the same code/doc trade-off as H3/H4: any fixed
  split between content types helps one kind of question and hurts the other.
- **Not pre-registered, reported as corroboration:** on WhaleCast the stage 3 candidate
  beats the flat hybrid at 4k (+0.23), 38k (+0.09) and files@5 (+0.13), all CIs above 0 —
  a second corpus unseen during design where the assembled system wins.

**Second embedder (R1).** `intfloat/multilingual-e5-small` (512-token window, e5
prefixes) replaces MiniLM-L12 (128 tokens) for hybrid and candidate on aifc360, cryonick
and healbot git cases. R1: the sign of (candidate − hybrid) at files@10 is the same as
with MiniLM on all three corpora. Reported: how much the hybrid itself moves.

#### R1 result (run 2026-09-27, `results/*-e5.json`)

Hybrid and candidate with `multilingual-e5-small` (512 tokens) instead of MiniLM-L12 (128).
Candidate − hybrid; bold = 95% CI excludes 0.

| Corpus | hybrid files@10 MiniLM → e5 | Δ files@10 MiniLM | Δ files@10 e5 | Δ 4k e5 | Δ 16k e5 |
|---|---|---|---|---|---|
| aifc360 | 0.70 → 0.72 | +0.06 | +0.002 | −0.03 | **+0.05** |
| cryonick | 0.49 → 0.56 | **+0.11** | +0.07 | **+0.08** | **+0.08** |
| healbot | 0.31 → 0.44 | **+0.16** | −0.02 | +0.01 | +0.09 |

- **R1 — not met.** The sign at files@10 flips on HealBot (+0.16 → −0.02).
- A better embedder lifts the flat hybrid most where it was weakest (HealBot +0.13 at
  files@10, Cryonick +0.07), and the graph's advantage shrinks by roughly the same amount.
  Reading: part of what the graph contributed was compensation for a weak dense channel.
  On Cryonick the advantage survives (+0.08 at 4k and 16k).
- Consequence: the embedder is a first-order factor and must be fixed before the graph is
  tuned further; results above are conditional on MiniLM.

### Stage 5 — ideas from the survey (pre-registered 2026-09-27; see `docs/related-work.md`)

All runs with `multilingual-e5-small`. C = `ppr:refs:transit=1&seeds=gfy`. Code sets: git cases
of aifc360, cryonick, healbot, whalecast. Decision sets: git-decisions of aifc360 (20),
cryonick (80), aifc360 decisions.json (26).

- **H8 — two tools instead of one shared ranking** (Letta, Graphiti MCP, OpenHands). Tools:
  `only:code:C` and `only:dec:hybrid`; upper bound = per case the better of the two
  (`oracle`). Decision rule D8: build two tools if the oracle beats the best shared policy
  (max of C and `channel:0.15:C`) by ≥0.05 at 16k on every decision set and every code set.
  Caveat stated in advance: git code gold is code by construction, so `only:code` is
  favoured on code sets; the decision sets are the informative part.
- **H9 — past-commits channel** (arXiv:2502.07067): BM25 over subjects of commits strictly
  before the case's commit → files they changed (score / files in commit, top 20
  commits), fused into the seed by RRF (`commits=1`). Criterion: C+commits beats C with CI
  above 0 in ≥1 cell on ≥2 of 4 code corpora and is not significantly worse anywhere.
- **H10 — Aider edge weights** instead of the hard caps: `mul·sqrt(mentions)`, ×10 long
  snake/camel names, ×0.1 `_private`, ×0.1 names defined in >5 files, no df/defs caps, plus
  personalization toward chunks defining identifiers named in the query
  (`w=aider&df=1e9&defs=1e9&qid=1`). Same criterion as H9.
- **R2 — scope leakage**: queries with the leading `scope:` removed (`--strip-scope`).
  Criterion: the sign of (C − hybrid) at files@10 is unchanged on corpora where it was
  significant with the scope kept.

#### Stage 5 results (run 2026-09-27, e5-small, `results/*-s5-*.json`)

Differences vs C; * = 95% CI excludes 0. Cells: 4k / 16k / 38k / files@5 / @10 / @20.

- **H9 (past commits) — not met.** Better only on aifc360 (files@10 +0.07*); significantly
  worse on HealBot (4k −0.10*) and WhaleCast (files@10 −0.05*).
- **H10 (Aider weights) — not met.** Worse on aifc360 (files@10 −0.06*), better only on
  HealBot (files@5 +0.05*); the hard caps stay.
- **H8 (two tools) — decision rule D8 not met as registered.** At 16k the oracle beats the
  best shared policy by +0.01 / +0.06 / +0.11 / +0.04 on the code sets (aifc360, cryonick,
  healbot, whalecast) and by +0.02 / 0.00 / +0.14 on the decision sets (aifc360 git, cryonick
  git, aifc360 own): the fixed 15% channel already holds the one decision document at 16k.
  Not pre-registered, but decisive for tool design: at the head of the list the separate
  decision tool dominates any shared ranking (files@5 +0.68 / +0.28 / +0.40 over C) and the
  code tool beats C on every code set (files@10 +0.07 / +0.08 / +0.16 / +0.08, all *).
- **R2 (scope leakage).** Removing the `scope:` prefix lowers the flat hybrid by 4–11 points
  (4k: aifc360 −0.08*, healbot −0.11*), so the prefix does leak. The graph's advantage
  survives or grows without it (C − hybrid at 16k: cryonick +0.07*, healbot +0.13*,
  whalecast +0.13*); the registered files@10 check is vacuous (no files@10 difference was
  significant with the scope kept).

### Stage 5b — H11: a code-capable embedder (pre-registered 2026-09-27)

`Qwen/Qwen3-Embedding-0.6B` (Apache-2.0; MTEB-Code 75.4 per arXiv:2506.05176; multilingual)
on GPU (RTX 2070, fp16), window capped at **512 tokens** to match e5 — so only model
quality differs. Query instruction: "Given a description of a code change or a question
about a software project, retrieve the relevant source files and documents"; documents
without prefix. Runs: `hybrid` and C on the 4 code sets and the 3 decision sets; compared
with the e5 runs case by case (`scripts/compare.py`).

- **Q1** hybrid(Qwen3) beats hybrid(e5) at files@10 with CI above 0 on ≥2 of 4 code corpora
  and is not significantly worse on any code corpus or decision set.
- Reported: C − hybrid under Qwen3 (does the graph still add anything with a strong
  embedder?).

#### H11 result (run 2026-09-27, `results/*-qwen*.json`)

Hybrid, Qwen3 − e5 (paired per case); * = CI excludes 0.

| Corpus | 4k | 16k | 38k | files@5 | files@10 | files@20 |
|---|---|---|---|---|---|---|
| aifc360 | +0.00 | +0.01 | −0.01 | −0.01 | −0.00 | +0.05 |
| cryonick | **−0.13*** | −0.03 | −0.00 | −0.06 | −0.06 | +0.02 |
| healbot | **+0.19*** | **+0.14*** | **+0.07*** | +0.04 | +0.13 | **+0.10*** |
| whalecast | −0.00 | −0.04 | 0.00 | −0.02 | +0.01 | −0.01 |

- **Q1 — not met.** No corpus shows a significant files@10 gain, and Qwen3 is significantly
  worse on Cryonick at 4k. No embedder is uniformly better: Qwen3 transforms HealBot and hurts
  Cryonick's head of the list.
- With Qwen3 the graph adds +0.10* / +0.09* (Cryonick 4k / 16k) and +0.09* / +0.10* (WhaleCast),
  +0.03* on aifc360 at 16k, and nothing on HealBot — the same pattern as with e5 and MiniLM.
- Decision sets (not part of Q1, reported): Qwen3 lifts the hybrid on decision questions —
  own 26 cases +0.27* / +0.23* / +0.23* (4k/16k/38k), files@10 +0.21*; aifc360 git-decisions
  +0.13 / +0.08 / +0.08 (lower bounds at 0); Cryonick git-decisions +0.02…+0.04 (n.s.). Prose
  in Russian is where the multilingual model pays off; for code the choice stays per project.
- The graph acts as a stabiliser across embedders: C on Qwen3 vs the e5 hybrid at 16k is
  +0.04 / +0.07* / +0.12* / +0.06 (aifc360 / cryonick / healbot / whalecast).

### Stage 6 — prototype tools on natural-language questions (pre-registered 2026-09-27)

A live smoke test of the MCP prototype on Russian natural-language questions failed where the
git cases looked fine: BM25 and the per-term seed rule latched onto function words ("как",
"и", "где"), and the BM25 half of the decision hybrid buried the right note that Qwen3 alone
ranked first. Git subjects are short technical phrases, so this is a query-distribution
shift. Test sets with natural questions: `datasets/aifc360/cases.json` (48) and
`decisions.json` (26); regression checks on git cases.

- **S1 — query stopwords** (RU/EN function words dropped from the query only, `--stopwords`),
  code tool = `only:code:C`, e5: not significantly worse in any cell on aifc360 natural or git
  cases, better in ≥1 cell on the natural set.
- **S2 — decision tool**: `only:dec:dense` vs `only:dec:hybrid`, Qwen3, with stopwords, on the
  26 natural decision questions and the two git-decision sets: pick dense-only if it is not
  significantly worse on any set and better on ≥1; otherwise keep the hybrid.

Also fixed: stage 1 README reported 25,931 chunks for aifc360 — that count predates the
chunker fix; every aifc360 run used 13,702 chunks (as recorded in the result files).

#### Result (run 2026-09-27, `results/*-s6-*.json`, log `results/stage6.log`)

**S1 — not met.** `only:code:C`, stopwords vs plain (paired, `scripts/compare.py`):

| set | 4k | 16k | 38k | files@5 | files@10 | files@20 |
|---|---|---|---|---|---|---|
| natural (48) | **+0.094** | **+0.064** | +0.031 | +0.042 | +0.063 | **+0.030** |
| git (80) | −0.038 | −0.015 | −0.008 | +0.020 | **−0.041** | −0.025 |

Clear gains on natural questions, but one significant loss on git subjects (files@10), so the
rule fails. Deployment decision, outside the rule: the server keeps query stopwords, because
agents send natural-language questions, not commit subjects — the natural set is the target
distribution. Either way the code tool beats plain hybrid: natural files@5/10/20
+0.17/+0.18/+0.15, git files@10/20 +0.06/+0.08 (all significant).

**S2 — not met; keep the hybrid.** `only:dec:dense` vs `only:dec:hybrid`: no significant
difference in any cell on any set (aifc360 natural files@5 +0.04, aifc360 git files@5 −0.05,
cryonick git files@5 −0.04, all CIs touch 0). Both are at or near ceiling (0.93–1.00) — these
sets no longer discriminate between decision retrievers; a harder set is needed.

**Live smoke test after the fixes** ("как считаются повторения при приседаниях и где
калибровка"): the code tool now returns `rep_counter.dart AdaptiveRepCounter` and
`pose_processor.dart processFrameExercise` (before: roadmap, specs and docs), but still ranks
`entitlements_provider.dart` first. The decision query "почему не приходят пуш-уведомления
неактивным пользователям" still misses: Qwen3 alone ranks the push/check-inactive note first,
BM25 has no match because Russian inflections ("уведомления", "неактивным") do not hit the
note's forms. Candidate next hypothesis: Russian stemming for BM25.

### Stage 7 — H12, Russian stemming in BM25 (pre-registered 2026-09-27)

BM25 matches surface forms, so a Russian question ("уведомления", "неактивным") misses notes
that use other inflections of the same words. H12: Snowball Russian stemming of Cyrillic
tokens, on both documents and queries (`--stem`); Latin tokens (identifiers) unchanged; query
stopwords are still dropped, before stemming. Paired comparison, stem vs no stem, both with
stopwords, for the two deployed tools:

- code tool `only:code:C`, e5: aifc360 natural (48), aifc360 git (80), cryonick git (80);
- decision tool `only:dec:hybrid`, Qwen3: aifc360 decisions (26), aifc360 git-decisions (20),
  cryonick git-decisions (80).

**H12 is met** if stemming is not significantly worse in any cell on any of the six sets and
significantly better in ≥1 cell. The decision sets sit near ceiling (stage 6), so a gain there
is unlikely to reach significance; the live push-notification query is reported as a
qualitative check, not as evidence.

#### Result (run 2026-09-27, `results/*-s7-*.json`, log `results/stage7.log`)

**H12 — not met.** Stem vs no stem, paired:

| set | 4k | 16k | 38k | files@5 | files@10 | files@20 |
|---|---|---|---|---|---|---|
| code, aifc360 natural | −0.004 | −0.010 | −0.005 | +0.014 | 0 | −0.009 |
| code, aifc360 git | 0 | 0 | 0 | 0 | 0 | 0 |
| code, cryonick git | −0.005 | +0.012 | +0.037 | **−0.058** | −0.059 | +0.023 |
| decisions, aifc360 natural | 0 | 0 | 0 | 0 | 0 | 0 |
| decisions, aifc360 git | 0 | 0 | 0 | 0 | 0 | 0 |
| decisions, cryonick git | +0.013 | 0 | 0 | +0.013 | +0.013 | +0.013 |

No significant gain anywhere, one significant loss (cryonick code files@5). aifc360 git
subjects are English, so stemming Cyrillic is a no-op there. Qualitative: on the push query
stemming lifts the re-engagement note into BM25's top 2, but the push/check-inactive note
still does not reach the top 5. The server stays without stemming. (The cryonick no-stem
baseline first crashed — `only:code:C` needs `C` in the method list — and was rerun.)

### Stage 8 — H13, a hard decision set (pre-registered 2026-09-27)

The existing decision sets are at ceiling (0.93–1.00), so they cannot choose between decision
retrievers. New set `datasets/{aifc360,cryonick}/hard-decisions.json`, 40 questions each:

- **Documents:** 40 per corpus sampled at random (seed 7) from the decision channel, ≥400
  characters (`scripts/sample_decisions.py`); older-set gold documents stay eligible.
- **Questions:** written by fresh subagents that see only the sampled documents, 10 per
  agent, never any retriever output. One Russian question per document, the way a developer
  would ask months later: symptoms and goals in their own words, no file names, identifiers,
  dates, or copied distinctive phrases; specific enough that this document answers it.
  Gold = the source document.
- **Lexical filter, fixed before any run:** a question is dropped if ≥60% of its content
  tokens (stopwords removed, Russian stems) occur in its document.

**H13:** on this set, `only:dec:dense` vs `only:dec:hybrid` (Qwen3, query stopwords). Pick
dense-only if it is not significantly worse in any cell on either corpus and significantly
better in ≥1 cell; otherwise keep the hybrid. Also reported: `bm25` alone, to show the set is
harder than the old ones (expected: clearly lower files@5 than on the stage 6 sets).
Known limitation: another document may also answer a question; gold has only the source.

#### Result (run 2026-09-27, `results/*-s8-*.json`, log `results/stage8.log`)

Set: 78 questions written (one agent skipped 2 of its 10 documents), the lexical filter
dropped 12 (4 aifc360, 8 cryonick) → **66 cases: aifc360 36, cryonick 30**; mean overlap of
kept questions 0.28 / 0.38.

**H13 — met; the decision tool becomes dense-only.** `only:dec:dense` vs `only:dec:hybrid`:

| corpus | files@5 | files@10 | files@20 | 4k / 16k / 38k |
|---|---|---|---|---|
| aifc360 (36) | 0.94 → 0.97, +0.03 | 1.00 → 1.00 | 1.00 → 1.00 | tie at 1.00 |
| cryonick (30) | 0.77 → 0.97, **+0.20** [+0.07, +0.37] | 0.93 → 0.97, +0.03 | tie | tie at 0.97–1.00 |

Never worse, significantly better once. How hard the set is (BM25 alone, files@5): cryonick
0.43 vs 0.93 on its old git-decision set — much harder; aifc360 0.69 vs 0.67 on its old
natural set — not harder for BM25, so on aifc360 the new set mostly adds cases, not
difficulty. Live check: "почему не приходят пуш-уведомления неактивным пользователям" now
returns the push/check-inactive note first.

### Stage 9 — memlab vs AMG on the same sets (pre-registered 2026-09-27)

AMG (the associative memory graph this project started from, PolyForm Strict) was never
measured here. It is now run as an external system, like graphify, through its own
`retrieve.py` (installed skill, unmodified), on a **copy** of each store — retrieval appends a
co-activation log, which must not touch the live memory.

- **Stores.** aifc360: the store committed in the corpus snapshot itself
  (`memlab-corpus/aifc360/.claude/amg`, same commit 47d4ba10 as every aifc360 run) — a
  like-for-like comparison. Cryonick, HealBot, WhaleCast: their corpus snapshots carry no
  store, so the live store of each project is copied; its sources may differ slightly from
  the snapshot (reported per corpus, secondary evidence).
- **What AMG returns.** One pack (default profile, ~38k tokens: strategic 4k, tactical 10k,
  operational 24k). Files are read from the pack's `path` pointers in order of appearance;
  the budget cells cut the pack text at budget × 4 characters, as for graphify; files@k use
  the order of appearance. AMG gets no split into code and decisions — that is its design.
- **Sets and memlab counterpart** (paired on the same cases):
  code sets → `only:code:C` with stopwords, e5 (aifc360 natural 48, aifc360 git 80,
  cryonick git 80, healbot git 25, whalecast git 22);
  decision sets → `only:dec:dense`, Qwen3 (aifc360 decisions 26, aifc360 git-decisions 20,
  aifc360 hard 36, cryonick git-decisions 80, cryonick hard 30).
- **Verdict per set:** memlab ahead if not significantly worse in any cell and significantly
  better in ≥1; AMG ahead by the mirror rule; otherwise a tie. **Overall claim "memlab beats
  AMG"** only if memlab is ahead on a majority of the ten sets and AMG is ahead on none —
  and on the like-for-like aifc360 sets in particular.
- Known confound: AMG stores ingest session transcripts and notes that may describe the very
  commits the git cases come from; this can favour AMG on git sets. Reported, not corrected.

#### Result (run 2026-09-27, `results/*-s9-*.json`, `results/ext/amg-*.json`, `results/stage9-summary.txt`)

Adapter check: AMG's in-process pack is byte-identical to its CLI pack on two probe queries.
AMG → memlab, mean per cell; * = 95% CI of the paired difference excludes 0.

| set | n | 4k | 16k | 38k | files@5 | files@10 | files@20 | ahead |
|---|---|---|---|---|---|---|---|---|
| aifc360 natural | 48 | 0.51→0.63 | **0.92→0.80** | **0.93→0.82** | 0.45→0.57 | 0.64→0.69 | 0.74→0.75 | AMG |
| aifc360 git | 80 | **0.45→0.71** | **0.75→0.90** | **0.77→0.93** | **0.28→0.59** | **0.43→0.76** | **0.60→0.88** | memlab |
| cryonick git | 80 | **0.21→0.70** | **0.39→0.87** | **0.39→0.91** | **0.09→0.57** | **0.21→0.72** | **0.30→0.82** | memlab |
| healbot git | 25 | **0.00→0.59** | **0.05→0.82** | **0.05→0.87** | **0.01→0.30** | **0.01→0.57** | **0.05→0.81** | memlab |
| whalecast git | 22 | 0.84→0.88 | 0.88→0.95 | **0.88→0.97** | 0.62→0.56 | 0.76→0.77 | 0.84→0.91 | memlab |
| aifc360 decisions | 26 | **0.04→1.00** | **0.15→1.00** | **0.21→1.00** | **0.04→1.00** | **0.04→1.00** | **0.04→1.00** | memlab |
| aifc360 git-dec. | 20 | **0.00→1.00** | **0.23→1.00** | **0.23→1.00** | **0.00→0.93** | **0.00→0.97** | **0.00→1.00** | memlab |
| aifc360 hard | 36 | **0.03→1.00** | **0.17→1.00** | **0.17→1.00** | **0.03→0.97** | **0.03→1.00** | **0.03→1.00** | memlab |
| cryonick git-dec. | 80 | **0.00→0.95** | **0.00→0.99** | **0.00→0.99** | **0.00→0.93** | **0.00→0.95** | **0.00→0.96** | memlab |
| cryonick hard | 30 | **0.00→0.97** | **0.00→1.00** | **0.00→1.00** | **0.00→0.97** | **0.00→0.97** | **0.00→0.97** | memlab |

**Overall rule — not met as written:** memlab is ahead on 9 of 10 sets, but AMG is ahead on
one (aifc360 natural), and the rule required "AMG ahead on none". AMG's two significant cells
there are budget cells, where the measures differ in kind: AMG's 38k pack *names* ~118 files
in one-line summaries, memlab's 38k budget holds ~20–40 files' *code*. On the ranking cells
of that set the difference is not significant (files@5 +0.12, CI crosses 0).

Why AMG scores so low elsewhere, checked per case, not an adapter fault:

- **Not indexed** (deployment scope): Cryonick's store indexes only code and `docs/dev` (user
  decision 2026-08-06), so none of its 81 ADR/postmortem gold documents has a node; HealBot's
  store holds 41 code nodes, covering 22 of 60 gold files; Cryonick runs with embeddings
  disabled.
- **Indexed but not surfaced:** aifc360 decision notes are in the store (22 of 30), yet AMG
  ranks them 70th–3000th of 14,193 nodes and they rarely enter the pack, which fills with hub,
  module and pattern summaries. On the covered subset (every gold file has an AMG node):
  aifc360 decisions files@5 0.05 → 1.00, hard set 0.04 → 1.00, aifc360 git 0.33 → 0.59,
  cryonick git 0.12 → 0.52 (all significant); aifc360 natural and WhaleCast stay ties on
  ranking cells.

Reading: on this project family AMG is competitive on broad natural-language questions about
a subsystem (its pack is built around module summaries) and on the small WhaleCast repo, and
far behind on locating the files of a concrete change and on retrieving decisions.
### Stage 10 — three-way table: memlab, AMG, graphify 0.9.70 (pre-registered 2026-09-28)

Stage 5 measured graphify 0.9.69 on code only (AST graph, no semantic pass), so its answer
to decision questions was never tested. graphify 0.9.70 is now built the way its README
recommends for mixed repos: `graphify extract` over a mirror of exactly the memlab corpus
files, code by tree-sitter, docs through its semantic pass with `--backend gemini`
(default model gemini-3-flash-preview; paid run approved by the user 2026-09-28).
**Amendment before any measurement (2026-09-28):** the Gemini project hit its monthly spend
cap during the Cryonick pass, so, at the user's request, every corpus is re-extracted from
scratch with `--backend openai` (graphify's default gpt-4.1-mini) — one model for all four;
the partial Gemini extractions were deleted unused. Queries go
through graphify's own `_query_graph_text` (BFS depth 3), read and cut to budget exactly as
in stage 5 (`tools/graphify_adapter.py --graph`). Same ten sets as stage 9; AMG and memlab
rows are reused from stage 9 unchanged.

- **Verdict per set and system pair** by the stage 9 rule (ahead = not significantly worse in
  any cell, significantly better in ≥1).
- **Claim "memlab is the best of the three"** only if memlab is ahead of graphify on a
  majority of sets and graphify is ahead of memlab on none. Otherwise the table is reported
  as is. The stage 9 caveat on budget cells (packs that name files vs. packs of code) applies
  to graphify too: its text answer is cut at budget × 4 characters.

#### Result (run 2026-09-28, `results/*-s10-*.json`, `results/stage10-summary.txt`)

Extraction: gpt-4.1-mini, three passes per corpus (graphify retries only omitted files);
graphify's own cost estimate ≈ $4.2 in total (HealBot and WhaleCast and part of aifc360 on one
OpenAI key, the rest on the key from the project `.env`, same model). Files@5 / 10 / 20:

| set | n | memlab | AMG | graphify 0.9.70 | memlab vs graphify | AMG vs graphify |
|---|---|---|---|---|---|---|
| aifc360 natural | 48 | 0.57 / 0.69 / 0.75 | 0.45 / 0.64 / 0.74 | 0.37 / 0.42 / 0.45 | memlab | AMG |
| aifc360 git | 80 | 0.59 / 0.76 / 0.88 | 0.28 / 0.43 / 0.60 | 0.29 / 0.38 / 0.45 | memlab | AMG |
| cryonick git | 80 | 0.57 / 0.72 / 0.82 | 0.09 / 0.21 / 0.30 | 0.08 / 0.13 / 0.19 | memlab | graphify |
| healbot git | 25 | 0.30 / 0.57 / 0.81 | 0.01 / 0.01 / 0.05 | 0.40 / 0.44 / 0.47 | memlab | graphify |
| whalecast git | 22 | 0.56 / 0.77 / 0.91 | 0.62 / 0.76 / 0.84 | 0.41 / 0.51 / 0.68 | memlab | AMG |
| aifc360 decisions | 26 | 1.00 / 1.00 / 1.00 | 0.04 / 0.04 / 0.04 | 0.04 / 0.04 / 0.04 | memlab | tie |
| aifc360 git-dec. | 20 | 0.93 / 0.97 / 1.00 | 0.00 / 0.00 / 0.00 | 0.05 / 0.05 / 0.05 | memlab | AMG |
| aifc360 hard | 36 | 0.97 / 1.00 / 1.00 | 0.03 / 0.03 / 0.03 | 0.00 / 0.00 / 0.00 | memlab | AMG |
| cryonick git-dec. | 80 | 0.93 / 0.95 / 0.96 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | memlab | tie |
| cryonick hard | 30 | 0.97 / 0.97 / 0.97 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | memlab | tie |

**Claim met:** memlab is ahead of graphify on all ten sets and graphify is ahead on none.
graphify's strongest case is HealBot's head (files@5 0.40 vs 0.30, 4k 0.67 vs 0.59 — both
single cells, memlab ahead on the rest). AMG vs graphify: AMG 5, graphify 2, ties 3.

Why graphify misses decisions, checked: its semantic pass returned nothing, on all three passes,
for every file under `.claude/` and nearly every file under `.ai-factory/` (aifc360: 0 of the
decision notes, 1 postmortem; cryonick: 0 of 79 `.ai-factory` gold docs in the git set) — not a
language effect (the omitted docs are *less* often Russian, 29% vs 39%); plausibly path handling
of dot-directories, not verified. Where the documents are present (cryonick hard set: 12 of 30
gold documents have nodes) it still returns none of them in the top 20.

