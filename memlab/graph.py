"""Code/document graph over chunks, and two graph-aware rankers built on a flat seed.

Edges (all derived from the sources, no LLM):
  siblings — chunks of the same file;
  imports  — file -> file from import/export/part directives (Dart, TS/JS) and
             relative markdown links / backticked repo paths in prose;
  refs     — chunk -> chunk that defines a symbol the chunk mentions by name.

Rankers:
  expand — H1, lexically anchored local expansion (after LARGER, arXiv:2605.16352):
           walk the seed ranking; after each of the first A anchors insert up to m of
           its graph neighbours, best seed score first, admitting only neighbours the
           seed itself ranks within its top R (the confidence filter).
  ppr    — global Personalized PageRank over the same graph (the H1 comparator),
           teleport vector = seed scores of the top-K seed chunks.
"""
from __future__ import annotations

import math
import posixpath
import re
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse

from .corpus import Chunk

IMPORT_DART = re.compile(r"""^\s*(?:import|export|part)\s+['"]([^'"]+)['"]""", re.M)
IMPORT_TS = re.compile(r"""(?:from\s+|import\s*\(\s*|require\(\s*|^import\s+)['"]([^'"]+)['"]""", re.M)
MD_LINK = re.compile(r"\]\(([^)#\s]+)")
REPO_PATH = re.compile(r"`([\w.-]+/[\w./-]+\.\w+)")
IDENT = re.compile(r"\b[A-Za-z_]\w{3,}\b")
CAMEL_INNER = re.compile(r"[a-z0-9][A-Z]")


def _resolve(src: str, target: str, files: set[str], dart_pkg: str, dart_root: str,
             aliases: dict[str, str]) -> str | None:
    alias = next((a for a in aliases if target.startswith(a)), None)
    if dart_pkg and target.startswith(f"package:{dart_pkg}/"):
        cand = posixpath.join(dart_root, target[len(f"package:{dart_pkg}/"):])
    elif alias:
        cand = aliases[alias] + target[len(alias):]
    elif target.startswith(("package:", "dart:", "http:", "https:", "mailto:", "@")):
        return None                                  # external package or unmapped scope
    else:                                            # relative path; a bare package name simply won't resolve
        cand = posixpath.normpath(posixpath.join(posixpath.dirname(src), target))
    for c in (cand, cand + ".ts", cand + ".tsx", cand + ".js", cand + ".jsx",
              cand + "/index.ts", cand + "/index.tsx", cand + "/index.js"):
        if c in files:
            return c
    return None


class Graph:
    def __init__(self, chunks: list[Chunk], dart_pkg: str = "", dart_root: str = "flutter/lib",
                 max_ref_df: int = 200, max_defs: int = 3, aliases: dict[str, str] | None = None,
                 weighting: str = "cap"):
        aliases = aliases or {}
        self.chunks = chunks
        self.by_file: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            self.by_file[c.path].append(i)
        files = set(self.by_file)

        # file -> file
        self.file_edges: dict[str, set[str]] = defaultdict(set)
        texts: dict[str, list[str]] = defaultdict(list)
        for c in chunks:
            texts[c.path].append(c.text)
        for path, parts in texts.items():
            text = "".join(parts)
            targets = []
            if path.endswith(".dart"):
                targets = IMPORT_DART.findall(text)
            elif path.endswith((".ts", ".tsx", ".js", ".jsx", ".astro")):
                targets = IMPORT_TS.findall(text)
            elif path.endswith((".md", ".mdx")):
                targets = MD_LINK.findall(text) + REPO_PATH.findall(text)
            for t in targets:
                r = _resolve(path, t, files, dart_pkg, dart_root, aliases) or (t if t in files else None)
                if r and r != path:
                    self.file_edges[path].add(r)
                    self.file_edges[r].add(path)

        # chunk -> defining chunk. "cap" (a priori): names defined in <= max_defs places and
        # mentioned in <= max_ref_df chunks, unit weight. "aider" (H10, aider/repomap.py):
        # no caps, weight mul*sqrt(mentions) with mul x10 for long snake/camel names,
        # x0.1 for _private names, x0.1 for names defined in more than 5 files.
        all_defs: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            if c.symbol:
                all_defs[c.symbol.split("~")[0]].append(i)
        self.defs = {k: v for k, v in all_defs.items() if len(k) >= 4}
        if weighting == "cap":
            defs = {k: v for k, v in self.defs.items() if len(v) <= max_defs}
        else:
            defs = self.defs
        mentions: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, c in enumerate(chunks):
            counts = Counter(IDENT.findall(c.text))
            for name in counts.keys() & defs.keys():
                mentions[name].append((i, counts[name]))
        self.refs: dict[int, set[int]] = defaultdict(set)
        self.ref_w: dict[tuple[int, int], float] = {}
        for name, users in mentions.items():
            if weighting == "cap" and len(users) > max_ref_df:
                continue
            mul = 1.0
            if weighting == "aider":
                if len(name) >= 8 and ("_" in name.strip("_") or CAMEL_INNER.search(name)):
                    mul *= 10
                if name.startswith("_"):
                    mul *= 0.1
                if len({chunks[d].path for d in defs[name]}) > 5:
                    mul *= 0.1
            for u, n in users:
                for d in defs[name]:
                    if u != d and chunks[u].path != chunks[d].path:
                        self.refs[u].add(d)
                        self.refs[d].add(u)
                        key = (min(u, d), max(u, d))
                        self.ref_w[key] = max(self.ref_w.get(key, 0.0), mul * math.sqrt(n))

    def neighbours(self, i: int, kinds: set[str]) -> set[int]:
        path = self.chunks[i].path
        out: set[int] = set()
        if "siblings" in kinds:
            out.update(self.by_file[path])
        if "imports" in kinds:
            for f in self.file_edges.get(path, ()):
                out.update(self.by_file[f])
        if "refs" in kinds:
            out.update(self.refs.get(i, ()))
        out.discard(i)
        return out

    def stats(self) -> str:
        fe = sum(len(v) for v in self.file_edges.values()) // 2
        re_ = sum(len(v) for v in self.refs.values()) // 2
        return f"file edges {fe}, ref edges {re_}, files with imports {len(self.file_edges)}"

    # ------------------------------------------------------------------ rankers
    def expand(self, seed_rank: np.ndarray, kinds: set[str], anchors: int = 10,
               per_anchor: int = 3, confidence_top: int = 2000) -> np.ndarray:
        pos = np.empty(len(seed_rank), dtype=np.int64)
        pos[seed_rank] = np.arange(len(seed_rank))
        out, seen = [], set()
        for a in seed_rank[:anchors]:
            if a not in seen:
                out.append(a); seen.add(a)
            nb = [n for n in self.neighbours(int(a), kinds) if n not in seen and pos[n] < confidence_top]
            for n in sorted(nb, key=lambda n: pos[n])[:per_anchor]:
                out.append(n); seen.add(n)
        out += [i for i in seed_rank if i not in seen]
        return np.array(out)

    def ppr(self, seed_scores: np.ndarray, kinds: set[str], seeds: int = 50,
            damping: float = 0.85, iters: int = 30, beta: float = 0.0,
            transit_block: bool = False, seed_ids: list[int] | None = None) -> np.ndarray:
        """beta > 0 divides the stationary mass by degree**beta (degree-normalized PPR),
        the standard correction for the bias of random walks toward high-degree nodes.

        transit_block: hubs (degree >= max(50, p99)) still receive mass but do not pass
        it on; their outflow returns to the teleport vector (graphify's BFS rule, as PPR).
        seed_ids: explicit teleport support instead of the top-`seeds` by score.
        """
        n = len(self.chunks)
        if not hasattr(self, "_adj") or self._adj_kinds != kinds:
            # Linear-size adjacency: a file is represented by its first chunk (star over
            # siblings), imports join file heads, refs join chunks directly.
            rows, cols = [], []
            def link(i, j):
                rows.extend((i, j)); cols.extend((j, i))
            head = {p: ids[0] for p, ids in self.by_file.items()}
            for p, ids in self.by_file.items():
                if "siblings" in kinds:
                    for i in ids[1:]:
                        link(head[p], i)
                if "imports" in kinds:
                    for f in self.file_edges.get(p, ()):
                        if p < f:
                            link(head[p], head[f])
            vals = [1.0] * len(rows)
            if "refs" in kinds:
                for (i, j), w in self.ref_w.items():
                    rows.extend((i, j)); cols.extend((j, i)); vals.extend((w, w))
            a = sparse.csr_matrix((np.array(vals), (rows, cols)), shape=(n, n))   # duplicates summed
            cnt = np.asarray((a > 0).sum(axis=1)).ravel().astype(float)
            deg = np.asarray(a.sum(axis=1)).ravel()
            deg[deg == 0] = 1
            cnt[cnt == 0] = 1
            self._deg, self._cnt = deg, cnt
            self._adj, self._adj_kinds = sparse.diags(1 / deg) @ a, kinds   # row-stochastic
        top = np.array(seed_ids) if seed_ids is not None else np.argsort(-seed_scores)[:seeds]
        p = np.zeros(n); p[top] = np.maximum(seed_scores[top], 1e-9); p /= p.sum()
        pi = p.copy()
        adj = self._adj
        hub = np.zeros(n, dtype=bool)
        if transit_block:
            hub = self._cnt >= max(50, np.percentile(self._cnt, 99))
            adj = sparse.diags((~hub).astype(float)) @ adj          # hubs emit nothing
        mt = adj.T.tocsr()
        for _ in range(iters):
            pi = (1 - damping) * p + damping * (mt @ pi + pi[hub].sum() * p)
        if beta:
            pi = pi / self._cnt ** beta
        # nodes the walk never reaches keep the seed order instead of index order
        s = seed_scores / (seed_scores.max() or 1)
        return np.argsort(-(pi + 1e-12 * s), kind="stable")
