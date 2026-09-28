"""H9: a "similar past commits" channel (after arXiv:2502.07067, BugLocator-style).

For a query, BM25 over the subjects of commits made strictly BEFORE the case's own commit
(time masking: the answer's commit and anything later are invisible), then each file
changed by the top commits gets score / (files in that commit). Files are turned into a
chunk ranking (file order by score, chunks inside a file by the seed order).
"""
from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from .corpus import Chunk
from .retrieve import BM25, tokenize


class CommitChannel:
    def __init__(self, root: Path, chunks: list[Chunk], top_commits: int = 20):
        log = subprocess.run(["git", "-C", str(root), "log", "--no-merges",
                              "--format=%x00%H%x09%ct%x09%s", "--name-only"],
                             capture_output=True, text=True, encoding="utf-8", check=True).stdout
        self.commits = []                                  # (sha, ts, tokens, files)
        for block in log.split("\x00")[1:]:
            head, *files = [x for x in block.splitlines() if x.strip()]
            sha, ts, subject = head.split("\t", 2)
            self.commits.append((sha, int(ts), tokenize(subject), files))
        self.ts_by_prefix = {sha[:8]: ts for sha, ts, _, _ in self.commits}
        self.by_file: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            self.by_file[c.path].append(i)
        self.n = len(chunks)
        self.top = top_commits

    def file_scores(self, case_id: str, query: str) -> dict[str, float]:
        sha8 = case_id.split("-", 1)[-1][:8]
        t = self.ts_by_prefix.get(sha8)
        if t is None:
            return {}
        past = [c for c in self.commits if c[1] < t]       # strictly earlier commits only
        if not past:
            return {}
        s = BM25([c[2] for c in past]).scores(tokenize(query))
        scores: dict[str, float] = defaultdict(float)
        for k in np.argsort(-s)[: self.top]:
            if s[k] <= 0:
                break
            files = [f for f in past[k][3] if f in self.by_file]
            for f in files:
                scores[f] += s[k] / len(past[k][3])
        return scores

    def ranking(self, case_id: str, query: str, seed_rank: np.ndarray) -> np.ndarray:
        fs = self.file_scores(case_id, query)
        pos = np.empty(self.n, dtype=np.int64)
        pos[seed_rank] = np.arange(self.n)
        out: list[int] = []
        for f in sorted(fs, key=fs.get, reverse=True):
            out += sorted(self.by_file[f], key=lambda i: pos[i])
        seen = set(out)
        out += [int(i) for i in seed_rank if int(i) not in seen]
        return np.array(out, dtype=np.int64)
