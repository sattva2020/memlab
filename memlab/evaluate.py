"""Matched-budget evaluation of retrievers against labeled cases.

Every method gets the same token budget: ranked chunks are taken in order and
kept while they fit. Recall is reported at two granularities (file and symbol),
with paired bootstrap confidence intervals of the difference to a reference.
"""
from __future__ import annotations

import numpy as np

from .corpus import Chunk


def select(ranking: np.ndarray, chunks: list[Chunk], budget: int) -> list[Chunk]:
    out, used = [], 0
    for i in ranking:
        c = chunks[i]
        if used + c.tokens > budget:
            continue
        out.append(c); used += c.tokens
        if budget - used < 50:
            break
    return out


def select_channel(side: np.ndarray, main: np.ndarray, chunks: list[Chunk], budget: int,
                   share: float) -> list[Chunk]:
    """Reserve `share` of the budget for a side channel (e.g. decisions), fill the rest
    from the main ranking; the reserve's unused tokens flow back to the main channel."""
    first = select(side, chunks, int(budget * share))
    taken = {c.id for c in first}
    used = sum(c.tokens for c in first)
    rest = [i for i in main if chunks[i].id not in taken]
    return first + select(np.array(rest), chunks, budget - used)


def hit(gold: dict, picked: list[Chunk], level: str) -> bool:
    path, sym = gold["path"], gold.get("symbol") or ""
    for c in picked:
        if c.path != path:
            continue
        if level == "file" or not sym:
            return True
        if c.symbol == sym or c.symbol.startswith(sym + "~"):
            return True
    return False


def recall(case: dict, picked: list[Chunk], level: str) -> float:
    g = case["gold"]
    return sum(hit(x, picked, level) for x in g) / max(len(g), 1)


def interleave(ranking: np.ndarray, is_code: np.ndarray, n_code: int, n_docs: int) -> np.ndarray:
    """Merge a ranking's code and non-code chunks in a fixed ratio, each side keeping its own
    order: n_code code chunks, then n_docs doc chunks, repeat; leftovers appended."""
    code = [i for i in ranking if is_code[i]]
    docs = [i for i in ranking if not is_code[i]]
    out, ci, di = [], 0, 0
    while ci < len(code) or di < len(docs):
        out += code[ci:ci + n_code]; ci += n_code
        out += docs[di:di + n_docs]; di += n_docs
    return np.array(out, dtype=np.int64)


def files_in_order(ranking, chunks: list[Chunk], k: int) -> list[str]:
    """First k distinct files in rank order (budget-free localization, like Acc@k)."""
    out: list[str] = []
    for i in ranking:
        p = chunks[i].path
        if p not in out:
            out.append(p)
            if len(out) == k:
                break
    return out


def recall_files(case: dict, files: list[str]) -> float:
    s = set(files)
    return sum(g["path"] in s for g in case["gold"]) / max(len(case["gold"]), 1)


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) over cases with a 95% paired-bootstrap interval."""
    d = a - b
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def alternate(base: list[str], side: list[str]) -> list[str]:
    """Base and side lists interleaved, base first, duplicates dropped (H14 pointer next to a tool)."""
    out: list[str] = []
    for i in range(max(len(base), len(side))):
        for lst in (base, side):
            if i < len(lst) and lst[i] not in out:
                out.append(lst[i])
    return out
