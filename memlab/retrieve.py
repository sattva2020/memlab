"""Baseline retrievers: BM25, dense embeddings, and their reciprocal-rank fusion.

Each retriever maps a query to a full ranking of chunk indices. Budgeting (how
many of them fit into the context) is the evaluator's job, not the retriever's.
"""
from __future__ import annotations

import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .corpus import Chunk, fingerprint

WORD = re.compile(r"[^\W_]+", re.UNICODE)
CAMEL = re.compile(r"(?<=[a-zа-яё0-9])(?=[A-ZА-ЯЁ])")


STOPWORDS = frozenset("""
a an and are as at be but by can do does for from how i in is it its not of on or so that the
this to was what when where which who why will with you your
а без бы в во вот все всё где да для до если есть еще ещё же за и из или им их к как ко когда
кто ли либо мне мы на над не нет ни но о об он она они от по под при про с со так там то тоже
у уже чем что чтобы это эта эти этот я
""".split())


CYRILLIC = re.compile(r"[а-яё]")
_RU = None


def _stem_ru(tok: str) -> str:
    """Snowball Russian stem for Cyrillic tokens (H12); Latin tokens — identifiers — stay as is."""
    global _RU
    if not CYRILLIC.search(tok):
        return tok
    if _RU is None:
        import snowballstemmer
        _RU = snowballstemmer.stemmer("russian")
    return _RU.stemWord(tok)


def query_tokens(text: str, drop_stopwords: bool = False, stem: bool = False) -> list[str]:
    """Query-side tokenization; stopword removal is a query-only step (documents keep them).
    Stopwords are matched before stemming, so the list stays in surface forms."""
    toks = tokenize(text)
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [_stem_ru(t) for t in toks] if stem else toks


def tokenize(text: str, stem: bool = False) -> list[str]:
    """Words, with camelCase/snake_case identifiers split into parts (parts + whole)."""
    out = []
    for w in WORD.findall(text):
        parts = CAMEL.sub(" ", w).lower().split()
        out += parts
        if len(parts) > 1:
            out.append(w.lower())
    return [_stem_ru(t) for t in out] if stem else out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.post: dict[str, list[int]] = {}
        for i, d in enumerate(self.tf):
            for t in d:
                self.post.setdefault(t, []).append(i)

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf))
        norm = self.k1 * (1 - self.b + self.b * self.len / (self.avg or 1))
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.post[t]:
                f = self.tf[i][t]
                s[i] += idf * f * (self.k1 + 1) / (f + norm[i])
        return s


def save_store(path: Path, new: dict[str, np.ndarray]) -> None:
    """Add vectors to the shared per-model store. Servers of every project build at the same time:
    re-read the file just before writing and replace it atomically, so one build does not drop
    the vectors another one wrote since it loaded the store (that cost Cryonick a full re-embed)."""
    store: dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path) as z:
            store = dict(zip(z["keys"].tolist(), z["vecs"]))
    store.update(new)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, keys=np.array(list(store)), vecs=np.stack(list(store.values())))
    # ponytail: no lock, two writers in the same instant can still lose one batch
    for attempt in range(10):   # Windows refuses the replace while another server is reading the store
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
    tmp.unlink(missing_ok=True)  # the store is only a cache: the vectors stay in memory, the build goes on


class Dense:
    """Sentence-transformer embeddings, cached on disk per corpus fingerprint."""

    def __init__(self, chunks: list[Chunk], model: str, cache_dir: Path,
                 query_prefix: str = "", doc_prefix: str = "", max_len: int | None = None,
                 encoder=None):
        """query_prefix/doc_prefix: instruction prefixes some models need (e5: "query: " /
        "passage: "; Qwen3: an instruction on the query only); the doc prefix and max_len are
        part of the cache key. Uses CUDA in fp16 when available. `encoder`: anything with a
        SentenceTransformer-style `.encode` (the shared embedder client); then no model is loaded here."""
        import hashlib
        if encoder is not None:
            self.model, cuda = encoder, True
        else:
            import torch
            from sentence_transformers import SentenceTransformer
            cuda = torch.cuda.is_available()
            self.model = SentenceTransformer(model, device="cuda" if cuda else "cpu",
                                             model_kwargs={"torch_dtype": torch.float16} if cuda else {})
            if max_len:
                self.model.max_seq_length = max_len
        self.query_prefix = query_prefix
        self.reembedded = 0
        tag = f"-{hashlib.sha1(doc_prefix.encode()).hexdigest()[:6]}" if doc_prefix else ""
        tag += f"-L{max_len}" if max_len else ""
        if not chunks:                           # e.g. a repository with no decision documents yet
            self.emb = np.zeros((0, 0), np.float32)
            return
        cache = cache_dir / f"emb-{fingerprint(chunks)}-{model.replace('/', '_')}{tag}.npy"
        if cache.exists():
            self.emb = np.load(cache)
            return
        # Per-chunk store keyed by text hash: after an edit only the changed chunks are re-embedded.
        texts = [f"{doc_prefix}{c.path} {c.symbol}\n{c.text}" for c in chunks]
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        store_path = cache_dir / f"vecstore-{model.replace('/', '_')}{tag}.npz"
        store: dict[str, np.ndarray] = {}
        if store_path.exists():
            with np.load(store_path) as z:
                store = dict(zip(z["keys"].tolist(), z["vecs"]))
        todo = [i for i, k in enumerate(keys) if k not in store]
        if todo:
            vecs = self.model.encode([texts[i] for i in todo], batch_size=16 if cuda else 64,
                                     normalize_embeddings=True, show_progress_bar=len(todo) > 500)
            new = {keys[i]: v.astype(np.float32) for i, v in zip(todo, vecs)}
            store.update(new)
            save_store(store_path, new)
        self.emb = np.stack([store[k] for k in keys]).astype(np.float32)
        self.reembedded = len(todo)
        np.save(cache, self.emb)

    def scores(self, query: str) -> np.ndarray:
        if not len(self.emb):
            return np.zeros(0, np.float32)
        q = self.model.encode([self.query_prefix + query], normalize_embeddings=True)[0]
        return self.emb @ q


def rank(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, kind="stable")


def rrf_scores(*rankings: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion scores (Cormack et al., 2009)."""
    fused = np.zeros(len(rankings[0]))
    for r in rankings:
        pos = np.empty(len(r)); pos[r] = np.arange(len(r))
        fused += 1.0 / (k + 1 + pos)
    return fused


def rrf(*rankings: np.ndarray, k: int = 60) -> np.ndarray:
    return rank(rrf_scores(*rankings, k=k))
