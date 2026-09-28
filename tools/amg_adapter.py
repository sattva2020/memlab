"""Run AMG (the associative memory graph skill) as an external baseline.

Calls AMG's own `retrieve()` from the installed skill, unmodified, with pack and
co-activation writes OFF. Store loading, the BM25 index, the adjacency and the node
embeddings do not change between queries, so they are memoized here to load once — the
per-query computation (seed, PPR, pack assembly) is AMG's own. Files are read from the pack
in order of appearance: a node id maps to its `source_path`, a note without one to its own
file under `.claude/amg/`. Budget cells cut the pack text at budget x 4 characters, as for
graphify. Run it on a COPY of the store (the embedding cache may be refreshed).

  <ml venv>/python tools/amg_adapter.py --config projects/aifc360.toml --store <copy> \
      --cases datasets/aifc360/cases.json --out results/ext/amg-aifc360-cases.json
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from memlab import corpus  # noqa: E402

AMG = Path.home() / ".claude/skills/amg-retrieve/scripts"
sys.path.insert(0, str(AMG))
import embed  # noqa: E402
import retrieve as amg  # noqa: E402

BUDGETS = (4000, 16000, 38000)
NODE_LINE = re.compile(r"^(?:- |### )(\S+?)(?:\s+—|\s+⟨|\s*$)", re.M)

# Load-once memoization of the query-independent parts (pure functions of the store).
amg.load_config = functools.lru_cache(None)(amg.load_config)
amg.load_nodes = functools.lru_cache(None)(amg.load_nodes)
_bm25_cls, _bm25 = amg.BM25, {}
amg.BM25 = lambda nodes, *a, **k: _bm25.setdefault(id(nodes), _bm25_cls(nodes, *a, **k))
_build_adj, _adj = amg.build_adjacency, {}
amg.build_adjacency = lambda nodes, cfg: _adj.setdefault(id(nodes), _build_adj(nodes, cfg))
embed.get_embedder = functools.lru_cache(None)(lambda cfg, _f=embed.get_embedder: _f(cfg))
_node_emb, _vecs = embed.node_embeddings, {}
embed.node_embeddings = lambda e, nodes, path: _vecs.setdefault(str(path), _node_emb(e, nodes, path))


class _Frozen(dict):          # hashable config for the lru_cache on get_embedder
    def __hash__(self):
        return id(self)


def node_path(node: dict) -> str:
    return node.get("source_path") or f".claude/amg/{node['_path']}"


def files_in(pack: str, nodes: dict, paths: set[str]) -> list[str]:
    out: list[str] = []
    for nid in NODE_LINE.findall(pack):
        n = nodes.get(nid)
        if n is None:
            continue
        p = node_path(n).replace("\\", "/")
        if p in paths and p not in out:
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cfg_m = tomllib.loads((REPO / a.config).read_text(encoding="utf-8"))
    c = cfg_m.get("corpus", {})
    paths = set(corpus.list_files(Path(cfg_m["root"]), c.get("include", []), c.get("exclude", []),
                                  c.get("max_bytes", 400_000), c.get("keep_always", [])))
    cfg = _Frozen(amg.load_config(a.store))
    nodes = amg.load_nodes(a.store)
    cases = json.loads((REPO / a.cases).read_text(encoding="utf-8"))
    out, t0 = {}, time.time()
    for i, k in enumerate(cases):
        res = amg.retrieve(a.store, k["query"], config=cfg, write_pack=False, log_coactivation=False)
        pack = res["pack"]
        out[k["id"]] = {"order": files_in(pack, nodes, paths)[:20],
                        "files": {str(b): files_in(pack[: b * 4], nodes, paths) for b in BUDGETS}}
        if i == 0:
            print(f"first query {time.time() - t0:.0f}s; {len(nodes)} nodes", flush=True)
    (REPO / a.out).parent.mkdir(parents=True, exist_ok=True)
    (REPO / a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(out)} cases -> {a.out}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
