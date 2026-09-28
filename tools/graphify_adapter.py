"""Run graphify (Apache-2.0, github.com/Graphify-Labs/graphify) as an external baseline.

Builds graphify's AST graph over exactly the memlab corpus files, asks each case's query
through graphify's own query path (serve._query_graph_text, BFS depth 3), and records the
repository files its answer surfaces. The answer text is cut to the same token budget as
memlab methods (chars / 4): graphify may return more than the budget it was asked for.

Run inside an environment with graphifyy installed:
  <venv>/python tools/graphify_adapter.py --config projects/aifc360.toml \
      --cases datasets/aifc360/git-cases.json --out results/ext/graphify-aifc360-git-cases.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import tempfile
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from memlab import corpus  # noqa: E402  (stdlib-only module)

SRC = re.compile(r"src=([^\s\]]+)")
BUDGETS = (4000, 16000, 38000)
ORDER_BUDGET = 200_000


def files_in(text: str, paths: set[str]) -> list[str]:
    out: list[str] = []
    for p in SRC.findall(text):
        p = p.replace("\\", "/")
        if p in paths and p not in out:
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--graph", type=Path, help="prebuilt graph.json (e.g. `graphify extract` with a "
                    "semantic doc pass over tools/mirror_corpus.py output) instead of an AST-only build")
    args = ap.parse_args()

    cfg = tomllib.loads((REPO / args.config).read_text(encoding="utf-8"))
    root = Path(cfg["root"])
    c = cfg.get("corpus", {})
    files = corpus.list_files(root, c.get("include", []), c.get("exclude", []),
                              c.get("max_bytes", 400_000), c.get("keep_always", []))
    paths = set(files)

    from graphify.build import build_from_json
    from graphify.extract import extract
    from graphify.serve import _query_graph_text
    import importlib.metadata as md

    from graphify.serve import _load_graph
    cache = REPO / ".cache" / f"graphify-{md.version('graphifyy')}-{args.config.stem}-{len(files)}.pkl"
    if args.graph:
        G = _load_graph(str(args.graph))
        print(f"graphify graph {args.graph}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    elif cache.exists():   # local cache written by this script only (gitignored .cache/), never shared input
        G = pickle.loads(cache.read_bytes())
    else:
        t = time.time()
        os.chdir(tempfile.mkdtemp(prefix="gfy-"))        # graphify writes graphify-out/ into the cwd
        ex = extract([root / f for f in files], root=root, parallel=False)
        G = build_from_json(ex)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(G))
        print(f"graphify graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {time.time() - t:.0f}s")

    cases = json.loads((REPO / args.cases).read_text(encoding="utf-8"))
    out = {}
    for k in cases:
        full = _query_graph_text(G, k["query"], token_budget=ORDER_BUDGET)
        res = {"order": files_in(full, paths)[:20], "files": {}}
        for b in BUDGETS:
            text = _query_graph_text(G, k["query"], token_budget=b)
            res["files"][str(b)] = files_in(text[: b * 4], paths)
        out[k["id"]] = res
    (REPO / args.out).parent.mkdir(parents=True, exist_ok=True)
    (REPO / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(out)} cases -> {args.out}")


if __name__ == "__main__":
    main()
