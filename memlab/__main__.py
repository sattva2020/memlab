"""memlab CLI.

  python -m memlab eval --config projects/aifc360.toml [--budgets 8000,32000] [--out results.json]
  python -m memlab stats --config projects/aifc360.toml
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import time
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np

from . import commits, corpus, evaluate, gitcases, graph, retrieve

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def load(cfg_path: Path):
    cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    root = Path(cfg["root"])
    c = cfg.get("corpus", {})
    chunks = corpus.build(root, c.get("include", []), c.get("exclude", []), c.get("max_bytes", 400_000),
                          c.get("keep_always", []))
    return cfg, root, chunks


def cmd_stats(args):
    cfg, root, chunks = load(args.config)
    by = Counter(c.path.split("/")[0] for c in chunks)
    print(f"{len(chunks)} chunks, {len({c.path for c in chunks})} files, "
          f"{sum(c.tokens for c in chunks):,} tokens, fingerprint {corpus.fingerprint(chunks)}")
    for k, v in by.most_common(12):
        print(f"  {v:6} {k}")


def cmd_eval(args):
    cfg, root, chunks = load(args.config)
    cases = json.loads(Path(args.cases or cfg["cases"]).read_text(encoding="utf-8"))
    if args.lang:                        # H15: Russian = the query has a Cyrillic letter
        cases = [k for k in cases if bool(re.search("[а-яё]", k["query"].lower())) == (args.lang == "ru")]
    paths = {c.path for c in chunks}
    missing = [(k["id"], g["path"]) for k in cases for g in k["gold"] if g["path"] not in paths]
    if missing:
        sys.exit(f"gold paths absent from corpus: {missing[:5]} ... ({len(missing)})")

    bm25 = retrieve.BM25([retrieve.tokenize(f"{c.path} {c.symbol} {c.text}", args.stem) for c in chunks])
    model = args.model or cfg.get("model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    e5, qwen = "e5" in model.lower(), "qwen3-embedding" in model.lower()
    qprefix = ("query: " if e5 else
               "Instruct: Given a description of a code change or a question about a software project, "
               "retrieve the relevant source files and documents\nQuery: " if qwen else "")
    dense = retrieve.Dense(chunks, model, Path(cfg.get("cache", ".cache")), query_prefix=qprefix,
                           doc_prefix="passage: " if e5 else "", max_len=512 if qwen else None)
    methods = args.methods.split(",")
    gc = cfg.get("graph", {})
    graphs: dict[tuple, graph.Graph] = {}

    def opts(m: str) -> dict:
        # trailing "k=v&k=v" segment of a method name, e.g. ppr:refs:df=1e9&beta=0.5
        last = m.split(":")[-1]
        return dict(kv.split("=") for kv in last.split("&")) if "=" in last else {}

    def graph_for(o: dict) -> graph.Graph:
        key = (float(o.get("df", 200)), float(o.get("defs", 3)), o.get("w", "cap"))
        if key not in graphs:
            graphs[key] = graph.Graph(chunks, gc.get("dart_package", ""), gc.get("dart_root", "flutter/lib"),
                                      max_ref_df=key[0], max_defs=key[1], aliases=gc.get("aliases", {}),
                                      weighting=key[2])
            print(f"graph df<={key[0]:g} defs<={key[1]:g} w={key[2]}: {graphs[key].stats()}")
        return graphs[key]

    is_code = np.array([Path(c.path).suffix.lower() in corpus.CODE_EXT for c in chunks])
    decision_globs = cfg.get("channels", {}).get("decisions", [])
    is_decision = np.array([any(fnmatch.fnmatch(c.path, gl) for gl in decision_globs) for c in chunks])

    def gfy_seeds(qt: list[str], bs: np.ndarray, rd: np.ndarray, rh: np.ndarray) -> list[int]:
        """graphify-style seeds (serve.py _pick_seeds, Apache-2.0): lexical gap 0.2 of the top,
        one seed per symbol label, at least one seed per matched query term. Dense top-5 kept
        so the hybrid's semantic seeds survive the lexical gap."""
        mx, dense5 = bs.max(), set(rd[:5].tolist())
        out, labels = [], set()
        for i in rh[:50]:
            i = int(i)
            if not (bs[i] >= 0.2 * mx or i in dense5):
                continue
            lab = chunks[i].symbol.split("~")[0]
            if lab and lab in labels:
                continue
            labels.add(lab); out.append(i)
        for tok in dict.fromkeys(qt):
            s = bm25.scores([tok])
            if s.max() > 0 and int(s.argmax()) not in out:
                out.append(int(s.argmax()))
        return out

    ext = {}
    for spec in args.external or []:
        name, path = spec.split("=", 1)
        ext[f"ext:{name}"] = json.loads(Path(path).read_text(encoding="utf-8"))
        methods.append(f"ext:{name}")

    reranker = None
    if any(m.startswith("rerank:") for m in methods):
        import torch
        from sentence_transformers import CrossEncoder
        cuda = torch.cuda.is_available()
        reranker = CrossEncoder(args.reranker, max_length=512, device="cuda" if cuda else "cpu",
                                model_kwargs={"torch_dtype": torch.float16} if cuda else {})
    rerank_seconds = []
    need_commits = any("commits=1" in m or m == "hybrid+c" or m.startswith("ptr:") for m in methods)
    code_paths = {c.path for c in chunks if Path(c.path).suffix.lower() in corpus.CODE_EXT}
    cch = commits.CommitChannel(root, chunks) if need_commits else None
    scope = re.compile(r"^[\w./-]+:\s+")

    rankings, gate_on = {}, {}
    for k in cases:
        q = scope.sub("", k["query"]) if args.strip_scope else k["query"]
        qt = retrieve.query_tokens(q, args.stopwords, args.stem)
        bs = bm25.scores(qt)
        rb = retrieve.rank(bs)
        rd = retrieve.rank(dense.scores(q))
        seed = retrieve.rrf_scores(rb, rd)
        r = {"bm25": rb, "dense": rd, "hybrid": retrieve.rank(seed)}
        seed_c = None
        if cch:                          # H9: fuse the past-commits channel into the seed
            rc = cch.ranking(k["id"], q, r["hybrid"])
            seed_c = retrieve.rrf_scores(rb, rd, rc)
            r["hybrid+c"] = retrieve.rank(seed_c)
        for m in methods:
            parts, o = m.split(":"), opts(m)
            s_m = seed_c if o.get("commits") == "1" else seed
            if parts[0] == "expand":     # expand:<kinds a+b>:<anchors>:<per_anchor>[:k=v&...]
                r[m] = graph_for(o).expand(r["hybrid"], set(parts[1].split("+")), int(parts[2]), int(parts[3]))
            elif parts[0] == "ppr":      # ppr:<kinds>[:df=&defs=&beta=&transit=1&seeds=gfy&w=aider&qid=1&commits=1]
                g = graph_for(o)
                rh = retrieve.rank(s_m)
                sids = gfy_seeds(qt, bs, rd, rh) if o.get("seeds") == "gfy" else None
                if o.get("qid") == "1":  # Aider: personalize toward chunks defining identifiers named in the query
                    named = [i for w in dict.fromkeys(re.findall(r"[A-Za-z_]\w{3,}", q)) for i in g.defs.get(w, [])]
                    base = sids if sids is not None else [int(i) for i in rh[:50]]
                    if named:
                        s_m = s_m.copy()
                        s_m[named] = np.maximum(s_m[named], s_m.max())
                    sids = list(dict.fromkeys(base + named))
                r[m] = g.ppr(s_m, set(parts[1].split("+")), beta=float(o.get("beta", 0)),
                             transit_block=o.get("transit") == "1", seed_ids=sids)
        dec = r["hybrid"][is_decision[r["hybrid"]]]
        for m in methods:
            if m.startswith("channel"):  # channel:<share>:<base method>
                _, share, base = m.split(":", 2)
                r[m] = (dec, r[base], float(share))
            elif m.startswith("mix"):    # mix:<code>:<docs>:<base> — H7, code/docs interleaved in a fixed ratio
                _, nc, nd, base = m.split(":", 3)
                r[m] = evaluate.interleave(r[base], is_code, int(nc), int(nd))
            elif m.startswith("only:"):  # only:code|dec:<base> — H8, a single-type tool
                _, kind, base = m.split(":", 2)
                mask = is_code if kind == "code" else is_decision
                r[m] = r[base][mask[r[base]]]
            elif m.startswith("gate"):   # gate:<k>:<share>:<base> — channel only if the seed's top-k holds a decision
                _, topk, share, base = m.split(":", 3)
                on = bool(is_decision[r["hybrid"][:int(topk)]].any())
                gate_on[m] = gate_on.get(m, 0) + on
                r[m] = (dec, r[base], float(share)) if on else r[base]
        for m in methods:
            if m.startswith("ptr:"):     # ptr:<k>:<cost>:<base> — H14, base tool + a k-path pointer from past commits
                _, kk, cost, base = m.split(":", 3)
                fs = {f: v for f, v in cch.file_scores(k["id"], q).items() if f in code_paths}
                r[m] = ("ptr", sorted(fs, key=fs.get, reverse=True)[:int(kk)], r[base], int(cost))
        for m in methods:
            if m.startswith("rerank:"):  # rerank:<k>:<base> — H15, cross-encoder over the base's top k
                _, kk, base = m.split(":", 2)
                head = r[base][:int(kk)]
                t0 = time.time()
                s = reranker.predict([(q, f"{chunks[i].path} {chunks[i].symbol}\n{chunks[i].text}") for i in head],
                                     batch_size=16, show_progress_bar=False)
                rerank_seconds.append(time.time() - t0)
                r[m] = np.concatenate([head[np.argsort(-np.asarray(s), kind="stable")], r[base][int(kk):]])
        for m, data in ext.items():
            r[m] = data[k["id"]]
        rankings[k["id"]] = r

    def picked(rk, budget):
        if isinstance(rk, dict):     # external system: files it surfaced at this token budget
            return [corpus.Chunk(p, p, "", "") for p in rk["files"][str(budget)]]
        if isinstance(rk, tuple) and rk[0] == "ptr":
            _, files, base, cost = rk
            return evaluate.select(base, chunks, budget - cost) + [corpus.Chunk(f, f, "", "") for f in files]
        if isinstance(rk, tuple):
            side, main, share = rk
            return evaluate.select_channel(side, main, chunks, budget, share)
        return evaluate.select(rk, chunks, budget)

    if rerank_seconds:
        print(f"rerank: median {1000 * float(np.median(rerank_seconds)):.0f} ms per query, {len(cases)} queries")
    for m, on in gate_on.items():
        print(f"{m}: channel on for {on}/{len(cases)} queries")
    def add_oracles(scores):
        # oracle:<m1>|<m2> — per case the better of two single-type tools (H8 upper bound)
        for m in methods:
            if m.startswith("oracle:"):
                a, b = m[len("oracle:"):].split("|")
                if a in scores and b in scores:
                    scores[m] = np.maximum(scores[a], scores[b])

    budgets = [int(b) for b in args.budgets.split(",")]
    report = {"corpus": corpus.fingerprint(chunks), "chunks": len(chunks), "cases": len(cases),
              "case_ids": [k["id"] for k in cases], "results": {}, "per_case": {}}
    for budget in budgets:
        for level in ("file", "symbol"):
            scores = {m: np.array([evaluate.recall(k, picked(rankings[k["id"]][m], budget), level)
                                   for k in cases]) for m in methods if not m.startswith("oracle:")}
            add_oracles(scores)
            key = f"{budget}/{level}"
            report["results"][key] = {m: float(s.mean()) for m, s in scores.items()}
            report["per_case"][key] = {m: s.tolist() for m, s in scores.items()}
            ref = scores[args.ref]
            print(f"budget {budget:>6} {level}")
            for m in scores:
                d = "" if m == args.ref else "  vs {} {:+.4f} [{:+.4f},{:+.4f}]".format(
                    args.ref, *evaluate.bootstrap_diff(scores[m], ref))
                print(f"   {m:40} {scores[m].mean():.2f}{d}")
    # Budget-free localization: first k distinct files (pointer-style systems vs chunk rankers).
    def order(rk):
        if isinstance(rk, dict):
            return rk["order"]
        if isinstance(rk, tuple) and rk[0] == "ptr":
            return evaluate.alternate(evaluate.files_in_order(rk[2], chunks, 20), rk[1])
        if isinstance(rk, tuple):
            return None
        return evaluate.files_in_order(rk, chunks, 20)
    ranked = [m for m in methods if not m.startswith("oracle:") and order(rankings[cases[0]["id"]][m]) is not None]
    for kk in (1, 5, 10, 20):
        scores = {m: np.array([evaluate.recall_files(k, order(rankings[k["id"]][m])[:kk]) for k in cases])
                  for m in ranked}
        add_oracles(scores)
        report["results"][f"files@{kk}"] = {m: float(s.mean()) for m, s in scores.items()}
        report["per_case"][f"files@{kk}"] = {m: s.tolist() for m, s in scores.items()}
        print(f"files@{kk}")
        for m in scores:
            d = "" if m == args.ref else "  vs {} {:+.4f} [{:+.4f},{:+.4f}]".format(
                args.ref, *evaluate.bootstrap_diff(scores[m], scores[args.ref]))
            print(f"   {m:40} {scores[m].mean():.2f}{d}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


def cmd_gitcases(args):
    cfg, root, chunks = load(args.config)
    paths = {c.path for c in chunks}
    if args.decisions:
        cases = gitcases.mine_decisions(root, paths, cfg.get("channels", {}).get("decisions", []), args.n)
    else:
        cases = gitcases.mine(root, paths, args.since, args.n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(cases)} cases, {sum(len(k['gold']) for k in cases)} gold files -> {args.out}")


def _explore_ctx(args):
    from . import config, explore, graph as graph_mod
    root = Path(args.root).resolve()
    cfg = config.load(root, args.config)
    c = cfg.get("corpus", {})
    chunks = corpus.build(root, c.get("include", []), c.get("exclude", []), c.get("max_bytes", 400_000),
                          c.get("keep_always", []))
    gc = cfg.get("graph", {})
    g = graph_mod.Graph(chunks, gc.get("dart_package", ""), gc.get("dart_root", ""), aliases=gc.get("aliases", {}))
    G = explore.file_graph(chunks, g, cfg.get("channels", {}).get("decisions", []))
    return explore, chunks, g, G


def cmd_explore(args):
    explore, chunks, g, G = _explore_ctx(args)
    if args.action == "explain":
        print(explore.explain(G, g, chunks, args.names[0]))
    elif args.action == "path":
        print(explore.path(G, g, chunks, args.names[0], args.names[1]))
    else:
        from . import config
        name = Path(args.config).stem if args.config else Path(args.root).resolve().name
        out = Path(args.out or config.home() / "out" / name)
        out.mkdir(parents=True, exist_ok=True)
        comm = explore.subsystems(G)
        if args.action in ("report", "all"):
            (out / "REPORT.md").write_text(explore.report(G, comm, name), encoding="utf-8")
        if args.action in ("html", "all"):
            (out / "graph.html").write_text(explore.to_html(G, comm, name), encoding="utf-8")
        print(f"{G.number_of_nodes()} files, {G.number_of_edges()} links, {len(set(comm.values()))} subsystems -> {out}")


HOOK_MARK = "# memlab: rebuild graph.html + REPORT.md and warm the search index after each commit"


def cmd_hook(args):
    """Append a background post-commit block to the repo's shared hooks dir (all worktrees);
    with --claude, add the session hooks (memlab/hooks.py) to ~/.claude/settings.json instead."""
    if args.claude:
        from . import hooks
        print(hooks.install())
        return
    import subprocess
    root = Path(args.root).resolve()
    common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=root, text=True).strip())
    hook = (common if common.is_absolute() else root / common) / "hooks" / "post-commit"
    py = Path(sys.executable).as_posix()
    repo = Path(__file__).resolve().parents[1].as_posix()
    cfg = f' --config "{Path(args.config).resolve().as_posix()}"' if args.config else ""
    nl = chr(10)
    block = (nl + HOOK_MARK + nl +
             f'( top="$(git rev-parse --show-toplevel)"; export PYTHONPATH="{repo}" PYTHONIOENCODING=utf-8; '
             f'"{py}" -m memlab explore all{cfg} --root "$top"; '
             f'"{py}" -m memlab warm{cfg} --root "$top" ) >/dev/null 2>&1 &' + nl)
    text = hook.read_text(encoding="utf-8") if hook.exists() else "#!/bin/sh" + nl
    if HOOK_MARK in text:
        print(f"already installed: {hook}")
        return
    hook.parent.mkdir(parents=True, exist_ok=True)
    # Right after the shebang: an existing hook may `exit 0` on every path, so an appended block would never run.
    first, _, rest = text.partition(nl)
    hook.write_text(first + nl + block + rest, encoding="utf-8")
    hook.chmod(0o755)
    print(f"installed: {hook}")


def cmd_warm(args):
    """Build the server index once so the embedding stores hold every current chunk."""
    from . import config, serve
    root = Path(args.root).resolve()
    serve.Index(root, config.load(root, args.config)).wait()


def main():
    p = argparse.ArgumentParser(prog="memlab")
    sub = p.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("serve", help="MCP server (stdio) with search_code and search_decisions")
    sv.add_argument("--config", type=Path, help="project config (default: built-in defaults + <root>/.memlab.toml)")
    sv.add_argument("--root", type=Path, default=Path("."), help="project working tree to index")
    sv.set_defaults(fn=lambda a: __import__("memlab.serve", fromlist=["serve"]).serve(a.config, a.root))
    ex = sub.add_parser("explore", help="graph views: html, report, explain NAME, path A B, all")
    ex.add_argument("action", choices=["html", "report", "explain", "path", "all"])
    ex.add_argument("names", nargs="*")
    ex.add_argument("--config", type=Path, help="project config (default: built-in defaults + <root>/.memlab.toml)")
    ex.add_argument("--root", type=Path, default=Path("."))
    ex.add_argument("--out", help="output dir for html/report (default out/<config name>)")
    ex.set_defaults(fn=cmd_explore)
    hk = sub.add_parser("hook-install", help="post-commit hook: rebuild graph views and warm the index")
    hk.add_argument("--claude", action="store_true",
                    help="instead: Claude Code SessionStart/UserPromptSubmit hooks in ~/.claude/settings.json")
    hk.add_argument("--config", type=Path, help="project config (default: built-in defaults + <root>/.memlab.toml)")
    hk.add_argument("--root", type=Path, default=Path("."))
    hk.set_defaults(fn=cmd_hook)
    em = sub.add_parser("embedder", help="shared embedding process (started by servers on demand)")
    em.set_defaults(fn=lambda a: __import__("memlab.embedder", fromlist=["run"]).run())
    wm = sub.add_parser("warm", help="build the search index once (embeds changed chunks)")
    wm.add_argument("--config", type=Path, help="project config (default: built-in defaults + <root>/.memlab.toml)")
    wm.add_argument("--root", type=Path, default=Path("."))
    wm.set_defaults(fn=cmd_warm)
    g = sub.add_parser("cases-from-git")
    g.add_argument("--config", type=Path, required=True)
    g.add_argument("--since", default="2026-06-01")
    g.add_argument("--n", type=int, default=80)
    g.add_argument("--out", required=True)
    g.add_argument("--decisions", action="store_true", help="mine commits that added a decision document")
    g.set_defaults(fn=cmd_gitcases)
    for name, fn in (("stats", cmd_stats), ("eval", cmd_eval)):
        s = sub.add_parser(name)
        s.add_argument("--config", type=Path, required=True)
        s.set_defaults(fn=fn)
        if name == "eval":
            s.add_argument("--budgets", default="8000,32000")
            s.add_argument("--methods", default="bm25,dense,hybrid")
            s.add_argument("--cases", help="override the config's cases file")
            s.add_argument("--ref", default="hybrid", help="method the others are compared to")
            s.add_argument("--model", help="override the config's embedding model")
            s.add_argument("--stopwords", action="store_true", help="drop RU/EN stopwords from queries")
            s.add_argument("--stem", action="store_true", help="Snowball-stem Russian tokens in BM25 (H12)")
            s.add_argument("--strip-scope", action="store_true",
                           help="drop a leading 'scope: ' from queries (leakage control)")
            s.add_argument("--external", action="append",
                           help="name=results.json from an external system (see tools/graphify_adapter.py)")
            s.add_argument("--lang", choices=["ru", "en"], help="keep only Russian (Cyrillic) or English queries")
            s.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3", help="cross-encoder for rerank:<k>:<base>")
            s.add_argument("--out")
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
