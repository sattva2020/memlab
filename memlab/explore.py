"""Human-facing views of the project graph: file graph, subsystems, explain, path, report, HTML.

Everything is derived from the same chunk graph the retrievers use (no LLM):
  files are nodes; an edge joins two files that import/link each other ("imports") or whose
  chunks reference each other's symbols ("refs", weight = number of chunk pairs, labelled
  with one shared symbol). Subsystems are Louvain communities of that file graph.
"""
from __future__ import annotations

import fnmatch
import html
import json
import posixpath
import re
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

from .corpus import CODE_EXT, MD_EXT, TEXT_EXT, Chunk
from .graph import Graph

GENERATED = ("*.g.dart", "*.freezed.dart", "*.gen.dart", "*.min.js")
TESTS = ("*/test/*", "*/tests/*", "test/*", "tests/*", "*_test.*", "*.test.*", "*.spec.*", "*integration_test/*")


def is_test(p: str) -> bool:
    return any(fnmatch.fnmatch(p, g) for g in TESTS)


def plain_word(name: str) -> bool:
    """`sign`, `report`, `threshold`: an ordinary word, so a match across files is usually a clash."""
    return name.isalpha() and name.islower()


def kind_of(path: str, decision_globs: list[str]) -> str:
    if any(fnmatch.fnmatch(path, g) for g in decision_globs):
        return "decision"
    return "code" if Path(path).suffix.lower() in CODE_EXT else "doc"


def file_graph(chunks: list[Chunk], graph: Graph, decision_globs: list[str]) -> nx.Graph:
    G = nx.Graph()
    lines = Counter()
    for c in chunks:
        lines[c.path] += c.text.count("\n") + 1
    for p in graph.by_file:
        G.add_node(p, kind=kind_of(p, decision_globs), lines=lines[p],
                   symbols=sorted({c.symbol.split("~")[0] for i in graph.by_file[p]
                                   if (c := chunks[i]).symbol})[:40])
    for a, bs in graph.file_edges.items():
        for b in bs:
            if a < b:
                G.add_edge(a, b, imports=True, refs=0, via="")
    for u, ds in graph.refs.items():
        for d in ds:
            a, b = chunks[u].path, chunks[d].path
            if a >= b:
                continue
            sym = chunks[d].symbol.split("~")[0]
            via = sym if sym and sym in chunks[u].text else chunks[u].symbol.split("~")[0]
            # Dart `_names` are library-private: two files never share one, the match is a name clash.
            if via.startswith("_") and a.endswith(".dart") and b.endswith(".dart") or plain_word(via):
                continue
            if not G.has_edge(a, b):
                G.add_edge(a, b, imports=False, refs=0, via="")
            e = G.edges[a, b]
            e["refs"] += 1
            e["via"] = e["via"] or via
    for _, _, e in G.edges(data=True):
        e["weight"] = (2.0 if e["imports"] else 0.0) + e["refs"]
    return G


def subsystems(G: nx.Graph) -> dict[str, int]:
    """file -> community id, largest community first (Louvain, fixed seed)."""
    comms = nx.community.louvain_communities(G, weight="weight", seed=0, resolution=1.0)
    comms = sorted(comms, key=len, reverse=True)
    return {p: i for i, c in enumerate(comms) for p in c}


def label(files: list[str]) -> str:
    """A community's name: its most common directory, two levels below the shared prefix."""
    dirs = Counter()
    code = [f for f in files if Path(f).suffix.lower() in CODE_EXT and not is_test(f)]
    files = code or files
    for f in files:
        parts = f.split("/")[:-1]
        dirs["/".join(parts[:4]) or "."] += 1
    top, n = dirs.most_common(1)[0]
    return top if n * 2 >= len(files) else f"{top} +{len(dirs) - 1}"


def resolve(G: nx.Graph, graph: Graph, chunks: list[Chunk], name: str) -> list[str]:
    """A symbol, a file path or a path fragment -> matching files (definitions first)."""
    if name in G:
        return [name]
    if name in graph.defs:
        return list(dict.fromkeys(chunks[i].path for i in graph.defs[name]))
    low = name.lower()
    ci = [k for k in graph.defs if k.lower() == low]
    if ci:
        return list(dict.fromkeys(chunks[i].path for i in graph.defs[ci[0]]))
    hits = [p for p in G if p.lower().endswith(low) or posixpath.basename(p).lower() == low]
    # `entitlementsProvider` -> entitlements_provider.dart (generated Riverpod names live in *.g.dart)
    norm = low.replace("_", "")
    hits = hits or [p for p in G if posixpath.splitext(posixpath.basename(p))[0].replace("_", "").lower() == norm]
    return hits or [p for p in G if low in p.lower()][:10]


def explain(G, graph, chunks, name: str, decisions=None, limit: int = 12) -> str:
    files = resolve(G, graph, chunks, name)
    if not files:
        return f"nothing named {name!r} in the index"
    out = []
    if name in graph.defs:
        out.append(f"## {name}\ndefined at: " + ", ".join(f"{chunks[i].path}:{chunks[i].line}" for i in graph.defs[name]))
        users = Counter(chunks[u].path for d in graph.defs[name] for u in graph.refs.get(d, ())
                        if name in chunks[u].text)
        if users:
            out.append("used by: " + ", ".join(f"{p} ({n})" for p, n in users.most_common(limit)))
        callees = sorted({chunks[d].symbol.split("~")[0] for i in graph.defs[name] for d in graph.refs.get(i, ())
                          if chunks[d].symbol and chunks[d].symbol.split("~")[0] in chunks[i].text})
        if callees:
            out.append("uses: " + ", ".join(callees[:limit]))
    for f in files[:3]:
        n = G.nodes[f]
        out.append(f"## {f}  [{n['kind']}, {n['lines']} lines, degree {G.degree(f)}]")
        if n["symbols"]:
            out.append("symbols: " + ", ".join(n["symbols"][:limit]))
        nb = sorted(G[f].items(), key=lambda kv: -kv[1]["weight"])
        for kind in ("code", "doc", "decision"):
            rows = [f"{p}" + (f" via {e['via']}" if e["via"] else " (import/link)")
                    for p, e in nb if G.nodes[p]["kind"] == kind][:limit]
            if rows:
                out.append(f"{kind} neighbours: " + "; ".join(rows))
    if decisions:
        out.append("## related decisions (search)\n" + decisions(name))
    return "\n".join(out)


def path(G, graph, chunks, a: str, b: str) -> str:
    fa, fb = resolve(G, graph, chunks, a), resolve(G, graph, chunks, b)
    if not fa or not fb:
        return f"not found: {a if not fa else b!r}"
    best = None
    for x in fa[:3]:
        for y in fb[:3]:
            try:
                p = nx.shortest_path(G, x, y)
            except nx.NetworkXNoPath:
                continue
            if best is None or len(p) < len(best):
                best = p
    if best is None:
        return f"no path between {a} and {b}"
    hops = [best[0]]
    for u, v in zip(best, best[1:]):
        e = G.edges[u, v]
        hops.append(f"  --{'import' if e['imports'] else 'ref'}{(' ' + e['via']) if e['via'] else ''}--> {v}")
    return f"path {a} -> {b}: {len(best) - 1} hops\n" + "\n".join(hops)


PATH_REF = re.compile(r"(?<![\w@./:-])[\w@.-]+(?:/[\w@.-]+)+")   # not inside a URL


def missing_refs(G: nx.Graph, chunks: list[Chunk], root: Path) -> list[tuple[str, str]]:
    """(decision record, path it names) for paths that no longer exist: decisions that drifted
    from the code. A path counts as present when it exists under root or ends an indexed file's path
    (records often name `lib/x.dart` for `app/lib/x.dart`)."""
    exts = CODE_EXT | MD_EXT | TEXT_EXT
    files = list(G)
    out = set()
    for c in chunks:
        if c.path not in G or G.nodes[c.path]["kind"] != "decision":
            continue
        for ref in PATH_REF.findall(c.text):
            ref = ref.split("#")[0].rstrip(".").removeprefix("./")   # keep `.ai-factory/...`
            if ref.startswith("/") or Path(ref).suffix.lower() not in exts or (root / ref).exists():
                continue
            if not any(f == ref or f.endswith("/" + ref) for f in files):
                out.add((c.path, ref))
    return sorted(out)


def report(G: nx.Graph, comm: dict[str, int], project: str, top: int = 15,
           missing: list[tuple[str, str]] = ()) -> str:
    gen = lambda p: any(fnmatch.fnmatch(p, g) for g in GENERATED)
    members = defaultdict(list)
    for p, c in comm.items():
        members[c].append(p)
    code = [p for p in G if G.nodes[p]["kind"] == "code" and not gen(p) and not is_test(p)]
    hubs = sorted(code, key=G.degree, reverse=True)[:top]
    out = [f"# {project}: project graph report", "",
           f"{G.number_of_nodes()} files ({len(code)} code), {G.number_of_edges()} links, "
           f"{len(members)} subsystems.", "", "## Hubs (most connected code files)", ""]
    out += [f"- `{p}` — {G.degree(p)} links" for p in hubs]
    out += ["", "## Subsystems", ""]
    for c in sorted(members, key=lambda c: -len(members[c]))[:top]:
        fs = members[c]
        lead = sorted(fs, key=lambda p: (G.nodes[p]["kind"] != "code" or is_test(p), -G.degree(p)))[:3]
        out.append(f"- **{label(fs)}** ({len(fs)} files): " + ", ".join(f"`{p}`" for p in lead))
    top_dir = lambda p: p.split("/")[0]
    cross = sorted(((u, v, e) for u, v, e in G.edges(data=True)
                    if comm[u] != comm[v] and top_dir(u) != top_dir(v)
                    and G.nodes[u]["kind"] == G.nodes[v]["kind"] == "code" and not is_test(u) and not is_test(v)),
                   key=lambda t: -t[2]["weight"])[:top]
    out += ["", "## Surprising connections (code across top-level folders and subsystems)", ""]
    out += [f"- `{u}` ↔ `{v}`" + (f" via `{e['via']}`" if e["via"] else " (import)") for u, v, e in cross] or ["- none"]
    undoc = [p for p in hubs if not any(G.nodes[n]["kind"] != "code" for n in G[p])]
    out += ["", "## Hubs no document or decision links to", ""]
    out += [f"- `{p}`" for p in undoc] or ["- none"]
    out += ["", "## Decision records naming files that no longer exist", ""]
    out += [f"- `{d}` → `{ref}`" for d, ref in missing] or ["- none"]
    return "\n".join(out) + "\n"


def community_layout(H: nx.Graph, members: dict[int, list[str]]) -> dict[str, tuple[float, float]]:
    """Each subsystem laid out on its own, subsystems placed on a sunflower spiral (largest in
    the middle, area ~ size), so clusters stay apart instead of collapsing into one blob."""
    import math
    pos, area = {}, 0.0
    golden = math.pi * (3 - math.sqrt(5))
    for i, c in enumerate(sorted(members, key=lambda c: -len(members[c]))):
        fs = members[c]
        radius = math.sqrt(len(fs))
        area += len(fs)
        r = math.sqrt(area) * 1.15 if i else 0.0
        cx, cy = r * math.cos(i * golden), r * math.sin(i * golden)
        sub = H.subgraph(fs)
        inner = nx.spring_layout(sub, weight="weight", iterations=50, seed=0) if len(fs) > 1 else {fs[0]: (0.0, 0.0)}
        for p, (x, y) in inner.items():
            pos[p] = (cx + x * radius * 0.5, cy + y * radius * 0.5)
    return pos


def to_html(G: nx.Graph, comm: dict[str, int], project: str) -> str:
    gen = lambda p: any(fnmatch.fnmatch(p, g) for g in GENERATED)
    keep = [p for p in G if not gen(p) and (G.degree(p) > 0 or G.nodes[p]["kind"] == "decision")]
    H = G.subgraph(keep)
    members = defaultdict(list)
    for p in keep:
        members[comm[p]].append(p)
    pos = community_layout(H, members)
    nodes = [{"id": p, "c": comm[p], "k": H.nodes[p]["kind"], "d": H.degree(p),
              "x": round(float(pos[p][0]), 4), "y": round(float(pos[p][1]), 4),
              "s": H.nodes[p]["symbols"][:25], "l": H.nodes[p]["lines"]} for p in keep]
    edges = [[u, v, int(e["imports"]), e["refs"], e["via"]] for u, v, e in H.edges(data=True)]
    labels = {c: label(fs) for c, fs in members.items()}
    data = json.dumps({"nodes": nodes, "edges": edges, "labels": labels}, ensure_ascii=False)
    tpl = (Path(__file__).with_name("graph_template.html")).read_text(encoding="utf-8")
    return tpl.replace("__TITLE__", html.escape(project)).replace("__DATA__", data.replace("</", "<\\/"))
