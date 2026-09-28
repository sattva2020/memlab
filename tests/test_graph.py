import numpy as np

from memlab.corpus import chunk_file
from memlab.graph import Graph

FILES = {
    "lib/a.dart": "import 'package:app/b.dart';\n\nclass Alpha {\n  Bravo b;\n}\n",
    "lib/b.dart": "class Bravo {}\n",
    "lib/c.dart": "class Charlie {}\n",
    "docs/x.md": "# X\nsee [b](../lib/b.dart) and `lib/c.dart`\n",
}


def make():
    chunks = [c for p, t in FILES.items() for c in chunk_file(p, t)]
    return chunks, Graph(chunks, dart_pkg="app", dart_root="lib")


def test_edges():
    chunks, g = make()
    assert g.file_edges["lib/a.dart"] == {"lib/b.dart"}
    assert g.file_edges["docs/x.md"] == {"lib/b.dart", "lib/c.dart"}
    alpha = next(i for i, c in enumerate(chunks) if c.symbol == "Alpha")
    bravo = next(i for i, c in enumerate(chunks) if c.symbol == "Bravo")
    assert bravo in g.refs[alpha]


def test_expand_inserts_neighbours_after_anchor():
    chunks, g = make()
    alpha = next(i for i, c in enumerate(chunks) if c.symbol == "Alpha")
    bravo = next(i for i, c in enumerate(chunks) if c.symbol == "Bravo")
    seed = np.array([alpha] + [i for i in range(len(chunks)) if i not in (alpha, bravo)] + [bravo])
    out = g.expand(seed, {"refs"}, anchors=1, per_anchor=1)
    assert list(out[:2]) == [alpha, bravo] and sorted(out) == sorted(seed)


def test_ppr_returns_permutation():
    chunks, g = make()
    s = np.zeros(len(chunks)); s[0] = 1.0
    r = g.ppr(s, {"refs", "imports", "siblings"}, seeds=1)
    assert sorted(r) == list(range(len(chunks))) and r[0] == 0


def test_degree_normalization_demotes_the_hub():
    from memlab.corpus import Chunk
    chunks = [Chunk(f"f{i}.dart", f"f{i}.dart", "", "x") for i in range(5)]
    g = Graph(chunks)
    g.file_edges = {"f0.dart": {"f1.dart", "f2.dart", "f3.dart", "f4.dart"},
                    **{f"f{i}.dart": {"f0.dart"} for i in range(1, 5)}}
    s = np.zeros(5); s[1] = 1.0; s[2] = 0.9
    assert g.ppr(s, {"imports"}, seeds=2)[0] == 0          # plain PPR: the hub wins
    assert g.ppr(s, {"imports"}, seeds=2, beta=1.0)[0] != 0  # normalized: a seed wins


def test_ts_alias_and_relative_imports():
    files = {
        "frontend/app/page.tsx": "import { Card } from '@/components/card'\nimport x from 'react'\n",
        "frontend/components/card.tsx": "import { T } from '@shared/types'\nexport function Card() {}\n",
        "shared/types.ts": "export type T = string\n",
    }
    chunks = [c for p, t in files.items() for c in chunk_file(p, t)]
    g = Graph(chunks, aliases={"@/": "frontend/", "@shared/": "shared/"})
    assert g.file_edges["frontend/app/page.tsx"] == {"frontend/components/card.tsx"}
    assert "shared/types.ts" in g.file_edges["frontend/components/card.tsx"]


def test_transit_block_keeps_hub_reachable_but_stops_propagation():
    from memlab.corpus import Chunk
    n = 60
    chunks = [Chunk(f"f{i}.dart", f"f{i}.dart", "", "x") for i in range(n)]
    g = Graph(chunks)
    # f0 is a hub linked to everyone; f1-f2 is a private pair
    g.file_edges = {"f0.dart": {f"f{i}.dart" for i in range(1, n)}}
    for i in range(1, n):
        g.file_edges.setdefault(f"f{i}.dart", set()).add("f0.dart")
    g.file_edges["f1.dart"].add("f2.dart"); g.file_edges["f2.dart"] = {"f0.dart", "f1.dart"}
    s = np.zeros(n); s[1] = 1.0
    plain = list(g.ppr(s, {"imports"}, seeds=1))
    blocked = list(g.ppr(s, {"imports"}, seeds=1, transit_block=True))
    assert plain[0] == 0                        # plain PPR: the hub outranks the seed itself
    assert blocked[0] == 1 and blocked[1] == 0  # blocked: seed first, hub still found
    assert blocked.index(2) < blocked.index(3)  # the seed's private neighbour beats hub-only nodes


def test_files_in_order_dedupes_paths():
    from memlab.corpus import Chunk
    from memlab.evaluate import files_in_order
    chunks = [Chunk("a::x", "a", "x", ""), Chunk("a::y", "a", "y", ""), Chunk("b", "b", "", "")]
    assert files_in_order(np.array([0, 1, 2]), chunks, 2) == ["a", "b"]


def test_aider_weighting_boosts_specific_names_and_damps_common_ones():
    files = {
        "lib/a.dart": "class RepCounterService {}\nclass _Helper {}\n",
        "lib/b.dart": "void f() { RepCounterService(); RepCounterService(); _Helper(); }\n",
    }
    chunks = [c for p, t in files.items() for c in chunk_file(p, t)]
    g = Graph(chunks, weighting="aider")
    rc = next(i for i, c in enumerate(chunks) if c.symbol == "RepCounterService")
    hp = next(i for i, c in enumerate(chunks) if c.symbol == "_Helper")
    user = next(i for i, c in enumerate(chunks) if c.path == "lib/b.dart" and c.symbol == "f")
    w = lambda a, b: g.ref_w[(min(a, b), max(a, b))]
    assert abs(w(user, rc) - 10 * 2 ** 0.5) < 1e-9     # long camel name x10, mentioned twice
    assert abs(w(user, hp) - 0.1) < 1e-9               # _private x0.1, mentioned once
