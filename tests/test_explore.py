from memlab.corpus import Chunk
from memlab.explore import explain, file_graph, path, plain_word, report, subsystems
from memlab.graph import Graph


def _setup():
    chunks = [
        Chunk("a.dart::Counter", "lib/a.dart", "Counter", "class Counter { void tick() {} }"),
        Chunk("b.dart::Screen", "lib/b.dart", "Screen", "class Screen { final c = Counter(); _clamp(); }"),
        Chunk("c.dart::_clamp", "lib/c.dart", "_clamp", "double _clamp(x) => x; Screen s;"),
        Chunk("d.dart::_clamp", "lib/d.dart", "_clamp", "double _clamp(x) => x;"),
        Chunk("n.md#note", "docs/adr/0001.md", "", "Why `Counter` counts reps"),
    ]
    g = Graph(chunks, max_ref_df=10, max_defs=3)
    return chunks, g, file_graph(chunks, g, ["docs/adr/*"])


def test_file_graph_kinds_and_private_dart_clash():
    chunks, g, G = _setup()
    assert G.nodes["docs/adr/0001.md"]["kind"] == "decision"
    assert G.has_edge("lib/a.dart", "lib/b.dart") and G.edges["lib/a.dart", "lib/b.dart"]["via"] == "Counter"
    assert not G.has_edge("lib/b.dart", "lib/d.dart")          # `_clamp` is file-private in Dart
    assert plain_word("report") and not plain_word("usedMinutes")


def test_explain_path_report():
    chunks, g, G = _setup()
    text = explain(G, g, chunks, "Counter")
    assert "lib/a.dart:1" in text and "lib/b.dart" in text and "docs/adr/0001.md" in text
    assert "1 hops" in path(G, g, chunks, "Counter", "Screen")
    assert "# demo" in report(G, subsystems(G), "demo")
