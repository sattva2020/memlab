from memlab.corpus import chunk_file

DART = '''import 'package:x/x.dart';

const String kVersion = '1';

/// Counts reps.
class RepCounter {
  RepCounter copyWith() => this;
  void calibrate(List<double> a) {
    debugPrint('x');
  }
}

enum Phase { idle, bottom }

Future<void> main() async {
  runApp(App());
}
'''

MD = "intro\n## Setup\nsteps\n## Usage\nrun it\n## Setup\nagain\n"


def test_dart_symbols_are_top_level_only():
    syms = [c.symbol for c in chunk_file("a.dart", DART)]
    assert syms == ["", "kVersion", "RepCounter", "Phase", "main"], syms


def test_markdown_headings_and_duplicates():
    ids = [c.id for c in chunk_file("d.md", MD)]
    assert ids == ["d.md", "d.md#setup", "d.md#usage", "d.md#setup-1"], ids


def test_long_chunk_is_split_into_parts():
    cs = chunk_file("b.dart", "class Big {\n" + "  int x = 1;\n" * 1000 + "}\n")
    assert len(cs) > 1 and all(c.symbol == "Big" for c in cs) and cs[1].id == "b.dart::Big@1"


def test_select_channel_reserves_budget_for_the_side_ranking():
    import numpy as np
    from memlab.corpus import Chunk
    from memlab.evaluate import select_channel
    chunks = [Chunk(f"c{i}", f"c{i}", "", "x" * 400) for i in range(10)]   # 100 tokens each
    main, side = np.arange(10), np.array([9])
    ids = [c.id for c in select_channel(side, main, chunks, 300, 0.34)]
    assert ids[0] == "c9" and len(ids) == 3 and "c9" not in ids[1:]


def test_interleave_keeps_each_side_in_order():
    import numpy as np
    from memlab.evaluate import interleave
    is_code = np.array([False, False, False, True, True, True, True])
    out = interleave(np.array([0, 1, 2, 3, 4, 5, 6]), is_code, 3, 1)
    assert list(out) == [3, 4, 5, 0, 6, 1, 2]


def test_cli_module_compiles():
    # the CLI is not imported by other tests; a syntax error there once broke a whole batch of runs
    import py_compile
    from pathlib import Path
    py_compile.compile(str(Path(__file__).resolve().parents[1] / "memlab" / "__main__.py"), doraise=True)


def test_chunks_carry_their_first_line():
    lines = {c.symbol: c.line for c in chunk_file("a.dart", DART)}
    assert lines["kVersion"] == 3 and lines["RepCounter"] == 6 and lines["main"] == 15, lines


def test_dense_store_reembeds_only_changed_chunks(tmp_path):
    from memlab.corpus import Chunk
    from memlab.retrieve import Dense
    m = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    a = [Chunk("a", "a", "", "alpha text"), Chunk("b", "b", "", "beta text")]
    d1 = Dense(a, m, tmp_path)
    assert d1.reembedded == 2
    d2 = Dense([a[0], Chunk("b", "b", "", "beta text edited")], m, tmp_path)
    assert d2.reembedded == 1 and d2.emb.shape == (2, d1.emb.shape[1])


def test_stem_ru_only_cyrillic():
    from memlab.retrieve import query_tokens, tokenize
    assert tokenize("уведомления AdaptiveRepCounter", stem=True) == \
        ["уведомлен", "adaptive", "rep", "counter", "adaptiverepcounter"]
    assert query_tokens("почему не приходят уведомлений", True, True) == ["поч", "приход", "уведомлен"]


def test_list_files_outside_git_walks_the_folder(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / ".hidden").mkdir(); (tmp_path / ".hidden" / "b.py").write_text("y = 2\n")
    (tmp_path / "node_modules").mkdir(); (tmp_path / "node_modules" / "c.js").write_text("z\n")
    from memlab.corpus import list_files
    assert list_files(tmp_path, [], [], 400_000) == ["a.py"]


def test_home_and_drive_root_are_not_walked():
    from pathlib import Path
    from memlab.corpus import _walk
    assert _walk(Path.home()) == []
    assert _walk(Path(Path.home().anchor)) == []
