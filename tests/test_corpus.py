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


def test_mjs_chunks_like_javascript():
    text = ("import './dep.mjs';\n"
            "export const answer = () => 42;\n"
            "export default class Runner {}\n"
            "export async function start() { return 1; }\n")
    cs = chunk_file("a.mjs", text)
    assert [(c.id, c.symbol, c.line) for c in cs] == [
        ("a.mjs", "", 1), ("a.mjs::answer", "answer", 2),
        ("a.mjs::Runner", "Runner", 3), ("a.mjs::start", "start", 4)]


def test_ps1_joins_the_corpus_as_text(tmp_path):
    from memlab.corpus import CODE_EXT, TEXT_EXT, chunk_file, list_files
    assert ".ps1" in TEXT_EXT and ".ps1" not in CODE_EXT and ".mjs" in CODE_EXT
    (tmp_path / "run.ps1").write_text("function Invoke-Task {}\n", encoding="utf-8")
    (tmp_path / "x.mjs").write_text("export const answer = () => 42;\n", encoding="utf-8")
    (tmp_path / "y.bin").write_bytes(b"\x00\x01")
    assert sorted(list_files(tmp_path, [], [], 400_000)) == ["run.ps1", "x.mjs"]
    win = chunk_file("a.ps1", "function Invoke-Task {}\n")
    assert len(win) == 1 and win[0].symbol == "" and win[0].id == "a.ps1"


def test_bom_marked_files_decode_and_bad_encoding_is_skipped(tmp_path):
    from memlab.corpus import build
    (tmp_path / "a.ps1").write_bytes("# PowerShell\r\n".encode("utf-16"))
    (tmp_path / "b.md").write_bytes("# Title\n".encode("utf-8-sig"))
    (tmp_path / "d.txt").write_bytes(b"\xff\xfe\x00\x00" + "ok\n".encode("utf-32-le"))
    (tmp_path / "c.txt").write_bytes(b"caf\xe9 menu")      # CP1252 bytes: not UTF-8, no BOM
    cs = build(tmp_path, [], [], 400_000)
    assert sorted((c.path, c.text.splitlines()[0]) for c in cs) == [
        ("a.ps1", "# PowerShell"), ("b.md", "# Title"), ("d.txt", "ok")]


def test_cr_and_crlf_newlines_are_normalized(tmp_path):
    from memlab.corpus import build
    (tmp_path / "cr.py").write_bytes(b"class Alpha {}\rclass Bravo {}\r")
    (tmp_path / "crlf.py").write_bytes(b"class Alpha {}\r\nclass Bravo {}\r\n")
    cs = sorted((c.path, c.text) for c in build(tmp_path, [], [], 400_000))
    assert cs == [("cr.py", "class Alpha {}\n"), ("cr.py", "class Bravo {}\n"),
                  ("crlf.py", "class Alpha {}\n"), ("crlf.py", "class Bravo {}\n")]
