import threading

import numpy as np

from memlab import jev, serve
from memlab.corpus import Chunk


class _Dense:
    def scores(self, query):
        return np.array([0.9, 0.8, 0.7])          # dense order: a, b, c


def _index(tmp_path, monkeypatch, key="k"):
    monkeypatch.setattr(serve, "LOG", tmp_path / "calls.jsonl")
    ix = serve.Index.__new__(serve.Index)
    ix.root, ix.error, ix.ready = tmp_path, None, threading.Event()
    ix.ready.set()
    ix.chunks = [Chunk(f"{p}.md", f"docs/adr/{p}.md", "", f"text {p}") for p in "abc"]
    ix.dec_ids, ix.dec_dense, ix.jev_key = np.arange(3), _Dense(), key
    return ix


def heads(text):
    return [l[4:].split(":")[0] for l in text.splitlines() if l.startswith("### ")]


def test_jev_reorders_and_says_no_match(tmp_path, monkeypatch):
    ix = _index(tmp_path, monkeypatch)
    monkeypatch.setattr(jev, "judge", lambda q, recs, key, timeout=4.0: ([0.1, 0.2, 0.9], 30))
    assert heads(ix.search_decisions("q", 2)) == ["docs/adr/c.md", "docs/adr/b.md"]
    monkeypatch.setattr(jev, "judge", lambda q, recs, key, timeout=4.0: ([0.1, 0.2, 0.3], 30))
    out = ix.search_decisions("q", 2)
    assert out.startswith("search_decisions: no recorded decision") and "- docs/adr/c.md:1" in out
    assert "text" not in (tmp_path / "calls.jsonl").read_text(encoding="utf-8")   # no record text in logs


def test_jev_failure_or_no_key_keeps_dense_order(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise TimeoutError("timed out")
    monkeypatch.setattr(jev, "judge", boom)
    assert heads(_index(tmp_path, monkeypatch).search_decisions("q", 2)) == ["docs/adr/a.md", "docs/adr/b.md"]
    assert '"status": "fallback"' in (tmp_path / "calls.jsonl").read_text(encoding="utf-8")
    assert heads(_index(tmp_path, monkeypatch, key=None).search_decisions("q", 2)) == ["docs/adr/a.md", "docs/adr/b.md"]
