import json

from memlab import serve


def test_journal_appends_json_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "LOG", tmp_path / "logs" / "calls.jsonl")
    serve.journal("start", root="E:/x")
    serve.journal("call", tool="search_code", args={"query": "повторы"}, top=["a.dart:1"])
    rows = [json.loads(l) for l in (tmp_path / "logs" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in rows] == ["start", "call"] and rows[1]["args"]["query"] == "повторы"
