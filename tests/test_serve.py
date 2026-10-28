"""Drive `memlab serve` over stdio like a client: tool advertisement and retrieval-only mode.

Requests are pipelined and read to EOF under a timeout, so a hung server fails the test
instead of hanging it. MEMLAB_HOME and MEMLAB_LOG point into tmp: no user-home writes.
"""
import json
import os
import subprocess
import sys


def _run(tmp_path, tools=None, calls=()):
    env = dict(os.environ, MEMLAB_HOME=str(tmp_path / "home"),
               MEMLAB_LOG=str(tmp_path / "home" / "logs" / "calls.jsonl"))
    env.pop("MEMLAB_TOOLS", None)
    if tools is not None:
        env["MEMLAB_TOOLS"] = tools
    reqs = [{"id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18",
                                                         "capabilities": {},
                                                         "clientInfo": {"name": "t", "version": "0"}}},
            {"id": None, "method": "notifications/initialized", "params": {}}]
    reqs += [{"id": i, "method": m, "params": prm} for i, (m, prm) in enumerate(calls, start=2)]
    p = subprocess.Popen([sys.executable, "-m", "memlab", "serve", "--root", str(tmp_path / "repo")],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    try:
        out, err = p.communicate("".join(json.dumps(r) + "\n" for r in reqs).encode(), timeout=120)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        raise AssertionError("serve did not answer and exit in time")
    replies = {}
    for line in out.decode("utf-8", "replace").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("id") is not None:
                replies[r["id"]] = r
    return p, replies, err.decode("utf-8", "replace")


def test_retrieval_mode_hides_add_note(tmp_path):
    (tmp_path / "repo").mkdir()
    p, r, _ = _run(tmp_path, tools="retrieval", calls=[("tools/list", {})])
    assert p.returncode == 0
    assert "add_note" not in r[1]["result"]["instructions"]
    assert "search_decisions" in r[1]["result"]["instructions"]      # retrieval guidance kept
    assert [t["name"] for t in r[2]["result"]["tools"]] == ["search_code", "search_decisions",
                                                            "explain", "find_path"]


def test_retrieval_mode_denies_add_note_before_any_work(tmp_path):
    (tmp_path / "repo").mkdir()
    other = tmp_path / "elsewhere"                                   # a worktree root the call names
    p, r, _ = _run(tmp_path, tools="retrieval",
                   calls=[("tools/call", {"name": "add_note",
                                          "arguments": {"summary": "s", "root": str(other)}})])
    assert p.returncode == 0
    res = r[2]["result"]
    assert res["isError"] and res["content"][0]["text"] == "error: unknown tool add_note"
    assert not (tmp_path / "repo" / "docs").exists() and not other.exists()


def test_default_mode_publishes_all_tools(tmp_path):
    (tmp_path / "repo").mkdir()
    p, r, _ = _run(tmp_path, calls=[("tools/list", {})])             # unset: the plugin's behavior
    assert p.returncode == 0
    assert "add_note" in r[1]["result"]["instructions"]
    assert [t["name"] for t in r[2]["result"]["tools"]] == ["search_code", "search_decisions",
                                                            "add_note", "explain", "find_path"]


def test_unknown_tools_value_fails_loud(tmp_path):
    (tmp_path / "repo").mkdir()
    p, _, err = _run(tmp_path, tools="readonly")                     # misspelled: fail, not silent-all
    assert p.returncode != 0
    assert "MEMLAB_TOOLS must be 'retrieval' or 'all'" in err
