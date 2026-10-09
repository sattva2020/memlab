import datetime
import json
import os
import subprocess
import time

from memlab import hooks, serve


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_digest_lists_newest_decision_records_with_summary(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t"); _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "docs/notes").mkdir(parents=True); (tmp_path / "docs/adr").mkdir()
    (tmp_path / "docs/adr/0001-x.md").write_text("# Use X\n")
    (tmp_path / "docs/notes/n.md").write_text('---\nsummary: "picked Y"\n---\n# picked Y\n')
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "."); _git(tmp_path, "commit", "-qm", "c")
    out = hooks.digest(tmp_path)
    assert len(out) == 2 and any("docs/adr/0001-x.md — Use X" in ln for ln in out)
    assert any("docs/notes/n.md — picked Y" in ln for ln in out)


def test_session_start_in_a_worktree_tells_the_agent_to_pass_root(tmp_path, monkeypatch):
    monkeypatch.setattr(hooks, "LOG", tmp_path / "none.jsonl")
    main, wt = tmp_path / "main", tmp_path / "wt"
    main.mkdir(); _git(main, "init", "-q")
    _git(main, "config", "user.email", "t@t"); _git(main, "config", "user.name", "t")
    (main / "a.py").write_text("x = 1\n"); _git(main, "add", "."); _git(main, "commit", "-qm", "c")
    _git(main, "worktree", "add", "-q", str(wt))
    assert f'root="{wt.resolve().as_posix()}"' in hooks.session_start({"cwd": str(wt)})
    assert hooks.session_start({"cwd": str(main)}) == ""


def test_prompt_hint_is_gated_by_length_recent_calls_and_cooldown(tmp_path, monkeypatch):
    log, state = tmp_path / "calls.jsonl", tmp_path / "hint-state.json"
    monkeypatch.setattr(hooks, "LOG", log); monkeypatch.setattr(hooks, "STATE", state)
    now = datetime.datetime(2026, 10, 5, 12, 0)
    data = {"cwd": str(tmp_path), "prompt": "x" * 300}
    assert hooks.prompt_hint({**data, "prompt": "short"}, now) == ""
    root = os.path.normcase(str(tmp_path.absolute()))
    log.write_text("\n" + json.dumps({"ts": "2026-10-05T11:55:00", "event": "call", "root": root}) + "\n")
    assert hooks.prompt_hint(data, now) == ""                      # memlab used 5 min ago
    log.write_text("")
    assert hooks.prompt_hint(data, now) == hooks.HINT
    assert hooks.prompt_hint(data, now + datetime.timedelta(minutes=5)) == ""   # cooldown


def test_install_replaces_only_its_own_entries(tmp_path):
    s = tmp_path / "settings.json"
    s.write_text(json.dumps({"hooks": {"SessionStart": [
        {"hooks": [{"type": "command", "command": "other"}]},
        {"hooks": [{"type": "command", "command": "py E:/old/memlab/hooks.py session-start"}]}]}}))
    hooks.install(s); hooks.install(s)
    starts = json.loads(s.read_text())["hooks"]["SessionStart"]
    assert [g["hooks"][0]["command"] == "other" for g in starts] == [True, False]
    assert len(json.loads(s.read_text())["hooks"]["UserPromptSubmit"]) == 1


def test_claude_md_block_is_appended_once_then_replaced_in_place(tmp_path):
    md = tmp_path / "CLAUDE.md"
    md.write_text("# Mine\n\nkeep this\n", encoding="utf-8")
    hooks.write_block(md)
    hooks.write_block(md)
    text = md.read_text(encoding="utf-8")
    assert text.startswith("# Mine\n\nkeep this\n") and text.count(hooks.BEGIN) == 1
    md.write_text(f"top\n{hooks.BEGIN}\nold\n{hooks.END}\nbottom\n", encoding="utf-8")
    hooks.write_block(md)
    text = md.read_text(encoding="utf-8")
    assert text.startswith("top\n" + hooks.BEGIN) and text.endswith(hooks.END + "\nbottom\n") and "old" not in text


def test_usage_marks_shown_files_the_session_edited_afterwards(tmp_path):
    def line(kind, item):
        return json.dumps({"type": kind, "timestamp": "t", "message": {"content": [item]}})
    edit = lambda p: line("assistant", {"type": "tool_use", "id": "e", "name": "Edit",
                                        "input": {"file_path": str(tmp_path / p)}})
    t = tmp_path / "t.jsonl"
    t.write_text("\n".join([
        edit("a.py"),                                                   # before the search: not "used"
        line("assistant", {"type": "tool_use", "id": "s1", "name": "mcp__plugin_memlab_memlab__search_code",
                           "input": {"query": "q"}}),
        line("user", {"type": "tool_result", "tool_use_id": "s1", "content": [
            {"type": "text", "text": "search_code: 2\n\n### a.py:1  f\nx\n\n### b.py:3  ⟨stale⟩\ny"}]}),
        edit("b.py"), edit("c.py"),
        line("assistant", {"type": "tool_use", "id": "w", "name": "Write",
                           "input": {"file_path": "C:/elsewhere/b.py"}}),
    ]), encoding="utf-8")
    assert hooks.usage(t, tmp_path) == [
        {"ts": "t", "tool": "search_code", "query": "q", "shown": ["a.py", "b.py"], "used": ["b.py"]}]


def test_results_from_files_changed_after_indexing_are_marked(tmp_path):
    idx = object.__new__(serve.Index)
    idx.root, idx.built, idx.fresh, idx.superseded = tmp_path, time.time() - 10, {"new.md"}, {"old.py": "n.md"}
    (tmp_path / "old.py").write_text("x"); os.utime(tmp_path / "old.py", (idx.built - 5, idx.built - 5))
    (tmp_path / "edited.py").write_text("y"); (tmp_path / "new.md").write_text("z")
    assert idx._mark("old.py") == "  ⟨superseded by n.md⟩" and idx._mark("new.md") == ""
    assert "stale" in idx._mark("edited.py") and "deleted" in idx._mark("gone.py")


def test_note_warnings_uncommitted_notes_and_main_behind_origin(tmp_path):
    import subprocess
    from memlab.hooks import note_warnings

    def git(cwd, *a):
        subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                       check=True, capture_output=True)
    origin, main, wt = tmp_path / "origin", tmp_path / "main", tmp_path / "wt"
    git(tmp_path, "init", "-q", "-b", "main", str(origin))
    git(origin, "commit", "-q", "--allow-empty", "-m", "a")
    git(tmp_path, "clone", "-q", str(origin), str(main))
    assert note_warnings(main) == []                                  # clean and up to date
    git(origin, "commit", "-q", "--allow-empty", "-m", "b")
    git(main, "fetch", "-q")
    git(main, "worktree", "add", "-q", "-b", "feat", str(wt))
    (wt / "docs" / "notes").mkdir(parents=True)
    (wt / "docs" / "notes" / "n.md").write_text("x")
    w = note_warnings(main)
    assert any("1 uncommitted" in x and "/wt/docs/notes" in x for x in w)
    assert any("1 commit(s) behind origin/main" in x for x in w)


def test_index_outside_git_waits_for_the_first_call(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "LOG", tmp_path / "calls.jsonl")
    assert serve.is_git(tmp_path) is False
    idx = serve.Index(tmp_path, {})
    assert idx._started is False and not idx.ready.is_set()
    assert '"deferred"' in (tmp_path / "calls.jsonl").read_text()
