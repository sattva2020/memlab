import datetime
import subprocess

import pytest

from memlab.serve import note_root, supersessions, write_note

NOW = datetime.datetime(2026, 9, 28, 12, 0, 0)


def test_note_is_a_new_file_with_frontmatter(tmp_path):
    rel = write_note(tmp_path, "Счётчик повторов: F1 на одной записи", "body", "decision", ["pose"], now=NOW)
    assert rel.startswith("docs/notes/2026-09-28-счётчик-повторов-f1-на-одной-записи-")
    text = (tmp_path / rel).read_text(encoding="utf-8")
    assert text.startswith("---\ntype: decision\n") and '"pose"' in text and text.endswith("body\n")


def test_different_notes_never_collide(tmp_path):
    a = write_note(tmp_path, "same title", "one", now=NOW)
    b = write_note(tmp_path, "same title", "two", now=NOW)
    assert a != b


def test_note_root_accepts_only_a_worktree_of_the_same_repo(tmp_path):
    git = lambda *a: subprocess.run(["git", *a], check=True, capture_output=True)
    main, wt, other = tmp_path / "main", tmp_path / "wt", tmp_path / "other"
    for r in (main, other):
        git("init", "-q", str(r))
        git("-C", str(r), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x")
    git("-C", str(main), "worktree", "add", "-q", str(wt))
    assert note_root(main, None) == main
    assert note_root(main, str(wt)) == wt.resolve()
    with pytest.raises(ValueError):
        note_root(main, str(other))
    with pytest.raises(ValueError):
        note_root(main, str(tmp_path / "missing"))


def test_a_new_note_declares_what_it_supersedes(tmp_path):
    old = write_note(tmp_path, "use X")
    new = write_note(tmp_path, "use Y instead of X", supersedes=[old, r".\docs\adr\0001-x.md"])
    assert supersessions(tmp_path, [old, new]) == {old: new, "docs/adr/0001-x.md": new}
    assert "supersedes" not in (tmp_path / old).read_text(encoding="utf-8")
