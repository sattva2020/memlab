import datetime

from memlab.serve import write_note

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
