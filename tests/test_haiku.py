import json

import pytest

from memlab import haiku


def test_message_numbers_records_and_cuts_long_text():
    m = json.loads(haiku.message("какой платёж?", [("a.md", "x" * 5000), ("b.md", "y")]))
    assert m["question"] == "какой платёж?"
    assert [r["id"] for r in m["records"]] == [0, 1] and len(m["records"][0]["text"]) == haiku.MAX_CHARS


def test_scores_in_record_order_missing_is_zero_and_clamped():
    out = {"structured_output": {"scores": [{"id": 2, "score": 0.9}, {"id": 0, "score": 1.4}, {"id": 7, "score": 1}]}}
    assert haiku.scores_from(out, 3) == [1.0, 0.0, 0.9]


def test_cli_error_raises():
    with pytest.raises(RuntimeError):
        haiku.scores_from({"is_error": True, "result": "Failed to authenticate"}, 3)
