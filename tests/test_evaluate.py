from memlab.evaluate import alternate


def test_alternate_base_first_dedup():
    assert alternate(["a", "b", "c", "d"], ["x", "b", "y"]) == ["a", "x", "b", "c", "y", "d"]
