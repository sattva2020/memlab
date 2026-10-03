import numpy as np

from memlab.retrieve import save_store


def test_second_writer_keeps_the_first_writers_vectors(tmp_path):
    path = tmp_path / "vecstore-m.npz"
    save_store(path, {"a": np.ones(2, np.float32)})          # project A's build
    save_store(path, {"b": np.zeros(2, np.float32)})         # project B loaded the store before A wrote
    z = np.load(path)
    assert sorted(z["keys"].tolist()) == ["a", "b"]
    assert not list(tmp_path.glob("*.tmp.npz"))
