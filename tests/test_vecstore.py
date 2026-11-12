import numpy as np

from memlab.retrieve import load_store, save_store


def test_second_writer_keeps_the_first_writers_vectors(tmp_path):
    path = tmp_path / "vecstore-m.npz"
    save_store(path, {"a": np.ones(2, np.float32)})          # project A's build
    save_store(path, {"b": np.zeros(2, np.float32)})         # project B loaded the store before A wrote
    z = np.load(path)
    assert sorted(z["keys"].tolist()) == ["a", "b"]
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_locked_store_does_not_fail_the_build(tmp_path, monkeypatch):
    path = tmp_path / "vecstore-m.npz"
    save_store(path, {"a": np.ones(2, np.float32)})

    def locked(src, dst):
        raise PermissionError(5, "another server is reading the store")
    monkeypatch.setattr("memlab.retrieve.os.replace", locked)
    monkeypatch.setattr("memlab.retrieve.time.sleep", lambda s: None)
    save_store(path, {"b": np.zeros(2, np.float32)})         # must not raise
    assert np.load(path)["keys"].tolist() == ["a"]
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_damaged_store_is_rebuilt_not_fatal(tmp_path):
    path = tmp_path / "vecstore-m.npz"
    save_store(path, {"a": np.ones(64, np.float32)})
    raw = bytearray(path.read_bytes())
    raw[raw.find(b"vecs.npy") + 200] ^= 0xFF               # flip a byte inside vecs.npy: bad CRC-32
    path.write_bytes(bytes(raw))
    assert load_store(path) == {}                          # was: BadZipFile, the whole index build died
    save_store(path, {"b": np.zeros(64, np.float32)})      # the next save rewrites a readable store
    assert list(load_store(path)) == ["b"]


def test_threads_of_one_server_do_not_share_a_tmp_file(tmp_path, monkeypatch):
    import threading
    path = tmp_path / "vecstore-m.npz"
    seen = []
    real = np.savez
    monkeypatch.setattr("memlab.retrieve.np.savez", lambda f, **kw: (seen.append(str(f)), real(f, **kw)))
    ts = [threading.Thread(target=save_store, args=(path, {k: np.ones(2, np.float32)})) for k in "ab"]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(set(seen)) == 2                             # was one pid-only name: two writers, one file


def test_damaged_embedding_cache_is_rebuilt(tmp_path):
    from memlab.corpus import Chunk
    from memlab.retrieve import Dense

    class Enc:
        def encode(self, texts, **kw):
            return np.ones((len(texts), 4), np.float32)
    chunks = [Chunk(id="a.py::f", path="a.py", symbol="f", text="x")]
    Dense(chunks, "m", tmp_path, encoder=Enc())
    cache = next(tmp_path.glob("emb-*.npy"))
    cache.write_bytes(cache.read_bytes()[:40])             # truncated, as after a killed write
    assert Dense(chunks, "m", tmp_path, encoder=Enc()).emb.shape == (1, 4)
