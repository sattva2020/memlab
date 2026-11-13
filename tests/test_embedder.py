import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np

from memlab import embedder


class Fake:
    def encode(self, texts, **_):
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


def test_client_round_trip_in_slices(monkeypatch):
    monkeypatch.setattr(embedder, "_load", lambda model, max_len: Fake())
    monkeypatch.setattr(embedder, "SLICE", 2)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), embedder._Handler)
    srv.last = time.time()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        v = embedder.Client("fake", port=srv.server_address[1]).encode(["a", "bb", "ccc", "dddd", "e"])
        assert v.shape == (5, 2) and v[:, 0].tolist() == [1, 2, 3, 4, 1]
    finally:
        srv.shutdown()


def test_dense_over_no_chunks_scores_nothing(tmp_path):
    from memlab.retrieve import Dense
    d = Dense([], "unused", tmp_path, encoder=Fake())
    assert d.scores("anything").shape == (0,)


class FakeCE:
    def predict(self, pairs, **_):
        return np.array([len(t) for _, t in pairs], dtype=np.float32)


def _server(monkeypatch, reranker):
    monkeypatch.setattr(embedder, "_load_reranker", lambda model: reranker)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), embedder._Handler)
    srv.last = time.time()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_rerank_scores_pairs_and_is_none_without_gpu(monkeypatch):
    embedder._models.clear()
    srv = _server(monkeypatch, FakeCE())
    try:
        s = embedder.Client("ce", port=srv.server_address[1]).rerank("q", ["a", "bbb", "cc"])
        assert s.tolist() == [1, 3, 2]
    finally:
        srv.shutdown()
    embedder._models.clear()
    srv = _server(monkeypatch, None)
    try:
        assert embedder.Client("ce", port=srv.server_address[1]).rerank("q", ["a"]) is None
    finally:
        srv.shutdown()
        embedder._models.clear()


def test_device_prefers_cuda_then_mps_then_cpu():
    from types import SimpleNamespace as NS
    from memlab.embedder import device
    t = lambda cuda, mps: NS(cuda=NS(is_available=lambda: cuda), backends=NS(mps=NS(is_available=lambda: mps)))
    assert device(t(True, True)) == "cuda"
    assert device(t(False, True)) == "mps"
    assert device(t(False, False)) == "cpu"
    assert device(NS(cuda=NS(is_available=lambda: False), backends=NS())) == "cpu"   # torch without MPS support



def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_run_reports_a_port_it_cannot_bind(monkeypatch, capsys):
    """WinError 10013 on a port nobody listens on: Windows reserved it (Hyper-V / WinNAT excluded
    range). Exiting silently left clients waiting 30 s for 'did not start' and an empty log."""
    import pytest

    def denied(*_a, **_k):
        raise PermissionError(13, "access forbidden by its access permissions")
    monkeypatch.setattr(embedder, "ThreadingHTTPServer", denied)
    port = _free_port()
    with pytest.raises(SystemExit) as e:
        embedder.run(port=port)
    assert e.value.code == 1
    assert str(port) in capsys.readouterr().err


def test_run_steps_aside_when_another_embedder_holds_the_port():
    import socket
    with socket.socket() as other:
        other.bind(("127.0.0.1", 0))
        other.listen()
        assert embedder.run(port=other.getsockname()[1]) is None


def test_load_kwargs_keeps_fp16_for_e5_and_qwen_on_cuda():
    import torch
    for m in ("intfloat/multilingual-e5-small", "Qwen/Qwen3-Embedding-0.6B"):
        assert embedder.load_kwargs(m, True) == ({"torch_dtype": torch.float16}, {})
        assert embedder.load_kwargs(m, False) == ({}, {})


def test_load_kwargs_gemma_is_fp32_text_only():
    import torch
    for cuda in (True, False):
        mk, ck = embedder.load_kwargs("google/embeddinggemma-2", cuda)
        assert mk == {"torch_dtype": torch.float32}
        assert ck == {"vision_config": None, "audio_config": None}


def test_reranker_load_failure_is_negative_cached(monkeypatch, capsys):
    """A model that cannot load (not cached, hub down) must answer scores=None, not kill the
    connection: the build's warm-up rerank used to crash the whole index with RemoteDisconnected."""
    attempts = []

    def broken(model):
        attempts.append(model)
        raise RuntimeError("not cached")

    embedder._models.clear()
    monkeypatch.setattr(embedder, "_load_reranker", broken)
    monkeypatch.setattr(embedder, "_load", lambda *a: Fake())
    srv = ThreadingHTTPServer(("127.0.0.1", 0), embedder._Handler)
    srv.last = time.time()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        for _ in range(3):                    # distinct clients share the negative cache
            assert embedder.Client("ce", port=srv.server_address[1]).rerank("q", ["a"]) is None
        assert attempts == ["ce"]
        assert embedder._models[("rerank", "ce")] is None
        assert embedder.Client("fake", port=srv.server_address[1]).encode(["a"]).shape == (1, 2)
    finally:
        srv.shutdown(); srv.server_close()
        embedder._models.clear()
    err = capsys.readouterr().err
    assert err.count("reranker_disabled") == 1 and "MEMLAB_RERANK=0" in err


def test_index_build_survives_a_missing_reranker(monkeypatch, tmp_path, capsys):
    """End to end: warm's warm-up rerank against a broken loader must leave the index usable."""
    import subprocess as sp
    from memlab import serve
    from memlab.corpus import Chunk

    attempts = []

    def broken(model):
        attempts.append(model)
        raise OSError("model not cached")

    embedder._models.clear()
    monkeypatch.setattr(embedder, "_load_reranker", broken)
    monkeypatch.setattr(embedder, "_load", lambda *a: Fake())
    srv = ThreadingHTTPServer(("127.0.0.1", 0), embedder._Handler)
    srv.last = time.time()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    real_client = embedder.Client
    monkeypatch.setattr(embedder, "Client",
                        lambda model, **kw: real_client(model, port=srv.server_address[1], **kw))
    monkeypatch.setattr(serve.corpus, "build", lambda *a, **k: [
        Chunk("a.py", "a.py", "alpha", "def alpha():\n    return 1\n")])
    monkeypatch.setattr(serve.config, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(serve, "journal", lambda *a, **k: None)
    monkeypatch.delenv("MEMLAB_RERANK", raising=False)
    try:
        sp.run(["git", "init", "-q", str(tmp_path)], check=True)
        ix = serve.Index(tmp_path, {})
        ix.wait()                             # before the fix: RuntimeError RemoteDisconnected
        assert ix.error is None and ix.reranker is None
        assert "a.py" in ix.search_code("alpha", 6000)
        assert attempts == [serve.RERANKER]   # one load attempt, then the negative cache holds
    finally:
        srv.shutdown(); srv.server_close()
        embedder._models.clear()
    assert "reranker_disabled" in capsys.readouterr().err
