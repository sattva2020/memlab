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
