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
