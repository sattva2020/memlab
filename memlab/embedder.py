"""One shared embedding process for all MCP servers on the machine.

Each Claude session starts its own memlab server; loading e5 + Qwen3 in every one of them cost
~1.5 GB VRAM and ~6 GB committed RAM per session. The models now live once, in a small HTTP
process on 127.0.0.1 that the first server starts and that exits after IDLE seconds without work.
Servers stay thin: no torch, only BM25, the graph and the vectors.

  python -m memlab embedder        # normally started by a server, not by hand
"""
from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

PORT = int(os.environ.get("MEMLAB_EMBED_PORT", "8765"))
IDLE = int(os.environ.get("MEMLAB_EMBED_IDLE", "1800"))
PROTO = 2                        # bump when requests change: clients replace an older running process
SLICE = 256                      # texts per request: long index builds stay under any HTTP timeout

_models: dict[tuple, object] = {}
_lock = threading.Lock()         # one GPU, one encode at a time


def _load(model: str, max_len: int | None):
    import torch
    from sentence_transformers import SentenceTransformer
    cuda = torch.cuda.is_available()
    m = SentenceTransformer(model, device="cuda" if cuda else "cpu",
                            model_kwargs={"torch_dtype": torch.float16} if cuda else {})
    if max_len:
        m.max_seq_length = max_len
    return m


def _load_reranker(model: str):
    """Cross-encoder for search_code's top 50 (H15). GPU only: on a CPU it costs seconds per query,
    so without CUDA there is no reranker and search_code keeps its plain ranking."""
    import torch
    if not torch.cuda.is_available():
        return None
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model, max_length=512, device="cuda", model_kwargs={"torch_dtype": torch.float16})


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._reply({"ok": True, "proto": PROTO, "pid": os.getpid(), "models": [k[0] for k in _models]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "query" in req:                                    # rerank: score (query, text) pairs
            key = ("rerank", req["model"])
            with _lock:
                if key not in _models:
                    _models[key] = _load_reranker(req["model"])
                m = _models[key]
                s = None if m is None else m.predict([(req["query"], t) for t in req["texts"]],
                                                     batch_size=16, show_progress_bar=False)
            self.server.last = time.time()
            self._reply({"scores": None if s is None else np.asarray(s, dtype=np.float32).tolist()})
            return
        key = (req["model"], req.get("max_len"))
        with _lock:
            if key not in _models:
                _models[key] = _load(*key)
            v = _models[key].encode(req["texts"], batch_size=req.get("batch_size", 16),
                                    normalize_embeddings=True, show_progress_bar=False)
        v = np.asarray(v, dtype=np.float32)
        self.server.last = time.time()
        self._reply({"shape": v.shape, "data": base64.b64encode(v.tobytes()).decode()})

    def _reply(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def run(port: int = PORT, idle: int = IDLE) -> None:
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    except OSError:
        return                   # another server's embedder won the race; use that one
    srv.last = time.time()

    def watchdog():
        while time.time() - srv.last < idle or _lock.locked():
            time.sleep(10)
        srv.shutdown()
    threading.Thread(target=watchdog, daemon=True).start()
    srv.serve_forever()


class Client:
    """Stands in for a SentenceTransformer: `.encode(texts, ...)` answered by the shared process."""

    def __init__(self, model: str, max_len: int | None = None, port: int = PORT):
        self.model, self.max_len = model, max_len
        self.url = f"http://127.0.0.1:{port}"
        self._checked = False

    def _health(self) -> dict | None:
        try:
            with urllib.request.urlopen(self.url, timeout=2) as r:
                return json.loads(r.read())
        except (OSError, ValueError):
            return None

    def _alive(self) -> bool:
        h = self._health()
        return bool(h and h.get("proto") == PROTO)

    def _start(self) -> None:
        h = self._health()
        if h and h.get("proto") == PROTO:
            return
        if h:                    # an older embedder that busy sessions keep from idling out: replace it
            try:
                os.kill(h["pid"], signal.SIGTERM)
            except OSError:
                pass
            for _ in range(20):
                if self._health() is None:
                    break
                time.sleep(0.5)
        repo = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(repo), PYTHONIOENCODING="utf-8")
        flags = 0
        if os.name == "nt":      # outlive the session that started it; no console window
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        for extra in ((0x01000000,) if os.name == "nt" else ()) + (0,):   # try CREATE_BREAKAWAY_FROM_JOB first
            try:
                subprocess.Popen([sys.executable, "-m", "memlab", "embedder"], cwd=Path.home(), env=env,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, creationflags=flags | extra,
                                 start_new_session=os.name != "nt")
                break
            except OSError:
                continue
        for _ in range(60):
            if self._alive():
                return
            time.sleep(0.5)
        raise RuntimeError(f"memlab embedder did not start on {self.url}")

    def _request(self, payload: dict) -> dict:
        if not self._checked:    # once per client: start the process, or replace an outdated one
            self._start()
            self._checked = True
        body = json.dumps(payload).encode()
        for attempt in (0, 1):
            try:
                req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    return json.loads(r.read())
            except urllib.error.URLError:            # not running yet, or exited while idle
                if attempt:
                    raise
                self._start()

    def encode(self, texts: list[str], batch_size: int = 16, **_) -> np.ndarray:
        parts = []
        for i in range(0, len(texts), SLICE):
            out = self._request({"model": self.model, "max_len": self.max_len, "texts": texts[i:i + SLICE],
                                 "batch_size": batch_size})
            parts.append(np.frombuffer(base64.b64decode(out["data"]), dtype=np.float32).reshape(out["shape"]))
        return np.vstack(parts) if parts else np.zeros((0, 0), np.float32)

    def rerank(self, query: str, texts: list[str]) -> np.ndarray | None:
        """Cross-encoder scores for (query, text) pairs; None where the machine has no GPU."""
        s = self._request({"model": self.model, "query": query, "texts": texts})["scores"]
        return None if s is None else np.asarray(s, dtype=np.float32)
