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


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._reply({"ok": True, "pid": os.getpid(), "models": [k[0] for k in _models]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
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

    def _alive(self) -> bool:
        try:
            with urllib.request.urlopen(self.url, timeout=2) as r:
                return json.loads(r.read())["ok"]
        except (OSError, ValueError):
            return False

    def _start(self) -> None:
        if self._alive():
            return
        repo = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(repo), PYTHONIOENCODING="utf-8")
        flags = 0
        if os.name == "nt":      # outlive the session that started it; no console window
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        for extra in ((0x01000000,) if os.name == "nt" else ()) + (0,):   # try CREATE_BREAKAWAY_FROM_JOB first
            try:
                subprocess.Popen([sys.executable, "-m", "memlab", "embedder"], cwd=repo, env=env,
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

    def _post(self, texts: list[str], batch_size: int) -> np.ndarray:
        body = json.dumps({"model": self.model, "max_len": self.max_len, "texts": texts,
                           "batch_size": batch_size}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read())
        return np.frombuffer(base64.b64decode(out["data"]), dtype=np.float32).reshape(out["shape"])

    def encode(self, texts: list[str], batch_size: int = 16, **_) -> np.ndarray:
        parts = []
        for i in range(0, len(texts), SLICE):
            try:
                parts.append(self._post(texts[i:i + SLICE], batch_size))
            except urllib.error.URLError:            # not running yet, or exited while idle
                self._start()
                parts.append(self._post(texts[i:i + SLICE], batch_size))
        return np.vstack(parts) if parts else np.zeros((0, 0), np.float32)
