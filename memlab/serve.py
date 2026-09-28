"""MCP server (stdio, JSON-RPC 2.0, newline-delimited) with two project-memory tools.

  search_code      — the stage 3 system: hybrid (BM25 + dense) seed → graphify-style seeds →
                     PPR over df-capped symbol references with hubs blocked as transit.
  search_decisions — Qwen3 dense over decision memory (session notes, ADRs, postmortems);
                     BM25 dropped: it buried paraphrased answers (H13).

Two tools instead of one shared ranking: every fixed or automatic split of one budget
between code and decisions lost in the experiments (H3, H4, H7); the agent knows which kind
of context it needs. The index is built from the working tree in a background thread at
start-up, so `initialize` and `tools/list` answer immediately.
"""
from __future__ import annotations

import datetime
import fnmatch
import hashlib
import json
import re
import sys
import threading
import time
import tomllib
from pathlib import Path

import numpy as np

from . import corpus, evaluate, explore, graph, retrieve

E5 = "intfloat/multilingual-e5-small"
QWEN = "Qwen/Qwen3-Embedding-0.6B"
QWEN_INSTRUCT = ("Instruct: Given a question about a software project, retrieve the notes, "
                 "decision records and postmortems that answer it\nQuery: ")

TOOLS = [
    {"name": "search_code",
     "description": ("Find the code in this repository relevant to a task or question: files, "
                     "classes and functions with path:line. Use for 'where is X implemented', "
                     "'what calls Y', 'which files does this change touch'."),
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "the task or question, any language"},
         "budget_tokens": {"type": "integer", "description": "context to return (default 6000, max 20000)"}},
         "required": ["query"]}},
    {"name": "search_decisions",
     "description": ("Find past decisions, session notes, ADRs and postmortems about this project: "
                     "why something was done, what was decided, what broke and how it was fixed."),
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string"},
         "k": {"type": "integer", "description": "documents to return (default 5, max 15)"}},
         "required": ["query"]}},
    {"name": "add_note",
     "description": ("Record a decision, conclusion or postmortem so later sessions find it with "
                     "search_decisions. Writes a new markdown file under docs/notes/ (never edits an "
                     "existing one, so worktrees merge cleanly); commit it with your change."),
     "inputSchema": {"type": "object", "properties": {
         "summary": {"type": "string", "description": "one line: what was decided and why"},
         "body": {"type": "string", "description": "details: context, alternatives, file paths"},
         "type": {"type": "string", "enum": ["decision", "note", "postmortem"]},
         "tags": {"type": "array", "items": {"type": "string"}}},
         "required": ["summary"]}},
    {"name": "explain",
     "description": ("Explain a symbol or file: where it is defined, which files use it and what it uses, "
                     "its code/doc/decision neighbours in the project graph, and related decisions."),
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "symbol (class/function) name, file path or file name"}},
         "required": ["name"]}},
    {"name": "find_path",
     "description": "Shortest chain of imports/symbol references connecting two symbols or files.",
     "inputSchema": {"type": "object", "properties": {
         "a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}},
]


NOTES_DIR = "docs/notes"


def write_note(root: Path, summary: str, body: str = "", kind: str = "decision",
               tags: list[str] | None = None, now: datetime.datetime | None = None) -> str:
    """A new note file, one per call: date + slug + content hash, created exclusively."""
    now = now or datetime.datetime.now()
    slug = "-".join(re.findall(r"[^\W_]+", summary.lower()))[:60].strip("-") or "note"
    text = (f"---\ntype: {kind}\ncreated: '{now.isoformat(timespec='seconds')}'\n"
            f"tags: {json.dumps(tags or [], ensure_ascii=False)}\n"
            f"summary: {json.dumps(summary, ensure_ascii=False)}\n---\n# {summary}\n\n{body}".rstrip() + "\n")
    h = hashlib.sha1(text.encode()).hexdigest()[:8]
    rel = f"{NOTES_DIR}/{now:%Y-%m-%d}-{slug}-{h}.md"
    (root / NOTES_DIR).mkdir(parents=True, exist_ok=True)
    with open(root / rel, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return rel


def log(msg: str) -> None:
    print(f"[memlab] {msg}", file=sys.stderr, flush=True)


class Index:
    def __init__(self, root: Path, cfg: dict):
        self.root, self.cfg = root, cfg
        self.ready = threading.Event()
        self.error: str | None = None
        threading.Thread(target=self._build, daemon=True).start()

    def _build(self) -> None:
        try:
            t = time.time()
            c = self.cfg.get("corpus", {})
            self.chunks = corpus.build(self.root, c.get("include", []), c.get("exclude", []),
                                       c.get("max_bytes", 400_000), c.get("keep_always", []))
            globs = self.cfg.get("channels", {}).get("decisions", []) + [f"{NOTES_DIR}/*"]
            self.is_dec = np.array([any(fnmatch.fnmatch(x.path, g) for g in globs) for x in self.chunks])
            self.is_code = np.array([Path(x.path).suffix.lower() in corpus.CODE_EXT for x in self.chunks])
            self.bm25 = retrieve.BM25([retrieve.tokenize(f"{x.path} {x.symbol} {x.text}") for x in self.chunks])
            repo = Path(__file__).resolve().parents[1]
            cache = repo / self.cfg.get("cache", ".cache")          # never inside the served project
            self.dense = retrieve.Dense(self.chunks, E5, cache, query_prefix="query: ", doc_prefix="passage: ")
            gc = self.cfg.get("graph", {})
            self.graph = graph.Graph(self.chunks, gc.get("dart_package", ""), gc.get("dart_root", ""),
                                     aliases=gc.get("aliases", {}))
            self.dec_ids = np.flatnonzero(self.is_dec)
            dec_chunks = [self.chunks[i] for i in self.dec_ids]
            self.files = explore.file_graph(self.chunks, self.graph, globs)
            self.dec_dense = retrieve.Dense(dec_chunks, QWEN, cache, query_prefix=QWEN_INSTRUCT, max_len=512)
            log(f"index ready: {len(self.chunks)} chunks, {len(self.dec_ids)} decision chunks, "
                f"re-embedded {self.dense.reembedded}+{self.dec_dense.reembedded}, {time.time() - t:.0f}s")
        except Exception as e:  # surfaced to the caller instead of a silent dead server
            self.error = f"{type(e).__name__}: {e}"
            log(f"index build failed: {self.error}")
        finally:
            self.ready.set()

    def wait(self) -> None:
        self.ready.wait()
        if self.error:
            raise RuntimeError(f"memlab index failed to build: {self.error}")

    # ------------------------------------------------------------------ tools
    def search_code(self, query: str, budget: int) -> str:
        self.wait()
        qt = retrieve.query_tokens(query, True)
        bs = self.bm25.scores(qt)
        rb, rd = retrieve.rank(bs), retrieve.rank(self.dense.scores(query))
        seed = retrieve.rrf_scores(rb, rd)
        rh = retrieve.rank(seed)
        seeds = self._gfy_seeds(qt, bs, rd, rh)
        ranking = self.graph.ppr(seed, {"refs"}, transit_block=True, seed_ids=seeds)
        ranking = ranking[self.is_code[ranking]]                      # code only (H8); prose has its own tool
        picked = evaluate.select(ranking, self.chunks, budget)
        return self._render(picked, f"search_code: {len(picked)} chunks, ~{budget} tokens budget")

    def search_decisions(self, query: str, k: int) -> str:
        self.wait()
        order = retrieve.rank(self.dec_dense.scores(query))                # dense only (H13)
        out, seen = [], set()
        for j in order:                                               # one best chunk per document
            c = self.chunks[self.dec_ids[j]]
            if c.path not in seen:
                seen.add(c.path)
                out.append(c)
                if len(out) == k:
                    break
        return self._render(out, f"search_decisions: {len(out)} documents", max_chars=1600)

    def add_note(self, summary: str, body: str, kind: str, tags: list[str]) -> str:
        """Write the note and make it searchable at once (decision channel only; the rest of
        the index picks it up at the next start)."""
        self.wait()
        rel = write_note(self.root, summary, body, kind, tags)
        new = corpus.chunk_file(rel, (self.root / rel).read_text(encoding="utf-8"))
        n = len(self.chunks)
        self.chunks = self.chunks + new
        self.is_dec = np.concatenate([self.is_dec, np.ones(len(new), bool)])
        self.is_code = np.concatenate([self.is_code, np.zeros(len(new), bool)])
        self.dec_ids = np.concatenate([self.dec_ids, np.arange(n, n + len(new))])
        vecs = self.dec_dense.model.encode([f"{c.path} {c.symbol}\n{c.text}" for c in new],
                                           normalize_embeddings=True)
        self.dec_dense.emb = np.vstack([self.dec_dense.emb, vecs.astype(np.float32)])
        return f"saved {rel} (searchable now; commit it with your change)"

    def explain(self, name: str) -> str:
        self.wait()
        heads = lambda q: chr(10).join(l for l in self.search_decisions(q, 3).splitlines() if l.startswith("### "))
        return explore.explain(self.files, self.graph, self.chunks, name, decisions=heads)

    def find_path(self, a: str, b: str) -> str:
        self.wait()
        return explore.path(self.files, self.graph, self.chunks, a, b)

    def _gfy_seeds(self, qt, bs, rd, rh) -> list[int]:
        mx, dense5 = bs.max(), set(rd[:5].tolist())
        out, labels = [], set()
        for i in rh[:50]:
            i = int(i)
            if not (bs[i] >= 0.2 * mx or i in dense5):
                continue
            lab = self.chunks[i].symbol.split("~")[0]
            if lab and lab in labels:
                continue
            labels.add(lab); out.append(i)
        for tok in dict.fromkeys(qt):
            s = self.bm25.scores([tok])
            if s.max() > 0 and int(s.argmax()) not in out:
                out.append(int(s.argmax()))
        return out

    @staticmethod
    def _render(chunks: list[corpus.Chunk], header: str, max_chars: int | None = None) -> str:
        parts = [header]
        for c in chunks:
            title = f"{c.path}:{c.line}" + (f"  {c.symbol}" if c.symbol else "")
            body = c.text if max_chars is None else c.text[:max_chars]
            parts.append(f"### {title}\n{body.rstrip()}")
        return "\n\n".join(parts)


def serve(config: Path, root: Path) -> None:
    cfg = tomllib.loads(config.read_text(encoding="utf-8"))
    index = Index(root.resolve(), cfg)
    stdin = sys.stdin.buffer
    out = sys.stdout.buffer

    def send(msg: dict) -> None:
        out.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
        out.flush()

    for raw in stdin:
        if not raw.strip():
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            continue
        rid, method, params = req.get("id"), req.get("method"), req.get("params") or {}
        if rid is None:                                               # notification
            continue
        try:
            if method == "initialize":
                result = {"protocolVersion": params.get("protocolVersion", "2025-06-18"),
                          "capabilities": {"tools": {}},
                          "serverInfo": {"name": "memlab", "version": "0.1.0"}}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                name, a = params.get("name"), params.get("arguments") or {}
                if name == "search_code":
                    text = index.search_code(a["query"], min(int(a.get("budget_tokens", 6000)), 20000))
                elif name == "search_decisions":
                    text = index.search_decisions(a["query"], min(int(a.get("k", 5)), 15))
                elif name == "add_note":
                    text = index.add_note(a["summary"], a.get("body", ""), a.get("type", "decision"),
                                          a.get("tags") or [])
                elif name == "explain":
                    text = index.explain(a["name"])
                elif name == "find_path":
                    text = index.find_path(a["a"], a["b"])
                else:
                    raise ValueError(f"unknown tool {name}")
                result = {"content": [{"type": "text", "text": text}], "isError": False}
            elif method == "ping":
                result = {}
            else:
                send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"no method {method}"}})
                continue
            send({"jsonrpc": "2.0", "id": rid, "result": result})
        except Exception as e:
            if method == "tools/call":
                send({"jsonrpc": "2.0", "id": rid,
                      "result": {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True}})
            else:
                send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": str(e)}})
