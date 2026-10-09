"""MCP server (stdio, JSON-RPC 2.0, newline-delimited) with two project-memory tools.

  search_code      — the stage 3 system: hybrid (BM25 + dense) seed → graphify-style seeds →
                     PPR over df-capped symbol references with hubs blocked as transit →
                     code only → top 50 re-ordered by a multilingual cross-encoder (H15; GPU only).
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
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import config, corpus, embedder, evaluate, explore, graph, retrieve

E5 = "intfloat/multilingual-e5-small"
QWEN = "Qwen/Qwen3-Embedding-0.6B"
RERANKER = "BAAI/bge-reranker-v2-m3"
RERANK_TOP = 50
QWEN_INSTRUCT = ("Instruct: Given a question about a software project, retrieve the notes, "
                 "decision records and postmortems that answer it\nQuery: ")

ROOT = {"type": "string", "description": (
    "absolute path of the git worktree you work in, when it differs from the server's root (e.g. a "
    "desktop-app worktree session); the first call builds that worktree's index. Default: server root")}

TOOLS = [
    {"name": "search_code",
     "description": ("Find the code in this repository relevant to a task or question: files, "
                     "classes and functions with path:line. Use for 'where is X implemented', "
                     "'what calls Y', 'which files does this change touch'."),
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "the task or question, any language"},
         "budget_tokens": {"type": "integer", "description": "context to return (default 6000, max 20000)"},
         "root": ROOT},
         "required": ["query"]}},
    {"name": "search_decisions",
     "description": ("Find past decisions, session notes, ADRs and postmortems about this project: "
                     "why something was done, what was decided, what broke and how it was fixed."),
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string"},
         "k": {"type": "integer", "description": "documents to return (default 5, max 15)"},
         "root": ROOT},
         "required": ["query"]}},
    {"name": "add_note",
     "description": ("Record a decision, conclusion or postmortem so later sessions find it with "
                     "search_decisions. Writes a new markdown file under docs/notes/ (never edits an "
                     "existing one, so worktrees merge cleanly); commit it with your change."),
     "inputSchema": {"type": "object", "properties": {
         "summary": {"type": "string", "description": "one line: what was decided and why"},
         "body": {"type": "string", "description": "details: context, alternatives, file paths"},
         "type": {"type": "string", "enum": ["decision", "note", "postmortem"]},
         "tags": {"type": "array", "items": {"type": "string"}},
         "supersedes": {"type": "array", "items": {"type": "string"}, "description": (
             "repo-relative paths of earlier decision records this one replaces; they stay searchable, "
             "marked superseded and ranked after current ones")},
         "root": ROOT},
         "required": ["summary"]}},
    {"name": "explain",
     "description": ("Explain a symbol or file: where it is defined, which files use it and what it uses, "
                     "its code/doc/decision neighbours in the project graph, and related decisions."),
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "symbol (class/function) name, file path or file name"},
         "root": ROOT},
         "required": ["name"]}},
    {"name": "find_path",
     "description": "Shortest chain of imports/symbol references connecting two symbols or files.",
     "inputSchema": {"type": "object", "properties": {
         "a": {"type": "string"}, "b": {"type": "string"}, "root": ROOT}, "required": ["a", "b"]}},
]


NOTES_DIR = "docs/notes"
# Sent in `initialize`; MCP clients put it into the session's system prompt, so every project
# that has the server connected gets the usage rules without a line in its CLAUDE.md.
INSTRUCTIONS = (
    "memlab is this project's memory: an index of the repository's code and of its decision records "
    "(docs/adr, docs/notes, .ai-factory/patches). Use it before searching files by hand.\n"
    "- Starting a task or entering an unfamiliar area: search_decisions for prior decisions, ADRs and "
    "postmortems on the topic, and search_code to find the files and symbols (path:line); open sources "
    "point-wise from there. explain NAME shows a symbol's or file's neighbours; find_path A B links two.\n"
    "- Before stating a code fact from a result, read the source: the index is built from the working tree "
    "when the server starts, so files changed later in the session are not in it (results from such files "
    "are marked ⟨stale⟩ or ⟨deleted⟩).\n"
    "- When a decision, conclusion or postmortem about the project emerges, record it with add_note "
    "(one line summary + body with context and file paths) and commit the new docs/notes file with the change; "
    "when it replaces an earlier decision from search_decisions, list that file in supersedes. "
    "If you work in a git worktree other than the server's root, pass it as root to every memlab tool.")
# One JSON line per server start, index build and tool call: which servers hang, what agents
# actually ask, what came back. Outside every served project; user data, never committed.
LOG = Path(os.environ.get("MEMLAB_LOG") or config.home() / "logs" / "calls.jsonl")
_log_lock = threading.Lock()


def journal(event: str, **fields) -> None:
    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "pid": os.getpid(), "event": event, **fields}
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock, open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass                     # the journal must never break a tool call


def write_note(root: Path, summary: str, body: str = "", kind: str = "decision",
               tags: list[str] | None = None, now: datetime.datetime | None = None,
               supersedes: list[str] | None = None) -> str:
    """A new note file, one per call: date + slug + content hash, created exclusively."""
    now = now or datetime.datetime.now()
    slug = "-".join(re.findall(r"[^\W_]+", summary.lower()))[:60].strip("-") or "note"
    text = (f"---\ntype: {kind}\ncreated: '{now.isoformat(timespec='seconds')}'\n"
            f"tags: {json.dumps(tags or [], ensure_ascii=False)}\n"
            + (f"supersedes: {json.dumps(supersedes, ensure_ascii=False)}\n" if supersedes else "")
            + f"summary: {json.dumps(summary, ensure_ascii=False)}\n---\n# {summary}\n\n{body}".rstrip() + "\n")
    h = hashlib.sha1(text.encode()).hexdigest()[:8]
    rel = f"{NOTES_DIR}/{now:%Y-%m-%d}-{slug}-{h}.md"
    (root / NOTES_DIR).mkdir(parents=True, exist_ok=True)
    with open(root / rel, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return rel


def supersessions(root: Path, paths) -> dict[str, str]:
    """{old record: the record whose frontmatter `supersedes:` lists it}. Notes are append-only,
    so the replacement is declared by the new record, never written into the old one."""
    out = {}
    for p in paths:
        try:
            with open(root / p, encoding="utf-8", errors="replace") as f:
                head = [next(f, "") for _ in range(12)]
        except OSError:
            continue
        for ln in head:
            if ln.startswith("supersedes:"):
                try:
                    olds = json.loads(ln[11:])
                except ValueError:
                    olds = []
                for old in olds if isinstance(olds, list) else []:
                    out[str(old).replace("\\", "/").removeprefix("./")] = p
    return out


def note_root(root: Path, target: str | None) -> Path:
    """Where add_note writes: the server root, or a worktree of the same repository. The desktop app
    starts the server in the main checkout while the session works in a worktree, and its roots/list
    answers with the same main checkout, so the agent names the worktree itself."""
    if not target:
        return root
    t = Path(target).resolve()
    common = lambda p: subprocess.run(["git", "-C", str(p), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                      capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout.strip()
    if not t.is_dir() or not common(t) or Path(common(t)).resolve() != Path(common(root)).resolve():
        raise ValueError(f"root {target} is not a worktree of {root}")
    return t


def is_git(root: Path) -> bool:
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"], capture_output=True,
                       text=True, stdin=subprocess.DEVNULL)
    return r.returncode == 0 and r.stdout.strip() == "true"


def log(msg: str) -> None:
    print(f"[memlab] {msg}", file=sys.stderr, flush=True)


class Index:
    def __init__(self, root: Path, cfg: dict):
        self.root, self.cfg = root, cfg
        self.ready = threading.Event()
        self.error: str | None = None
        self.built = time.time()     # files read after this; a newer mtime means the chunk may be out of date
        self.fresh: set[str] = set()  # notes added through add_note: newer than the build, yet indexed
        self.superseded: dict[str, str] = {}
        self._started, self._lock = False, threading.Lock()
        if is_git(root):
            self._start()
        else:                        # app folders, scratch dirs (a user-scope server starts in every session):
            journal("deferred", root=str(root))   # build only when a tool is called
            log("not a git repository: the index is built on the first call")

    def _start(self) -> None:
        with self._lock:
            if not self._started:
                self._started = True
                threading.Thread(target=self._build, daemon=True).start()

    def _build(self) -> None:
        try:
            t = self.built = time.time()
            c = self.cfg.get("corpus", {})
            self.chunks = corpus.build(self.root, c.get("include", []), c.get("exclude", []),
                                       c.get("max_bytes", 400_000), c.get("keep_always", []))
            globs = self.cfg.get("channels", {}).get("decisions", []) + [f"{NOTES_DIR}/*"]
            self.is_dec = np.array([any(fnmatch.fnmatch(x.path, g) for g in globs) for x in self.chunks])
            self.is_code = np.array([Path(x.path).suffix.lower() in corpus.CODE_EXT for x in self.chunks])
            self.bm25 = retrieve.BM25([retrieve.tokenize(f"{x.path} {x.symbol} {x.text}") for x in self.chunks])
            cache = config.home() / self.cfg.get("cache", ".cache")  # never inside the served project
            self.dense = retrieve.Dense(self.chunks, E5, cache, query_prefix="query: ", doc_prefix="passage: ",
                                        encoder=embedder.Client(E5))
            gc = self.cfg.get("graph", {})
            self.graph = graph.Graph(self.chunks, gc.get("dart_package", ""), gc.get("dart_root", ""),
                                     aliases=gc.get("aliases", {}))
            self.dec_ids = np.flatnonzero(self.is_dec)
            dec_chunks = [self.chunks[i] for i in self.dec_ids]
            self.superseded = supersessions(self.root, dict.fromkeys(c.path for c in dec_chunks))
            self.files = explore.file_graph(self.chunks, self.graph, globs)
            self.dec_dense = retrieve.Dense(dec_chunks, QWEN, cache, query_prefix=QWEN_INSTRUCT, max_len=512,
                                            encoder=embedder.Client(QWEN, max_len=512))
            log(f"index ready: {len(self.chunks)} chunks, {len(self.dec_ids)} decision chunks, "
                f"re-embedded {self.dense.reembedded}+{self.dec_dense.reembedded}, {time.time() - t:.0f}s")
            self.dense.scores("warm up")        # start the shared embedder and load both models now,
            self.dec_dense.scores("warm up")    # not inside the first tool call (was 41 s after an idle exit)
            self.reranker = None if os.environ.get("MEMLAB_RERANK") == "0" else embedder.Client(RERANKER)
            if self.reranker and self.reranker.rerank("warm up", ["warm up"]) is None:
                self.reranker = None            # no GPU: keep the plain ranking
            journal("ready", root=str(self.root), chunks=len(self.chunks), seconds=round(time.time() - t, 1),
                    reembedded=self.dense.reembedded + self.dec_dense.reembedded)
        except Exception as e:  # surfaced to the caller instead of a silent dead server
            self.error = f"{type(e).__name__}: {e}"
            log(f"index build failed: {self.error}")
            journal("build_failed", root=str(self.root), error=self.error)
        finally:
            self.ready.set()

    def wait(self) -> None:
        self._start()
        self.ready.wait()
        if self.error:
            raise RuntimeError(f"memlab index failed to build: {self.error}")

    # ------------------------------------------------------------------ tools
    def search_code(self, query: str, budget: int) -> str:
        self.wait()
        if not self.chunks:
            return f"search_code: 0 chunks — nothing is indexed under {self.root}"
        qt = retrieve.query_tokens(query, True)
        bs = self.bm25.scores(qt)
        rb, rd = retrieve.rank(bs), retrieve.rank(self.dense.scores(query))
        seed = retrieve.rrf_scores(rb, rd)
        rh = retrieve.rank(seed)
        seeds = self._gfy_seeds(qt, bs, rd, rh)
        ranking = self.graph.ppr(seed, {"refs"}, transit_block=True, seed_ids=seeds)
        ranking = ranking[self.is_code[ranking]]                      # code only (H8); prose has its own tool
        if self.reranker:
            head = ranking[:RERANK_TOP]
            s = self.reranker.rerank(query, [self._pair_text(i) for i in head])
            if s is not None:
                ranking = np.concatenate([head[np.argsort(-s, kind="stable")], ranking[RERANK_TOP:]])
        picked = evaluate.select(ranking, self.chunks, budget)
        return self._render(picked, f"search_code: {len(picked)} chunks, ~{budget} tokens budget")

    def _pair_text(self, i: int) -> str:
        c = self.chunks[i]
        return f"{c.path} {c.symbol}" + chr(10) + c.text

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
        out.sort(key=lambda c: c.path in self.superseded)              # stable: superseded go last
        return self._render(out, f"search_decisions: {len(out)} documents", max_chars=1600)

    def add_note(self, summary: str, body: str, kind: str, tags: list[str], target: str | None = None,
                 supersedes: list[str] | None = None) -> str:
        """Write the note and make it searchable at once (decision channel only; the rest of
        the index picks it up at the next start)."""
        self.wait()
        dest = note_root(self.root, target)
        rel = write_note(dest, summary, body, kind, tags, supersedes=supersedes)
        self.fresh.add(rel)
        self.superseded.update(supersessions(dest, [rel]))
        new = corpus.chunk_file(rel, (dest / rel).read_text(encoding="utf-8"))
        n = len(self.chunks)
        self.chunks = self.chunks + new
        self.is_dec = np.concatenate([self.is_dec, np.ones(len(new), bool)])
        self.is_code = np.concatenate([self.is_code, np.zeros(len(new), bool)])
        self.dec_ids = np.concatenate([self.dec_ids, np.arange(n, n + len(new))])
        vecs = self.dec_dense.model.encode([f"{c.path} {c.symbol}\n{c.text}" for c in new],
                                           normalize_embeddings=True)
        old = self.dec_dense.emb
        self.dec_dense.emb = np.vstack([old, vecs.astype(np.float32)]) if len(old) else vecs.astype(np.float32)
        return f"saved {dest / rel} (searchable now; commit it with your change)"

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

    def _mark(self, path: str) -> str:
        """⟨deleted⟩ / ⟨stale⟩ when the file changed after the index read it: read it before relying on the chunk."""
        try:
            changed = (self.root / path).stat().st_mtime > self.built and path not in self.fresh
        except OSError:
            return "  ⟨deleted⟩"
        mark = f"  ⟨superseded by {self.superseded[path]}⟩" if path in self.superseded else ""
        return mark + ("  ⟨stale: file changed after indexing⟩" if changed else "")

    def _render(self, chunks: list[corpus.Chunk], header: str, max_chars: int | None = None) -> str:
        parts = [header]
        for c in chunks:
            title = f"{c.path}:{c.line}" + (f"  {c.symbol}" if c.symbol else "") + self._mark(c.path)
            body = c.text if max_chars is None else c.text[:max_chars]
            parts.append(f"### {title}\n{body.rstrip()}")
        return "\n\n".join(parts)


def serve(config_path: Path | None, root: Path) -> None:
    cfg = config.load(root.resolve(), config_path)
    # Read JSON-RPC from a private duplicate of stdin and point fd 0 at NUL before any work starts.
    # On Windows a thread that touches the process stdin (a DLL loading, a child `git`) blocks
    # while our main thread has a read pending on that pipe: the index never finished building.
    stdin = os.fdopen(os.dup(0), "rb")
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    sys.stdin = open(os.devnull)
    journal("start", root=str(root.resolve()), config=str(config_path) if config_path else None)
    index = Index(root.resolve(), cfg)
    worktrees: dict[Path, Index] = {}
    out = sys.stdout.buffer

    def pick(a: dict, build: bool = True) -> Index:
        """The index for the call's `root`: the server's, or a worktree's (validated by note_root,
        built in the background on first use; the shared vector store keeps unchanged chunks)."""
        dest = note_root(index.root, a.get("root"))
        if dest == index.root:
            return index
        if dest not in worktrees and build:
            journal("start", root=str(dest), config=str(config_path) if config_path else None, worktree_of=str(index.root))
            worktrees[dest] = Index(dest, config.load(dest, config_path))
        return worktrees.get(dest, index)

    def journal_call(name, a, t0, text=None, error=None, root=None):
        try:
            args = {k: (v[:300] if isinstance(v, str) else v) for k, v in a.items() if k != "body"}
            top = [ln[4:].split()[0] for ln in (text or "").splitlines() if ln.startswith("### ")][:10]
            journal("call", root=str(root or index.root), tool=name, args=args, ms=round(1000 * (time.time() - t0)),
                    top=top, error=error)
        except Exception:
            pass                 # malformed arguments are the caller's error, not a reason to stop serving

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
                          "serverInfo": {"name": "memlab", "version": "0.1.0"},
                          "instructions": INSTRUCTIONS}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                name, a = params.get("name"), params.get("arguments") or {}
                t0, ix = time.time(), index
                ix = pick(a, build=name != "add_note")   # a note alone does not justify building an index
                if name == "search_code":
                    text = ix.search_code(a["query"], min(int(a.get("budget_tokens", 6000)), 20000))
                elif name == "search_decisions":
                    text = ix.search_decisions(a["query"], min(int(a.get("k", 5)), 15))
                elif name == "add_note":
                    text = ix.add_note(a["summary"], a.get("body", ""), a.get("type", "decision"),
                                       a.get("tags") or [], a.get("root"), a.get("supersedes") or None)
                elif name == "explain":
                    text = ix.explain(a["name"])
                elif name == "find_path":
                    text = ix.find_path(a["a"], a["b"])
                else:
                    raise ValueError(f"unknown tool {name}")
                result = {"content": [{"type": "text", "text": text}], "isError": False}
                journal_call(name, a, t0, text, root=ix.root)
            elif method == "ping":
                result = {}
            else:
                send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"no method {method}"}})
                continue
            send({"jsonrpc": "2.0", "id": rid, "result": result})
        except Exception as e:
            if method == "tools/call":
                journal_call(params.get("name"), params.get("arguments") or {}, t0, error=f"{type(e).__name__}: {e}")
                send({"jsonrpc": "2.0", "id": rid,
                      "result": {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True}})
            else:
                send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": str(e)}})
    journal("exit", root=str(index.root))     # the client closed the session: a start without ready was not a hang
