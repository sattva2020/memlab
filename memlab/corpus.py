"""Corpus: git-tracked text files of a project, cut into retrievable chunks.

A chunk is the unit a retriever ranks and a budget pays for. Code is cut at
top-level declarations (so a chunk id carries the symbol name), markdown at
headings, everything else into fixed windows. Long chunks are split into parts.
"""
from __future__ import annotations

import fnmatch
import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 6000          # ~1.5k tokens; longer chunks are split into parts
WINDOW = 3000             # window size for unstructured text

CODE_EXT = {".dart", ".ts", ".tsx", ".js", ".jsx", ".py", ".kt", ".swift", ".astro"}
MD_EXT = {".md", ".mdx"}
TEXT_EXT = {".yml", ".yaml", ".json", ".sql", ".sh", ".toml", ".txt", ".html", ".css"}

# Top-level declarations (column 0). Group 1 = symbol name.
DECL = re.compile(
    r"^(?:@\w+(?:\([^)]*\))?\s*)*"
    r"(?:export\s+)?(?:default\s+)?(?:abstract\s+|sealed\s+|final\s+|base\s+|interface\s+|data\s+|async\s+|private\s+|public\s+)*"
    r"(?:class|enum|mixin|extension(?:\s+type)?|typedef|interface|type|object|struct|protocol|function|def|fun|func)\s+"
    r"([A-Za-z_$][\w$]*)",
    re.M)
# Top-level variables/constants, optionally typed: `const Type name =`, `final name =`, `export const name =`.
VAR = re.compile(r"^(?:export\s+)?(?:const|final|var|let|val)\s+(?:[\w<>?,.\[\] ]+?\s+)?([A-Za-z_$][\w$]*)\s*[=:;]", re.M)
# Dart/Kotlin/TS top-level functions written as `Type name(` starting at column 0 (no indentation).
FUNC = re.compile(r"^(?:[A-Za-z_][\w<>\[\]?,. ]*\s)?([a-z_$][\w$]*)\s*(?:<[^>]*>)?\(", re.M)
HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$", re.M)


@dataclass(frozen=True)
class Chunk:
    id: str        # path, path::Symbol, path#slug, path@n (+ @part)
    path: str
    symbol: str    # "" for file-level / non-code chunks
    text: str
    line: int = 1  # 1-based line of the chunk's first character in its file

    @property
    def tokens(self) -> int:
        return max(1, len(self.text) // 4)


def slug(s: str) -> str:
    return re.sub(r"[^\w]+", "-", s.lower()).strip("-")[:60] or "section"


def _split(cid: str, path: str, symbol: str, text: str, line: int = 1) -> list[Chunk]:
    if len(text) <= MAX_CHARS:
        return [Chunk(cid, path, symbol, text, line)]
    return [Chunk(f"{cid}@{i}", path, symbol, text[o:o + MAX_CHARS], line + text.count("\n", 0, o))
            for i, o in enumerate(range(0, len(text), MAX_CHARS))]


def _cut(text: str, starts: list[tuple[int, str]]) -> list[tuple[str, str, int]]:
    """Cut text at (offset, name) marks -> (name, body, first line); the prefix is named ''."""
    starts = sorted(set(starts))
    out, prev, name = [], 0, ""
    for off, nxt in starts:
        if text[prev:off].strip():
            out.append((name, text[prev:off], text.count("\n", 0, prev) + 1))
        prev, name = off, nxt
    if text[prev:].strip():
        out.append((name, text[prev:], text.count("\n", 0, prev) + 1))
    return out


def chunk_file(path: str, text: str) -> list[Chunk]:
    ext = Path(path).suffix.lower()
    chunks: list[Chunk] = []
    if ext in CODE_EXT:
        marks = [(m.start(), m.group(1)) for m in DECL.finditer(text)]
        marks += [(m.start(), m.group(1)) for m in VAR.finditer(text) if m.start() not in {o for o, _ in marks}]
        if ext in {".dart", ".kt", ".ts", ".tsx"}:
            taken = {o for o, _ in marks}
            marks += [(m.start(), m.group(1)) for m in FUNC.finditer(text)
                      if m.start() not in taken and m.group(1) not in {"if", "for", "while", "switch", "return"}]
        seen: dict[str, int] = {}
        for name, body, line in _cut(text, marks):
            key = name or ""
            n = seen.get(key, 0); seen[key] = n + 1
            sym = key if n == 0 else f"{key}~{n}"
            cid = f"{path}::{sym}" if sym else path
            chunks += _split(cid, path, sym, body, line)
    elif ext in MD_EXT:
        marks = [(m.start(), slug(m.group(2))) for m in HEADING.finditer(text)]
        seen = {}
        for name, body, line in _cut(text, marks):
            n = seen.get(name, 0); seen[name] = n + 1
            s = name if n == 0 else f"{name}-{n}"
            chunks += _split(f"{path}#{s}" if s else path, path, "", body, line)
    else:
        for i, o in enumerate(range(0, len(text), WINDOW)):
            chunks.append(Chunk(f"{path}@{i}" if o else path, path, "", text[o:o + WINDOW],
                                text.count("\n", 0, o) + 1))
    return chunks


def list_files(root: Path, include: list[str], exclude: list[str], max_bytes: int,
               keep_always: list[str] = ()) -> list[str]:
    """Text files git knows or would track (untracked but not ignored: a note written this
    session counts before its commit); `keep_always` globs override `exclude`."""
    out = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard"],
                         capture_output=True,
                         text=True, encoding="utf-8", check=True).stdout.splitlines()
    keep = []
    for p in out:
        if Path(p).suffix.lower() not in CODE_EXT | MD_EXT | TEXT_EXT:
            continue
        forced = any(fnmatch.fnmatch(p, g) for g in keep_always)
        if not forced and include and not any(fnmatch.fnmatch(p, g) for g in include):
            continue
        if not forced and any(fnmatch.fnmatch(p, g) for g in exclude):
            continue
        f = root / p
        if f.is_file() and f.stat().st_size <= max_bytes:
            keep.append(p)
    return keep


def build(root: Path, include: list[str], exclude: list[str], max_bytes: int,
          keep_always: list[str] = ()) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in list_files(root, include, exclude, max_bytes, keep_always):
        try:
            text = (root / p).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        chunks += chunk_file(p, text)
    return chunks


def fingerprint(chunks: list[Chunk]) -> str:
    h = hashlib.sha1()
    for c in chunks:
        h.update(c.id.encode()); h.update(b"\0"); h.update(c.text.encode())
    return h.hexdigest()[:12]
