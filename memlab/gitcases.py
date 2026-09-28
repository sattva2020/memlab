"""Retrieval cases mined from git history (task-localization style, cf. SWE-bench).

query = the commit subject (what the developer set out to do), conventional-commit type
        and trailing version tags stripped, scope kept (a developer knows the area);
gold  = the source files the commit changed that still exist in the corpus.

Mined cases are independent of whoever runs the evaluation: nobody chooses the wording
or the answer. Kept only: feat/fix/refactor/perf commits touching 1..max_files source
files, excluding tests and generated code.
"""
from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path

from .corpus import CODE_EXT

TYPE = re.compile(r"^(feat|fix|refactor|perf)(\(([^)]*)\))?!?:\s*(.+)$", re.I)
TAIL = re.compile(r"\s*\((v\d+[\w.]*|#\d+)\)\s*$")
TEST = ("*test*", "*spec*", "*/__tests__/*", "*/__mocks__/*", "*.g.dart", "*.freezed.dart",
        "*/generated/*", "*.d.ts")


ANY_TYPE = re.compile(r"^(\w+)(\(([^)]*)\))?!?:\s*(.+)$")
SKIP_DECISION = re.compile(r"versionCode|^release|\brelease\b.*\bpatch", re.I)
ADR_PREFIX = re.compile(r"^ADR-\d+\s*[—–-]\s*", re.I)


def mine_decisions(root: Path, corpus_paths: set[str], decision_globs: list[str], n: int,
                   min_words: int = 4, max_docs: int = 2,
                   skip_scopes: tuple[str, ...] = ("amg", "release")) -> list[dict]:
    """Decision cases: a commit that ADDED a decision document (ADR, postmortem).

    query = the commit subject (what was being done), gold = the added decision docs.
    """
    log = subprocess.run(["git", "-C", str(root), "log", "--no-merges", "--diff-filter=A",
                          "--format=%x00%H%x09%s", "--name-only"],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    cases, seen = [], set()
    for block in log.split("\x00")[1:]:
        head, *files = [x for x in block.splitlines() if x.strip()]
        sha, subject = head.split("\t", 1)
        gold = [f for f in files if f in corpus_paths and any(fnmatch.fnmatch(f, g) for g in decision_globs)]
        if not 1 <= len(gold) <= max_docs or SKIP_DECISION.search(subject):
            continue          # bulk memory-store commits carry no question
        m = ANY_TYPE.match(subject.strip())
        scope, text = (m.group(3) or "", m.group(4)) if m else ("", subject)
        if scope in skip_scopes:
            continue
        text = ADR_PREFIX.sub("", TAIL.sub("", text).strip())
        query = f"{scope}: {text}" if scope and scope not in {"docs", "patches"} else text
        if len(re.findall(r"\w+", text)) < min_words or query.lower() in seen:
            continue
        seen.add(query.lower())
        cases.append({"id": f"dec-{sha[:8]}", "query": query, "gold": [{"path": f} for f in gold]})
        if len(cases) >= n:
            break
    return cases


def mine(root: Path, corpus_paths: set[str], since: str, n: int, max_files: int = 4,
         min_words: int = 4) -> list[dict]:
    log = subprocess.run(["git", "-C", str(root), "log", "--no-merges", f"--since={since}",
                          "--format=%x00%H%x09%s", "--name-only"],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    cases, seen = [], set()
    for block in log.split("\x00")[1:]:
        head, *files = [x for x in block.splitlines() if x.strip()]
        sha, subject = head.split("\t", 1)
        m = TYPE.match(subject.strip())
        if not m:
            continue
        scope, text = m.group(3) or "", TAIL.sub("", m.group(4)).strip()
        query = f"{scope}: {text}" if scope else text
        if len(re.findall(r"\w+", text)) < min_words or query.lower() in seen:
            continue
        gold = [f for f in files
                if Path(f).suffix.lower() in CODE_EXT and f in corpus_paths
                and not any(fnmatch.fnmatch(f.lower(), t) for t in TEST)]
        if not 1 <= len(gold) <= max_files:
            continue
        seen.add(query.lower())
        cases.append({"id": f"git-{sha[:8]}", "query": query, "gold": [{"path": f} for f in gold]})
        if len(cases) >= n:
            break
    return cases
