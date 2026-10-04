"""TypeSafe Jev as a judge of search_decisions candidates (H16).

One Noul per (question, record) pair, as in TypeSafe's re-ranking cookbook: does the record state
the decision the question asks about, or is it only on a related topic? The nouls re-order the
dense shortlist; when none reaches NO_MATCH the tool answers that no record was found.
Only the query and decision records are sent, never code. Any failure raises; the caller keeps
its local ranking.
"""
from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"            # pinned: NO_MATCH was registered against this version
PRICE = 0.042e-6                # $ per input token, output free (docs.typesafe.ai/models, 2026-10-04)
TOP = 10                        # documents of the dense shortlist that Jev re-orders
NO_MATCH = 0.5                  # best noul below this: no record answers the question
MAX_CHARS = 4000
KEY_FILE = Path.home() / ".memlab" / ".env"   # user-level, outside every repository

QUESTION = {
    "type": "noul",
    "instructions": ("A developer asks `question` about their software project. Does `record` state the "
                     "decision, conclusion or fix that answers this specific question?"),
    "criteria": {
        "true": ("The record states what was decided, concluded, changed or fixed about the specific "
                 "matter `question` asks about."),
        "false": ("The record is only on a related topic, mentions the matter in passing, or covers a "
                  "different decision in the same area."),
    },
}


def api_key() -> str | None:
    """TYPESAFE_API_KEY from the environment, else from ~/.memlab/.env."""
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    try:
        for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "TYPESAFE_API_KEY":
                return value.strip().strip("'\"") or None
    except OSError:
        pass
    return None


def _ask(key: str, query: str, path: str, text: str, timeout: float) -> tuple[float, int]:
    body = {"model": MODEL, "questions": {"answers": QUESTION},
            "state": {"question": query, "record": {"path": path, "text": text[:MAX_CHARS]}}}
    req = urllib.request.Request(URL, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                  "User-Agent": "memlab"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return float(out["answers"]["answers"]["noul"]), int(out["usage"]["input_tokens"])


def judge(query: str, records: list[tuple[str, str]], key: str, timeout: float = 4.0) -> tuple[list[float], int]:
    """A noul per (path, text) record, all requests in parallel, and the input tokens used."""
    with ThreadPoolExecutor(max(len(records), 1)) as pool:
        res = list(pool.map(lambda r: _ask(key, query, r[0], r[1], timeout), records))
    return [n for n, _ in res], sum(t for _, t in res)
