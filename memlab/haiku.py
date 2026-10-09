"""Claude Haiku 5.5 as a listwise judge of search_decisions candidates (H17, eval only).

One call per query through the Claude Code CLI on the owner's plan (no API key on this machine):
the model sees the question and up to 10 decision records and scores each 0..1. Only the query and
decision records are sent. The prompt is pre-registered in README stage 14; do not tune it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

MODEL = "claude-haiku-5-5"
TOP = 10
NO_MATCH = 0.5                  # best score below this: no record answers the question
MAX_CHARS = 4000

SYSTEM = ("You judge search results for a developer's question about their own software project. You get "
          "the question and up to 10 decision records (ADRs, notes, postmortems), each with an id. For each "
          "record give a score from 0 to 1: how clearly the record itself states the decision, conclusion or "
          "fix that answers this specific question. 1 = it states exactly that; 0.5 = it is about the same "
          "matter but does not state the answer; 0 = a different matter, only a related topic, or a passing "
          "mention. Compare the records: usually at most one or two state the answer. Questions and records "
          "may be Russian or English; a question may be a commit subject rather than a question.")

SCHEMA = {"type": "object", "required": ["scores"], "properties": {"scores": {"type": "array", "items": {
    "type": "object", "required": ["id", "score"],
    "properties": {"id": {"type": "integer"}, "score": {"type": "number"}}}}}}

_CWD = Path(tempfile.gettempdir()) / "memlab-h17"   # empty folder: no CLAUDE.md, no repo, no project hooks


def message(query: str, records: list[tuple[str, str]]) -> str:
    return json.dumps({"question": query, "records": [{"id": i, "path": p, "text": t[:MAX_CHARS]}
                                                      for i, (p, t) in enumerate(records)]}, ensure_ascii=False)


def scores_from(out: dict, n: int) -> list[float]:
    """Scores in record order from the CLI's json output; a record without a score is 0."""
    if out.get("is_error") or not isinstance(out.get("structured_output"), dict):
        raise RuntimeError(f"no structured output: {str(out.get('result'))[:200]}")
    got = {int(s["id"]): float(s["score"]) for s in out["structured_output"]["scores"] if 0 <= int(s["id"]) < n}
    return [min(max(got.get(i, 0.0), 0.0), 1.0) for i in range(n)]


def judge(query: str, records: list[tuple[str, str]], timeout: float = 180) -> tuple[list[float], dict]:
    """A score per (path, text) record and the call's usage ({input_tokens, output_tokens, cost, models})."""
    _CWD.mkdir(exist_ok=True)
    (_CWD / "mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")
    env = dict(os.environ, MEMLAB_HINT_MIN_CHARS="100000000")   # keep memlab's own prompt hint out of the judge
    cmd = [shutil.which("claude") or "claude", "-p", "--model", MODEL, "--system-prompt", SYSTEM,
           "--json-schema", json.dumps(SCHEMA), "--output-format", "json", "--tools", "",
           "--strict-mcp-config", "--mcp-config", str(_CWD / "mcp.json"),
           "--disable-slash-commands", "--no-session-persistence"]
    # the prompt goes through stdin: claude.CMD on Windows would mangle %, ^ and & in an argument
    r = subprocess.run(cmd, cwd=_CWD, env=env, input=message(query, records), capture_output=True, text=True,
                       encoding="utf-8", timeout=timeout)
    out = json.loads(r.stdout)
    u = out.get("usage") or {}
    usage = {"input_tokens": u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
             + u.get("cache_creation_input_tokens", 0),
             "output_tokens": u.get("output_tokens", 0), "cost": out.get("total_cost_usd", 0.0),
             "models": sorted(out.get("modelUsage") or {})}
    return scores_from(out, len(records)), usage
