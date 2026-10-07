"""Claude Code hooks: what reaches a session without a tool call.

  session-start — newest decision records of the repo (a digest) and a warning when the
                  last index build for it failed; prints nothing when there is neither.
  prompt-hint   — one line pointing at memlab when a long task prompt arrives and the
                  repo's server has had no tool call for a while; silent otherwise.
  session-end   — usage log: for each memlab call of the session, which shown files the session
                  edited afterwards (logs/usage.jsonl), a non-circular source of eval cases.

Run as a file (`python <checkout>/memlab/hooks.py session-start`), standard library only:
a hook starts on every session and prompt, the numpy import of the CLI is too slow for that.
`install` merges all three into ~/.claude/settings.json, replacing only its own entries.
"""
from __future__ import annotations

import datetime
import fnmatch
import json
import os
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None, ""):                     # run as a file by the hook command
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from memlab import config  # noqa: E402

LOG = Path(os.environ.get("MEMLAB_LOG") or config.home() / "logs" / "calls.jsonl")
STATE = LOG.parent / "hint-state.json"
USAGE = LOG.parent / "usage.jsonl"
EDIT_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")
DIGEST_N = 5
HINT_MIN_CHARS = int(os.environ.get("MEMLAB_HINT_MIN_CHARS", "200"))
HINT_QUIET_MIN = int(os.environ.get("MEMLAB_HINT_QUIET_MIN", "15"))       # no memlab call for this long
HINT_COOLDOWN_MIN = int(os.environ.get("MEMLAB_HINT_COOLDOWN_MIN", "10"))
HINT = ("memlab: before working on this, call search_decisions and search_code on the topic "
        "(prior decisions, path:line), then read the sources they point to.")


def _git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, encoding="utf-8",
                       stdin=subprocess.DEVNULL)
    return r.stdout if r.returncode == 0 else ""


def roots(cwd: Path) -> list[str]:
    """Roots a server for this session may have indexed: the cwd and, in a worktree, the main checkout."""
    out = [os.path.normcase(str(cwd.absolute()))]
    common = _git(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir").strip()
    if common:
        main = os.path.normcase(str(Path(common).parent))
        if main not in out:
            out.append(main)
    return out


def log_tail(max_bytes: int = 512_000) -> list[dict]:
    try:
        with open(LOG, "rb") as f:
            start = max(0, f.seek(0, 2) - max_bytes)
            f.seek(start)
            lines = f.read().decode("utf-8", "replace").splitlines()[1 if start else 0:]   # cut line
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def _mine(events: list[dict], rs: list[str]) -> list[dict]:
    return [e for e in events if os.path.normcase(e.get("root", "")) in rs]


def decision_globs(root: Path) -> list[str]:
    globs = list(config.DEFAULTS["channels"]["decisions"])
    local = root / ".memlab.toml"
    if local.exists():
        globs = tomllib.loads(local.read_text(encoding="utf-8")).get("channels", {}).get("decisions", globs)
    return globs + ["docs/notes/*"]


def _summary(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]
    except OSError:
        return ""
    for ln in lines:
        if ln.startswith("summary:"):
            s = ln[8:].strip()
            try:
                return json.loads(s) if s.startswith('"') else s.strip("'")
            except ValueError:
                return s
    return next((ln.lstrip("# ").strip() for ln in lines if ln.startswith("# ")), "")


def digest(root: Path) -> list[str]:
    """Newest DIGEST_N committed decision records: date, path, one-line summary."""
    globs = decision_globs(root)
    seen, out, date = set(), [], ""
    for ln in _git(root, "log", "--format=@%cs", "--name-only", "--diff-filter=AM", "-n", "200").splitlines():
        if ln.startswith("@"):
            date = ln[1:]
        elif ln and ln not in seen and any(fnmatch.fnmatch(ln, g) for g in globs) and (root / ln).is_file():
            seen.add(ln)
            out.append(f"- {date} {ln} — {_summary(root / ln)[:160]}")
            if len(out) == DIGEST_N:
                break
    return out


def note_warnings(top: Path) -> list[str]:
    """Where notes can still go missing: uncommitted files in docs/notes of any worktree of the repo
    (lost when that worktree is removed), and a main checkout behind origin (its index misses notes
    merged elsewhere). Uses the last fetched origin refs, no network."""
    out = []
    wts = [Path(ln[9:]) for ln in _git(top, "worktree", "list", "--porcelain").splitlines()
           if ln.startswith("worktree ")] or [top]
    count = lambda wt: sum(1 for ln in _git(wt, "status", "--porcelain", "--", "docs/notes").splitlines() if ln.strip())
    with ThreadPoolExecutor(16) as pool:                  # fitness has ~80 worktrees: 6 s one by one
        counts = list(pool.map(count, wts))
    for wt, n in zip(wts, counts):
        if n:
            out.append(f"{n} uncommitted file(s) in {wt.as_posix()}/docs/notes — commit them with their change, "
                       "or they are lost when that worktree is removed")
    remote = _git(top, "symbolic-ref", "--short", "refs/remotes/origin/HEAD").strip()     # e.g. origin/main
    main = wts[0]                                                                          # the main checkout
    if remote and _git(main, "branch", "--show-current").strip() == remote.split("/", 1)[-1]:
        behind = _git(main, "rev-list", "--count", f"HEAD..{remote}").strip()
        if behind not in ("", "0"):
            out.append(f"the main checkout {main.as_posix()} is {behind} commit(s) behind {remote} "
                       "(as of the last fetch) — its memlab index misses notes merged there until a pull")
    return out


def session_start(data: dict) -> str:
    cwd = Path(data.get("cwd") or os.getcwd())
    top = _git(cwd, "rev-parse", "--show-toplevel").strip()
    root = Path(top) if top else cwd
    parts = []
    lines = digest(root) if top else []
    if lines:
        parts.append("memlab — newest decision records here (search_decisions finds the rest):\n" + "\n".join(lines))
    rs = roots(root)
    if len(rs) > 1:      # a linked worktree: the desktop app starts memlab in the main checkout
        parts.append(f'memlab: this session works in the git worktree {Path(top).as_posix()}, while the memlab '
                     f'server may index the main checkout. Pass root="{Path(top).as_posix()}" to every memlab tool '
                     "(search_code, search_decisions, explain, find_path, add_note) to search and write this branch.")
    warn = note_warnings(root) if top else []
    if warn:
        parts.append("memlab warning — decision notes at risk (tell the user):\n" + "\n".join(f"- {w}" for w in warn))
    builds = [e for e in _mine(log_tail(), rs) if e.get("event") in ("ready", "build_failed")]
    if builds and builds[-1]["event"] == "build_failed":
        parts.append(f"memlab warning: the last index build for this repo failed ({builds[-1]['ts']}): "
                     f"{builds[-1].get('error', '')[:300].rstrip('.')}. memlab tools may answer with that error.")
    return "\n\n".join(parts)


def prompt_hint(data: dict, now: datetime.datetime | None = None) -> str:
    prompt = data.get("prompt") or ""
    if len(prompt) < HINT_MIN_CHARS or prompt.lstrip().startswith("/"):
        return ""
    now = now or datetime.datetime.now()
    rs = roots(Path(data.get("cwd") or os.getcwd()))
    calls = [e for e in _mine(log_tail(), rs) if e.get("event") == "call"]
    if calls and now - datetime.datetime.fromisoformat(calls[-1]["ts"]) < datetime.timedelta(minutes=HINT_QUIET_MIN):
        return ""
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    last = state.get(rs[0])
    if last and now - datetime.datetime.fromisoformat(last) < datetime.timedelta(minutes=HINT_COOLDOWN_MIN):
        return ""
    state[rs[0]] = now.isoformat(timespec="seconds")
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass
    return HINT


def _result_text(content) -> str:
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return content if isinstance(content, str) else ""


def usage(transcript: Path, root: Path) -> list[dict]:
    """One record per memlab search in the transcript: the files it showed and those of them the
    session edited after it (`used`). Edits made through Bash are not seen."""
    # ponytail: Edit/Write tools only; sed or scripted edits would need a git diff per call
    calls, results, edits = {}, {}, []                 # edits: (order, repo-relative path)
    with open(transcript, encoding="utf-8", errors="replace") as f:
        for n, ln in enumerate(f):
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            content = (e.get("message") or {}).get("content")
            for it in content if isinstance(content, list) else []:
                if not isinstance(it, dict):
                    continue
                if it.get("type") == "tool_use":
                    name, args = it.get("name", ""), it.get("input") or {}
                    if name.startswith("mcp__memlab__search_"):
                        calls[it.get("id")] = (n, e.get("timestamp", ""), name.rsplit("__", 1)[-1], args.get("query", ""))
                    elif name in EDIT_TOOLS and args.get("file_path"):
                        try:
                            rel = Path(args["file_path"]).absolute().relative_to(root).as_posix()
                        except ValueError:
                            continue                   # outside the repo
                        edits.append((n, rel))
                elif it.get("type") == "tool_result" and it.get("tool_use_id") in calls:
                    results[it["tool_use_id"]] = _result_text(it.get("content"))
    out = []
    for cid, (n, ts, tool, query) in calls.items():
        shown = list(dict.fromkeys(ln[4:].split(":")[0] for ln in results.get(cid, "").splitlines()
                                   if ln.startswith("### ")))
        later = {p for m, p in edits if m > n}
        out.append({"ts": ts, "tool": tool, "query": query, "shown": shown,
                    "used": [p for p in shown if p in later]})
    return out


def session_end(data: dict) -> str:
    cwd = Path(data.get("cwd") or os.getcwd())
    top = _git(cwd, "rev-parse", "--show-toplevel").strip()
    path = data.get("transcript_path")
    if not top or not path or not Path(path).exists():
        return ""
    recs = usage(Path(path), Path(top))
    if recs:
        try:
            with open(USAGE, "a", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps({"session": data.get("session_id"), "root": top, **r},
                                       ensure_ascii=False) + "\n")
        except OSError:
            pass
    return ""                                          # SessionEnd output is not shown anyway


def install(settings: Path | None = None) -> str:
    """Add both hooks to the user's Claude Code settings; earlier memlab entries are replaced."""
    settings = settings or Path.home() / ".claude" / "settings.json"
    data = json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else {}
    if settings.exists():
        settings.with_suffix(".json.bak").write_text(settings.read_text(encoding="utf-8"), encoding="utf-8")
    me = Path(__file__).absolute().as_posix()
    hooks = data.setdefault("hooks", {})
    for event, verb in (("SessionStart", "session-start"), ("UserPromptSubmit", "prompt-hint"),
                        ("SessionEnd", "session-end")):
        kept = [g for g in hooks.get(event, [])
                if not any("memlab/hooks.py" in h.get("command", "") for h in g.get("hooks", []))]
        kept.append({"hooks": [{"type": "command", "command": f'"{Path(sys.executable).as_posix()}" "{me}" {verb}',
                                "timeout": 15}]})
        hooks[event] = kept
    settings.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return f"installed SessionStart + UserPromptSubmit + SessionEnd hooks in {settings} (backup: settings.json.bak)"


def main(argv: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    verb = argv[0] if argv else ""
    if verb == "install":
        print(install())
        return
    if verb == "notes":                               # for /memlab status: python hooks.py notes [cwd]
        top = _git(Path(argv[1] if len(argv) > 1 else os.getcwd()), "rev-parse", "--show-toplevel").strip()
        print("\n".join(note_warnings(Path(top))) if top else "", end="")
        return
    try:
        data = json.loads(sys.stdin.read() or "{}")
        text = {"session-start": session_start, "prompt-hint": prompt_hint, "session-end": session_end}[verb](data)
    except Exception:            # a hook must never get in the way of a session
        return
    if text:
        print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
