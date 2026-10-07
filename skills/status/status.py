"""memlab status: what the last server session did for this cwd and, in a worktree, its main checkout.

Reads only logs/calls.jsonl under memlab's home (MEMLAB_HOME / CLAUDE_PLUGIN_DATA, see config.home).
Usage: python status.py [cwd]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

PLUGIN = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(PLUGIN))
from memlab import config  # noqa: E402

LOG = config.home() / "logs" / "calls.jsonl"


def norm(p):
    return os.path.normcase(os.path.abspath(p))


def roots(cwd):
    out = [norm(cwd)]
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            cwd=cwd, capture_output=True, text=True).stdout.strip()
    if common and norm(Path(common).parent) not in out:
        out.append(norm(Path(common).parent))
    return out


def events():
    try:
        with open(LOG, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
    except OSError:
        return


def summary(evs, root):
    mine = [e for e in evs if norm(e.get("root", "")) == root]
    starts = [e for e in mine if e["event"] == "start"]
    if not starts:
        return f"{root}\n  no server start logged"
    s = starts[-1]
    sess = [e for e in mine if e.get("pid") == s["pid"] and e["ts"] >= s["ts"]]
    ready = next((e for e in sess if e["event"] == "ready"), None)
    failed = next((e for e in sess if e["event"] == "build_failed"), None)
    calls = [e for e in sess if e["event"] == "call"]
    errors = [e for e in calls if e.get("error")]
    if ready:
        state = f"ready: {ready['chunks']} chunks in {ready['seconds']}s"
    elif failed:
        state = f"BUILD FAILED: {failed['error'][:300]}"
    else:
        state = "started, no ready yet (building or hung)"
    lines = [root, f"  last start {s['ts']} pid {s['pid']} - {state}",
             f"  calls this session: {len(calls)}, errors: {len(errors)}"]
    lines += [f"    {e['ts']} {e['tool']}: {str(e['error'])[:200]}" for e in errors[-3:]]
    fails = sum(e["event"] == "build_failed" for e in mine)
    lines.append(f"  history: {len(starts)} starts, {fails} build failures")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cwd = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    evs = list(events())
    rs = roots(cwd)
    print(f"log: {LOG}\ncwd: {rs[0]}")
    if len(rs) > 1:
        print(f"main checkout: {rs[1]} (desktop worktree sessions may index this one)")
    for r in rs:
        print()
        print(summary(evs, r))
    warn = subprocess.run([sys.executable, str(PLUGIN / "memlab" / "hooks.py"), "notes", cwd],
                          capture_output=True, text=True, encoding="utf-8").stdout.strip()
    print("\nnotes at risk:\n  " + warn.replace("\n", "\n  ") if warn else "\nnotes at risk: none")
