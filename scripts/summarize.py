"""Compact table from result JSONs: method means and paired diffs vs a reference, chosen cells.

  python scripts/summarize.py REF_METHOD file1.json[:label] file2.json[:label] ... -- m1 m2 ...
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from memlab.evaluate import bootstrap_diff  # noqa: E402

CELLS = ("4000/file", "16000/file", "38000/file", "files@5", "files@10", "files@20")
args = sys.argv[1:]
ref = args[0]
split = args.index("--")
files, methods = args[1:split], args[split + 1:]
for spec in files:
    path, _, label = spec.partition(":")
    if not Path(path).exists():
        print(f"{label or path}: (not ready)")
        continue
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"## {label or path}  (n={d['cases']})")
    for m in [ref] + methods:
        row = []
        for c in CELLS:
            pc = d.get("per_case", {}).get(c, {})
            if m not in pc or ref not in pc:
                row.append("   —   ")
                continue
            x, r = np.array(pc[m]), np.array(pc[ref])
            if m == ref:
                row.append(f" {x.mean():.2f}  ")
            else:
                diff, lo, hi = bootstrap_diff(x, r)
                row.append(f"{diff:+.2f}{'*' if lo > 0 or hi < 0 else ' '} ")
        print(f"  {m[:44]:44} " + " ".join(row))
    print()
