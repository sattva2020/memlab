"""Paired comparison of one method across two runs (e.g. two embedders) using per-case scores.

  python scripts/compare.py results/a.json METHOD results/b.json METHOD
Prints, per metric cell, mean(b) - mean(a) with a 95% paired bootstrap interval.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from memlab.evaluate import bootstrap_diff  # noqa: E402

a_path, a_m, b_path, b_m = sys.argv[1:5]
a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (a_path, b_path))
assert a["case_ids"] == b["case_ids"], "runs must cover the same cases in the same order"
for cell in a["per_case"]:
    if cell.endswith("/symbol") or cell not in b["per_case"]:
        continue
    x, y = np.array(a["per_case"][cell][a_m]), np.array(b["per_case"][cell][b_m])
    d, lo, hi = bootstrap_diff(y, x)
    mark = " *" if lo > 0 or hi < 0 else ""
    print(f"{cell:12} {x.mean():.2f} -> {y.mean():.2f}  {d:+.4f} [{lo:+.4f},{hi:+.4f}]{mark}")
