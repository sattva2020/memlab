"""Sample decision documents for a hard question set; write batches for question writers.

  python scripts/sample_decisions.py <config.toml> <n> <out_dir> [--seed 7] [--batch 10]
Documents under 400 characters are skipped (nothing to ask about). Documents that are gold in
older sets stay eligible: the question is new, and retrievers are not trained on the old sets.
"""
import argparse
import fnmatch
import json
import random
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memlab import corpus  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config"); ap.add_argument("n", type=int); ap.add_argument("out")
ap.add_argument("--seed", type=int, default=7); ap.add_argument("--batch", type=int, default=10)
a = ap.parse_args()
cfg = tomllib.loads(Path(a.config).read_text(encoding="utf-8"))
root, c = Path(cfg["root"]), cfg.get("corpus", {})
chunks = corpus.build(root, c.get("include", []), c.get("exclude", []), c.get("max_bytes", 400_000),
                      c.get("keep_always", []))
globs = cfg["channels"]["decisions"]
docs = sorted({x.path for x in chunks if any(fnmatch.fnmatch(x.path, g) for g in globs)})
name = Path(a.config).stem
pool = [p for p in docs if len((root / p).read_text(encoding="utf-8", errors="replace")) >= 400]
random.Random(a.seed).shuffle(pool)
pick = pool[:a.n]
print(f"{name}: {len(docs)} decision docs, {len(pool)} eligible, picked {len(pick)}")
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for i in range(0, len(pick), a.batch):
    items = [{"path": p, "text": (root / p).read_text(encoding="utf-8", errors="replace")[:4000]}
             for p in pick[i:i + a.batch]]
    (out / f"{name}-batch{i // a.batch}.json").write_text(json.dumps(items, ensure_ascii=False, indent=1),
                                                          encoding="utf-8")
