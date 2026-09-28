"""Assemble the hard decision set from subagent questions, applying the pre-registered filter.

  python scripts/build_hard_set.py <config.toml> <questions_dir>
Drops a question when >=60% of its content tokens (stopwords removed, Russian stems) occur in
its gold document. Writes datasets/<name>/hard-decisions.json and prints the dropped ones.
"""
import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memlab.retrieve import query_tokens, tokenize  # noqa: E402

cfg_path, qdir = Path(sys.argv[1]), Path(sys.argv[2])
name, root = cfg_path.stem, Path(tomllib.loads(cfg_path.read_text(encoding="utf-8"))["root"])
qs = [q for f in sorted(qdir.glob(f"{name}-q*.json")) for q in json.loads(f.read_text(encoding="utf-8"))]
cases, dropped = [], []
for i, q in enumerate(qs):
    toks = query_tokens(q["query"], True, True)
    doc = set(tokenize((root / q["path"]).read_text(encoding="utf-8", errors="replace"), True))
    ov = sum(t in doc for t in toks) / max(len(toks), 1)
    (dropped if ov >= 0.6 else cases).append(
        {"id": f"hard-{name}-{i:02d}", "query": q["query"], "gold": [{"path": q["path"]}], "overlap": round(ov, 2)})
out = Path(f"datasets/{name}/hard-decisions.json")
out.write_text("[\n" + ",\n".join(" " + json.dumps(c, ensure_ascii=False) for c in cases) + "\n]\n", encoding="utf-8")
print(f"{name}: {len(qs)} questions, kept {len(cases)}, dropped {len(dropped)}; "
      f"mean overlap kept {sum(c['overlap'] for c in cases) / max(len(cases), 1):.2f}")
for d in dropped:
    print(f"  dropped {d['overlap']:.2f}  {d['query']}")
