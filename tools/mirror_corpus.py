"""Copy exactly the memlab corpus files of a project into a directory (for external tools
that index a folder, e.g. `graphify extract`), keeping repository-relative paths.

  python tools/mirror_corpus.py projects/healbot.toml /path/to/mirror
"""
import shutil
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memlab import corpus  # noqa: E402

cfg_path, dest = Path(sys.argv[1]), Path(sys.argv[2])
cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
root, c = Path(cfg["root"]), cfg.get("corpus", {})
files = corpus.list_files(root, c.get("include", []), c.get("exclude", []),
                          c.get("max_bytes", 400_000), c.get("keep_always", []))
for f in files:
    (dest / f).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / f, dest / f)
print(f"{len(files)} files -> {dest}")
