"""Project configuration without a config file.

`memlab serve --root .` must work in any repository, so everything a project config used to
spell out has a default or is read from the project itself:
  - corpus exclusions and decision folders: DEFAULTS below;
  - the Dart package and its lib/ root: from pubspec.yaml;
  - TypeScript import aliases (`@/…`): from tsconfig.json `paths`.
A `.memlab.toml` in the project root overrides any section; `--config FILE` replaces all of it
(the research configs in projects/ use that, unchanged).
"""
from __future__ import annotations

import copy
import json
import os
import posixpath
import re
import subprocess
import tomllib
from pathlib import Path

DEFAULTS = {
    "corpus": {
        "max_bytes": 400_000,
        "exclude": [
            ".claude/*", ".agents/*", ".codex/*", ".github/*", "node_modules/*", "*/node_modules/*",
            "*package-lock.json", "*pnpm-lock.yaml", "*yarn.lock", "*.lock", "*/pubspec.lock",
            "*.g.dart", "*.freezed.dart", "*.d.ts", "*.min.js", "*/generated/*",
        ],
    },
    "channels": {
        "decisions": ["docs/adr/*", "docs/decisions/*", "docs/postmortems/*", ".ai-factory/patches/*"],
    },
    "graph": {},
}


def home() -> Path:
    """Where caches and the journal live: MEMLAB_HOME, else the Claude Code plugin's data dir (set for
    the plugin's hooks), else the source checkout, else ~/.memlab."""
    for var in ("MEMLAB_HOME", "CLAUDE_PLUGIN_DATA"):
        if os.environ.get(var):
            return Path(os.environ[var])
    repo = Path(__file__).resolve().parents[1]
    return repo if (repo / ".git").exists() else Path.home() / ".memlab"


def _tracked(root: Path, name: str) -> list[str]:
    out = subprocess.run(["git", "-C", str(root), "ls-files", f"*{name}"], capture_output=True, text=True,
                         encoding="utf-8", stdin=subprocess.DEVNULL).stdout.split()
    return sorted((p for p in out if posixpath.basename(p) == name and "node_modules/" not in p),
                  key=lambda p: p.count("/"))


def _dart(root: Path) -> dict:
    for p in _tracked(root, "pubspec.yaml"):
        m = re.search(r"^name:\s*([\w]+)", (root / p).read_text(encoding="utf-8", errors="ignore"), re.M)
        if m:
            return {"dart_package": m[1], "dart_root": posixpath.join(posixpath.dirname(p), "lib")}
    return {}


def _ts_aliases(root: Path) -> dict:
    """`"@/*": ["./src/*"]` in dir/tsconfig.json → {"@/": "dir/src/"}; tsconfig is JSON with comments."""
    aliases: dict[str, str] = {}
    for p in _tracked(root, "tsconfig.json"):
        text = (root / p).read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r'("(?:\\.|[^"\\])*")|//[^\n]*|/\*.*?\*/', lambda m: m[1] or "", text, flags=re.S)
        text = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            opts = json.loads(text).get("compilerOptions", {})
        except (ValueError, AttributeError):
            continue
        base = posixpath.normpath(posixpath.join(posixpath.dirname(p), opts.get("baseUrl", ".")))
        for key, targets in (opts.get("paths") or {}).items():
            if key.endswith("/*") and targets and targets[0].endswith("/*"):
                target = posixpath.normpath(posixpath.join(base, targets[0][:-2]))
                aliases.setdefault(key[:-1], "" if target == "." else target + "/")
    return aliases


def load(root: Path, config: Path | None = None) -> dict:
    if config:
        return tomllib.loads(Path(config).read_text(encoding="utf-8"))
    cfg = copy.deepcopy(DEFAULTS)
    local = root / ".memlab.toml"
    if local.exists():
        for section, values in tomllib.loads(local.read_text(encoding="utf-8")).items():
            if isinstance(values, dict):
                cfg.setdefault(section, {}).update(values)
            else:
                cfg[section] = values
    g = cfg["graph"]
    if "dart_package" not in g:
        g.update(_dart(root))
    if "aliases" not in g:
        g["aliases"] = _ts_aliases(root)
    return cfg
