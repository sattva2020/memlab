import subprocess

from memlab import config


def _repo(tmp_path, files):
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    return tmp_path


def test_defaults_detect_dart_ts_and_local_override(tmp_path):
    root = _repo(tmp_path, {
        "flutter/pubspec.yaml": "name: ai_fitness_coach_360\nversion: 1.0.0\n",
        "frontend/tsconfig.json": '{ // app\n "compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["./*"], "@lib/*": ["src/lib/*"],},}}',
        ".memlab.toml": '[channels]\ndecisions = ["notes/*"]\n',
    })
    cfg = config.load(root)
    assert cfg["graph"]["dart_package"] == "ai_fitness_coach_360"
    assert cfg["graph"]["dart_root"] == "flutter/lib"
    assert cfg["graph"]["aliases"] == {"@/": "frontend/", "@lib/": "frontend/src/lib/"}
    assert cfg["channels"]["decisions"] == ["notes/*"]
    assert "*.g.dart" in cfg["corpus"]["exclude"]


def test_explicit_config_is_used_as_is(tmp_path):
    f = tmp_path / "p.toml"
    f.write_text('root = "x"\n[graph]\ndart_package = "p"\n', encoding="utf-8")
    assert config.load(tmp_path, f) == {"root": "x", "graph": {"dart_package": "p"}}
