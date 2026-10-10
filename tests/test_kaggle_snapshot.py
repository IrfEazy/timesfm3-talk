"""Offline tests: code snapshot creation, dataset discovery, commit-sha env fallback."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from tfm3lab import cellstore, manifest
from tfm3lab import kaggle as kg

ROOT = Path(__file__).resolve().parents[1]


def _git_repo(root: Path) -> str:
    def git(*a):
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *a],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    (root / "src/tfm3lab").mkdir(parents=True)
    (root / "src/tfm3lab/__init__.py").write_text("x = 1\n")
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    (root / ".gitignore").write_text("ignored.txt\n")
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    return git("rev-parse", "HEAD")


# --------------------------------------------------------------- snapshot


def test_snapshot_contains_tracked_files_and_commit_sha(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    sha = _git_repo(repo)
    (repo / "untracked.txt").write_text("u")
    (repo / "ignored.txt").write_text("i")
    archive, names = kg.create_code_snapshot(tmp_path / "out", repo)
    assert archive.name == f"tfm3lab-code-{sha[:12]}.tar.gz"
    assert "untracked.txt" not in names and "ignored.txt" not in names
    assert {"pyproject.toml", "src/tfm3lab/__init__.py", "COMMIT_SHA"} <= set(names)
    assert not any(n.startswith(".git/") or n == ".git" for n in names)
    dest = kg.materialize_code_snapshot(archive, tmp_path / "x")
    assert kg.read_snapshot_commit(dest) == (sha, True)


def test_snapshot_refuses_dirty_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    (repo / "pyproject.toml").write_text("changed\n")
    with pytest.raises(RuntimeError, match="dirty"):
        kg.create_code_snapshot(tmp_path / "out", repo)
    assert not (tmp_path / "out").exists()


def test_snapshot_cli_prints_file_list(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    sha = _git_repo(repo)
    real = kg.create_code_snapshot
    monkeypatch.setattr(kg, "create_code_snapshot", lambda out, repo_=None: real(out, repo))
    assert kg.main(["code-snapshot", "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "COMMIT_SHA" in out and sha[:12] in out
    with tarfile.open(tmp_path / "o" / f"tfm3lab-code-{sha[:12]}.tar.gz") as tar:
        assert "COMMIT_SHA" in tar.getnames()


# -------------------------------------------------------------- discovery


def test_find_input_dir_flat_and_nested(tmp_path):
    (tmp_path / "flat-slug").mkdir()
    (tmp_path / "datasets/owner/Nested-Slug").mkdir(parents=True)
    assert kg.find_input_dir("flat-slug", tmp_path) == tmp_path / "flat-slug"
    assert kg.find_input_dir("nested-slug", tmp_path) == tmp_path / "datasets/owner/Nested-Slug"
    with pytest.raises(FileNotFoundError, match="missing-slug"):
        kg.find_input_dir("missing-slug", tmp_path)


def test_find_code_snapshot_directory_and_archive(tmp_path):
    flat = tmp_path / "a"
    (flat / "tfm3lab-code-abc/src/tfm3lab").mkdir(parents=True)
    (flat / "tfm3lab-code-abc/pyproject.toml").write_text("")
    assert kg.find_code_snapshot("tfm3lab-code-abc", flat) == flat / "tfm3lab-code-abc"
    nested = tmp_path / "b"
    (nested / "datasets/me/tfm3lab-code-abc").mkdir(parents=True)
    arc = nested / "datasets/me/tfm3lab-code-abc/tfm3lab-code-abc123.tar.gz"
    arc.write_bytes(b"")
    assert kg.find_code_snapshot("tfm3lab-code-abc", nested) == arc
    with pytest.raises(FileNotFoundError):
        kg.find_code_snapshot("nope", tmp_path / "empty")


def test_find_code_snapshot_slug_missing_does_not_fall_back(tmp_path):
    other = tmp_path / "tfm3lab-code-other"
    (other / "src/tfm3lab").mkdir(parents=True)
    (other / "pyproject.toml").write_text("")
    with pytest.raises(FileNotFoundError):
        kg.find_code_snapshot("tfm3lab-code-aaaaaaaaaaaa", tmp_path)


def test_find_code_snapshot_selects_by_sha12_or_fails(tmp_path):
    ds = tmp_path / "datasets/me/tfm3lab-code-aaaaaaaaaaaa"
    ds.mkdir(parents=True)
    a = ds / "tfm3lab-code-aaaaaaaaaaaa.tar.gz"
    b = ds / "tfm3lab-code-bbbbbbbbbbbb.tar.gz"
    a.write_bytes(b"")
    b.write_bytes(b"")
    assert kg.find_code_snapshot("tfm3lab-code-aaaaaaaaaaaa", tmp_path) == a
    ds2 = tmp_path / "datasets/me/tfm3lab-code-zzzzzzzzzzzz"
    ds2.mkdir()
    (ds2 / "tfm3lab-code-aaaaaaaaaaaa.tar.gz").write_bytes(b"")
    (ds2 / "tfm3lab-code-bbbbbbbbbbbb.tar.gz").write_bytes(b"")
    with pytest.raises(RuntimeError, match="bbbbbbbbbbbb"):
        kg.find_code_snapshot("tfm3lab-code-zzzzzzzzzzzz", tmp_path)
    d1 = tmp_path / "x/tfm3lab-code-cccccccccccc"
    for name, sha in (("one", "c" * 12), ("two", "d" * 12)):
        (d1 / name / "src/tfm3lab").mkdir(parents=True)
        (d1 / name / "pyproject.toml").write_text("")
        (d1 / name / "COMMIT_SHA").write_text(sha * 1 + "\n")
    assert kg.find_code_snapshot("tfm3lab-code-cccccccccccc", tmp_path / "x") == d1 / "one"


def test_materialize_is_writable_from_read_only_source(tmp_path):
    import os

    src = tmp_path / "ro"
    (src / "src/tfm3lab").mkdir(parents=True)
    (src / "pyproject.toml").write_text("")
    (src / "src/tfm3lab/m.py").write_text("x = 1\n")
    for cur, _dirs, files in os.walk(src):
        for n in files:
            (Path(cur) / n).chmod(0o444)
    for cur, _dirs, _files in os.walk(src, topdown=False):
        Path(cur).chmod(0o555)
    dest = tmp_path / "dest"
    try:
        kg.materialize_code_snapshot(src, dest)
        (dest / "src/tfm3lab/m.py").write_text("x = 2\n")
        (dest / "src/tfm3lab/new.py").write_text("y = 1\n")
        kg.materialize_code_snapshot(src, dest)  # replacing a writable copy works too
    finally:
        for cur, _dirs, files in os.walk(src):
            Path(cur).chmod(0o755)
            for n in files:
                (Path(cur) / n).chmod(0o644)


def test_helper_module_loads_standalone_from_path():
    """The notebook loads kaggle.py by file path before the snapshot is extracted."""
    path = ROOT / "src/tfm3lab/kaggle.py"
    spec = importlib.util.spec_from_file_location("_boot_t", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        del sys.modules[spec.name]
    assert callable(mod.find_code_snapshot)


def test_notebook_dataset_branch_present():
    nb = json.loads((ROOT / "notebooks/run_on_kaggle.ipynb").read_text())
    text = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for needle in ("REPO_SOURCE", "CODE_DATASET_SLUG", "snapshot_environ", "find_input_dir"):
        assert needle in text


# ------------------------------------------------------- sha env fallback


def test_git_sha_uses_env_when_no_git(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "_REPO_ROOT", tmp_path)
    monkeypatch.delenv("TFM3LAB_GIT_SHA", raising=False)
    monkeypatch.delenv("TFM3LAB_GIT_DIRTY", raising=False)
    with pytest.raises(RuntimeError, match="TFM3LAB_GIT_SHA"):
        manifest._git_sha()
    monkeypatch.setenv("TFM3LAB_GIT_SHA", "a" * 40)
    assert manifest._git_sha() == "a" * 40
    assert manifest._git_dirty() is None  # clean not declared
    monkeypatch.setenv("TFM3LAB_GIT_DIRTY", "0")
    assert manifest._git_dirty() is False
    assert manifest._prereg_commit() == "unknown"


def test_git_sha_ignores_env_when_git_exists(tmp_path, monkeypatch):
    repo = tmp_path / "r"
    repo.mkdir()
    sha = _git_repo(repo)
    monkeypatch.setattr(manifest, "_REPO_ROOT", repo)
    monkeypatch.setenv("TFM3LAB_GIT_SHA", "b" * 40)
    assert manifest._git_sha() == sha


def test_snapshot_code_digest_tracks_content(tmp_path):
    root = tmp_path / "snap"
    (root / "src/tfm3lab").mkdir(parents=True)
    (root / "src/tfm3lab/a.py").write_text("a = 1\n")
    (root / "pyproject.toml").write_text("p\n")
    base = cellstore.git_worktree_digest(root)
    assert base is not None and cellstore.git_worktree_digest(root) == base
    (root / "README.md").write_text("ignored\n")
    assert cellstore.git_worktree_digest(root) == base
    (root / "src/tfm3lab/a.py").write_text("a = 2\n")
    assert cellstore.git_worktree_digest(root) != base
