"""Unit tests for tfm3lab.manifest — pure functions, no network."""

from __future__ import annotations

import datetime as dt
import json
import subprocess

import pandas as pd

from tfm3lab.manifest import build_fetch_manifest, write_manifest


def test_write_manifest_includes_common_meta_fields(tmp_path):
    path = write_manifest(
        {"hello": "world"}, tmp_path / "m.json", as_of=dt.date(2026, 9, 1), live_end=False
    )
    data = json.loads(path.read_text())
    assert data["hello"] == "world"
    assert data["_meta"]["as_of"] == "2026-09-01"
    assert data["_meta"]["live_end"] is False
    assert "git_sha" in data["_meta"]
    assert "written_at_utc" in data["_meta"]
    assert set(data["_meta"]["package_versions"]) >= {
        "requests",
        "py7zr",
        "pandas",
        "timesfm",
        "torch",
    }


def test_write_manifest_creates_parent_directories(tmp_path):
    path = write_manifest({}, tmp_path / "nested" / "m.json", as_of=dt.date(2026, 9, 1))
    assert path.exists()


def test_build_fetch_manifest_computes_price_field_pct():
    payload = build_fetch_manifest(
        date_range=(dt.date(2024, 2, 8), dt.date(2026, 9, 1)),
        resolved_cards=pd.DataFrame([{"label": "A", "group_id": 1, "product_id": 2}]),
        archive_hashes={"2024-02-08": "abc123"},
        price_field_counts={"market": 3, "mid": 1},
        subtype_counts={"Normal": 4},
        coverage_stats=[{"series": "A", "observed_rate": 0.9, "fallback_rate": 0.1}],
    )
    assert payload["price_field_used_pct"] == {"market": 0.75, "mid": 0.25}
    assert payload["date_range"] == {"start": "2024-02-08", "end": "2026-09-01"}
    assert payload["archive_hashes"] == {"2024-02-08": "abc123"}
    assert payload["resolved_cards"] == [{"label": "A", "group_id": 1, "product_id": 2}]
    assert payload["coverage"][0]["series"] == "A"


def test_build_fetch_manifest_empty_price_field_counts_gives_empty_pct():
    payload = build_fetch_manifest(
        date_range=(dt.date(2024, 2, 8), dt.date(2024, 2, 8)),
        resolved_cards=pd.DataFrame(columns=["label"]),
        archive_hashes={},
        price_field_counts={},
        subtype_counts={},
        coverage_stats=[],
    )
    assert payload["price_field_used_pct"] == {}


def test_build_fetch_manifest_passes_through_subtype_counts():
    payload = build_fetch_manifest(
        date_range=(dt.date(2024, 2, 8), dt.date(2024, 2, 8)),
        resolved_cards=pd.DataFrame(columns=["label"]),
        archive_hashes={},
        price_field_counts={},
        subtype_counts={"Normal": 5},
        coverage_stats=[],
    )
    assert payload["subtype_counts"] == {"Normal": 5}


def test_write_manifest_handles_git_timeout_gracefully(tmp_path, monkeypatch):
    """Verify that a git subprocess timeout degrades to 'unknown' SHA."""
    monkeypatch.setattr(
        "tfm3lab.manifest.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(cmd=["git"], timeout=5)
        ),
    )
    path = write_manifest({}, tmp_path / "m.json", as_of=dt.date(2026, 9, 1))
    data = json.loads(path.read_text())
    assert data["_meta"]["git_sha"] == "unknown"


def test_write_manifest_records_reproducibility_fields(tmp_path):
    import datetime as dt
    import json

    from tfm3lab.manifest import write_manifest

    ckpt = {"repo_id": "google/x", "revision_requested": None, "revision_resolved": "abc"}
    path = write_manifest(
        {"k": 1}, tmp_path / "m.json", as_of=dt.date(2026, 9, 1), checkpoint=ckpt, seed=7
    )
    meta = json.loads(path.read_text())["_meta"]
    assert isinstance(meta["git_dirty"], bool) or meta["git_dirty"] is None
    assert "prereg_commit" in meta
    assert {"gpu_name", "cuda_version", "torch_version", "device_count"} <= set(meta["environment"])
    assert meta["checkpoint"] == ckpt
    assert meta["seed"] == 7
    for pkg in ("numpy", "scipy", "huggingface_hub", "timesfm3"):
        assert pkg in meta["package_versions"]


def test_write_manifest_defaults_checkpoint_and_seed_to_none(tmp_path):
    import datetime as dt
    import json

    from tfm3lab.manifest import write_manifest

    path = write_manifest({}, tmp_path / "m.json", as_of=dt.date(2026, 9, 1))
    meta = json.loads(path.read_text())["_meta"]
    assert meta["checkpoint"] is None
    assert meta["seed"] is None


def test_write_run_manifest_lists_outputs(tmp_path):
    import datetime as dt
    import json

    from tfm3lab.manifest import write_run_manifest

    out = tmp_path / "a.parquet"
    out.write_bytes(b"abc")
    path = write_run_manifest(
        "exp_x",
        [out, tmp_path / "gone.parquet"],
        as_of=dt.date(2026, 9, 1),
        manifest_dir=tmp_path / "man",
    )
    data = json.loads(path.read_text())
    assert data["script"] == "exp_x"
    assert data["outputs"][0]["bytes"] == 3
    assert len(data["outputs"][0]["sha256"]) == 64
    assert data["outputs"][1]["missing"] is True


def test_git_dirty_ignores_results_and_data_outputs(tmp_path, monkeypatch):
    import shutil

    from tfm3lab import manifest

    if shutil.which("git") is None:
        import pytest

        pytest.skip("git not available")
    run = lambda *a: subprocess.run(a, cwd=tmp_path, check=True, capture_output=True)  # noqa: E731
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "t")
    (tmp_path / "code.py").write_text("x = 1\n")
    run("git", "add", "code.py")
    run("git", "commit", "-qm", "init")
    monkeypatch.setattr(manifest, "_REPO_ROOT", tmp_path)
    assert manifest._git_dirty() is False
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "out.parquet").write_text("o")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "c.parquet").write_text("o")
    assert manifest._git_dirty() is False
    (tmp_path / "code.py").write_text("x = 2\n")
    assert manifest._git_dirty() is True


def test_write_manifest_uses_supplied_git_dirty(tmp_path):
    path = write_manifest({}, tmp_path / "m.json", as_of=dt.date(2026, 9, 1), git_dirty=True)
    assert json.loads(path.read_text())["_meta"]["git_dirty"] is True
