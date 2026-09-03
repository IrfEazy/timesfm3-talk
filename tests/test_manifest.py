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
    assert set(data["_meta"]["package_versions"]) == {"requests", "py7zr", "pandas"}


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
