"""Reproducibility manifest for tfm3lab data-fetch (and, later, model) runs.

write_manifest() carries the parts every run shares (git SHA, as_of,
live_end, timestamp, package versions); callers merge in a payload built by
a per-script function like build_fetch_manifest() below. JSON, not parquet
— small, human-diffable, meant to be read next to the results/ artifact it
describes, not queried at scale.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

# requests/py7zr/pandas cover the data-fetch runs; timesfm/torch cover the
# model runs (scripts/02*, 03, 05) -- a model result is only reproducible if
# the checkpoint-loading stack's versions are recorded alongside it. Any
# package not installed in the current environment records "unknown" rather
# than failing the run.
_TRACKED_PACKAGES = ("requests", "py7zr", "pandas", "timesfm", "torch")


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        OSError,
    ):
        return "unknown"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def write_manifest(
    payload: dict, path: Path, *, as_of: dt.date, live_end: bool = False
) -> Path:
    """Merges `payload` under a common `_meta` block and writes indented
    JSON to `path` (parent directories created as needed)."""
    manifest = {
        "_meta": {
            "git_sha": _git_sha(),
            "as_of": as_of.isoformat(),
            "live_end": live_end,
            "written_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "package_versions": _package_versions(),
        },
        **payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def build_fetch_manifest(
    *,
    date_range: tuple[dt.date, dt.date],
    resolved_cards: pd.DataFrame,
    archive_hashes: dict[str, str],
    price_field_counts: dict[str, int],
    subtype_counts: dict[str, int],
    coverage_stats: list[dict],
) -> dict:
    """MTG-specific fetch payload: date range, resolved card specs, archive
    hashes, price-field usage %, subtype usage counts (e.g. how often the
    resolver picked "Normal" vs. a foil variant — see
    mtg.py's _resolve_subtype_row), and per-card observed/forward-filled %
    (`coverage_stats` — see figdata.data_quality_table, which computes the
    same numbers for the data-quality figure; this function doesn't
    recompute them, just carries what the caller already has).
    """
    total_points = sum(price_field_counts.values())
    price_field_pct = (
        {k: v / total_points for k, v in price_field_counts.items()} if total_points else {}
    )
    return {
        "date_range": {"start": date_range[0].isoformat(), "end": date_range[1].isoformat()},
        "resolved_cards": resolved_cards.to_dict(orient="records"),
        "archive_hashes": archive_hashes,
        "price_field_used_pct": price_field_pct,
        "subtype_counts": subtype_counts,
        "coverage": coverage_stats,
    }
