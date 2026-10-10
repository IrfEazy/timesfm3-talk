"""Reproducibility manifest for tfm3lab data-fetch (and, later, model) runs.

write_manifest() carries the parts every run shares (git SHA, as_of,
live_end, timestamp, package versions); callers merge in a payload built by
a per-script function like build_fetch_manifest() below. JSON, not parquet
— small, human-diffable, meant to be read next to the results/ artifact it
describes, not queried at scale.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

# requests/py7zr/pandas cover the data-fetch runs; timesfm/torch cover the
# model runs (scripts/02*, 03, 05) -- a model result is only reproducible if
# the checkpoint-loading stack's versions are recorded alongside it. Any
# package not installed in the current environment records "unknown" rather
# than failing the run.
_TRACKED_PACKAGES = (
    "requests",
    "py7zr",
    "pandas",
    "numpy",
    "scipy",
    "huggingface_hub",
    "timesfm",
    "timesfm3",
    "torch",
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_UNSET = object()


GIT_SHA_ENV = "TFM3LAB_GIT_SHA"
GIT_DIRTY_ENV = "TFM3LAB_GIT_DIRTY"


def _has_git() -> bool:
    """False for a code snapshot (git archive extract) that carries no .git."""
    return (_REPO_ROOT / ".git").exists()


def _git_sha() -> str:
    if not _has_git():
        sha = os.environ.get(GIT_SHA_ENV, "").strip()
        if not sha:
            raise RuntimeError(
                f"no .git under {_REPO_ROOT}: this is a code snapshot, so the commit sha must "
                f"be given in the {GIT_SHA_ENV} environment variable (the Kaggle notebook sets "
                "it from the snapshot's COMMIT_SHA file)"
            )
        return sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=_REPO_ROOT,
        )
        return result.stdout.strip()
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        OSError,
    ):
        return "unknown"


def _git_dirty() -> bool | None:
    """True if the work tree has uncommitted changes; None if git is unavailable.

    Without a .git (code snapshot) it is False only when the snapshot declared itself
    clean through TFM3LAB_GIT_DIRTY=0, else None (unknown)."""
    if not _has_git():
        declared = os.environ.get(GIT_DIRTY_ENV, "").strip().lower()
        return False if declared in {"0", "false"} else None
    try:
        result = subprocess.run(
            # results/ and data/ hold the run's own outputs; they must not
            # make the code state look dirty.
            ["git", "status", "--porcelain", "--", ".", ":(exclude)results", ":(exclude)data"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
            cwd=_REPO_ROOT,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None


def capture_git_state() -> bool | None:
    """Code-state dirtiness; call at script start and pass to write_manifest(git_dirty=...)."""
    return _git_dirty()


def _prereg_commit() -> str:
    """Last commit touching docs/analysis-plan.md (the preregistration)."""
    if not _has_git():
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "log", "-n", "1", "--format=%H", "--", "docs/analysis-plan.md"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=_REPO_ROOT,
        )
        return result.stdout.strip() or "unknown"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
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
    payload: dict,
    path: Path,
    *,
    as_of: dt.date,
    live_end: bool = False,
    checkpoint: dict | None = None,
    seed: int | None = None,
    git_dirty: bool | None | object = _UNSET,
) -> Path:
    """Merges `payload` under a common `_meta` block and writes indented
    JSON to `path` (parent directories created as needed).

    `checkpoint` is {repo_id, revision_requested, revision_resolved} for model
    runs; `git_dirty` is a value captured by capture_git_state() at script
    start (default: computed now); `seed` is the run's RNG seed (None when the
    run uses no randomness).
    """
    from .preflight import run_environment

    meta = {
        "_meta": {
            "git_sha": _git_sha(),
            "git_dirty": _git_dirty() if git_dirty is _UNSET else git_dirty,
            "prereg_commit": _prereg_commit(),
            "environment": run_environment(),
            "checkpoint": checkpoint,
            "seed": seed,
            "as_of": as_of.isoformat(),
            "live_end": live_end,
            "written_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "package_versions": _package_versions(),
        },
    }
    manifest = {**meta, **payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_manifest(
    script: str,
    outputs: list[Path],
    *,
    as_of: dt.date,
    live_end: bool = False,
    checkpoint: dict | None = None,
    seed: int | None = None,
    extra: dict | None = None,
    git_dirty: bool | None | object = _UNSET,
    manifest_dir: Path | None = None,
) -> Path:
    """Per-run manifest for scripts 02-05: lists the output files (name, size,
    sha256) next to the common `_meta` block. Missing outputs are recorded as
    such rather than failing the run."""
    from . import config

    entries = []
    for out in outputs:
        out = Path(out)
        if out.exists():
            entries.append({"path": str(out), "bytes": out.stat().st_size, "sha256": _sha256(out)})
        else:
            entries.append({"path": str(out), "missing": True})
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    target = (manifest_dir or config.RESULTS_DIR / "manifests") / (
        f"{script}-{as_of.isoformat()}-{ts}.json"
    )
    payload = {"script": script, "outputs": entries, **(extra or {})}
    return write_manifest(
        payload,
        target,
        as_of=as_of,
        live_end=live_end,
        checkpoint=checkpoint,
        seed=seed,
        git_dirty=git_dirty,
    )


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
