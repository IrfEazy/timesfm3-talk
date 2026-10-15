#!/usr/bin/env python3
# scripts/14_v3_forecast.py
"""v3 studies (S1/S2/S3): TimesFM-3 forecasts for a requests file, resumable.

Reads the requests parquet (scaled price contexts only, never outcomes; schema
in tfm3lab.studies.requests), forecasts it block by block (block 1 first, so a
session that ends early still leaves the most valuable block complete) and
writes

    <out>/block=<k>/part-<n>.parquet     one file per batch, written atomically
    <out>/block_status.json              per block: requests, done, floored_values,
                                         seconds, status (complete|partial|pending)
    results/manifests/v3_forecast-<as_of>-<UTC timestamp>.json

The manifest records the requests file's sha256 and counts, the model
revision (configured and resolved), the device and the per-block status; it
never holds forecast values.

Usage:
    uv run --no-sync scripts/14_v3_forecast.py --as-of 2026-09-04 --dry-run
    uv run --no-sync scripts/14_v3_forecast.py --as-of 2026-09-04 --resume
    uv run --no-sync scripts/14_v3_forecast.py --as-of 2026-09-04 --blocks 1,2 --allow-cpu

Defaults: --requests data/cache/v3_requests.parquet (found like every other
cache file: under TFM3LAB_DATA_ROOT/cache, which the Kaggle notebook fills from
the private cache dataset) and --out results/benchmark/v3_forecasts (gitignored
through results/benchmark/; the Kaggle heavy archive carries it, and a restored
previous session resumes from it).

A fresh run refuses a --out that already holds parts; --resume continues it:
verified parts are skipped and no request is ever written twice. Every part
records the sha256 of the requests content it was forecast from (the manifest's
requests.content_sha256), the resolved checkpoint revision and the two inference
flags (also in block_status.json); --resume refuses parts made from different
requests, at another checkpoint revision or with other flags, and then a fresh
--out is needed. The configured revision must be a full 40-hex commit sha
(any run, --dry-run included, otherwise the script exits at once); it is checked
against the parts before any preflight. --dry-run only reports and never refuses an
--out that holds parts. A GPU is required (preflight: GPU, HF token, pinned
checkpoint) unless --allow-cpu is given for a smoke run. With nothing left to
forecast, no preflight runs and no model is loaded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

import pandas as pd

from tfm3lab import cellstore, config
from tfm3lab.manifest import capture_git_state, write_run_manifest
from tfm3lab.model import load_forecaster
from tfm3lab.preflight import preflight_model, run_environment
from tfm3lab.studies.requests import (
    BLOCK_MAX,
    BLOCK_MIN,
    read_forecasts,
    read_requests,
    requests_digest,
    validate_forecasts,
)
from tfm3lab.studies.runner import (
    MAKE_POSITIVE,
    STATUS_FILE,
    USE_SYMMETRIC_AVERAGING,
    current_status,
    has_forecast_parts,
    pending_counts,
    run_requests,
)

SCRIPT_NAME = "v3_forecast"
REQUESTS_NAME = "v3_requests.parquet"
DEFAULT_BATCH_SIZE = 128
_FULL_SHA = re.compile(r"[0-9a-f]{40}")


def default_requests_path() -> Path:
    return config.CACHE_DIR / REQUESTS_NAME


def default_out_dir() -> Path:
    # results/benchmark/ is gitignored (see .gitignore): nothing under it is committed
    return config.RESULTS_DIR / "benchmark" / "v3_forecasts"


def pinned_revision() -> str:
    """The configured checkpoint revision, which must be a full 40-hex commit
    sha for every run of this script (dry-run and status checks included): the
    hub resolves a full sha to itself, so it can be checked against the parts
    before any preflight and a branch or tag cannot move under a resume."""
    revision = (config.CHECKPOINT_REVISION or "").strip().lower()
    if not _FULL_SHA.fullmatch(revision):
        raise SystemExit(
            f"error: the configured checkpoint revision {config.CHECKPOINT_REVISION!r} is not a "
            "full 40-hex commit sha; pin tfm3lab.config.CHECKPOINT_REVISION to the commit "
            "(a branch or tag can move, so a resume could mix two checkpoints)"
        )
    return revision


def parse_blocks(text: str) -> list[int]:
    try:
        blocks = sorted({int(t) for t in text.split(",") if t.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad --blocks {text!r}; use e.g. 1,2,3,4") from exc
    if not blocks or any(not BLOCK_MIN <= b <= BLOCK_MAX for b in blocks):
        raise argparse.ArgumentTypeError(
            f"--blocks must list integers in {BLOCK_MIN}..{BLOCK_MAX}, got {text!r}"
        )
    return blocks


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--requests",
        type=Path,
        default=default_requests_path(),
        help="requests parquet (default: data/cache/v3_requests.parquet under the data root)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out_dir(),
        help="forecast directory (default: results/benchmark/v3_forecasts, gitignored)",
    )
    parser.add_argument("--adapter", choices=("timesfm3",), default="timesfm3")
    parser.add_argument("--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--blocks",
        type=parse_blocks,
        default=None,
        metavar="1,2,3,4",
        help="comma-separated priority blocks to run (default: every block in the file)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the --out directory: verified parts are skipped, nothing is duplicated",
    )
    parser.add_argument(
        "--as-of",
        type=dt.date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help="the cached data's cutoff date, recorded in the manifest (required unless --dry-run)",
    )
    parser.add_argument(
        "--allow-cpu", action="store_true", help="skip the GPU requirement (smoke runs only)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the pending requests per block; no preflight, no model, no writes",
    )
    return parser


def requests_summary(path: Path, requests_df: pd.DataFrame) -> dict:
    """Identity and counts of the requests file; no contexts, no values."""
    return {
        "path": str(path),
        "sha256": cellstore.file_sha256(path),
        "bytes": path.stat().st_size,
        "content_sha256": requests_digest(requests_df),
        "n_requests": len(requests_df),
        "by_study": {str(k): int(v) for k, v in requests_df["study"].value_counts().items()},
        "by_block": {
            str(int(k)): int(v) for k, v in requests_df["block"].value_counts().sort_index().items()
        },
        "horizon_max": int(requests_df["horizon"].max()),
        "context_length_max": int(requests_df["context"].map(len).max()),
    }


def validate_complete_blocks(requests_df: pd.DataFrame, out_dir: Path, status: dict) -> dict:
    """validate_forecasts over every complete block on disk (this session's and
    earlier ones): steps 1..horizon, finite, quantiles non-decreasing."""
    complete = sorted(int(b) for b, rec in status.items() if rec["status"] == "complete")
    if not complete:
        return {"blocks": [], "ok": True, "error": None}
    subset = requests_df[requests_df["block"].isin(complete)]
    forecasts = read_forecasts(out_dir)
    forecasts = forecasts[forecasts["request_id"].isin(set(subset["request_id"]))]
    try:
        validate_forecasts(subset, forecasts)
    except ValueError as exc:
        return {"blocks": complete, "ok": False, "error": str(exc)}
    return {"blocks": complete, "ok": True, "error": None}


def write_session_manifest(
    *,
    args: argparse.Namespace,
    summary: dict,
    status: dict,
    pending: dict[int, int],
    checkpoint: dict | None,
    validation: dict | None,
    error: str | None,
    git_dirty: bool | None,
) -> Path:
    parts = sorted(args.out.glob("block=*/part-*.parquet"))
    status_file = args.out / STATUS_FILE
    outputs = [*parts, *([status_file] if status_file.is_file() else [])]
    this_session = {
        str(b): status[str(b)]["done"] - (status[str(b)]["requests"] - n)
        for b, n in pending.items()
        if str(b) in status
    }
    totals = {
        "requests": sum(rec["requests"] for rec in status.values()),
        "done": sum(rec["done"] for rec in status.values()),
        "floored_values": sum(rec["floored_values"] for rec in status.values()),
        "parts": len(parts),
    }
    return write_run_manifest(
        SCRIPT_NAME,
        outputs,
        as_of=args.as_of,
        checkpoint=checkpoint
        or {
            "repo_id": config.CHECKPOINT_ID,
            "revision_requested": config.CHECKPOINT_REVISION,
            "revision_resolved": None,
        },
        seed=None,
        git_dirty=git_dirty,
        extra={
            "requests": summary,
            "model": {
                "adapter": args.adapter,
                "repo_id": config.CHECKPOINT_ID,
                "revision": config.CHECKPOINT_REVISION,
                "revision_resolved": (checkpoint or {}).get("revision_resolved"),
                "use_symmetric_averaging": USE_SYMMETRIC_AVERAGING,
                "make_positive": MAKE_POSITIVE,
            },
            "device": run_environment()["gpu_name"],
            "batch_size": args.batch_size,
            "blocks_selected": sorted(pending),
            "out_dir": str(args.out),
            "block_status": status,
            "counts": totals,
            "session": {
                "resume": args.resume,
                "pending_before": {str(b): n for b, n in pending.items()},
                "forecast_this_session": this_session,
            },
            "validation": validation,
            "error": error,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run and args.as_of is None:
        parser.error("--as-of YYYY-MM-DD is required for a real run (recorded in the manifest)")
    check_revision = pinned_revision()

    if not args.requests.is_file():
        raise SystemExit(
            f"error: requests file {args.requests} not found. It is built locally by the v3 "
            f"request builder and shipped inside the PRIVATE cache dataset as {REQUESTS_NAME} "
            "(docs/kaggle-runbook.md, 'v3 studies session')."
        )
    try:
        requests_df = read_requests(args.requests)
    except ValueError as exc:
        raise SystemExit(f"error: invalid requests file {args.requests}: {exc}") from exc
    summary = requests_summary(args.requests, requests_df)
    print(
        f"requests: {summary['n_requests']} ({summary['by_study']}), blocks {summary['by_block']}, "
        f"sha256 {summary['sha256'][:12]}"
    )

    holds_parts = has_forecast_parts(args.out)
    if holds_parts and not args.resume and not args.dry_run:
        raise SystemExit(
            f"error: {args.out} already holds forecast parts; pass --resume to continue them "
            "(parts are verified and never duplicated) or choose another --out"
        )
    try:
        pending = pending_counts(
            requests_df,
            args.out,
            args.blocks,
            checkpoint_revision=check_revision,
            use_symmetric_averaging=USE_SYMMETRIC_AVERAGING,
            make_positive=MAKE_POSITIVE,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    for block, n in sorted(pending.items()):
        print(f"block {block}: {n} pending of {summary['by_block'][str(block)]}")
    if args.dry_run:
        if holds_parts and not args.resume:
            print(
                f"note: {args.out} already holds forecast parts; the pending counts above are "
                "what --resume would still forecast, and a real run needs --resume"
            )
        print("dry run: no preflight, no model, nothing written")
        return 0

    git_dirty = capture_git_state()
    checkpoint: dict | None = None
    error: str | None = None
    status: dict = {}
    try:
        if sum(pending.values()):
            checkpoint = preflight_model(
                config.CHECKPOINT_ID, config.CHECKPOINT_REVISION, allow_cpu=args.allow_cpu
            )
            check_revision = checkpoint["revision_resolved"]
            forecaster = load_forecaster(revision=check_revision)
            status = run_requests(
                requests_df,
                forecaster,
                args.out,
                batch_size=args.batch_size,
                resume=args.resume,
                blocks=args.blocks,
                checkpoint_revision=check_revision,
                use_symmetric_averaging=USE_SYMMETRIC_AVERAGING,
                make_positive=MAKE_POSITIVE,
            )
        else:
            print("nothing pending: no preflight, no model load")
            status = current_status(
                requests_df,
                args.out,
                checkpoint_revision=check_revision,
                use_symmetric_averaging=USE_SYMMETRIC_AVERAGING,
                make_positive=MAKE_POSITIVE,
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        try:
            status = current_status(
                requests_df,
                args.out,
                checkpoint_revision=check_revision,
                use_symmetric_averaging=USE_SYMMETRIC_AVERAGING,
                make_positive=MAKE_POSITIVE,
            )
        except ValueError:
            status = {}
        manifest_path = write_session_manifest(
            args=args,
            summary=summary,
            status=status,
            pending=pending,
            checkpoint=checkpoint,
            validation=None,
            error=error,
            git_dirty=git_dirty,
        )
        print(f"run failed ({error}); partial progress kept, manifest {manifest_path}")
        raise

    validation = validate_complete_blocks(requests_df, args.out, status)
    manifest_path = write_session_manifest(
        args=args,
        summary=summary,
        status=status,
        pending=pending,
        checkpoint=checkpoint,
        validation=validation,
        error=None,
        git_dirty=git_dirty,
    )
    for block, rec in sorted(status.items()):
        print(
            f"block {block}: {rec['status']} {rec['done']}/{rec['requests']} "
            f"({rec['floored_values']} floored values, {rec['seconds']} s)"
        )
    print(f"manifest {manifest_path}")
    if not validation["ok"]:
        print(f"VALIDATION FAILED: {validation['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
