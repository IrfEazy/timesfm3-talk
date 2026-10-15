#!/usr/bin/env python3
"""Build the local per-day Magic price panel from the cached TCGCSV archives.

Parses every cached data/raw/tcgcsv-prices-<date>.ppmd.7z once into
data/cache/panel/<date>.parquet: one row per (product, subtype) with
marketPrice and midPrice (low/high/directLow are never stored), Normal and
Foil as separate SKUs. src/tfm3lab/data/panel.py:load_panel() then reads any
SKU's daily history back without re-parsing archives.

No network: only archives already under data/raw/ are read, and a date
without one is recorded as `missing_archive` in the manifest. The date range
is [--start, --as-of] inclusive, the same as scripts/01_fetch_data.py's
--as-of (so a run frozen at the same --as-of covers the same days). Re-running
resumes: dates whose parquet already exists are skipped (--force rebuilds
them). A day is parsed in one worker's memory (roughly 0.3-0.6 GB peak) and takes about
5-6 s, so the full ~940-day history needs ~20 min with 4 workers.

Writes results/manifests/panel-<as_of>-<UTC timestamp>.json: parameters, git
state, per-date records (archive sha256, output sha256, rows, status) and
counts -- no prices.

Usage:
    uv run scripts/12_build_panel.py --as-of 2026-09-04
    uv run scripts/12_build_panel.py --as-of 2026-09-04 --workers 6
    uv run scripts/12_build_panel.py --as-of 2026-09-04 --start 2026-08-01 --out /tmp/panel
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

from tfm3lab import config
from tfm3lab.data.mtg import MAGIC_CATEGORY_ID, TCGCSV_ARCHIVE_START
from tfm3lab.data.panel import (
    STATUS_MISSING,
    STATUS_SKIPPED,
    STATUS_WRITTEN,
    build_panel,
)
from tfm3lab.manifest import capture_git_state, write_manifest

_PROGRESS_EVERY = 50
_STATUSES = (STATUS_WRITTEN, STATUS_SKIPPED, STATUS_MISSING)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--as-of",
        required=True,
        type=dt.date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="last date to include, inclusive (as in scripts/01_fetch_data.py)",
    )
    parser.add_argument(
        "--start",
        default=TCGCSV_ARCHIVE_START,
        type=dt.date.fromisoformat,
        metavar="YYYY-MM-DD",
        help=f"first date to include (default {TCGCSV_ARCHIVE_START})",
    )
    parser.add_argument(
        "--workers",
        default=min(4, os.cpu_count() or 1),
        type=_positive_int,
        help="parallel processes, one day in memory per worker (default min(4, cpu_count))",
    )
    parser.add_argument(
        "--out",
        default=None,
        type=Path,
        help="output directory for the day files (default <data root>/cache/panel)",
    )
    parser.add_argument(
        "--force", action="store_true", help="rebuild days whose parquet already exists"
    )
    return parser


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.REPO_ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.start > args.as_of:
        parser.error(f"--start {args.start} is after --as-of {args.as_of}")
    out_dir = args.out if args.out is not None else config.CACHE_DIR / "panel"
    raw_dir = config.RAW_DIR
    git_dirty = capture_git_state()
    utc_ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")

    n_dates = (args.as_of - args.start).days + 1
    print(
        f"Building panel {args.start} .. {args.as_of} ({n_dates} dates, {args.workers} workers) "
        f"from {_display_path(raw_dir)} into {_display_path(out_dir)}",
        file=sys.stderr,
    )
    began = time.monotonic()
    done = 0

    def progress(record: dict) -> None:
        nonlocal done
        done += 1
        if done % _PROGRESS_EVERY == 0 or done == n_dates:
            print(
                f"  {done}/{n_dates} dates, {time.monotonic() - began:.0f}s elapsed",
                file=sys.stderr,
            )

    records = build_panel(
        raw_dir, out_dir, args.start, args.as_of, args.workers, args.force, progress=progress
    )
    elapsed = time.monotonic() - began

    counts = {status: sum(r["status"] == status for r in records) for status in _STATUSES}
    counts["dates"] = len(records)
    counts["rows"] = sum(r["rows"] or 0 for r in records)
    payload = {
        "script": "12_build_panel",
        "parameters": {
            "start": args.start.isoformat(),
            "end": args.as_of.isoformat(),
            "as_of": args.as_of.isoformat(),
            "workers": args.workers,
            "force": args.force,
            "category_id": MAGIC_CATEGORY_ID,
            "raw_dir": _display_path(raw_dir),
            "out_dir": _display_path(out_dir),
            "stored_price_fields": ["marketPrice", "midPrice"],
        },
        "counts": counts,
        "missing_dates": [r["date"] for r in records if r["status"] == STATUS_MISSING],
        "records": records,
    }
    manifest_path = write_manifest(
        payload,
        config.RESULTS_DIR / "manifests" / f"panel-{args.as_of.isoformat()}-{utc_ts}.json",
        as_of=args.as_of,
        live_end=False,
        git_dirty=git_dirty,
    )
    print(
        f"panel {args.start}..{args.as_of}: {counts['dates']} dates "
        f"(written {counts[STATUS_WRITTEN]}, skipped_existing {counts[STATUS_SKIPPED]}, "
        f"missing_archive {counts[STATUS_MISSING]}), {counts['rows']} rows, "
        f"{elapsed:.0f}s; manifest {_display_path(manifest_path)}"
    )


if __name__ == "__main__":
    main()
