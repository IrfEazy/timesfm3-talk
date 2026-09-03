#!/usr/bin/env python3
"""Fetch and cache the raw data every experiment reads.

  - MTG card prices (TCGCSV, full history since 2024-02-08 by default)
  - Market series: S&P 500, VIX, gold, oil (yfinance)
  - CPI YoY inflation (FRED)

Idempotent: TCGCSV's own daily archives are cached individually under
data/raw/ (re-running only downloads missing days); this script's parquet
outputs under data/cache/ are simply overwritten each run.

The full MTG backfill (~2.5 years, ~900+ daily archives) takes a while on
first run — use --mtg-start to fetch a shorter window (e.g. for a quick
local sanity check before committing to the full history on Colab).

A full run must freeze its end date with --as-of YYYY-MM-DD, recorded in
the run's manifest — --allow-live-end (dev only) uses today() instead and
is not reproducible.

Usage:
    uv run scripts/01_fetch_data.py --as-of 2026-09-01
    uv run scripts/01_fetch_data.py --as-of 2026-09-01 --mtg-start 2026-08-01  # quick check
    uv run scripts/01_fetch_data.py --as-of 2026-09-01 --skip-mtg             # market+CPI only
    uv run scripts/01_fetch_data.py --allow-live-end                         # dev only
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import matplotlib.pyplot as plt
import pandas as pd

from tfm3lab import config, figdata, plots
from tfm3lab.backtest import SeriesData
from tfm3lab.data.macro import build_cpi_series
from tfm3lab.data.market import build_market_series
from tfm3lab.data.mtg import TCGCSV_ARCHIVE_START, PriceSelectionPolicy, build_card_series
from tfm3lab.manifest import build_fetch_manifest, write_manifest

plots.apply_style()


def _series_list_to_frame(series_list: list[SeriesData]) -> pd.DataFrame:
    frames = [
        pd.DataFrame(
            {"date": s.dates, "value": s.values, "observed": s.observed, "series": s.name}
        )
        for s in series_list
    ]
    return pd.concat(frames, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--skip-mtg", action="store_true", help="skip the (slow) MTG/TCGCSV ingest")
    parser.add_argument(
        "--mtg-start",
        default=None,
        help="override MTG start date (YYYY-MM-DD); default is the full TCGCSV history",
    )
    end_group = parser.add_mutually_exclusive_group()
    end_group.add_argument(
        "--as-of",
        default=None,
        metavar="YYYY-MM-DD",
        help="freeze the run's end date for reproducibility; required unless --allow-live-end",
    )
    end_group.add_argument(
        "--allow-live-end",
        action="store_true",
        help="use today() as the end date (dev only) — not reproducible, "
        "records live_end=true in the manifest",
    )
    return parser


def resolve_as_of(args: argparse.Namespace) -> tuple[dt.date, bool]:
    """Returns (as_of, live_end). Exits (SystemExit) if neither --as-of nor
    --allow-live-end was given — a full run must not silently fall back to
    today()."""
    if args.allow_live_end:
        print(
            "WARNING: --allow-live-end — using today() as the run's end date; "
            "not reproducible, do not use for a full/final experiment run.",
            file=sys.stderr,
        )
        return dt.date.today(), True
    if args.as_of:
        return dt.date.fromisoformat(args.as_of), False
    raise SystemExit(
        "error: one of --as-of YYYY-MM-DD or --allow-live-end is required for a full fetch run"
    )


def main() -> None:
    args = build_parser().parse_args()
    as_of, live_end = resolve_as_of(args)
    utc_ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")

    if not args.skip_mtg:
        start = dt.date.fromisoformat(args.mtg_start) if args.mtg_start else TCGCSV_ARCHIVE_START
        print(f"Fetching MTG card prices from TCGCSV, {start} .. {as_of} (as_of)...")
        mtg_series, ingest_report = build_card_series(
            start=start, end=as_of, price_policy=PriceSelectionPolicy.MARKET_THEN_MID
        )
        mtg_df = _series_list_to_frame(mtg_series)
        mtg_df.to_parquet(config.CACHE_DIR / "mtg_prices.parquet", index=False)
        print(
            f"  {len(mtg_series)} cards, {mtg_df['date'].min()} .. {mtg_df['date'].max()}, "
            f"{mtg_df['observed'].mean():.1%} observed (rest forward-filled)"
        )

        quality_table = figdata.data_quality_table(mtg_series)
        quality_table.to_parquet(config.RESULTS_DIR / "mtg_data_quality.parquet", index=False)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        plots.plot_data_quality(quality_table, axes=axes)
        plots.save(fig, "mtg_data_quality")
        print(f"  data-quality table + figure written under {config.RESULTS_DIR}")

        manifest_payload = build_fetch_manifest(
            date_range=(start, as_of),
            resolved_cards=ingest_report.resolved_cards,
            archive_hashes=ingest_report.archive_hashes,
            price_field_counts=ingest_report.price_field_counts,
            subtype_counts=ingest_report.subtype_counts,
            coverage_stats=quality_table[["series", "observed_rate", "fallback_rate"]].to_dict(
                orient="records"
            ),
        )
        manifest_path = write_manifest(
            manifest_payload,
            config.RESULTS_DIR / "manifests" / f"fetch-{as_of.isoformat()}-{utc_ts}.json",
            as_of=as_of,
            live_end=live_end,
        )
        print(f"  manifest written to {manifest_path}")
    else:
        print("Skipping MTG ingest (--skip-mtg)")

    print("Fetching market series (S&P 500, VIX, gold, oil) via yfinance...")
    market_series = build_market_series(end=as_of.isoformat())
    market_df = _series_list_to_frame(market_series)
    market_df.to_parquet(config.CACHE_DIR / "market_prices.parquet", index=False)
    print(
        f"  {len(market_series)} tickers, {market_df['date'].min()} .. {market_df['date'].max()}, "
        f"{len(market_series[0].values)} trading days each"
    )

    print("Fetching CPI YoY from FRED...")
    cpi = build_cpi_series(end=as_of)
    cpi_df = _series_list_to_frame([cpi])
    cpi_df.to_parquet(config.CACHE_DIR / "cpi_yoy.parquet", index=False)
    print(f"  {len(cpi.values)} months, {cpi_df['date'].min()} .. {cpi_df['date'].max()}")

    print(f"\nAll caches written to {config.CACHE_DIR}")


if __name__ == "__main__":
    main()
