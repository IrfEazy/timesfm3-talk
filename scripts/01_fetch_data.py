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

Usage:
    uv run scripts/01_fetch_data.py
    uv run scripts/01_fetch_data.py --mtg-start 2026-08-01   # quick check
    uv run scripts/01_fetch_data.py --skip-mtg               # market+CPI only
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from tfm3lab import config
from tfm3lab.backtest import SeriesData
from tfm3lab.data.macro import build_cpi_series
from tfm3lab.data.market import build_market_series
from tfm3lab.data.mtg import TCGCSV_ARCHIVE_START, build_card_series


def _series_list_to_frame(series_list: list[SeriesData]) -> pd.DataFrame:
    frames = [
        pd.DataFrame(
            {"date": s.dates, "value": s.values, "observed": s.observed, "series": s.name}
        )
        for s in series_list
    ]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--skip-mtg", action="store_true", help="skip the (slow) MTG/TCGCSV ingest")
    parser.add_argument(
        "--mtg-start",
        default=None,
        help="override MTG start date (YYYY-MM-DD); default is the full TCGCSV history",
    )
    args = parser.parse_args()

    if not args.skip_mtg:
        start = dt.date.fromisoformat(args.mtg_start) if args.mtg_start else TCGCSV_ARCHIVE_START
        print(f"Fetching MTG card prices from TCGCSV, {start} .. today...")
        mtg_series = build_card_series(start=start)
        mtg_df = _series_list_to_frame(mtg_series)
        mtg_df.to_parquet(config.CACHE_DIR / "mtg_prices.parquet", index=False)
        print(
            f"  {len(mtg_series)} cards, {mtg_df['date'].min()} .. {mtg_df['date'].max()}, "
            f"{mtg_df['observed'].mean():.1%} observed (rest forward-filled)"
        )
    else:
        print("Skipping MTG ingest (--skip-mtg)")

    print("Fetching market series (S&P 500, VIX, gold, oil) via yfinance...")
    market_series = build_market_series()
    market_df = _series_list_to_frame(market_series)
    market_df.to_parquet(config.CACHE_DIR / "market_prices.parquet", index=False)
    print(
        f"  {len(market_series)} tickers, {market_df['date'].min()} .. {market_df['date'].max()}, "
        f"{len(market_series[0].values)} trading days each"
    )

    print("Fetching CPI YoY from FRED...")
    cpi = build_cpi_series()
    cpi_df = _series_list_to_frame([cpi])
    cpi_df.to_parquet(config.CACHE_DIR / "cpi_yoy.parquet", index=False)
    print(f"  {len(cpi.values)} months, {cpi_df['date'].min()} .. {cpi_df['date'].max()}")

    print(f"\nAll caches written to {config.CACHE_DIR}")


if __name__ == "__main__":
    main()
