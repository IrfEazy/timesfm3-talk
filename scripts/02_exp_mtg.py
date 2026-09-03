#!/usr/bin/env python3
"""Experiment A — "camera pulita": Magic: The Gathering card prices.

Domanda: su un dominio quasi certamente assente dal pretraining di
TimesFM-3, su dati interamente successivi al cutoff (config.PRETRAIN_CUTOFF),
il modello batte la naive? Il multivariato batte l'univariato? log1p batte
raw (o solo make_positive)?

Requires:
  - scripts/01_fetch_data.py already run (reads data/cache/mtg_prices.parquet)
  - a loaded TimesFM-3 forecaster: GPU recommended (Colab), gated HF
    checkpoint accepted + `hf auth login` or HF_TOKEN — see README.md.
    Runs on CPU for a handful of short series, slowly.

Usage:
    uv run scripts/02_exp_mtg.py
    uv run scripts/02_exp_mtg.py --context-len 128 --max-horizon 28 --max-origins 50

Writes to results/:
    exp_mtg_raw_predictions.parquet  (every origin x horizon-step row, all mode/transform combos)
    exp_mtg_accuracy.parquet         (summarize_accuracy, grouped by mode/transform/series/horizon)
    exp_mtg_calibration.parquet      (summarize_calibration, same grouping)
"""

from __future__ import annotations

import argparse

import pandas as pd

from tfm3lab import config
from tfm3lab.backtest import (
    IDENTITY_TRANSFORM,
    LOG1P_TRANSFORM,
    SeriesData,
    run_multivariate_backtest,
    run_univariate_backtest,
)
from tfm3lab.model import load_forecaster
from tfm3lab.summarize import compute_mase_scales, summarize_accuracy, summarize_calibration
from tfm3lab.windows import valid_origins

GROUP_COLS = ("mode", "transform", "series", "horizon_step")


def load_cached_series() -> list[SeriesData]:
    path = config.CACHE_DIR / "mtg_prices.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/01_fetch_data.py first")
    df = pd.read_parquet(path)

    series_list = [
        SeriesData(
            name=name,
            values=group.sort_values("date")["value"].to_numpy(dtype=float),
            dates=group.sort_values("date")["date"].to_numpy(),
            observed=group.sort_values("date")["observed"].to_numpy(dtype=bool),
        )
        for name, group in df.groupby("series")
    ]

    # Multivariate stacking requires a common calendar (backtest._assert_aligned
    # enforces this) — trim every series to the shortest one's length, keeping
    # the most recent history, and log it: silently truncating coverage
    # without saying so is exactly what this project's plan rules out.
    lengths = {s.name: len(s.values) for s in series_list}
    n = min(lengths.values())
    if len(set(lengths.values())) > 1:
        print(f"  trimming all series to the shortest common length ({n} days): {lengths}")
    return [SeriesData(s.name, s.values[-n:], s.dates[-n:], s.observed[-n:]) for s in series_list]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--max-horizon", type=int, default=28)
    parser.add_argument(
        "--max-origins", type=int, default=None, help="cap origins per series (debug/quick runs)"
    )
    args = parser.parse_args()

    series_list = load_cached_series()
    n = len(series_list[0].values)
    print(f"Loaded {len(series_list)} series, {n} days each: {[s.name for s in series_list]}")

    origins = valid_origins(
        n=n, context_len=args.context_len, horizon=args.max_horizon, max_origins=args.max_origins
    )
    if len(origins) == 0:
        raise RuntimeError(
            f"no valid origins: {n} days is too short for context_len={args.context_len} "
            f"+ max_horizon={args.max_horizon}. Fetch more history or shrink these."
        )
    print(
        f"{len(origins)} origins per series "
        f"(context_len={args.context_len}, max_horizon={args.max_horizon})"
    )

    forecaster = load_forecaster()

    all_results = []
    for transform, transform_name in ((IDENTITY_TRANSFORM, "raw"), (LOG1P_TRANSFORM, "log1p")):
        print(f"\n--- transform: {transform_name} ---")
        print("  univariate...")
        all_results.append(
            run_univariate_backtest(
                forecaster,
                series_list,
                origins,
                args.context_len,
                args.max_horizon,
                transform=transform,
            )
        )
        print("  multivariate...")
        all_results.append(
            run_multivariate_backtest(
                forecaster,
                series_list,
                origins,
                args.context_len,
                args.max_horizon,
                transform=transform,
            )
        )

    raw_df = pd.concat(all_results, ignore_index=True)
    # zstd instead of the default snappy: ~33% smaller for the same content
    # (measured: 49.3 MB -> 32.8 MB on the full run), which matters because
    # this file is gitignored (see .gitignore) but the repo still has to
    # move it around (Kaggle -> Drive -> local) more than once per talk.
    raw_df.to_parquet(
        config.RESULTS_DIR / "exp_mtg_raw_predictions.parquet", index=False, compression="zstd"
    )

    # A small, committed slice for notebooks/demo.ipynb and scripts/06_make_figures.py:
    # the full file is gitignored (49 MB, over GitHub's warning threshold),
    # but the demo must still work fully offline. Written here, in the same
    # block that writes the parent file, rather than in a separate script —
    # a separate script is a second thing that can be forgotten after a
    # re-run and would silently describe a *previous* run's forecasts.
    # transform=="identity" & mode=="timesfm3_univariate" is the one
    # combination every demo/slide chart actually reads; measured at 8.42 MB
    # with all 9 quantile columns kept (dropping to q10/q90 only saves
    # little and breaks the quantile-bin calibration diagnostic, which needs the full grid).
    demo_slice = raw_df[
        (raw_df["transform"] == "identity") & (raw_df["mode"] == "timesfm3_univariate")
    ].drop(columns=["mode", "transform", "baseline_drift", "baseline_ets"])
    demo_slice = demo_slice.assign(series=demo_slice["series"].astype("category"))
    demo_slice.to_parquet(
        config.RESULTS_DIR / "exp_mtg_demo_slice.parquet", index=False, compression="zstd"
    )

    mase_scales = compute_mase_scales(series_list, boundary_index=int(origins.min()))
    accuracy = summarize_accuracy(raw_df, mase_scales, group_cols=GROUP_COLS)
    accuracy.to_parquet(config.RESULTS_DIR / "exp_mtg_accuracy.parquet", index=False)

    calibration = summarize_calibration(raw_df, group_cols=GROUP_COLS)
    calibration.to_parquet(config.RESULTS_DIR / "exp_mtg_calibration.parquet", index=False)

    print(
        f"\nWrote {len(raw_df)} prediction rows, {len(demo_slice)} demo-slice rows, "
        f"{len(accuracy)} accuracy rows, {len(calibration)} calibration rows "
        f"to {config.RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
