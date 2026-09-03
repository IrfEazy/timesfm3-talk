#!/usr/bin/env python3
"""Regenerates every slide figure from results/*.parquet — nothing here
reads from the model or the network. Safe to run locally, without a GPU,
right after pulling fresh results from a Colab/Kaggle run.

All data-shaping lives in tfm3lab.figdata, all rendering in tfm3lab.plots —
this script is just a registry of (name, builder) pairs plus a loop, so
notebooks/demo.ipynb can import and reuse the exact same functions.

Usage:
    uv run scripts/06_make_figures.py
    uv run scripts/06_make_figures.py --only exp_mtg_forecast_slice

Writes PNGs to results/figures/.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from tfm3lab import config, figdata, plots

plots.apply_style()


def _read(name: str) -> pd.DataFrame | None:
    path = config.RESULTS_DIR / name
    if not path.exists():
        print(f"  skip (missing {name})")
        return None
    return pd.read_parquet(path)


def build_hero_slice() -> None:
    preds, source = figdata.load_mtg_predictions()
    print(f"  (loaded MTG predictions from {source.name})")
    truth = figdata.reconstruct_truth(preds)
    sl = figdata.build_forecast_slice(
        preds, truth, "The One Ring [LTR]", origin_index=238, require_observed_targets=True
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plots.plot_forecast_slice(sl, ax=ax, reveal=True)
    plots.save(fig, "exp_mtg_forecast_slice")


def build_shock_reactions() -> None:
    df = _read("exp_shock_raw_predictions.parquet")
    if df is None:
        return
    lag_df = _read("exp_shock_adaptation_lag.parquet")
    for event in ("Crollo Covid", "Shock dazi"):
        sub = df[
            (df["series"] == "SP500")
            & (df["mode"] == "timesfm3_multivariate")
            & (df["event"] == event)
        ]
        if sub.empty:
            print(f"  skip (no rows for event={event!r})")
            continue
        threshold = None
        if lag_df is not None:
            detail = figdata.adaptation_lag_detail(df, lag_df, series="SP500")
            row = detail[detail["event"] == event]
            if not row.empty:
                threshold = float(row["threshold"].iloc[0])
        fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
        plots.plot_shock_reaction(sub, axes=axes, threshold=threshold, title=event)
        plots.save(fig, f"exp_shock_reaction_{event.replace(' ', '_')}")


def build_horizon_profile() -> None:
    accuracy = _read("exp_mtg_accuracy.parquet")
    calibration = _read("exp_mtg_calibration.parquet")
    if accuracy is None or calibration is None:
        return
    profile = figdata.horizon_profile(accuracy, calibration)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    plots.plot_horizon_profile(profile, axes=axes)
    plots.save(fig, "exp_mtg_horizon_profile")


def build_quantile_bin_calibration() -> None:
    preds, _ = figdata.load_mtg_predictions()
    hist = figdata.quantile_bin_calibration(preds, horizon_steps=(1, 7, 28))
    if hist.empty:
        print("  skip (no rows at horizon steps 1/7/28)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4), sharey=True)
    plots.plot_quantile_bin_calibration(hist, axes=axes)
    plots.save(fig, "exp_mtg_quantile_bin_calibration")


def build_calibration_curve() -> None:
    curve = _read("exp_calibration_curve.parquet")
    if curve is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    plots.plot_calibration_curve(curve, ax=ax)
    plots.save(fig, "exp_calibration_curve")


def build_adaptation_dots() -> None:
    shock = _read("exp_shock_raw_predictions.parquet")
    lag_df = _read("exp_shock_adaptation_lag.parquet")
    if shock is None or lag_df is None:
        return
    detail = figdata.adaptation_lag_detail(shock, lag_df)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5))
    plots.plot_adaptation_dots(detail, axes=axes)
    plots.save(fig, "exp_shock_adaptation_dots")


def build_card_relative_mae() -> None:
    accuracy = _read("exp_mtg_accuracy.parquet")
    if accuracy is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plots.plot_card_relative_mae(accuracy, ax=ax)
    plots.save(fig, "exp_mtg_relative_mae")


def build_glitch_vignette() -> None:
    preds, _ = figdata.load_mtg_predictions()
    truth = figdata.reconstruct_truth(preds)
    glitches = figdata.find_glitches(truth)
    if glitches.empty:
        print("  skip (no glitches detected)")
        return
    fig, axes = plt.subplots(1, len(glitches), figsize=(4.5 * len(glitches), 4))
    plots.plot_glitch_vignette(truth, glitches, ax=axes)
    plots.save(fig, "exp_mtg_data_glitch")


FIGURES = [
    ("exp_mtg_forecast_slice", build_hero_slice),
    ("exp_shock_reaction", build_shock_reactions),
    ("exp_mtg_horizon_profile", build_horizon_profile),
    ("exp_mtg_quantile_bin_calibration", build_quantile_bin_calibration),
    ("exp_calibration_curve", build_calibration_curve),
    ("exp_shock_adaptation_dots", build_adaptation_dots),
    ("exp_mtg_relative_mae", build_card_relative_mae),
    ("exp_mtg_data_glitch", build_glitch_vignette),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", help="regenerate only the named figure (see the list below)")
    args = parser.parse_args()

    print(f"Writing figures to {config.FIGURES_DIR}")
    names = [n for n, _ in FIGURES]
    if args.only and args.only not in names:
        raise SystemExit(f"unknown figure {args.only!r}. Choices: {names}")

    for name, builder in FIGURES:
        if args.only and name != args.only:
            continue
        print(f"{name}:")
        builder()


if __name__ == "__main__":
    main()
