"""Shared post-backtest summary logic.

Turns a raw long-format backtest DataFrame (from backtest.run_*_backtest)
into compact metrics tables — accuracy-with-significance and calibration —
so all four experiment scripts (scripts/02..05) report numbers the same
way, computed from the same tested primitives (metrics.py), instead of
each script growing its own slightly-different aggregation logic.

Every function here filters to `observed` rows first: forward-filled
target points are not real observations (plan findings #9/#10).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .metrics import (
    coverage,
    diebold_mariano,
    in_sample_scale,
    mae,
    mase,
    pinball_loss_multi,
    pit_values,
    relative_mae,
    rmse,
    smape,
)

QUANTILE_COLUMNS = [f"q{round(level * 100):02d}" for level in config.QUANTILE_LEVELS]

# Diebold-Mariano's long-run-variance estimate is not meaningful below this
# many observations per group (metrics.diebold_mariano enforces >= 2*horizon
# itself; this is a looser, practical floor for horizon=1 groups).
MIN_OBSERVATIONS_FOR_DM_TEST = 8


def compute_mase_scales(series_list, boundary_index: int, seasonality: int = 1) -> dict[str, float]:
    """One MASE scale per series, from the portion of its history strictly
    BEFORE `boundary_index` (e.g. the first evaluated origin's target
    start) — never from data inside or after the evaluated window, or the
    scale leaks information the model didn't have either.
    """
    return {
        s.name: in_sample_scale(s.values[:boundary_index], seasonality=seasonality)
        for s in series_list
    }


def summarize_accuracy(
    df: pd.DataFrame,
    mase_scales: dict[str, float],
    baseline_col: str = "baseline_naive",
    group_cols: tuple[str, ...] = ("mode", "series", "horizon_step"),
) -> pd.DataFrame:
    """One row per group: MAE/RMSE/sMAPE/MASE for the model and the chosen
    baseline, relative MAE (below 1 = model beats baseline), and a
    Diebold-Mariano test of the model's absolute error against the
    baseline's (negative statistic + small p-value = model significantly
    better).
    """
    if "series" not in group_cols:
        raise ValueError("group_cols must include 'series' to look up its MASE scale")
    series_idx = group_cols.index("series")

    observed = df[df["observed"]]
    rows = []
    for keys, group in observed.groupby(list(group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        scale = mase_scales.get(keys[series_idx], 1.0)

        actual = group["actual"].to_numpy()
        model_pred = group["forecast"].to_numpy()
        baseline_pred = group[baseline_col].to_numpy()
        model_abs_err = np.abs(actual - model_pred)
        baseline_abs_err = np.abs(actual - baseline_pred)

        mae_model_val = mae(actual, model_pred)
        mae_baseline_val = mae(actual, baseline_pred)
        # A baseline with zero error (e.g. a genuinely flat series where
        # "tomorrow = today" is exact) is a real, if rare, degenerate case —
        # relative_mae's own zero-baseline guard would crash a whole batch
        # summary over one such group. Define it explicitly instead: tied
        # at 0 if the model also has zero error, otherwise +inf (the model
        # lost to a perfect baseline — a real, reportable signal).
        if mae_baseline_val <= 0:
            relative_mae_val = 0.0 if mae_model_val <= 0 else np.inf
        else:
            relative_mae_val = relative_mae(mae_model_val, mae_baseline_val)

        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "n": len(group),
                "mae_model": mae_model_val,
                "rmse_model": rmse(actual, model_pred),
                "smape_model": smape(actual, model_pred),
                "mase_model": mase(model_abs_err, scale),
                f"mae_{baseline_col}": mae_baseline_val,
                f"mase_{baseline_col}": mase(baseline_abs_err, scale),
                "relative_mae_vs_baseline": relative_mae_val,
            }
        )
        if len(group) >= MIN_OBSERVATIONS_FOR_DM_TEST:
            dm_stat, dm_p = diebold_mariano(model_abs_err, baseline_abs_err, horizon=1)
        else:
            dm_stat, dm_p = np.nan, np.nan
        row["dm_stat"] = dm_stat
        row["dm_pvalue"] = dm_p
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_calibration(
    df: pd.DataFrame, group_cols: tuple[str, ...] = ("mode", "series", "horizon_step")
) -> pd.DataFrame:
    """Pinball loss (averaged over all 9 quantiles), P10-P90 coverage, and
    mean PIT value per group. A well-calibrated group has coverage near
    0.80 and mean PIT near 0.5 — see docs/talk-outline.md for how this
    feeds Experiment C.
    """
    observed = df[df["observed"]]
    rows = []
    for keys, group in observed.groupby(list(group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        actual = group["actual"].to_numpy()
        quantiles = group[QUANTILE_COLUMNS].to_numpy()
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "n": len(group),
                "pinball_avg": pinball_loss_multi(actual, quantiles, config.QUANTILE_LEVELS),
                "coverage_p10_p90": coverage(
                    actual, group["q10"].to_numpy(), group["q90"].to_numpy()
                ),
                "pit_mean": float(np.mean(pit_values(actual, quantiles, config.QUANTILE_LEVELS))),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
