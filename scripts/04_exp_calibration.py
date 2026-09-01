#!/usr/bin/env python3
"""Experiment C — sono calibrati i 9 quantili? Regime calmo vs regime di shock.

Design choice: this script does NOT call the model again. It reads the raw
predictions already produced by scripts/02_exp_mtg.py and
scripts/03_exp_shock.py and re-slices them by regime:
  - "calm"  = every MTG row (no shock defined there), plus market rows far
    from any known event (|offset| > CALM_OFFSET_THRESHOLD days)
  - "shock" = market rows close to a known event (|offset| <= that threshold)

Reusing cached predictions avoids paying for GPU inference a second time
just to re-slice the same forecasts by regime, and is only possible because
scripts/02 and 03 already ran and wrote their raw_predictions parquet files.

Domanda: la copertura empirica dei quantili coincide con quella nominale?
Peggiora vicino a uno shock (code sotto-coperte proprio quando servirebbero
di più)? Verificare, non assumere.

Requires: scripts/02_exp_mtg.py and scripts/03_exp_shock.py already run.

Usage: uv run scripts/04_exp_calibration.py [--calm-offset-threshold 3]

Writes to results/:
    exp_calibration_curve.parquet    (nominal vs empirical coverage, per regime x quantile level)
    exp_calibration_summary.parquet  (pinball/coverage/PIT per regime; see summarize.py)
"""

from __future__ import annotations

import argparse

import pandas as pd

from tfm3lab import config
from tfm3lab.summarize import QUANTILE_COLUMNS, summarize_calibration


def calibration_curve(df: pd.DataFrame, group_col: str = "regime") -> pd.DataFrame:
    """Empirical vs nominal coverage at every quantile level: for a
    well-calibrated forecaster, P(actual <= q_level) should equal
    `level` — e.g. the actual should fall below the q70 forecast 70% of
    the time. Deviations show up directly as empirical != nominal.
    """
    observed = df[df["observed"]]
    rows = []
    for regime, group in observed.groupby(group_col):
        for level, col in zip(config.QUANTILE_LEVELS, QUANTILE_COLUMNS, strict=True):
            empirical = float((group["actual"] <= group[col]).mean())
            rows.append(
                {
                    group_col: regime,
                    "nominal_level": level,
                    "empirical_coverage": empirical,
                    "gap": empirical - level,
                    "n": len(group),
                }
            )
    return pd.DataFrame(rows)


def load_mtg_calm() -> pd.DataFrame:
    path = config.RESULTS_DIR / "exp_mtg_raw_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/02_exp_mtg.py first")
    df = pd.read_parquet(path)
    # Keep the raw/identity-transform univariate arm — plenty of rows,
    # simplest single slice to compare against the shock experiment.
    df = df[(df["transform"] == "identity") & (df["mode"] == "timesfm3_univariate")].copy()
    df["regime"] = "calm"
    return df


def load_shock_by_regime(calm_offset_threshold: int) -> pd.DataFrame:
    path = config.RESULTS_DIR / "exp_shock_raw_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/03_exp_shock.py first")
    df = pd.read_parquet(path)
    df = df[df["mode"] == "timesfm3_multivariate"].copy()
    df["regime"] = df["offset"].abs().le(calm_offset_threshold).map({True: "shock", False: "calm"})
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--calm-offset-threshold",
        type=int,
        default=3,
        help="market rows within this many days of a known event count as 'shock' regime",
    )
    args = parser.parse_args()

    mtg_calm = load_mtg_calm()
    shock_split = load_shock_by_regime(args.calm_offset_threshold)
    print(f"MTG (calm, by construction): {len(mtg_calm)} rows")
    print(f"Market, split by proximity to a known event (threshold={args.calm_offset_threshold}d):")
    print(shock_split["regime"].value_counts().to_string())

    cols = ["regime", "observed", "actual", *QUANTILE_COLUMNS]
    combined = pd.concat([mtg_calm[cols], shock_split[cols]], ignore_index=True)

    curve = calibration_curve(combined)
    curve.to_parquet(config.RESULTS_DIR / "exp_calibration_curve.parquet", index=False)
    print("\nCalibration curve (empirical vs nominal coverage):")
    curve_pivot = curve.pivot(index="nominal_level", columns="regime", values="empirical_coverage")
    print(curve_pivot.to_string())

    summary = summarize_calibration(combined, group_cols=("regime",))
    summary.to_parquet(config.RESULTS_DIR / "exp_calibration_summary.parquet", index=False)
    print("\nPinball / P10-P90 coverage / mean PIT by regime:")
    print(summary.to_string(index=False))

    print(f"\nWrote calibration curve + summary to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
