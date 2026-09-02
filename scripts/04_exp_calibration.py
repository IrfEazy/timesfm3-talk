#!/usr/bin/env python3
"""Experiment C — sono calibrati i 9 quantili? Calmo vs shock, a parità di dominio e orizzonte.

Design choice: this script does NOT call the model again. It reads the raw
predictions already produced by scripts/02_exp_mtg.py and
scripts/03_exp_shock.py and re-slices them by regime.

IMPORTANT — this used to pool MTG rows across all 28 horizon steps into
"calm" and compare them against market rows at horizon_step==1 only into
"shock". Coverage degrades with horizon (see exp_mtg_calibration.parquet:
0.822 at h=1, 0.538 at h=28), so that comparison mixed two confounds
(domain AND horizon) into one axis and made the shock regime look *better*
calibrated than calm — the opposite of the truth. Held within one domain
at a fixed horizon (market, h=1), shock is worse: P10-P90 coverage 0.839
(calm) vs 0.571 (shock) for SP500 alone. This script now produces three
regimes, all at horizon_step==1, never mixing domain or horizon into a
single "calm"/"shock" pool:
  - "market_calm"  = market rows far from any known event
                     (|offset| > CALM_OFFSET_THRESHOLD days)
  - "market_shock" = market rows close to a known event
                     (|offset| <= that threshold)
  - "mtg"          = MTG rows (no shock defined there; a cross-domain
                     reference point at the same horizon, not a "calm" bucket)

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

HORIZON_STEP = 1  # every regime is compared at this one horizon step only


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


def load_mtg_h1() -> pd.DataFrame:
    """MTG rows at horizon_step==1 only — a cross-domain reference point at
    the same horizon as the market regimes below, not a "calm" bucket."""
    path = config.RESULTS_DIR / "exp_mtg_raw_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/02_exp_mtg.py first")
    df = pd.read_parquet(path)
    df = df[
        (df["transform"] == "identity")
        & (df["mode"] == "timesfm3_univariate")
        & (df["horizon_step"] == HORIZON_STEP)
    ].copy()
    df["regime"] = "mtg"
    return df


def load_market_by_regime(calm_offset_threshold: int) -> pd.DataFrame:
    """Market rows at horizon_step==1, split into market_calm/market_shock
    by proximity to a known event. The shock file only ever has
    horizon_step==1 rows (scripts/03_exp_shock.py runs one-step-ahead), so
    this filter is a no-op there — kept explicit so this script does not
    silently start mixing horizons again if that ever changes."""
    path = config.RESULTS_DIR / "exp_shock_raw_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/03_exp_shock.py first")
    df = pd.read_parquet(path)
    df = df[(df["mode"] == "timesfm3_multivariate") & (df["horizon_step"] == HORIZON_STEP)].copy()
    is_shock = df["offset"].abs().le(calm_offset_threshold)
    df["regime"] = is_shock.map({True: "market_shock", False: "market_calm"})
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--calm-offset-threshold",
        type=int,
        default=3,
        help="market rows within this many days of a known event count as 'market_shock'",
    )
    args = parser.parse_args()

    mtg = load_mtg_h1()
    market = load_market_by_regime(args.calm_offset_threshold)
    print(f"MTG, horizon_step={HORIZON_STEP}: {len(mtg)} rows")
    print(f"Market, horizon_step={HORIZON_STEP}, split by proximity to a known event "
          f"(threshold={args.calm_offset_threshold}d):")
    print(market["regime"].value_counts().to_string())

    cols = ["regime", "observed", "actual", *QUANTILE_COLUMNS]
    combined = pd.concat([mtg[cols], market[cols]], ignore_index=True)

    curve = calibration_curve(combined)
    curve.to_parquet(config.RESULTS_DIR / "exp_calibration_curve.parquet", index=False)
    print("\nCalibration curve (empirical vs nominal coverage), three regimes, all at h=1:")
    curve_pivot = curve.pivot(index="nominal_level", columns="regime", values="empirical_coverage")
    print(curve_pivot.to_string())

    summary = summarize_calibration(combined, group_cols=("regime",))
    summary.to_parquet(config.RESULTS_DIR / "exp_calibration_summary.parquet", index=False)
    print("\nPinball / P10-P90 coverage / mean PIT by regime:")
    print("(pinball_avg is in the regime's own units — dollars for mtg, index points for")
    print(" market_* — never compare pinball_avg ACROSS regimes, only coverage/PIT)")
    print(summary.to_string(index=False))

    print(f"\nWrote calibration curve + summary to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
