#!/usr/bin/env python3
"""Experiment B — pre-cutoff vs post-cutoff shocks. The core result of the talk.

Domanda: la reazione del modello agli shock di mercato cambia a seconda che
l'evento sia dentro o fuori la finestra di pretraining di TimesFM-3
(config.PRETRAIN_CUTOFF)? Ogni evento noto (config.KNOWN_EVENTS) è
un'ancora di controllo per un rilevatore automatico basato sui dati
(detect_shock_days) — la data dell'evento non è mai presa per fede da una
fonte giornalistica, solo confermata contro di essa.

Requires:
  - scripts/01_fetch_data.py already run (reads data/cache/market_prices.parquet)
  - a loaded TimesFM-3 forecaster (see README.md / scripts/02_exp_mtg.py's header)

Usage:
    uv run scripts/03_exp_shock.py
    uv run scripts/03_exp_shock.py --context-len 256 --window-before 12 --window-after 25

Writes to results/:
    exp_shock_raw_predictions.parquet   (every event x origin row, one-step-ahead)
    exp_shock_accuracy.parquet          (summarize_accuracy, grouped by mode/event/arm/series)
    exp_shock_adaptation_lag.parquet    (one row per event x arm x mode x lag-multiplier)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tfm3lab import config
from tfm3lab.backtest import SeriesData, run_multivariate_backtest, run_univariate_backtest
from tfm3lab.data.market import detect_shock_days
from tfm3lab.model import load_forecaster
from tfm3lab.summarize import compute_mase_scales, summarize_accuracy
from tfm3lab.windows import nearest_origin

DETECTION_Z_THRESHOLD = 4.0
DETECTION_TOLERANCE_DAYS = 3
ADAPTATION_LAG_MULTIPLIERS = (1.25, 1.5, 2.0)
HEADLINE_MULTIPLIER = 1.5


def load_cached_market_series() -> list[SeriesData]:
    path = config.CACHE_DIR / "market_prices.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/01_fetch_data.py first")
    df = pd.read_parquet(path)
    return [
        SeriesData(
            name=name,
            values=g.sort_values("date")["value"].to_numpy(dtype=float),
            dates=g.sort_values("date")["date"].to_numpy(),
            observed=g.sort_values("date")["observed"].to_numpy(dtype=bool),
        )
        for name, g in df.groupby("series")
    ]


def validate_known_events(sp500: SeriesData) -> None:
    """Confirms the data-driven detector fires near each known event date —
    a sanity check on the detector, not a substitute for it (the backtest
    below always anchors on the known date, detected or not)."""
    detected = set(detect_shock_days(sp500.values, z_threshold=DETECTION_Z_THRESHOLD).tolist())
    for event in config.KNOWN_EVENTS:
        origin = nearest_origin(sp500.dates, np.datetime64(event.date))
        nearby = any(abs(d - origin) <= DETECTION_TOLERANCE_DAYS for d in detected)
        status = "OK" if nearby else "NOT DETECTED (using the known date anyway)"
        print(f"  {event.name} ({event.date}, {event.arm}): {status}")


def estimate_adaptation_lag(
    offsets: np.ndarray,
    abs_pct_errors: np.ndarray,
    multiplier: float,
    consecutive_points: int = 3,
) -> float | None:
    """Days after the event until abs_pct_error drops to `multiplier` times
    the pre-event median error, for `consecutive_points` in a row. Returns
    None if it never does within the observed window (a real, reportable
    outcome — the model may simply not have re-converged yet)."""
    pre = abs_pct_errors[offsets < 0]
    if len(pre) == 0:
        return None
    threshold = np.median(pre) * multiplier

    post_mask = offsets >= 0
    order = np.argsort(offsets[post_mask])
    post_offsets = offsets[post_mask][order]
    post_errors = abs_pct_errors[post_mask][order]

    for i in range(len(post_errors) - consecutive_points + 1):
        if np.all(post_errors[i : i + consecutive_points] <= threshold):
            return float(post_offsets[i])
    return None


def run_event_backtest(
    forecaster,
    series_list: list[SeriesData],
    sp500_dates: np.ndarray,
    event: config.Event,
    context_len: int,
    window_before: int,
    window_after: int,
) -> pd.DataFrame:
    """One-step-ahead rolling forecasts around `event`, both univariate and
    multivariate, so the multivariate-vs-univariate ablation (plan finding
    #15) runs on market data too, not only on MTG (scripts/02)."""
    center = nearest_origin(sp500_dates, np.datetime64(event.date))
    n = len(sp500_dates)
    first = max(context_len, center - window_before)
    last = min(n - 1, center + window_after)
    if last < first:
        raise RuntimeError(
            f"no valid origins around '{event.name}': context_len={context_len} leaves no "
            f"room in a series of {n} days at this event's position. Fetch more history "
            "or shrink --context-len/--window-before."
        )
    origins = np.arange(first, last + 1)

    multi = run_multivariate_backtest(forecaster, series_list, origins, context_len, max_horizon=1)
    uni = run_univariate_backtest(forecaster, series_list, origins, context_len, max_horizon=1)
    combined = pd.concat([multi, uni], ignore_index=True)

    combined["event"] = event.name
    combined["event_date"] = pd.Timestamp(event.date)
    combined["arm"] = event.arm
    combined["offset"] = combined["origin_index"] - center
    denom = combined["actual"].abs().clip(lower=1e-8)
    combined["abs_pct_error"] = (combined["actual"] - combined["forecast"]).abs() / denom * 100
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--context-len", type=int, default=256)
    parser.add_argument("--window-before", type=int, default=12)
    parser.add_argument("--window-after", type=int, default=25)
    args = parser.parse_args()

    series_list = load_cached_market_series()
    sp500 = next(s for s in series_list if s.name == "SP500")
    start_date, end_date = pd.Timestamp(sp500.dates[0]).date(), pd.Timestamp(sp500.dates[-1]).date()
    print(
        f"Loaded {len(series_list)} market series, {len(sp500.values)} trading days "
        f"({start_date} .. {end_date})"
    )

    print("\nValidating known events against the data-driven shock detector:")
    validate_known_events(sp500)

    forecaster = load_forecaster()

    event_dfs = []
    for event in config.KNOWN_EVENTS:
        if not (start_date <= event.date <= end_date):
            print(f"\n  skipping {event.name}: outside the fetched date range")
            continue
        print(f"\nEvent: {event.name} ({event.date}, {event.arm})")
        event_dfs.append(
            run_event_backtest(
                forecaster,
                series_list,
                sp500.dates,
                event,
                args.context_len,
                args.window_before,
                args.window_after,
            )
        )

    if not event_dfs:
        raise RuntimeError("no known events fall inside the fetched date range")

    raw_df = pd.concat(event_dfs, ignore_index=True)
    raw_df.to_parquet(config.RESULTS_DIR / "exp_shock_raw_predictions.parquet", index=False)

    mase_scales = compute_mase_scales(series_list, boundary_index=args.context_len)
    group_cols = ("mode", "event", "arm", "series")
    accuracy = summarize_accuracy(raw_df, mase_scales, group_cols=group_cols)
    accuracy.to_parquet(config.RESULTS_DIR / "exp_shock_accuracy.parquet", index=False)

    lag_rows = []
    sp500_rows = raw_df[raw_df["series"] == "SP500"]
    for (event_name, arm, series_name, mode), group in sp500_rows.groupby(
        ["event", "arm", "series", "mode"]
    ):
        for mult in ADAPTATION_LAG_MULTIPLIERS:
            lag = estimate_adaptation_lag(
                group["offset"].to_numpy(), group["abs_pct_error"].to_numpy(), mult
            )
            lag_rows.append(
                {
                    "event": event_name,
                    "arm": arm,
                    "series": series_name,
                    "mode": mode,
                    "multiplier": mult,
                    "adaptation_lag_days": lag,
                }
            )
    lag_df = pd.DataFrame(lag_rows)
    lag_df.to_parquet(config.RESULTS_DIR / "exp_shock_adaptation_lag.parquet", index=False)

    print(f"\n--- Headline: pre vs post-cutoff, mean adaptation lag (x{HEADLINE_MULTIPLIER}) ---")
    at_mult = lag_df[lag_df["multiplier"] == HEADLINE_MULTIPLIER]
    headline = at_mult.groupby("arm")["adaptation_lag_days"].mean()
    print(headline.to_string())

    print(
        f"\nWrote {len(raw_df)} rows, {len(accuracy)} accuracy rows, "
        f"{len(lag_df)} lag rows to {config.RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
