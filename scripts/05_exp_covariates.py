#!/usr/bin/env python3
"""Experiment D — covariate lecite (fatte bene), e una fatta male apposta.

Two genuinely-known-in-advance covariates:
  - market: day-of-week, cyclically encoded (sin/cos) — trivially known for
    any future calendar date, for any series.
  - MTG: days until the next set release, from TCGCSV's own `publishedOn`
    field (Wizards announces sets months ahead of release — this is real
    lead time, not a leak).

Plus one deliberate NEGATIVE CONTROL: a card's forecast is handed the
card's OWN actual future price as a "covariate" — a textbook leakage bug.
The resulting (artificially inflated) accuracy is the live-demo payload:
"here is what a metric looks like when the future leaks into the
pipeline" — the same failure mode the speaker's day job (agentic claims
document triage) has to design against structurally, not just avoid by
discipline.

This script calls tfm3lab.model.forecast_batch directly rather than
backtest.run_*_backtest: covariates vary per-origin in a way the other
three experiments don't need, so that plumbing lives here instead of
complicating the shared engine.

Requires:
  - scripts/01_fetch_data.py already run
  - a loaded TimesFM-3 forecaster (see scripts/02_exp_mtg.py's header)

Usage: uv run scripts/05_exp_covariates.py

Writes to results/:
    exp_covariates_market_dow.parquet
    exp_covariates_mtg_release.parquet
    exp_covariates_leakage_demo.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from tfm3lab import config
from tfm3lab.backtest import SeriesData
from tfm3lab.data.mtg import DEFAULT_CARDS, MAGIC_CATEGORY_ID, fetch_groups
from tfm3lab.metrics import mae, relative_mae
from tfm3lab.model import forecast_batch, load_forecaster
from tfm3lab.windows import context_slice, target_indices, valid_origins

CONTEXT_LEN = 32
HORIZON = 5
MAX_ORIGINS = 40


def _load_series(cache_name: str) -> list[SeriesData]:
    df = pd.read_parquet(config.CACHE_DIR / cache_name)
    return [
        SeriesData(
            name=name,
            values=g.sort_values("date")["value"].to_numpy(dtype=float),
            dates=g.sort_values("date")["date"].to_numpy(),
            observed=g.sort_values("date")["observed"].to_numpy(dtype=bool),
        )
        for name, g in df.groupby("series")
    ]


def day_of_week_covariate(dates: np.ndarray) -> np.ndarray:
    """Cyclical day-of-week encoding, shape (2, len(dates)): sin/cos of the
    weekday — known for any date, past or future, by construction."""
    dow = pd.DatetimeIndex(dates).dayofweek.to_numpy().astype(float)
    angle = 2 * np.pi * dow / 7.0
    return np.stack([np.sin(angle), np.cos(angle)], axis=0)


def days_until_next_release_covariate(
    dates: np.ndarray, release_dates: list[np.datetime64]
) -> np.ndarray:
    """Days until the nearest upcoming set release on or after each date,
    shape (1, len(dates)) — genuinely known in advance (announced release
    calendar), capped at 365 to keep the covariate's scale bounded."""
    dates64 = np.asarray(dates, dtype="datetime64[D]")
    releases = np.sort(np.asarray(release_dates, dtype="datetime64[D]"))
    out = np.empty(len(dates64), dtype=float)
    for i, d in enumerate(dates64):
        future = releases[releases >= d]
        out[i] = float((future[0] - d).astype(int)) if len(future) else 365.0
    return np.clip(out, 0, 365)[None, :]


def run_covariate_ablation(
    forecaster,
    s: SeriesData,
    origins: np.ndarray,
    build_covariate_fn,
    label: str,
) -> pd.DataFrame:
    """Runs the same origins with and without one past_future covariate."""
    rows = []
    for with_cov in (False, True):
        contexts, covariates, ts_ids = [], [], []
        for origin in origins:
            origin = int(origin)
            contexts.append(s.values[context_slice(origin, CONTEXT_LEN)])
            ts_ids.append(f"{s.name}::{origin}::{'cov' if with_cov else 'nocov'}")
            if with_cov:
                full = slice(origin - CONTEXT_LEN, origin + HORIZON)
                covariates.append(build_covariate_fn(s.dates[full]))
            else:
                covariates.append(None)

        batch = forecast_batch(
            forecaster,
            contexts,
            HORIZON,
            ts_ids=ts_ids,
            past_future_covariates=covariates if with_cov else None,
        )
        for i, origin in enumerate(origins):
            origin = int(origin)
            tgt = target_indices(origin, HORIZON)
            for h, t_idx in enumerate(tgt):
                if t_idx >= len(s.values):
                    break
                rows.append(
                    {
                        "covariate": label,
                        "series": s.name,
                        "origin_index": origin,
                        "horizon_step": h + 1,
                        "with_covariate": with_cov,
                        "actual": float(s.values[t_idx]),
                        "forecast": float(batch.forecast[i, h]),
                    }
                )
    return pd.DataFrame(rows)


def leakage_demo(forecaster, s: SeriesData, origins: np.ndarray) -> pd.DataFrame:
    """NEGATIVE CONTROL: hands the model the series' own actual FUTURE
    values as a 'covariate'. Must show a dramatic, artificial accuracy
    improvement — if it doesn't, the demo itself is broken, not the model.

    Three arms, not two — "clean" plus TWO leaked variants:

    - "leaked_flat_past": the first version of this fix, kept as a
      documented dead end. It padded the covariate's PAST with a constant
      (`ctx[-1]` repeated) before appending the real future. TimesFM-3
      treats a past_future_covariate as an extra variate and RevIN-
      normalizes it with its OWN per-variate running mean/std (see the
      installed timesfm3.model.py's `_preprocess`, which calls
      `util.get_running_stats` per variate before `util.revin`). A
      constant past has zero variance, so `util._make_safe_for_division`
      clamps that variate's std to 1.0 — an arbitrary scale unrelated to
      every other variate's real scale — AND gives the model nothing in
      the covariate's past to recognize as "this tracks the target",
      since a flat line correlates with nothing. The leak is present but
      illegible: measured effect on this card was ~0.33% (see
      docs/talk-outline.md), not the dramatic blowup the demo needs.
    - "leaked": the actual fix. The covariate's past IS the series' own
      real past (`ctx`), so it shares the target's real scale and a
      genuine, recognizable one-step-behind relationship with it before
      the leaked future ever appears.
    - "clean": no covariate, the baseline both leaked arms are compared
      against.
    """
    rows = []
    arms = ("clean", "leaked_flat_past", "leaked")
    for arm in arms:
        contexts, covariates, ts_ids = [], [], []
        for origin in origins:
            origin = int(origin)
            ctx = s.values[context_slice(origin, CONTEXT_LEN)]
            contexts.append(ctx)
            ts_ids.append(f"{s.name}::{origin}::{arm}")
            if arm == "clean":
                covariates.append(None)
            elif arm == "leaked_flat_past":
                future = s.values[target_indices(origin, HORIZON)]
                past_filler = np.full(CONTEXT_LEN, ctx[-1])
                covariates.append(np.concatenate([past_filler, future])[None, :])
            else:  # "leaked": real past + real future
                future = s.values[target_indices(origin, HORIZON)]
                covariates.append(np.concatenate([ctx, future])[None, :])

        batch = forecast_batch(
            forecaster,
            contexts,
            HORIZON,
            ts_ids=ts_ids,
            past_future_covariates=covariates if arm != "clean" else None,
        )
        for i, origin in enumerate(origins):
            origin = int(origin)
            tgt = target_indices(origin, HORIZON)
            for h, t_idx in enumerate(tgt):
                rows.append(
                    {
                        "series": s.name,
                        "origin_index": origin,
                        "horizon_step": h + 1,
                        "arm": arm,
                        "leaked": arm != "clean",  # back-compat column, see build_values() below
                        "actual": float(s.values[t_idx]),
                        "forecast": float(batch.forecast[i, h]),
                    }
                )
    return pd.DataFrame(rows)


def fetch_mtg_release_dates(cards=DEFAULT_CARDS) -> list[np.datetime64]:
    session = requests.Session()
    groups = fetch_groups(MAGIC_CATEGORY_ID, session)
    abbrevs = {c.group_abbreviation.upper() for c in cards}
    matched = groups[groups["abbreviation"].str.upper().isin(abbrevs)]
    return [np.datetime64(d[:10]) for d in matched["publishedOn"].dropna()]


def _require_origins(origins: np.ndarray, what: str) -> None:
    if len(origins) == 0:
        raise RuntimeError(
            f"no valid origins for {what} with context_len={CONTEXT_LEN}, horizon={HORIZON} — "
            "the cached series is too short. Fetch more history (scripts/01_fetch_data.py) or "
            "lower CONTEXT_LEN/HORIZON at the top of this script."
        )


def _print_ablation_summary(df: pd.DataFrame, group_col: str, label_col: str) -> None:
    for key, group in df.groupby(group_col):
        with_c = group[group[label_col]]
        without_c = group[~group[label_col]]
        mae_with = mae(with_c["actual"], with_c["forecast"])
        mae_without = mae(without_c["actual"], without_c["forecast"])
        rel = relative_mae(mae_with, mae_without) if mae_without > 0 else float("nan")
        print(f"  {key}: MAE without={mae_without:.4f}, with={mae_with:.4f}, relative={rel:.3f}")


def main() -> None:
    forecaster = load_forecaster()

    print("=== Legitimate covariate 1: market day-of-week ===")
    market_series = _load_series("market_prices.parquet")
    sp500 = next(s for s in market_series if s.name == "SP500")
    origins = valid_origins(len(sp500.values), CONTEXT_LEN, HORIZON, max_origins=MAX_ORIGINS)
    _require_origins(origins, "the market day-of-week ablation")
    market_df = run_covariate_ablation(
        forecaster, sp500, origins, day_of_week_covariate, "day_of_week"
    )
    market_df.to_parquet(config.RESULTS_DIR / "exp_covariates_market_dow.parquet", index=False)
    _print_ablation_summary(market_df, "covariate", "with_covariate")

    print("\n=== Legitimate covariate 2: MTG days-until-next-set-release ===")
    mtg_series = _load_series("mtg_prices.parquet")
    release_dates = fetch_mtg_release_dates()
    print(f"  {len(release_dates)} release dates found for the tracked sets")
    card = mtg_series[0]
    origins_mtg = valid_origins(len(card.values), CONTEXT_LEN, HORIZON, max_origins=MAX_ORIGINS)
    _require_origins(origins_mtg, "the MTG release-date ablation")

    def release_cov_fn(dates: np.ndarray) -> np.ndarray:
        return days_until_next_release_covariate(dates, release_dates)

    mtg_df = run_covariate_ablation(
        forecaster, card, origins_mtg, release_cov_fn, "days_to_release"
    )
    mtg_df.to_parquet(config.RESULTS_DIR / "exp_covariates_mtg_release.parquet", index=False)
    _print_ablation_summary(mtg_df, "covariate", "with_covariate")

    print("\n=== NEGATIVE CONTROL: leaking the actual future into the covariate ===")
    print("  three arms: clean (no covariate), leaked_flat_past (documented dead end,")
    print("  see leakage_demo's docstring), leaked (real past + real future)")
    leakage_df = leakage_demo(forecaster, card, origins_mtg)
    leakage_df.to_parquet(config.RESULTS_DIR / "exp_covariates_leakage_demo.parquet", index=False)
    mae_by_arm = {
        arm: mae(g["actual"], g["forecast"]) for arm, g in leakage_df.groupby("arm")
    }
    for arm, value in mae_by_arm.items():
        print(f"  MAE {arm}={value:.4f}")
    mae_clean = mae_by_arm["clean"]
    mae_leaked = mae_by_arm["leaked"]
    if mae_leaked >= mae_clean:
        print(
            "  WARNING: leaking the future (real past + real future) did not improve "
            "MAE — the demo did not reproduce the expected leakage effect even with a "
            "readable covariate. Report this as-is, don't force a positive result."
        )
    else:
        ratio = relative_mae(mae_leaked, mae_clean)
        print(
            f"  Confirmed: leaking the future drops MAE to {ratio:.1%} of the clean "
            "error — DO NOT present this as a real result."
        )
    flat_ratio = relative_mae(mae_by_arm["leaked_flat_past"], mae_clean)
    print(
        f"  For comparison, leaked_flat_past (the illegible variant) moved MAE to "
        f"{flat_ratio:.1%} of clean — the size of the effect a badly-wired leak produces."
    )

    print(f"\nWrote covariate experiment results to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
