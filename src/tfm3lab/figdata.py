"""Data-shaping for the demo/slide figures — pure pandas/numpy, no matplotlib.

Split out from plotting (see plots.py) so the fragile parts — reconstructing
a truth series across a gap the data doesn't cover, picking a demo window
without leaking, guarding a near-zero baseline MAE — are unit-testable
without a display backend. Every function here takes a DataFrame shaped
like results/exp_mtg_*.parquet or results/exp_shock_*.parquet and returns a
plain DataFrame or dataclass; nothing here reads a file or renders a plot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .metrics import coverage, mae
from .summarize import QUANTILE_COLUMNS

IDENTITY = "identity"
UNIVARIATE = "timesfm3_univariate"
MULTIVARIATE = "timesfm3_multivariate"

# The mode/transform baked into results/exp_mtg_demo_slice.parquet (written by
# scripts/02_exp_mtg.py alongside the full raw-predictions file — see that
# script for why only one combination is sliced). load_mtg_predictions only
# uses the slice when the caller asked for exactly this combination.
DEMO_SLICE_MODE = UNIVARIATE
DEMO_SLICE_TRANSFORM = IDENTITY
DEMO_SLICE_FILENAME = "exp_mtg_demo_slice.parquet"
FULL_RAW_FILENAME = "exp_mtg_raw_predictions.parquet"


def load_mtg_predictions(
    mode: str = UNIVARIATE,
    transform: str = IDENTITY,
    path: Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Loads MTG raw predictions, preferring the small demo slice over the
    49 MB full file whenever the request matches what the slice contains.

    Returns (dataframe, path_used) — callers (the demo notebook especially)
    should print the path: silently falling back to a file that may not
    exist offline is a `FileNotFoundError` in front of an audience, not a
    graceful degradation, so make the choice visible instead of silent.
    """
    if path is not None:
        df = pd.read_parquet(path)
        return _filter_mode_transform(df, mode, transform), path

    slice_path = config.RESULTS_DIR / DEMO_SLICE_FILENAME
    if mode == DEMO_SLICE_MODE and transform == DEMO_SLICE_TRANSFORM and slice_path.exists():
        return pd.read_parquet(slice_path), slice_path

    full_path = config.RESULTS_DIR / FULL_RAW_FILENAME
    if not full_path.exists():
        raise FileNotFoundError(
            f"neither {slice_path.name} nor {full_path.name} found in {config.RESULTS_DIR} — "
            "run scripts/02_exp_mtg.py first"
        )
    df = pd.read_parquet(full_path)
    return _filter_mode_transform(df, mode, transform), full_path


def _filter_mode_transform(df: pd.DataFrame, mode: str, transform: str) -> pd.DataFrame:
    if "mode" in df.columns:
        df = df[df["mode"] == mode]
    if "transform" in df.columns:
        df = df[df["transform"] == transform]
    return df


def reconstruct_truth(preds: pd.DataFrame) -> pd.DataFrame:
    """Rebuilds the observed price series from a raw-predictions frame.

    Every horizon_step==1 row's target_index equals its own origin_index
    (windows.py: origin is the first predicted index), so those rows alone
    cover every index from the first origin through the last. The very last
    origin's full horizon then extends coverage a further `horizon - 1`
    steps past that. Indices before the first origin (the initial context
    window, e.g. 0..63 at context_len=64) are NOT recoverable from this
    file — there is no origin early enough to have predicted them — and
    this function does not pretend otherwise: callers asking for history
    before the first origin simply get a shorter window than requested.
    """
    cols = ["series", "target_index", "target_date", "actual", "observed"]
    parts = []
    for _series, g in preds.groupby("series"):
        h1 = g[g["horizon_step"] == 1][cols]
        last_origin = g["origin_index"].max()
        tail = g[g["origin_index"] == last_origin][cols]
        parts.append(pd.concat([h1, tail], ignore_index=True))
    truth = pd.concat(parts, ignore_index=True)
    truth = truth.drop_duplicates(subset=["series", "target_index"])
    truth = truth.sort_values(["series", "target_index"]).reset_index(drop=True)
    return truth.rename(columns={"target_index": "index", "target_date": "date", "actual": "value"})


def find_glitches(
    truth: pd.DataFrame, *, return_threshold: float = 0.25, revert_threshold: float = 0.15
) -> pd.DataFrame:
    """Flags single-day round-trips: a day where the value jumps by more
    than `return_threshold` and the very next day reverts by more than
    `revert_threshold` in the opposite direction. Verified against two real
    TCGCSV artifacts (The One Ring and Urza's Saga, both spiking on
    2024-11-15) — two different cards moving on the same date is the
    evidence it is a feed artifact, not a market event. A genuine trend
    (e.g. Urza's Saga's real run-up) does not revert the next day and is
    not flagged.
    """
    rows = []
    for series, g in truth.groupby("series"):
        g = g.sort_values("index")
        ret = g["value"].pct_change()
        nxt = ret.shift(-1)
        mask = (ret.abs() > return_threshold) & (nxt.abs() > revert_threshold) & (
            np.sign(ret) != np.sign(nxt)
        )
        mask = mask.fillna(False)
        for _, row in g[mask].iterrows():
            rows.append({"series": series, "index": int(row["index"]), "date": row["date"]})
    return pd.DataFrame(rows, columns=["series", "index", "date"])


def _max_consecutive_false(mask: np.ndarray) -> int:
    """Longest run of False (unobserved) entries in a boolean array."""
    best = current = 0
    for observed in mask:
        if observed:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def data_quality_table(series_list: list) -> pd.DataFrame:
    """Per-card data-quality summary: observed rate, forward-fill rate, the
    longest run of consecutive unobserved points (assumes daily frequency —
    true for build_card_series's output, the only caller), glitch count
    (reusing find_glitches against the raw, not truth-reconstructed, price
    frame), price range, and log-return volatility restricted to observed
    points.
    """
    frames = [
        pd.DataFrame({
            "series": s.name,
            "index": np.arange(len(s.values)),
            "date": s.dates,
            "value": s.values,
        })
        for s in series_list
    ]
    raw = pd.concat(frames, ignore_index=True)
    glitch_counts = find_glitches(raw).groupby("series").size()

    rows = []
    for s in series_list:
        n = len(s.values)
        observed_rate = float(np.mean(s.observed)) if n else float("nan")
        fallback_rate = 1.0 - observed_rate if n else float("nan")
        obs_values = s.values[s.observed]
        if s.observed.sum() >= 2:
            volatility = float(np.std(np.diff(np.log(obs_values))))
        else:
            volatility = float("nan")
        rows.append(
            {
                "series": s.name,
                "observed_rate": observed_rate,
                "fallback_rate": fallback_rate,
                "max_gap_days": _max_consecutive_false(s.observed),
                "glitch_count": int(glitch_counts.get(s.name, 0)),
                "price_min": float(np.min(s.values)) if n else float("nan"),
                "price_max": float(np.max(s.values)) if n else float("nan"),
                "log_return_volatility": volatility,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ForecastSlice:
    """One origin's forecast, paired with the history that preceded it and
    the truth that followed — everything plots.plot_forecast_slice needs."""

    series: str
    origin_index: int
    origin_date: pd.Timestamp
    history_dates: np.ndarray
    history_values: np.ndarray
    target_dates: np.ndarray
    actual: np.ndarray
    forecast: np.ndarray
    q10: np.ndarray
    q90: np.ndarray
    naive: float
    coverage: float
    relative_mae: float
    contains_glitch: bool
    observed_mask: np.ndarray  # bool, aligned with target_dates/actual — True where the
    #   target is a real observation, False where it was forward-filled
    history_observed: np.ndarray  # bool, aligned with history_dates/history_values


def build_forecast_slice(
    preds: pd.DataFrame,
    truth: pd.DataFrame,
    series: str,
    origin_index: int,
    history_days: int = 120,
    glitches: pd.DataFrame | None = None,
    require_observed_targets: bool = False,
) -> ForecastSlice:
    """Builds one hero-chart window: `history_days` of real history strictly
    before `origin_index`, then the model's forecast for that origin against
    what actually happened. Raises if `origin_index` has no forecast rows —
    a silently empty chart is worse than a loud error here.

    `require_observed_targets=True` raises if any target point in this
    window is forward-filled rather than a real observation — set this for
    any window whose chart will be presented as "reale" (e.g. the hero
    slide), so a forward-filled value can never get drawn as if it were an
    actual market/TCGCSV print. Callers that only explore data (e.g. the
    demo notebook scrubbing through origins) should leave it False and
    instead use `observed_mask`/`history_observed` to render imputed points
    distinctly (see plots.plot_forecast_slice).

    Regardless of `require_observed_targets`, the returned `coverage` and
    `relative_mae` scalars are always scored on observed targets only —
    `nan` if none of this window's targets are observed. `actual`/
    `forecast`/`q10`/`q90` on the slice remain the FULL (unfiltered) arrays,
    for plotting.
    """
    fc = preds[(preds["series"] == series) & (preds["origin_index"] == origin_index)]
    if fc.empty:
        raise ValueError(f"no forecast rows for series={series!r}, origin_index={origin_index}")
    fc = fc.sort_values("horizon_step")

    observed_mask = fc["observed"].to_numpy(dtype=bool)
    if require_observed_targets and not observed_mask.all():
        unobserved = fc.loc[~observed_mask, "target_index"].astype(int).tolist()
        raise ValueError(
            f"series={series!r}, origin_index={origin_index}: target index(es) "
            f"{unobserved} are forward-filled, not observed — refusing to build a "
            "slice that would plot imputed values as real (require_observed_targets=True)"
        )

    origin_date = pd.Timestamp(fc["origin_date"].iloc[0])
    hist = truth[(truth["series"] == series) & (truth["index"] < origin_index)].sort_values("index")
    hist = hist.tail(history_days)

    actual = fc["actual"].to_numpy(dtype=float)
    forecast = fc["forecast"].to_numpy(dtype=float)
    q10 = fc["q10"].to_numpy(dtype=float)
    q90 = fc["q90"].to_numpy(dtype=float)
    naive = float(fc["baseline_naive"].iloc[0])

    # coverage/relative_mae are scored on observed targets only — a
    # forward-filled target's `actual` is just yesterday's price repeated,
    # and letting it into these scalars would silently launder an imputed
    # point into the printed accuracy story even though the chart itself
    # (plots.plot_forecast_slice) draws it as visually distinct. The full
    # (unfiltered) arrays are still returned on the slice for plotting.
    if observed_mask.any():
        obs_actual = actual[observed_mask]
        obs_forecast = forecast[observed_mask]
        obs_q10 = q10[observed_mask]
        obs_q90 = q90[observed_mask]
        naive_arr = np.full_like(obs_actual, naive)
        naive_mae = mae(obs_actual, naive_arr)
        model_mae = mae(obs_actual, obs_forecast)
        relative = model_mae / naive_mae if naive_mae > 1e-9 else float("nan")
        observed_coverage = coverage(obs_actual, obs_q10, obs_q90)
    else:
        relative = float("nan")
        observed_coverage = float("nan")

    if glitches is None:
        glitches = find_glitches(truth)
    window_indices = set(fc["target_index"].astype(int).tolist()) | set(
        hist["index"].astype(int).tolist()
    )
    series_glitch_indices = set(glitches.loc[glitches["series"] == series, "index"])
    contains_glitch = bool(window_indices & series_glitch_indices)

    return ForecastSlice(
        series=series,
        origin_index=int(origin_index),
        origin_date=origin_date,
        history_dates=hist["date"].to_numpy(),
        history_values=hist["value"].to_numpy(dtype=float),
        target_dates=fc["target_date"].to_numpy(),
        actual=actual,
        forecast=forecast,
        q10=q10,
        q90=q90,
        naive=naive,
        coverage=observed_coverage,
        relative_mae=relative,
        contains_glitch=contains_glitch,
        observed_mask=observed_mask,
        history_observed=hist["observed"].to_numpy(dtype=bool),
    )


def rank_windows(
    preds: pd.DataFrame,
    *,
    exclude_glitches: bool = True,
    min_naive_mae: float = 1e-9,
    require_all_observed: bool = True,
) -> pd.DataFrame:
    """One row per (series, origin): relative MAE, P10-P90 coverage, and the
    naive-relative price move over that window, guarded against the
    near-zero-naive-MAE blowup that makes a per-window relative-MAE MEAN
    unusable (a single near-flat card can send it to 1e8+) — callers must
    use the win-rate or the median across this table, never the mean of
    `relative_mae` directly.

    Policy for unobserved targets: a window with even one forward-filled
    (non-observed) target is, by default, dropped from the ranking entirely
    (`require_all_observed=True`) — a "hero" demo window picked by relative
    MAE or coverage must not be able to win by scoring well against an
    imputed value instead of a real observation. Set
    `require_all_observed=False` to keep such windows (e.g. for a
    data-quality diagnostic); `observed_fraction`/`all_targets_observed` are
    always reported so callers can see what was dropped.
    """
    truth = reconstruct_truth(preds)
    glitches = find_glitches(truth)
    glitch_keys = set(zip(glitches["series"], glitches["index"], strict=True))

    rows = []
    for (series, origin), g in preds.groupby(["series", "origin_index"]):
        g = g.sort_values("horizon_step")
        actual = g["actual"].to_numpy(dtype=float)
        forecast = g["forecast"].to_numpy(dtype=float)
        naive_val = float(g["baseline_naive"].iloc[0])
        naive_arr = np.full_like(actual, naive_val)
        naive_mae = mae(actual, naive_arr)
        model_mae = mae(actual, forecast)
        relative = model_mae / naive_mae if naive_mae >= min_naive_mae else float("nan")
        pct_change = (actual[-1] - naive_val) / naive_val if naive_val != 0 else float("nan")
        window_indices = set(g["target_index"].astype(int).tolist()) | {int(origin)}
        contains_glitch = any((series, idx) in glitch_keys for idx in window_indices)
        rows.append(
            {
                "series": series,
                "origin_index": int(origin),
                "pct_change": pct_change,
                "relative_mae": relative,
                "coverage": coverage(actual, g["q10"].to_numpy(), g["q90"].to_numpy()),
                "contains_glitch": contains_glitch,
                "beats_naive": bool(model_mae < naive_mae),
                "observed_fraction": float(g["observed"].mean()),
                "all_targets_observed": bool(g["observed"].all()),
            }
        )
    out = pd.DataFrame(rows)
    if exclude_glitches:
        out = out[~out["contains_glitch"]].reset_index(drop=True)
    if require_all_observed:
        out = out[out["all_targets_observed"]].reset_index(drop=True)
    return out


def horizon_profile(
    accuracy: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    mode: str = UNIVARIATE,
    transform: str = IDENTITY,
) -> pd.DataFrame:
    """Relative MAE and P10-P90 coverage vs horizon_step, averaged (with
    min/max) across series — the mechanism behind the calibration section:
    intervals sized about right at h=1 fail to widen fast enough, which is
    invisible in any single aggregate number.
    """
    acc = accuracy[(accuracy["mode"] == mode) & (accuracy["transform"] == transform)]
    acc_by_h = acc.groupby("horizon_step")["relative_mae_vs_baseline"].agg(["mean", "min", "max"])
    acc_by_h.columns = ["relative_mae_mean", "relative_mae_min", "relative_mae_max"]

    cal = calibration[(calibration["mode"] == mode) & (calibration["transform"] == transform)]
    cov_by_h = cal.groupby("horizon_step")["coverage_p10_p90"].mean().rename("coverage_mean")

    return acc_by_h.join(cov_by_h).reset_index()


def quantile_bin_calibration(
    preds: pd.DataFrame, horizon_steps: tuple[int, ...] = (1, 7, 28)
) -> pd.DataFrame:
    """Discrete quantile-bin calibration: 10 bins built directly from the 9
    known quantile forecasts (0.1..0.9), one horizon step at a time.

    Bin i is "how many of the 9 quantile forecasts does `actual` exceed":
    bin 0 is "actual <= q10", bin k (1..8) is "q{10k} < actual <= q{10(k+1)}",
    bin 9 is "actual > q90". Each bin has nominal probability 1/10 if the
    quantiles are calibrated.

    This replaces the old `pit_histogram`, which ran metrics.pit_values'
    *interpolated* PIT through `np.histogram(pit, bins=QUANTILE_LEVELS)` —
    9 edges make 8 bins, not 9 — and then mislabeled the outer two as
    "<= q10"/">= q90" when they actually covered [0.1, 0.2) and [0.8, 0.9].
    Counting directly against the quantile columns (no interpolation step)
    gets both the bin count and the labels right by construction.

    Rows with `observed=False` (forward-filled targets) are excluded before
    binning, matching the project-wide rule (see summarize.py's module
    docstring and `summarize_accuracy`/`summarize_calibration`) that a
    fabricated "yesterday's price repeated" value must never be scored
    against the model's forecast as if it were a real observation.
    """
    levels = config.QUANTILE_LEVELS
    n_bins = len(levels) + 1  # 10
    rows = []
    for h in horizon_steps:
        g = preds[(preds["horizon_step"] == h) & preds["observed"]]
        if g.empty:
            continue
        actual = g["actual"].to_numpy(dtype=float)
        quantiles = g[QUANTILE_COLUMNS].to_numpy(dtype=float)  # shape (n, 9), sorted per row
        n = len(actual)
        # bin_index[k] = count of quantile forecasts actual[k] strictly exceeds (0..9)
        bin_index = np.sum(quantiles < actual[:, None], axis=1)
        counts = np.bincount(bin_index, minlength=n_bins)[:n_bins]
        for i in range(n_bins):
            if i == 0:
                label = "≤ q10"
            elif i == n_bins - 1:
                label = "> q90"
            else:
                label = f"({levels[i - 1]:.1f}, {levels[i]:.1f}]"
            rows.append(
                {
                    "horizon_step": h,
                    "bin_index": i,
                    "label": label,
                    "count": int(counts[i]),
                    "fraction": float(counts[i]) / n if n else float("nan"),
                    "nominal_fraction": 1.0 / n_bins,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


# Deprecated alias: the name "PIT histogram" claimed continuous-PIT semantics
# this diagnostic never had. Kept only so a leftover caller doesn't hard
# break — prefer quantile_bin_calibration in new code.
pit_histogram = quantile_bin_calibration


def naive_gap(
    preds: pd.DataFrame, *, horizon_step: int = 1, group_col: str = "series"
) -> pd.DataFrame:
    """How close the model's forecast sits to a flat naive, per group — the
    number behind "the model is the naive in disguise": ratio near 0 means
    the forecast barely deviates from yesterday's value relative to how much
    the actual series really moved; corr near 1 means it moves in lockstep
    with the naive rather than anticipating anything.
    """
    g_all = preds[preds["horizon_step"] == horizon_step]
    rows = []
    for key, g in g_all.groupby(group_col):
        diff_model = float((g["forecast"] - g["baseline_naive"]).abs().mean())
        diff_actual = float((g["actual"] - g["baseline_naive"]).abs().mean())
        ratio = diff_model / diff_actual if diff_actual > 0 else float("nan")
        corr = (
            float(np.corrcoef(g["forecast"], g["baseline_naive"])[0, 1])
            if len(g) > 1
            else float("nan")
        )
        rows.append(
            {
                group_col: key,
                "mean_abs_dev_from_naive": diff_model,
                "mean_abs_actual_move": diff_actual,
                "ratio": ratio,
                "corr": corr,
                "n": len(g),
            }
        )
    return pd.DataFrame(rows)


def adaptation_lag_detail(
    shock_preds: pd.DataFrame,
    lag_df: pd.DataFrame,
    *,
    multiplier: float = 1.5,
    mode: str = MULTIVARIATE,
    series: str = "SP500",
) -> pd.DataFrame:
    """One row per event: the pre-event median error the adaptation-lag
    threshold is derived from, the threshold itself, and the resulting lag.
    The threshold column is the point of this function — it varies 7x+
    across events depending on how calm the pre-event window happened to
    be, which can generate most of a pre/post-cutoff gap on its own,
    independent of any real difference in adaptation speed.
    """
    sub = shock_preds[(shock_preds["mode"] == mode) & (shock_preds["series"] == series)]
    lag_sub = lag_df[
        (lag_df["multiplier"] == multiplier)
        & (lag_df["mode"] == mode)
        & (lag_df["series"] == series)
    ]
    rows = []
    for event, g in sub.groupby("event"):
        pre = g.loc[g["offset"] < 0, "abs_pct_error"]
        median_err = float(pre.median()) if len(pre) else float("nan")
        arm = g["arm"].iloc[0]
        lag_row = lag_sub[lag_sub["event"] == event]
        lag = float(lag_row["adaptation_lag_days"].iloc[0]) if len(lag_row) else float("nan")
        rows.append(
            {
                "event": event,
                "arm": arm,
                "pre_event_median_error": median_err,
                "threshold": median_err * multiplier,
                "adaptation_lag_days": lag,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["n_events_in_arm"] = out.groupby("arm")["event"].transform("count")
    return out.sort_values(["arm", "event"]).reset_index(drop=True)
