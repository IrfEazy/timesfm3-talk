import numpy as np
import pandas as pd
import pytest

from tfm3lab import config, figdata

QC = list(config.QUANTILE_LEVELS)


def _pred_row(
    series="A",
    origin_index=10,
    horizon_step=1,
    origin_date=None,
    actual=100.0,
    forecast=100.0,
    baseline_naive=100.0,
    q10=90.0,
    q90=110.0,
    observed=True,
) -> dict:
    # Default origin_date derived from origin_index (one calendar day per
    # index) so a series of origins is chronologically consistent — a fixed
    # default date here would make every origin share the same date.
    origin_dt = pd.Timestamp(origin_date) if origin_date is not None else (
        pd.Timestamp("2024-01-01") + pd.Timedelta(days=origin_index)
    )
    target_dt = origin_dt + pd.Timedelta(days=horizon_step - 1)
    row = {
        "series": series,
        "origin_index": origin_index,
        "origin_date": origin_dt,
        "target_index": origin_index + horizon_step - 1,
        "target_date": target_dt,
        "horizon_step": horizon_step,
        "actual": actual,
        "observed": observed,
        "forecast": forecast,
        "baseline_naive": baseline_naive,
    }
    # spread quantiles linearly between q10 and q90, q50 = midpoint
    for level, col in zip(config.QUANTILE_LEVELS, figdata.QUANTILE_COLUMNS, strict=True):
        row[col] = q10 + (q90 - q10) * (level - 0.1) / 0.8
    return row


def _make_origin_block(series, origin_index, base_price, horizon=3, drift=0.0) -> list[dict]:
    """One origin's full horizon, with `actual` following a small drift from
    base_price and the model forecasting flat (a stand-in naive-ish model)."""
    rows = []
    for h in range(1, horizon + 1):
        actual = base_price + drift * h
        rows.append(
            _pred_row(
                series=series,
                origin_index=origin_index,
                horizon_step=h,
                actual=actual,
                forecast=base_price,
                baseline_naive=base_price,
                q10=base_price - 5,
                q90=base_price + 5,
            )
        )
    return rows


def _synthetic_preds(n_origins=5, horizon=3, start_origin=10, drift=1.0) -> pd.DataFrame:
    rows = []
    for i in range(n_origins):
        origin = start_origin + i
        rows.extend(
            _make_origin_block("A", origin, base_price=100.0 + i, horizon=horizon, drift=drift)
        )
    return pd.DataFrame(rows)


# --- reconstruct_truth -------------------------------------------------------


def test_reconstruct_truth_covers_h1_range_plus_last_origin_tail():
    preds = _synthetic_preds(n_origins=5, horizon=3, start_origin=10)
    truth = figdata.reconstruct_truth(preds)
    # h=1 rows give target_index == origin_index for origins 10..14 (5 rows);
    # the last origin (14) extends 2 more steps (target_index 15, 16).
    assert set(truth["index"]) == {10, 11, 12, 13, 14, 15, 16}
    assert list(truth.columns) == ["series", "index", "date", "value", "observed"]


def test_reconstruct_truth_no_duplicate_indices():
    preds = _synthetic_preds(n_origins=3, horizon=3, start_origin=5)
    truth = figdata.reconstruct_truth(preds)
    assert truth["index"].is_unique


def test_reconstruct_truth_minimum_index_is_first_origin():
    preds = _synthetic_preds(n_origins=4, horizon=2, start_origin=64)
    truth = figdata.reconstruct_truth(preds)
    assert truth["index"].min() == 64  # the initial context window (< 64) is not recoverable


# --- find_glitches ------------------------------------------------------------


def test_find_glitches_catches_spike_and_revert():
    truth = pd.DataFrame(
        {
            "series": ["A"] * 5,
            "index": [0, 1, 2, 3, 4],
            "date": pd.date_range("2024-01-01", periods=5),
            "value": [10.0, 10.1, 20.0, 10.2, 10.3],  # index 2: +98%, index 3: -49% (revert)
            "observed": [True] * 5,
        }
    )
    hits = figdata.find_glitches(truth)
    assert (hits["series"] == "A").any()
    assert 2 in set(hits["index"])


def test_find_glitches_ignores_monotone_ramp():
    truth = pd.DataFrame(
        {
            "series": ["A"] * 5,
            "index": [0, 1, 2, 3, 4],
            "date": pd.date_range("2024-01-01", periods=5),
            "value": [10.0, 15.0, 22.0, 33.0, 49.0],  # steady ~50% growth, never reverts
            "observed": [True] * 5,
        }
    )
    hits = figdata.find_glitches(truth)
    assert hits.empty


# --- build_forecast_slice -----------------------------------------------------


def test_build_forecast_slice_history_strictly_before_origin():
    # Two origins, drift=0 so each origin's h=1 actual settles exactly at
    # its base_price. Origin 66's baseline_naive is set to match truth at
    # index 65 (origin 65's own h=1 actual) — as it would be for real data,
    # where baseline_naive is literally the last observed context value.
    rows = _make_origin_block("A", 65, base_price=50.0, horizon=2)
    rows += _make_origin_block("A", 66, base_price=55.0, horizon=2)
    preds = pd.DataFrame(rows)
    preds.loc[preds["origin_index"] == 66, "baseline_naive"] = 50.0

    truth = figdata.reconstruct_truth(preds)
    sl = figdata.build_forecast_slice(preds, truth, "A", origin_index=66, history_days=5)
    assert sl.history_dates.max() < sl.origin_date  # windows.assert_no_leakage's spirit, on dates
    assert len(sl.target_dates) == 2  # horizon
    assert sl.naive == pytest.approx(sl.history_values[-1])


def test_build_forecast_slice_missing_origin_raises():
    preds = _synthetic_preds(n_origins=3, horizon=2, start_origin=10)
    truth = figdata.reconstruct_truth(preds)
    with pytest.raises(ValueError, match="no forecast rows"):
        figdata.build_forecast_slice(preds, truth, "A", origin_index=999)


def test_build_forecast_slice_coverage_and_relative_mae():
    # model forecasts flat at base_price, actual drifts away fast enough to
    # exit the q10/q90 band and to lose to the (also-flat) naive baseline.
    preds = _synthetic_preds(n_origins=1, horizon=3, start_origin=64, drift=20.0)
    truth = figdata.reconstruct_truth(preds)
    sl = figdata.build_forecast_slice(preds, truth, "A", origin_index=64, history_days=5)
    assert sl.coverage < 1.0
    assert sl.relative_mae == pytest.approx(1.0)  # forecast == naive by construction here


# --- rank_windows --------------------------------------------------------------


def test_rank_windows_flags_and_excludes_glitch_windows():
    # Origin 9 anchors a pre-spike value so the jump at origin 10 has a
    # baseline to compute a return against; origin 11 supplies the revert.
    rows = _make_origin_block("A", 9, base_price=100.0, horizon=2)
    rows += _make_origin_block("A", 10, base_price=100.0, horizon=2)
    rows += _make_origin_block("A", 11, base_price=100.0, horizon=2)
    df = pd.DataFrame(rows)
    df.loc[(df["origin_index"] == 10) & (df["horizon_step"] == 1), "actual"] = 250.0
    df.loc[(df["origin_index"] == 11) & (df["horizon_step"] == 1), "actual"] = 95.0

    ranked_all = figdata.rank_windows(df, exclude_glitches=False)
    assert bool(ranked_all.loc[ranked_all["origin_index"] == 10, "contains_glitch"].iloc[0])

    ranked_clean = figdata.rank_windows(df, exclude_glitches=True)
    assert 10 not in set(ranked_clean["origin_index"])


def test_rank_windows_guards_near_zero_naive_mae():
    # naive is exactly flat and actual barely moves -> naive MAE ~ 0; must
    # yield NaN, never an astronomically large relative_mae.
    rows = _make_origin_block("A", 10, base_price=0.1, horizon=3, drift=0.0)
    df = pd.DataFrame(rows)
    df.loc[df["horizon_step"] == 2, "actual"] = 0.1 + 1e-12  # forces a nonzero, tiny model error
    ranked = figdata.rank_windows(df, min_naive_mae=1e-9)
    row = ranked[ranked["origin_index"] == 10].iloc[0]
    assert np.isnan(row["relative_mae"])


# --- horizon_profile -----------------------------------------------------------


def test_horizon_profile_merges_accuracy_and_calibration():
    base = {"mode": "m", "transform": "t"}
    accuracy = pd.DataFrame(
        [
            {**base, "series": "A", "horizon_step": 1, "relative_mae_vs_baseline": 1.2},
            {**base, "series": "B", "horizon_step": 1, "relative_mae_vs_baseline": 0.8},
            {**base, "series": "A", "horizon_step": 2, "relative_mae_vs_baseline": 1.0},
        ]
    )
    calibration = pd.DataFrame(
        [
            {**base, "series": "A", "horizon_step": 1, "coverage_p10_p90": 0.8},
            {**base, "series": "B", "horizon_step": 1, "coverage_p10_p90": 0.6},
            {**base, "series": "A", "horizon_step": 2, "coverage_p10_p90": 0.5},
        ]
    )
    profile = figdata.horizon_profile(accuracy, calibration, mode="m", transform="t")
    row1 = profile[profile["horizon_step"] == 1].iloc[0]
    assert row1["relative_mae_mean"] == pytest.approx(1.0)
    assert row1["relative_mae_min"] == pytest.approx(0.8)
    assert row1["relative_mae_max"] == pytest.approx(1.2)
    assert row1["coverage_mean"] == pytest.approx(0.7)


# --- pit_histogram ---------------------------------------------------------------


def test_pit_histogram_fractions_sum_to_one_per_horizon():
    preds = _synthetic_preds(n_origins=20, horizon=1, start_origin=64, drift=0.5)
    hist = figdata.pit_histogram(preds, horizon_steps=(1,))
    assert hist["fraction"].sum() == pytest.approx(1.0)


def test_pit_histogram_outer_bins_labeled_as_clipped():
    preds = _synthetic_preds(n_origins=5, horizon=1, start_origin=64)
    hist = figdata.pit_histogram(preds, horizon_steps=(1,))
    labels = hist.sort_values("bin_left")["label"].tolist()
    assert labels[0] == "≤ q10"
    assert labels[-1] == "≥ q90"


# --- naive_gap -------------------------------------------------------------------


def test_naive_gap_ratio_and_corr_for_a_near_naive_model():
    # naive varies a lot origin-to-origin (like a real price series); the
    # model tracks it closely, only correcting by 30% of the day's move —
    # the "model is the naive in disguise" pattern verified on real shock
    # data (corr(forecast, naive) = 0.99+).
    naive_vals = [100.0, 120.0, 140.0, 160.0, 180.0]
    noise = [5.0, -8.0, 12.0, -3.0, 7.0]
    rows = []
    for i, (naive, n) in enumerate(zip(naive_vals, noise, strict=True)):
        actual = naive + n
        forecast = naive + 0.3 * n
        rows.append(_pred_row(series="X", origin_index=i, horizon_step=1, actual=actual,
                               forecast=forecast, baseline_naive=naive))
    df = pd.DataFrame(rows)
    gap = figdata.naive_gap(df, horizon_step=1)
    row = gap[gap["series"] == "X"].iloc[0]
    assert row["ratio"] == pytest.approx(0.3, abs=1e-6)
    assert row["corr"] > 0.99


# --- adaptation_lag_detail --------------------------------------------------------


def test_adaptation_lag_detail_threshold_is_pre_event_median_times_multiplier():
    base = {"mode": "m", "series": "SP500", "event": "E1", "arm": "pre_cutoff"}
    shock_preds = pd.DataFrame(
        [
            {**base, "offset": -2, "abs_pct_error": 2.0},
            {**base, "offset": -1, "abs_pct_error": 4.0},
            {**base, "offset": 0, "abs_pct_error": 10.0},
        ]
    )
    lag_df = pd.DataFrame([{**base, "multiplier": 1.5, "adaptation_lag_days": 3.0}])
    detail = figdata.adaptation_lag_detail(
        shock_preds, lag_df, multiplier=1.5, mode="m", series="SP500"
    )
    row = detail.iloc[0]
    assert row["pre_event_median_error"] == pytest.approx(3.0)  # median(2.0, 4.0)
    assert row["threshold"] == pytest.approx(4.5)
    assert row["adaptation_lag_days"] == pytest.approx(3.0)
    assert row["n_events_in_arm"] == 1
