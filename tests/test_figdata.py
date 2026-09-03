import numpy as np
import pandas as pd
import pytest

from tfm3lab import config, figdata
from tfm3lab.backtest import SeriesData
from tfm3lab.metrics import coverage as _coverage
from tfm3lab.metrics import mae as _mae

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


def test_build_forecast_slice_exposes_observed_mask():
    rows = _make_origin_block("A", 65, base_price=50.0, horizon=3)
    preds = pd.DataFrame(rows)
    preds.loc[(preds["origin_index"] == 65) & (preds["horizon_step"] == 2), "observed"] = False

    truth = figdata.reconstruct_truth(preds)
    sl = figdata.build_forecast_slice(preds, truth, "A", origin_index=65, history_days=5)
    np.testing.assert_array_equal(sl.observed_mask, [True, False, True])
    assert sl.history_observed.dtype == bool


def test_build_forecast_slice_coverage_and_relative_mae_ignore_imputed_targets():
    # Two real observations (h=1, h=2) sit inside a tight q10/q90 band and
    # track the naive baseline closely (relative_mae == 1.0, coverage ==
    # 1.0). A third, forward-filled target (h=3, observed=False) is
    # deliberately way outside the band, with a forecast far worse than the
    # naive baseline on that row too — if it were wrongly folded into the
    # score, coverage would drop (2/3) and relative_mae would rise well
    # above 1.0.
    rows = [
        _pred_row(series="A", origin_index=65, horizon_step=1, actual=51.0, forecast=50.0,
                   baseline_naive=50.0, q10=45.0, q90=55.0, observed=True),
        _pred_row(series="A", origin_index=65, horizon_step=2, actual=49.0, forecast=50.0,
                   baseline_naive=50.0, q10=45.0, q90=55.0, observed=True),
        _pred_row(series="A", origin_index=65, horizon_step=3, actual=5000.0, forecast=-1000.0,
                   baseline_naive=50.0, q10=45.0, q90=55.0, observed=False),
    ]
    preds = pd.DataFrame(rows)
    truth = figdata.reconstruct_truth(preds)
    sl = figdata.build_forecast_slice(preds, truth, "A", origin_index=65, history_days=5)

    observed = preds[preds["observed"]].sort_values("horizon_step")
    obs_actual = observed["actual"].to_numpy(dtype=float)
    obs_forecast = observed["forecast"].to_numpy(dtype=float)
    obs_q10 = observed["q10"].to_numpy(dtype=float)
    obs_q90 = observed["q90"].to_numpy(dtype=float)
    naive_val = float(observed["baseline_naive"].iloc[0])
    naive_arr = np.full_like(obs_actual, naive_val)
    expected_relative = _mae(obs_actual, obs_forecast) / _mae(obs_actual, naive_arr)
    expected_coverage = _coverage(obs_actual, obs_q10, obs_q90)

    assert sl.coverage == pytest.approx(expected_coverage)
    assert sl.coverage == pytest.approx(1.0)
    assert sl.relative_mae == pytest.approx(expected_relative)
    assert sl.relative_mae == pytest.approx(1.0)

    # Sanity check the "wrongly included" claim itself: if all 3 rows (the
    # imputed one too) had been scored, coverage would drop and relative_mae
    # would rise well past the observed-only values above.
    all_actual = preds.sort_values("horizon_step")["actual"].to_numpy(dtype=float)
    all_forecast = preds.sort_values("horizon_step")["forecast"].to_numpy(dtype=float)
    all_q10 = preds.sort_values("horizon_step")["q10"].to_numpy(dtype=float)
    all_q90 = preds.sort_values("horizon_step")["q90"].to_numpy(dtype=float)
    naive_arr_all = np.full_like(all_actual, naive_val)
    coverage_if_included = _coverage(all_actual, all_q10, all_q90)
    relative_if_included = _mae(all_actual, all_forecast) / _mae(all_actual, naive_arr_all)
    assert coverage_if_included < sl.coverage
    assert relative_if_included > sl.relative_mae


def test_build_forecast_slice_coverage_and_relative_mae_nan_when_nothing_observed():
    rows = _make_origin_block("A", 65, base_price=50.0, horizon=2, drift=0.0)
    preds = pd.DataFrame(rows)
    preds.loc[preds["origin_index"] == 65, "observed"] = False

    truth = figdata.reconstruct_truth(preds)
    sl = figdata.build_forecast_slice(preds, truth, "A", origin_index=65, history_days=5)
    assert np.isnan(sl.coverage)
    assert np.isnan(sl.relative_mae)


def test_build_forecast_slice_require_observed_targets_raises():
    rows = _make_origin_block("A", 65, base_price=50.0, horizon=3)
    preds = pd.DataFrame(rows)
    preds.loc[(preds["origin_index"] == 65) & (preds["horizon_step"] == 2), "observed"] = False
    truth = figdata.reconstruct_truth(preds)

    with pytest.raises(ValueError, match="forward-filled"):
        figdata.build_forecast_slice(
            preds, truth, "A", origin_index=65, history_days=5, require_observed_targets=True
        )


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


def test_rank_windows_excludes_windows_with_unobserved_targets_by_default():
    rows = _make_origin_block("A", 20, base_price=100.0, horizon=3)
    df = pd.DataFrame(rows)
    df.loc[(df["origin_index"] == 20) & (df["horizon_step"] == 2), "observed"] = False

    ranked = figdata.rank_windows(df, exclude_glitches=False)
    assert 20 not in set(ranked["origin_index"])


def test_rank_windows_keeps_unobserved_windows_when_flag_disabled():
    rows = _make_origin_block("A", 20, base_price=100.0, horizon=3)
    df = pd.DataFrame(rows)
    df.loc[(df["origin_index"] == 20) & (df["horizon_step"] == 2), "observed"] = False

    ranked = figdata.rank_windows(df, exclude_glitches=False, require_all_observed=False)
    row = ranked[ranked["origin_index"] == 20].iloc[0]
    assert not row["all_targets_observed"]
    assert row["observed_fraction"] == pytest.approx(2 / 3)


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


# --- quantile_bin_calibration ---------------------------------------------------


def test_quantile_bin_calibration_fractions_sum_to_one_per_horizon():
    preds = _synthetic_preds(n_origins=20, horizon=1, start_origin=64, drift=0.5)
    hist = figdata.quantile_bin_calibration(preds, horizon_steps=(1,))
    assert hist["fraction"].sum() == pytest.approx(1.0)


def test_quantile_bin_calibration_has_ten_bins_with_correct_labels():
    preds = _synthetic_preds(n_origins=5, horizon=1, start_origin=64)
    hist = figdata.quantile_bin_calibration(preds, horizon_steps=(1,))
    labels = hist.sort_values("bin_index")["label"].tolist()
    assert len(labels) == 10
    assert labels[0] == "≤ q10"
    assert labels[1] == "(0.1, 0.2]"
    assert labels[-2] == "(0.8, 0.9]"
    assert labels[-1] == "> q90"
    assert hist["nominal_fraction"].eq(0.1).all()


def test_quantile_bin_calibration_excludes_unobserved_rows():
    # 20 origins x horizon=1 -> 20 rows, all observed by default. Flip one
    # row to observed=False AND move its actual far outside its own
    # quantile band, so if it were wrongly counted it would land in a
    # different (outer) bin than where the rest of the mass sits — the
    # count-sum check below is the load-bearing assertion either way.
    preds = _synthetic_preds(n_origins=20, horizon=1, start_origin=64, drift=0.5)
    target_row = preds[preds["horizon_step"] == 1].index[0]
    preds.loc[target_row, "observed"] = False
    preds.loc[target_row, "actual"] = -1_000_000.0  # would land in the "<= q10" bin if included

    hist = figdata.quantile_bin_calibration(preds, horizon_steps=(1,))
    n_observed = int(preds["observed"].sum())
    assert n_observed == 19
    assert hist["count"].sum() == n_observed
    assert hist["count"].sum() != len(preds)  # i.e. the unobserved row was actually dropped
    assert hist["n"].iloc[0] == n_observed


def test_quantile_bin_calibration_below_q10_lands_in_first_bin():
    # actual well below every quantile forecast -> must count in bin 0 ("<= q10"),
    # never silently merged into the (q10, q20] bin the old 8-bin histogram produced.
    preds = _synthetic_preds(n_origins=1, horizon=1, start_origin=64)
    preds.loc[preds["horizon_step"] == 1, "actual"] = -1000.0
    hist = figdata.quantile_bin_calibration(preds, horizon_steps=(1,))
    row0 = hist[hist["bin_index"] == 0].iloc[0]
    assert row0["count"] == 1
    assert hist[hist["bin_index"] != 0]["count"].sum() == 0


def test_pit_histogram_alias_still_works():
    preds = _synthetic_preds(n_origins=5, horizon=1, start_origin=64)
    pd.testing.assert_frame_equal(
        figdata.pit_histogram(preds, horizon_steps=(1,)),
        figdata.quantile_bin_calibration(preds, horizon_steps=(1,)),
    )


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


# --- data_quality_table ---


def _series(name, values, observed, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(values)).to_numpy()
    return SeriesData(
        name=name, values=np.array(values, dtype=float), dates=dates, observed=np.array(observed)
    )


def test_data_quality_table_observed_and_fallback_rates():
    s = _series("A", [10.0] * 8, [True, True, False, False, True, True, True, True])
    table = figdata.data_quality_table([s])
    row = table.iloc[0]
    assert row["series"] == "A"
    assert row["observed_rate"] == pytest.approx(6 / 8)
    assert row["fallback_rate"] == pytest.approx(2 / 8)


def test_data_quality_table_max_gap_days_is_longest_unobserved_run():
    observed = [True, False, False, False, True, False, True]
    s = _series("A", [10.0] * 7, observed)
    table = figdata.data_quality_table([s])
    assert table.iloc[0]["max_gap_days"] == 3


def test_data_quality_table_glitch_count_reuses_find_glitches():
    values = [10.0, 10.0, 20.0, 10.0, 10.0]  # spike-and-revert at index 2
    s = _series("A", values, [True] * 5)
    table = figdata.data_quality_table([s])
    assert table.iloc[0]["glitch_count"] == 1


def test_data_quality_table_price_range_and_volatility():
    s = _series("A", [10.0, 20.0, 10.0], [True, True, True])
    table = figdata.data_quality_table([s])
    row = table.iloc[0]
    assert row["price_min"] == 10.0
    assert row["price_max"] == 20.0
    assert row["log_return_volatility"] > 0


def test_data_quality_table_one_row_per_series():
    s1 = _series("A", [10.0, 10.0], [True, True])
    s2 = _series("B", [5.0, 5.0], [True, True])
    table = figdata.data_quality_table([s1, s2])
    assert list(table["series"]) == ["A", "B"]


def test_data_quality_table_volatility_is_nan_with_fewer_than_two_observed_points():
    s = _series("A", [10.0, 20.0, 30.0], [False, True, False])
    table = figdata.data_quality_table([s])
    assert np.isnan(table.iloc[0]["log_return_volatility"])
