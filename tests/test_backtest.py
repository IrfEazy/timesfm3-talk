import numpy as np
import pandas as pd
import pytest

from tfm3lab.backtest import (
    LOG1P_TRANSFORM,
    SeriesData,
    run_multivariate_backtest,
    run_univariate_backtest,
)
from tfm3lab.windows import target_indices, valid_origins

from .conftest import FakeForecaster


def _make_series(name: str, values, start="2024-01-01") -> SeriesData:
    values = np.asarray(values, dtype=float)
    dates = pd.date_range(start, periods=len(values), freq="D").to_numpy()
    observed = np.ones(len(values), dtype=bool)
    return SeriesData(name=name, values=values, dates=dates, observed=observed)


def test_series_data_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        SeriesData(
            name="bad",
            values=np.array([1.0, 2.0, 3.0]),
            dates=np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            observed=np.array([True, True, True]),
        )


def test_multivariate_backtest_rejects_misaligned_calendars():
    a = _make_series("a", np.arange(20.0), start="2024-01-01")
    b = _make_series("b", np.arange(20.0), start="2024-02-01")  # different calendar
    origins = valid_origins(n=20, context_len=5, horizon=3)
    with pytest.raises(ValueError, match="date-aligned"):
        run_multivariate_backtest(FakeForecaster(), [a, b], origins, context_len=5, max_horizon=3)


def test_univariate_backtest_row_count_and_target_alignment():
    a = _make_series("a", np.arange(30.0))
    b = _make_series("b", np.arange(30.0) * 10)
    context_len, horizon = 5, 3
    origins = valid_origins(n=30, context_len=context_len, horizon=horizon)

    df = run_univariate_backtest(FakeForecaster(), [a, b], origins, context_len, horizon)

    assert len(df) == 2 * len(origins) * horizon
    # spot-check target index/date arithmetic against windows.py directly
    one_origin = int(origins[0])
    expected_targets = target_indices(one_origin, horizon)
    got = df[(df["series"] == "a") & (df["origin_index"] == one_origin)].sort_values("horizon_step")
    np.testing.assert_array_equal(got["target_index"].to_numpy(), expected_targets)
    np.testing.assert_array_equal(got["target_date"].to_numpy(), a.dates[expected_targets])


def test_univariate_backtest_fake_forecast_matches_naive_baseline():
    # FakeForecaster repeats the last context value, exactly like the naive
    # baseline -> the two columns must agree everywhere in this synthetic case.
    a = _make_series("a", np.linspace(100, 130, 30))
    origins = valid_origins(n=30, context_len=5, horizon=4)
    df = run_univariate_backtest(FakeForecaster(), [a], origins, context_len=5, max_horizon=4)
    np.testing.assert_allclose(df["forecast"].to_numpy(), df["baseline_naive"].to_numpy())


def test_backtest_quantile_columns_present_and_ordered():
    a = _make_series("a", np.arange(20.0))
    origins = valid_origins(n=20, context_len=4, horizon=2)
    df = run_univariate_backtest(FakeForecaster(), [a], origins, context_len=4, max_horizon=2)
    q_cols = [f"q{q:02d}" for q in (10, 20, 30, 40, 50, 60, 70, 80, 90)]
    assert all(c in df.columns for c in q_cols)
    values = df[q_cols].to_numpy()
    assert np.all(np.diff(values, axis=1) >= -1e-9)  # monotone non-decreasing


def test_observed_flag_propagates_for_filled_points():
    values = np.arange(20.0)
    dates = pd.date_range("2024-01-01", periods=20, freq="D").to_numpy()
    observed = np.ones(20, dtype=bool)
    observed[12] = False  # pretend index 12 was forward-filled, not a real observation
    s = SeriesData(name="a", values=values, dates=dates, observed=observed)

    origins = valid_origins(n=20, context_len=4, horizon=3)
    df = run_univariate_backtest(FakeForecaster(), [s], origins, context_len=4, max_horizon=3)

    filled_rows = df[df["target_index"] == 12]
    assert len(filled_rows) > 0
    assert not filled_rows["observed"].any()
    other_rows = df[df["target_index"] != 12]
    assert other_rows["observed"].all()


def test_baseline_naive_and_drift_match_baselines_module_directly():
    from tfm3lab import baselines
    from tfm3lab.windows import context_slice

    a = _make_series("a", [1.0, 3.0, 2.0, 5.0, 4.0, 8.0, 7.0, 11.0])
    context_len, horizon = 4, 2
    origins = valid_origins(n=len(a.values), context_len=context_len, horizon=horizon)
    df = run_univariate_backtest(FakeForecaster(), [a], origins, context_len, horizon)

    one_origin = int(origins[0])
    ctx = a.values[context_slice(one_origin, context_len)]
    expected_naive = baselines.naive_forecast(ctx, horizon)
    expected_drift = baselines.drift_forecast(ctx, horizon)

    got = df[df["origin_index"] == one_origin].sort_values("horizon_step")
    np.testing.assert_allclose(got["baseline_naive"].to_numpy(), expected_naive)
    np.testing.assert_allclose(got["baseline_drift"].to_numpy(), expected_drift)


def test_seasonal_naive_column_absent_without_season_length():
    a = _make_series("a", np.arange(20.0))
    origins = valid_origins(n=20, context_len=4, horizon=2)
    df = run_univariate_backtest(FakeForecaster(), [a], origins, context_len=4, max_horizon=2)
    assert "baseline_seasonal_naive" not in df.columns


def test_seasonal_naive_column_present_with_season_length():
    a = _make_series("a", np.arange(30.0))
    origins = valid_origins(n=30, context_len=8, horizon=2)
    df = run_univariate_backtest(
        FakeForecaster(), [a], origins, context_len=8, max_horizon=2, season_length=4
    )
    assert "baseline_seasonal_naive" in df.columns


def test_multivariate_backtest_indexes_series_correctly():
    # Series "b" is 100x series "a" at every point -> with the FakeForecaster
    # (repeats last context value per variate), the stacked multivariate
    # call must still map forecast[:, j] back to the right series j, not a
    # transposed or swapped axis.
    a = _make_series("a", np.arange(1.0, 21.0))
    b = _make_series("b", np.arange(1.0, 21.0) * 100)
    origins = valid_origins(n=20, context_len=5, horizon=3)

    df = run_multivariate_backtest(FakeForecaster(), [a, b], origins, context_len=5, max_horizon=3)

    assert set(df["series"].unique()) == {"a", "b"}
    assert len(df) == 2 * len(origins) * 3
    merged = df.pivot_table(
        index=["origin_index", "horizon_step"], columns="series", values="forecast"
    )
    np.testing.assert_allclose(merged["b"].to_numpy(), merged["a"].to_numpy() * 100)


def test_log1p_transform_round_trips_to_the_same_result_as_identity():
    # FakeForecaster repeats the last CONTEXT value it's given. Under
    # LOG1P_TRANSFORM the model sees log1p(context) and its output is
    # inverse-transformed back — for this fake, that must exactly reproduce
    # the identity-transform result (both just "repeat the last raw value").
    a = _make_series("a", np.array([10.0, 20.0, 15.0, 30.0, 25.0, 40.0]))
    origins = valid_origins(n=6, context_len=3, horizon=2)

    df_identity = run_univariate_backtest(
        FakeForecaster(), [a], origins, context_len=3, max_horizon=2
    )
    df_log1p = run_univariate_backtest(
        FakeForecaster(), [a], origins, context_len=3, max_horizon=2, transform=LOG1P_TRANSFORM
    )

    np.testing.assert_allclose(df_log1p["forecast"].to_numpy(), df_identity["forecast"].to_numpy())
    np.testing.assert_allclose(
        df_log1p["baseline_naive"].to_numpy(), df_identity["baseline_naive"].to_numpy()
    )
    assert set(df_log1p["transform"].unique()) == {"log1p"}
    assert set(df_identity["transform"].unique()) == {"identity"}


def test_log1p_transform_forecast_stays_nonnegative_for_positive_series():
    a = _make_series("a", np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    origins = valid_origins(n=6, context_len=3, horizon=2)
    df = run_univariate_backtest(
        FakeForecaster(), [a], origins, context_len=3, max_horizon=2, transform=LOG1P_TRANSFORM
    )
    assert (df["forecast"] >= 0).all()


def test_multivariate_mode_label_default():
    a = _make_series("a", np.arange(20.0))
    b = _make_series("b", np.arange(20.0))
    origins = valid_origins(n=20, context_len=4, horizon=2)
    df = run_multivariate_backtest(FakeForecaster(), [a, b], origins, context_len=4, max_horizon=2)
    assert set(df["mode"].unique()) == {"timesfm3_multivariate"}


def test_univariate_backtest_correct_with_reversed_output_order():
    from .conftest import ReversedFakeForecaster

    a = _make_series("a", np.arange(30.0))
    b = _make_series("b", np.arange(30.0) * 10)
    context_len, horizon = 5, 3
    origins = valid_origins(n=30, context_len=context_len, horizon=horizon)

    df_forward = run_univariate_backtest(FakeForecaster(), [a, b], origins, context_len, horizon)
    df_reversed = run_univariate_backtest(
        ReversedFakeForecaster(), [a, b], origins, context_len, horizon
    )

    sort_cols = ["series", "origin_index", "horizon_step"]
    left = df_forward.sort_values(sort_cols).reset_index(drop=True)
    right = df_reversed.sort_values(sort_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_multivariate_backtest_correct_with_reversed_output_order():
    from .conftest import ReversedFakeForecaster

    a = _make_series("a", np.arange(1.0, 21.0))
    b = _make_series("b", np.arange(1.0, 21.0) * 100)
    origins = valid_origins(n=20, context_len=5, horizon=3)

    df_forward = run_multivariate_backtest(
        FakeForecaster(), [a, b], origins, context_len=5, max_horizon=3
    )
    df_reversed = run_multivariate_backtest(
        ReversedFakeForecaster(), [a, b], origins, context_len=5, max_horizon=3
    )

    sort_cols = ["series", "origin_index", "horizon_step"]
    left = df_forward.sort_values(sort_cols).reset_index(drop=True)
    right = df_reversed.sort_values(sort_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_series_data_rejects_unsorted_dates():
    values = np.array([1.0, 2.0, 3.0])
    dates = np.array(["2024-01-03", "2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="strictly increasing"):
        SeriesData(name="bad", values=values, dates=dates, observed=np.ones(3, dtype=bool))


def test_series_data_rejects_duplicate_dates():
    values = np.array([1.0, 2.0, 3.0])
    dates = np.array(["2024-01-01", "2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="strictly increasing"):
        SeriesData(name="bad", values=values, dates=dates, observed=np.ones(3, dtype=bool))


def test_series_data_rejects_non_finite_values():
    values = np.array([1.0, np.nan, 3.0])
    dates = pd.date_range("2024-01-01", periods=3).to_numpy()
    with pytest.raises(ValueError, match="non-finite"):
        SeriesData(name="bad", values=values, dates=dates, observed=np.ones(3, dtype=bool))


def test_series_data_accepts_valid_series():
    values = np.array([1.0, 2.0, 3.0])
    dates = pd.date_range("2024-01-01", periods=3).to_numpy()
    s = SeriesData(name="ok", values=values, dates=dates, observed=np.ones(3, dtype=bool))
    assert s.name == "ok"  # must not raise
