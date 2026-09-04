import numpy as np
import pandas as pd
import pytest

from tfm3lab import config
from tfm3lab.backtest import SeriesData
from tfm3lab.metrics import in_sample_scale
from tfm3lab.summarize import (
    MIN_OBSERVATIONS_FOR_DM_TEST,
    QUANTILE_COLUMNS,
    aggregate_leaderboard,
    compute_mase_scales,
    summarize_accuracy,
    summarize_calibration,
    summarize_leaderboard,
)


def _row(mode="m", series="a", horizon_step=1, actual=0.0, observed=True, forecast=0.0,
         baseline_naive=0.0, quantiles=None) -> dict:
    quantiles = quantiles if quantiles is not None else [forecast] * config.N_QUANTILES
    row = {
        "mode": mode,
        "series": series,
        "horizon_step": horizon_step,
        "actual": actual,
        "observed": observed,
        "forecast": forecast,
        "baseline_naive": baseline_naive,
    }
    row.update(dict(zip(QUANTILE_COLUMNS, quantiles, strict=True)))
    return row


def test_compute_mase_scales_uses_only_pre_boundary_history():
    values = np.array([1.0, 3.0, 2.0, 100.0, 100.0, 100.0])  # a huge move AFTER the boundary
    dates = pd.date_range("2024-01-01", periods=6, freq="D").to_numpy()
    s = SeriesData(name="a", values=values, dates=dates, observed=np.ones(6, dtype=bool))

    boundary = 3
    scales = compute_mase_scales([s], boundary_index=boundary)
    expected = in_sample_scale(values[:boundary])
    assert scales["a"] == pytest.approx(expected)
    # sanity: the huge post-boundary move must NOT have leaked into the scale
    assert scales["a"] < 10.0


def test_summarize_accuracy_excludes_unobserved_rows():
    rows = [
        _row(actual=10.0, forecast=10.0, baseline_naive=8.0, observed=True),
        _row(actual=10.0, forecast=10.0, baseline_naive=8.0, observed=True),
        # huge error, but filled -> must not pollute the aggregate below
        _row(actual=10.0, forecast=999.0, baseline_naive=8.0, observed=False),
    ]
    df = pd.DataFrame(rows)
    summary = summarize_accuracy(df, mase_scales={"a": 1.0})
    assert len(summary) == 1
    assert summary.iloc[0]["mae_model"] == pytest.approx(0.0)
    assert summary.iloc[0]["n"] == 2


def test_summarize_accuracy_perfect_model_zero_error():
    rows = [_row(actual=v, forecast=v, baseline_naive=v + 1.0) for v in [10.0, 20.0, 30.0]]
    df = pd.DataFrame(rows)
    summary = summarize_accuracy(df, mase_scales={"a": 1.0})
    row = summary.iloc[0]
    assert row["mae_model"] == pytest.approx(0.0)
    assert row["relative_mae_vs_baseline"] == pytest.approx(0.0)
    assert row["mae_baseline_naive"] == pytest.approx(1.0)


def test_summarize_accuracy_skips_dm_test_below_threshold():
    n = MIN_OBSERVATIONS_FOR_DM_TEST - 1
    rows = [
        _row(actual=float(i), forecast=float(i), baseline_naive=float(i) + 5.0) for i in range(n)
    ]
    df = pd.DataFrame(rows)
    summary = summarize_accuracy(df, mase_scales={"a": 1.0})
    assert np.isnan(summary.iloc[0]["dm_stat"])
    assert np.isnan(summary.iloc[0]["dm_pvalue"])


def test_summarize_accuracy_runs_dm_test_above_threshold_and_favors_better_model():
    rng = np.random.default_rng(3)
    n = MIN_OBSERVATIONS_FOR_DM_TEST + 10
    actual = rng.normal(loc=100, scale=1, size=n)
    # model tracks actual closely; baseline is consistently far off.
    forecast = actual + rng.normal(scale=0.1, size=n)
    baseline = actual + 10.0 + rng.normal(scale=0.1, size=n)
    rows = [
        _row(actual=a, forecast=f, baseline_naive=b)
        for a, f, b in zip(actual, forecast, baseline, strict=True)
    ]
    df = pd.DataFrame(rows)
    summary = summarize_accuracy(df, mase_scales={"a": 1.0})
    row = summary.iloc[0]
    assert not np.isnan(row["dm_stat"])
    assert row["dm_stat"] < 0  # model's error is significantly lower
    assert row["dm_pvalue"] < 0.05


def test_summarize_accuracy_zero_baseline_error_does_not_crash():
    # A genuinely flat series: naive ("tomorrow=today") is exact, mae_baseline=0.
    # This must report a defined value, not raise (metrics.relative_mae alone
    # would raise on a zero baseline — summarize_accuracy must guard it).
    tied_rows = [_row(series="flat_tied", actual=5.0, forecast=5.0, baseline_naive=5.0)]
    worse_rows = [_row(series="flat_worse", actual=5.0, forecast=6.0, baseline_naive=5.0)]
    df = pd.DataFrame(tied_rows + worse_rows)
    summary = summarize_accuracy(df, mase_scales={"flat_tied": 1.0, "flat_worse": 1.0})

    tied = summary[summary["series"] == "flat_tied"].iloc[0]
    worse = summary[summary["series"] == "flat_worse"].iloc[0]
    assert tied["relative_mae_vs_baseline"] == pytest.approx(0.0)
    assert worse["relative_mae_vs_baseline"] == np.inf


def test_summarize_accuracy_rejects_group_cols_without_series():
    df = pd.DataFrame([_row()])
    with pytest.raises(ValueError, match="series"):
        summarize_accuracy(df, mase_scales={}, group_cols=("mode", "horizon_step"))


def test_summarize_accuracy_scale_lookup_defaults_to_one_for_unknown_series():
    rows = [_row(series="unknown_series", actual=10.0, forecast=8.0, baseline_naive=10.0)]
    df = pd.DataFrame(rows)
    summary = summarize_accuracy(df, mase_scales={"a": 5.0})  # "unknown_series" not in scales
    assert summary.iloc[0]["mase_model"] == pytest.approx(2.0)  # |10-8| / scale(1.0)


def test_summarize_calibration_hand_computed_pit_and_coverage():
    q_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # maps exactly to config.QUANTILE_LEVELS via interp
    rows = [
        _row(actual=5.0, quantiles=q_values),  # PIT exactly 0.5, inside [q10,q90]=[1,9]
        _row(actual=0.0, quantiles=q_values),  # below range -> PIT clipped to 0.1, outside [1,9]
    ]
    df = pd.DataFrame(rows)
    summary = summarize_calibration(df)
    row = summary.iloc[0]
    assert row["n"] == 2
    assert row["coverage_p10_p90"] == pytest.approx(0.5)  # only the first row is inside [1, 9]
    assert row["pit_mean"] == pytest.approx((0.5 + 0.1) / 2)


def test_summarize_calibration_excludes_unobserved_rows():
    levels_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    rows = [
        _row(actual=5.0, quantiles=levels_values, observed=True),
        _row(actual=999.0, quantiles=levels_values, observed=False),
    ]
    df = pd.DataFrame(rows)
    summary = summarize_calibration(df)
    assert summary.iloc[0]["n"] == 1


def _row_with_extra_baseline(baseline_drift=None, **kwargs) -> dict:
    row = _row(**kwargs)
    if baseline_drift is not None:
        row["baseline_drift"] = baseline_drift
    return row


def test_summarize_accuracy_includes_skill_column():
    rows = [_row(actual=10.0, forecast=9.0, baseline_naive=8.0)]
    df = pd.DataFrame(rows)
    summary = summarize_accuracy(df, mase_scales={"a": 1.0})
    row = summary.iloc[0]
    assert row["skill_vs_baseline"] == pytest.approx(1.0 - row["relative_mae_vs_baseline"])


def test_summarize_leaderboard_multiple_baseline_methods():
    rows = [
        _row_with_extra_baseline(
            actual=10.0, forecast=9.0, baseline_naive=8.0, baseline_drift=11.0
        ),
        _row_with_extra_baseline(
            actual=12.0, forecast=11.0, baseline_naive=10.0, baseline_drift=13.0
        ),
    ]
    df = pd.DataFrame(rows)
    leaderboard = summarize_leaderboard(
        df, mase_scales={"a": 1.0}, baseline_cols=("baseline_naive", "baseline_drift")
    )
    assert set(leaderboard["baseline_method"]) == {"naive", "drift"}
    assert "mae_baseline" in leaderboard.columns
    assert "mase_baseline" in leaderboard.columns
    assert len(leaderboard) == 2  # one group x 2 methods


def test_summarize_leaderboard_filters_nan_rows_per_baseline_independently():
    rows = [
        _row_with_extra_baseline(
            actual=10.0, forecast=9.0, baseline_naive=8.0, baseline_drift=None
        ),
        _row_with_extra_baseline(
            actual=12.0, forecast=11.0, baseline_naive=10.0, baseline_drift=13.0
        ),
    ]
    df = pd.DataFrame(rows)
    leaderboard = summarize_leaderboard(
        df, mase_scales={"a": 1.0}, baseline_cols=("baseline_naive", "baseline_drift")
    )
    naive_row = leaderboard[leaderboard["baseline_method"] == "naive"].iloc[0]
    drift_row = leaderboard[leaderboard["baseline_method"] == "drift"].iloc[0]
    assert naive_row["n"] == 2
    assert drift_row["n"] == 1  # the None-baseline_drift row excluded for drift only


def test_summarize_leaderboard_rejects_when_no_baseline_columns_present():
    df = pd.DataFrame([_row()])
    with pytest.raises(ValueError, match="none of"):
        summarize_leaderboard(df, mase_scales={"a": 1.0}, baseline_cols=("baseline_ets",))


def test_aggregate_leaderboard_weighted_mean_documents_the_weighting():
    df = pd.DataFrame(
        [
            {
                "mode": "m",
                "baseline_method": "naive",
                "horizon_step": 1,
                "series": "a",
                "n": 10,
                "relative_mae_vs_baseline": 0.5,
            },
            {
                "mode": "m",
                "baseline_method": "naive",
                "horizon_step": 1,
                "series": "b",
                "n": 1000,
                "relative_mae_vs_baseline": 1.5,
            },
        ]
    )
    agg = aggregate_leaderboard(df)
    row = agg.iloc[0]
    naive_mean = (0.5 + 1.5) / 2  # what a naive unweighted mean-of-ratios would give
    expected_weighted = (0.5 * 10 + 1.5 * 1000) / 1010
    assert row["relative_mae_mean_weighted"] == pytest.approx(expected_weighted)
    assert row["relative_mae_mean_weighted"] != pytest.approx(naive_mean)
    assert row["relative_mae_median"] == pytest.approx(1.0)
    assert row["skill_mean_weighted"] == pytest.approx(1.0 - expected_weighted)
    assert row["skill_median"] == pytest.approx(0.0)
    assert row["n_cards"] == 2
