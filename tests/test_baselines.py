import numpy as np
import pytest

from tfm3lab.baselines import (
    drift_forecast,
    ets_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)


def test_naive_forecast_repeats_last_value():
    np.testing.assert_array_equal(naive_forecast([1, 2, 5], horizon=3), [5, 5, 5])


def test_naive_forecast_rejects_empty_context():
    with pytest.raises(ValueError):
        naive_forecast([], horizon=2)


def test_seasonal_naive_repeats_last_season():
    context = [10, 20, 30, 40, 50, 60, 70]  # season_length=3 -> last season [50,60,70]
    np.testing.assert_array_equal(
        seasonal_naive_forecast(context, horizon=3, season_length=3), [50, 60, 70]
    )


def test_seasonal_naive_cycles_when_horizon_exceeds_season():
    context = [1, 2, 3]  # season_length=3, one full season
    got = seasonal_naive_forecast(context, horizon=7, season_length=3)
    np.testing.assert_array_equal(got, [1, 2, 3, 1, 2, 3, 1])


def test_seasonal_naive_rejects_context_shorter_than_season():
    with pytest.raises(ValueError):
        seasonal_naive_forecast([1, 2], horizon=2, season_length=3)


def test_drift_forecast_extrapolates_linear_trend_exactly():
    # context [0, 2, 4, 6]: slope = (6-0)/3 = 2 per step
    got = drift_forecast([0, 2, 4, 6], horizon=3)
    np.testing.assert_allclose(got, [8, 10, 12])


def test_drift_forecast_flat_context_gives_flat_forecast():
    got = drift_forecast([5, 5, 5, 5], horizon=2)
    np.testing.assert_allclose(got, [5, 5])


def test_drift_forecast_rejects_single_point_context():
    with pytest.raises(ValueError):
        drift_forecast([5.0], horizon=2)


def test_ets_forecast_returns_correct_shape_and_finite_values():
    rng = np.random.default_rng(1)
    context = np.arange(30, dtype=float) * 1.5 + rng.normal(scale=0.1, size=30)
    got = ets_forecast(context, horizon=5)
    assert got.shape == (5,)
    assert np.all(np.isfinite(got))


def test_ets_forecast_continues_upward_trend_direction():
    context = np.arange(20, dtype=float) * 2.0  # noiseless, clearly rising
    got = ets_forecast(context, horizon=4)
    # A trend-fit ETS forecast on a noiseless upward line must keep rising.
    assert np.all(np.diff(got) > 0)
    assert got[0] > context[-1]


def test_ets_forecast_rejects_too_short_context():
    with pytest.raises(ValueError):
        ets_forecast([1.0, 2.0], horizon=2)
