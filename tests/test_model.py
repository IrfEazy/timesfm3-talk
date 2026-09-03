"""Unit tests for tfm3lab.model using a fake Forecaster — no GPU, no
Hugging Face login, no network. The real checkpoint is exercised only by
tests/test_model_smoke.py, opt-in via TFM3LAB_RUN_MODEL_SMOKE=1.
"""

from __future__ import annotations

import numpy as np
import pytest

from tfm3lab import config
from tfm3lab.model import BatchForecast, assert_quantile_shape, forecast_batch

from .conftest import FakeForecaster as _FakeForecaster
from .conftest import MismatchedTsIdForecaster, ReversedFakeForecaster


def test_assert_quantile_shape_accepts_valid_grid():
    quantiles = np.sort(np.random.default_rng(0).normal(size=(3, 5, config.N_QUANTILES)), axis=-1)
    assert_quantile_shape(quantiles)  # must not raise


def test_assert_quantile_shape_rejects_wrong_count():
    quantiles = np.zeros((2, 5, 7))
    with pytest.raises(AssertionError, match="expected 9 quantiles"):
        assert_quantile_shape(quantiles)


def test_assert_quantile_shape_rejects_non_monotonic():
    quantiles = np.zeros((1, 1, config.N_QUANTILES))
    quantiles[0, 0] = [0.1, 0.2, 0.9, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # dip at index 2
    with pytest.raises(AssertionError, match="monotonically"):
        assert_quantile_shape(quantiles)


def test_forecast_batch_stacks_multiple_series_and_records_latency():
    fake = _FakeForecaster()
    contexts = [np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])]
    result = forecast_batch(fake, contexts, max_horizon=5, ts_ids=["a", "b"])

    assert isinstance(result, BatchForecast)
    assert result.n_series == 2
    assert result.ts_ids == ["a", "b"]
    assert result.forecast.shape == (2, 5)
    assert result.quantiles.shape == (2, 5, config.N_QUANTILES)
    assert result.latency_seconds >= 0.0
    # fake forecaster repeats the last context value -> series "a" forecasts 3.0
    np.testing.assert_allclose(result.forecast[0], 3.0)
    np.testing.assert_allclose(result.forecast[1], 30.0)


def test_forecast_batch_defaults_match_evaluator_benchmark_defaults():
    fake = _FakeForecaster()
    forecast_batch(fake, [np.array([1.0, 2.0])], max_horizon=3)
    assert fake.last_call_kwargs["use_symmetric_averaging"] is True
    assert fake.last_call_kwargs["make_positive"] is True
    assert fake.last_call_kwargs["return_quantiles"] is True


def test_forecast_batch_generates_ts_ids_when_not_given():
    fake = _FakeForecaster()
    result = forecast_batch(fake, [np.array([1.0]), np.array([2.0])], max_horizon=1)
    assert result.ts_ids == ["0", "1"]


def test_at_horizon_slices_the_free_multi_horizon_batch():
    fake = _FakeForecaster()
    result = forecast_batch(fake, [np.array([1.0, 2.0, 5.0])], max_horizon=28, ts_ids=["only"])
    point7, quant7 = result.at_horizon(7)
    assert point7.shape == (1, 7)
    assert quant7.shape == (1, 7, config.N_QUANTILES)
    np.testing.assert_allclose(point7, result.forecast[:, :7])


def test_at_horizon_rejects_horizon_beyond_the_call():
    fake = _FakeForecaster()
    result = forecast_batch(fake, [np.array([1.0])], max_horizon=7, ts_ids=["only"])
    with pytest.raises(ValueError, match="exceeds"):
        result.at_horizon(28)


def test_forecast_batch_rejects_duplicate_ts_ids():
    fake = _FakeForecaster()
    with pytest.raises(ValueError, match="unique"):
        forecast_batch(fake, [np.array([1.0]), np.array([2.0])], max_horizon=1, ts_ids=["a", "a"])


def test_forecast_batch_rejects_ts_id_count_mismatch():
    fake = _FakeForecaster()
    with pytest.raises(ValueError, match="got 1 ts_ids"):
        forecast_batch(fake, [np.array([1.0]), np.array([2.0])], max_horizon=1, ts_ids=["a"])


def test_forecast_batch_rejects_output_ts_ids_not_matching_request():
    fake = MismatchedTsIdForecaster()
    with pytest.raises(ValueError, match="missing="):
        forecast_batch(fake, [np.array([1.0]), np.array([2.0])], max_horizon=1, ts_ids=["a", "b"])


def test_forecast_batch_preserves_reversed_output_order_in_ts_ids():
    fake = ReversedFakeForecaster()
    result = forecast_batch(
        fake, [np.array([1.0, 2.0]), np.array([10.0, 20.0])], max_horizon=1, ts_ids=["a", "b"]
    )
    assert result.ts_ids == ["b", "a"]
    # forecast[0] must belong to "b" (last context value 20.0), not "a" — BatchForecast.ts_ids
    # and BatchForecast.forecast must stay in lockstep with whatever order the forecaster used.
    np.testing.assert_allclose(result.forecast[0], 20.0)
    np.testing.assert_allclose(result.forecast[1], 2.0)
