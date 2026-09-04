"""Unit tests for tfm3lab.model_2p5's TimesFM2p5Adapter using a fake
underlying 2.5 model -- no real checkpoint, no network, no torch download.
The real checkpoint is never loaded by this branch (no GPU run performed) --
see load_forecaster_2p5's docstring for the live path, exercised only by a
future opt-in smoke test.
"""

from __future__ import annotations

import numpy as np
import pytest

from tfm3lab import config
from tfm3lab.model import forecast_batch
from tfm3lab.model_2p5 import TimesFM2p5Adapter


class _FakeTimesFM25:
    """Mimics TimesFM_2p5_200M_torch.forecast(horizon, inputs) -> (point, quantiles):
    repeats each input's last value; quantiles = point +/- fixed offsets,
    `n_levels` configurable so tests can prove the adapter itself doesn't
    hardcode TimesFM-3's 9-level grid (config.N_QUANTILES).
    """

    def __init__(self, n_levels: int = 5):
        self.n_levels = n_levels

    def forecast(self, horizon, inputs):
        points, quants = [], []
        levels = np.linspace(-0.2, 0.2, self.n_levels)
        for ctx in inputs:
            ctx = np.asarray(ctx, dtype=float)
            point = np.full(horizon, ctx[-1])
            quant = point[:, None] + levels
            points.append(point)
            quants.append(quant)
        return np.stack(points, axis=0), np.stack(quants, axis=0)


def test_predict_batch_shapes_and_ts_id_order():
    adapter = TimesFM2p5Adapter(_FakeTimesFM25(n_levels=5))
    contexts = [np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0])]
    outputs = adapter.predict_batch(contexts, horizon=4, ts_ids=["a", "b"])
    assert [o.ts_id for o in outputs] == ["a", "b"]
    assert outputs[0].forecast.shape == (4,)
    np.testing.assert_allclose(outputs[0].forecast, 3.0)
    np.testing.assert_allclose(outputs[1].forecast, 20.0)
    assert outputs[1].quantiles.shape == (4, 5)


def test_predict_batch_generates_ts_ids_when_not_given():
    adapter = TimesFM2p5Adapter(_FakeTimesFM25())
    outputs = adapter.predict_batch([np.array([1.0]), np.array([2.0])], horizon=1)
    assert [o.ts_id for o in outputs] == ["0", "1"]


def test_predict_batch_rejects_covariates():
    adapter = TimesFM2p5Adapter(_FakeTimesFM25())
    with pytest.raises(NotImplementedError, match="covariates"):
        adapter.predict_batch(
            [np.array([1.0])], horizon=1, past_only_covariates=[np.array([0.0])]
        )


def test_predict_batch_rejects_multivariate_context():
    # run_multivariate_backtest stacks one (n_series, context_len) array per
    # origin; TimesFM_2p5_200M_torch.forecast() has no variate concept, so
    # this must raise rather than silently forecast garbage.
    adapter = TimesFM2p5Adapter(_FakeTimesFM25())
    stacked = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(NotImplementedError, match="multivariate"):
        adapter.predict_batch([stacked], horizon=2)


def test_predict_batch_rejects_multivariate_context_among_valid_ones():
    adapter = TimesFM2p5Adapter(_FakeTimesFM25())
    with pytest.raises(NotImplementedError, match="multivariate"):
        adapter.predict_batch(
            [np.array([1.0, 2.0]), np.array([[1.0, 2.0], [3.0, 4.0]])], horizon=1
        )


def test_adapter_with_matching_quantile_grid_works_through_model_forecast_batch():
    adapter = TimesFM2p5Adapter(_FakeTimesFM25(n_levels=config.N_QUANTILES))
    result = forecast_batch(adapter, [np.array([1.0, 2.0, 5.0])], max_horizon=3, ts_ids=["only"])
    assert result.ts_ids == ["only"]
    assert result.forecast.shape == (1, 3)


def test_adapter_with_mismatched_quantile_grid_raises_loudly_through_model_forecast_batch():
    # Documents the open/unverified risk from model_2p5.py's module
    # docstring: if TimesFM-2.5's real quantile grid differs from
    # TimesFM-3's, routing it through the shared model.forecast_batch fails
    # loudly at assert_quantile_shape rather than silently producing a
    # misaligned results table.
    adapter = TimesFM2p5Adapter(_FakeTimesFM25(n_levels=5))
    with pytest.raises(AssertionError, match="expected 9 quantiles"):
        forecast_batch(adapter, [np.array([1.0])], max_horizon=2, ts_ids=["only"])
