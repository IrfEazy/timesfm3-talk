"""Unit tests for tfm3lab.model_2p5's TimesFM2p5Adapter using a fake
underlying 2.5 model -- no real checkpoint, no network, no torch download.
The real checkpoint is never loaded by this branch (no GPU run performed) --
see load_forecaster_2p5's docstring for the live path, exercised only by a
future opt-in smoke test.
"""

from __future__ import annotations

import numpy as np
import pytest
from timesfm import ForecastConfig as _RealForecastConfig

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
            # real layout: column 0 = mean (distinct), columns 1.. = quantiles
            quant = np.concatenate([(point + 99.0)[:, None], point[:, None] + levels], axis=1)
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
    assert not np.any(outputs[1].quantiles > 50)  # mean column dropped


def test_predict_batch_generates_ts_ids_when_not_given():
    adapter = TimesFM2p5Adapter(_FakeTimesFM25())
    outputs = adapter.predict_batch([np.array([1.0]), np.array([2.0])], horizon=1)
    assert [o.ts_id for o in outputs] == ["0", "1"]


def test_predict_batch_rejects_covariates():
    adapter = TimesFM2p5Adapter(_FakeTimesFM25())
    with pytest.raises(NotImplementedError, match="covariates"):
        adapter.predict_batch([np.array([1.0])], horizon=1, past_only_covariates=[np.array([0.0])])


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
        adapter.predict_batch([np.array([1.0, 2.0]), np.array([[1.0, 2.0], [3.0, 4.0]])], horizon=1)


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


def _fake_timesfm(monkeypatch, loaded):
    import sys
    import types

    class FakeModel:
        def __init__(self, **kw):
            pass

        def load_checkpoint(self, path):
            loaded.append(path)

        def compile(self, cfg):
            pass

    monkeypatch.setitem(
        sys.modules,
        "timesfm",
        types.SimpleNamespace(TimesFM_2p5_200M_torch=FakeModel, ForecastConfig=_RealForecastConfig),
    )


def test_load_forecaster_2p5_without_revision_loads_repo_id(monkeypatch):
    from tfm3lab import config
    from tfm3lab.model_2p5 import load_forecaster_2p5

    loaded = []
    _fake_timesfm(monkeypatch, loaded)
    monkeypatch.setattr(config, "CHECKPOINT_REVISION_2P5", None)
    load_forecaster_2p5("google/x")
    assert loaded == ["google/x"]


def test_load_forecaster_2p5_with_revision_uses_snapshot_download(monkeypatch):
    import huggingface_hub

    from tfm3lab.model_2p5 import load_forecaster_2p5

    loaded, calls = [], []
    _fake_timesfm(monkeypatch, loaded)

    def fake_snapshot(repo_id, revision=None, **kw):
        calls.append((repo_id, revision))
        return "/local/snap"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    load_forecaster_2p5("google/x", revision="abc")
    assert calls == [("google/x", "abc")]
    assert loaded == ["/local/snap"]


def test_load_compiles_before_forecast_with_adequate_limits(monkeypatch):
    import sys
    import types

    from tfm3lab import model_2p5

    events = []

    class Fake(_FakeTimesFM25):
        def __init__(self, **kw):
            super().__init__(n_levels=9)

        def load_checkpoint(self, path, **kw):
            events.append("load")

        def compile(self, cfg):
            events.append(("compile", cfg))

        def forecast(self, horizon, inputs):
            assert any(e[0] == "compile" for e in events if isinstance(e, tuple))
            events.append("forecast")
            return super().forecast(horizon, inputs)

    fake_mod = types.SimpleNamespace(
        TimesFM_2p5_200M_torch=Fake, ForecastConfig=_RealForecastConfig
    )
    monkeypatch.setitem(sys.modules, "timesfm", fake_mod)
    monkeypatch.setattr(config, "CHECKPOINT_REVISION_2P5", None)
    adapter = model_2p5.load_forecaster_2p5()
    cfg = next(e[1] for e in events if isinstance(e, tuple))
    assert cfg.max_context >= 512 and cfg.max_horizon >= 64
    adapter.predict_batch([np.ones(512)], horizon=64)
    assert events.index("forecast") > events.index(next(e for e in events if isinstance(e, tuple)))
    with pytest.raises(ValueError, match="max_horizon"):
        adapter.predict_batch([np.ones(8)], horizon=65)
    with pytest.raises(ValueError, match="max_context"):
        adapter.predict_batch([np.ones(513)], horizon=4)


def test_nine_quantile_columns_and_q50_is_point_forecast():
    adapter = TimesFM2p5Adapter(_FakeTimesFM25(n_levels=9))
    (out,) = adapter.predict_batch([np.array([1.0, 4.0])], horizon=3)
    assert out.quantiles.shape == (3, 9)
    np.testing.assert_allclose(out.quantiles[:, 4], out.forecast)  # q50 == median point


def test_ten_column_real_shape_through_forecast_batch():
    adapter = TimesFM2p5Adapter(_FakeTimesFM25(n_levels=9))
    result = forecast_batch(adapter, [np.array([1.0, 2.0, 5.0])], max_horizon=3, ts_ids=["s"])
    assert result.forecast.shape == (1, 3)
    np.testing.assert_allclose(result.forecast, 5.0)


def test_compile_config_batch_size_and_flags():
    from tfm3lab.model_2p5 import forecast_config_kwargs

    kw = forecast_config_kwargs()
    assert kw["per_core_batch_size"] == 128
    assert kw["force_flip_invariance"] and kw["normalize_inputs"]


def test_adapter_runs_end_to_end_through_univariate_backtest():
    import pandas as pd

    from tfm3lab.backtest import SeriesData, run_univariate_backtest
    from tfm3lab.windows import valid_origins

    vals = np.linspace(100, 130, 30)
    dates = pd.date_range("2024-01-01", periods=30, freq="D").to_numpy()
    s = SeriesData(name="a", values=vals, dates=dates, observed=np.ones(30, dtype=bool))
    origins = valid_origins(n=30, context_len=5, horizon=3)
    adapter = TimesFM2p5Adapter(_FakeTimesFM25(n_levels=9))
    df = run_univariate_backtest(adapter, [s], origins, 5, 3, mode_label="timesfm2.5")
    assert len(df) == len(origins) * 3
    np.testing.assert_allclose(df["q50"].to_numpy(), df["forecast"].to_numpy())
