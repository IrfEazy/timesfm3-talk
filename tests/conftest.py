"""Shared test fixtures: a fake Forecaster mimicking timesfm3's
predict_batch shape contract, so backtest.py and model.py logic can be unit
tested without a GPU, a Hugging Face login, or network access.
"""

from __future__ import annotations

import numpy as np

from tfm3lab import config


class FakeOutput:
    def __init__(self, ts_id, forecast, quantiles):
        self.ts_id = ts_id
        self.forecast = forecast
        self.quantiles = quantiles


class FakeForecaster:
    """Repeats each context's last value for the whole horizon — makes its
    output trivially comparable to the `naive` baseline in tests, and its
    quantiles are `point + levels`, monotone in the quantile axis by
    construction (mimicking the real API's sorted-quantile guarantee)."""

    def __init__(self, n_quantiles: int = config.N_QUANTILES):
        self.n_quantiles = n_quantiles
        self.last_call_kwargs: dict | None = None

    def predict_batch(self, contexts, horizon, **kwargs):
        self.last_call_kwargs = kwargs
        levels = np.linspace(0.1, 0.9, self.n_quantiles)
        ts_ids = kwargs.get("ts_ids") or [None] * len(contexts)
        for ts_id, ctx in zip(ts_ids, contexts, strict=True):
            ctx = np.asarray(ctx, dtype=float)
            last = ctx[-1] if ctx.ndim == 1 else ctx[:, -1]
            point = np.broadcast_to(np.asarray(last)[..., None], (*np.shape(last), horizon)).copy()
            quant = point[..., None] + levels
            yield FakeOutput(ts_id, point, quant)


class ReversedFakeForecaster(FakeForecaster):
    """Same outputs as FakeForecaster but yielded in reverse ts_id order —
    exercises callers that must not assume predict_batch preserves input
    order (the P0 "ts_id association" fix in model.py/backtest.py)."""

    def predict_batch(self, contexts, horizon, **kwargs):
        outputs = list(super().predict_batch(contexts, horizon, **kwargs))
        yield from reversed(outputs)


class MismatchedTsIdForecaster(FakeForecaster):
    """Relabels the first output's ts_id to one the caller never requested —
    simulates a forecaster that drops/renames a ts_id, which forecast_batch
    must reject rather than silently misassociate."""

    def predict_batch(self, contexts, horizon, **kwargs):
        outputs = list(super().predict_batch(contexts, horizon, **kwargs))
        if outputs:
            outputs[0].ts_id = "unrequested_id"
        yield from outputs
