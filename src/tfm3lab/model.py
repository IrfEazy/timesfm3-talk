"""Thin, testable wrapper around timesfm3.TimesFM3Evaluator.

Experiment scripts go through this module rather than importing timesfm3
directly, for three reasons:

1. One place enforces the invariant the whole talk's calibration claims
   rest on — 9 sorted quantiles, median at a known index — via
   `assert_quantile_shape`, checked on every batch, not just in a test.
2. One place implements the "one call, many horizons" optimization: a
   single non-autoregressive decode always covers a full 64-step output
   patch internally (see config.OUTPUT_PATCH_LENGTH), so horizon=7 and
   horizon=28 cost the same forward pass. `forecast_batch` requests the
   largest horizon an experiment needs once; `BatchForecast.at_horizon`
   slices the rest for free.
3. The forecaster is an injected object, not a hardcoded import — this
   module's own logic (shape checks, batching, latency) is unit-tested
   locally with a fake, without a GPU or a Hugging Face login.
"""

from __future__ import annotations

import dataclasses
import os
import time
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from . import config


class Forecaster(Protocol):
    """Structural type matching timesfm3.TimesFM3Evaluator.predict_batch."""

    def predict_batch(
        self,
        contexts: list[np.ndarray],
        horizon: int,
        past_only_covariates: list[np.ndarray | None] | None = None,
        past_future_covariates: list[np.ndarray | None] | None = None,
        ts_ids: list[str] | None = None,
        return_quantiles: bool = True,
        use_symmetric_averaging: bool = True,
        make_positive: bool = True,
        sort_quantiles: bool = True,
        use_znorm: bool = False,
        padding_mode: str = "none",
    ) -> Any: ...


def assert_quantile_shape(quantiles: np.ndarray) -> None:
    """Fail loudly if the checkpoint's quantile grid isn't what the talk
    assumes. Cheap enough to call on every batch, not just in tests."""
    if quantiles.shape[-1] != config.N_QUANTILES:
        raise AssertionError(
            f"expected {config.N_QUANTILES} quantiles, got shape {quantiles.shape} "
            "— Google may have changed the checkpoint's quantile grid."
        )
    if np.any(np.diff(quantiles, axis=-1) < -1e-6):
        raise AssertionError("quantiles are not monotonically non-decreasing")


def load_forecaster(
    checkpoint_id: str = config.CHECKPOINT_ID,
    per_core_batch_size: int = 16,
    device: str | None = None,
) -> Forecaster:
    """Loads the real TimesFM3Evaluator.

    Requires accepting the gated checkpoint's license on Hugging Face and
    either `hf auth login` once, or an HF_TOKEN env var — see README.md.
    timesfm3 is imported inside this function, not at module load, so the
    rest of tfm3lab — and every test that injects a fake Forecaster — works
    without torch/timesfm3 as a hard import-time dependency.
    """
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    token = os.environ.get("HF_TOKEN") or True
    cfg = ModelConfig(
        checkpoint_path=checkpoint_id,
        per_core_batch_size=per_core_batch_size,
        device=device,
        token=token,
    )
    return TimesFM3Evaluator(cfg)


@dataclasses.dataclass
class BatchForecast:
    """One predict_batch call's results, sliceable to any horizon <= the
    horizon the call was actually made at."""

    ts_ids: list[str]
    forecast: np.ndarray  # shape (..., max_horizon)
    quantiles: np.ndarray  # shape (..., max_horizon, N_QUANTILES)
    latency_seconds: float
    n_series: int

    def at_horizon(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Point forecast and quantiles truncated to `horizon` steps — the
        free multi-horizon slicing described in the module docstring."""
        max_horizon = self.forecast.shape[-1]
        if horizon > max_horizon:
            raise ValueError(f"horizon {horizon} exceeds this batch's max_horizon {max_horizon}")
        return self.forecast[..., :horizon], self.quantiles[..., :horizon, :]


def forecast_batch(
    forecaster: Forecaster,
    contexts: Sequence[np.ndarray],
    max_horizon: int,
    ts_ids: list[str] | None = None,
    past_only_covariates: list[np.ndarray | None] | None = None,
    past_future_covariates: list[np.ndarray | None] | None = None,
    use_symmetric_averaging: bool = True,
    make_positive: bool = True,
) -> BatchForecast:
    """Runs one predict_batch call at `max_horizon`.

    `use_symmetric_averaging` and `make_positive` default to
    TimesFM3Evaluator's own official benchmark defaults (True, True) so
    numbers stay comparable to what Google reports; the cost/latency
    experiment (scripts/03..05) is exactly where a caller should override
    `use_symmetric_averaging=False` to measure the ~2x compute it costs
    (each context is run once as-is and once negated, then averaged).
    """
    ts_ids = list(ts_ids) if ts_ids is not None else [str(i) for i in range(len(contexts))]

    start = time.perf_counter()
    outputs = list(
        forecaster.predict_batch(
            contexts=list(contexts),
            horizon=max_horizon,
            past_only_covariates=past_only_covariates,
            past_future_covariates=past_future_covariates,
            ts_ids=ts_ids,
            return_quantiles=True,
            use_symmetric_averaging=use_symmetric_averaging,
            make_positive=make_positive,
        )
    )
    latency = time.perf_counter() - start

    forecasts = np.stack([o.forecast for o in outputs], axis=0)
    quantiles = np.stack([o.quantiles for o in outputs], axis=0)
    assert_quantile_shape(quantiles)

    return BatchForecast(
        ts_ids=[o.ts_id for o in outputs],
        forecast=forecasts,
        quantiles=quantiles,
        latency_seconds=latency,
        n_series=len(outputs),
    )
