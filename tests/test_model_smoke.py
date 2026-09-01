"""Opt-in smoke test against the REAL TimesFM-3 checkpoint.

Skipped by default: needs network, the gated checkpoint's license accepted
on huggingface.co/google/timesfm-3.0-pytorch, and either `hf auth login` or
an HF_TOKEN env var — see README.md. Runs fine on CPU for a single short
series (slow, but this is a shape/sanity check, not a benchmark).

Run explicitly with:
    TFM3LAB_RUN_MODEL_SMOKE=1 uv run pytest tests/test_model_smoke.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tfm3lab import config
from tfm3lab.model import assert_quantile_shape, forecast_batch, load_forecaster

pytestmark = pytest.mark.skipif(
    os.environ.get("TFM3LAB_RUN_MODEL_SMOKE") != "1",
    reason=(
        "opt-in only: needs network + the gated HF checkpoint + a GPU or patience; "
        "set TFM3LAB_RUN_MODEL_SMOKE=1"
    ),
)


def test_real_checkpoint_returns_expected_quantile_shape():
    forecaster = load_forecaster(per_core_batch_size=1)
    rng = np.random.default_rng(0)
    context = np.cumsum(rng.normal(size=64)) + 100.0
    result = forecast_batch(forecaster, [context], max_horizon=7, ts_ids=["smoke"])
    assert_quantile_shape(result.quantiles)
    assert result.forecast.shape == (1, 7)
    assert result.quantiles.shape == (1, 7, config.N_QUANTILES)
    assert np.all(np.isfinite(result.forecast))


def test_real_checkpoint_forecast_is_the_median_quantile():
    # Ground truth from timesfm3.TimesFM3Forecaster.predict_batch: `forecast`
    # is defined as raw[..., median_quantile_index], not a separately
    # estimated mean — this is what makes the log1p ablation (docs/talk-
    # outline.md) safe: any monotonic transform's point forecast survives
    # inversion exactly like its quantiles do, because both are the same
    # kind of statistic.
    forecaster = load_forecaster(per_core_batch_size=1)
    context = np.linspace(10, 20, 64)
    result = forecast_batch(forecaster, [context], max_horizon=1, ts_ids=["smoke2"])
    np.testing.assert_allclose(
        result.forecast[0], result.quantiles[0, :, config.MEDIAN_QUANTILE_INDEX]
    )
