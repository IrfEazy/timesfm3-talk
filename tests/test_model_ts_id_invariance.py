"""Opt-in test: does the real TimesFM-3 checkpoint's forecast depend on the
ts_id label passed to predict_batch? It shouldn't — ts_id is meant to be an
opaque tracking label, not a model input. If this test ever fails, a card's
NAME (e.g. "The One Ring [LTR]") would be a channel for metadata leakage
into the forecast, which would silently invalidate every zero-shot claim in
this project's talk (the model would be reacting to the label, not learning
from the time series alone) — see openai.review.md's ts_id-invariance ask.

Skipped by default: needs network, the gated HF checkpoint, and either
`hf auth login` or an HF_TOKEN env var — reuses the same opt-in gate as
tests/test_model_smoke.py rather than inventing a second env var.

Run explicitly with:
    TFM3LAB_RUN_MODEL_SMOKE=1 uv run pytest tests/test_model_ts_id_invariance.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tfm3lab.model import forecast_batch, load_forecaster

pytestmark = pytest.mark.skipif(
    os.environ.get("TFM3LAB_RUN_MODEL_SMOKE") != "1",
    reason=(
        "opt-in only: needs network + the gated HF checkpoint + a GPU or patience; "
        "set TFM3LAB_RUN_MODEL_SMOKE=1"
    ),
)


def test_forecast_is_invariant_to_ts_id_label():
    forecaster = load_forecaster(per_core_batch_size=1)
    rng = np.random.default_rng(0)
    context = np.cumsum(rng.normal(size=64)) + 100.0

    original = forecast_batch(forecaster, [context], max_horizon=7, ts_ids=["The One Ring [LTR]"])
    anonymous = forecast_batch(forecaster, [context], max_horizon=7, ts_ids=["series_0001"])
    randomized = forecast_batch(
        forecaster, [context], max_horizon=7, ts_ids=[f"x{int(rng.integers(0, 10**9))}"]
    )

    np.testing.assert_allclose(original.forecast, anonymous.forecast, atol=1e-6)
    np.testing.assert_allclose(original.forecast, randomized.forecast, atol=1e-6)
    np.testing.assert_allclose(original.quantiles, anonymous.quantiles, atol=1e-6)
    np.testing.assert_allclose(original.quantiles, randomized.quantiles, atol=1e-6)
