"""Regression test for scripts/05_exp_covariates.py's leakage_demo wiring.

Loaded via importlib rather than a package import: scripts/ are thin CLIs,
not part of the tfm3lab package, so this mirrors how `uv run scripts/...`
actually executes the file while still exercising `leakage_demo` directly,
without a GPU or the gated checkpoint (a fake forecaster stands in for
timesfm3.TimesFM3Evaluator, same pattern as tests/conftest.py's
FakeForecaster).

What this locks in: the root-cause fix for the leakage negative control
that didn't fire (docs/talk-outline.md, "Esperimento D"). The original
"leaked" arm padded the covariate's PAST with a constant, which TimesFM-3
normalizes with its own per-variate RevIN stats — a constant past has zero
variance, gets clamped to an arbitrary scale, and gives the model nothing
to recognize as "this tracks the target". The fix hands the covariate its
own real past. This test checks the WIRING (what past_future_covariates
the forecaster actually receives), not the model's response to it — that
part needs a real forecaster and is out of scope for a unit test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from tfm3lab.backtest import SeriesData
from tfm3lab.windows import context_slice, target_indices, valid_origins

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "05_exp_covariates.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("exp_covariates_05", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exp05():
    return _load_script_module()


class CapturingForecaster:
    """Records the past_future_covariates array passed to each call and
    returns a trivial flat forecast — this test only checks wiring, not
    what a real model would do with the covariate."""

    def __init__(self):
        self.calls: list[dict] = []

    def predict_batch(self, contexts, horizon, **kwargs):
        self.calls.append(kwargs)
        from tests.conftest import FakeOutput

        levels = np.linspace(0.1, 0.9, 9)
        ts_ids = kwargs.get("ts_ids") or [None] * len(contexts)
        for ts_id, ctx in zip(ts_ids, contexts, strict=True):
            ctx = np.asarray(ctx, dtype=float)
            point = np.full(horizon, ctx[-1])
            quant = point[:, None] + levels
            yield FakeOutput(ts_id, point, quant)


def _series(values) -> SeriesData:
    values = np.asarray(values, dtype=float)
    n = len(values)
    dates = np.arange(n).astype("datetime64[D]")
    return SeriesData(name="Test Card", values=values, dates=dates, observed=np.ones(n, dtype=bool))


def test_leaked_arm_covariate_past_matches_real_context(exp05):
    values = np.linspace(10.0, 20.0, 60) + np.sin(np.linspace(0, 12, 60))
    s = _series(values)
    origins = valid_origins(len(values), exp05.CONTEXT_LEN, exp05.HORIZON, max_origins=3)

    forecaster = CapturingForecaster()
    exp05.leakage_demo(forecaster, s, origins)

    leaked_calls = [c for c in forecaster.calls if c["ts_ids"][0].endswith("::leaked")]
    assert leaked_calls, "no call made for the 'leaked' arm"
    for call in leaked_calls:
        for origin, cov in zip(origins, call["past_future_covariates"], strict=True):
            origin = int(origin)
            expected_past = s.values[context_slice(origin, exp05.CONTEXT_LEN)]
            got_past = cov[0, : exp05.CONTEXT_LEN]
            np.testing.assert_allclose(got_past, expected_past)
            expected_future = s.values[target_indices(origin, exp05.HORIZON)]
            np.testing.assert_allclose(cov[0, exp05.CONTEXT_LEN :], expected_future)


def test_leaked_arm_covariate_past_has_nonzero_variance(exp05):
    # Real price series are never perfectly flat over a 32-day context —
    # guards against ever reintroducing the constant-past bug (zero
    # variance -> RevIN clamps the covariate to an arbitrary scale).
    values = np.linspace(10.0, 20.0, 60) + np.sin(np.linspace(0, 12, 60))
    s = _series(values)
    origins = valid_origins(len(values), exp05.CONTEXT_LEN, exp05.HORIZON, max_origins=3)

    forecaster = CapturingForecaster()
    exp05.leakage_demo(forecaster, s, origins)

    leaked_calls = [c for c in forecaster.calls if c["ts_ids"][0].endswith("::leaked")]
    for call in leaked_calls:
        for cov in call["past_future_covariates"]:
            past = cov[0, : exp05.CONTEXT_LEN]
            assert np.std(past) > 1e-6


def test_leaked_flat_past_arm_still_has_a_constant_past(exp05):
    # Documents the dead-end arm's actual behavior (kept deliberately, see
    # leakage_demo's docstring) rather than silently changing meaning.
    values = np.linspace(10.0, 20.0, 60)
    s = _series(values)
    origins = valid_origins(len(values), exp05.CONTEXT_LEN, exp05.HORIZON, max_origins=2)

    forecaster = CapturingForecaster()
    exp05.leakage_demo(forecaster, s, origins)

    flat_calls = [c for c in forecaster.calls if c["ts_ids"][0].endswith("::leaked_flat_past")]
    assert flat_calls
    for call in flat_calls:
        for cov in call["past_future_covariates"]:
            past = cov[0, : exp05.CONTEXT_LEN]
            assert np.std(past) == pytest.approx(0.0)


def test_clean_arm_gets_no_covariate(exp05):
    values = np.linspace(10.0, 20.0, 60)
    s = _series(values)
    origins = valid_origins(len(values), exp05.CONTEXT_LEN, exp05.HORIZON, max_origins=2)

    forecaster = CapturingForecaster()
    exp05.leakage_demo(forecaster, s, origins)

    clean_calls = [c for c in forecaster.calls if c["ts_ids"][0].endswith("::clean")]
    assert clean_calls
    for call in clean_calls:
        assert call["past_future_covariates"] is None


def test_leakage_demo_output_has_three_arms(exp05):
    values = np.linspace(10.0, 20.0, 60)
    s = _series(values)
    origins = valid_origins(len(values), exp05.CONTEXT_LEN, exp05.HORIZON, max_origins=2)

    forecaster = CapturingForecaster()
    df = exp05.leakage_demo(forecaster, s, origins)

    assert set(df["arm"]) == {"clean", "leaked_flat_past", "leaked"}
    # back-compat boolean: only "clean" is not leaked
    assert set(df.loc[~df["leaked"], "arm"]) == {"clean"}
    assert set(df.loc[df["leaked"], "arm"]) == {"leaked_flat_past", "leaked"}
