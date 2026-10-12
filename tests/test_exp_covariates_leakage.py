"""Leaked-arm wiring and exclusion for scripts/05_exp_covariates.py (v2).

The leaked arm hands the model the series' own real past plus real future; it
is reachable only through the explicit teaching-only path and may never enter
a summary. Loaded via importlib (scripts/ are CLIs, not package modules).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfm3lab import covariates as cv
from tfm3lab.backtest import SeriesData

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "05_exp_covariates.py"


@pytest.fixture(scope="module")
def exp05():
    spec = importlib.util.spec_from_file_location("exp_covariates_05", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CapturingForecaster:
    def __init__(self):
        self.calls: list[dict] = []

    def predict_batch(self, contexts, horizon, **kwargs):
        from tests.conftest import FakeOutput

        self.calls.append(kwargs)
        levels = np.linspace(0.1, 0.9, 9)
        for ts_id, ctx in zip(kwargs["ts_ids"], contexts, strict=True):
            point = np.full(horizon, float(np.asarray(ctx)[-1]))
            yield FakeOutput(ts_id, point, point[:, None] + levels)


def _series(n=60):
    values = np.linspace(10.0, 20.0, n) + np.sin(np.linspace(0, 12, n))
    return SeriesData("Card", values, np.arange(n).astype("datetime64[D]"), np.ones(n, bool))


def test_leaked_covariate_is_real_past_plus_real_future(exp05):
    s = _series()
    f = CapturingForecaster()
    origins = np.array([20, 27])
    exp05.run_arm(
        f,
        s,
        origins,
        cv.LEAKED_ARM,
        cv.LEAKED_FUTURE,
        lambda ser, o: cv.teaching_only_leaked_covariate(ser.values, o, 16, 7),
        context_len=16,
        horizon=7,
        domain="mtg",
        teaching_only=True,
    )
    for origin, covar in zip(origins, f.calls[0]["past_future_covariates"], strict=True):
        np.testing.assert_allclose(covar[0, :16], s.values[origin - 16 : origin])
        np.testing.assert_allclose(covar[0, 16:], s.values[origin : origin + 7])
        assert np.std(covar[0, :16]) > 1e-6


def test_leaked_rows_never_enter_summary_or_main_outputs(exp05, tmp_path):
    mtg = [_series(), _series()]
    mtg[1] = SeriesData("Card2", mtg[1].values * 1.1, mtg[1].dates, mtg[1].observed)
    paths = exp05.run_experiment(
        CapturingForecaster(),
        market_series=[],
        mtg_series=mtg,
        release_dates=np.array(["1970-01-30"], dtype="datetime64[D]"),
        excluded=[],
        results_dir=tmp_path,
        context_len=16,
        horizon=7,
        stride=7,
        n_boot=50,
    )
    for key in ("predictions", "summary"):
        df = pd.read_parquet(paths[key])
        assert "leaked" not in set(df.get("arm", []))
        assert not df.astype(str).apply(lambda c: c.str.contains("LEAKED")).any().any()
    leak = pd.read_parquet(paths["leakage_demo"])
    assert set(leak["arm"]) == {"leaked"}
    assert (leak["watermark"] == "LEAKED — teaching contrast, not a result").all()
    with pytest.raises(cv.LeakedRowsError):
        cv.paired_skill_summary(leak, primary_horizon=7, n_boot=20)


def test_slides_leakage_slide_has_no_numbers():
    tpl = (ROOT / "slides" / "talk.md.template").read_text(encoding="utf-8")
    slide = next(s for s in tpl.split("\n---\n") if s.lstrip().startswith("## LEAKED"))
    assert "LEAKED - teaching contrast, not a result" in slide
    assert "{{" not in slide
    assert not re.search(r"\d", slide.replace("v2_covariates_leakage_demo.png", ""))
