"""End-to-end scripts/05 on tiny synthetic data with a fake forecaster."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfm3lab import covariates as cv
from tfm3lab.backtest import SeriesData

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "05_exp_covariates.py"


@pytest.fixture(scope="module")
def exp05():
    spec = importlib.util.spec_from_file_location("exp_covariates_05_v2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CovForecaster:
    """Forecast = last value, nudged by the covariate's mean when present."""

    def __init__(self):
        self.calls = []

    def predict_batch(self, contexts, horizon, **kwargs):
        from tests.conftest import FakeOutput

        self.calls.append(kwargs)
        covs = kwargs.get("past_future_covariates") or [None] * len(contexts)
        levels = np.linspace(0.1, 0.9, 9)
        for ts_id, ctx, cov in zip(kwargs["ts_ids"], contexts, covs, strict=True):
            ctx = np.asarray(ctx, dtype=float)
            nudge = 0.0 if cov is None else 0.001 * float(np.mean(cov[0]))
            point = np.full(horizon, ctx[-1] * (1 + nudge))
            yield FakeOutput(ts_id, point, point[:, None] + levels)


def _series(name, n=80, start="2025-01-01", seed=0):
    rng = np.random.default_rng(seed)
    vals = 1000.0 + np.cumsum(rng.normal(0, 5, n))
    dates = pd.date_range(start, periods=n).to_numpy()
    return SeriesData(name, vals, dates, np.ones(n, dtype=bool))


def test_end_to_end(exp05, tmp_path):
    market = [_series("SP500", seed=1)]
    mtg = [_series(f"card{i}", seed=10 + i) for i in range(3)]
    release = np.array(["2025-02-15", "2025-03-20"], dtype="datetime64[D]")
    f = CovForecaster()
    paths = exp05.run_experiment(
        f,
        market_series=market,
        mtg_series=mtg,
        release_dates=release,
        excluded=[{"domain": "mtg", "series": "gone", "reason": "no cached price series"}],
        results_dir=tmp_path,
        context_len=16,
        horizon=7,
        stride=7,
        n_boot=100,
    )
    preds = pd.read_parquet(paths["predictions"])
    summary = pd.read_parquet(paths["summary"])
    leak = pd.read_parquet(paths["leakage_demo"])

    assert set(preds["arm"]) == {"no_covariate", "legit", "placebo"}
    assert (
        "leaked" not in set(summary.get("arm", []))
        and not summary.astype(str).apply(lambda c: c.str.contains("LEAKED")).any().any()
    )
    assert set(leak["arm"]) == {"leaked"}
    assert (leak["watermark"] == cv.LEAKED_WATERMARK).all()
    # normalized, no raw prices
    for df in (preds, leak):
        assert "actual" not in df.columns and "forecast" not in df.columns
        assert df["actual_norm"].abs().max() < 10
    assert set(summary["comparison"]) == {"legit_vs_no_covariate", "legit_vs_placebo"}
    assert set(summary["verdict"]) <= {"helps", "does not help", "inconclusive"}
    assert set(summary["domain"]) == {"market", "mtg"}
    assert (summary["label"] == "small-cluster/exploratory").all()
    assert summary["n_clusters"].max() == 3
    ex = pd.read_parquet(paths["excluded"])
    assert list(ex["series"]) == ["gone"]
    # legacy names not written
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {
        exp05.OUT_PREDICTIONS,
        exp05.OUT_SUMMARY,
        exp05.OUT_EXCLUDED,
        exp05.OUT_LEAKAGE,
    }
    assert not any("market_dow" in n or "mtg_release" in n for n in names)
    # origin stride 7
    o = np.sort(preds[preds["series"] == "SP500"]["origin_index"].unique())
    assert set(np.diff(o)) == {7}
    assert (preds["horizon_step"].max()) == 7


def test_leaked_arm_requires_teaching_path(exp05):
    s = _series("c")
    origins = np.array([20, 27])
    with pytest.raises(cv.CovariateAvailabilityError):
        exp05.run_arm(
            CovForecaster(),
            s,
            origins,
            "leaked",
            cv.LEAKED_FUTURE,
            lambda ser, o: cv.teaching_only_leaked_covariate(ser.values, o, 16, 7),
            context_len=16,
            horizon=7,
            domain="mtg",
        )
    # a legit-arm run with a not-future-known spec is refused too
    with pytest.raises(cv.CovariateAvailabilityError):
        exp05.run_arm(
            CovForecaster(),
            s,
            origins,
            "legit",
            cv.LEAKED_FUTURE,
            lambda ser, o: np.zeros((1, 23)),
            context_len=16,
            horizon=7,
            domain="mtg",
        )


def test_build_mtg_inputs_excludes_with_reasons(exp05, tmp_path, monkeypatch):
    pool = tmp_path / "pool.csv"
    pool.write_text(
        "label,group_abbreviation,product_name,product_id\n"
        "A [AAA],AAA,A,1\nB [BBB],BBB,B,2\nC [CCC],CCC,C,3\n"
    )
    monkeypatch.setattr(
        exp05, "_load_series", lambda name: [_series("A [AAA]"), _series("B [BBB]")]
    )
    monkeypatch.setattr(
        exp05,
        "fetch_release_map",
        lambda abbrevs: {"AAA": np.datetime64("2025-02-01"), "CCC": np.datetime64("2025-03-01")},
    )
    series, release, excluded = exp05.build_mtg_inputs(pool)
    assert [s.name for s in series] == ["A [AAA]"]
    reasons = {e["series"]: e["reason"] for e in excluded}
    assert reasons["B [BBB]"] == "set release date unresolved"
    assert reasons["C [CCC]"] == "no cached price series"
    assert list(release) == [np.datetime64("2025-02-01")]


def test_per_series_skill_equals_raw_error_ratio(exp05):
    s = _series("c", seed=3)
    origins = np.array([20, 27, 34])
    kw = {"context_len": 16, "horizon": 7, "domain": "mtg"}
    base = exp05.run_arm(CovForecaster(), s, origins, "no_covariate", None, None, **kw)
    legit = exp05.run_arm(
        CovForecaster(),
        s,
        origins,
        "legit",
        cv.DAYS_TO_RELEASE,
        lambda ser, o: np.full((1, 23), 5.0),
        **kw,
    )
    assert "abs_err_norm" not in base.columns
    assert base["abs_err_scaled"].notna().all() and legit["abs_err_scaled"].notna().all()
    fc_base = np.array([s.values[o - 1] for o in base["origin_index"]])
    idx = (base["origin_index"] + base["horizon_step"] - 1).to_numpy()
    actual = s.values[idx]
    raw_base = np.abs(fc_base - actual).sum()
    nudge = 0.001 * 5.0
    raw_legit = np.abs(fc_base * (1 + nudge) - actual).sum()
    got = 1 - legit["abs_err_scaled"].sum() / base["abs_err_scaled"].sum()
    assert got == pytest.approx(1 - raw_legit / raw_base)
