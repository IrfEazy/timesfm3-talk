import datetime as dt
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tfm3lab import shock
from tfm3lab.backtest import SeriesData

from .conftest import FakeForecaster

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _panel(n=700):
    dates = pd.bdate_range("2018-01-01", periods=n).to_numpy()
    out = {}
    for i, label in enumerate(("SP500", "Gold", "Nikkei225")):
        rng = np.random.default_rng(i)
        v = 1000 * (i + 1) * np.exp(np.cumsum(rng.normal(scale=0.005, size=n)))
        out[label] = SeriesData(label, v, dates, np.ones(n, bool))
    return out


def test_cli_requires_as_of_and_has_flags():
    mod = _load("s03", "03_exp_shock.py")
    args = mod.build_parser().parse_args(
        ["--as-of", "2026-09-04", "--adapter", "timesfm2.5", "--legacy-events"]
    )
    assert args.adapter == "timesfm2.5" and args.legacy_events


def test_rule_events_skips_excluded(tmp_path):
    mod = _load("s03", "03_exp_shock.py")
    csv = tmp_path / "e.csv"
    csv.write_text(
        "event_id,onset_date,arm,onset_index,triggering_indices,max_abs_z,excluded_reason\n"
        "a,2020-01-02,pre,Gold,Gold,5,\nb,2023-07-01,guard,Gold,Gold,5,guard_band\n"
    )
    specs = mod.rule_events(csv)
    assert [s.event_id for s in specs] == ["a"]


def test_end_to_end_writes_normalized_outputs_only(tmp_path):
    mod = _load("s03", "03_exp_shock.py")
    series = _panel()
    dates = series["SP500"].dates
    specs = [
        shock.EventSpec("e1", "rule", "pre", pd.Timestamp(dates[560]).date(), "SP500"),
        shock.EventSpec("e2", "rule", "pre", pd.Timestamp(dates[600]).date(), "Gold"),
        shock.EventSpec("e3", "rule", "post", pd.Timestamp(dates[620]).date(), "SP500"),
        shock.EventSpec("e4", "rule", "post", pd.Timestamp(dates[640]).date(), "Nikkei225"),
        shock.EventSpec("early", "rule", "post", pd.Timestamp(dates[100]).date(), "SP500"),
    ]
    paths, notes = mod.run_experiment(
        FakeForecaster(), "timesfm3", specs, series, tmp_path, n_boot=50
    )
    assert any("early" in n for n in notes)
    pred = pq.read_table(paths[0]).to_pandas()
    lags = pd.read_parquet(paths[1])
    summ = pd.read_parquet(paths[2])

    assert set(pred.columns) == {
        "event_id",
        "source",
        "arm",
        "forecaster",
        "perturbation",
        "seed",
        "origin_offset",
        "horizon_step",
        "actual_norm",
        "forecast_norm",
        "ape",
        "q10_norm",
        "q50_norm",
        "q90_norm",
        "origin_date",
        "target_date",
        "onset_index",
    }
    tfm = pred[(pred["forecaster"] == "timesfm3_univariate") & (pred["perturbation"] == "none")]
    assert tfm[["q10_norm", "q90_norm", "origin_date", "target_date"]].notna().all().all()
    assert (tfm["q10_norm"] <= tfm["q90_norm"]).all()
    assert (tfm["target_date"] >= tfm["origin_date"]).all()
    assert pred[pred["forecaster"] == "naive"]["q10_norm"].isna().all()
    assert pred["actual_norm"].between(0.5, 2).all()  # normalized, not raw prices (~1000+)
    assert {"timesfm3_univariate", "timesfm3_multivariate", "naive", "ets"} <= set(
        pred["forecaster"]
    )
    assert set(pred["perturbation"]) == {"none", "noise", "splice"}
    assert pred[pred["perturbation"] != "none"]["horizon_step"].max() == 5
    assert pred["horizon_step"].max() == 28
    assert set(pred["event_id"]) == {"e1", "e2", "e3", "e4"}

    # The fake repeats the last value == naive, so D is exactly 0 for it.
    m = lags[(lags["forecaster"] == "timesfm3_univariate") & (lags["perturbation"] == "none")]
    assert (m["d_naive"] == 0).all()
    assert {"censored", "multiplier", "lag"} <= set(lags.columns)
    assert set(lags["multiplier"]) == {1.25, 1.5, 2.0}
    assert "promotion" in set(summ["metric"])
    promo = summ[summ["metric"] == "promotion"].iloc[0]
    assert promo["verdict"] == "appendix"  # only 2 events per arm
    assert "anticipation_hit_rate" in set(summ["metric"])
    assert {"n_pre_noise", "n_post_splice", "notes"} <= set(promo.index)
    assert "unperturbed naive/ets baseline lags" in promo["notes"]
    defs = summ.groupby("metric")["effect_definition"].first()
    assert defs["anticipation_hit_rate"] == "pre_minus_post"
    assert defs[[m for m in defs.index if m.startswith("lag_D")]].eq("post_minus_pre").all()
    assert not (tmp_path / "exp_shock_raw_predictions.parquet").exists()


def test_adapter_2p5_suffix_and_no_multivariate_or_promotion(tmp_path):
    mod = _load("s03", "03_exp_shock.py")
    series = _panel()
    dates = series["SP500"].dates
    specs = [
        shock.EventSpec("e1", "rule", "pre", pd.Timestamp(dates[600]).date(), "SP500"),
        shock.EventSpec("e3", "rule", "post", pd.Timestamp(dates[640]).date(), "SP500"),
    ]
    paths, _ = mod.run_experiment(
        FakeForecaster(), "timesfm2p5", specs, series, tmp_path, perturb=False, n_boot=20
    )
    assert all(p.name.endswith("_2p5.parquet") for p in paths)
    pred = pd.read_parquet(paths[0])
    assert "timesfm3_multivariate" not in set(pred["forecaster"])
    assert "promotion" not in set(pd.read_parquet(paths[2])["metric"])


def test_legacy_events_marked_and_separate():
    mod = _load("s03", "03_exp_shock.py")
    leg = mod.legacy_events()
    assert leg and all(s.source == "legacy" and s.arm in ("pre", "post") for s in leg)
    assert isinstance(leg[0].onset_date, dt.date)
