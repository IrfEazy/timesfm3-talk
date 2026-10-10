"""Argparse-only tests for scripts 02-05's new flags; no model, no network."""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=["02_exp_mtg.py", "03_exp_shock.py", "05_exp_covariates.py"])
def model_script(request):
    return _load("script_" + request.param.split("_")[0], request.param)


def test_model_scripts_require_as_of(model_script):
    with pytest.raises(SystemExit):
        model_script.build_parser().parse_args([])


def test_model_scripts_parse_as_of_and_allow_cpu(model_script):
    args = model_script.build_parser().parse_args(["--as-of", "2026-09-01", "--allow-cpu"])
    assert args.as_of == dt.date(2026, 9, 1)
    assert args.allow_cpu is True
    assert model_script.build_parser().parse_args(["--as-of", "2026-09-01"]).allow_cpu is False


def test_calibration_script_requires_as_of_or_live_end(capsys):
    script = _load("script_04", "04_exp_calibration.py")
    args = script.build_parser().parse_args([])
    with pytest.raises(SystemExit):
        script.resolve_as_of(args)
    args = script.build_parser().parse_args(["--as-of", "2026-09-01"])
    assert script.resolve_as_of(args) == (dt.date(2026, 9, 1), False)
    args = script.build_parser().parse_args(["--allow-live-end"])
    as_of, live = script.resolve_as_of(args)
    assert live is True and as_of == dt.date.today()
    assert "WARNING" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        script.build_parser().parse_args(["--as-of", "2026-09-01", "--allow-live-end"])


def _fake_calibration_inputs(script, monkeypatch, tmp_path):
    import pandas as pd

    from tfm3lab import config
    from tfm3lab.summarize import QUANTILE_COLUMNS

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    rows = {"observed": [True] * 4, "actual": [1.0, 2.0, 3.0, 4.0]}
    for c in QUANTILE_COLUMNS:
        rows[c] = [2.0] * 4
    base = pd.DataFrame(rows)
    monkeypatch.setattr(script, "load_mtg_h1", lambda: base.assign(regime="mtg"))
    monkeypatch.setattr(
        script,
        "load_market_by_regime",
        lambda t: base.assign(regime="market_calm", offset=10),
    )
    monkeypatch.setattr(
        script, "summarize_calibration", lambda df, group_cols: pd.DataFrame({"regime": ["mtg"]})
    )


def test_calibration_main_writes_manifest_with_live_end(tmp_path, monkeypatch):
    import json

    script = _load("script_04", "04_exp_calibration.py")
    _fake_calibration_inputs(script, monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["04", "--as-of", "2026-09-01"])
    script.main()
    (manifest,) = (tmp_path / "manifests").glob("exp_calibration-2026-09-01-*.json")
    meta = json.loads(manifest.read_text())["_meta"]
    assert meta["as_of"] == "2026-09-01" and meta["live_end"] is False

    monkeypatch.setattr(sys, "argv", ["04", "--allow-live-end"])
    script.main()
    metas = [
        json.loads(p.read_text())["_meta"] for p in (tmp_path / "manifests").glob("*.json")
    ]
    assert any(m["live_end"] is True and m["as_of"] == dt.date.today().isoformat() for m in metas)


def test_calibration_main_without_flags_exits(monkeypatch):
    script = _load("script_04", "04_exp_calibration.py")
    monkeypatch.setattr(sys, "argv", ["04"])
    with pytest.raises(SystemExit):
        script.main()


def test_mtg_main_preflight_first_and_revision_recorded(tmp_path, monkeypatch):
    import json

    import numpy as np
    import pandas as pd

    from tfm3lab import config
    from tfm3lab.backtest import SeriesData

    script = _load("script_02", "02_exp_mtg.py")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    calls: list[str] = []
    resolved = "a" * 40

    def fake_preflight(**kwargs):
        calls.append("preflight")
        return {"repo_id": "r", "revision_requested": None, "revision_resolved": resolved}

    def fake_load_series():
        calls.append("data")
        n = 200
        dates = pd.date_range("2026-01-01", periods=n)
        return [SeriesData("s", np.arange(n, dtype=float), dates, np.ones(n, dtype=bool))]

    loaded: dict = {}

    def fake_loader(revision=None):
        calls.append("loader")
        loaded["revision"] = revision
        return object()

    raw = pd.DataFrame(
        {
            "transform": ["identity"],
            "mode": ["timesfm3_univariate"],
            "baseline_drift": [0.0],
            "baseline_ets": [0.0],
            "series": ["s"],
        }
    )
    monkeypatch.setattr(script, "preflight_model", fake_preflight)
    monkeypatch.setattr(script, "load_cached_series", fake_load_series)
    monkeypatch.setattr(script, "load_forecaster", fake_loader)
    monkeypatch.setattr(script, "run_univariate_backtest", lambda *a, **k: raw)
    monkeypatch.setattr(script, "run_multivariate_backtest", lambda *a, **k: raw)
    empty = pd.DataFrame({"x": [1]})
    monkeypatch.setattr(script, "compute_mase_scales", lambda *a, **k: {})
    monkeypatch.setattr(script, "summarize_accuracy", lambda *a, **k: empty)
    monkeypatch.setattr(script, "summarize_calibration", lambda *a, **k: empty)
    monkeypatch.setattr(sys, "argv", ["02", "--as-of", "2026-09-01", "--max-origins", "2"])
    script.main()

    assert calls.index("preflight") < calls.index("data") < calls.index("loader")
    assert loaded["revision"] == resolved
    (manifest,) = (tmp_path / "manifests").glob("exp_mtg-*.json")
    meta = json.loads(manifest.read_text())["_meta"]
    assert meta["checkpoint"]["revision_resolved"] == resolved
