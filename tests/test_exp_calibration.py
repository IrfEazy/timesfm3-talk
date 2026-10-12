import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from tfm3lab import config

from .test_calibration import _mtg_primary, _shock_pred

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "04_exp_calibration.py"


def _load():
    spec = importlib.util.spec_from_file_location("script_04v2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_04v2"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_missing_input_names_producer(tmp_path, monkeypatch, capsys):
    mod = _load()
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["04", "--as-of", "2026-09-01"])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert "02b_exp_mtg_benchmark.py" in str(e.value)
    _mtg_primary(n_days=5).to_parquet(tmp_path / "exp_mtg_benchmark_primary.parquet")
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert "03_exp_shock.py" in str(e.value)
    assert not list(tmp_path.glob("exp_calibration_v2_*"))


def test_main_writes_v2_outputs_and_manifest(tmp_path, monkeypatch, capsys):
    mod = _load()
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _mtg_primary(n_days=90).to_parquet(tmp_path / "exp_mtg_benchmark_primary.parquet")
    _shock_pred().to_parquet(tmp_path / "exp_shock_v2_predictions.parquet")
    monkeypatch.setattr(sys, "argv", ["04", "--as-of", "2026-09-01", "--n-boot", "50"])
    mod.main()
    cov = pd.read_parquet(tmp_path / "exp_calibration_v2_coverage.parquet")
    conf = pd.read_parquet(tmp_path / "exp_calibration_v2_conformal.parquet")
    assert len(cov) == 3 + 3 + 4 and len(conf) == len(cov) + 4
    assert (conf["analysis"] == "exploratory_post_hoc_resplit").sum() == 4
    assert "Exploratory post-hoc MTG split date" in capsys.readouterr().out
    assert not (tmp_path / "exp_calibration_curve.parquet").exists()
    (manifest,) = (tmp_path / "manifests").glob("exp_calibration-2026-09-01-*.json")
    meta = json.loads(manifest.read_text())["_meta"]
    assert meta["as_of"] == "2026-09-01" and meta["live_end"] is False

    monkeypatch.setattr(sys, "argv", ["04", "--allow-live-end", "--n-boot", "50"])
    mod.main()
    metas = [json.loads(p.read_text())["_meta"] for p in (tmp_path / "manifests").glob("*.json")]
    assert any(m["live_end"] is True and m["as_of"] == dt.date.today().isoformat() for m in metas)


def test_unfitted_cqr_warns_and_is_null(tmp_path, monkeypatch, capsys):
    mod = _load()
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    # every MTG origin is after the split date -> empty calibration split
    _mtg_primary(n_days=30, start="2025-07-01").to_parquet(
        tmp_path / "exp_mtg_benchmark_primary.parquet"
    )
    _shock_pred().to_parquet(tmp_path / "exp_shock_v2_predictions.parquet")
    monkeypatch.setattr(sys, "argv", ["04", "--as-of", "2026-09-01", "--n-boot", "50"])
    mod.main()
    err = capsys.readouterr().err
    assert "CQR not fitted for regime=mtg horizon=1" in err
    conf = pd.read_parquet(tmp_path / "exp_calibration_v2_conformal.parquet")
    r = conf[
        (conf["analysis"] == "preregistered") & (conf["regime"] == "mtg") & (conf["horizon"] == 1)
    ].iloc[0]
    assert r["status"] != "ok" and pd.isna(r["conformal_fixes_it"])
