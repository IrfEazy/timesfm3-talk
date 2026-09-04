"""CLI tests for scripts/02b_exp_mtg_benchmark.py -- --dry-run and argument
validation only, no forecaster load, no network. Mirrors
tests/test_fetch_data_cli.py's importlib-loading pattern (scripts/ are thin
CLIs, not part of the tfm3lab package).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "02b_exp_mtg_benchmark.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("exp_mtg_benchmark_02b", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench02b():
    return _load_script_module()


def _write_config(tmp_path, **overrides):
    payload = {
        "config_id": "test_grid",
        "context_lengths": [4, 8],
        "horizons": [1, 2],
        "modes": ["univariate", "multivariate"],
        "transforms": ["raw"],
        "make_positive": [True],
        "cards": "showcase",
    }
    payload.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_config_is_required(bench02b):
    parser = bench02b.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dry-run"])


def test_dry_run_reports_expected_combo_count(bench02b, tmp_path, monkeypatch, capsys):
    cfg_path = _write_config(tmp_path)
    # No data/cache/mtg_prices.parquet in this tmp environment -> dry-run
    # must still succeed, with n_days=0 (a NOTE printed to stderr, not a crash).
    monkeypatch.setattr(bench02b.config, "CACHE_DIR", tmp_path)
    bench02b.main(["--config", str(cfg_path), "--dry-run"])
    out = capsys.readouterr()
    report = json.loads(out.out)
    assert report["config_id"] == "test_grid"
    # 2 context_lengths x 2 horizons x 1 transform x 1 make_positive x 2 modes
    assert report["n_combos"] == 8
    assert "NOTE" in out.err


def test_dry_run_out_writes_file_instead_of_stdout(bench02b, tmp_path, monkeypatch):
    cfg_path = _write_config(tmp_path)
    monkeypatch.setattr(bench02b.config, "CACHE_DIR", tmp_path)
    out_path = tmp_path / "report.json"
    bench02b.main(["--config", str(cfg_path), "--dry-run", "--dry-run-out", str(out_path)])
    report = json.loads(out_path.read_text())
    assert report["config_id"] == "test_grid"


def test_cards_benchmark_without_path_exits(bench02b, tmp_path):
    cfg_path = _write_config(tmp_path)
    with pytest.raises(SystemExit, match="requires a manifest path"):
        bench02b.main(["--config", str(cfg_path), "--dry-run", "--cards", "benchmark"])


def test_cards_showcase_dry_run_uses_seven_card_pool(bench02b, tmp_path, monkeypatch, capsys):
    cfg_path = _write_config(tmp_path)
    monkeypatch.setattr(bench02b.config, "CACHE_DIR", tmp_path)
    bench02b.main(["--config", str(cfg_path), "--dry-run"])
    report = json.loads(capsys.readouterr().out)
    assert report["card_pool_size"] == 7
