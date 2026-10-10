"""Fast test of scripts/08_inference_selfcheck.py with tiny settings."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "08_inference_selfcheck.py"


@pytest.fixture(scope="module")
def selfcheck():
    spec = importlib.util.spec_from_file_location("inference_selfcheck_08", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_as_of_is_required(selfcheck):
    with pytest.raises(SystemExit):
        selfcheck.build_parser().parse_args([])


def test_defaults_match_preregistration(selfcheck):
    args = selfcheck.build_parser().parse_args(["--as-of", "2026-10-10"])
    assert args.groups == (8, 20, 40)
    assert args.n_datasets == 400
    assert args.n_boot == 2000
    assert args.seed == 20261010
    assert selfcheck.ACCEPTANCE_BAND == (0.92, 0.98)


def test_tiny_run_writes_parquet_manifest_and_verdict(selfcheck, tmp_path, capsys):
    out = tmp_path / "sc.parquet"
    argv = ["--as-of", "2026-10-10", "--groups", "4,40", "--n-datasets", "3",
            "--n-boot", "20", "--truth-panels", "20", "--truth-cards", "10",
            "--days", "30", "--min-days", "10",
            "--output", str(out), "--manifest-dir", str(tmp_path / "m")]
    assert selfcheck.main(argv) == 0
    line, stress = capsys.readouterr().out.strip().splitlines()[-2:]
    assert stress.startswith("STRESS (sensitivity, not acceptance): ")
    assert line.startswith(("PASS:", "FAIL:"))
    assert line.endswith(
        "(baseline DGP constants chosen post-preview; see docs/inference-selfcheck.md)"
    )
    table = pd.read_parquet(out)
    assert len(table) == 12
    assert set(table["statistic"]) == {"median", "weighted_mean", "pooled"}
    assert table["used_for_acceptance"].sum() == 2
    assert set(table["scenario"]) == {"baseline", "stress"}
    assert not table.loc[table["scenario"] == "stress", "used_for_acceptance"].any()
    assert {"mc_ci_low", "mc_ci_high", "mean_estimate", "true_value", "mean_ci_width"} <= set(
        table.columns
    )
    assert (table["mc_ci_low"] <= table["coverage"] + 1e-12).all()
    assert (table["coverage"] <= table["mc_ci_high"] + 1e-12).all()
    assert (table.loc[table["n_clusters"] == 4, "label"] == "small-cluster/exploratory").all()
    manifest = json.loads((tmp_path / "m" / "inference-selfcheck-2026-10-10.json").read_text())
    assert manifest["_meta"]["as_of"] == "2026-10-10"
    assert manifest["verdict_line"] == line
    assert manifest["settings"]["seed"] == 20261010
    assert manifest["dgp_tuned_after_preview"] is True
    assert "40-dataset preview" in manifest["design_provenance"]
    assert manifest["stress_line"].startswith("STRESS (sensitivity, not acceptance): ")
    # deterministic given the seed
    selfcheck.main(argv)
    pd.testing.assert_frame_equal(table, pd.read_parquet(out))


def test_verdict_not_evaluated_without_g40(selfcheck):
    table = selfcheck.run_selfcheck(groups=(4,), n_datasets=2, n_boot=10, seed=1,
                                    n_days=20, min_days=5)
    status, line = selfcheck.verdict(table)
    assert status == "NOT_EVALUATED"
