"""CLI test of scripts/08b_exp_mtg_inference.py with synthetic inputs."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.test_exp_a import CARDS, make_leaderboard, make_primary, make_selfcheck

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "08b_exp_mtg_inference.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("exp_mtg_inference_08b", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path):
    primary = tmp_path / "primary.parquet"
    lb = tmp_path / "lb.parquet"
    sc = tmp_path / "sc.parquet"
    pool = tmp_path / "pool.csv"
    make_primary(CARDS, {c: 0.6 for c in CARDS}).to_parquet(primary)
    make_leaderboard(CARDS, CARDS[:7]).to_parquet(lb)
    make_selfcheck().to_parquet(sc)
    pd.DataFrame({"label": CARDS, "is_showcase": [c in ("c0", "c1") for c in CARDS]}).to_csv(
        pool, index=False
    )
    return dict(primary=primary, lb=lb, sc=sc, pool=pool)


def argv(tmp_path, i, *extra):
    return [
        "--primary",
        str(i["primary"]),
        "--leaderboard",
        str(i["lb"]),
        "--pool",
        str(i["pool"]),
        "--selfcheck",
        str(i["sc"]),
        "--output-dir",
        str(tmp_path / "out"),
        "--manifest-dir",
        str(tmp_path / "man"),
        "--n-boot",
        "100",
        *extra,
    ]


def test_end_to_end_writes_outputs_and_manifest(script, inputs, tmp_path, capsys):
    assert script.main(argv(tmp_path, inputs, "--as-of", "2026-09-04")) == 0
    out = tmp_path / "out"
    for name in ("verdict", "secondary", "significance"):
        assert len(pd.read_parquet(out / f"exp_mtg_v2_{name}.parquet")) > 0
    man = json.loads((tmp_path / "man" / "exp-mtg-v2-inference-2026-09-04.json").read_text())
    assert man["_meta"]["seed"] == 20261010 and man["settings"]["n_boot"] == 100
    assert man["n_pool_csv"] == 24 and man["n_cards_in_export"] == 24
    assert man["pool_cards_missing_from_export"] == []
    assert set(man["inputs"]) == {"primary", "leaderboard", "pool", "selfcheck"}
    assert "pool h=28 beats naive" in capsys.readouterr().out


@pytest.mark.parametrize(
    "key,producer",
    [
        ("primary", "02b_exp_mtg_benchmark"),
        ("lb", "02b_exp_mtg_benchmark"),
        ("pool", "01b_select_pool"),
        ("sc", "08_inference_selfcheck"),
    ],
)
def test_missing_input_names_producer(script, inputs, tmp_path, key, producer):
    inputs[key].unlink()
    with pytest.raises(SystemExit) as e:
        script.main(argv(tmp_path, inputs, "--as-of", "2026-09-04"))
    assert producer in str(e.value)


def test_requires_as_of_or_live_end(script, inputs, tmp_path):
    with pytest.raises(SystemExit):
        script.main(argv(tmp_path, inputs))


def test_optional_inputs_absent_do_not_fail_and_showcase_says_not_run(script, inputs, tmp_path):
    assert script.main(argv(tmp_path, inputs, "--as-of", "2026-09-04")) == 0
    verdict = pd.read_parquet(tmp_path / "out" / "exp_mtg_v2_verdict.parquet")
    assert set(verdict[verdict["subset"] == "showcase_only"]["verdict"]) == {"not run"}
    sec = pd.read_parquet(tmp_path / "out" / "exp_mtg_v2_secondary.parquet")
    assert not sec["comparison"].str.startswith(("timesfm2p5", "ffill")).any()
    man = json.loads((tmp_path / "man" / "exp-mtg-v2-inference-2026-09-04.json").read_text())
    assert man["optional_inputs"] == {}
    assert "showcase_primary" in man["optional_inputs_not_present"]


def test_optional_inputs_present_are_read(script, inputs, tmp_path):
    from tests.test_exp_a import make_primary as mk_primary

    sib = inputs["primary"].parent
    show = ["s0", "s1", "s2"]
    mk_primary(show, {c: 0.7 for c in show}, seed=2).to_parquet(
        sib / "exp_mtg_benchmark_primary__preregistered_v2_showcase.parquet"
    )
    lb25 = make_leaderboard(CARDS, CARDS[:7])
    lb25 = lb25[lb25["mode"] == "timesfm3_univariate"].copy()
    lb25["mode"] = "timesfm2p5_univariate"
    lb25.to_parquet(sib / "exp_mtg_benchmark_leaderboard__preregistered_v2_timesfm2p5.parquet")
    mk_primary(CARDS, {c: 0.5 for c in CARDS}, seed=4).to_parquet(
        sib / "exp_mtg_benchmark_primary__preregistered_v2_ffill1.parquet"
    )
    assert script.main(argv(tmp_path, inputs, "--as-of", "2026-09-04")) == 0
    verdict = pd.read_parquet(tmp_path / "out" / "exp_mtg_v2_verdict.parquet")
    assert set(verdict[verdict["subset"] == "showcase_only"]["n_cards"]) == {3}
    sec = pd.read_parquet(tmp_path / "out" / "exp_mtg_v2_secondary.parquet")
    comps = set(sec["comparison"])
    assert {"timesfm2p5_vs_timesfm3", "ffill1_vs_naive", "ffill1_vs_ffill3"} <= comps
    assert "ffill0_vs_naive" not in comps
    man = json.loads((tmp_path / "man" / "exp-mtg-v2-inference-2026-09-04.json").read_text())
    assert set(man["optional_inputs"]) == {"showcase_primary", "leaderboard_2p5", "ffill1_primary"}
