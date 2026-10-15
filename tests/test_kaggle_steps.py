"""Offline tests of the opt-in `v3_forecasts` Kaggle step: its place in the step
table, how a session treats it, how its requests file travels (private cache
dataset, cache manifest) and where its outputs end up in the packed archives."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from tests.test_kaggle_notebook import (
    NOTEBOOK,
    _code_cells,
    _load_script,
    _make_cache,
    _session,
    _src,
)
from tfm3lab import config
from tfm3lab import kaggle as kg
from tfm3lab.studies import requests as rq

ROOT = Path(__file__).resolve().parents[1]
AS_OF = "2026-09-04"
V2_IDS = [
    "probe",
    "02b_preregistered_v2",
    "02b_timesfm2p5",
    "02b_showcase",
    "02b_ffill0",
    "02b_ffill1",
    "03_timesfm3",
    "03_timesfm2p5",
    "05_covariates",
    "09_latency_timesfm3",
    "09_latency_timesfm2p5",
]


def _requests_file(path: Path) -> Path:
    df = pd.DataFrame(
        {
            "request_id": ["a", "b"],
            "study": ["S1", "S2"],
            "block": [1, 2],
            "horizon": [3, 5],
            "context": [[1.0, 1.1, 1.2], [1.0, 0.9]],
        }
    )
    return rq.write_requests(df, path)


# ------------------------------------------------------------------ step table


def test_v3_step_is_appended_last_and_the_existing_steps_are_unchanged():
    assert [s.step_id for s in kg.STEPS] == [*V2_IDS, "v3_forecasts"]
    step = kg.STEP_BY_ID["v3_forecasts"]
    assert step == kg.Step(
        "v3_forecasts",
        "scripts/14_v3_forecast.py",
        ("--resume",),
        "timesfm3",
        "v3 studies: S1/S2/S3 forecasts",
    )
    assert (ROOT / step.script).is_file()
    assert kg.validate_experiments(["probe", "v3_forecasts"]) == ["probe", "v3_forecasts"]


def test_v3_step_is_opt_in_and_never_part_of_the_default_session():
    assert set(kg.OPT_IN_STEPS) == {"v3_forecasts"}
    assert list(kg.DEFAULT_EXPERIMENTS) == V2_IDS
    assert "v3_forecasts" in kg.STEP_BY_ID


def test_v3_step_command_line():
    step = kg.STEP_BY_ID["v3_forecasts"]
    argv = kg.step_argv(step, AS_OF)
    assert argv == [
        sys.executable,
        "-m",
        "uv",
        "run",
        "--no-sync",
        "scripts/14_v3_forecast.py",
        "--resume",
        "--as-of",
        AS_OF,
    ]
    assert kg.step_argv(step, AS_OF, allow_cpu=True)[-1] == "--allow-cpu"


def test_v3_script_parses_the_step_command_line(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setattr(config, "CACHE_DIR", cache)
    script = _load_script("14_v3_forecast.py")
    argv = kg.step_argv(kg.STEP_BY_ID["v3_forecasts"], AS_OF, allow_cpu=True)
    args = script.build_parser().parse_args(argv[6:])
    assert args.resume is True and args.allow_cpu is True and str(args.as_of) == AS_OF
    assert args.adapter == kg.STEP_BY_ID["v3_forecasts"].adapter == "timesfm3"
    # found like every other cache file: <data root>/cache/<name>
    assert args.requests == cache / "v3_requests.parquet"


# ------------------------------------------------------------------ session behaviour


def test_default_session_skips_the_v3_step_without_running_it(tmp_path, monkeypatch):
    s, calls = _session(tmp_path, monkeypatch, {})
    rec = s.run("v3_forecasts")
    assert calls == [] and rec["status"] == "skipped" and rec["reason"] == "not in EXPERIMENTS"


def test_probe_plus_v3_session_runs_the_script_and_records_it(tmp_path, monkeypatch):
    s, calls = _session(tmp_path, monkeypatch, {}, experiments=["probe", "v3_forecasts"])
    s.run("probe")
    rec = s.run("v3_forecasts")
    assert rec["status"] == "ok" and rec["exit_code"] == 0 and rec["commit_sha"] == "abc"
    assert [c[5] for c in calls] == ["scripts/00b_runtime_probe.py", "scripts/14_v3_forecast.py"]
    assert calls[0][calls[0].index("--steps") + 1] == "v3_forecasts"
    assert "--resume" in calls[1] and "--as-of" in calls[1]
    table = kg.format_status_table(s.state["steps"])
    assert "v3_forecasts" in table and "probe" in table  # the status table lists opt-in steps

    saved = json.loads(s.status_path.read_text())
    assert saved["steps"]["v3_forecasts"]["status"] == "ok"
    # a restored ok on the same commit is skipped under RESUME, a failed one is retried
    s2, calls2 = _session(tmp_path, monkeypatch, {}, experiments=["v3_forecasts"])
    s2.run("v3_forecasts")
    assert calls2 == []


def test_a_failing_v3_step_is_recorded_and_retried(tmp_path, monkeypatch):
    s, _ = _session(tmp_path, monkeypatch, {"14_v3_forecast.py--resume": 1}, ["v3_forecasts"])
    assert s.run("v3_forecasts")["status"] == "failed"
    s2, calls2 = _session(tmp_path, monkeypatch, {}, experiments=["v3_forecasts"])
    assert s2.run("v3_forecasts")["status"] == "ok" and len(calls2) == 1


def test_probe_accepts_the_v3_step_but_has_no_work_model_for_it(tmp_path, monkeypatch):
    s, _ = _session(tmp_path, monkeypatch, {}, experiments=["probe", "v3_forecasts"])
    extra = s.probe_extra_args()
    assert extra[:2] == ["--steps", "v3_forecasts"]
    spec = importlib.util.spec_from_file_location(
        "probe_t", ROOT / "scripts" / "00b_runtime_probe.py"
    )
    probe = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = probe
    spec.loader.exec_module(probe)
    args = probe.build_parser().parse_args(["--as-of", AS_OF, *extra])
    assert args.steps == ["v3_forecasts"]
    assert probe.gather_work(["v3_forecasts"], 20, 3) == []  # documented: not projected


# ------------------------------------------------------------------ notebook


def test_notebook_documents_the_v3_option_in_the_experiments_cell():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = _src(_code_cells(nb)[0])
    assert '# "v3_forecasts",' in source  # commented out of the default list
    assert '["probe", "v3_forecasts"]' in source
    tree = ast.parse(source)
    experiments = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and n.targets[0].id == "EXPERIMENTS"
    )
    assert "v3_forecasts" not in experiments and experiments == V2_IDS


def test_notebook_has_the_v3_step_cell_right_before_packing():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]
    run_cell = next(
        i
        for i, c in enumerate(cells)
        if c["cell_type"] == "code" and _src(c).strip() == 'session.run("v3_forecasts")'
    )
    assert _src(cells[run_cell - 1]).startswith("### v3 studies")
    assert _src(cells[run_cell + 1]).startswith("## 7. Pack the results")
    assert _src(cells[run_cell - 2]).strip() == 'session.run("09_latency_timesfm2p5")'


# ------------------------------------------------------------------ cache coverage


def test_v3_requests_file_is_covered_by_the_existing_cache_manifest_mechanism(tmp_path):
    as_of, cache, man = _make_cache(tmp_path, with_cache_manifest=False)
    _requests_file(cache / "v3_requests.parquet")
    kg.write_cache_manifest(cache, man, as_of)  # the documented `cache-manifest` command
    listed = json.loads((man / f"cache-{as_of}.json").read_text())["files"]
    assert listed["v3_requests.parquet"]["sha256"] == kg.sha256_file(cache / "v3_requests.parquet")
    report = kg.verify_cache(cache, man, as_of)
    assert len(report["files"]["v3_requests.parquet"]) == 64
    assert f"cache_manifest:{len(listed)}_files" in report["checked"]


def test_no_cache_check_is_weakened_for_the_v3_requests_file(tmp_path):
    # the dataset gained the file but the committed manifest was not rebuilt: refused
    as_of, cache, man = _make_cache(tmp_path)
    _requests_file(cache / "v3_requests.parquet")
    with pytest.raises(kg.CacheMismatch, match="not in cache-.*v3_requests.parquet"):
        kg.verify_cache(cache, man, as_of)

    # the manifest lists the file but the attached dataset version lacks it: refused
    kg.write_cache_manifest(cache, man, as_of)
    (cache / "v3_requests.parquet").unlink()
    with pytest.raises(kg.CacheMismatch, match="cache is missing v3_requests.parquet"):
        kg.verify_cache(cache, man, as_of)

    # same name, other bytes (requests rebuilt locally, dataset not re-uploaded): refused
    _requests_file(cache / "v3_requests.parquet")
    kg.write_cache_manifest(cache, man, as_of)
    other = pd.DataFrame(
        {
            "request_id": ["a"],
            "study": ["S3"],
            "block": [4],
            "horizon": [2],
            "context": [[1.0, 2.0]],
        }
    )
    rq.write_requests(other, cache / "v3_requests.parquet")
    with pytest.raises(kg.CacheMismatch, match="v3_requests.parquet: sha256"):
        kg.verify_cache(cache, man, as_of)


def test_dataset_copy_puts_the_requests_file_where_the_script_looks(tmp_path, monkeypatch):
    dataset = tmp_path / "input" / "tfm3lab-cache-2026-09-04" / "nested"
    dataset.mkdir(parents=True)
    _requests_file(dataset / "v3_requests.parquet")
    cache = tmp_path / "data" / "cache"
    copied = kg.copy_dataset_into_cache(tmp_path / "input", cache)
    assert [p.name for p in copied] == ["v3_requests.parquet"]

    monkeypatch.setattr(config, "CACHE_DIR", cache)
    script = _load_script("14_v3_forecast.py")
    default = script.build_parser().parse_args([]).requests
    assert default == copied[0]
    assert len(rq.read_requests(default)) == 2


# ------------------------------------------------------------------ outputs and resume


def _fake_v3_outputs(results: Path) -> None:
    forecasts = results / "benchmark" / "v3_forecasts"
    (forecasts / "block=1").mkdir(parents=True)
    (forecasts / "block=1" / "part-000000.parquet").write_bytes(b"forecasts")
    (forecasts / "block_status.json").write_text("{}")
    (results / "manifests").mkdir()
    (results / "manifests" / f"v3_forecast-{AS_OF}-20261015T000000Z.json").write_text("{}")


def test_forecasts_travel_in_the_heavy_archive_and_manifest_in_the_light_one(tmp_path):
    results = tmp_path / "results"
    _fake_v3_outputs(results)
    out = tmp_path / "out"
    kg.pack_results(results, out, AS_OF, state={"steps": {"v3_forecasts": {"status": "ok"}}})
    with tarfile.open(out / f"tfm3lab-results-{AS_OF}.tar.gz") as tar:
        light = tar.getnames()
    with tarfile.open(out / f"tfm3lab-heavy-{AS_OF}.tar.gz") as tar:
        heavy = tar.getnames()
    assert f"results/manifests/v3_forecast-{AS_OF}-20261015T000000Z.json" in light
    assert "results/run_status.json" in light
    assert not any("v3_forecasts" in name for name in light)
    assert "results/benchmark/v3_forecasts/block=1/part-000000.parquet" in heavy
    assert "results/benchmark/v3_forecasts/block_status.json" in heavy


def test_a_restored_previous_session_brings_the_v3_parts_back_for_resume(tmp_path):
    previous = tmp_path / "prev" / "tfm3lab-data" / "results"
    _fake_v3_outputs(previous)
    (previous / "run_status.json").write_text('{"commit_sha": "abc", "steps": {}}')
    results = tmp_path / "res"
    results.mkdir()
    restored = kg.restore_previous(tmp_path / "prev", results, "abc")
    assert "benchmark/v3_forecasts/block=1/part-000000.parquet" in restored
    assert (results / "benchmark" / "v3_forecasts" / "block=1" / "part-000000.parquet").is_file()
