"""Offline checks of the Kaggle notebook and the helpers it calls."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from tfm3lab import kaggle as kg

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "run_on_kaggle.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _src(cell) -> str:
    return "".join(cell["source"])


def _load_script(filename: str):
    spec = importlib.util.spec_from_file_location(
        "kaggle_t_" + filename[:3], ROOT / "scripts" / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ notebook


def test_notebook_is_valid_nbformat4_without_outputs(nb):
    assert nb["nbformat"] == 4
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.from_dict(nb))
    for cell in _code_cells(nb):
        assert cell["outputs"] == [] and cell["execution_count"] is None
        ast.parse(_src(cell))  # every code cell is plain Python


def test_parameter_cell_is_first_code_cell(nb):
    first = _code_cells(nb)[0]
    tree = ast.parse(_src(first))
    values = {
        t.id: ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
    }
    assert values["REPO_URL"] == "https://github.com/IrfEazy/timesfm3-talk"
    assert values["GIT_REF"] == "feat/phase1-defensible"
    assert values["AS_OF"] == "2026-09-04"
    assert {"DATASET_DIR", "PREVIOUS_OUTPUT_DIR", "RESUME", "ALLOW_CPU"} <= set(values)
    assert values["PREVIOUS_OUTPUT_DIR"] is None
    assert values["RESUME"] is True and values["ALLOW_CPU"] is False
    assert values["EXPERIMENTS"] == list(kg.DEFAULT_EXPERIMENTS)
    assert not any(isinstance(n, ast.Import | ast.ImportFrom) for n in tree.body)


def test_setup_cells_cover_the_required_plumbing(nb):
    text = "\n".join(_src(c) for c in _code_cells(nb))
    for needle in (
        "uv sync",
        "--frozen",
        "--extra",
        "cuda",
        "UserSecretsClient",
        "HF_TOKEN",
        "TFM3LAB_DATA_ROOT",
        "/kaggle/working/tfm3lab-data",
        "verify_cache",
        "restore_previous",
        "tfm3lab.kaggle",
        "COMMIT_SHA",
        "UV_PROJECT_ENVIRONMENT",
        "/kaggle/tmp",
    ):
        assert needle in text, needle
    assert 'REPO_DIR = TMP_DIR + "/timesfm3-talk"' in text and 'TMP_DIR = "/kaggle/tmp"' in text
    assert '"/kaggle/working/timesfm3-talk"' not in text  # clone + venv stay off the saved dir
    assert "restore_previous(" in text and "COMMIT_SHA\n)" in text  # commit checked on restore
    assert "git_push" not in text and "git push" not in text


def test_each_step_has_its_own_cell_in_table_order(nb):
    calls = []
    for cell in _code_cells(nb):
        src = _src(cell).strip()
        if src.startswith("session.run("):
            calls.append(ast.literal_eval(src[len("session.run(") : -1]))
    assert calls == [s.step_id for s in kg.STEPS]
    assert calls[1] == "02b_preregistered_v2"  # primary claim right after the probe


@pytest.mark.parametrize("step", kg.STEPS, ids=lambda s: s.step_id)
def test_step_argv_parses_with_the_scripts_own_parser(step):
    script = ROOT / step.script
    assert script.is_file()
    module = _load_script(script.name)
    argv = kg.step_argv(step, "2026-09-04")
    assert argv[:6] == [sys.executable, "-m", "uv", "run", "--no-sync", step.script]
    args = module.build_parser().parse_args(argv[6:])
    assert str(args.as_of) == "2026-09-04"
    if "--config" in step.args:
        cfg = ROOT / step.args[step.args.index("--config") + 1]
        assert cfg.is_file()
        assert args.resume is True
    if "--adapter" in step.args:
        assert args.adapter == "timesfm2.5"
        assert step.adapter == "timesfm2.5"
    assert module.build_parser().parse_args([*argv[6:], "--allow-cpu"]).allow_cpu is True


def test_probe_extra_args_parse():
    module = _load_script("00b_runtime_probe.py")
    session = kg.Session.__new__(kg.Session)
    session.experiments = ["probe", "05_covariates", "09_latency_timesfm3"]
    session.quota_hours = 12.5
    extra = kg.Session.probe_extra_args(session)
    args = module.build_parser().parse_args(["--as-of", "2026-09-04", *extra])
    assert args.steps == ["05_covariates", "09_latency_timesfm3"]
    assert args.quota_hours == 12.5


def test_pack_cell_uses_pack_results_and_prints_hashes(nb):
    text = _src(_code_cells(nb)[-1])
    assert "pack_results" in text and "sha256" in text and "bytes" in text
    assert "/kaggle/working" in text


# ------------------------------------------------------------------ packing


def test_pack_results_excludes_benchmark_from_light_archive(tmp_path):
    res = tmp_path / "results"
    (res / "benchmark" / "cfg" / "abc123").mkdir(parents=True)
    (res / "benchmark" / "cfg" / "abc123" / "cell.parquet").write_bytes(b"heavy")
    (res / "manifests").mkdir()
    (res / "manifests" / "m.json").write_text("{}")
    (res / "latency_v2.parquet").write_bytes(b"light")
    out = tmp_path / "out"
    info = kg.pack_results(res, out, "2026-09-04", state={"steps": {"probe": {"exit_code": 0}}})

    with tarfile.open(out / "tfm3lab-results-2026-09-04.tar.gz") as tar:
        names = tar.getnames()
    assert "results/latency_v2.parquet" in names
    assert "results/manifests/m.json" in names
    assert "results/run_status.json" in names
    assert not any("benchmark" in n for n in names)
    with tarfile.open(out / "tfm3lab-heavy-2026-09-04.tar.gz") as tar:
        assert tar.getnames() == ["results/benchmark/cfg/abc123/cell.parquet"]
    assert len(info["light"]["sha256"]) == 64 and info["heavy"]["bytes"] > 0
    assert json.loads((res / "run_status.json").read_text())["steps"]["probe"]["exit_code"] == 0


def test_pack_results_without_benchmark_or_heavy(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    (res / "a.parquet").write_bytes(b"x")
    assert kg.pack_results(res, tmp_path / "o", "d")["heavy"] is None
    (res / "benchmark").mkdir()
    (res / "benchmark" / "f").write_bytes(b"y")
    assert kg.pack_results(res, tmp_path / "o2", "d", include_heavy=False)["heavy"] is None


# ------------------------------------------------------------------ cache, resume


def _make_cache(
    tmp_path: Path,
    *,
    tamper: bool = False,
    with_cache_manifest: bool = True,
    tamper_cache_manifest: bool = False,
):
    as_of = "2026-09-04"
    cache = tmp_path / "cache"
    cache.mkdir()
    dates = pd.date_range("2026-08-01", periods=5)
    pd.DataFrame(
        {"series": ["A [X]"] * 5 + ["B [X]"] * 5, "date": list(dates) * 2, "value": 1.0}
    ).to_parquet(cache / "mtg_prices.parquet")
    pd.DataFrame({"a": [1]}).to_parquet(cache / "market_prices.parquet")
    pd.DataFrame({"a": [1]}).to_parquet(cache / "cpi_yoy.parquet")
    closes = {"SP500": [1.0, 2.0, 3.0, 4.0, 5.0]}
    shock_hash = {}
    for label, vals in closes.items():
        pd.DataFrame({"date": dates, "close": vals}).to_parquet(
            cache / f"shock_v2_{label}_{as_of}.parquet"
        )
        shock_hash[label] = kg.series_sha256(dates, vals if not tamper else [9.0] * 5)
    man = tmp_path / "manifests"
    man.mkdir()
    (man / f"shock-events-v2-{as_of}.json").write_text(json.dumps({"data_sha256": shock_hash}))
    (man / f"fetch-{as_of}-20260904T071425Z.json").write_text(
        json.dumps(
            {
                "date_range": {"start": "2026-08-01", "end": "2026-08-05"},
                "resolved_cards": [{"label": "A [X]"}, {"label": "B [X]"}],
            }
        )
    )
    files = kg.build_cache_files(cache)
    if tamper_cache_manifest:
        files["cpi_yoy.parquet"]["sha256"] = "0" * 64
    if with_cache_manifest:
        (man / f"cache-{as_of}.json").write_text(json.dumps({"files": files}))
    return as_of, cache, man


def test_verify_cache_accepts_matching_data(tmp_path):
    as_of, cache, man = _make_cache(tmp_path)
    report = kg.verify_cache(cache, man, as_of)
    assert "shock:SP500" in report["checked"] and "fetch:mtg_series" in report["checked"]
    assert len(report["files"]["mtg_prices.parquet"]) == 64


def test_verify_cache_requires_the_cache_manifest(tmp_path):
    as_of, cache, man = _make_cache(tmp_path, with_cache_manifest=False)
    with pytest.raises(kg.CacheMismatch, match="cache-manifest"):
        kg.verify_cache(cache, man, as_of)


def test_verify_cache_compares_every_file_sha(tmp_path):
    as_of, cache, man = _make_cache(tmp_path, tamper_cache_manifest=True)
    with pytest.raises(kg.CacheMismatch, match="cpi_yoy"):
        kg.verify_cache(cache, man, as_of)


def test_verify_cache_rejects_missing_and_extra_files(tmp_path):
    as_of, cache, man = _make_cache(tmp_path)
    pd.DataFrame({"a": [2]}).to_parquet(cache / "extra.parquet")
    with pytest.raises(kg.CacheMismatch, match="extra.parquet"):
        kg.verify_cache(cache, man, as_of)
    (cache / "extra.parquet").unlink()
    (cache / "cpi_yoy.parquet").unlink()
    with pytest.raises(kg.CacheMismatch, match="missing cpi_yoy"):
        kg.verify_cache(cache, man, as_of)


def test_write_cache_manifest_roundtrip(tmp_path, monkeypatch):
    as_of, cache, man = _make_cache(tmp_path, with_cache_manifest=False)
    out = kg.write_cache_manifest(cache, man, as_of)
    assert out == man / f"cache-{as_of}.json"
    data = json.loads(out.read_text())
    assert set(data["files"]) == {p.name for p in cache.glob("*.parquet")}
    one = data["files"]["cpi_yoy.parquet"]
    assert one["sha256"] == kg.sha256_file(cache / "cpi_yoy.parquet")
    assert one["bytes"] == (cache / "cpi_yoy.parquet").stat().st_size
    assert "_meta" in data
    assert "cache_manifest" in kg.verify_cache(cache, man, as_of)


def test_cache_manifest_cli(tmp_path, monkeypatch):
    from tfm3lab import config

    as_of, cache, man = _make_cache(tmp_path, with_cache_manifest=False)
    monkeypatch.setattr(config, "CACHE_DIR", cache)
    monkeypatch.setattr(config, "RESULTS_DIR", man.parent)
    assert kg.main(["cache-manifest", "--as-of", as_of]) == 0
    assert (man / f"cache-{as_of}.json").is_file()
    with pytest.raises(SystemExit):
        kg.main(["cache-manifest"])


def test_verify_cell_is_not_swallowed(nb):
    cell = next(_src(c) for c in _code_cells(nb) if "verify_cache" in _src(c))
    assert "try:" not in cell and "except" not in cell  # fails loudly


def test_verify_cache_fails_fast_on_shock_hash_mismatch(tmp_path):
    as_of, cache, man = _make_cache(tmp_path, tamper=True)
    with pytest.raises(kg.CacheMismatch, match="shock_v2_SP500"):
        kg.verify_cache(cache, man, as_of)


def test_verify_cache_detects_missing_card_and_declared_sha(tmp_path):
    as_of, cache, man = _make_cache(tmp_path)
    fetch = next(man.glob("fetch-*.json"))
    meta = json.loads(fetch.read_text())
    meta["resolved_cards"].append({"label": "C [X]"})
    fetch.write_text(json.dumps(meta))
    with pytest.raises(kg.CacheMismatch, match="C \\[X\\]"):
        kg.verify_cache(cache, man, as_of)
    meta["resolved_cards"].pop()
    meta["cache_sha256"] = {"cpi_yoy.parquet": "0" * 64}
    fetch.write_text(json.dumps(meta))
    with pytest.raises(kg.CacheMismatch, match="cpi_yoy"):
        kg.verify_cache(cache, man, as_of)


def test_copy_dataset_into_cache_flattens_nested_dirs(tmp_path):
    ds = tmp_path / "ds" / "data" / "cache"
    ds.mkdir(parents=True)
    (ds / "a.parquet").write_bytes(b"1")
    (tmp_path / "ds" / "readme.txt").write_text("x")
    out = kg.copy_dataset_into_cache(tmp_path / "ds", tmp_path / "cache")
    assert [p.name for p in out] == ["a.parquet"]
    with pytest.raises(kg.CacheMismatch):
        kg.copy_dataset_into_cache(tmp_path / "missing", tmp_path / "cache")


def test_restore_previous_copies_run_dirs_without_overwriting(tmp_path):
    prev = tmp_path / "prev" / "tfm3lab-data" / "results"
    (prev / "benchmark" / "c" / "f1").mkdir(parents=True)
    (prev / "benchmark" / "c" / "f1" / "cell.parquet").write_bytes(b"old")
    (prev / "run_status.json").write_text('{"steps": {}}')
    res = tmp_path / "res"
    res.mkdir()
    (res / "run_status.json").write_text("NEW")
    restored = kg.restore_previous(tmp_path / "prev", res)
    assert restored == ["benchmark/c/f1/cell.parquet"]
    assert (res / "run_status.json").read_text() == "NEW"
    assert kg.restore_previous(None, res) == []
    with pytest.raises(FileNotFoundError):
        kg.restore_previous(tmp_path / "empty_nowhere", res)


def test_restore_previous_stops_on_commit_mismatch(tmp_path):
    prev = tmp_path / "prev" / "tfm3lab-data" / "results"
    (prev / "benchmark").mkdir(parents=True)
    (prev / "benchmark" / "f").write_bytes(b"1")
    (prev / "run_status.json").write_text('{"commit_sha": "aaa111", "steps": {}}')
    res = tmp_path / "res"
    res.mkdir()
    with pytest.raises(kg.ResumeCommitMismatch, match="GIT_REF = 'aaa111'"):
        kg.restore_previous(tmp_path / "prev", res, "bbb222")
    assert list(res.iterdir()) == []  # nothing copied
    assert kg.restore_previous(tmp_path / "prev", res, "aaa111")


def test_session_does_not_overwrite_commit_or_skip_other_commit_steps(tmp_path, monkeypatch):
    s, _ = _session(tmp_path, monkeypatch, {}, experiments=["03_timesfm3"])
    s.run("03_timesfm3")
    assert s.state["steps"]["03_timesfm3"]["commit_sha"] == "abc"
    monkeypatch.setattr(kg, "run_command", lambda *a, **k: (0, 1.0))
    with pytest.raises(kg.ResumeCommitMismatch, match="abc"):
        kg.Session(
            as_of="2026-09-04",
            repo_dir=tmp_path,
            data_root=tmp_path / "d",
            experiments=["03_timesfm3"],
            commit_sha="def",
        )
    status = json.loads(s.status_path.read_text())
    status["steps"]["03_timesfm3"]["commit_sha"] = "old"  # ok, but on another commit
    s.status_path.write_text(json.dumps(status))
    s2, calls = _session(tmp_path, monkeypatch, {}, experiments=["03_timesfm3"])
    s2.run("03_timesfm3")
    assert len(calls) == 1  # re-run, not skipped


# ------------------------------------------------------------------ session


def _session(tmp_path, monkeypatch, codes, experiments=kg.DEFAULT_EXPERIMENTS, resume=True):
    calls = []

    def fake_run(argv, *, cwd, log_path, env=None, echo=None):
        calls.append(argv)
        key = argv[5].split("/")[-1] + " ".join(argv[6:7])
        return codes.get(key, codes.get("default", 0)), 60.0

    monkeypatch.setattr(kg, "run_command", fake_run)
    s = kg.Session(
        as_of="2026-09-04",
        repo_dir=tmp_path,
        data_root=tmp_path / "d",
        experiments=experiments,
        resume=resume,
        quota_hours=10.0,
        commit_sha="abc",
    )
    return s, calls


def test_session_records_failure_and_continues(tmp_path, monkeypatch):
    s, calls = _session(tmp_path, monkeypatch, {"02b_exp_mtg_benchmark.py--config": 1})
    s.run("02b_preregistered_v2")
    s.run("03_timesfm3")
    assert s.state["steps"]["02b_preregistered_v2"]["status"] == "failed"
    assert s.state["steps"]["02b_preregistered_v2"]["exit_code"] == 1
    assert s.state["steps"]["03_timesfm3"]["status"] == "ok"
    assert s.state["steps"]["03_timesfm3"]["elapsed_s"] == 60.0
    saved = json.loads(s.status_path.read_text())
    assert saved["commit_sha"] == "abc"
    assert set(saved["steps"]) == {"02b_preregistered_v2", "03_timesfm3"}
    assert "--resume" in calls[0] and "--as-of" in calls[0]


def test_session_skips_unselected_and_completed_steps(tmp_path, monkeypatch):
    s, calls = _session(tmp_path, monkeypatch, {}, experiments=["03_timesfm3", "probe"])
    s.run("02b_showcase")
    assert calls == [] and s.state["steps"]["02b_showcase"]["status"] == "skipped"
    s.run("03_timesfm3")
    s2, calls2 = _session(tmp_path, monkeypatch, {}, experiments=["03_timesfm3", "probe"])
    s2.run("03_timesfm3")  # restored as ok -> skipped under RESUME
    s2.run("probe")  # the probe always re-runs
    assert [c[5] for c in calls2] == ["scripts/00b_runtime_probe.py"]
    s3, calls3 = _session(tmp_path, monkeypatch, {}, experiments=["03_timesfm3"], resume=False)
    s3.run("03_timesfm3")
    assert len(calls3) == 1


def test_run_command_streams_and_reports_exit(tmp_path):
    lines = []
    code, elapsed = kg.run_command(
        [sys.executable, "-c", "print('hi'); raise SystemExit(3)"],
        cwd=tmp_path,
        log_path=tmp_path / "log" / "x.log",
        echo=lines.append,
    )
    assert code == 3 and elapsed >= 0 and "".join(lines).strip() == "hi"
    assert "hi" in (tmp_path / "log" / "x.log").read_text()
    code, _ = kg.run_command(
        ["definitely-not-a-command-xyz"], cwd=tmp_path, log_path=tmp_path / "l2"
    )
    assert code == 127


def test_validate_experiments_rejects_unknown():
    with pytest.raises(ValueError, match="unknown"):
        kg.validate_experiments(["probe", "nope"])


def test_no_step_or_notebook_command_can_re_resolve_the_environment():
    for step in kg.STEPS:
        argv = kg.step_argv(step, "2026-09-04")
        assert argv[3:5] == ["run", "--no-sync"]
    nb = (ROOT / "notebooks" / "run_on_kaggle.ipynb").read_text(encoding="utf-8")
    assert 'uv_argv(\\"run\\", \\"--no-sync\\"' in nb
    assert 'uv_argv(\\"run\\", \\"python\\"' not in nb
    assert "torch.__version__" in nb
