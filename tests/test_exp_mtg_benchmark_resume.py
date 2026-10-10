"""End-to-end streaming/commit/resume tests for scripts/02b_exp_mtg_benchmark.py
with a tiny deterministic fake forecaster. Offline: preflight, git state and
the model loader are replaced; ETS is replaced by a cheap deterministic stub.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import FakeForecaster
from tfm3lab import baselines, cellstore
from tfm3lab.benchmark import PRIMARY_EXPORT_FORBIDDEN
from tfm3lab.data.mtg import DEFAULT_CARDS

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "02b_exp_mtg_benchmark.py"
N_DAYS = 530
N_CELLS = 8  # 2 contexts x 2 horizons x 1 transform x 1 make_positive x 2 modes


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("exp_mtg_benchmark_02b_resume", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CountingForecaster(FakeForecaster):
    """Deterministic: point = 1.01 * last context value. Records, per call,
    how many cells were already committed on disk."""

    def __init__(self, results_root):
        super().__init__()
        self.calls = 0
        self.markers_seen: list[int] = []
        self.results_root = results_root

    def predict_batch(self, contexts, horizon, **kwargs):
        self.calls += 1
        self.markers_seen.append(len(list(Path(self.results_root()).rglob("done.json"))))
        for out in super().predict_batch(contexts, horizon, **kwargs):
            out.forecast = out.forecast * 1.01
            out.quantiles = out.quantiles * 1.01
            yield out


def _write_cache(cache_dir: Path) -> None:
    dates = pd.date_range("2024-02-08", periods=N_DAYS, freq="D")
    frames = []
    for i, card in enumerate(DEFAULT_CARDS):
        t = np.arange(N_DAYS)
        values = 5.0 + i + 0.02 * t + np.sin(t / (3.0 + i))
        observed = (t + i) % 6 != 0
        frames.append(
            pd.DataFrame(
                {"series": card.label, "date": dates, "value": values, "observed": observed}
            )
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(cache_dir / "mtg_prices.parquet")


class Env:
    def __init__(self, bench, tmp_path, monkeypatch):
        self.bench = bench
        self.tmp = tmp_path
        self.mp = monkeypatch
        self.cache = tmp_path / "cache"
        _write_cache(self.cache)
        monkeypatch.setattr(bench.config, "CACHE_DIR", self.cache)
        self.use_results("a")
        self.cfg_path = self.write_config()
        self.git = {"sha": "f" * 40, "dirty": False, "dirty_digest": None}
        self.revision = "rev-1"
        self.forecasters: list[CountingForecaster] = []
        self.loads = 0
        monkeypatch.setattr(bench, "_git_state", lambda: dict(self.git))
        monkeypatch.setattr(
            bench,
            "preflight_model",
            lambda repo_id, revision, allow_cpu=False: {
                "repo_id": repo_id,
                "revision_requested": revision,
                "revision_resolved": self.revision,
            },
        )
        monkeypatch.setattr(bench, "_load_forecaster", self._load)
        monkeypatch.setattr(
            baselines,
            "ets_forecast",
            lambda ctx, h, seasonal_periods=None: np.full(h, float(np.mean(ctx))),
        )

    def _load(self, adapter, revision):
        assert revision == self.revision
        self.loads += 1
        f = CountingForecaster(lambda: self.results)
        self.forecasters.append(f)
        return f, True

    @property
    def calls(self) -> int:
        return sum(f.calls for f in self.forecasters)

    def use_results(self, name: str) -> None:
        self.results = self.tmp / f"results_{name}"
        self.results.mkdir(exist_ok=True)
        self.mp.setattr(self.bench.config, "RESULTS_DIR", self.results)

    def write_config(self, **over) -> Path:
        payload = {
            "config_id": "tiny",
            "context_lengths": [8, 512],
            "horizons": [1, 3],
            "primary_horizons": [3],
            "origin_stride": 2,
            "max_origins": 3,
            "cards": "showcase",
            "transforms": ["raw"],
            "make_positive": [True],
            "modes": ["univariate", "multivariate"],
            "season_length": 7,
        }
        payload.update(over)
        path = self.tmp / "tiny.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def run(self, *extra: str) -> None:
        self.bench.main(
            ["--config", str(self.cfg_path), "--as-of", "2026-09-04", "--allow-cpu", *extra]
        )

    def run_dirs(self) -> list[Path]:
        root = self.results / "benchmark" / "tiny"
        return sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []

    def run_dir(self) -> Path:
        (d,) = self.run_dirs()
        return d

    def logical_rows(self) -> pd.DataFrame:
        parts = [
            pd.read_parquet(c / "part.parquet")
            for c in sorted(self.run_dir().glob("cell=*"))
            if (c / "done.json").exists()
        ]
        return pd.concat(parts, ignore_index=True)

    def outputs(self, partial: bool = False) -> dict[str, pd.DataFrame]:
        suffix = "_partial" if partial else ""
        return {
            name: pd.read_parquet(self.results / f"{name}{suffix}.parquet")
            for name in (
                "exp_mtg_benchmark_leaderboard",
                "exp_mtg_benchmark_leaderboard_aggregate",
                "exp_mtg_benchmark_primary",
            )
        }

    def snapshot(self) -> dict:
        root = self.results / "benchmark"
        return {
            str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
            for p in sorted(root.rglob("*"))
        }


@pytest.fixture
def env(bench, tmp_path, monkeypatch):
    return Env(bench, tmp_path, monkeypatch)


@pytest.fixture
def reference(env):
    """An uninterrupted run in results_ref; leaves env pointed at results_a."""
    env.use_results("ref")
    env.run()
    ref = {"rows": env.logical_rows(), **env.outputs()}
    env.use_results("a")
    env.forecasters.clear()
    return ref


def _assert_same(env, ref):
    pd.testing.assert_frame_equal(env.logical_rows(), ref["rows"])
    for name, df in env.outputs().items():
        pd.testing.assert_frame_equal(df, ref[name], obj=name)


def _crash_at(stage: str, cell_index: int):
    """Raise at `stage` while the cell with 0-based execution index
    `cell_index` is being committed."""
    state = {"done": 0}

    def hook(s, cell_id):
        if s == stage and state["done"] == cell_index:
            raise RuntimeError(f"injected crash at {stage}")
        if s == "after_marker_rename":
            state["done"] += 1

    return hook


def test_uninterrupted_run_streams_one_cell_at_a_time(env):
    env.run()
    rd = env.run_dir()
    assert len(list(rd.glob("cell=*/done.json"))) == N_CELLS
    assert json.loads((rd / "manifest.json").read_text())["run_id"] == rd.name
    # each inference call happens after all previous cells were committed
    seen = env.forecasters[0].markers_seen
    assert seen == sorted(seen) and sorted(set(seen)) == list(range(N_CELLS))
    lb = env.outputs()["exp_mtg_benchmark_leaderboard"]
    assert {"config_id", "run_id"} <= set(lb.columns)
    assert lb["summary_complete"].all()
    agg = env.outputs()["exp_mtg_benchmark_leaderboard_aggregate"]
    assert set(agg["run_id"]) == {rd.name} and set(agg["config_id"]) == {"tiny"}


@pytest.mark.parametrize("cell_index", [0, N_CELLS - 1], ids=["first_cell", "last_cell"])
@pytest.mark.parametrize(
    "stage",
    [
        "after_temp_write",
        "during_validation",
        "after_part_rename",
        "during_marker_write",
        "after_marker_rename",
    ],
)
def test_crash_then_resume_matches_uninterrupted(env, reference, monkeypatch, stage, cell_index):
    monkeypatch.setattr(cellstore, "_FAULT_HOOK", _crash_at(stage, cell_index))
    with pytest.raises(RuntimeError, match="injected"):
        env.run()
    monkeypatch.setattr(cellstore, "_FAULT_HOOK", None)
    committed_before = len(list(env.run_dir().glob("cell=*/done.json")))
    expected = cell_index + (stage == "after_marker_rename")
    assert committed_before == expected

    env.forecasters.clear()
    env.run("--resume")
    quarantine = env.run_dir() / "_quarantine"
    if stage == "after_marker_rename":
        assert not quarantine.exists()
    else:
        assert len(list(quarantine.iterdir())) == 1
    if expected == N_CELLS:
        assert env.forecasters == []  # nothing pending: no model load
    else:
        assert env.forecasters[0].markers_seen[0] == expected
    _assert_same(env, reference)


def test_second_resume_does_no_inference_and_replaces_nothing(env):
    env.run()
    env.run("--resume")
    loads_before = env.loads
    rd = env.run_dir()
    stats = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in rd.rglob("cell=*/*")}
    env.run("--resume")
    assert env.loads == loads_before == 1
    assert {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in rd.rglob("cell=*/*")} == stats


def test_existing_run_without_resume_refuses(env):
    env.run()
    with pytest.raises(SystemExit, match="--resume"):
        env.run()


@pytest.mark.parametrize("damage", ["truncate", "flip", "delete_file", "bad_marker"])
def test_damaged_cell_is_never_skipped(env, reference, damage):
    env.run()
    cell = sorted(env.run_dir().glob("cell=*"))[3]
    part, marker = cell / "part.parquet", cell / "done.json"
    if damage == "truncate":
        part.write_bytes(part.read_bytes()[: part.stat().st_size // 2])
    elif damage == "flip":
        data = bytearray(part.read_bytes())
        data[len(data) // 3] ^= 0x01
        part.write_bytes(bytes(data))
    elif damage == "delete_file":
        part.unlink()
    else:
        marker.write_text(marker.read_text()[:-20])
    before = env.snapshot()
    loads = env.loads
    with pytest.raises(SystemExit, match="failed verification"):
        env.run("--resume")
    assert env.snapshot() == before and env.loads == loads
    env.forecasters.clear()
    env.run("--resume", "--repair")
    assert sum(f.calls for f in env.forecasters) >= 1
    _assert_same(env, reference)


@pytest.mark.parametrize("change", ["config", "git", "checkpoint", "cache"])
def test_fingerprint_change_refuses_resume_before_mutation(env, monkeypatch, change):
    monkeypatch.setattr(cellstore, "_FAULT_HOOK", _crash_at("after_part_rename", 1))
    with pytest.raises(RuntimeError):
        env.run()
    monkeypatch.setattr(cellstore, "_FAULT_HOOK", None)
    before = env.snapshot()
    if change == "config":
        env.cfg_path = env.write_config(season_length=5)
    elif change == "git":
        env.git["sha"] = "e" * 40
    elif change == "checkpoint":
        monkeypatch.setattr(env.bench.config, "CHECKPOINT_REVISION", "rev-2")
    else:
        df = pd.read_parquet(env.cache / "mtg_prices.parquet")
        df.loc[0, "value"] += 1.0
        df.to_parquet(env.cache / "mtg_prices.parquet")
    loads = env.loads
    with pytest.raises(SystemExit, match="fingerprint"):
        env.run("--resume")
    assert env.snapshot() == before and env.loads == loads


def test_config_description_is_not_part_of_the_fingerprint(env):
    env.run()
    env.cfg_path = env.write_config(description="edited prose")
    env.run("--resume")
    assert len(env.run_dirs()) == 1 and env.loads == 1


def test_reversed_completion_order_gives_identical_results(env, reference, monkeypatch):
    monkeypatch.setattr(env.bench, "_execution_order", lambda combos: list(reversed(combos)))
    env.run()
    _assert_same(env, reference)


def test_partial_summary_refused_unless_allowed(env, monkeypatch):
    monkeypatch.setattr(cellstore, "_FAULT_HOOK", _crash_at("after_temp_write", 5))
    with pytest.raises(RuntimeError):
        env.run()
    monkeypatch.setattr(cellstore, "_FAULT_HOOK", None)
    with pytest.raises(SystemExit, match="partial"):
        env.run("--resume", "--summary-only")
    loads = env.loads
    env.run("--resume", "--summary-only", "--allow-partial-summary")
    assert env.loads == loads
    lb = pd.read_parquet(env.results / "exp_mtg_benchmark_leaderboard_partial.parquet")
    assert not lb["summary_complete"].any()
    assert set(lb["n_cells_committed"]) == {5} and set(lb["n_cells_expected"]) == {N_CELLS}
    assert not (env.results / "exp_mtg_benchmark_leaderboard.parquet").exists()


def test_primary_export_is_normalized_and_price_free(env):
    env.run()
    primary = env.outputs()["exp_mtg_benchmark_primary"]
    assert not (PRIMARY_EXPORT_FORBIDDEN & set(primary.columns))
    assert not any(c.startswith("baseline_") for c in primary.columns)
    for col in (
        "card",
        "origin_date",
        "target_date",
        "horizon",
        "observed",
        "ae_model_scaled",
        "ae_baseline_naive_scaled",
        "y_norm",
        "q10_norm",
        "q50_norm",
        "q90_norm",
    ):
        assert col in primary.columns
    assert set(primary["context_len"]) == {512}
    assert set(primary["mode"]) == {"timesfm3_univariate"}
    assert set(primary["requested_horizon"]) == {1, 3}

    rows = env.logical_rows()
    prim = rows[
        (rows["context_len"] == 512)
        & (rows["mode"] == "timesfm3_univariate")
        & (rows["horizon_step"] == rows["requested_horizon"])
    ]
    prim = prim.reset_index(drop=True)
    assert len(prim) == len(primary) > 0
    # one row per (card, origin, requested horizon): far smaller than the heavy cells
    assert (
        len(primary)
        == primary[["card", "origin_date", "requested_horizon"]].drop_duplicates().shape[0]
    )
    assert (primary["horizon"] == primary["requested_horizon"]).all()
    assert primary["scale_valid"].all()
    assert np.isfinite(primary.select_dtypes("number").to_numpy(dtype=float)).all()
    np.testing.assert_allclose(primary["y_norm"], prim["actual"] / prim["context_last_observed"])
    np.testing.assert_allclose(
        primary["ae_model_scaled"],
        (prim["actual"] - prim["forecast"]).abs() / prim["context_last_observed"],
    )
    # the normalizer is the last OBSERVED context value, not just the last value
    cache = pd.read_parquet(env.cache / "mtg_prices.parquet")
    r = prim.iloc[0]
    s = cache[cache["series"] == r["series"]].sort_values("date").reset_index(drop=True)
    ctx = s.iloc[int(r["origin"]) - 512 : int(r["origin"])]
    assert r["context_last_observed"] == ctx[ctx["observed"]]["value"].iloc[-1]


def test_session_manifest_records_provenance(env):
    env.run()
    (path,) = sorted((env.results / "manifests").glob("benchmark-tiny-*.json"))
    m = json.loads(path.read_text())
    rd = env.run_dir()
    assert m["run_id"] == rd.name and m["fingerprint"].startswith(rd.name)
    assert m["_meta"]["checkpoint"]["revision_resolved"] == "rev-1"
    assert m["_meta"]["seed"] == 42
    assert len(m["cells"]) == N_CELLS
    for cid, info in m["committed_cells"].items():
        marker = json.loads((rd / f"cell={cid}" / "done.json").read_text())
        assert info["sha256"] == marker["sha256"]
    assert m["summarize_origin_stride_passed"] is True
    assert len(m["outputs"]) == 3


def _no_preflight(*_a, **_k):
    raise AssertionError("preflight must not run (no GPU/token/network needed)")


def test_summary_only_needs_no_preflight_gpu_or_model(env, reference, monkeypatch):
    env.run()
    monkeypatch.setattr(env.bench, "preflight_model", _no_preflight)
    monkeypatch.setattr(env.bench, "_load_forecaster", _no_preflight)
    env.run("--resume", "--summary-only")
    _assert_same(env, reference)
    (*_, last) = sorted((env.results / "manifests").glob("benchmark-tiny-*.json"))
    m = json.loads(last.read_text())
    assert m["_meta"]["checkpoint"]["revision_resolved"] is None
    assert {"torch", "numpy", "pandas", "pyarrow", "statsmodels"} <= set(m["dependency_versions"])


def test_resume_with_nothing_pending_needs_no_preflight(env, monkeypatch):
    env.run()
    monkeypatch.setattr(env.bench, "preflight_model", _no_preflight)
    monkeypatch.setattr(env.bench, "_load_forecaster", _no_preflight)
    env.run("--resume")
    assert env.loads == 1


def test_summary_only_without_a_run_refuses_and_creates_nothing(env, monkeypatch):
    monkeypatch.setattr(env.bench, "preflight_model", _no_preflight)
    with pytest.raises(SystemExit, match="no run directory"):
        env.run("--summary-only")
    assert env.run_dirs() == []


def test_dependency_versions_are_not_in_the_fingerprint(env):
    env.run()
    m = json.loads((env.run_dir() / "manifest.json").read_text())
    assert "dependency_versions" not in json.dumps(m["fingerprint_payload"])
