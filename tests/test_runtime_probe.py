"""Probe logic with fakes: no GPU, no network, no model."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import FakeForecaster
from tfm3lab import kaggle as kg

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location(
        "probe_t", ROOT / "scripts" / "00b_runtime_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def m(adapter, context, batch, median_s, status="ok"):
    return {
        "adapter": adapter, "context": context, "batch": batch, "median_s": median_s,
        "status": status,
    }


MEAS = [
    m("a", 128, 1, 0.5), m("a", 128, 32, 3.2),  # 0.1 s per forecast at the largest batch
    m("a", 512, 1, 1.0), m("a", 512, 32, 6.4),  # 0.2 s per forecast
]


def test_per_forecast_uses_largest_batch_and_interpolates():
    assert kg.per_forecast_seconds(MEAS, "a", 128) == (pytest.approx(0.1), "measured")
    spf, basis = kg.per_forecast_seconds(MEAS, "a", 320)
    assert spf == pytest.approx(0.15) and "interpolated" in basis
    spf, basis = kg.per_forecast_seconds(MEAS, "a", 64)
    assert spf == pytest.approx(0.1) and "clamped" in basis
    spf, basis = kg.per_forecast_seconds(MEAS, "a", 1024)
    assert spf == pytest.approx(0.4) and "scaled" in basis


def test_per_forecast_ignores_failed_points_and_other_adapters():
    meas = [*MEAS, m("a", 512, 128, 1.0, status="oom"), m("b", 512, 8, 80.0)]
    assert kg.per_forecast_seconds(meas, "a", 512)[0] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        kg.per_forecast_seconds(MEAS, "zzz", 128)


def test_project_plan_cumulative_and_quota():
    works = [
        kg.StepWork("s1", "a", ((128, 360), (512, 180)), "n1"),  # 36 + 36 = 72 s
        kg.StepWork("s2", "a", ((512, 3600),), ""),  # 720 s
    ]
    rows = kg.project_plan(works, MEAS, {"a": 10.0}, quota_hours=0.1)  # 360 s
    assert rows[0]["compute_s"] == pytest.approx(72.0)
    assert rows[0]["projected_s"] == pytest.approx(82.0)
    assert rows[1]["cumulative_s"] == pytest.approx(82.0 + 730.0)
    assert rows[0]["fits_quota"] is True and rows[1]["fits_quota"] is False
    text = kg.format_plan(rows)
    assert "EXCEEDS" in text and "s1" in text
    none = kg.project_plan(works, MEAS, {}, None)
    assert none[0]["fits_quota"] is None and "quota" not in kg.format_plan(none)


def test_projection_scales_with_measurement_not_constants():
    slow = [{**x, "median_s": x["median_s"] * 2} for x in MEAS]
    w = [kg.StepWork("s", "a", ((512, 100),))]
    a = kg.project_plan(w, MEAS, {}, None)[0]["projected_s"]
    b = kg.project_plan(w, slow, {}, None)[0]["projected_s"]
    assert b == pytest.approx(2 * a)


def test_work_sizes():
    report = {
        "n_origins": 10, "card_pool_size": 4,
        "combos": [{"context_len": 64}, {"context_len": 512}],
    }
    w = kg.work_from_dry_run("x", "a", report)
    assert w.items == ((64, 40), (512, 40))
    s = kg.work_from_shock(
        "y", "a", n_events=2, context_len=256, origins_per_event=3, n_perturbation_runs=10,
        panel_series=5,
    )
    assert s.items == ((256, 2 * 3 * 16),)
    c = kg.work_from_covariates("z", "a", context_len=128, market_origins=[2, 3], mtg_origins=[4])
    assert c.items == ((128, 5 * 3 + 4 * 4),)
    lat = kg.work_from_latency("l", "a", [(1, 128, 28), (8, 512, 64)], repeats=20, warmup=3)
    assert lat.items == () and lat.cells == ((128, 1, 23), (512, 8, 23))


def test_latency_cells_priced_per_call_from_the_same_batch():
    meas = [m("a", 128, 1, 0.5), m("a", 512, 1, 1.0), m("a", 128, 8, 2.0), m("a", 512, 8, 4.0)]
    sec, basis = kg.per_call_seconds(meas, "a", 128, 8)
    assert sec == 2.0 and basis.startswith("measured") and "lower bound" not in basis
    # batch 32 not measured: per-forecast from the largest measured batch x 32, a lower bound
    sec, basis = kg.per_call_seconds(meas, "a", 512, 32)
    assert sec == pytest.approx(4.0 / 8 * 32) and "lower bound" in basis
    work = kg.work_from_latency("l", "a", [(8, 128, 28), (8, 512, 64), (32, 512, 64)], 2, 1)
    row = kg.project_step(work, meas, {})
    assert row["compute_s"] == pytest.approx(3 * (2.0 + 4.0) + 3 * 16.0)
    assert "lower bound" in row["basis"] and row["n_forecasts"] == 3 * 8 * 2 + 3 * 32


def test_default_points_cover_the_latency_grid_and_largest_context(probe):
    from tfm3lab import latency

    assert probe.grid_contexts_missing(probe.DEFAULT_POINTS) == []
    for b in latency.BATCH_SIZES:
        assert (max(latency.CONTEXTS), max(latency.HORIZONS), b) in probe.DEFAULT_POINTS
    assert probe.grid_contexts_missing(((128, 64, 1),)) == [512]


class Clock:
    def __init__(self):
        self.t = 0.0
        self.calls = 0

    def __call__(self):
        self.calls += 1
        self.t += 0.25
        return self.t


def test_measure_points_with_fakes_warmup_and_sync():
    series = [np.linspace(1, 2, 700), np.linspace(2, 3, 600)]
    syncs = []
    fc = FakeForecaster()
    rows = kg.measure_points(
        fc, series, "a", [(128, 64, 3), (512, 64, 5)], repeats=3, warmup=2,
        sync=lambda: syncs.append(1), clock=Clock(),
    )
    assert [r["status"] for r in rows] == ["ok", "ok"]
    assert rows[0]["median_s"] == pytest.approx(0.25) and rows[0]["warmup"] == 2
    assert rows[1]["batch"] == 5 and rows[1]["data_source"] == "real"
    assert len(syncs) == 2 * (2 + 2 * 3)  # warmup syncs once, timed calls twice


def test_measure_points_records_oom_and_continues():
    class Boom(FakeForecaster):
        def predict_batch(self, contexts, horizon, **kw):
            if len(contexts) > 2:
                raise RuntimeError("CUDA out of memory. Tried to allocate")
            return super().predict_batch(contexts, horizon, **kw)

    rows = kg.measure_points(
        Boom(), [np.linspace(1, 2, 700)], "a", [(128, 64, 4), (128, 64, 1)], repeats=1, warmup=0
    )
    assert rows[0]["status"] == "oom" and rows[1]["status"] == "ok"
    assert kg.per_forecast_seconds(rows, "a", 128)[1] == "measured"


def test_too_short_series_is_an_error_row_not_synthetic():
    rows = kg.measure_points(
        FakeForecaster(), [np.ones(10)], "a", [(512, 64, 2)], repeats=1, warmup=0
    )
    assert rows[0]["status"] == "error" and "synthetic" in rows[0]["message"]


def test_parser_and_helpers(probe):
    args = probe.build_parser().parse_args(["--as-of", "2026-09-04", "--quota-hours", "7.5"])
    assert args.quota_hours == 7.5 and "probe" not in args.steps
    assert probe.parse_points("128x64x1, 512x28x8") == ((128, 64, 1), (512, 28, 8))
    with pytest.raises(argparse.ArgumentTypeError):
        probe.parse_points("128x64")
    with pytest.raises(argparse.ArgumentTypeError):
        probe.parse_steps("nope")
    with pytest.raises(SystemExit):
        probe.build_parser().parse_args([])


def test_to_frame_has_all_kinds(probe):
    plan = kg.project_plan([kg.StepWork("s", "a", ((512, 10),))], MEAS, {"a": 1.0}, 1.0)
    df = probe.to_frame(MEAS, plan, {"a": 1.0})
    assert set(df["kind"]) == {"measurement", "load", "projection"}
    assert df.loc[df["kind"] == "projection", "fits_quota"].iloc[0]


def test_gather_work_for_shock_and_latency_is_offline(probe):
    works = probe.gather_work(
        ["03_timesfm3", "03_timesfm2p5", "09_latency_timesfm3"], 20, 3
    )
    by = {w.step_id: w for w in works}
    assert by["03_timesfm3"].items[0][1] > by["03_timesfm2p5"].items[0][1] > 0  # panel only for 3
    assert len(by["09_latency_timesfm3"].cells) == 16  # 4 batches x 2 contexts x 2 horizons
