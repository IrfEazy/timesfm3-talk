import numpy as np
import pytest

from tfm3lab.latency import grid, make_contexts, summarize_timings, time_calls, time_cell


class FakeClock:
    def __init__(self, step=1.0):
        self.t = 0.0
        self.step = step

    def __call__(self):
        return self.t


def test_warmup_excluded_and_sync_around_timed_calls():
    events = []
    clock = FakeClock()

    def fn():
        events.append("call")
        clock.t += 2.0

    times = time_calls(fn, repeats=3, warmup=2, sync=lambda: events.append("sync"), clock=clock)
    assert times == [2.0, 2.0, 2.0]
    assert events.count("call") == 5
    # timed section: sync, call, sync for each repeat
    assert events[-9:] == ["sync", "call", "sync"] * 3


def test_median_p95_and_per_1k():
    s = summarize_timings([1.0, 2.0, 3.0, 4.0, 100.0], n_forecasts=8)
    assert s["median_s"] == 3.0
    assert s["p95_s"] == pytest.approx(np.percentile([1, 2, 3, 4, 100], 95))
    assert s["median_s_per_1k"] == pytest.approx(3.0 / 8 * 1000)


def test_time_cell_with_fake_model():
    clock = FakeClock()
    calls = []

    def model():
        calls.append(1)
        clock.t += 0.5

    r = time_cell(model, n_forecasts=32, repeats=4, warmup=1, clock=clock)
    assert len(calls) == 5
    assert r["median_s"] == 0.5 and r["median_s_per_1k"] == pytest.approx(0.5 / 32 * 1000)


def test_invalid_args():
    with pytest.raises(ValueError):
        time_calls(lambda: None, repeats=0, warmup=0)
    with pytest.raises(ValueError):
        summarize_timings([1.0], 0)


def test_grid_size():
    assert len(grid()) == 16


def test_make_contexts_real_and_synthetic():
    ctx, src = make_contexts([np.arange(1, 300.0)], 8, 128, allow_synthetic=False)
    assert src == "real" and len(ctx) == 8 and all(len(c) == 128 for c in ctx)
    with pytest.raises(ValueError):
        make_contexts([np.arange(10.0)], 2, 128, allow_synthetic=False)
    ctx, src = make_contexts([], 2, 128, allow_synthetic=True)
    assert src == "synthetic" and ctx[0].shape == (128,)


# ---- grid robustness ----
from tfm3lab.latency import (  # noqa: E402
    check_data_sufficiency,
    context_stats,
    guarded_ets,
    run_grid,
    write_rows_atomic,
)


class OutOfMemoryError(RuntimeError):
    pass


def _methods(fail_cells=(), exc=None):
    exc = exc or OutOfMemoryError("CUDA out of memory")

    def methods_for(ctxs, horizon):
        def model():
            if (len(ctxs), horizon) in fail_cells:
                raise exc

        return [("m", model, None, "gpu", None), ("naive", lambda: None, None, "cpu", None)]

    return methods_for


SERIES = [np.arange(1, 600.0)]


def test_oom_in_one_cell_recorded_and_grid_continues():
    cleaned = []
    cells = grid((1, 8), (128,), (28,))
    rows = run_grid(
        cells,
        SERIES,
        _methods(fail_cells={(1, 28)}),
        repeats=1,
        warmup=0,
        allow_synthetic=False,
        cleanup=lambda: cleaned.append(1),
    )
    assert [r["status"] for r in rows] == ["oom", "ok", "ok", "ok"]
    assert "out of memory" in rows[0]["error"] and cleaned == [1]
    assert rows[2]["batch"] == 8 and rows[2]["series_repeated"] is True
    assert rows[2]["n_distinct_series"] == 1 and rows[2]["p95_method"]


def test_other_exception_is_error_status():
    rows = run_grid(
        grid((1,), (128,), (28,)),
        SERIES,
        _methods(fail_cells={(1, 28)}, exc=ValueError("boom")),
        repeats=1,
        warmup=0,
        allow_synthetic=False,
    )
    assert rows[0]["status"] == "error" and "boom" in rows[0]["error"]


def test_results_file_exists_after_mid_grid_exception(tmp_path):
    out = tmp_path / "r.parquet"
    calls = {"n": 0}

    def methods_for(ctxs, horizon):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyError("mid-grid")
        return [("m", lambda: None, None, "gpu", None)]

    with pytest.raises(KeyError):
        run_grid(
            grid((1, 8), (128,), (28,)),
            SERIES,
            methods_for,
            repeats=1,
            warmup=0,
            allow_synthetic=False,
            on_row=lambda row, rows: write_rows_atomic(rows, out),
        )
    import pandas as pd

    df = pd.read_parquet(out)
    assert len(df) == 1 and df["status"].iloc[0] == "ok"
    assert not (tmp_path / "r.parquet.tmp").exists()


def test_insufficient_data_fails_before_model_load():
    loaded = []

    def load_model():
        loaded.append(1)

    with pytest.raises(ValueError, match="insufficient data"):
        check_data_sufficiency([np.arange(100.0)], allow_synthetic=False)
        load_model()
    assert loaded == []
    assert check_data_sufficiency([], allow_synthetic=True) == 0
    assert check_data_sufficiency([np.arange(600.0)], allow_synthetic=False) == 1


def test_ets_failures_counted_per_cell():
    def ets(c, h):
        if c[0] == 0:
            raise ValueError("degenerate")

    ctxs = [np.zeros(5), np.ones(5), np.zeros(5)]
    run, failures = guarded_ets(ets)
    rows = run_grid(
        grid((3,), (128,), (28,)),
        [np.arange(600.0)],
        lambda c, h: [
            ("ets", lambda: run(ctxs, h), None, "cpu", lambda: {"ets_failed_series": failures()})
        ],
        repeats=2,
        warmup=1,
        allow_synthetic=False,
    )
    assert rows[0]["status"] == "ok" and rows[0]["ets_failed_series"] == 2


def test_context_stats_synthetic():
    assert context_stats([], 4, 128, "synthetic") == {
        "n_distinct_series": 4,
        "series_repeated": False,
    }


def test_output_path_is_per_adapter():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "latency_script", Path(__file__).resolve().parents[1] / "scripts" / "09_latency.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p3 = mod.output_path("timesfm3")
    p25 = mod.output_path("timesfm2.5")
    assert p3.name == "latency_v2.parquet"
    assert p25.name == "latency_v2_2p5.parquet"
    assert p3 != p25
