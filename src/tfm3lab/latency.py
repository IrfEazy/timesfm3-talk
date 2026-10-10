"""Latency benchmark primitives (Preregistration v2 s6): descriptive only.

The timing function is pure and injectable (callable, sync hook, clock) so it
is unit-tested offline with a fake model and fake clock. Results are seconds
per call and seconds per 1k forecasts (one forecast = one series x one
horizon window). No dollar figures, no deployment recommendation.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

BATCH_SIZES = (1, 8, 32, 128)
CONTEXTS = (128, 512)
HORIZONS = (28, 64)
P95_METHOD = "numpy.percentile(q=95, method='linear')"


def time_calls(
    fn: Callable[[], object],
    *,
    repeats: int,
    warmup: int,
    sync: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> list[float]:
    """Run fn `warmup` times (discarded), then `repeats` timed times.
    `sync` is called before and after every timed call (GPU barrier)."""
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()
    times: list[float] = []
    for _ in range(repeats):
        if sync is not None:
            sync()
        start = clock()
        fn()
        if sync is not None:
            sync()
        times.append(clock() - start)
    return times


def summarize_timings(times: Sequence[float], n_forecasts: int) -> dict:
    """Median and p95 seconds per call, and seconds per 1k forecasts
    (median / n_forecasts * 1000) where n_forecasts = series per call.
    p95 uses numpy.percentile's default linear interpolation (P95_METHOD);
    with few repeats it is interpolated between the two largest timings."""
    if n_forecasts < 1:
        raise ValueError(f"n_forecasts must be >= 1, got {n_forecasts}")
    arr = np.asarray(times, dtype=float)
    if arr.size == 0:
        raise ValueError("no timings to summarize")
    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    return {
        "median_s": median,
        "p95_s": p95,
        "median_s_per_1k": median / n_forecasts * 1000.0,
        "p95_s_per_1k": p95 / n_forecasts * 1000.0,
    }


def time_cell(
    fn: Callable[[], object],
    *,
    n_forecasts: int,
    repeats: int,
    warmup: int,
    sync: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    times = time_calls(fn, repeats=repeats, warmup=warmup, sync=sync, clock=clock)
    return {**summarize_timings(times, n_forecasts), "repeats": repeats, "warmup": warmup}


def grid(
    batches: Sequence[int] = BATCH_SIZES,
    contexts: Sequence[int] = CONTEXTS,
    horizons: Sequence[int] = HORIZONS,
) -> list[tuple[int, int, int]]:
    return [(b, c, h) for b in batches for c in contexts for h in horizons]


def make_contexts(
    series: Sequence[np.ndarray],
    batch: int,
    context: int,
    *,
    allow_synthetic: bool,
    seed: int = 0,
) -> tuple[list[np.ndarray], str]:
    """`batch` contexts of length `context` from real series (cycled, last
    `context` points of each series long enough). Falls back to a synthetic
    positive random walk only when allow_synthetic; returns (contexts, source)."""
    usable = [np.asarray(s, dtype=float)[-context:] for s in series if len(s) >= context]
    if usable:
        return [usable[i % len(usable)] for i in range(batch)], "real"
    if not allow_synthetic:
        raise ValueError(f"no cached series has >= {context} points; refusing synthetic data")
    rng = np.random.default_rng(seed)
    walks = 10.0 + np.cumsum(rng.normal(0, 0.1, size=(batch, context)), axis=1)
    return [np.abs(w) + 1.0 for w in walks], "synthetic"


def check_data_sufficiency(
    series: Sequence[np.ndarray],
    contexts: Sequence[int] = CONTEXTS,
    *,
    allow_synthetic: bool,
) -> int:
    """Fail fast (before model load) unless at least one series holds the
    largest context. Returns the number of usable series (0 only when
    synthetic data is allowed)."""
    need = max(contexts)
    n = sum(1 for s in series if len(s) >= need)
    if n == 0 and not allow_synthetic:
        raise ValueError(
            f"insufficient data: no cached series has >= {need} points "
            f"({len(series)} series loaded); refusing synthetic data without --allow-cpu"
        )
    return n


def context_stats(series: Sequence[np.ndarray], batch: int, context: int, source: str) -> dict:
    """n_distinct_series feeding a batch and whether series were repeated to fill it."""
    if source == "real":
        n_usable = sum(1 for s in series if len(s) >= context)
        n_distinct = min(batch, n_usable)
    else:
        n_distinct = batch
    return {"n_distinct_series": n_distinct, "series_repeated": n_distinct < batch}


def is_oom(exc: BaseException) -> bool:
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def guarded_ets(ets_fn: Callable[[np.ndarray, int], object]):
    """Wrap an ETS forecaster so per-series failures are counted, not raised.
    Returns (run(ctxs, horizon), failures()) where failures() is the count of
    failed series in the most recent call."""
    state = {"failed": 0}

    def run(ctxs, horizon):
        failed = 0
        for c in ctxs:
            try:
                ets_fn(c, horizon)
            except Exception:
                failed += 1
        state["failed"] = failed

    return run, lambda: state["failed"]


def write_rows_atomic(rows: list[dict], path: Path) -> None:
    """Rewrite the whole results parquet via temp file + os.replace."""
    import pandas as pd

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, path)


def run_grid(
    cells: Sequence[tuple[int, int, int]],
    series: Sequence[np.ndarray],
    methods_for: Callable[[list[np.ndarray], int], list[tuple]],
    *,
    repeats: int,
    warmup: int,
    allow_synthetic: bool,
    on_row: Callable[[dict, list[dict]], None] | None = None,
    cleanup: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    row_extra: dict | None = None,
) -> list[dict]:
    """Time every method in every cell. `methods_for(ctxs, horizon)` returns
    tuples (name, fn, sync, device, extra_fn) with extra_fn() -> dict merged
    into the row after timing (may be None). A failing method records
    status="oom" or "error" (with message), calls `cleanup`, and the grid
    continues. `on_row(row, all_rows)` fires after each row (incremental
    persistence); exceptions outside the timed methods propagate, with rows so
    far already delivered."""
    rows: list[dict] = []
    for batch, context, horizon in cells:
        ctxs, source = make_contexts(series, batch, context, allow_synthetic=allow_synthetic)
        stats = context_stats(series, batch, context, source)
        for name, fn, sync, device, extra_fn in methods_for(ctxs, horizon):
            row = {
                "method": name,
                "device": device,
                "batch": batch,
                "context": context,
                "horizon": horizon,
                "data_source": source,
                "p95_method": P95_METHOD,
                **stats,
                **(row_extra or {}),
            }
            try:
                res = time_cell(
                    fn, n_forecasts=batch, repeats=repeats, warmup=warmup, sync=sync, clock=clock
                )
                row.update(status="ok", error=None, **res)
                if extra_fn is not None:
                    row.update(extra_fn())
            except Exception as exc:
                row.update(
                    status="oom" if is_oom(exc) else "error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                if cleanup is not None:
                    cleanup()
            rows.append(row)
            if on_row is not None:
                on_row(row, rows)
    return rows
