#!/usr/bin/env python3
"""Latency benchmark (Preregistration v2 s6): descriptive only.

Grid batch {1,8,32,128} x context {128,512} x horizon {28,64}. TimesFM runs on
the GPU with torch.cuda.synchronize() around every timed call; CPU naive and
ETS are timed on the same contexts (ETS is fitted per series). Reported as
seconds per call and seconds per 1k forecasts on the recorded GPU. No dollar
figures and no deployment recommendation (the checkpoint licence is
non-commercial). Run on the GPU host, not locally: CPU runs are smoke only.

    uv run scripts/09_latency.py --as-of 2026-09-04
    uv run scripts/09_latency.py --as-of 2026-09-04 --adapter timesfm2.5
    uv run scripts/09_latency.py --as-of 2026-09-04 --allow-cpu --repeats 2 --warmup 1

Writes results/latency_v2.parquet (timesfm3) or results/latency_v2_2p5.parquet
(timesfm2.5) and a run manifest.
"""

from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from tfm3lab import config
from tfm3lab.baselines import ets_forecast, naive_forecast
from tfm3lab.latency import (
    BATCH_SIZES,
    CONTEXTS,
    HORIZONS,
    P95_METHOD,
    check_data_sufficiency,
    grid,
    guarded_ets,
    run_grid,
    write_rows_atomic,
)
from tfm3lab.manifest import capture_git_state, write_run_manifest
from tfm3lab.preflight import preflight_model, run_environment


def output_path(adapter: str):
    """Per-adapter output so the TimesFM-3 and 2.5 runs do not overwrite each other."""
    suffix = "" if adapter == "timesfm3" else "_2p5"
    return config.RESULTS_DIR / f"latency_v2{suffix}.parquet"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--as-of", required=True, type=dt.date.fromisoformat, metavar="YYYY-MM-DD")
    p.add_argument("--adapter", choices=("timesfm3", "timesfm2.5"), default="timesfm3")
    p.add_argument("--allow-cpu", action="store_true", help="skip GPU preflight (smoke only)")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    return p


def load_series() -> list[np.ndarray]:
    path = config.CACHE_DIR / "mtg_prices.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return [g.sort_values("date")["value"].to_numpy(dtype=float) for _, g in df.groupby("series")]


def _cuda_sync():
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return torch.cuda.synchronize
    return None


def _cuda_cleanup() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


PER_CORE_BATCH_SIZE = 16  # tfm3lab.model.load_forecaster default


def main() -> None:
    args = build_parser().parse_args()
    out = output_path(args.adapter)
    git_dirty = capture_git_state()
    if args.adapter == "timesfm2.5":
        repo, rev = config.CHECKPOINT_ID_2P5, config.CHECKPOINT_REVISION_2P5
    else:
        repo, rev = config.CHECKPOINT_ID, config.CHECKPOINT_REVISION
    # Data sufficiency before any model/GPU work.
    series = load_series()
    n_usable = check_data_sufficiency(series, CONTEXTS, allow_synthetic=args.allow_cpu)
    checkpoint = preflight_model(repo, rev, allow_cpu=args.allow_cpu)
    env = run_environment()
    sync = _cuda_sync()

    if args.adapter == "timesfm2.5":
        from tfm3lab.model_2p5 import load_forecaster_2p5

        forecaster = load_forecaster_2p5(repo, revision=checkpoint["revision_resolved"])
    else:
        from tfm3lab.model import load_forecaster

        forecaster = load_forecaster(
            per_core_batch_size=PER_CORE_BATCH_SIZE, revision=checkpoint["revision_resolved"]
        )

    def methods_for(ctxs, horizon):
        def run_model():
            list(forecaster.predict_batch(contexts=ctxs, horizon=horizon, return_quantiles=True))

        def run_naive():
            for c in ctxs:
                naive_forecast(c, horizon)

        run_ets_all, ets_failures = guarded_ets(ets_forecast)

        def run_ets():
            run_ets_all(ctxs, horizon)

        return [
            (args.adapter, run_model, sync, env["gpu_name"], None),
            ("naive", run_naive, None, "cpu", None),
            ("ets", run_ets, None, "cpu", lambda: {"ets_failed_series": ets_failures()}),
        ]

    cells = grid()
    rows: list[dict] = []
    outcome = "complete"

    def on_row(row, all_rows):
        write_rows_atomic(all_rows, out)
        print(
            f"{row['method']} b={row['batch']} c={row['context']} h={row['horizon']}: "
            f"{row['status']}"
            + (f" {row['median_s']:.4f}s median" if row["status"] == "ok" else "")
        )

    env_cols = {f"env_{k}": v for k, v in env.items()}
    try:
        rows = run_grid(
            cells,
            series,
            methods_for,
            repeats=args.repeats,
            warmup=args.warmup,
            allow_synthetic=args.allow_cpu,
            on_row=on_row,
            cleanup=_cuda_cleanup,
            row_extra={"per_core_batch_size": PER_CORE_BATCH_SIZE, **env_cols},
        )
    except BaseException as exc:
        outcome = f"aborted: {type(exc).__name__}: {exc}"
        raise
    finally:
        path = write_run_manifest(
            "latency",
            [out],
            as_of=args.as_of,
            checkpoint=checkpoint,
            seed=None,
            git_dirty=git_dirty,
            extra={
                "args": {k: str(v) for k, v in vars(args).items()},
                "environment": env,
                "run_outcome": outcome,
                "grid": {
                    "batch_sizes": list(BATCH_SIZES),
                    "contexts": list(CONTEXTS),
                    "horizons": list(HORIZONS),
                    "cells": [list(c) for c in cells],
                },
                "repeats": args.repeats,
                "warmup": args.warmup,
                "p95_method": P95_METHOD,
                "n_usable_series_max_context": n_usable,
                "cell_status_counts": {
                    st: sum(1 for r in rows if r["status"] == st) for st in ("ok", "oom", "error")
                },
                "per_core_batch_size": PER_CORE_BATCH_SIZE,
                "batching_note": (
                    "timesfm3 TimesFM3Evaluator.predict_batch iterates queries in chunks of "
                    "per_core_batch_size (timesfm3_forecaster.py); a batch > "
                    f"{PER_CORE_BATCH_SIZE} is run as sequential forward passes inside one call, "
                    "so batch 32/128 time several chunks, not one large forward pass."
                ),
                "data_sources": sorted({r["data_source"] for r in rows}),
            },
        )
        print(f"Wrote {out}; manifest {path}")


if __name__ == "__main__":
    main()
