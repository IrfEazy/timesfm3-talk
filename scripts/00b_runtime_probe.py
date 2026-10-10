#!/usr/bin/env python3
"""Runtime probe: project per-step GPU time from throughput measured now.

After the preflight, load each adapter, time forecast_batch on real cached
contexts at a few (context, horizon, batch) points (warmup, CUDA sync), then
combine the measured seconds-per-forecast with each step's work size (02b
--dry-run, events x origins for 03, series x arms x origins for 05, the
latency grid for 09) to print projected minutes per step and in total against
the remaining weekly GPU quota. Nothing is hard-coded: every duration comes
from the measurement taken on the GPU attached to this session.

    uv run scripts/00b_runtime_probe.py --as-of 2026-09-04 --quota-hours 12
    uv run scripts/00b_runtime_probe.py --as-of 2026-09-04 \
        --steps 02b_preregistered_v2,05_covariates

Writes results/runtime_probe.parquet and a run manifest. The projection is an
estimate of model time only (CPU baselines, I/O and covariate overhead are not
measured); treat it as a planning aid, not a guarantee.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from tfm3lab import config, latency
from tfm3lab import kaggle as kg
from tfm3lab.manifest import capture_git_state, write_run_manifest
from tfm3lab.preflight import preflight_model, run_environment

SCRIPTS = Path(__file__).resolve().parent
OUT_NAME = "runtime_probe.parquet"
# (context, horizon, batch): every context and batch size of the step-09
# latency grid (latency.grid()), measured at its longest horizon, so each
# latency cell is priced per call from a measurement with the same batch.
# 02b/03/05 are priced per forecast from the largest measured batch.
DEFAULT_POINTS = tuple(sorted({(c, max(latency.HORIZONS), b) for b, c, _h in latency.grid()}))


def grid_contexts_missing(points) -> list[int]:
    """Latency-grid contexts that no measured point covers (a custom --points
    that omits one would make step 09 extrapolate)."""
    have = {c for c, _h, _b in points}
    return sorted({c for _b, c, _h in latency.grid()} - have)


def parse_points(text: str) -> tuple[tuple[int, int, int], ...]:
    """'128x64x1,512x64x32' -> ((128, 64, 1), (512, 64, 32))"""
    out = []
    for chunk in text.split(","):
        parts = chunk.strip().lower().split("x")
        if len(parts) != 3 or not all(p.isdigit() and int(p) > 0 for p in parts):
            raise argparse.ArgumentTypeError(f"bad point {chunk!r}; use CONTEXTxHORIZONxBATCH")
        out.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return tuple(out)


def parse_steps(text: str) -> list[str]:
    ids = [t.strip() for t in text.split(",") if t.strip()]
    try:
        return kg.validate_experiments(ids)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--as-of", required=True, type=dt.date.fromisoformat, metavar="YYYY-MM-DD")
    p.add_argument(
        "--quota-hours",
        type=float,
        default=None,
        help="remaining weekly GPU quota in hours; the plan is compared against it",
    )
    p.add_argument(
        "--steps",
        type=parse_steps,
        default=[s for s in kg.DEFAULT_EXPERIMENTS if s != kg.PROBE_ID],
        help="comma-separated step ids to project (default: every GPU step)",
    )
    p.add_argument("--points", type=parse_points, default=DEFAULT_POINTS, help="CxHxB,CxHxB,...")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--allow-cpu", action="store_true", help="skip the GPU requirement (smoke only)")
    return p


def _load_script(filename: str):
    spec = importlib.util.spec_from_file_location(
        "probe_" + filename.split("_")[0], SCRIPTS / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dry_run(config_path: str) -> dict:
    """02b --dry-run report for a config (no preflight, no writes)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dry.json"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "02b_exp_mtg_benchmark.py"),
                "--config",
                config_path,
                "--dry-run",
                "--dry-run-out",
                str(out),
            ],
            check=True,
            cwd=kg.REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(out.read_text(encoding="utf-8"))


def _origin_counts(script, series, context_len: int, horizon: int, stride: int) -> list[int]:
    return [
        len(script.strided_origins(len(s.values), context_len, horizon, stride)) for s in series
    ]


def gather_work(step_ids: list[str], repeats_09: int, warmup_09: int) -> list[kg.StepWork]:
    """Work size per requested step, in step-table order."""
    works: list[kg.StepWork] = []
    shock_script = cov_script = None
    for sid in [s for s in kg.DEFAULT_EXPERIMENTS if s in step_ids]:
        step = kg.STEP_BY_ID[sid]
        adapter = step.adapter
        if sid == kg.PROBE_ID or adapter is None:
            continue
        if step.script.startswith("scripts/02b"):
            cfg_path = step.args[step.args.index("--config") + 1]
            works.append(kg.work_from_dry_run(sid, adapter, _dry_run(cfg_path)))
        elif step.script.startswith("scripts/03"):
            from tfm3lab import shock

            shock_script = shock_script or _load_script("03_exp_shock.py")
            specs = shock_script.rule_events(shock_script.EVENTS_CSV)
            panel = 5 if adapter == "timesfm3" else 0  # multivariate panel: timesfm3 only
            works.append(
                kg.work_from_shock(
                    sid,
                    adapter,
                    n_events=len(specs),
                    context_len=shock.CONTEXT_LEN,
                    origins_per_event=shock.WINDOW_BEFORE + shock.WINDOW_AFTER,
                    n_perturbation_runs=len(shock_script.PERTURBATIONS) * shock.N_PERTURB_SEEDS,
                    panel_series=panel,
                )
            )
        elif step.script.startswith("scripts/05"):
            cov_script = cov_script or _load_script("05_exp_covariates.py")
            cl, hz, st = cov_script.CONTEXT_LEN, cov_script.HORIZON, cov_script.ORIGIN_STRIDE
            market = [
                s
                for s in cov_script._load_series("market_prices.parquet")
                if s.name in cov_script.MARKET_SERIES
            ]
            pool = {c.label for c in cov_script.load_card_manifest(cov_script.POOL_PATH)}
            mtg = [s for s in cov_script._load_series("mtg_prices.parquet") if s.name in pool]
            works.append(
                kg.work_from_covariates(
                    sid,
                    adapter,
                    context_len=cl,
                    market_origins=_origin_counts(cov_script, market, cl, hz, st),
                    mtg_origins=_origin_counts(cov_script, mtg, cl, hz, st),
                )
            )
        elif step.script.startswith("scripts/09"):
            p9 = _load_script("09_latency.py").build_parser().parse_args(["--as-of", "2000-01-01"])
            works.append(kg.work_from_latency(sid, adapter, latency.grid(), p9.repeats, p9.warmup))
    return works


def _load_forecaster(adapter: str, checkpoint: dict):
    rev = checkpoint["revision_resolved"]
    if adapter == "timesfm2.5":
        from tfm3lab.model_2p5 import load_forecaster_2p5

        return load_forecaster_2p5(config.CHECKPOINT_ID_2P5, revision=rev)
    from tfm3lab.model import load_forecaster

    return load_forecaster(revision=rev)


def _cuda():
    try:
        import torch
    except ImportError:
        return None, lambda: None
    if torch.cuda.is_available():
        return torch, torch.cuda.synchronize
    return None, lambda: None


def to_frame(measurements: list[dict], plan: list[dict], load_s: dict) -> pd.DataFrame:
    rows = [{"kind": "measurement", **m} for m in measurements]
    rows += [{"kind": "load", "adapter": a, "load_s": s} for a, s in load_s.items()]
    rows += [{"kind": "projection", **r} for r in plan]
    return pd.DataFrame(rows)


def adapters_measured(measurements: list[dict], adapters: list[str]) -> int:
    """Exit code: 0 if at least one adapter has an ok measurement, else 1."""
    ok = {m["adapter"] for m in measurements if m.get("status", "ok") == "ok"}
    failed = [a for a in adapters if a not in ok]
    if failed:
        print(f"warning: no usable measurement for {failed}; their steps are not projected")
    return 0 if ok else 1


def main() -> int:
    args = build_parser().parse_args()
    missing = grid_contexts_missing(args.points)
    if missing:
        print(
            f"warning: --points does not measure latency-grid contexts {missing}; "
            "step 09 projections for them are extrapolated"
        )
    git_dirty = capture_git_state()
    adapters = sorted({kg.STEP_BY_ID[s].adapter for s in args.steps if kg.STEP_BY_ID[s].adapter})
    if not adapters:
        raise SystemExit("error: --steps selects nothing the probe can project")

    series_path = config.CACHE_DIR / "mtg_prices.parquet"
    if not series_path.exists():
        raise SystemExit(f"error: {series_path} not found; restore the cache first")
    df = pd.read_parquet(series_path)
    series = [g.sort_values("date")["value"].to_numpy(float) for _, g in df.groupby("series")]

    torch, sync = _cuda()
    env = run_environment()
    measurements: list[dict] = []
    load_s: dict[str, float] = {}
    checkpoints: dict[str, dict] = {}
    for adapter in adapters:
        if adapter == "timesfm2.5":
            repo, rev = config.CHECKPOINT_ID_2P5, config.CHECKPOINT_REVISION_2P5
        else:
            repo, rev = config.CHECKPOINT_ID, config.CHECKPOINT_REVISION
        checkpoints[adapter] = preflight_model(repo, rev, allow_cpu=args.allow_cpu)
        print(f"[{adapter}] {repo}@{checkpoints[adapter]['revision_resolved']}")
        t0 = time.perf_counter()
        try:
            forecaster = _load_forecaster(adapter, checkpoints[adapter])
        except Exception as exc:  # noqa: BLE001 - one adapter failing must not end the probe
            print(f"[{adapter}] load failed: {type(exc).__name__}: {exc}")
            measurements.append(
                {
                    "adapter": adapter,
                    "status": "error",
                    "message": f"load failed: {type(exc).__name__}: {exc}",
                }
            )
            continue
        sync()
        load_s[adapter] = time.perf_counter() - t0
        print(f"[{adapter}] model load {load_s[adapter]:.1f} s")
        rows = kg.measure_points(
            forecaster,
            series,
            adapter,
            args.points,
            repeats=args.repeats,
            warmup=args.warmup,
            sync=sync,
        )
        for r in rows:
            detail = f"{r['median_s']:.4f} s/call" if r["status"] == "ok" else r.get("message")
            point = f"c={r['context']} h={r['horizon']} b={r['batch']}"
            print(f"[{adapter}] {point}: {r['status']} {detail}")
        measurements += rows
        del forecaster
        if torch is not None:
            torch.cuda.empty_cache()

    works = gather_work(args.steps, 20, 3)
    plan = kg.project_plan(works, measurements, load_s, args.quota_hours)
    print()
    print(f"GPU: {env['gpu_name']} (torch {env['torch_version']}, CUDA {env['cuda_version']})")
    print(kg.format_plan(plan))
    print("Model time only: CPU baselines, I/O and covariate overhead are not measured.")

    out = config.RESULTS_DIR / OUT_NAME
    to_frame(measurements, plan, load_s).to_parquet(out, index=False)
    manifest = write_run_manifest(
        "runtime_probe",
        [out],
        as_of=args.as_of,
        checkpoint=checkpoints,
        seed=None,
        git_dirty=git_dirty,
        extra={
            "args": {k: str(v) for k, v in vars(args).items()},
            "environment": env,
            "load_seconds": load_s,
            "plan": plan,
        },
    )
    print(f"Wrote {out}; manifest {manifest}")
    return adapters_measured(measurements, adapters)


if __name__ == "__main__":
    raise SystemExit(main())
