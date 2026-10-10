#!/usr/bin/env python3
"""Experiment C (Preregistration v2 s4) - calibration of the P10-P90 interval
by regime x horizon, plus chronological split-conformal (CQR) recalibration.

Offline: reads the committed normalized outputs only, never calls the model.
Inputs (produced elsewhere; a missing one aborts with the producing script):
    results/exp_mtg_benchmark_primary.parquet   <- scripts/02b_exp_mtg_benchmark.py
    results/exp_shock_v2_predictions.parquet    <- scripts/03_exp_shock.py
        (timesfm3_univariate, perturbation none)

Usage: uv run scripts/04_exp_calibration.py --as-of 2026-09-01
Exactly one of --as-of or --allow-live-end is required (no silent default to
today()); --allow-live-end (dev only) uses today() and records live_end=true.
A manifest is written to results/manifests/.

Writes to results/:
    exp_calibration_v2_coverage.parquet   (coverage + 3 CIs by regime x horizon)
    exp_calibration_v2_conformal.parquet  (CQR before/after on the evaluation split)
Regime results are empirical subgroup evaluations, not conformal guarantees.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import pandas as pd

from tfm3lab import bootstrap, calibration, config
from tfm3lab.manifest import capture_git_state, write_run_manifest

MTG_INPUT = ("exp_mtg_benchmark_primary.parquet", "scripts/02b_exp_mtg_benchmark.py")
SHOCK_INPUT = ("exp_shock_v2_predictions.parquet", "scripts/03_exp_shock.py")


def read_input(spec: tuple[str, str]) -> pd.DataFrame:
    name, producer = spec
    path = config.RESULTS_DIR / name
    if not path.exists():
        raise SystemExit(f"error: {path} not found - run {producer} first")
    return pd.read_parquet(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    end_group = parser.add_mutually_exclusive_group()
    end_group.add_argument(
        "--as-of",
        default=None,
        type=dt.date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="data freeze date recorded in the manifest; required unless --allow-live-end",
    )
    end_group.add_argument(
        "--allow-live-end",
        action="store_true",
        help="use today() as the as-of date (dev only); records live_end=true",
    )
    parser.add_argument("--n-boot", type=int, default=bootstrap.DEFAULT_N_BOOT)
    return parser


def resolve_as_of(args: argparse.Namespace) -> tuple[dt.date, bool]:
    """Returns (as_of, live_end); exits if neither --as-of nor --allow-live-end."""
    if args.allow_live_end:
        print(
            "WARNING: --allow-live-end — using today() as as_of; not reproducible, "
            "do not use for a final run.",
            file=sys.stderr,
        )
        return dt.date.today(), True
    if args.as_of is not None:
        return args.as_of, False
    raise SystemExit("error: one of --as-of YYYY-MM-DD or --allow-live-end is required")


def main() -> None:
    args = build_parser().parse_args()
    as_of, live_end = resolve_as_of(args)
    git_dirty = capture_git_state()

    frame = pd.concat(
        [
            calibration.mtg_frame(read_input(MTG_INPUT)),
            calibration.market_frame(read_input(SHOCK_INPUT)),
        ],
        ignore_index=True,
    )
    cov = calibration.coverage_table(frame, n_boot=args.n_boot)
    conf = calibration.conformal_table(frame, n_boot=args.n_boot)
    for _, r in conf[conf["status"] != "ok"].iterrows():
        print(
            f"WARNING: CQR not fitted for regime={r['regime']} horizon={r['horizon']} "
            f"(status={r['status']}); conformal_fixes_it left undetermined (null).",
            file=sys.stderr,
        )
    cov_path = config.RESULTS_DIR / "exp_calibration_v2_coverage.parquet"
    conf_path = config.RESULTS_DIR / "exp_calibration_v2_conformal.parquet"
    cov.to_parquet(cov_path, index=False)
    conf.to_parquet(conf_path, index=False)
    print(cov.drop(columns="label").to_string(index=False))
    print(conf.drop(columns="label").to_string(index=False))

    manifest_path = write_run_manifest(
        "exp_calibration",
        [cov_path, conf_path],
        as_of=as_of,
        live_end=live_end,
        seed=bootstrap.DEFAULT_SEED,
        git_dirty=git_dirty,
        extra={"args": {k: str(v) for k, v in vars(args).items()}},
    )
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
