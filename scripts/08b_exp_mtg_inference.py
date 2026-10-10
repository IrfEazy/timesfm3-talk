#!/usr/bin/env python3
# scripts/08b_exp_mtg_inference.py
"""Experiment A inference (Preregistration v2 sections 1-2), offline.

Reads the committed 02b outputs and writes the preregistered verdicts:

    uv run scripts/08b_exp_mtg_inference.py --as-of 2026-09-04

Inputs (each missing one stops the run naming its producer):
    results/exp_mtg_benchmark_primary.parquet      scripts/02b_exp_mtg_benchmark.py
    results/exp_mtg_benchmark_leaderboard.parquet  scripts/02b_exp_mtg_benchmark.py
    configs/benchmark_pool_v2.csv  (is_showcase)   scripts/01b_select_pool.py
    results/inference_selfcheck.parquet            scripts/08_inference_selfcheck.py

Outputs:
    results/exp_mtg_v2_verdict.parquet       headline rows, verdict per (subset, h)
    results/exp_mtg_v2_secondary.parquet     descriptive paired comparisons
    results/exp_mtg_v2_significance.parquet  DM p-values with BH / BY q-values
    results/manifests/exp-mtg-v2-inference-<as_of>.json

Exactly one of --as-of or --allow-live-end (dev only) is required.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

import pandas as pd

from tfm3lab import bootstrap, exp_a, manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
SCRIPT_NAME = "scripts/08b_exp_mtg_inference.py"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--as-of",
        type=dt.date.fromisoformat,
        metavar="YYYY-MM-DD",
        default=None,
        help="data freeze date recorded in the manifest",
    )
    g.add_argument(
        "--allow-live-end",
        action="store_true",
        help="use today() as as-of (dev only); records live_end=true",
    )
    p.add_argument("--primary", type=Path, default=RESULTS / "exp_mtg_benchmark_primary.parquet")
    p.add_argument(
        "--leaderboard", type=Path, default=RESULTS / "exp_mtg_benchmark_leaderboard.parquet"
    )
    p.add_argument("--pool", type=Path, default=REPO_ROOT / "configs" / "benchmark_pool_v2.csv")
    p.add_argument("--selfcheck", type=Path, default=RESULTS / "inference_selfcheck.parquet")
    p.add_argument("--output-dir", type=Path, default=RESULTS)
    p.add_argument("--manifest-dir", type=Path, default=RESULTS / "manifests")
    p.add_argument("--n-boot", type=int, default=bootstrap.DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=bootstrap.DEFAULT_SEED)
    return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(path: Path, producer: str) -> Path:
    if not path.exists():
        raise SystemExit(f"error: missing input {path}; produce it with {producer}")
    return path


def _resolve_as_of(args) -> tuple[dt.date, bool]:
    if args.allow_live_end:
        print("WARNING: --allow-live-end uses today() as as_of; not reproducible.", file=sys.stderr)
        return dt.date.today(), True
    if args.as_of is not None:
        return args.as_of, False
    raise SystemExit("error: one of --as-of YYYY-MM-DD or --allow-live-end is required")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    as_of, live_end = _resolve_as_of(args)
    inputs = {
        "primary": _require(args.primary, "scripts/02b_exp_mtg_benchmark.py"),
        "leaderboard": _require(args.leaderboard, "scripts/02b_exp_mtg_benchmark.py"),
        "pool": _require(args.pool, "scripts/01b_select_pool.py"),
        "selfcheck": _require(args.selfcheck, "scripts/08_inference_selfcheck.py"),
    }
    primary = pd.read_parquet(inputs["primary"])
    leaderboard = pd.read_parquet(inputs["leaderboard"])
    selfcheck = pd.read_parquet(inputs["selfcheck"])
    pool = pd.read_csv(inputs["pool"])
    if "is_showcase" not in pool.columns or "label" not in pool.columns:
        raise SystemExit(
            f"error: {inputs['pool']} lacks label/is_showcase; rerun scripts/01b_select_pool.py"
        )
    flag = pool["is_showcase"].astype(str).str.lower().isin(("true", "1"))
    showcase = set(pool.loc[flag, "label"].astype(str))

    pool_cards = set(pool["label"].astype(str))
    verdict = exp_a.headline_table(
        primary, showcase, selfcheck, pool_cards=pool_cards, n_boot=args.n_boot, seed=args.seed
    )
    coverage = exp_a.pool_coverage(primary, pool_cards)
    secondary = exp_a.secondary_table(leaderboard, n_boot=args.n_boot, seed=args.seed)
    significance = exp_a.significance_frame(leaderboard)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "verdict": args.output_dir / "exp_mtg_v2_verdict.parquet",
        "secondary": args.output_dir / "exp_mtg_v2_secondary.parquet",
        "significance": args.output_dir / "exp_mtg_v2_significance.parquet",
    }
    for key, frame in (
        ("verdict", verdict),
        ("secondary", secondary),
        ("significance", significance),
    ):
        frame.to_parquet(outputs[key], index=False)

    first = verdict.drop_duplicates(["subset", "horizon"])
    lines = [f"{r.subset} h={r.horizon} {r.verdict}" for r in first.itertuples(index=False)]
    payload = {
        "script": SCRIPT_NAME,
        "inputs": {k: {"path": str(v), "sha256": _sha256(v)} for k, v in inputs.items()},
        "outputs": {k: {"path": str(v), "sha256": _sha256(v)} for k, v in outputs.items()},
        "settings": {
            "n_boot": args.n_boot,
            "seed": args.seed,
            "ci_level": bootstrap.DEFAULT_CI,
            "primary_ci": "paired whole-card cluster bootstrap",
            "sensitivity_ci": "synchronized calendar-block bootstrap (origin_date)",
            "horizons": list(exp_a.PRIMARY_HORIZONS),
        },
        "n_showcase_cards_in_pool_csv": len(showcase),
        "n_pool_csv": coverage["n_pool_csv"],
        "n_cards_in_export": coverage["n_cards_in_export"],
        "pool_cards_missing_from_export": coverage["missing"],
        "selfcheck_acceptance_failed": exp_a.selfcheck_acceptance_failed(selfcheck),
        "verdicts": lines,
    }
    manifest_path = args.manifest_dir / f"exp-mtg-v2-inference-{as_of.isoformat()}.json"
    manifest.write_manifest(payload, manifest_path, as_of=as_of, live_end=live_end, seed=args.seed)
    for line in lines:
        print(line)
    print(
        f"wrote {', '.join(str(p) for p in outputs.values())} and {manifest_path}", file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
