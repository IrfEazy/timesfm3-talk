#!/usr/bin/env python3
# scripts/08b_exp_mtg_inference.py
"""Experiment A inference (Preregistration v2 sections 1-2), offline.

Reads the committed 02b outputs and writes the preregistered verdicts (strict rule of
the 2026-10-10 amendment in `verdict`; original rule in `verdict_original_rule`):

    uv run scripts/08b_exp_mtg_inference.py --as-of 2026-09-04

Inputs (each missing one stops the run naming its producer):
    results/exp_mtg_benchmark_primary.parquet      scripts/02b_exp_mtg_benchmark.py
    results/exp_mtg_benchmark_leaderboard.parquet  scripts/02b_exp_mtg_benchmark.py
    configs/benchmark_pool_v2.csv  (is_showcase)   scripts/01b_select_pool.py
    results/inference_selfcheck.parquet            scripts/08_inference_selfcheck.py

Optional inputs (missing ones never fail the run; their rows say "not run" or
are simply absent):
    results/exp_mtg_benchmark_primary__preregistered_v2_showcase.parquet
        -> showcase_only rows (descriptive); without it they say "not run"
    results/exp_mtg_benchmark_leaderboard__preregistered_v2_timesfm2p5.parquet
        -> TimesFM-2.5 vs 3 secondary ("model comparison only, not a cutoff control")
    results/exp_mtg_benchmark_primary__preregistered_v2_ffill{0,1}.parquet
        -> forward-fill 0/1 vs 3 days secondary

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
    p.add_argument(
        "--showcase-primary",
        type=Path,
        default=None,
        help="default: sibling exp_mtg_benchmark_primary__preregistered_v2_showcase.parquet",
    )
    p.add_argument(
        "--leaderboard-2p5",
        type=Path,
        default=None,
        help="default: sibling exp_mtg_benchmark_leaderboard__preregistered_v2_timesfm2p5.parquet",
    )
    p.add_argument(
        "--ffill-primary",
        action="append",
        default=None,
        metavar="K=PATH",
        help="forward-fill sensitivity export, repeatable (default: <dir of --primary>/"
        "exp_mtg_benchmark_primary__preregistered_v2_ffill{0,1}.parquet when present)",
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


def _ffill_paths(specs: list[str] | None, base: Path) -> dict[int, Path]:
    if specs:
        out = {}
        for spec in specs:
            k, _, path = spec.partition("=")
            if not path or not k.isdigit():
                raise SystemExit(f"error: --ffill-primary expects K=PATH, got {spec!r}")
            out[int(k)] = Path(path)
        return out
    return {
        k: base / f"exp_mtg_benchmark_primary__preregistered_v2_ffill{k}.parquet" for k in (0, 1)
    }


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
    sibling = args.primary.parent
    showcase_path = args.showcase_primary or (
        sibling / "exp_mtg_benchmark_primary__preregistered_v2_showcase.parquet"
    )
    lb_2p5_path = args.leaderboard_2p5 or (
        sibling / "exp_mtg_benchmark_leaderboard__preregistered_v2_timesfm2p5.parquet"
    )
    optional: dict[str, Path] = {}
    showcase_primary = None
    if showcase_path.exists():
        showcase_primary = pd.read_parquet(showcase_path)
        optional["showcase_primary"] = showcase_path
    verdict = exp_a.replace_showcase_rows(
        verdict, showcase_primary, selfcheck, n_boot=args.n_boot, seed=args.seed
    )
    secondary = exp_a.secondary_table(leaderboard, n_boot=args.n_boot, seed=args.seed)
    extra = []
    if lb_2p5_path.exists():
        optional["leaderboard_2p5"] = lb_2p5_path
        extra.append(
            exp_a.model_comparison_table(
                pd.read_parquet(lb_2p5_path),
                leaderboard,
                n_boot=args.n_boot,
                seed=args.seed,
            )
        )
    ffill_paths = _ffill_paths(args.ffill_primary, args.primary.parent)
    ffill_frames = {}
    for k, path in ffill_paths.items():
        if path.exists():
            ffill_frames[k] = pd.read_parquet(path)
            optional[f"ffill{k}_primary"] = path
    if ffill_frames:
        extra.append(
            exp_a.ffill_sensitivity_table(ffill_frames, primary, n_boot=args.n_boot, seed=args.seed)
        )
    extra = [e for e in extra if len(e)]
    if extra:
        secondary = pd.concat([secondary, *extra], ignore_index=True)
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
    lines_original = [
        f"{r.subset} h={r.horizon} {r.verdict_original_rule}" for r in first.itertuples(index=False)
    ]
    payload = {
        "script": SCRIPT_NAME,
        "inputs": {k: {"path": str(v), "sha256": _sha256(v)} for k, v in inputs.items()},
        "optional_inputs": {k: {"path": str(v), "sha256": _sha256(v)} for k, v in optional.items()},
        "optional_inputs_not_present": sorted(
            {"showcase_primary", "leaderboard_2p5", *(f"ffill{k}_primary" for k in ffill_paths)}
            - set(optional)
        ),
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
        "rule_version": exp_a.RULE_VERSION,
        "rule_description": (
            "beats/loses iff primary cluster CI and calendar-block CI, for both median and "
            "n-weighted mean skill, all lie entirely above/below 0; else inconclusive"
        ),
        "original_rule_version": exp_a.ORIGINAL_RULE_VERSION,
        "verdicts": lines,
        "verdicts_original_rule": lines_original,
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
