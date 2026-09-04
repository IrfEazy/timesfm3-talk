#!/usr/bin/env python3
# scripts/02b_exp_mtg_benchmark.py
"""Experiment A, configurable: declarative context/horizon/ablation grid
over the MTG card pool, multi-baseline leaderboard, TimesFM-3 or TimesFM-2.5.

This is Experiment A's PREREGISTERED evaluation entry point -- distinct from
scripts/02_exp_mtg.py, whose fixed single-config output
(results/exp_mtg_*.parquet) backs the numbers already committed in
docs/talk-outline.md. This script's own outputs are prefixed
exp_mtg_benchmark_* and are never read by 02_exp_mtg.py, 04-07, or the
existing slides -- running this script does not change any existing number.

Usage:
    uv run scripts/02b_exp_mtg_benchmark.py \
        --config configs/benchmark_preregistered.example.json --dry-run
    uv run scripts/02b_exp_mtg_benchmark.py \
        --config configs/benchmark_preregistered.example.json --as-of 2026-09-01

`--as-of YYYY-MM-DD` is REQUIRED for a real (non-dry-run) run: it is the
cached data's own cutoff date, recorded verbatim in the run manifest. It is
not inferred from the clock, because a run against months-old cached data
would otherwise record today's date as if it were the data's cutoff.

Requires (non-dry-run only):
  - scripts/01_fetch_data.py already run for the cards this config needs
  - a loaded forecaster: TimesFM-3 (default, gated HF checkpoint) or
    TimesFM-2.5 (--adapter timesfm2.5, Apache-2.0, ungated). The 2.5
    adapter is univariate-only -- a config with any `multivariate*` mode is
    rejected up front, since TimesFM_2p5_200M_torch.forecast() has no
    variate concept.

MEMORY WARNING: this script accumulates every combo's predictions in memory
and writes them in one shot at the end. The shipped example config's full
grid (4 context lengths x 5 horizons x 2 transforms x 2 make_positive x
2 modes x 7 cards) can produce ~10+ million prediction rows and several GB
of resident memory -- doubling transiently during the final pd.concat --
which will likely OOM a free-tier Colab instance (12.7 GB). For a first
real run, start from a smaller custom config (fewer context_lengths and
horizons, or a larger `origin_stride` / a `max_origins` cap), especially on
a memory-constrained GPU instance. --dry-run reports the combo count and an
estimated predict_batch call count before you commit to one.

Writes to results/ (non-dry-run only):
    exp_mtg_benchmark_raw_predictions.parquet
    exp_mtg_benchmark_leaderboard.parquet
    exp_mtg_benchmark_leaderboard_aggregate.parquet
    manifests/benchmark-<config_id>-<run_id>.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import secrets
import sys
from pathlib import Path

import pandas as pd

from tfm3lab import config, manifest
from tfm3lab.backtest import (
    IDENTITY_TRANSFORM,
    LOG1P_TRANSFORM,
    SeriesData,
    run_multivariate_backtest,
    run_univariate_backtest,
)
from tfm3lab.benchmark import (
    common_origin_set,
    dry_run_report,
    iter_ablation_combos,
    select_placebo_panel,
)
from tfm3lab.benchmark_config import BenchmarkConfig, load_benchmark_config
from tfm3lab.data.mtg import DEFAULT_CARDS, CardSpec, load_card_manifest
from tfm3lab.summarize import aggregate_leaderboard, compute_mase_scales, summarize_leaderboard

_TRANSFORMS = {"raw": IDENTITY_TRANSFORM, "log1p": LOG1P_TRANSFORM}


def _resolve_card_pool(cfg: BenchmarkConfig, cards_override: str | None) -> tuple[CardSpec, ...]:
    spec = cards_override or cfg.cards
    if spec == "showcase":
        return DEFAULT_CARDS
    if spec == "benchmark":
        raise SystemExit(
            "--cards benchmark requires a manifest path -- there is no built-in "
            "wider card list (see configs/benchmark_cards.example.csv's header "
            "for why, and the criteria a real one must satisfy)"
        )
    return load_card_manifest(spec)


def load_cached_series(cards: tuple[CardSpec, ...]) -> list[SeriesData]:
    path = config.CACHE_DIR / "mtg_prices.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/01_fetch_data.py first")
    df = pd.read_parquet(path)
    wanted = {c.label for c in cards}
    df = df[df["series"].isin(wanted)]
    missing = wanted - set(df["series"].unique())
    if missing:
        raise ValueError(
            f"{sorted(missing)} not found in cached series at {path} -- "
            "re-run scripts/01_fetch_data.py with these cards included"
        )
    series_list = [
        SeriesData(
            name=name,
            values=group.sort_values("date")["value"].to_numpy(dtype=float),
            dates=group.sort_values("date")["date"].to_numpy(),
            observed=group.sort_values("date")["observed"].to_numpy(dtype=bool),
        )
        for name, group in df.groupby("series")
    ]
    lengths = {s.name: len(s.values) for s in series_list}
    n = min(lengths.values())
    if len(set(lengths.values())) > 1:
        print(f"  trimming all series to the shortest common length ({n} days): {lengths}")
    return [SeriesData(s.name, s.values[-n:], s.dates[-n:], s.observed[-n:]) for s in series_list]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="path to a BenchmarkConfig JSON file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the grid, no forecaster calls, no results/ writes",
    )
    parser.add_argument(
        "--dry-run-out",
        type=Path,
        default=None,
        help="write the dry-run report JSON here instead of stdout",
    )
    parser.add_argument(
        "--cards",
        default=None,
        help="override the config's 'cards' field: showcase | a manifest path",
    )
    parser.add_argument("--adapter", choices=("timesfm3", "timesfm2.5"), default="timesfm3")
    parser.add_argument(
        "--as-of",
        default=None,
        type=str,
        help=(
            "the cached data's cutoff date (YYYY-MM-DD), recorded in the run "
            "manifest -- required for a real run, ignored by --dry-run"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.dry_run and args.as_of is None:
        parser.error(
            "--as-of YYYY-MM-DD is required for a real run -- it records the cached "
            "data's own cutoff in the manifest and is never inferred from today's "
            "clock (only --dry-run, which writes no manifest, may omit it)"
        )
    as_of = dt.date.fromisoformat(args.as_of) if args.as_of else None

    cfg = load_benchmark_config(args.config)
    cards = _resolve_card_pool(cfg, args.cards)

    if args.adapter == "timesfm2.5":
        incompatible = sorted(
            {
                c.mode
                for c in iter_ablation_combos(cfg, card_pool_size=len(cards))
                if c.mode.startswith("multivariate")
            }
        )
        if incompatible:
            raise SystemExit(
                f"--adapter timesfm2.5 cannot run mode(s) {incompatible}: "
                "TimesFM_2p5_200M_torch.forecast() has no variate concept, so a "
                "stacked (n_series, context_len) context would be misread rather "
                "than jointly modelled -- rerun with --adapter timesfm3, or with a "
                "config whose modes are univariate only"
            )

    if args.dry_run:
        cache_path = config.CACHE_DIR / "mtg_prices.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            wanted = {c.label for c in cards}
            present = df[df["series"].isin(wanted)]
            n_days = int(present.groupby("series").size().min()) if len(present) else 0
        else:
            n_days = 0
            print(
                f"NOTE: {cache_path} not found -- dry-run report uses n_days=0 "
                "(run scripts/01_fetch_data.py first for a realistic origin count)",
                file=sys.stderr,
            )
        report = dry_run_report(cfg, n_days=n_days, card_pool_size=len(cards))
        text = json.dumps(report, indent=2)
        if args.dry_run_out:
            args.dry_run_out.write_text(text, encoding="utf-8")
            print(f"wrote dry-run report to {args.dry_run_out}")
        else:
            print(text)
        return

    series_list = load_cached_series(cards)
    n = len(series_list[0].values)
    origins = common_origin_set(
        n=n,
        context_lengths=cfg.context_lengths,
        horizons=cfg.horizons,
        origin_stride=cfg.origin_stride,
        max_origins=cfg.max_origins,
    )
    if len(origins) == 0:
        raise RuntimeError(
            f"no valid origins: {n} days too short for context_lengths={cfg.context_lengths} "
            f"+ horizons={cfg.horizons}"
        )

    if args.adapter == "timesfm3":
        from tfm3lab.model import load_forecaster

        forecaster = load_forecaster()
        use_symmetric_averaging = True
    else:
        from tfm3lab.model_2p5 import load_forecaster_2p5

        forecaster = load_forecaster_2p5()
        use_symmetric_averaging = False  # not supported by TimesFM2p5Adapter

    combos = iter_ablation_combos(cfg, card_pool_size=len(cards))
    placebo_series: list[SeriesData] = []
    if any(c.mode == "multivariate_placebo" for c in combos):
        placebo_panel = select_placebo_panel(cards, cfg.placebo_panel_size, cfg.placebo_seed)
        placebo_labels = {c.label for c in placebo_panel}
        placebo_series = [s for s in series_list if s.name in placebo_labels]

    run_id = f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"

    all_results = []
    for combo in combos:
        transform = _TRANSFORMS[combo.transform]
        combo_series = placebo_series if combo.mode == "multivariate_placebo" else series_list
        run_fn = (
            run_univariate_backtest if combo.mode == "univariate" else run_multivariate_backtest
        )
        adapter_label = "timesfm3" if args.adapter == "timesfm3" else "timesfm2p5"
        mode_label = f"{adapter_label}_{combo.mode}"
        df = run_fn(
            forecaster,
            combo_series,
            origins,
            combo.context_len,
            combo.horizon,
            season_length=cfg.season_length,
            use_symmetric_averaging=use_symmetric_averaging,
            make_positive=combo.make_positive,
            mode_label=mode_label,
            transform=transform,
        )
        df = df.assign(
            run_id=run_id,
            config_id=cfg.config_id,
            context_len=combo.context_len,
            requested_horizon=combo.horizon,
        )
        all_results.append(df)

    raw_df = pd.concat(all_results, ignore_index=True)
    raw_df = raw_df.rename(columns={"origin_index": "origin"})
    raw_df.to_parquet(
        config.RESULTS_DIR / "exp_mtg_benchmark_raw_predictions.parquet",
        index=False,
        compression="zstd",
    )

    mase_scales = compute_mase_scales(series_list, boundary_index=int(origins.min()))
    group_cols = (
        "mode",
        "transform",
        "make_positive",
        "context_len",
        "requested_horizon",
        "series",
        "horizon_step",
    )
    baseline_cols = tuple(f"baseline_{b}" for b in cfg.baselines)
    leaderboard = summarize_leaderboard(
        raw_df, mase_scales, group_cols=group_cols, baseline_cols=baseline_cols
    )
    leaderboard.to_parquet(
        config.RESULTS_DIR / "exp_mtg_benchmark_leaderboard.parquet", index=False
    )

    aggregate = aggregate_leaderboard(leaderboard)
    aggregate.to_parquet(
        config.RESULTS_DIR / "exp_mtg_benchmark_leaderboard_aggregate.parquet", index=False
    )

    manifest_payload = {
        "config_id": cfg.config_id,
        "run_id": run_id,
        "adapter": args.adapter,
        "resolved_cards": [c.label for c in cards],
        "grid": {
            "context_lengths": list(cfg.context_lengths),
            "horizons": list(cfg.horizons),
            "primary_horizons": list(cfg.primary_horizons),
            "transforms": list(cfg.transforms),
            "make_positive": list(cfg.make_positive),
            "modes": list(cfg.modes),
            "origin_stride": cfg.origin_stride,
            "placebo_seed": cfg.placebo_seed,
            "season_length": cfg.season_length,
            "n_combos": len(combos),
            "n_origins": len(origins),
        },
        "baselines": list(cfg.baselines),
        "use_symmetric_averaging": use_symmetric_averaging,
    }
    manifest.write_manifest(
        manifest_payload,
        config.RESULTS_DIR / "manifests" / f"benchmark-{cfg.config_id}-{run_id}.json",
        as_of=as_of,
    )

    print(
        f"Wrote {len(raw_df)} prediction rows, {len(leaderboard)} leaderboard rows, "
        f"{len(aggregate)} aggregate rows to {config.RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
