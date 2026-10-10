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
        --config configs/benchmark_preregistered_v2.json --dry-run
    uv run scripts/02b_exp_mtg_benchmark.py \
        --config configs/benchmark_preregistered_v2.json --as-of 2026-09-04 --resume

`--as-of YYYY-MM-DD` is REQUIRED for a real (non-dry-run) run: it is the
cached data's own cutoff date, recorded verbatim in the run manifest. It is
not inferred from the clock, because a run against months-old cached data
would otherwise record today's date as if it were the data's cutoff.

Requires (non-dry-run only):
  - scripts/01_fetch_data.py already run for the cards this config needs
  - preflight (GPU, HF token, checkpoint access + resolved revision) passes;
    it runs before any model load. --allow-cpu permits a CPU smoke run.
  - a forecaster: TimesFM-3 (default, gated HF checkpoint) or TimesFM-2.5
    (--adapter timesfm2.5). The 2.5 adapter is univariate-only -- a config
    with any `multivariate*` mode is rejected up front.

STREAMING + RESUME: each grid cell (context, horizon, transform,
make_positive, mode) is computed, committed and released on its own, so
memory is bounded by one cell, never the whole grid. Cells live in a run
directory keyed by the execution fingerprint (config incl. defaults, git
state, checkpoint revision, card manifest and input-cache hashes, schema and
metric versions, seed):

    results/benchmark/<config_id>/<fingerprint12>/        (gitignored)
        manifest.json, cell=<cell_id>/{part.parquet,done.json}

A cell counts only once its done.json commit marker is published (see
tfm3lab.cellstore for the protocol). After a crash or a session timeout,
rerun the same command with --resume: valid cells are skipped, uncommitted
debris is quarantined and recomputed, damaged committed cells fail loudly
(--repair quarantines and recomputes them), and a changed fingerprint is
refused before anything is touched. A resume with nothing left to do runs
no inference. The summary is computed by reading committed cells only, in
sorted cell_id order, so it does not depend on completion order. A summary
over an incomplete run is refused unless --allow-partial-summary (outputs
then carry a `_partial` suffix and coverage columns); --summary-only skips
inference entirely.

FINGERPRINT + CODE CHANGES: the fingerprint contains the git sha, so
COMMITTING code between sessions changes the sha and therefore requires a new
run directory (--resume will refuse; start fresh or restore the commit). The
uncommitted-code digest covers only result-determining paths (src/tfm3lab/,
this script, configs/, pyproject.toml): editing uv.lock, docs/, tests/,
notebooks or logs does not block --resume. Dependency versions (torch,
timesfm, timesfm3, numpy, pandas, pyarrow, statsmodels) are recorded in the
session manifest but are not part of the fingerprint.

--summary-only and a --resume with no pending cells never run preflight or
touch the GPU, the HF token or the network (the fingerprint uses the
configured checkpoint revision, config.CHECKPOINT_REVISION).

FILESYSTEM: TFM3LAB_DATA_ROOT and the results run directory must be on a
local disk, NOT a FUSE/Drive/network mount (the commit protocol needs POSIX
atomic rename and fsync). Run there, then copy the finished results out at
the end.

Writes to results/ (non-dry-run only):
    exp_mtg_benchmark_leaderboard.parquet
    exp_mtg_benchmark_leaderboard_aggregate.parquet
    exp_mtg_benchmark_primary.parquet   (primary cell only, rows with
        horizon_step == requested_horizon, normalized by the last observed
        context value: errors and y/q10/q50/q90 ratios, `scale_valid` flags
        NaN/zero scales, no actual prices -- small and committable)
    manifests/benchmark-<config_id>-<run_id>-<UTC timestamp>.json
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import inspect
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa

from tfm3lab import cellstore, config, manifest
from tfm3lab.backtest import (
    IDENTITY_TRANSFORM,
    LOG1P_TRANSFORM,
    SeriesData,
    run_multivariate_backtest,
    run_univariate_backtest,
)
from tfm3lab.benchmark import (
    CELL_SCHEMA,
    CELL_SCHEMA_VERSION,
    METRIC_VERSION,
    AblationCombo,
    canonicalize_cell_frame,
    common_origin_set,
    dry_run_report,
    iter_ablation_combos,
    primary_export_frame,
    select_placebo_panel,
)
from tfm3lab.benchmark_config import BenchmarkConfig, load_benchmark_config
from tfm3lab.data.mtg import DEFAULT_CARDS, CardSpec, load_card_manifest
from tfm3lab.preflight import preflight_model
from tfm3lab.summarize import aggregate_leaderboard, compute_mase_scales, summarize_leaderboard

_TRANSFORMS = {"raw": IDENTITY_TRANSFORM, "log1p": LOG1P_TRANSFORM}
_GROUP_COLS = (
    "mode",
    "transform",
    "make_positive",
    "context_len",
    "requested_horizon",
    "series",
    "horizon_step",
)
# Config fields that do not affect results; excluded from the fingerprint.
_NON_SCIENTIFIC_CONFIG_FIELDS = ("description",)

LEADERBOARD_NAME = "exp_mtg_benchmark_leaderboard"
AGGREGATE_NAME = "exp_mtg_benchmark_leaderboard_aggregate"
PRIMARY_NAME = "exp_mtg_benchmark_primary"


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


def _card_manifest_sha256(cfg: BenchmarkConfig, cards_override: str | None) -> str:
    spec = cards_override or cfg.cards
    if spec == "showcase":
        return cellstore.sha256_text(
            cellstore.canonical_json([dataclasses.asdict(c) for c in DEFAULT_CARDS])
        )
    return cellstore.file_sha256(spec)


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
        help="report the grid and projected rows/bytes per cell; no preflight, no writes",
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
    parser.add_argument(
        "--allow-cpu", action="store_true", help="skip the GPU requirement (smoke runs)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the run directory of this exact execution fingerprint",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="with --resume: quarantine and recompute committed cells that fail verification",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="no inference: summarize the cells already committed",
    )
    parser.add_argument(
        "--allow-partial-summary",
        action="store_true",
        help="summarize an incomplete run (outputs get a _partial suffix and coverage columns)",
    )
    return parser


_DEPENDENCIES = ("torch", "timesfm", "timesfm3", "numpy", "pandas", "pyarrow", "statsmodels")


def _dependency_versions() -> dict[str, str]:
    """Recorded in the session manifest; deliberately NOT in the fingerprint."""
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in _DEPENDENCIES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "unknown"
    return out


def _git_state() -> dict:
    """sha, dirty flag and a digest of the result-determining sources only
    (cellstore.RESULT_DETERMINING_PATHS)."""
    return {
        "sha": manifest._git_sha(),
        "dirty": manifest.capture_git_state(),
        "dirty_digest": cellstore.git_worktree_digest(manifest._REPO_ROOT),
    }


def _load_forecaster(adapter: str, revision: str | None):
    """Returns (forecaster, use_symmetric_averaging) at the preflight-resolved revision."""
    if adapter == "timesfm3":
        from tfm3lab.model import load_forecaster

        return load_forecaster(revision=revision), True
    from tfm3lab.model_2p5 import load_forecaster_2p5

    return load_forecaster_2p5(revision=revision), False  # no symmetric averaging in 2.5


def _execution_order(combos: list[AblationCombo]) -> list[AblationCombo]:
    """Order in which pending cells are computed (enumeration order). The
    summary never depends on it -- tests reverse it to prove that."""
    return list(combos)


def _summarize_kwargs(cfg: BenchmarkConfig) -> dict:
    """origin_stride is passed only if this summarize_leaderboard accepts it."""
    if "origin_stride" in inspect.signature(summarize_leaderboard).parameters:
        return {"origin_stride": cfg.origin_stride}
    return {}


def _run_cell(forecaster, combo, combo_series, origins, cfg, use_sym, adapter) -> pd.DataFrame:
    run_fn = run_univariate_backtest if combo.mode == "univariate" else run_multivariate_backtest
    adapter_label = "timesfm3" if adapter == "timesfm3" else "timesfm2p5"
    return run_fn(
        forecaster,
        combo_series,
        origins,
        combo.context_len,
        combo.horizon,
        season_length=cfg.season_length,
        use_symmetric_averaging=use_sym,
        make_positive=combo.make_positive,
        mode_label=f"{adapter_label}_{combo.mode}",
        transform=_TRANSFORMS[combo.transform],
    )


def summarize_store(store: cellstore.RunStore, cfg: BenchmarkConfig, mase_scales: dict):
    """Leaderboard + aggregate computed by streaming over committed cells in
    sorted cell_id order. Every leaderboard group lies inside one cell (all
    cell dimensions are group columns), so per-cell summaries concatenated
    equal the whole-grid summary."""
    baseline_cols = tuple(f"baseline_{b}" for b in cfg.baselines)
    kwargs = _summarize_kwargs(cfg)
    parts = []
    for _cid, table in store.iter_committed():
        df = table.to_pandas()
        # an all-NaN baseline column means "not computed in this cell": drop
        # it, as summarize_leaderboard skips absent baselines.
        present = tuple(c for c in baseline_cols if df[c].notna().any())
        df = df.drop(columns=[c for c in baseline_cols if c not in present])
        parts.append(
            summarize_leaderboard(
                df, mase_scales, group_cols=_GROUP_COLS, baseline_cols=present, **kwargs
            )
        )
        del df, table
    if not parts:
        raise SystemExit("no committed cells to summarize")
    sort_cols = ["baseline_method", *_GROUP_COLS]
    leaderboard = (
        pd.concat(parts, ignore_index=True)
        .sort_values(sort_cols, kind="mergesort")
        .reset_index(drop=True)
    )
    aggregate = aggregate_leaderboard(leaderboard)
    agg_sort = [c for c in sort_cols if c in aggregate.columns]
    aggregate = aggregate.sort_values(agg_sort, kind="mergesort").reset_index(drop=True)
    return leaderboard, aggregate, bool(kwargs)


def export_primary(store: cellstore.RunStore, combos, cfg: BenchmarkConfig) -> pd.DataFrame | None:
    primary_ids = sorted(c.cell_id for c in combos if c.is_primary)
    committed = [cid for cid in primary_ids if cid in store.committed]
    if not committed:
        return None
    frames = [
        primary_export_frame(table.to_pandas(), cfg.baselines)
        for _cid, table in store.iter_committed(cell_ids=committed)
    ]
    return pd.concat(frames, ignore_index=True)


def _label(df: pd.DataFrame, cfg, run_id, n_committed, n_expected) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "run_id", run_id)
    df.insert(0, "config_id", cfg.config_id)
    df["n_cells_committed"] = n_committed
    df["n_cells_expected"] = n_expected
    df["summary_complete"] = n_committed == n_expected
    return df


def _write_light(df: pd.DataFrame, name: str, partial: bool) -> Path:
    path = config.RESULTS_DIR / f"{name}{'_partial' if partial else ''}.parquet"
    cellstore.atomic_write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    return path


def _dry_run(cfg, cards, args) -> None:
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


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.dry_run and args.as_of is None:
        parser.error(
            "--as-of YYYY-MM-DD is required for a real run -- it records the cached "
            "data's own cutoff in the manifest and is never inferred from today's "
            "clock (only --dry-run, which writes no manifest, may omit it)"
        )
    if args.repair and not args.resume:
        parser.error("--repair only applies together with --resume")
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
        _dry_run(cfg, cards, args)
        return

    git = _git_state()
    if args.adapter == "timesfm3":
        repo_id, revision = config.CHECKPOINT_ID, config.CHECKPOINT_REVISION
    else:
        repo_id, revision = config.CHECKPOINT_ID_2P5, config.CHECKPOINT_REVISION_2P5
    # The fingerprint uses the configured revision (no network lookup), so
    # --summary-only and a --resume with nothing pending need no GPU/token/hub.
    checkpoint: dict | None = None
    if not args.resume and not args.summary_only:
        # fresh run: every cell is pending, so fail fast before creating anything
        checkpoint = preflight_model(repo_id, revision, allow_cpu=args.allow_cpu)

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

    combos = iter_ablation_combos(cfg, card_pool_size=len(cards))
    placebo_series: list[SeriesData] = []
    if any(c.mode == "multivariate_placebo" for c in combos):
        placebo_panel = select_placebo_panel(cards, cfg.placebo_panel_size, cfg.placebo_seed)
        placebo_labels = {c.label for c in placebo_panel}
        placebo_series = [s for s in series_list if s.name in placebo_labels]

    def series_for(combo: AblationCombo) -> list[SeriesData]:
        return placebo_series if combo.mode == "multivariate_placebo" else series_list

    origin_list = tuple(int(o) for o in origins)
    specs = [
        cellstore.CellSpec(
            cell_id=c.cell_id,
            key_domains=(
                ("series", tuple(sorted(s.name for s in series_for(c)))),
                ("origin", origin_list),
                ("horizon_step", tuple(range(1, c.horizon + 1))),
            ),
            meta={"combo": dataclasses.asdict(c)},
        )
        for c in combos
    ]

    resolved_config = {
        k: v for k, v in dataclasses.asdict(cfg).items() if k not in _NON_SCIENTIFIC_CONFIG_FIELDS
    }
    payload = cellstore.build_fingerprint_payload(
        resolved_config=resolved_config,
        git_sha=git["sha"],
        git_dirty=git["dirty"],
        git_dirty_digest=git["dirty_digest"],
        checkpoint={"repo_id": repo_id, "revision": revision},
        card_manifest_sha256=_card_manifest_sha256(cfg, args.cards),
        input_cache_sha256=cellstore.file_sha256(config.CACHE_DIR / "mtg_prices.parquet"),
        schema_version=CELL_SCHEMA_VERSION,
        metric_version=METRIC_VERSION,
        seed=cfg.placebo_seed,
        extra={
            "adapter": args.adapter,
            "cards_spec": args.cards or cfg.cards,
            "resolved_cards": [dataclasses.asdict(c) for c in cards],
            "n_days": n,
            "origins_sha256": cellstore.sha256_text(cellstore.canonical_json(origin_list)),
            "cell_ids": [c.cell_id for c in combos],
        },
    )
    fingerprint = cellstore.fingerprint_of(payload)

    if args.summary_only and not (
        config.RESULTS_DIR / "benchmark" / cfg.config_id / fingerprint[:12]
    ).is_dir():
        raise SystemExit(
            f"error: --summary-only: no run directory for execution fingerprint "
            f"{fingerprint[:12]} under {config.RESULTS_DIR / 'benchmark' / cfg.config_id}"
        )

    try:
        store, opened = cellstore.RunStore.open(
            config.RESULTS_DIR / "benchmark",
            cfg.config_id,
            payload,
            CELL_SCHEMA,
            CELL_SCHEMA_VERSION,
            specs,
            resume=args.resume or args.summary_only,
            repair=args.repair,
        )
    except cellstore.CellStoreError as exc:
        raise SystemExit(f"error: {exc}") from exc
    run_id = store.run_id
    print(
        f"run {cfg.config_id}/{run_id}: {len(opened.committed)} committed, "
        f"{len(opened.pending)} pending, {len(opened.quarantined)} quarantined "
        f"({store.run_dir})"
    )

    computed: list[str] = []
    pending = set(opened.pending)
    if pending and not args.summary_only:
        if checkpoint is None:
            checkpoint = preflight_model(repo_id, revision, allow_cpu=args.allow_cpu)
        forecaster, use_sym = _load_forecaster(args.adapter, checkpoint["revision_resolved"])
        for combo in _execution_order(combos):
            if combo.cell_id not in pending:
                continue
            cell_series = series_for(combo)
            df = _run_cell(forecaster, combo, cell_series, origins, cfg, use_sym, args.adapter)
            table = canonicalize_cell_frame(
                df, combo=combo, run_id=run_id, config_id=cfg.config_id, series_list=cell_series
            )
            del df
            store.commit_cell(combo.cell_id, table)
            del table
            computed.append(combo.cell_id)
            print(f"  committed {combo.cell_id} ({len(computed)}/{len(pending)})")
        use_symmetric_averaging: bool | None = use_sym
    else:
        use_symmetric_averaging = args.adapter == "timesfm3"

    n_expected, n_committed = len(store.cells), len(store.committed)
    partial = not store.complete
    if partial and not args.allow_partial_summary:
        raise SystemExit(
            f"error: only {n_committed}/{n_expected} cells are committed; refusing a "
            "partial summary (rerun with --resume, or pass --allow-partial-summary)"
        )

    mase_scales = compute_mase_scales(series_list, boundary_index=int(origins.min()))
    leaderboard, aggregate, stride_passed = summarize_store(store, cfg, mase_scales)
    outputs = [
        _write_light(
            _label(leaderboard, cfg, run_id, n_committed, n_expected), LEADERBOARD_NAME, partial
        ),
        _write_light(
            _label(aggregate, cfg, run_id, n_committed, n_expected), AGGREGATE_NAME, partial
        ),
    ]
    primary = export_primary(store, combos, cfg)
    if primary is not None:
        outputs.append(_write_light(primary, PRIMARY_NAME, partial))
    else:
        print("NOTE: no primary cell (univariate/raw/make_positive/context 512) in this grid")

    if checkpoint is None:  # no inference this session: nothing was resolved
        checkpoint = {
            "repo_id": repo_id,
            "revision_requested": revision,
            "revision_resolved": None,
        }
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest_payload = {
        "dependency_versions": _dependency_versions(),
        "config_id": cfg.config_id,
        "run_id": run_id,
        "fingerprint": fingerprint,
        "fingerprint_payload": payload,
        "run_dir": str(store.run_dir),
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
            "max_origins": cfg.max_origins,
            "placebo_seed": cfg.placebo_seed,
            "season_length": cfg.season_length,
            "n_combos": len(combos),
            "n_origins": len(origins),
        },
        "baselines": list(cfg.baselines),
        "use_symmetric_averaging": use_symmetric_averaging,
        "summarize_origin_stride_passed": stride_passed,
        "cells": [c.cell_id for c in combos],
        "committed_cells": store.committed_hashes(),
        "session": {
            "computed": computed,
            "skipped_committed": opened.committed,
            "quarantined": opened.quarantined,
            "summary_complete": not partial,
            "n_cells_committed": n_committed,
            "n_cells_expected": n_expected,
        },
        "outputs": [
            {"path": str(p), "bytes": p.stat().st_size, "sha256": cellstore.file_sha256(p)}
            for p in outputs
        ],
    }
    manifest_path = manifest.write_manifest(
        manifest_payload,
        config.RESULTS_DIR / "manifests" / f"benchmark-{cfg.config_id}-{run_id}-{ts}.json",
        as_of=as_of,
        checkpoint=checkpoint,
        seed=cfg.placebo_seed,
        git_dirty=git["dirty"],
    )

    print(
        f"{len(computed)} cell(s) computed this session; {n_committed}/{n_expected} committed; "
        f"{len(leaderboard)} leaderboard rows, {len(aggregate)} aggregate rows"
        f"{' (PARTIAL)' if partial else ''}; manifest {manifest_path}"
    )


if __name__ == "__main__":
    main()
