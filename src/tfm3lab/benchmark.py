"""Grid orchestration for the declarative MTG benchmark: one shared origin
set across the whole context/horizon grid, ablation-combo enumeration, and
placebo-panel card sampling. Origin *index* math itself stays entirely in
windows.py (untouched) -- this module only decides which combos to run and
reuses windows.valid_origins for the actual computation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .benchmark_config import BenchmarkConfig
from .windows import valid_origins


def common_origin_set(
    n: int,
    context_lengths: tuple[int, ...],
    horizons: tuple[int, ...],
    origin_stride: int = 1,
    max_origins: int | None = None,
) -> np.ndarray:
    """One origin array valid for the grid's *widest* cell
    (max(context_lengths), max(horizons)) -- every smaller cell's own valid
    range is a superset of this (windows.valid_origins' start/end bounds
    are monotonic in context_len/horizon), so reusing this one array for
    every combo satisfies "all configurations must be compared on the same
    origins" exactly, with no per-combo recomputation.
    """
    origins = valid_origins(
        n=n,
        context_len=max(context_lengths),
        horizon=max(horizons),
        max_origins=max_origins,
    )
    if origin_stride > 1:
        origins = origins[::origin_stride]
    return origins


@dataclass(frozen=True)
class AblationCombo:
    """One cell of the ablation grid -- what run_univariate_backtest /
    run_multivariate_backtest (backtest.py, unmodified) get called with."""

    context_len: int
    horizon: int
    transform: str
    make_positive: bool
    mode: str


def iter_ablation_combos(cfg: BenchmarkConfig, card_pool_size: int) -> list[AblationCombo]:
    """Full cartesian product of the grid's ablation dimensions, in a fixed
    deterministic order (context_lengths x horizons x transforms x
    make_positive x modes, each in the order given in the config).

    `multivariate_placebo` combos are dropped unless `card_pool_size` is
    STRICTLY greater than `cfg.placebo_panel_size`. Equality is a skip, not
    a run: sampling a `placebo_panel_size` panel out of a pool of exactly
    that size returns the whole pool, i.e. a "placebo" panel identical to
    the real multivariate panel, which silently defeats the comparison the
    placebo mode exists to make. The caller (dry_run_report / the CLI) is
    responsible for reporting that skip; this function just doesn't emit an
    uninformative combo.
    """
    combos = []
    for context_len in cfg.context_lengths:
        for horizon in cfg.horizons:
            for transform in cfg.transforms:
                for make_positive in cfg.make_positive:
                    for mode in cfg.modes:
                        skip_placebo = (
                            mode == "multivariate_placebo"
                            and card_pool_size <= cfg.placebo_panel_size
                        )
                        if skip_placebo:
                            continue
                        combos.append(
                            AblationCombo(context_len, horizon, transform, make_positive, mode)
                        )
    return combos


def select_placebo_panel(pool: tuple, panel_size: int, seed: int) -> tuple:
    """Deterministic random sample of `panel_size` items from `pool`
    (typically CardSpec instances), seeded so the same config always
    selects the same placebo panel. Raises if the pool is smaller than
    `panel_size` -- the caller must have already checked this via
    iter_ablation_combos' skip logic; this function fails loudly rather
    than silently sampling with replacement or truncating.
    """
    if len(pool) < panel_size:
        msg = f"card pool has {len(pool)} cards, need >= {panel_size} for a placebo panel"
        raise ValueError(msg)
    rng = random.Random(seed)
    return tuple(rng.sample(list(pool), panel_size))


def dry_run_report(cfg: BenchmarkConfig, n_days: int, card_pool_size: int) -> dict:
    """Everything --dry-run needs to report, with zero forecaster calls: the
    shared origin set's size (0 if n_days <= 0 -- e.g. no cached data yet),
    every combo that would run, how many were skipped (and why), and a rough
    predict_batch call-count estimate (one call per univariate series-origin
    pair, one call per multivariate origin, per combo).
    """
    if n_days > 0:
        origins = common_origin_set(
            n=n_days,
            context_lengths=cfg.context_lengths,
            horizons=cfg.horizons,
            origin_stride=cfg.origin_stride,
            max_origins=cfg.max_origins,
        )
    else:
        origins = np.array([], dtype=int)

    combos = iter_ablation_combos(cfg, card_pool_size)
    full_count = (
        len(cfg.context_lengths)
        * len(cfg.horizons)
        * len(cfg.transforms)
        * len(cfg.make_positive)
        * len(cfg.modes)
    )
    n_origins = len(origins)
    estimated_calls = sum(
        n_origins * card_pool_size if c.mode == "univariate" else n_origins for c in combos
    )

    return {
        "config_id": cfg.config_id,
        "n_days": n_days,
        "n_origins": n_origins,
        "n_combos": len(combos),
        "n_combos_skipped_placebo_pool_too_small": full_count - len(combos),
        "card_pool_size": card_pool_size,
        "estimated_predict_batch_calls": estimated_calls,
        "combos": [
            {
                "context_len": c.context_len,
                "horizon": c.horizon,
                "transform": c.transform,
                "make_positive": c.make_positive,
                "mode": c.mode,
            }
            for c in combos
        ],
    }
