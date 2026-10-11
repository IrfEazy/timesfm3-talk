"""Grid orchestration for the declarative MTG benchmark: one shared origin
set across the whole context/horizon grid, ablation-combo enumeration, and
placebo-panel card sampling. Origin *index* math itself stays entirely in
windows.py (untouched) -- this module only decides which combos to run and
reuses windows.valid_origins for the actual computation.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa

from . import config
from .backtest import SeriesData
from .benchmark_config import BenchmarkConfig
from .windows import context_slice, valid_origins

# Bump when the per-cell row layout (CELL_SCHEMA) or the metric definitions
# applied to it change: both enter the execution fingerprint, so a bump
# forces a fresh run directory instead of mixing incompatible cells.
CELL_SCHEMA_VERSION = 1
METRIC_VERSION = 1

BASELINE_NAMES = ("naive", "seasonal_naive", "drift", "ets")
QUANTILE_COLUMNS = tuple(f"q{round(level * 100):02d}" for level in config.QUANTILE_LEVELS)
CELL_KEY_COLUMNS = ("series", "origin", "horizon_step")

# Rough uncompressed bytes per stored row (16 float64 + 6 int64/timestamp +
# 2 bool + short strings); used only for --dry-run projections.
BYTES_PER_ROW_ESTIMATE = 220

# Experiment A's primary cell (Preregistration v2 s2), minus the horizon:
# every grid horizon of this (mode, transform, make_positive, context) slice
# goes into the small committed export.
PRIMARY_CELL = {"mode": "univariate", "transform": "raw", "make_positive": True, "context_len": 512}


def refill_series(series: SeriesData, ffill_days: int) -> tuple[SeriesData, np.ndarray]:
    """Forward-fill sensitivity (prereg s2): non-observed points are set to
    NaN, then re-filled forward from the last observed value with at most
    `ffill_days` consecutive fills. Returns (series, gap_mask): `gap_mask` is
    True where a NaN remains (a gap longer than the limit, or a leading gap).

    Neither the model (forecast_batch) nor the baselines accept NaN, and
    SeriesData forbids it, so the returned series keeps the cached value at
    gap positions as an inert placeholder: `keep_origins_per_card`
    drops, per card, every origin whose context window touches the mask, so a
    placeholder never reaches a model or baseline. Targets keep their
    `observed` flags, so metrics still use observed points only."""
    vals = np.asarray(series.values, dtype=float).copy()
    vals[~np.asarray(series.observed, dtype=bool)] = np.nan
    # pandas rejects limit=0; zero fills means the NaNs simply stay
    refilled = np.array(pd.Series(vals).ffill(limit=ffill_days)) if ffill_days > 0 else vals.copy()
    gap = np.isnan(refilled)
    refilled[gap] = np.asarray(series.values, dtype=float)[gap]
    return SeriesData(series.name, refilled, series.dates, series.observed), gap


def keep_origins_per_card(
    origins: np.ndarray, gap_masks: Mapping[str, np.ndarray], context_len: int
) -> dict[str, np.ndarray]:
    """Per-card gap rule: card c keeps origin o iff c's widest context window
    [o-context_len, o) holds no remaining gap in c's OWN series. Cards never
    drop each other's origins. Returns {card: kept origins} (possibly empty).
    Widest window => superset of every smaller context's window."""
    out: dict[str, np.ndarray] = {}
    for name, mask in gap_masks.items():
        keep = np.array(
            [not mask[max(int(o) - context_len, 0) : int(o)].any() for o in origins], dtype=bool
        )
        out[name] = origins[keep]
    return out


def gap_drop_summary(origins: np.ndarray, kept: Mapping[str, np.ndarray]) -> dict:
    """Manifest/fingerprint summary of a per-card gap drop: per-card dropped
    counts, per-card kept fraction, overall kept fraction, and the cards left
    with zero origins (excluded from the run's summaries and export)."""
    n = len(origins)
    dropped = {c: int(n - len(k)) for c, k in kept.items()}
    frac = {c: (len(k) / n if n else 0.0) for c, k in kept.items()}
    total = n * len(kept)
    return {
        "n_candidate_origins": int(n),
        "dropped_per_card": dropped,
        "kept_fraction_per_card": frac,
        "kept_fraction_overall": (sum(len(k) for k in kept.values()) / total if total else 0.0),
        "excluded_cards": sorted(c for c, k in kept.items() if len(k) == 0),
    }


def filter_kept_rows(df: pd.DataFrame, kept: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Keeps only rows whose (series, origin) pair is in the per-card kept
    set; a card absent from `kept` or with no kept origins loses all rows."""
    keep_pairs = {(c, int(o)) for c, k in kept.items() for o in k}
    idx = pd.MultiIndex.from_arrays([df["series"].astype(str), df["origin"].astype("int64")])
    return df[idx.isin(list(keep_pairs))]


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

    @property
    def cell_id(self) -> str:
        """Stable id derived from the combo fields only (never from
        enumeration order); zero-padded so lexical order is numeric order."""
        return (
            f"ctx{self.context_len:05d}-h{self.horizon:04d}-{self.transform}"
            f"-mp{int(self.make_positive)}-{self.mode}"
        )

    @property
    def is_primary(self) -> bool:
        return (
            self.mode == PRIMARY_CELL["mode"]
            and self.transform == PRIMARY_CELL["transform"]
            and self.make_positive == PRIMARY_CELL["make_positive"]
            and self.context_len == PRIMARY_CELL["context_len"]
        )


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


def primary_first_order(combos: list[AblationCombo], cfg: BenchmarkConfig) -> list[AblationCombo]:
    """Stable reorder that puts the primary slice (raw, make_positive, univariate,
    largest context -- 512 if in the grid) first, primary horizons before the
    other horizons. Execution order only: cell ids, cell content and summaries
    (sorted by cell id) do not depend on it."""
    ctx = 512 if 512 in cfg.context_lengths else max(cfg.context_lengths)
    primary_h = set(cfg.primary_horizons)

    def key(c: AblationCombo) -> tuple[int, int]:
        in_slice = (
            c.mode == PRIMARY_CELL["mode"]
            and c.transform == PRIMARY_CELL["transform"]
            and c.make_positive == PRIMARY_CELL["make_positive"]
            and c.context_len == ctx
        )
        return (0 if in_slice else 1, 0 if c.horizon in primary_h else 1)

    return sorted(combos, key=key)


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
    cells = []
    for c in combos:
        rows = projected_cell_rows(c, n_origins, card_pool_size, cfg.placebo_panel_size)
        cells.append(
            {
                "cell_id": c.cell_id,
                "projected_rows": rows,
                "projected_bytes_estimate": rows * BYTES_PER_ROW_ESTIMATE,
            }
        )

    return {
        "config_id": cfg.config_id,
        "n_days": n_days,
        "n_origins": n_origins,
        "n_combos": len(combos),
        "n_combos_skipped_placebo_pool_too_small": full_count - len(combos),
        "card_pool_size": card_pool_size,
        "estimated_predict_batch_calls": estimated_calls,
        "bytes_per_row_estimate": BYTES_PER_ROW_ESTIMATE,
        "projected_rows_total": sum(c["projected_rows"] for c in cells),
        "projected_bytes_total_estimate": sum(c["projected_bytes_estimate"] for c in cells),
        "cells": cells,
        "combos": [
            {
                "cell_id": c.cell_id,
                "context_len": c.context_len,
                "horizon": c.horizon,
                "transform": c.transform,
                "make_positive": c.make_positive,
                "mode": c.mode,
            }
            for c in combos
        ],
    }


def projected_cell_rows(
    combo: AblationCombo, n_origins: int, card_pool_size: int, placebo_panel_size: int
) -> int:
    """Rows one cell stores: one per (series, origin, horizon step)."""
    n_series = placebo_panel_size if combo.mode == "multivariate_placebo" else card_pool_size
    return n_series * n_origins * combo.horizon


def cell_arrow_schema() -> pa.Schema:
    """The fixed per-cell row layout (CELL_SCHEMA_VERSION). All four baseline
    columns are always present (NaN where a baseline was not computed), so
    every cell of every run shares one schema."""
    fields = [
        ("cell_id", pa.string()),
        ("run_id", pa.string()),
        ("config_id", pa.string()),
        ("mode", pa.string()),
        ("transform", pa.string()),
        ("make_positive", pa.bool_()),
        ("context_len", pa.int64()),
        ("requested_horizon", pa.int64()),
        ("series", pa.string()),
        ("origin", pa.int64()),
        ("origin_date", pa.timestamp("ns")),
        ("target_index", pa.int64()),
        ("target_date", pa.timestamp("ns")),
        ("horizon_step", pa.int64()),
        ("actual", pa.float64()),
        ("observed", pa.bool_()),
        ("forecast", pa.float64()),
        *[(q, pa.float64()) for q in QUANTILE_COLUMNS],
        *[(f"baseline_{b}", pa.float64()) for b in BASELINE_NAMES],
        # last OBSERVED value of the model's context window (price units);
        # the normalizer of the committed primary export.
        ("context_last_observed", pa.float64()),
    ]
    return pa.schema(fields)


CELL_SCHEMA = cell_arrow_schema()


def last_observed_context_value(
    values: np.ndarray, observed: np.ndarray, origin: int, context_len: int
) -> float:
    """Value of the last observed point inside the context window
    values[origin-context_len:origin]; NaN if none is observed or the value
    is not a positive finite number (it is used as a divisor)."""
    window = context_slice(origin, context_len)
    idx = np.flatnonzero(np.asarray(observed[window], dtype=bool))
    if len(idx) == 0:
        return float("nan")
    value = float(np.asarray(values[window], dtype=float)[idx[-1]])
    return value if np.isfinite(value) and value > 0 else float("nan")


def _to_naive_ns(col: pd.Series) -> pd.Series:
    out = pd.to_datetime(col)
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert(None)
    return out.astype("datetime64[ns]")


def canonicalize_cell_frame(
    df: pd.DataFrame,
    *,
    combo: AblationCombo,
    run_id: str,
    config_id: str,
    series_list: list,
) -> pa.Table:
    """Backtest output -> one cell's canonical Arrow table: fixed columns
    and dtypes (CELL_SCHEMA), the normalizer column, rows sorted by
    CELL_KEY_COLUMNS. `series_list` are the SeriesData this cell forecast."""
    out = df.rename(columns={"origin_index": "origin"}).copy()
    out["cell_id"] = combo.cell_id
    out["run_id"] = run_id
    out["config_id"] = config_id
    out["context_len"] = combo.context_len
    out["requested_horizon"] = combo.horizon
    for b in BASELINE_NAMES:
        if f"baseline_{b}" not in out.columns:
            out[f"baseline_{b}"] = np.nan
    origins = sorted({int(o) for o in out["origin"].unique()})
    scale_rows = [
        (s.name, o, last_observed_context_value(s.values, s.observed, o, combo.context_len))
        for s in series_list
        for o in origins
    ]
    scales = pd.DataFrame(scale_rows, columns=["series", "origin", "context_last_observed"])
    out["series"] = out["series"].astype(str)
    out["origin"] = out["origin"].astype("int64")
    scales["series"] = scales["series"].astype(str)
    scales["origin"] = scales["origin"].astype("int64")
    out = out.merge(scales, on=["series", "origin"], how="left", validate="many_to_one")
    for c in ("origin_date", "target_date"):
        out[c] = _to_naive_ns(out[c])
    out = out.sort_values(list(CELL_KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
    names = CELL_SCHEMA.names
    return pa.Table.from_pandas(out[names], schema=CELL_SCHEMA, preserve_index=False)


PRIMARY_EXPORT_FORBIDDEN = frozenset(
    {"actual", "forecast", "context_last_observed", *QUANTILE_COLUMNS}
    | {f"baseline_{b}" for b in BASELINE_NAMES}
)


def primary_export_frame(
    cell_df: pd.DataFrame,
    baselines: tuple[str, ...],
    error_scales: Mapping[str, float],
) -> pd.DataFrame:
    """Normalized, price-free view of one primary cell for the committed
    export (Preregistration v2 s7, data option b). Absolute errors
    (`ae_*_scaled`) are divided by a per-series CONSTANT from
    `error_scales` (normalize.series_error_scale), so per-card ratios of
    loss sums equal the raw-error ratios of Preregistration v2 s1;
    y/q10/q50/q90 are divided by the last observed context value (per
    origin) for calibration. The constants are never written out. No
    column holds an actual price.

    Only the row at horizon_step == requested_horizon is kept (the horizon
    each cell was run for); intermediate steps stay in the heavy run
    directory. Rows whose scale is NaN, zero or negative get
    `scale_valid=False` and NaN normalized values instead of inf."""
    cell_df = cell_df[cell_df["horizon_step"] == cell_df["requested_horizon"]]
    scale_raw = cell_df["context_last_observed"].to_numpy(dtype=float)
    valid = np.isfinite(scale_raw) & (scale_raw > 0)
    scale = np.where(valid, scale_raw, np.nan)
    err_raw = cell_df["series"].astype(str).map(error_scales).to_numpy(dtype=float)
    err_valid = np.isfinite(err_raw) & (err_raw > 0)
    err_scale = np.where(err_valid, err_raw, np.nan)
    y = cell_df["actual"].to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "config_id": cell_df["config_id"].astype(str).to_numpy(),
            "run_id": cell_df["run_id"].astype(str).to_numpy(),
            "mode": cell_df["mode"].astype(str).to_numpy(),
            "context_len": cell_df["context_len"].to_numpy(),
            "card": cell_df["series"].astype(str).to_numpy(),
            "origin_date": cell_df["origin_date"].to_numpy(),
            "target_date": cell_df["target_date"].to_numpy(),
            "requested_horizon": cell_df["requested_horizon"].to_numpy(),
            "horizon": cell_df["horizon_step"].to_numpy(),
            "observed": cell_df["observed"].to_numpy(dtype=bool),
            "scale_valid": valid,
            "error_scale_valid": err_valid,
            "ae_model_scaled": np.abs(y - cell_df["forecast"].to_numpy(dtype=float)) / err_scale,
        }
    )
    for b in baselines:
        pred = cell_df[f"baseline_{b}"].to_numpy(dtype=float)
        out[f"ae_baseline_{b}_scaled"] = np.abs(y - pred) / err_scale
    out["y_norm"] = y / scale
    for q in ("q10", "q50", "q90"):
        out[f"{q}_norm"] = cell_df[q].to_numpy(dtype=float) / scale
    leaked = PRIMARY_EXPORT_FORBIDDEN & set(out.columns)
    if leaked:  # defensive: never ship a price column
        raise AssertionError(f"primary export would contain price columns {sorted(leaked)}")
    return out
