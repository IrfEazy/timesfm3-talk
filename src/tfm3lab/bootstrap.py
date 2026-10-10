"""Resampling inference for aggregate skill and interval coverage
(Preregistration v2, sections 1 and 4).

Three designs, each answering a different question:

- PRIMARY, `paired_cluster_bootstrap`: whole-card cluster bootstrap. Cards
  are resampled uniformly with replacement; each drawn card keeps its full
  history; model and baseline losses share the same draws (pairing). The
  statistic is recomputed from per-card loss sums with a (B x G) bootstrap
  multiplicity matrix, so a card drawn twice counts twice (duplicates are
  never collapsed) and no observation weighting is applied on top of the
  uniform cluster sampling. Target: performance across a population of
  cards.
- SENSITIVITY, `calendar_block_bootstrap`: synchronized circular
  calendar-block bootstrap. One set of calendar blocks (block length in
  DAYS = max(h, 7)) is drawn per replicate and applied to EVERY card, so
  common shocks stay aligned; per-card sums are recomputed from the
  resampled days. Label: "time uncertainty, conditional on this panel".
- DIAGNOSTIC ONLY, `two_stage_bootstrap`: cards, then independent
  within-card calendar blocks. It double-counts within-card noise and must
  never be the headline interval.

Skill definitions (per card i, over observed rows only):
    r_i = sum|e_model| / sum|e_baseline|   (relative MAE; s_i = 1 - r_i)
    median skill          = 1 - median_i(r_i)
    n-weighted mean skill = 1 - sum_i n_i r_i / sum_i n_i
                            (exactly summarize.aggregate_leaderboard)
    pooled skill          = 1 - sum_it|e_model| / sum_it|e_baseline|
                            (SECONDARY, never a decision input)
Zero-baseline policy matches summarize.summarize_accuracy: r_i = 0 if both
sums are 0, +inf if only the baseline sum is 0.

Intervals are percentile intervals (numpy.quantile, linear interpolation)
over the VALID replicates; replicates whose statistic is not finite are
counted in `n_invalid` and reported, never silently dropped. Defaults:
B = 2000, seed 20261010, numpy default_rng (PCG64). Everything operates on
plain arrays; cluster ids are canonicalized by sorting so row order never
changes a result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 20261010
DEFAULT_CI = 0.95
MIN_BLOCK_DAYS = 7
SKILL_STATISTICS = ("median", "weighted_mean", "pooled")


@dataclass(frozen=True)
class BootstrapCI:
    """One statistic's point estimate and percentile interval."""

    statistic: str
    method: str
    estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_boot: int
    seed: int | None
    n_invalid: int
    n_clusters: int
    block_days: int | None = None
    # Card-replicates left out because the card had zero resampled rows
    # (calendar-block and two-stage designs only; 0 for cluster designs).
    n_excluded_card_reps: int = 0
    replicates: np.ndarray = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        out = asdict(self)
        out.pop("replicates")
        return out


def block_days_for_horizon(horizon: int) -> int:
    """Calendar block length in days: max(h, 7) (Preregistration v2)."""
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return max(horizon, MIN_BLOCK_DAYS)


def _rng(seed: int | None, rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng(seed)


def _check_level(ci: float) -> None:
    if not 0 < ci < 1:
        raise ValueError(f"ci must be in (0, 1), got {ci}")


def _as_float(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values; filter rows before bootstrapping")
    return arr


def _encode_clusters(cluster_ids, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(cluster_ids)
    if ids.shape != (n_rows,):
        raise ValueError(f"cluster_ids must have shape ({n_rows},), got {ids.shape}")
    uniques, codes = np.unique(ids, return_inverse=True)
    return uniques, codes


def _encode_days(dates, n_rows: int) -> tuple[np.ndarray, int, np.datetime64]:
    days = pd.to_datetime(np.asarray(dates)).to_numpy(dtype="datetime64[D]")
    if days.shape != (n_rows,):
        raise ValueError(f"dates must have shape ({n_rows},), got {days.shape}")
    if np.isnat(days).any():
        raise ValueError("dates contain NaT")
    if n_rows == 0:
        return np.zeros(0, dtype=int), 0, np.datetime64("NaT")
    start = days.min()
    offsets = (days - start).astype(int)
    return offsets, int(offsets.max()) + 1, start


def percentile_interval(replicates, ci: float = DEFAULT_CI) -> tuple[float, float, int]:
    """(low, high, n_invalid) over the finite replicates; NaN bounds if none."""
    _check_level(ci)
    reps = np.asarray(replicates, dtype=float)
    valid = np.isfinite(reps)
    n_invalid = int(reps.size - valid.sum())
    if not valid.any():
        return float("nan"), float("nan"), n_invalid
    alpha = 1.0 - ci
    low, high = np.quantile(reps[valid], [alpha / 2, 1 - alpha / 2])
    return float(low), float(high), n_invalid


# --- resampling weights -----------------------------------------------------


def cluster_multiplicities(
    n_clusters: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_boot x n_clusters) integer matrix: how many times each cluster is
    drawn in each replicate (uniform draws with replacement, row sums = G)."""
    if n_clusters < 1:
        raise ValueError("need at least one cluster")
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    flat = draws + n_clusters * np.arange(n_boot)[:, None]
    return np.bincount(flat.ravel(), minlength=n_boot * n_clusters).reshape(
        n_boot, n_clusters
    )


def calendar_block_multiplicities(
    n_days: int, block_days: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_boot x n_days) integer matrix of circular calendar-block weights.

    Each replicate draws ceil(n_days / block_days) block starts uniformly on
    the circular calendar 0..n_days-1, concatenates the blocks and truncates
    to n_days days; the weight of a day is how often it appears. Applying one
    row to every card synchronizes the calendar across cards.
    """
    if n_days < 1:
        raise ValueError("need at least one calendar day")
    if block_days < 1:
        raise ValueError(f"block_days must be >= 1, got {block_days}")
    n_blocks = -(-n_days // block_days)
    starts = rng.integers(0, n_days, size=(n_boot, n_blocks))
    days = (starts[:, :, None] + np.arange(block_days)[None, None, :]) % n_days
    days = days.reshape(n_boot, n_blocks * block_days)[:, :n_days]
    flat = days + n_days * np.arange(n_boot)[:, None]
    return np.bincount(flat.ravel(), minlength=n_boot * n_days).reshape(n_boot, n_days)


# --- statistics from (resampled) per-card sums ------------------------------


def _relative_mae(model_sum: np.ndarray, base_sum: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = model_sum / base_sum
    zero_base = base_sum <= 0
    ratio = np.where(zero_base & (model_sum <= 0), 0.0, ratio)
    ratio = np.where(zero_base & (model_sum > 0), np.inf, ratio)
    return ratio


def weighted_median(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Row-wise median of `values` where entry j is repeated counts[j] times
    (integer counts) — identical to np.median of the expanded array, i.e. the
    mean of the two middle order statistics for an even total. Rows with
    total count 0 give NaN."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    counts = np.atleast_2d(np.asarray(counts, dtype=np.int64))
    values, counts = np.broadcast_arrays(values, counts)
    order = np.argsort(values, axis=1, kind="mergesort")
    v = np.take_along_axis(values, order, axis=1)
    c = np.take_along_axis(counts, order, axis=1)
    cum = np.cumsum(c, axis=1)
    total = cum[:, -1]
    lo_k = (total - 1) // 2
    hi_k = total // 2
    lo = np.argmax(cum > lo_k[:, None], axis=1)
    hi = np.argmax(cum > hi_k[:, None], axis=1)
    rows = np.arange(v.shape[0])
    with np.errstate(invalid="ignore"):
        med = (v[rows, lo] + v[rows, hi]) / 2.0
    return np.where(total > 0, med, np.nan)


def _skill_from_components(
    ratio: np.ndarray,
    median_counts: np.ndarray,
    weights: np.ndarray,
    model_total: np.ndarray,
    base_total: np.ndarray,
) -> dict[str, np.ndarray]:
    """Rows = replicates. `median_counts` are integer repetition counts for
    the median, `weights` the n-weights of the weighted mean."""
    median_r = weighted_median(ratio, median_counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        safe = np.where(weights > 0, ratio, 0.0)
        wsum = weights.sum(axis=1)
        mean_r = np.where(wsum > 0, (weights * safe).sum(axis=1) / wsum, np.nan)
        # +inf ratios with positive weight make the weighted mean +inf
        has_inf = ((weights > 0) & np.isinf(ratio)).any(axis=1)
        mean_r = np.where(has_inf, np.inf, mean_r)
        pooled_r = np.where(base_total > 0, model_total / base_total, np.nan)
    return {
        "median": 1.0 - median_r,
        "weighted_mean": 1.0 - mean_r,
        "pooled": 1.0 - pooled_r,
    }


def skill_from_multiplicities(
    model_sum, base_sum, n_obs, multiplicities
) -> dict[str, np.ndarray]:
    """Median, n-weighted mean and pooled skill for each row of a
    (B x G) cluster multiplicity matrix, from per-card loss sums and observed
    row counts. A multiplicity of k is exactly equivalent to the card
    appearing k times."""
    model_sum = np.asarray(model_sum, dtype=float)
    base_sum = np.asarray(base_sum, dtype=float)
    n_obs = np.asarray(n_obs, dtype=float)
    mult = np.atleast_2d(np.asarray(multiplicities, dtype=np.int64))
    ratio = np.broadcast_to(_relative_mae(model_sum, base_sum), mult.shape)
    return _skill_from_components(
        ratio,
        mult,
        mult * n_obs[None, :],
        mult @ model_sum,
        mult @ base_sum,
    )


def per_cluster_sums(model_abs_err, baseline_abs_err, cluster_ids):
    """(cluster_ids_sorted, model_sum, baseline_sum, n_obs) per cluster."""
    m = _as_float(model_abs_err, "model_abs_err")
    b = _as_float(baseline_abs_err, "baseline_abs_err")
    if m.shape != b.shape:
        raise ValueError(f"shape mismatch: {m.shape} vs {b.shape}")
    uniques, codes = _encode_clusters(cluster_ids, len(m))
    g = len(uniques)
    return (
        uniques,
        np.bincount(codes, weights=m, minlength=g),
        np.bincount(codes, weights=b, minlength=g),
        np.bincount(codes, minlength=g).astype(float),
    )


def _results(
    point: dict[str, np.ndarray],
    reps: dict[str, np.ndarray],
    *,
    method: str,
    ci: float,
    n_boot: int,
    seed: int | None,
    n_clusters: int,
    block_days: int | None,
    statistics: tuple[str, ...],
    n_excluded_card_reps: int = 0,
) -> dict[str, BootstrapCI]:
    out = {}
    for name in statistics:
        if name not in point:
            raise ValueError(f"unknown statistic {name!r}; choose from {SKILL_STATISTICS}")
        low, high, n_invalid = percentile_interval(reps[name], ci)
        out[name] = BootstrapCI(
            statistic=name,
            method=method,
            estimate=float(point[name][0]),
            ci_low=low,
            ci_high=high,
            ci_level=ci,
            n_boot=n_boot,
            seed=seed,
            n_invalid=n_invalid,
            n_clusters=n_clusters,
            block_days=block_days,
            n_excluded_card_reps=n_excluded_card_reps,
            replicates=reps[name],
        )
    return out


def paired_cluster_bootstrap_from_sums(
    model_sum,
    base_sum,
    n_obs,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int | None = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
    rng: np.random.Generator | None = None,
    statistics: tuple[str, ...] = SKILL_STATISTICS,
) -> dict[str, BootstrapCI]:
    """PRIMARY whole-card cluster bootstrap from preaggregated per-card sums
    (arrays of length G in a fixed canonical order). See module docstring."""
    _check_level(ci)
    model_sum = np.asarray(model_sum, dtype=float)
    base_sum = np.asarray(base_sum, dtype=float)
    n_obs = np.asarray(n_obs, dtype=float)
    g = len(model_sum)
    generator = _rng(seed, rng)
    mult = cluster_multiplicities(g, n_boot, generator)
    point = skill_from_multiplicities(model_sum, base_sum, n_obs, np.ones((1, g), dtype=int))
    reps = skill_from_multiplicities(model_sum, base_sum, n_obs, mult)
    return _results(
        point,
        reps,
        method="cluster",
        ci=ci,
        n_boot=n_boot,
        seed=None if rng is not None else seed,
        n_clusters=g,
        block_days=None,
        statistics=statistics,
    )


def paired_cluster_bootstrap(
    model_abs_err,
    baseline_abs_err,
    cluster_ids,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int | None = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
    rng: np.random.Generator | None = None,
    statistics: tuple[str, ...] = SKILL_STATISTICS,
) -> dict[str, BootstrapCI]:
    """PRIMARY paired whole-card cluster bootstrap of aggregate skill.

    Row-level inputs for ONE (comparison, horizon) cell: absolute model and
    baseline errors on observed rows, and the card id of each row. Returns
    {statistic: BootstrapCI} for median, n-weighted mean and pooled skill.
    """
    _, m_sum, b_sum, n = per_cluster_sums(model_abs_err, baseline_abs_err, cluster_ids)
    return paired_cluster_bootstrap_from_sums(
        m_sum, b_sum, n, n_boot=n_boot, seed=seed, ci=ci, rng=rng, statistics=statistics
    )


# --- calendar-indexed designs -----------------------------------------------


@dataclass(frozen=True)
class CalendarPanel:
    """Per-card x per-day loss sums on a common daily calendar."""

    cluster_ids: np.ndarray
    start: np.datetime64
    model: np.ndarray  # (G, T)
    base: np.ndarray  # (G, T)
    count: np.ndarray  # (G, T)


def calendar_panel(model_abs_err, baseline_abs_err, cluster_ids, dates) -> CalendarPanel:
    """Aggregates rows onto a (cards x days) grid spanning min..max date.
    Days without observations for a card hold zeros, so gaps travel with the
    calendar instead of being concatenated away."""
    m = _as_float(model_abs_err, "model_abs_err")
    b = _as_float(baseline_abs_err, "baseline_abs_err")
    if m.shape != b.shape:
        raise ValueError(f"shape mismatch: {m.shape} vs {b.shape}")
    if len(m) == 0:
        raise ValueError("no rows")
    uniques, codes = _encode_clusters(cluster_ids, len(m))
    offsets, n_days, start = _encode_days(dates, len(m))
    g = len(uniques)
    flat = codes * n_days + offsets
    size = g * n_days
    return CalendarPanel(
        cluster_ids=uniques,
        start=start,
        model=np.bincount(flat, weights=m, minlength=size).reshape(g, n_days),
        base=np.bincount(flat, weights=b, minlength=size).reshape(g, n_days),
        count=np.bincount(flat, minlength=size).reshape(g, n_days).astype(float),
    )


def calendar_block_resample_sums(
    panel: CalendarPanel, day_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Applies the SAME (B x T) day weights to every card: returns resampled
    per-card (model_sum, base_sum, n_obs), each (B x G)."""
    w = np.atleast_2d(np.asarray(day_weights, dtype=float))
    return w @ panel.model.T, w @ panel.base.T, w @ panel.count.T


def _skill_from_resampled(m: np.ndarray, b: np.ndarray, n: np.ndarray) -> dict[str, np.ndarray]:
    # A card with no resampled observations has no relative MAE in that
    # replicate: it is left out of the median (count 0) and the weighted
    # mean (weight 0), exactly as summarize would produce no row for it.
    present = n > 0
    ratio = np.where(present, _relative_mae(m, b), np.nan)
    return _skill_from_components(
        ratio, present.astype(np.int64), n, m.sum(axis=1), b.sum(axis=1)
    )


def calendar_block_bootstrap(
    model_abs_err,
    baseline_abs_err,
    cluster_ids,
    dates,
    horizon: int,
    *,
    block_days: int | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int | None = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
    rng: np.random.Generator | None = None,
    statistics: tuple[str, ...] = SKILL_STATISTICS,
) -> dict[str, BootstrapCI]:
    """SENSITIVITY synchronized circular calendar-block bootstrap of
    aggregate skill — "time uncertainty, conditional on this panel".

    `dates` is the calendar date that indexes each row (normally the
    forecast origin date); block length in days defaults to max(h, 7). The
    same blocks are applied to every card and to both model and baseline
    losses; the statistic is recomputed from the resampled per-card sums.
    """
    _check_level(ci)
    panel = calendar_panel(model_abs_err, baseline_abs_err, cluster_ids, dates)
    blen = block_days_for_horizon(horizon) if block_days is None else int(block_days)
    generator = _rng(seed, rng)
    weights = calendar_block_multiplicities(panel.model.shape[1], blen, n_boot, generator)
    point = _skill_from_resampled(
        panel.model.sum(axis=1)[None, :],
        panel.base.sum(axis=1)[None, :],
        panel.count.sum(axis=1)[None, :],
    )
    resampled = calendar_block_resample_sums(panel, weights)
    reps = _skill_from_resampled(*resampled)
    return _results(
        point,
        reps,
        method="calendar_block",
        ci=ci,
        n_boot=n_boot,
        seed=None if rng is not None else seed,
        n_clusters=len(panel.cluster_ids),
        block_days=blen,
        n_excluded_card_reps=int((resampled[2] <= 0).sum()),
        statistics=statistics,
    )


def two_stage_bootstrap(
    model_abs_err,
    baseline_abs_err,
    cluster_ids,
    dates,
    horizon: int,
    *,
    block_days: int | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int | None = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
    rng: np.random.Generator | None = None,
    statistics: tuple[str, ...] = SKILL_STATISTICS,
) -> dict[str, BootstrapCI]:
    """DIAGNOSTIC ONLY — never a headline interval.

    Stage 1 resamples cards uniformly with replacement; stage 2 draws an
    independent circular calendar-block resample within EACH drawn card.
    This adds within-card noise on top of a between-card resample that
    already contains it, so it tends to be too wide (Preregistration v2
    section 1). Kept to show the size of that effect.
    """
    _check_level(ci)
    panel = calendar_panel(model_abs_err, baseline_abs_err, cluster_ids, dates)
    g, n_days = panel.model.shape
    blen = block_days_for_horizon(horizon) if block_days is None else int(block_days)
    generator = _rng(seed, rng)
    m_rep = np.empty((n_boot, g))
    b_rep = np.empty((n_boot, g))
    n_rep = np.empty((n_boot, g))
    for i in range(n_boot):
        drawn = generator.integers(0, g, size=g)
        w = calendar_block_multiplicities(n_days, blen, g, generator).astype(float)
        m_rep[i] = np.einsum("gt,gt->g", w, panel.model[drawn])
        b_rep[i] = np.einsum("gt,gt->g", w, panel.base[drawn])
        n_rep[i] = np.einsum("gt,gt->g", w, panel.count[drawn])
    point = _skill_from_resampled(
        panel.model.sum(axis=1)[None, :],
        panel.base.sum(axis=1)[None, :],
        panel.count.sum(axis=1)[None, :],
    )
    reps = _skill_from_resampled(m_rep, b_rep, n_rep)
    return _results(
        point,
        reps,
        method="two_stage_diagnostic",
        ci=ci,
        n_boot=n_boot,
        seed=None if rng is not None else seed,
        n_clusters=g,
        block_days=blen,
        n_excluded_card_reps=int((n_rep <= 0).sum()),
        statistics=statistics,
    )


# --- coverage intervals -----------------------------------------------------


def _coverage_empty(method: str, ci: float, n_boot: int, seed, block_days=None) -> BootstrapCI:
    # Policy for an empty group (no evaluated rows): no estimate, every
    # replicate counted invalid; never an error that hides the group.
    return BootstrapCI(
        statistic="coverage",
        method=method,
        estimate=float("nan"),
        ci_low=float("nan"),
        ci_high=float("nan"),
        ci_level=ci,
        n_boot=n_boot,
        seed=seed,
        n_invalid=n_boot,
        n_clusters=0,
        block_days=block_days,
        replicates=np.full(n_boot, np.nan),
    )


def _as_covered(covered) -> np.ndarray:
    arr = np.asarray(covered)
    if arr.ndim != 1:
        raise ValueError(f"covered must be 1-D, got shape {arr.shape}")
    if arr.dtype != bool:
        if not np.all(np.isin(arr, (0, 1))):
            raise ValueError("covered must be boolean (or 0/1)")
        arr = arr.astype(bool)
    return arr.astype(float)


def coverage_calendar_block_ci(
    covered,
    cluster_ids,
    dates,
    horizon: int,
    *,
    block_days: int | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int | None = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """PRIMARY coverage interval: synchronized circular calendar-block
    bootstrap of pooled coverage sum(covered) / n (fixed-panel, future-time
    estimand, conditional on the fitted intervals). Block length in days
    defaults to max(h, 7). Empty input -> NaN estimate with n_invalid=n_boot.
    All-covered (or none-covered) input -> every replicate equals the point
    estimate, i.e. a degenerate [1, 1] (or [0, 0]) interval, reported as is.
    """
    _check_level(ci)
    blen = block_days_for_horizon(horizon) if block_days is None else int(block_days)
    cov = _as_covered(covered)
    recorded_seed = None if rng is not None else seed
    if cov.size == 0:
        return _coverage_empty("calendar_block", ci, n_boot, recorded_seed, blen)
    uniques, _ = _encode_clusters(cluster_ids, len(cov))
    offsets, n_days, _ = _encode_days(dates, len(cov))
    k_day = np.bincount(offsets, weights=cov, minlength=n_days)
    n_day = np.bincount(offsets, minlength=n_days).astype(float)
    w = calendar_block_multiplicities(n_days, blen, n_boot, _rng(seed, rng)).astype(float)
    k_rep, n_rep = w @ k_day, w @ n_day
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = np.where(n_rep > 0, k_rep / n_rep, np.nan)
    low, high, n_invalid = percentile_interval(reps, ci)
    return BootstrapCI(
        statistic="coverage",
        method="calendar_block",
        estimate=float(cov.mean()),
        ci_low=low,
        ci_high=high,
        ci_level=ci,
        n_boot=n_boot,
        seed=recorded_seed,
        n_invalid=n_invalid,
        n_clusters=len(uniques),
        block_days=blen,
        replicates=reps,
    )


def coverage_cluster_ci(
    covered,
    cluster_ids,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int | None = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """SENSITIVITY coverage interval: whole-card (or whole-event) cluster
    bootstrap of pooled coverage sum_i k_i / sum_i n_i with multiplicities.
    Same empty-group and all-covered policies as the calendar-block version.
    """
    _check_level(ci)
    cov = _as_covered(covered)
    recorded_seed = None if rng is not None else seed
    if cov.size == 0:
        return _coverage_empty("cluster", ci, n_boot, recorded_seed)
    uniques, codes = _encode_clusters(cluster_ids, len(cov))
    g = len(uniques)
    k = np.bincount(codes, weights=cov, minlength=g)
    n = np.bincount(codes, minlength=g).astype(float)
    mult = cluster_multiplicities(g, n_boot, _rng(seed, rng)).astype(float)
    reps = (mult @ k) / (mult @ n)  # every cluster has n_i >= 1, so no zero
    low, high, n_invalid = percentile_interval(reps, ci)
    return BootstrapCI(
        statistic="coverage",
        method="cluster",
        estimate=float(cov.mean()),
        ci_low=low,
        ci_high=high,
        ci_level=ci,
        n_boot=n_boot,
        seed=recorded_seed,
        n_invalid=n_invalid,
        n_clusters=g,
        replicates=reps,
    )


@dataclass(frozen=True)
class WilsonDesignResult:
    estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_obs: int
    n_clusters: int
    n_eff: float
    n_eff_adjusted: float
    policy: str

    def to_dict(self) -> dict:
        return asdict(self)


def coverage_wilson_design_adjusted(
    covered, cluster_ids, *, ci: float = DEFAULT_CI
) -> WilsonDesignResult:
    """DIAGNOSTIC design-adjusted Wilson interval for clustered coverage.

    p = sum k_i / N; V = G/(G-1) * sum_i (k_i - p n_i)^2 / N^2;
    n_eff = p(1-p) / V (unrounded); the Wilson interval uses
    n_eff * (z_{1-a/2} / t_{G-1, 1-a/2})^2 as its sample size.

    Explicit policies (`policy` field):
    - "empty": no rows -> NaN everywhere.
    - "single_cluster": G < 2, no between-cluster variance -> NaN interval.
    - "degenerate_iid": V == 0 (e.g. all covered, none covered, or identical
      per-cluster rates) -> n_eff falls back to N, the iid Wilson size,
      still multiplied by the small-cluster factor.
    - "ok": the formula above.
    An approximation, not a finite-sample guarantee.
    """
    _check_level(ci)
    cov = _as_covered(covered)
    nan = float("nan")
    if cov.size == 0:
        return WilsonDesignResult(nan, nan, nan, ci, 0, 0, nan, nan, "empty")
    uniques, codes = _encode_clusters(cluster_ids, len(cov))
    g = len(uniques)
    k = np.bincount(codes, weights=cov, minlength=g)
    n = np.bincount(codes, minlength=g).astype(float)
    total = float(n.sum())
    p = float(k.sum() / total)
    if g < 2:
        return WilsonDesignResult(p, nan, nan, ci, int(total), g, nan, nan, "single_cluster")
    var = g / (g - 1) * float(np.sum((k - p * n) ** 2)) / total**2
    if var <= 0 or p in (0.0, 1.0):
        n_eff, policy = total, "degenerate_iid"
    else:
        n_eff, policy = p * (1 - p) / var, "ok"
    z = float(stats.norm.ppf(1 - (1 - ci) / 2))
    t = float(stats.t.ppf(1 - (1 - ci) / 2, df=g - 1))
    n_adj = n_eff * (z / t) ** 2
    denom = 1 + z**2 / n_adj
    center = (p + z**2 / (2 * n_adj)) / denom
    half = z * np.sqrt(p * (1 - p) / n_adj + z**2 / (4 * n_adj**2)) / denom
    return WilsonDesignResult(
        p,
        float(max(0.0, center - half)),
        float(min(1.0, center + half)),
        ci,
        int(total),
        g,
        float(n_eff),
        float(n_adj),
        policy,
    )

