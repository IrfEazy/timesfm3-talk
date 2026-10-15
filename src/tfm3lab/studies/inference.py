"""Preregistered inference primitives of the v3 studies.

All functions work on unit-level numbers (never prices) and raise ValueError
on non-finite input. Conventions: B = 2000 bootstrap replicates, seed
20261010, confirmatory two-sided level 1 - 0.05/3 (98.33%), exploratory 95%.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import stats

from .. import bootstrap

ALPHA_CONFIRMATORY = 0.05 / 3
CI_CONFIRMATORY = 1.0 - ALPHA_CONFIRMATORY
CI_EXPLORATORY = 0.95
B = 2000
SEED = 20261010
EXACT_MAX_G = 20
MC_DRAWS = 20000
S1_MIN_EVENTS = 10

Interval = tuple[float, float]


def _finite_1d(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def t_interval(diffs, level: float) -> tuple[float, float, float]:
    """(mean, low, high): Student-t interval of the mean of unit-level paired
    differences, df = G - 1. Needs G >= 2."""
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    d = _finite_1d(diffs, "diffs")
    g = d.size
    if g < 2:
        raise ValueError(f"t_interval needs at least 2 units, got {g}")
    mean = float(d.mean())
    se = float(d.std(ddof=1)) / np.sqrt(g)
    crit = float(stats.t.ppf(0.5 + level / 2.0, df=g - 1))
    return mean, mean - crit * se, mean + crit * se


def sign_flip_test(diffs, seed: int = SEED) -> float:
    """Two-sided p-value of mean(diffs) under random sign flips of the units.
    Exact enumeration of all 2^G sign patterns if G <= 20; otherwise 20000
    Monte Carlo draws (plus-one corrected) from default_rng(seed)."""
    d = _finite_1d(diffs, "diffs")
    g = d.size
    if g < 1:
        raise ValueError("sign_flip_test needs at least one unit")
    obs = abs(float(d.sum()))
    tol = 1e-12 * max(1.0, float(np.abs(d).sum()))
    if g <= EXACT_MAX_G:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=g)))
        sums = np.abs(signs @ d)
        return float(np.mean(sums >= obs - tol))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(MC_DRAWS, g))
    sums = np.abs(signs @ d)
    return float((1 + np.count_nonzero(sums >= obs - tol)) / (MC_DRAWS + 1))


def cluster_bootstrap_mean(
    values, cluster_ids, level: float = CI_EXPLORATORY, n_boot: int = B, seed: int = SEED
) -> Interval:
    """Percentile CI of the overall mean of unit values when whole clusters are
    resampled with replacement (row-weighted mean over the drawn clusters;
    duplicates counted)."""
    v = _finite_1d(values, "values")
    if v.size == 0:
        raise ValueError("values is empty")
    uniques, codes = bootstrap._encode_clusters(cluster_ids, v.size)
    n_clusters = uniques.size
    sums = np.bincount(codes, weights=v, minlength=n_clusters)
    counts = np.bincount(codes, minlength=n_clusters).astype(float)
    mult = bootstrap.cluster_multiplicities(n_clusters, n_boot, np.random.default_rng(seed))
    reps = (mult @ sums) / (mult @ counts)
    low, high, _ = bootstrap.percentile_interval(reps, level)
    return low, high


def calendar_block_bootstrap_mean(
    values,
    dates,
    block_days: int,
    level: float = CI_EXPLORATORY,
    n_boot: int = B,
    seed: int = SEED,
) -> Interval:
    """Percentile CI of the mean of unit values under a circular calendar-block
    bootstrap over the daily calendar spanned by the unit origin dates; a unit
    is weighted by the multiplicity of its origin day."""
    v = _finite_1d(values, "values")
    if v.size == 0:
        raise ValueError("values is empty")
    offsets, n_days, _ = bootstrap._encode_days(dates, v.size)
    mult = bootstrap.calendar_block_multiplicities(
        n_days, int(block_days), n_boot, np.random.default_rng(seed)
    )
    w = mult[:, offsets].astype(float)
    wsum = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = np.where(wsum > 0, (w @ v) / wsum, np.nan)
    low, high, _ = bootstrap.percentile_interval(reps, level)
    return low, high


def verdict_two_procedures(ci_a: Interval, ci_b: Interval, lower_is_better: bool) -> str:
    """'beats' only if both intervals lie entirely on the favourable side of 0,
    'loses' if both lie entirely on the unfavourable side, else 'inconclusive'."""
    for ci in (ci_a, ci_b):
        if len(ci) != 2 or not all(np.isfinite(ci)):
            raise ValueError(f"interval must be two finite numbers, got {ci!r}")
        if ci[0] > ci[1]:
            raise ValueError(f"interval lower bound exceeds upper bound: {ci!r}")
    below = all(ci[1] < 0 for ci in (ci_a, ci_b))
    above = all(ci[0] > 0 for ci in (ci_a, ci_b))
    beats, loses = (below, above) if lower_is_better else (above, below)
    return "beats" if beats else "loses" if loses else "inconclusive"


def s1_verdict(event_diffs, seed: int = SEED) -> str:
    """S1 verdict on event-level (TimesFM - naive) loss differences."""
    d = _finite_1d(event_diffs, "event_diffs")
    if d.size < S1_MIN_EVENTS:
        return "descriptive only"
    mean, low, high = t_interval(d, CI_CONFIRMATORY)
    p = sign_flip_test(d, seed)
    if mean < 0 and high < 0 and p < ALPHA_CONFIRMATORY:
        return "beats naive"
    if mean > 0 and low > 0 and p < ALPHA_CONFIRMATORY:
        return "loses to naive"
    return "inconclusive"
