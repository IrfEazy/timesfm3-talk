# Statistical Inference & Probabilistic Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add paired/panel moving-block bootstrap, Benjamini-Hochberg-corrected
significance, calibration CI/WIS/normalized-width, and causally-valid
conformal post-processing on top of the existing MTG backtest results, all
offline/local-fixture-tested, with new artifacts stamped `run_id` +
manifest.

**Architecture:** Two new standalone modules (`bootstrap.py`, `conformal.py`),
additive functions on three existing modules (`metrics.py`, `summarize.py`,
`figdata.py`), an additive extension of `scripts/04_exp_calibration.py`, three
new plot functions in `plots.py`, and one new CLI script
(`scripts/08_exp_inference.py`) that ties everything together into 5 new
parquet artifacts + 2 new figures + 1 manifest. Every new script/function
reads already-written `results/*.parquet` files — nothing here calls the
model or the network.

**Tech Stack:** Python 3.12, numpy, pandas, scipy.stats, statsmodels
(already a dependency), matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-statistical-inference-calibration-design.md`

## Global Constraints

- `uv` only — never `pip`/`uv pip` (not needed in this plan: no new
  dependencies; statsmodels/scipy/matplotlib are already in `pyproject.toml`).
- Never manually edit a `results/*.parquet` file or a slide number.
- Never invent a result, metric, source, or data point.
- Every test in this plan runs fully offline (no network, no HF download, no
  GPU) — local synthetic fixtures only, matching the rest of `tests/`.
- No leakage: every function that touches "the past" must enforce it as a
  checked boundary (`origin_index` comparison), never trust caller-supplied
  pre-sorted order.
- Every metric/statistic respects `observed=True` — a forward-filled row is
  never scored or calibrated against.
- `windows.py`'s origin convention (origin = first predicted index) is not
  touched by this plan.
- A bootstrap call always has a seed (default `config.SEED`), and always
  echoes it back on the result — no unseeded, unreproducible statistics.
- `block_size >= horizon` is a checked `ValueError`, not a docstring promise.
- Every new artifact this plan's script writes carries a `run_id` column and
  is referenced from one `manifest.write_manifest` call.
- Add a unit test for every bug found while implementing (the moving-block
  bootstrap "widens the CI on autocorrelated data" test in Task 1 and the
  causal-boundary tests in Task 5 are the direct instances of this rule
  already designed into the plan; if implementation surfaces another one,
  add a test for it too before moving on).

---

### Task 1: `bootstrap.py` — paired and panel moving-block bootstrap

**Files:**
- Create: `src/tfm3lab/bootstrap.py`
- Test: `tests/test_bootstrap.py`

**Interfaces:**
- Produces: `BootstrapResult` (frozen dataclass: `delta_mean, ci_low,
  ci_high, ci_level, n_boot, block_size, seed, n_origins`, all defined on
  every successful call — no partial results); `paired_moving_block_bootstrap(model_abs_err, baseline_abs_err, horizon, block_size=None, n_boot=1000, ci=0.9, seed=config.SEED) -> BootstrapResult`;
  `panel_paired_block_bootstrap(deltas: dict[str, np.ndarray], horizon, block_size=None, n_boot=1000, ci=0.9, seed=config.SEED, weights: dict[str, int] | None = None) -> BootstrapResult`.
  Task 3 imports `paired_moving_block_bootstrap`; `scripts/08_exp_inference.py`
  (Task 9) imports `panel_paired_block_bootstrap`.
- Consumes: `config.SEED` from the existing `tfm3lab.config` module.

- [ ] **Step 1: Write the failing test file**

```python
"""Tests for bootstrap.py — paired and panel moving-block bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from tfm3lab.bootstrap import (
    BootstrapResult,
    paired_moving_block_bootstrap,
    panel_paired_block_bootstrap,
)


def test_delta_mean_is_hand_computed():
    model_err = np.array([1.0, 2.0, 3.0, 4.0])
    baseline_err = np.array([2.0, 2.0, 5.0, 4.0])
    # delta = baseline - model = [1, 0, 2, 0] -> mean 0.75
    result = paired_moving_block_bootstrap(
        model_err, baseline_err, horizon=1, block_size=1, n_boot=200, seed=1
    )
    assert result.delta_mean == pytest.approx(0.75)


def test_block_size_below_horizon_raises():
    model_err = np.arange(10.0)
    baseline_err = np.arange(10.0) + 1.0
    with pytest.raises(ValueError, match="block_size"):
        paired_moving_block_bootstrap(model_err, baseline_err, horizon=5, block_size=2)


def test_block_size_defaults_to_horizon():
    model_err = np.arange(10.0)
    baseline_err = np.arange(10.0) + 1.0
    result = paired_moving_block_bootstrap(model_err, baseline_err, horizon=3, n_boot=50, seed=1)
    assert result.block_size == 3


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        paired_moving_block_bootstrap(np.arange(5.0), np.arange(4.0), horizon=1)


def test_n_boot_must_be_positive():
    model_err = np.arange(5.0)
    baseline_err = np.arange(5.0) + 1.0
    with pytest.raises(ValueError, match="n_boot"):
        paired_moving_block_bootstrap(model_err, baseline_err, horizon=1, n_boot=0)


def test_same_seed_reproducible():
    rng = np.random.default_rng(0)
    n = 60
    model_err = np.abs(rng.normal(size=n))
    baseline_err = model_err + rng.normal(scale=0.1, size=n) ** 2
    r1 = paired_moving_block_bootstrap(model_err, baseline_err, horizon=3, n_boot=300, seed=7)
    r2 = paired_moving_block_bootstrap(model_err, baseline_err, horizon=3, n_boot=300, seed=7)
    assert r1 == r2


def test_different_seed_differs():
    n = 60
    rng = np.random.default_rng(1)
    model_err = np.abs(rng.normal(size=n))
    baseline_err = model_err + 0.3
    r1 = paired_moving_block_bootstrap(model_err, baseline_err, horizon=3, n_boot=300, seed=1)
    r2 = paired_moving_block_bootstrap(model_err, baseline_err, horizon=3, n_boot=300, seed=2)
    assert r1.seed != r2.seed
    assert (r1.ci_low, r1.ci_high) != (r2.ci_low, r2.ci_high)


def test_seed_is_echoed_on_result():
    model_err = np.arange(10.0)
    baseline_err = np.arange(10.0) + 1.0
    result = paired_moving_block_bootstrap(model_err, baseline_err, horizon=1, seed=123, n_boot=20)
    assert result.seed == 123


def test_block_resampling_widens_ci_on_autocorrelated_series():
    """An autocorrelated delta sequence: block_size=1 (effectively i.i.d.
    resampling of individual points, ignoring dependence) understates the
    true sampling variance compared to a properly sized block -- direct
    proof block resampling changes the answer, not just a documented claim.
    """
    rng = np.random.default_rng(3)
    n = 200
    noise = rng.normal(size=n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.9 * ar[i - 1] + noise[i]
    model_err = np.zeros(n)
    baseline_err = ar  # delta = baseline_err - model_err = ar

    small_block = paired_moving_block_bootstrap(
        model_err, baseline_err, horizon=1, block_size=1, n_boot=500, seed=42
    )
    large_block = paired_moving_block_bootstrap(
        model_err, baseline_err, horizon=1, block_size=20, n_boot=500, seed=42
    )
    width_small = small_block.ci_high - small_block.ci_low
    width_large = large_block.ci_high - large_block.ci_low
    assert width_large > width_small * 1.5


def test_panel_requires_equal_length_arrays():
    deltas = {"a": np.arange(10.0), "b": np.arange(5.0)}
    with pytest.raises(ValueError, match="same number of origins"):
        panel_paired_block_bootstrap(deltas, horizon=1)


def test_panel_requires_nonempty_deltas():
    with pytest.raises(ValueError, match="non-empty"):
        panel_paired_block_bootstrap({}, horizon=1)


def test_panel_point_estimate_is_weighted_average():
    deltas = {"a": np.full(10, 1.0), "b": np.full(10, 3.0)}
    weights = {"a": 1, "b": 9}
    result = panel_paired_block_bootstrap(
        deltas, horizon=1, block_size=1, n_boot=10, seed=1, weights=weights
    )
    # weighted mean of constant per-series deltas: (1*1 + 3*9) / 10 = 2.8
    assert result.delta_mean == pytest.approx(2.8)


def test_panel_ci_differs_from_naive_unweighted_average_of_per_series_cis():
    rng = np.random.default_rng(5)
    n = 60
    a = rng.normal(loc=1.0, scale=0.5, size=n)
    b = rng.normal(loc=1.0, scale=0.5, size=n)
    deltas = {"a": a, "b": b}
    weights = {"a": 50, "b": 1}  # heavily skewed weight

    panel_result = panel_paired_block_bootstrap(
        deltas, horizon=1, block_size=1, n_boot=500, seed=9, weights=weights
    )
    per_series_a = paired_moving_block_bootstrap(
        np.zeros(n), a, horizon=1, block_size=1, n_boot=500, seed=9
    )
    per_series_b = paired_moving_block_bootstrap(
        np.zeros(n), b, horizon=1, block_size=1, n_boot=500, seed=9
    )
    naive_unweighted_mean = (per_series_a.delta_mean + per_series_b.delta_mean) / 2
    assert panel_result.delta_mean != pytest.approx(naive_unweighted_mean, abs=1e-9)


def test_panel_block_size_below_horizon_raises():
    deltas = {"a": np.arange(10.0), "b": np.arange(10.0)}
    with pytest.raises(ValueError, match="block_size"):
        panel_paired_block_bootstrap(deltas, horizon=5, block_size=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bootstrap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tfm3lab.bootstrap'`

- [ ] **Step 3: Write the implementation**

```python
"""Paired moving-block bootstrap for the model-vs-baseline error delta.

A rolling-origin forecast's errors overlap in time (neighboring origins
share most of their context and near-identical horizons), so treating them
as i.i.d. observations understates the true sampling variance of any
aggregate statistic built from them -- this is the same problem
`metrics.block_bootstrap_ci` already solves for a single array of values.
This module applies the same block-resampling idea to a PAIRED delta
(model error vs baseline error at the same origin), so pairing survives
resampling, and adds a panel variant for the multi-series showcase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class BootstrapResult:
    """One bootstrap run's summary: point estimate, CI, and everything
    needed to reproduce it exactly (seed, block_size, n_boot)."""

    delta_mean: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_boot: int
    block_size: int
    seed: int
    n_origins: int


def _block_resample_indices(
    n: int, block_size: int, n_blocks: int, rng: np.random.Generator
) -> np.ndarray:
    """One bootstrap replicate's resampled index array: `n_blocks` contiguous
    blocks of `block_size`, drawn with replacement from every valid start
    position, concatenated and truncated to length `n`."""
    starts = np.arange(0, n - block_size + 1)
    chosen = rng.choice(starts, size=n_blocks, replace=True)
    idx = np.concatenate([np.arange(s, s + block_size) for s in chosen])
    return idx[:n]


def _validate_common(block_size: int, horizon: int, n: int, n_boot: int) -> None:
    if block_size < horizon:
        raise ValueError(f"block_size ({block_size}) must be >= horizon ({horizon})")
    if not (1 <= block_size <= n):
        raise ValueError(f"block_size must be in [1, {n}], got {block_size}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")


def paired_moving_block_bootstrap(
    model_abs_err,
    baseline_abs_err,
    horizon: int,
    block_size: int | None = None,
    n_boot: int = 1000,
    ci: float = 0.9,
    seed: int = config.SEED,
) -> BootstrapResult:
    """Moving-block bootstrap CI for the mean error-reduction delta
    (`baseline_abs_err - model_abs_err`, one value per forecast ORIGIN,
    positive = model better -- same sign convention as `skill = 1 -
    relative_mae`).

    `model_abs_err`/`baseline_abs_err` must already be paired by origin (the
    same origin at the same index in both arrays) and, like
    `metrics.diebold_mariano`, hold one loss per origin at a FIXED horizon
    step -- not averaged across horizon steps and not mixed across horizons.
    The delta is computed once, before any resampling, and every block
    resample moves the paired delta as a single unit, so pairing can never
    be broken.

    `block_size` defaults to `horizon` and must be >= `horizon`: a rolling
    h-step-ahead forecast has up to h-1 lags of autocorrelation in its
    error, so a block shorter than the horizon would still understate the
    true variance -- this is a checked invariant, not just a docstring
    promise.

    `seed` always has a value and is always returned on the result: there is
    no unseeded, unreproducible call.
    """
    model_abs_err = np.asarray(model_abs_err, dtype=float)
    baseline_abs_err = np.asarray(baseline_abs_err, dtype=float)
    if model_abs_err.shape != baseline_abs_err.shape:
        raise ValueError(
            f"shape mismatch: {model_abs_err.shape} vs {baseline_abs_err.shape}"
        )
    n = len(model_abs_err)
    if block_size is None:
        block_size = horizon
    _validate_common(block_size, horizon, n, n_boot)

    delta = baseline_abs_err - model_abs_err
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = _block_resample_indices(n, block_size, n_blocks, rng)
        means[b] = delta[idx].mean()

    alpha = 1 - ci
    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        delta_mean=float(delta.mean()),
        ci_low=float(lower),
        ci_high=float(upper),
        ci_level=ci,
        n_boot=n_boot,
        block_size=block_size,
        seed=seed,
        n_origins=n,
    )


def panel_paired_block_bootstrap(
    deltas: dict[str, np.ndarray],
    horizon: int,
    block_size: int | None = None,
    n_boot: int = 1000,
    ci: float = 0.9,
    seed: int = config.SEED,
    weights: dict[str, int] | None = None,
) -> BootstrapResult:
    """Panel (multi-series) moving-block bootstrap: ONE set of block-start
    positions is drawn per replicate and applied IDENTICALLY to every
    series' delta array, so the resample preserves cross-sectional
    correlation at a shared origin on top of each series' own
    autocorrelation -- a true panel block bootstrap, not per-series
    bootstraps reported side by side.

    Every array in `deltas` must be the same length and in the same origin
    order (the caller aligns this via `benchmark.common_origin_set`
    upstream -- this function does not re-derive that alignment). Each
    replicate's per-series resampled means are combined into one
    panel-level number via a weighted average; `weights` defaults to each
    series' observation count (`len(deltas[name])`), matching
    `summarize.aggregate_leaderboard`'s existing weighting convention --
    deliberately not a naive unweighted mean across series.
    """
    if not deltas:
        raise ValueError("deltas must be non-empty")
    names = list(deltas.keys())
    arrays = [np.asarray(deltas[name], dtype=float) for name in names]
    n = len(arrays[0])
    for name, arr in zip(names, arrays, strict=True):
        if len(arr) != n:
            raise ValueError(
                f"all series must have the same number of origins; "
                f"{names[0]} has {n}, {name} has {len(arr)}"
            )
    if block_size is None:
        block_size = horizon
    _validate_common(block_size, horizon, n, n_boot)

    if weights is None:
        weight_arr = np.full(len(names), float(n))
    else:
        weight_arr = np.array([float(weights[name]) for name in names])

    stacked = np.stack(arrays, axis=0)  # shape (n_series, n_origins)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    combined_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = _block_resample_indices(n, block_size, n_blocks, rng)
        per_series_mean = stacked[:, idx].mean(axis=1)
        combined_means[b] = np.average(per_series_mean, weights=weight_arr)

    point_per_series = stacked.mean(axis=1)
    point_estimate = float(np.average(point_per_series, weights=weight_arr))

    alpha = 1 - ci
    lower, upper = np.quantile(combined_means, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        delta_mean=point_estimate,
        ci_low=float(lower),
        ci_high=float(upper),
        ci_level=ci,
        n_boot=n_boot,
        block_size=block_size,
        seed=seed,
        n_origins=n,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_bootstrap.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/bootstrap.py tests/test_bootstrap.py
git commit -m "feat: add paired and panel moving-block bootstrap for error-delta CI"
```

---

### Task 2: `metrics.py` — binomial CI, weighted interval score, normalized width

**Files:**
- Modify: `src/tfm3lab/metrics.py` (append after `block_bootstrap_ci`)
- Test: `tests/test_metrics.py` (extend)

**Interfaces:**
- Produces: `binomial_ci(successes, n, ci=0.9) -> tuple[float, float]`;
  `weighted_interval_score(actual, quantile_forecasts, levels) -> float`;
  `interval_width_normalized(lower, upper, scale) -> float`. Task 4
  (`summarize_calibration`) and Task 6 (`figdata.quantile_bin_calibration`)
  and Task 7 (`calibration_curve`) import these.
- Consumes: nothing new — `scipy.stats` is already imported in `metrics.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metrics.py` (extend the existing import block at the
top of the file to add `binomial_ci`, `interval_width_normalized`, and
`weighted_interval_score` to the `from tfm3lab.metrics import (...)` list,
and add `from tfm3lab import config` near the top):

```python
def test_binomial_ci_known_clopper_pearson_values():
    # Cross-checked against scipy.stats.beta directly -- the same
    # construction binomial_ci uses -- since Clopper-Pearson has no closed
    # form to hand-compute independently.
    lower, upper = binomial_ci(8, 10, ci=0.90)
    expected_lower = stats.beta.ppf(0.05, 8, 3)
    expected_upper = stats.beta.ppf(0.95, 9, 2)
    assert lower == pytest.approx(expected_lower)
    assert upper == pytest.approx(expected_upper)


def test_binomial_ci_zero_successes_lower_bound_is_zero():
    lower, upper = binomial_ci(0, 10)
    assert lower == 0.0
    assert upper < 1.0


def test_binomial_ci_all_successes_upper_bound_is_one():
    lower, upper = binomial_ci(10, 10)
    assert upper == 1.0
    assert lower > 0.0


def test_binomial_ci_n_zero_returns_nan():
    lower, upper = binomial_ci(0, 0)
    assert np.isnan(lower)
    assert np.isnan(upper)


def test_binomial_ci_rejects_successes_out_of_range():
    with pytest.raises(ValueError):
        binomial_ci(11, 10)


def test_weighted_interval_score_perfect_quantiles_is_zero():
    levels = config.QUANTILE_LEVELS
    actual = np.array([5.0])
    quantile_forecasts = np.array([[5.0] * 9])
    wis = weighted_interval_score(actual, quantile_forecasts, levels)
    assert wis == pytest.approx(0.0)


def test_weighted_interval_score_undercovered_is_larger_than_wellcovered():
    levels = config.QUANTILE_LEVELS
    actual = np.array([100.0])
    well_covered = np.array([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 200.0]])
    undercovered = np.array([[45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0]])
    wis_well = weighted_interval_score(actual, well_covered, levels)
    wis_under = weighted_interval_score(actual, undercovered, levels)
    assert wis_under > wis_well


def test_interval_width_normalized_hand_computed():
    lower = np.array([8.0, 18.0])
    upper = np.array([12.0, 22.0])
    # widths [4, 4], mean width 4, scale 2 -> normalized 2.0
    assert interval_width_normalized(lower, upper, scale=2.0) == pytest.approx(2.0)


def test_interval_width_normalized_rejects_nonpositive_scale():
    with pytest.raises(ValueError):
        interval_width_normalized(np.array([1.0]), np.array([2.0]), scale=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ImportError: cannot import name 'binomial_ci'`

- [ ] **Step 3: Write the implementation**

Append to `src/tfm3lab/metrics.py` (after `block_bootstrap_ci`):

```python
def binomial_ci(successes: int, n: int, ci: float = 0.9) -> tuple[float, float]:
    """Exact Clopper-Pearson confidence interval for a binomial proportion
    (e.g. empirical coverage/calibration-bin fraction out of `n` trials).

    Unlike a normal-approximation interval, this stays valid at the extremes
    (successes=0 or successes=n) where a Wald interval collapses to zero
    width or goes outside [0, 1]. `n == 0` returns `(nan, nan)` -- there is
    no observation to bound.
    """
    if n == 0:
        return float("nan"), float("nan")
    if not (0 <= successes <= n):
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    alpha = 1 - ci
    lower = (
        0.0
        if successes == 0
        else float(stats.beta.ppf(alpha / 2, successes, n - successes + 1))
    )
    upper = (
        1.0
        if successes == n
        else float(stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    )
    return lower, upper


# Weighted interval score: the 4 nested central intervals available from the
# project's 9-level quantile grid (config.QUANTILE_LEVELS = 0.1..0.9), each
# paired with its Bracher, Held & Krymova (2021) weight alpha/2, plus the
# median at weight 0.5. K=4 intervals -> normalizer 1/(K + 0.5).
_WIS_PAIRS = (
    (0.1, 0.9, 0.2),
    (0.2, 0.8, 0.4),
    (0.3, 0.7, 0.6),
    (0.4, 0.6, 0.8),
)
_WIS_K = len(_WIS_PAIRS)


def weighted_interval_score(actual, quantile_forecasts, levels) -> float:
    """Weighted interval score (WIS), a proper scoring rule for a set of
    central prediction intervals plus the median -- Bracher, Held & Krymova
    (2021). This is the project's documented variant built from the 9
    quantile levels already computed everywhere else
    (`config.QUANTILE_LEVELS`): the 4 nested pairs (10/90, 20/80, 30/70,
    40/60) each scored as an interval score at their own alpha, plus
    |actual - median| at weight 0.5, normalized by 1/(K + 0.5) with K=4.

    `quantile_forecasts` has the quantile axis last, in the same order as
    `levels` (matches `pinball_loss_multi`'s convention). `levels` must
    include all of 0.1..0.9 in steps of 0.1 -- a different grid needs
    different pairs and is not handled here.
    """
    actual = np.asarray(actual, dtype=float)
    quantile_forecasts = np.asarray(quantile_forecasts, dtype=float)
    levels = list(levels)
    level_index = {round(level, 4): i for i, level in enumerate(levels)}

    def q(level: float) -> np.ndarray:
        return quantile_forecasts[..., level_index[round(level, 4)]]

    median = q(0.5)
    total = 0.5 * np.abs(actual - median)
    for lower_level, upper_level, alpha in _WIS_PAIRS:
        lower = q(lower_level)
        upper = q(upper_level)
        width = upper - lower
        under = (2.0 / alpha) * np.maximum(lower - actual, 0.0)
        over = (2.0 / alpha) * np.maximum(actual - upper, 0.0)
        interval_score = width + under + over
        weight = alpha / 2.0
        total = total + weight * interval_score
    wis = total / (_WIS_K + 0.5)
    return float(np.mean(wis))


def interval_width_normalized(lower, upper, scale: float) -> float:
    """Mean prediction-interval width divided by `scale` -- the SAME
    per-series in-sample scale used for MASE (`in_sample_scale`), so
    "normalized" means the same thing here as it does for the accuracy
    metrics instead of introducing a second, unrelated denominator.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return float(np.mean(upper - lower) / scale)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (all tests, existing + 9 new)

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/metrics.py tests/test_metrics.py
git commit -m "feat: add binomial CI, weighted interval score, normalized interval width"
```

---

### Task 3: `summarize.py` — Benjamini-Hochberg correction + significance table

**Files:**
- Modify: `src/tfm3lab/summarize.py` (extend import block, append two functions)
- Test: `tests/test_summarize.py` (extend)

**Interfaces:**
- Consumes: `bootstrap.paired_moving_block_bootstrap` (Task 1),
  `statsmodels.stats.multitest.multipletests` (already a project dependency).
- Produces: `apply_bh_correction(df, pvalue_col="dm_pvalue", family_cols=()) -> pd.DataFrame`;
  `summarize_significance(df, mase_scales, group_cols=("mode","series","horizon_step"), baseline_col="baseline_naive", n_boot=1000, ci=0.9, seed=config.SEED) -> pd.DataFrame`.
  `scripts/08_exp_inference.py` (Task 9) imports `summarize_significance`.

- [ ] **Step 1: Write the failing tests**

Extend the `from tfm3lab.summarize import (...)` block in
`tests/test_summarize.py` to add `apply_bh_correction` and
`summarize_significance`, and append:

```python
def test_apply_bh_correction_matches_statsmodels_fdr_bh():
    from statsmodels.stats.multitest import multipletests

    df = pd.DataFrame({"dm_pvalue": [0.001, 0.02, 0.04, 0.5, 0.8]})
    result = apply_bh_correction(df)
    _, expected_q, _, _ = multipletests(df["dm_pvalue"].to_numpy(), method="fdr_bh")
    np.testing.assert_allclose(result["dm_qvalue_bh"].to_numpy(), expected_q)


def test_apply_bh_correction_leaves_nan_pvalues_as_nan():
    df = pd.DataFrame({"dm_pvalue": [0.01, np.nan, 0.03, np.nan]})
    result = apply_bh_correction(df)
    assert result["dm_qvalue_bh"].isna().tolist() == [False, True, False, True]


def test_apply_bh_correction_splits_by_family():
    # same raw p-values, once corrected as one family of 2, once pooled with
    # 4 more into a family of 6 -- BH's correction depends on how many tests
    # are in the family, so the per-family result must differ from pooling
    # everything together.
    df = pd.DataFrame(
        {
            "family": ["a", "a", "b", "b", "b", "b"],
            "dm_pvalue": [0.01, 0.02, 0.01, 0.02, 0.03, 0.04],
        }
    )
    per_family = apply_bh_correction(df, family_cols=("family",))
    pooled = apply_bh_correction(df)
    assert per_family.loc[0, "dm_qvalue_bh"] != pooled.loc[0, "dm_qvalue_bh"]


def test_summarize_significance_end_to_end():
    rows = []
    err_ratios = [0.5, 1.5, 0.8, 1.2, 0.6, 1.4, 0.9, 1.1, 0.7, 1.3]
    for i, v in enumerate(err_ratios):
        rows.append(
            _row(
                mode="timesfm3_univariate",
                series="a",
                horizon_step=1,
                actual=10.0 + i,
                observed=True,
                forecast=10.0 + i + 0.1,  # small model error
                baseline_naive=10.0 + i + v,  # larger baseline error
            )
        )
    df = pd.DataFrame(rows)
    mase_scales = {"a": 1.0}
    result = summarize_significance(df, mase_scales, n_boot=200, seed=1)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["delta_mean"] > 0  # model beats baseline on average
    assert row["ci_low"] <= row["delta_mean"] <= row["ci_high"]
    assert "dm_qvalue_bh" in result.columns


def test_summarize_significance_tiny_group_gets_nan_bootstrap():
    df = pd.DataFrame(
        [
            _row(
                mode="m", series="a", horizon_step=5, actual=1.0, observed=True,
                forecast=1.0, baseline_naive=1.5,
            )
        ]
    )
    mase_scales = {"a": 1.0}
    result = summarize_significance(df, mase_scales, n_boot=50, seed=1)
    assert np.isnan(result.iloc[0]["delta_mean"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_summarize.py -v`
Expected: FAIL with `ImportError: cannot import name 'apply_bh_correction'`

- [ ] **Step 3: Write the implementation**

Extend the import block at the top of `src/tfm3lab/summarize.py`:

```python
from statsmodels.stats.multitest import multipletests

from .bootstrap import paired_moving_block_bootstrap
```

Append to `src/tfm3lab/summarize.py` (after `aggregate_leaderboard`):

```python
def apply_bh_correction(
    df: pd.DataFrame,
    pvalue_col: str = "dm_pvalue",
    family_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Adds a Benjamini-Hochberg corrected q-value column
    (`dm_qvalue_bh` when `pvalue_col == "dm_pvalue"`, else
    `{pvalue_col}_qvalue_bh`), one correction per family.

    `family_cols=()` (the default) treats the WHOLE `df` as one family --
    correct when `df` already covers exactly one logical comparison (e.g.
    one call to `summarize_significance`, already filtered to one
    mode/transform/baseline combination before this runs). Pass non-empty
    `family_cols` only if a caller stacks several comparisons into one df
    and wants each corrected independently.

    Rows where `pvalue_col` is NaN (e.g. underpowered groups excluded by
    `MIN_OBSERVATIONS_FOR_DM_TEST`) are excluded from the correction and
    keep NaN at their original row -- never coerced to 0 or 1, never
    dropped from the returned table.
    """
    out = df.copy()
    qcol = "dm_qvalue_bh" if pvalue_col == "dm_pvalue" else f"{pvalue_col}_qvalue_bh"
    out[qcol] = np.nan

    index_groups = (
        list(out.groupby(list(family_cols)).groups.values()) if family_cols else [out.index]
    )
    for idx in index_groups:
        pvals = out.loc[idx, pvalue_col].to_numpy(dtype=float)
        valid_mask = ~np.isnan(pvals)
        if not valid_mask.any():
            continue
        _, qvals, _, _ = multipletests(pvals[valid_mask], method="fdr_bh")
        out.loc[idx[valid_mask], qcol] = qvals
    return out


def summarize_significance(
    df: pd.DataFrame,
    mase_scales: dict[str, float],
    group_cols: tuple[str, ...] = ("mode", "series", "horizon_step"),
    baseline_col: str = "baseline_naive",
    n_boot: int = 1000,
    ci: float = 0.9,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """One row per group: Diebold-Mariano stat/p-value (reusing
    `summarize_accuracy`'s existing logic, not duplicating it) plus a
    paired moving-block bootstrap effect size and CI on the error-reduction
    delta (`bootstrap.paired_moving_block_bootstrap`), plus a
    Benjamini-Hochberg corrected q-value across every group in the returned
    table (one call to this function is expected to already cover exactly
    one comparison family -- see `apply_bh_correction`).

    Reporting rule: `delta_mean`/`ci_low`/`ci_high` (equivalently
    `skill_vs_baseline = 1 - relative_mae_vs_baseline` from
    `summarize_accuracy`) is the HEADLINE effect size. `dm_pvalue` /
    `dm_qvalue_bh` are supplementary significance signals, never the
    primary claim -- see docs/analysis-plan.md's Claim rule.

    A group with fewer observed rows than its own horizon_step cannot
    satisfy the bootstrap's `block_size >= horizon` requirement -- such a
    group's bootstrap columns are NaN (not an invented number), matching
    the project-wide "NaN over a fabricated statistic" convention already
    used by the DM columns.
    """
    accuracy = summarize_accuracy(
        df, mase_scales, baseline_col=baseline_col, group_cols=group_cols
    )
    horizon_index = group_cols.index("horizon_step") if "horizon_step" in group_cols else None

    observed = df[df["observed"]]
    boot_rows = []
    for keys, group in observed.groupby(list(group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        horizon = int(keys[horizon_index]) if horizon_index is not None else 1
        row = dict(zip(group_cols, keys, strict=True))
        if len(group) < horizon:
            row.update(
                {
                    "delta_mean": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "ci_level": ci,
                    "block_size": horizon,
                    "n_boot": n_boot,
                    "bootstrap_seed": seed,
                }
            )
            boot_rows.append(row)
            continue
        actual = group["actual"].to_numpy()
        model_pred = group["forecast"].to_numpy()
        baseline_pred = group[baseline_col].to_numpy()
        model_abs_err = np.abs(actual - model_pred)
        baseline_abs_err = np.abs(actual - baseline_pred)
        result = paired_moving_block_bootstrap(
            model_abs_err, baseline_abs_err, horizon=horizon, n_boot=n_boot, ci=ci, seed=seed
        )
        row.update(
            {
                "delta_mean": result.delta_mean,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "ci_level": result.ci_level,
                "block_size": result.block_size,
                "n_boot": result.n_boot,
                "bootstrap_seed": result.seed,
            }
        )
        boot_rows.append(row)
    boot_df = pd.DataFrame(boot_rows)

    merged = accuracy.merge(boot_df, on=list(group_cols), how="left")
    return apply_bh_correction(merged)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_summarize.py -v`
Expected: PASS (all tests, existing + 5 new)

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/summarize.py tests/test_summarize.py
git commit -m "feat: add Benjamini-Hochberg correction and bootstrap-backed significance table"
```

---

### Task 4: `summarize.py` — WIS and normalized width in `summarize_calibration`

**Files:**
- Modify: `src/tfm3lab/summarize.py` (extend import block, extend
  `summarize_calibration`)
- Test: `tests/test_summarize.py` (extend)

**Interfaces:**
- Consumes: `metrics.interval_width_normalized`, `metrics.weighted_interval_score` (Task 2).
- Produces: `summarize_calibration(df, group_cols=(...), mase_scales=None) -> pd.DataFrame`
  — additive optional param; existing callers (`scripts/02_exp_mtg.py`,
  `scripts/04_exp_calibration.py`) pass no `mase_scales` and see no
  behavior change. `scripts/08_exp_inference.py` (Task 9) passes
  `mase_scales`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_summarize.py`:

```python
def test_summarize_calibration_without_mase_scales_keeps_existing_columns():
    df = pd.DataFrame([_row(actual=1.0, forecast=1.0, quantiles=[1.0] * config.N_QUANTILES)])
    summary = summarize_calibration(df)
    assert "wis" not in summary.columns
    assert "interval_width_normalized" not in summary.columns


def test_summarize_calibration_with_mase_scales_adds_wis_and_normalized_width():
    df = pd.DataFrame(
        [
            _row(
                series="a", actual=10.0, forecast=10.0,
                quantiles=[8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0],
            )
        ]
    )
    summary = summarize_calibration(df, mase_scales={"a": 2.0})
    assert "wis" in summary.columns
    assert "interval_width_normalized" in summary.columns
    # width = q90 - q10 = 12.0 - 8.0 = 4.0, scale 2.0 -> normalized 2.0
    assert summary.iloc[0]["interval_width_normalized"] == pytest.approx(2.0)


def test_summarize_calibration_requires_series_in_group_cols_for_mase_scales():
    df = pd.DataFrame([_row()])
    with pytest.raises(ValueError, match="series"):
        summarize_calibration(df, group_cols=("mode",), mase_scales={"a": 1.0})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_summarize.py -v`
Expected: FAIL — `summarize_calibration() got an unexpected keyword argument 'mase_scales'`

- [ ] **Step 3: Write the implementation**

Extend the `.metrics` import in `src/tfm3lab/summarize.py` (the existing
`from .metrics import (...)` block) to add `interval_width_normalized` and
`weighted_interval_score`, alphabetically ordered with the rest.

Replace `summarize_calibration`'s body:

```python
def summarize_calibration(
    df: pd.DataFrame,
    group_cols: tuple[str, ...] = ("mode", "series", "horizon_step"),
    mase_scales: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Pinball loss (averaged over all 9 quantiles), P10-P90 coverage, and
    mean PIT value per group. A well-calibrated group has coverage near
    0.80 and mean PIT near 0.5 -- see docs/talk-outline.md for how this
    feeds Experiment C.

    `mase_scales`, when given, adds `interval_width_normalized` (P10-P90
    width divided by the same in-sample scale used for MASE) and `wis`
    (weighted interval score, see `metrics.weighted_interval_score`)
    columns -- both look up each group's series in `mase_scales` the same
    way `summarize_accuracy` does. Omitted (the default) keeps the
    existing column set unchanged, so callers that don't pass it see no
    behavior change.
    """
    if mase_scales is not None and "series" not in group_cols:
        raise ValueError("group_cols must include 'series' to look up its MASE scale")
    series_idx = group_cols.index("series") if mase_scales is not None else None

    observed = df[df["observed"]]
    rows = []
    for keys, group in observed.groupby(list(group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        actual = group["actual"].to_numpy()
        quantiles = group[QUANTILE_COLUMNS].to_numpy()
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "n": len(group),
                "pinball_avg": pinball_loss_multi(actual, quantiles, config.QUANTILE_LEVELS),
                "coverage_p10_p90": coverage(
                    actual, group["q10"].to_numpy(), group["q90"].to_numpy()
                ),
                "pit_mean": float(np.mean(pit_values(actual, quantiles, config.QUANTILE_LEVELS))),
            }
        )
        if mase_scales is not None:
            scale = mase_scales.get(keys[series_idx], 1.0)
            row["interval_width_normalized"] = interval_width_normalized(
                group["q10"].to_numpy(), group["q90"].to_numpy(), scale
            )
            row["wis"] = weighted_interval_score(actual, quantiles, config.QUANTILE_LEVELS)
        rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_summarize.py -v`
Expected: PASS (all tests, existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/summarize.py tests/test_summarize.py
git commit -m "feat: add WIS and normalized interval width to summarize_calibration"
```

---

### Task 5: `conformal.py` — causally-valid conformal interval calibration

**Files:**
- Create: `src/tfm3lab/conformal.py`
- Test: `tests/test_conformal.py`

**Interfaces:**
- Consumes: `metrics.coverage` (existing).
- Produces: `conformalize_intervals(df, alpha=0.2, min_calibration_origins=20, group_cols=("series","horizon_step")) -> pd.DataFrame`;
  `evaluate_conformal_coverage(df, group_cols=("series","horizon_step")) -> pd.DataFrame`.
  `scripts/08_exp_inference.py` (Task 9) imports both.

- [ ] **Step 1: Write the failing test file**

```python
"""Tests for conformal.py — causally-valid conformal interval calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm3lab.conformal import conformalize_intervals, evaluate_conformal_coverage


def _make_df(n, actual_fn, q10_fn, q90_fn, observed_fn=lambda i: True, horizon_step=1, series="a"):
    rows = []
    for i in range(n):
        rows.append(
            {
                "series": series,
                "horizon_step": horizon_step,
                "origin_index": i,
                "actual": actual_fn(i),
                "observed": observed_fn(i),
                "q10": q10_fn(i),
                "q90": q90_fn(i),
            }
        )
    return pd.DataFrame(rows)


def test_perfect_quantiles_conformal_adjustment_near_zero():
    df = _make_df(n=40, actual_fn=lambda i: 10.0, q10_fn=lambda i: 9.0, q90_fn=lambda i: 11.0)
    result = conformalize_intervals(df, alpha=0.2, min_calibration_origins=10)
    conformalized = result[result["conformalized"]]
    assert (conformalized["conformal_score_threshold"] <= 0).all()
    coverage_after = evaluate_conformal_coverage(result)
    assert coverage_after.iloc[0]["coverage_conformal"] == pytest.approx(1.0)


def test_undercovered_quantiles_conformal_improves_coverage():
    rng = np.random.default_rng(0)
    n = 200
    actual = rng.normal(loc=0.0, scale=1.0, size=n)
    q10 = np.full(n, -0.3)
    q90 = np.full(n, 0.3)
    df = _make_df(n, lambda i: actual[i], lambda i: q10[i], lambda i: q90[i])

    result = conformalize_intervals(df, alpha=0.2, min_calibration_origins=30)
    coverage_table = evaluate_conformal_coverage(result)
    row = coverage_table.iloc[0]
    assert row["coverage_raw"] < 0.80
    assert row["coverage_conformal"] > row["coverage_raw"]


def test_below_minimum_calibration_stays_raw():
    df = _make_df(n=5, actual_fn=lambda i: 1.0, q10_fn=lambda i: 0.5, q90_fn=lambda i: 1.5)
    result = conformalize_intervals(df, min_calibration_origins=20)
    assert not result["conformalized"].any()
    assert (result["q10_conformal"] == result["q10"]).all()
    assert (result["q90_conformal"] == result["q90"]).all()


def test_calibration_never_uses_current_or_future_origin():
    n = 50
    actual = [10.0] * n
    actual[45] = 1000.0  # huge future outlier, late in the series
    df = _make_df(n, lambda i: actual[i], lambda i: 9.5, lambda i: 10.5)

    result = conformalize_intervals(df, alpha=0.2, min_calibration_origins=20)
    row_20 = result[result["origin_index"] == 20].iloc[0]
    # calibration set for origin 20 is origins 0..19 -- none is the outlier
    # at origin 45, so the threshold must equal exactly the calm-data score,
    # unaffected by a value that (from origin 20's point of view) hasn't
    # happened yet.
    assert row_20["conformal_score_threshold"] == pytest.approx(-0.5)


def test_forward_filled_actuals_excluded_from_calibration():
    n = 40
    observed = [True] * n
    observed[10] = False
    actual = [10.0] * n
    actual[10] = 999.0  # forward-filled row, would be an outlier if used
    df = _make_df(
        n, lambda i: actual[i], lambda i: 9.5, lambda i: 10.5, observed_fn=lambda i: observed[i]
    )
    result = conformalize_intervals(df, alpha=0.2, min_calibration_origins=20)
    row_30 = result[result["origin_index"] == 30].iloc[0]
    assert row_30["conformal_calibration_n"] == 29  # origins 0..29 minus the excluded origin 10
    assert row_30["conformal_score_threshold"] == pytest.approx(-0.5)


def test_alpha_out_of_range_raises():
    df = _make_df(n=5, actual_fn=lambda i: 1.0, q10_fn=lambda i: 0.5, q90_fn=lambda i: 1.5)
    with pytest.raises(ValueError, match="alpha"):
        conformalize_intervals(df, alpha=1.5)


def test_evaluate_conformal_coverage_reports_skipped_rows():
    df = _make_df(n=15, actual_fn=lambda i: 1.0, q10_fn=lambda i: 0.5, q90_fn=lambda i: 1.5)
    result = conformalize_intervals(df, min_calibration_origins=20)  # nothing qualifies
    coverage_table = evaluate_conformal_coverage(result)
    row = coverage_table.iloc[0]
    assert row["n_skipped_insufficient_calibration"] == 15
    assert row["n_evaluated"] == 0
    assert np.isnan(row["coverage_conformal"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_conformal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tfm3lab.conformal'`

- [ ] **Step 3: Write the implementation**

```python
"""Causally-valid conformal post-processing of TimesFM's P10-P90 interval.

Split-conformal / CQR-style calibration, computed strictly from a series'
OWN PAST forecast errors -- never a future target, never the same origin's
own error, never a forward-filled (non-observed) actual. This is an ONLINE
POST-HOC WRAPPER around the zero-shot forecasts, not a zero-shot result
itself: report raw and conformalized results side by side, never blended
into one number, and never call the conformalized columns "zero-shot".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import coverage


def _nonconformity_scores(actual: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> np.ndarray:
    """CQR nonconformity score: how far outside [q_low, q_high] `actual`
    fell (negative when comfortably inside -- more negative for a wider
    interval than needed)."""
    return np.maximum(q_low - actual, actual - q_high)


def conformalize_intervals(
    df: pd.DataFrame,
    alpha: float = 0.2,
    min_calibration_origins: int = 20,
    group_cols: tuple[str, ...] = ("series", "horizon_step"),
) -> pd.DataFrame:
    """Adds `q10_conformal`, `q90_conformal`, `conformal_score_threshold`,
    `conformal_calibration_n`, `conformalized` columns to a COPY of `df`.
    Raw `q10`/`q90` are never modified.

    Per `group_cols` group, sorted by `origin_index`: row i's calibration
    set is every STRICTLY EARLIER row in the same group (`origin_index` <
    row i's) with `observed == True` -- never the current or a later
    origin, never a forward-filled target. Below `min_calibration_origins`
    such rows, the row is passed through unchanged with
    `conformalized = False`. Otherwise the threshold is the finite-sample
    split-conformal empirical quantile of past nonconformity scores (the
    `ceil((n+1)*(1-alpha))`-th smallest, 1-indexed, clipped to the
    available count -- the standard split-conformal / Romano et al. CQR
    convention), and the interval widens (or narrows, if the threshold is
    negative -- not clamped at 0, or the coverage guarantee breaks) by that
    threshold on both sides.
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    out = df.copy()
    out["q10_conformal"] = out["q10"].astype(float)
    out["q90_conformal"] = out["q90"].astype(float)
    out["conformal_score_threshold"] = np.nan
    out["conformal_calibration_n"] = 0
    out["conformalized"] = False

    for _keys, group in out.groupby(list(group_cols)):
        group_sorted = group.sort_values("origin_index")
        origins = group_sorted["origin_index"].to_numpy()
        actual = group_sorted["actual"].to_numpy(dtype=float)
        q_low = group_sorted["q10"].to_numpy(dtype=float)
        q_high = group_sorted["q90"].to_numpy(dtype=float)
        observed_mask = group_sorted["observed"].to_numpy(dtype=bool)
        scores = _nonconformity_scores(actual, q_low, q_high)
        row_labels = group_sorted.index.to_numpy()

        for i in range(len(group_sorted)):
            past_mask = (origins < origins[i]) & observed_mask
            n_cal = int(past_mask.sum())
            if n_cal < min_calibration_origins:
                continue
            past_scores = np.sort(scores[past_mask])
            rank = min(int(np.ceil((n_cal + 1) * (1 - alpha))), n_cal)  # 1-indexed, clipped
            threshold = float(past_scores[rank - 1])

            label = row_labels[i]
            out.loc[label, "q10_conformal"] = q_low[i] - threshold
            out.loc[label, "q90_conformal"] = q_high[i] + threshold
            out.loc[label, "conformal_score_threshold"] = threshold
            out.loc[label, "conformal_calibration_n"] = n_cal
            out.loc[label, "conformalized"] = True

    return out


def evaluate_conformal_coverage(
    df: pd.DataFrame, group_cols: tuple[str, ...] = ("series", "horizon_step")
) -> pd.DataFrame:
    """Per group (restricted to `observed == True` AND `conformalized ==
    True` rows, for a fair apples-to-apples comparison): coverage and mean
    interval width, raw vs conformalized. `n_skipped_insufficient_calibration`
    reports how many observed rows in the group never got conformalized
    (too little calibration history) -- never silently dropped from the
    picture.
    """
    observed = df[df["observed"]]
    rows = []
    for keys, group in observed.groupby(list(group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys, strict=True))
        conformalized = group[group["conformalized"]]
        row["n_skipped_insufficient_calibration"] = int((~group["conformalized"]).sum())
        row["n_evaluated"] = len(conformalized)
        if len(conformalized) == 0:
            row.update(
                {
                    "coverage_raw": float("nan"),
                    "coverage_conformal": float("nan"),
                    "width_raw_mean": float("nan"),
                    "width_conformal_mean": float("nan"),
                }
            )
        else:
            actual = conformalized["actual"].to_numpy(dtype=float)
            row["coverage_raw"] = coverage(
                actual, conformalized["q10"].to_numpy(), conformalized["q90"].to_numpy()
            )
            row["coverage_conformal"] = coverage(
                actual,
                conformalized["q10_conformal"].to_numpy(),
                conformalized["q90_conformal"].to_numpy(),
            )
            row["width_raw_mean"] = float((conformalized["q90"] - conformalized["q10"]).mean())
            row["width_conformal_mean"] = float(
                (conformalized["q90_conformal"] - conformalized["q10_conformal"]).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_conformal.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/conformal.py tests/test_conformal.py
git commit -m "feat: add causally-valid conformal interval post-processing"
```

---

### Task 6: `figdata.py` — binomial CI on quantile-bin calibration

**Files:**
- Modify: `src/tfm3lab/figdata.py`
- Test: `tests/test_figdata.py` (extend)

**Interfaces:**
- Consumes: `metrics.binomial_ci` (Task 2).
- Produces: `quantile_bin_calibration(...)` output gains `ci_low`/`ci_high`
  columns — additive, existing columns unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_figdata.py` (in the `--- quantile_bin_calibration
---` section):

```python
def test_quantile_bin_calibration_bin_ci_matches_binomial_ci():
    from tfm3lab.metrics import binomial_ci

    preds = _synthetic_preds(n_origins=30, horizon=1, start_origin=64)
    preds.loc[preds["horizon_step"] == 1, "actual"] = -1_000_000.0  # always below every quantile
    hist = figdata.quantile_bin_calibration(preds, horizon_steps=(1,))
    bin0 = hist[hist["bin_index"] == 0].iloc[0]
    expected_low, expected_high = binomial_ci(30, 30)
    assert bin0["ci_low"] == pytest.approx(expected_low)
    assert bin0["ci_high"] == pytest.approx(expected_high)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_figdata.py -v -k quantile_bin_calibration`
Expected: FAIL with `KeyError: 'ci_low'`

- [ ] **Step 3: Write the implementation**

Change the import line in `src/tfm3lab/figdata.py`:

```python
from .metrics import binomial_ci, coverage, mae
```

In `quantile_bin_calibration`, replace the `rows.append({...})` block
inside the `for i in range(n_bins):` loop:

```python
        for i in range(n_bins):
            if i == 0:
                label = "≤ q10"
            elif i == n_bins - 1:
                label = "> q90"
            else:
                label = f"({levels[i - 1]:.1f}, {levels[i]:.1f}]"
            ci_low, ci_high = binomial_ci(int(counts[i]), n) if n else (float("nan"), float("nan"))
            rows.append(
                {
                    "horizon_step": h,
                    "bin_index": i,
                    "label": label,
                    "count": int(counts[i]),
                    "fraction": float(counts[i]) / n if n else float("nan"),
                    "nominal_fraction": 1.0 / n_bins,
                    "n": n,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_figdata.py -v -k quantile_bin_calibration`
Expected: PASS (existing 4 + 1 new; `test_pit_histogram_alias_still_works`
still passes since both sides of its `assert_frame_equal` gain the same
new columns)

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/figdata.py tests/test_figdata.py
git commit -m "feat: add binomial CI to quantile-bin calibration"
```

---

### Task 7: `scripts/04_exp_calibration.py` — CI band on the calibration curve

**Files:**
- Modify: `scripts/04_exp_calibration.py`
- Test: Create `tests/test_exp_calibration_cli.py`

**Interfaces:**
- Consumes: `metrics.binomial_ci` (Task 2).
- Produces: `calibration_curve(df, group_col="regime")` output gains
  `ci_low`/`ci_high` columns.

- [ ] **Step 1: Write the failing test file**

```python
"""Tests for scripts/04_exp_calibration.py's calibration_curve — argparse-free
pure function, loaded via importlib since scripts/ isn't a package (same
pattern as tests/test_fetch_data_cli.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "04_exp_calibration.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("exp_calibration_04", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def calib04():
    return _load_script_module()


def test_calibration_curve_ci_matches_binomial_ci(calib04):
    from tfm3lab.metrics import binomial_ci

    rows = []
    for i in range(20):
        row = {"regime": "mtg", "observed": True, "actual": -1000.0}
        for level, col in zip(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"],
            strict=True,
        ):
            row[col] = float(level)  # every quantile forecast well above `actual`
        rows.append(row)
    df = pd.DataFrame(rows)

    curve = calib04.calibration_curve(df)
    row0 = curve[curve["nominal_level"] == 0.1].iloc[0]
    # actual always below every quantile -> empirical coverage 1.0 at every level
    expected_low, expected_high = binomial_ci(20, 20)
    assert row0["empirical_coverage"] == pytest.approx(1.0)
    assert row0["ci_low"] == pytest.approx(expected_low)
    assert row0["ci_high"] == pytest.approx(expected_high)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_exp_calibration_cli.py -v`
Expected: FAIL with `KeyError: 'ci_low'`

- [ ] **Step 3: Write the implementation**

Add to the imports at the top of `scripts/04_exp_calibration.py`:

```python
from tfm3lab.metrics import binomial_ci
```

Replace `calibration_curve`'s body:

```python
def calibration_curve(df: pd.DataFrame, group_col: str = "regime") -> pd.DataFrame:
    """Empirical vs nominal coverage at every quantile level: for a
    well-calibrated forecaster, P(actual <= q_level) should equal
    `level` -- e.g. the actual should fall below the q70 forecast 70% of
    the time. Deviations show up directly as empirical != nominal.

    `ci_low`/`ci_high` are an exact Clopper-Pearson binomial CI on the
    empirical coverage (`metrics.binomial_ci`) -- how much sampling noise
    alone could explain a gap from the nominal level, at this group's `n`.
    """
    observed = df[df["observed"]]
    rows = []
    for regime, group in observed.groupby(group_col):
        n = len(group)
        for level, col in zip(config.QUANTILE_LEVELS, QUANTILE_COLUMNS, strict=True):
            successes = int((group["actual"] <= group[col]).sum())
            empirical = float(successes) / n if n else float("nan")
            ci_low, ci_high = binomial_ci(successes, n)
            rows.append(
                {
                    group_col: regime,
                    "nominal_level": level,
                    "empirical_coverage": empirical,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "gap": empirical - level,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_exp_calibration_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/04_exp_calibration.py tests/test_exp_calibration_cli.py
git commit -m "feat: add binomial CI band to the calibration curve"
```

---

### Task 8: `plots.py` — CI band, bootstrap delta forest plot, conformal coverage bars

**Files:**
- Modify: `src/tfm3lab/plots.py`
- Test: `tests/test_plots_smoke.py` (extend)

**Interfaces:**
- Consumes: nothing new (pure rendering of DataFrames already shaped by
  Tasks 3/5/7).
- Produces: `plot_calibration_curve` gains an optional CI band (backward
  compatible); new `plot_bootstrap_delta(significance, ax=None, *, label_col="series")`;
  new `plot_conformal_coverage(coverage_df, ax=None, *, label_col="series")`.
  `scripts/08_exp_inference.py` (Task 9) calls both new functions.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_plots_smoke.py`:

```python
def test_plot_calibration_curve_with_ci_band_still_three_lines():
    curve = pd.DataFrame(
        {
            "regime": ["market_calm"] * 3 + ["market_shock"] * 3,
            "nominal_level": [0.1, 0.5, 0.9] * 2,
            "empirical_coverage": [0.09, 0.5, 0.88, 0.15, 0.55, 0.8],
            "ci_low": [0.05, 0.45, 0.83, 0.10, 0.50, 0.75],
            "ci_high": [0.13, 0.55, 0.93, 0.20, 0.60, 0.85],
            "n": [100] * 6,
        }
    )
    ax = plots.plot_calibration_curve(curve)
    assert len(ax.lines) == 3  # ci band uses fill_between (collections), not extra lines
    assert len(ax.collections) == 2  # one filled band per regime


def test_plot_bootstrap_delta_one_row_per_group():
    significance = pd.DataFrame(
        {
            "series": ["A", "B", "C"],
            "delta_mean": [0.5, -0.2, 1.0],
            "ci_low": [0.1, -0.5, 0.6],
            "ci_high": [0.9, 0.1, 1.4],
        }
    )
    ax = plots.plot_bootstrap_delta(significance)
    assert len(ax.lines) >= 1  # errorbar draws at least the point markers


def test_plot_conformal_coverage_two_bars_per_group():
    coverage_df = pd.DataFrame(
        {"series": ["A", "B"], "coverage_raw": [0.6, 0.7], "coverage_conformal": [0.78, 0.82]}
    )
    ax = plots.plot_conformal_coverage(coverage_df)
    assert len(ax.patches) == 4  # 2 series x 2 bars each
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_plots_smoke.py -v`
Expected: FAIL — `test_plot_calibration_curve_with_ci_band_still_three_lines`
fails on `assert len(ax.collections) == 2` (currently 0, no CI band drawn);
`plot_bootstrap_delta`/`plot_conformal_coverage` fail with `AttributeError`

- [ ] **Step 3: Write the implementation**

Replace `plot_calibration_curve`'s body in `src/tfm3lab/plots.py`:

```python
def plot_calibration_curve(curve: pd.DataFrame, ax=None, *, group_col: str = "regime"):
    """Nominal vs empirical coverage, one line per group in `group_col`
    (expects results/exp_calibration_curve.parquet's regimes: market_calm,
    market_shock, mtg — all at the same horizon, never pooled across
    domain or horizon, see scripts/04_exp_calibration.py).

    When `curve` has `ci_low`/`ci_high` columns, each regime's line gets a
    shaded band for its exact binomial CI at each nominal level — absent
    those columns (an older/plain curve table), the plot is unchanged."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5))
    colors = {
        "market_calm": PALETTE["pre"], "market_shock": PALETTE["alert"], "mtg": PALETTE["post"]
    }
    ax.plot(
        [0, 1], [0, 1], color=PALETTE["baseline"], linestyle="--", label="calibrazione perfetta"
    )
    has_ci = "ci_low" in curve.columns and "ci_high" in curve.columns
    for key, g in curve.sort_values("nominal_level").groupby(group_col):
        n = int(g["n"].iloc[0])
        color = colors.get(key)
        if has_ci:
            ax.fill_between(g["nominal_level"], g["ci_low"], g["ci_high"], color=color, alpha=0.15)
        ax.plot(
            g["nominal_level"], g["empirical_coverage"], marker="o",
            color=color, label=f"{key} (n={n})",
        )
    ax.set_xlabel("livello nominale del quantile")
    ax.set_ylabel("copertura empirica")
    ax.set_title("Esperimento C — calibrazione, stesso orizzonte (h=1)")
    ax.legend(fontsize=9)
    return ax


def plot_bootstrap_delta(significance: pd.DataFrame, ax=None, *, label_col: str = "series"):
    """Forest plot: one row per group in `significance` (see
    summarize.summarize_significance), point = delta_mean (error reduction,
    positive = model better than baseline), whiskers = its bootstrap CI."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, max(3.0, 0.4 * len(significance))))
    ordered = significance.reset_index(drop=True)
    y = np.arange(len(ordered))
    lower_err = ordered["delta_mean"] - ordered["ci_low"]
    upper_err = ordered["ci_high"] - ordered["delta_mean"]
    ax.errorbar(
        ordered["delta_mean"], y, xerr=[lower_err, upper_err],
        fmt="o", color=PALETTE["model"], ecolor=PALETTE["baseline"], capsize=3,
    )
    ax.axvline(0.0, color=PALETTE["alert"], linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered[label_col].astype(str))
    ax.set_xlabel("delta (baseline_err - model_err), CI bootstrap")
    ax.set_title("Significativita' — delta di errore vs baseline")
    return ax


def plot_conformal_coverage(coverage_df: pd.DataFrame, ax=None, *, label_col: str = "series"):
    """Bar comparison: raw vs conformalized P10-P90 coverage per group (see
    conformal.evaluate_conformal_coverage), with a dashed reference line at
    the nominal 0.80."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.5))
    ordered = coverage_df.reset_index(drop=True)
    x = np.arange(len(ordered))
    width = 0.35
    ax.bar(x - width / 2, ordered["coverage_raw"], width, label="raw", color=PALETTE["baseline"])
    ax.bar(
        x + width / 2, ordered["coverage_conformal"], width,
        label="conformalized", color=PALETTE["post"],
    )
    ax.axhline(0.80, color=PALETTE["alert"], linestyle="--", linewidth=1, label="nominale (0.80)")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered[label_col].astype(str), rotation=45, ha="right")
    ax.set_ylabel("copertura P10-P90")
    ax.set_title("Conformal — copertura raw vs conformalizzata")
    ax.legend(fontsize=9)
    return ax
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_plots_smoke.py -v`
Expected: PASS (all tests, existing + 3 new;
`test_plot_calibration_curve_one_line_per_regime` — the pre-existing test
with no `ci_low`/`ci_high` columns — still passes unchanged since `has_ci`
is `False` for that fixture)

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/plots.py tests/test_plots_smoke.py
git commit -m "feat: add calibration CI band, bootstrap forest plot, conformal coverage bars"
```

---

### Task 9: `scripts/08_exp_inference.py` — significance, panel bootstrap, conformal artifacts

**Files:**
- Create: `scripts/08_exp_inference.py`
- Test: Create `tests/test_exp_inference_cli.py`

**Interfaces:**
- Consumes: `bootstrap.panel_paired_block_bootstrap` (Task 1),
  `summarize.compute_mase_scales`/`summarize_significance`/`summarize_calibration`
  (Tasks 3, 4 and existing), `conformal.conformalize_intervals`/
  `evaluate_conformal_coverage` (Task 5), `plots.plot_bootstrap_delta`/
  `plot_conformal_coverage` (Task 8), `manifest.write_manifest` (existing),
  `backtest.SeriesData` (existing).
- Produces: `exp_significance.parquet`, `exp_significance_panel.parquet`,
  `exp_conformal_predictions.parquet`, `exp_conformal_coverage.parquet`,
  `exp_calibration_extended.parquet`, `figures/exp_bootstrap_delta.png`,
  `figures/exp_conformal_coverage.png`, `manifests/inference-<run_id>.json`
  — none of these exist until this script is actually run (it is, in this
  plan — see Task 10's migration note).

- [ ] **Step 1: Write the failing test file**

```python
"""CLI test for scripts/08_exp_inference.py — small synthetic fixtures,
proving the script writes every artifact with run_id/manifest, no network,
no model call. Loaded via importlib, same pattern as
tests/test_fetch_data_cli.py / tests/test_exp_mtg_benchmark_cli.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "08_exp_inference.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("exp_inference_08", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inference08():
    return _load_script_module()


def _write_raw_predictions(path: Path, n_origins: int = 15, series_names=("A", "B")):
    rows = []
    rng = np.random.default_rng(0)
    for series in series_names:
        for i in range(n_origins):
            origin = 50 + i
            actual = 100.0 + i + rng.normal(scale=0.5)
            forecast = actual + rng.normal(scale=0.2)
            baseline = actual + rng.normal(scale=0.6)
            row = {
                "mode": "timesfm3_univariate",
                "transform": "identity",
                "series": series,
                "origin_index": origin,
                "target_index": origin,
                "target_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(origin)),
                "horizon_step": 1,
                "actual": actual,
                "observed": True,
                "forecast": forecast,
                "baseline_naive": baseline,
            }
            for level in range(1, 10):
                row[f"q{level * 10:02d}"] = forecast - 1.0 + level * 0.2
            rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_cache_series(path: Path, n_days: int = 100, series_names=("A", "B")):
    rows = []
    for series in series_names:
        dates = pd.date_range("2023-01-01", periods=n_days)
        values = 100.0 + np.arange(n_days) * 0.3
        for date, value in zip(dates, values, strict=True):
            rows.append({"series": series, "date": date, "value": value, "observed": True})
    pd.DataFrame(rows).to_parquet(path, index=False)


@pytest.fixture
def wired_dirs(inference08, tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    results_dir = tmp_path / "results"
    figures_dir = results_dir / "figures"
    cache_dir.mkdir()
    results_dir.mkdir()
    figures_dir.mkdir()
    monkeypatch.setattr(inference08.config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(inference08.config, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(inference08.config, "FIGURES_DIR", figures_dir)

    _write_raw_predictions(results_dir / inference08.RAW_PREDICTIONS_FILENAME)
    _write_cache_series(cache_dir / "mtg_prices.parquet")
    return results_dir


def test_missing_raw_predictions_raises(inference08, tmp_path, monkeypatch):
    monkeypatch.setattr(inference08.config, "RESULTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="run scripts/02_exp_mtg.py"):
        inference08.load_raw_predictions()


def test_main_writes_all_artifacts(wired_dirs, inference08):
    results_dir = wired_dirs
    inference08.main(["--n-boot", "20", "--min-calibration-origins", "5"])

    for name in (
        "exp_significance.parquet",
        "exp_significance_panel.parquet",
        "exp_conformal_predictions.parquet",
        "exp_conformal_coverage.parquet",
        "exp_calibration_extended.parquet",
    ):
        assert (results_dir / name).exists(), name

    for name in ("exp_bootstrap_delta.png", "exp_conformal_coverage.png"):
        assert (results_dir / "figures" / name).exists(), name

    manifest_files = list((results_dir / "manifests").glob("inference-*.json"))
    assert len(manifest_files) == 1
    payload = json.loads(manifest_files[0].read_text())
    assert payload["_meta"]["git_sha"]
    assert payload["n_boot"] == 20


def test_run_id_is_consistent_across_outputs(wired_dirs, inference08):
    results_dir = wired_dirs
    inference08.main(["--n-boot", "20", "--min-calibration-origins", "5"])

    significance = pd.read_parquet(results_dir / "exp_significance.parquet")
    conformal_predictions = pd.read_parquet(results_dir / "exp_conformal_predictions.parquet")
    assert significance["run_id"].nunique() == 1
    assert conformal_predictions["run_id"].nunique() == 1
    assert significance["run_id"].iloc[0] == conformal_predictions["run_id"].iloc[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_exp_inference_cli.py -v`
Expected: FAIL with `FileNotFoundError` (no such script) /
`ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
"""Esperimento D — inferenza statistica: bootstrap pareggiato (panel),
correzione BH, calibrazione con CI/WIS, conformal prediction causale.

Design choice: come scripts/04_exp_calibration.py, questo script NON
richiama il modello. Legge le predizioni raw gia' scritte da
scripts/02_exp_mtg.py (results/exp_mtg_raw_predictions.parquet) e la cache
locale dei prezzi (data/cache/mtg_prices.parquet, per calcolare le scale
MASE con storia pre-finestra genuina, mai inventata) -- CPU-only, nessuna
chiamata di rete, nessun opt-in richiesto.

Requires: scripts/01_fetch_data.py and scripts/02_exp_mtg.py already run.

Usage:
    uv run scripts/08_exp_inference.py
    uv run scripts/08_exp_inference.py --n-boot 2000 --alpha 0.1

Writes to results/:
    exp_significance.parquet          (DM stat/p, BH q-value, bootstrap delta+CI per series x horizon_step)
    exp_significance_panel.parquet    (panel bootstrap delta+CI across the showcase panel, per horizon_step)
    exp_conformal_predictions.parquet (raw + conformalized q10/q90, per row)
    exp_conformal_coverage.parquet    (coverage/width raw vs conformal, per series x horizon_step)
    exp_calibration_extended.parquet  (pinball/coverage/PIT + WIS + normalized interval width)
    figures/exp_bootstrap_delta.png
    figures/exp_conformal_coverage.png
    manifests/inference-<run_id>.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import secrets

import matplotlib.pyplot as plt
import pandas as pd

from tfm3lab import config, manifest, plots
from tfm3lab.backtest import SeriesData
from tfm3lab.bootstrap import panel_paired_block_bootstrap
from tfm3lab.conformal import conformalize_intervals, evaluate_conformal_coverage
from tfm3lab.summarize import compute_mase_scales, summarize_calibration, summarize_significance

MODE = "timesfm3_univariate"
TRANSFORM = "identity"
RAW_PREDICTIONS_FILENAME = "exp_mtg_raw_predictions.parquet"

plots.apply_style()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap replicates")
    parser.add_argument("--seed", type=int, default=config.SEED, help="bootstrap RNG seed")
    parser.add_argument("--ci", type=float, default=0.9, help="bootstrap/BH confidence level")
    parser.add_argument(
        "--alpha", type=float, default=0.2,
        help="conformal miscoverage target (1-alpha nominal coverage, default 0.8)",
    )
    parser.add_argument(
        "--min-calibration-origins", type=int, default=20,
        help="minimum prior origins required before a row gets conformalized",
    )
    return parser


def load_raw_predictions() -> pd.DataFrame:
    path = config.RESULTS_DIR / RAW_PREDICTIONS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/02_exp_mtg.py first")
    df = pd.read_parquet(path)
    return df[(df["mode"] == MODE) & (df["transform"] == TRANSFORM)].copy()


def load_cached_series() -> list[SeriesData]:
    """Same source file as scripts/02_exp_mtg.py's loader, but WITHOUT its
    common-length trim -- that trim exists there for multivariate stacking
    alignment, which this script never does; here each series only needs
    its OWN pre-boundary history for compute_mase_scales, so trimming
    would only needlessly shorten it."""
    path = config.CACHE_DIR / "mtg_prices.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/01_fetch_data.py first")
    df = pd.read_parquet(path)
    return [
        SeriesData(
            name=name,
            values=group.sort_values("date")["value"].to_numpy(dtype=float),
            dates=group.sort_values("date")["date"].to_numpy(),
            observed=group.sort_values("date")["observed"].to_numpy(dtype=bool),
        )
        for name, group in df.groupby("series")
    ]


def _panel_bootstrap_table(
    df: pd.DataFrame, n_boot: int, ci: float, seed: int
) -> pd.DataFrame:
    """One row per horizon_step: panel_paired_block_bootstrap across every
    series present at that horizon_step, aligned to their shared trailing
    window of origins (the showcase panel's common length may differ
    slightly from series to series if a card has a few extra/missing
    observed rows -- trimming to the shortest keeps every series'
    origin-position alignment exact, which the panel bootstrap requires)."""
    rows = []
    observed = df[df["observed"]]
    for horizon_step, group in observed.groupby("horizon_step"):
        deltas: dict[str, object] = {}
        weights: dict[str, int] = {}
        for series, sg in group.groupby("series"):
            sg = sg.sort_values("origin_index")
            model_err = (sg["actual"] - sg["forecast"]).abs().to_numpy()
            baseline_err = (sg["actual"] - sg["baseline_naive"]).abs().to_numpy()
            deltas[series] = baseline_err - model_err
            weights[series] = len(sg)
        min_len = min(len(v) for v in deltas.values())
        if min_len < int(horizon_step):
            continue
        aligned = {name: arr[-min_len:] for name, arr in deltas.items()}
        result = panel_paired_block_bootstrap(
            aligned, horizon=int(horizon_step), n_boot=n_boot, ci=ci, seed=seed, weights=weights
        )
        rows.append(
            {
                "horizon_step": int(horizon_step),
                "delta_mean": result.delta_mean,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "ci_level": result.ci_level,
                "block_size": result.block_size,
                "n_boot": result.n_boot,
                "seed": result.seed,
                "n_origins": result.n_origins,
                "n_series": len(aligned),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    df = load_raw_predictions()
    series_list = load_cached_series()
    boundary_index = int(df["origin_index"].min())
    mase_scales = compute_mase_scales(series_list, boundary_index=boundary_index)

    run_id = f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"

    significance = summarize_significance(
        df, mase_scales, n_boot=args.n_boot, ci=args.ci, seed=args.seed
    )
    significance["run_id"] = run_id
    significance.to_parquet(config.RESULTS_DIR / "exp_significance.parquet", index=False)
    print(f"Significance table: {len(significance)} rows")
    print(significance.to_string(index=False))

    panel_df = _panel_bootstrap_table(df, args.n_boot, args.ci, args.seed)
    panel_df["run_id"] = run_id
    panel_df.to_parquet(config.RESULTS_DIR / "exp_significance_panel.parquet", index=False)
    print(f"\nPanel bootstrap: {len(panel_df)} horizon_step rows")
    print(panel_df.to_string(index=False))

    calibration_extended = summarize_calibration(
        df, group_cols=("series", "horizon_step"), mase_scales=mase_scales
    )
    calibration_extended["run_id"] = run_id
    calibration_extended.to_parquet(
        config.RESULTS_DIR / "exp_calibration_extended.parquet", index=False
    )

    conformal_df = conformalize_intervals(
        df, alpha=args.alpha, min_calibration_origins=args.min_calibration_origins
    )
    conformal_df["run_id"] = run_id
    conformal_df.to_parquet(config.RESULTS_DIR / "exp_conformal_predictions.parquet", index=False)

    conformal_coverage = evaluate_conformal_coverage(conformal_df)
    conformal_coverage["run_id"] = run_id
    conformal_coverage.to_parquet(
        config.RESULTS_DIR / "exp_conformal_coverage.parquet", index=False
    )
    print("\nConformal coverage (raw vs conformalized):")
    print(conformal_coverage.to_string(index=False))

    significance_labeled = significance.copy()
    significance_labeled["label"] = (
        significance_labeled["series"] + " h=" + significance_labeled["horizon_step"].astype(str)
    )
    fig1, ax1 = plt.subplots(figsize=(7, max(3.0, 0.4 * len(significance_labeled))))
    plots.plot_bootstrap_delta(significance_labeled, ax=ax1, label_col="label")
    fig1.tight_layout()
    plots.save(fig1, "exp_bootstrap_delta")

    conformal_coverage_labeled = conformal_coverage.copy()
    conformal_coverage_labeled["label"] = (
        conformal_coverage_labeled["series"]
        + " h="
        + conformal_coverage_labeled["horizon_step"].astype(str)
    )
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    plots.plot_conformal_coverage(conformal_coverage_labeled, ax=ax2, label_col="label")
    fig2.tight_layout()
    plots.save(fig2, "exp_conformal_coverage")

    as_of = pd.Timestamp(df["target_date"].max()).date()
    payload = {
        "run_id": run_id,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "ci": args.ci,
        "alpha": args.alpha,
        "min_calibration_origins": args.min_calibration_origins,
        "outputs": [
            "exp_significance.parquet",
            "exp_significance_panel.parquet",
            "exp_conformal_predictions.parquet",
            "exp_conformal_coverage.parquet",
            "exp_calibration_extended.parquet",
            "figures/exp_bootstrap_delta.png",
            "figures/exp_conformal_coverage.png",
        ],
    }
    manifest.write_manifest(
        payload, config.RESULTS_DIR / "manifests" / f"inference-{run_id}.json", as_of=as_of
    )
    print(f"\nWrote inference artifacts to {config.RESULTS_DIR} (run_id={run_id})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_exp_inference_cli.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/08_exp_inference.py tests/test_exp_inference_cli.py
git commit -m "feat: add scripts/08_exp_inference.py for significance/conformal artifacts"
```

---

### Task 10: Documentation — `docs/analysis-plan.md` and `README.md`

**Files:**
- Modify: `docs/analysis-plan.md`
- Modify: `README.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Update `docs/analysis-plan.md`'s Uncertainty section**

Replace the existing `Uncertainty:` block:

```
Uncertainty:
  paired moving-block bootstrap;
  block length >= horizon;
  correzione BH per p-value multipli.
```

with:

```
Uncertainty:
  paired moving-block bootstrap sul delta di errore (baseline - model),
  pairing preservato per origine, block length >= horizon (verificato,
  non solo documentato) -- tfm3lab.bootstrap.paired_moving_block_bootstrap;
  supporto panel (piu' carte, stesso blocco di origini ricampionato per
  tutte insieme, media pesata per n) -- tfm3lab.bootstrap.panel_paired_block_bootstrap;
  seed sempre esplicito, sempre riportato nel risultato -- mai una
  statistica non riproducibile;
  Diebold-Mariano corretto con Benjamini-Hochberg sulle combinazioni
  carta x horizon di UNA comparazione (mai su comparazioni diverse
  insieme) -- tfm3lab.summarize.apply_bh_correction/summarize_significance;
  headline = effect size (delta_mean / skill) + CI bootstrap, MAI il
  p-value/q-value da solo -- vedi tfm3lab.summarize.summarize_significance;
  curva di calibrazione nominale vs empirica con CI binomiale esatta
  (Clopper-Pearson) -- tfm3lab.metrics.binomial_ci, applicata sia alla
  curva (scripts/04_exp_calibration.py) sia al quantile-bin (tfm3lab.figdata.quantile_bin_calibration);
  weighted interval score (variante documentata, Bracher et al. 2021) e
  ampiezza dell'intervallo normalizzata sulla stessa scala MASE --
  tfm3lab.metrics.weighted_interval_score/interval_width_normalized;
  conformal prediction split-conformal/CQR, causale (solo origini
  strettamente precedenti, mai target futuro, mai riga non osservata) --
  tfm3lab.conformal.conformalize_intervals -- NON e' piu' zero-shot puro:
  raw e conformalized sono riportati sempre separati, mai fusi in un solo
  numero. Nessun run reale eseguito su GPU per questi artifact --
  scripts/08_exp_inference.py e' CPU-only e legge solo predizioni gia'
  scritte, quindi E' stato eseguito in questo branch (a differenza della
  griglia di benchmark GPU-bound di scripts/02b, che resta non eseguita).
```

- [ ] **Step 2: Update `README.md`**

Add one paragraph after the existing description of
`scripts/02b_exp_mtg_benchmark.py` in the "What's here" section (find that
paragraph and insert immediately after it):

```markdown
`scripts/08_exp_inference.py` adds the inferential layer on top of
`02_exp_mtg.py`'s raw predictions: a paired (and panel) moving-block
bootstrap CI on the model-vs-baseline error delta, Diebold-Mariano
p-values with a Benjamini-Hochberg correction across card x horizon
combinations, a calibration table extended with WIS and normalized
interval width, and a causally-valid conformal (split-conformal/CQR)
post-processing of the P10-P90 interval — raw and conformalized results
kept strictly separate, since the conformalized numbers are no longer a
zero-shot result. CPU-only, no network, no GPU — unlike `02b`'s benchmark
grid, this script *was* run in this branch (see
`docs/analysis-plan.md`'s Uncertainty section for every function name).
```

In "Running it", add after the existing `02b_exp_mtg_benchmark.py` line:

```
uv run scripts/08_exp_inference.py
```

In the `src/tfm3lab/` project-layout tree, add two lines (alphabetically
placed alongside the existing `benchmark.py`/`benchmark_config.py` entries):

```
bootstrap.py       — paired/panel moving-block bootstrap for the error delta
conformal.py        — causally-valid conformal (split-conformal/CQR) post-processing
```

Change the `scripts/` project-layout line's script-count description to
include `08` (e.g. `00-08` in place of the current upper bound).

- [ ] **Step 3: Commit**

```bash
git add docs/analysis-plan.md README.md
git commit -m "docs: describe bootstrap/BH/conformal inference layer in analysis-plan and README"
```
