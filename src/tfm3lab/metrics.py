"""Forecast accuracy, calibration, and significance — the checks that let us
say "beats naive" instead of just "has a smaller number".

Every function takes plain numpy arrays and returns a plain float or array;
grouping by card/series/origin/horizon-step is the caller's job (usually a
pandas groupby in an experiment script). Keeping these primitives ignorant
of the DataFrame layout makes them easy to unit-test against hand-computed
cases, which is the point: the draft notebook this project replaces reported
`BeatNaive_%` — a per-observation win rate — as if it were an accuracy
metric, which it isn't (see `relative_mae`/`mase` below for why).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mae(actual, predicted) -> float:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted) -> float:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def smape(actual, predicted) -> float:
    """Symmetric MAPE, percent, 0-200 scale (200 = maximally wrong)."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    denom = np.abs(actual) + np.abs(predicted)
    denom = np.where(denom == 0, 1e-8, denom)
    return float(np.mean(200.0 * np.abs(actual - predicted) / denom))


def in_sample_scale(history: np.ndarray, seasonality: int = 1) -> float:
    """Mean absolute seasonal difference of `history` — MASE's denominator.

    `history` must be data observed *before* the test period (e.g. the
    series up to the first test origin's context start) — using data from
    inside or after the test window would leak information into the scale
    and understate MASE for a genuinely good model. `seasonality=1` gives
    the ordinary (non-seasonal) naive scale; use e.g. 7 for a weekly effect.
    """
    history = np.asarray(history, dtype=float)
    if len(history) <= seasonality:
        raise ValueError(f"history too short ({len(history)}) for seasonality={seasonality}")
    diffs = np.abs(history[seasonality:] - history[:-seasonality])
    scale = float(np.mean(diffs))
    return scale if scale > 1e-12 else 1e-12  # guard against a near-constant history


def mase(abs_errors, scale) -> float:
    """Mean absolute scaled error.

    `scale` is normally a single float from `in_sample_scale` (the standard
    Hyndman-Koehler definition: one scale per series). It may also be an
    array broadcastable against `abs_errors` for a locally-scaled variant —
    that changes what the number means (see docs/talk-outline.md) and
    should be a deliberate, documented choice at the call site, not a
    silent default.
    """
    abs_errors = np.asarray(abs_errors, dtype=float)
    return float(np.mean(abs_errors / np.asarray(scale, dtype=float)))


def relative_mae(mae_model: float, mae_baseline: float) -> float:
    """MAE_model / MAE_baseline. Below 1 means the model beats the baseline.

    Use this (or MASE) instead of a per-observation win rate: a model can
    win 60% of individual observations and still have worse aggregate MAE
    if it loses badly on the other 40% (fat tails, e.g. around a shock).
    """
    if mae_baseline <= 0:
        raise ValueError(f"mae_baseline must be positive, got {mae_baseline}")
    return mae_model / mae_baseline


def pinball_loss(actual, predicted, quantile: float) -> float:
    """Pinball (quantile) loss at one quantile level in (0, 1)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    diff = actual - predicted
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def pinball_loss_multi(actual, quantile_forecasts, levels) -> float:
    """Average pinball loss across all quantile levels.

    `quantile_forecasts` has the quantile axis last, shape (..., len(levels));
    `actual` broadcasts against the leading dimensions.
    """
    quantile_forecasts = np.asarray(quantile_forecasts, dtype=float)
    losses = [pinball_loss(actual, quantile_forecasts[..., i], q) for i, q in enumerate(levels)]
    return float(np.mean(losses))


def coverage(actual, lower, upper) -> float:
    """Fraction of `actual` inside [lower, upper], inclusive on both ends."""
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((actual >= lower) & (actual <= upper)))


def pit_values(actual, quantile_forecasts, levels) -> np.ndarray:
    """Approximate probability integral transform via linear interpolation of
    the empirical CDF implied by the quantile forecasts.

    This is an approximation with only `len(levels)` (9) known points:
    values outside the lowest/highest forecast quantile are clipped to that
    quantile's level rather than extrapolated. A calibration histogram built
    from this will therefore never show mass exactly at 0.0 or 1.0 even for
    a genuinely miscalibrated tail — that's a limitation of the method to
    disclose, not evidence the tails are fine.
    """
    actual = np.atleast_1d(np.asarray(actual, dtype=float))
    quantile_forecasts = np.asarray(quantile_forecasts, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if quantile_forecasts.ndim == 1:
        quantile_forecasts = quantile_forecasts[None, :]
    out = np.empty(actual.shape[0], dtype=float)
    for i in range(actual.shape[0]):
        q_vals = quantile_forecasts[i]
        order = np.argsort(q_vals)  # defensive: output is normally pre-sorted
        out[i] = np.interp(actual[i], q_vals[order], levels[order])
    return np.clip(out, levels.min(), levels.max())


def diebold_mariano(loss1, loss2, horizon: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample
    correction, for whether model 1's average loss differs from model 2's.

    `loss1`/`loss2` are one loss value per forecast ORIGIN at a fixed
    horizon step (e.g. absolute error) — not averaged across horizon steps
    within an origin, and not per-individual-step across mixed horizons.
    `horizon` is that fixed horizon step h: h-step-ahead rolling forecasts
    have MA(h-1) autocorrelation in the loss differential, which the test's
    long-run variance estimate must account for.

    Returns (statistic, two_sided_p_value). A negative statistic with a
    small p-value means model 1 has significantly lower average loss.
    Raises ValueError with fewer than 2*horizon observations — below that
    the long-run variance estimate is not meaningful.
    """
    loss1 = np.asarray(loss1, dtype=float)
    loss2 = np.asarray(loss2, dtype=float)
    if loss1.shape != loss2.shape:
        raise ValueError(f"shape mismatch: {loss1.shape} vs {loss2.shape}")
    d = loss1 - loss2
    t = len(d)
    if t < 2 * horizon:
        raise ValueError(f"need at least {2 * horizon} observations, got {t}")
    dbar = float(np.mean(d))
    max_lag = horizon - 1
    # Long-run variance: gamma_0 + 2 * sum_{k=1}^{h-1} gamma_k, each gamma_k
    # normalized by T (not T-k) — matches Diebold & Mariano (1995) and the
    # widely-used R `forecast::dm.test` reference implementation.
    gamma0 = float(np.sum((d - dbar) ** 2) / t)
    lrv = gamma0
    for lag in range(1, max_lag + 1):
        gamma_k = float(np.sum((d[lag:] - dbar) * (d[:-lag] - dbar)) / t)
        lrv += 2 * gamma_k
    var_dbar = lrv / t
    if var_dbar <= 0:
        return 0.0, 1.0  # no variation in the loss differential: nothing to test
    dm_stat = dbar / np.sqrt(var_dbar)
    correction = np.sqrt((t + 1 - 2 * horizon + horizon * (horizon - 1) / t) / t)
    dm_adj = dm_stat * correction
    p_value = float(2 * (1 - stats.t.cdf(np.abs(dm_adj), df=t - 1)))
    return float(dm_adj), p_value


def block_bootstrap_ci(
    values,
    block_size: int,
    n_boot: int = 1000,
    ci: float = 0.9,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Moving-block bootstrap confidence interval for the mean of `values`.

    Rolling-origin errors overlap (neighboring origins share most of their
    context), so an i.i.d. bootstrap understates the true variance. Block
    resampling preserves local autocorrelation instead. `block_size` should
    be at least the forecast horizon. Pass `rng` for reproducible tests;
    omit it for genuine randomness in real analysis.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if not (1 <= block_size <= n):
        raise ValueError(f"block_size must be in [1, {n}], got {block_size}")
    if rng is None:
        rng = np.random.default_rng()
    n_blocks = int(np.ceil(n / block_size))
    starts = np.arange(0, n - block_size + 1)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([values[s : s + block_size] for s in chosen])[:n]
        means[b] = sample.mean()
    alpha = 1 - ci
    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)
