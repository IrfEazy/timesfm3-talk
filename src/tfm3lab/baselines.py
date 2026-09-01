"""Baseline forecasters: naive, seasonal naive, drift, ETS.

Every experiment compares TimesFM-3 against these. An accuracy number with
no baseline attached says nothing — on daily financial/collectible prices
the naive ("tomorrow = today") baseline is notoriously hard to beat, so its
absence from the original draft (finding #4 in the project plan) would have
made every result impossible to interpret.
"""

from __future__ import annotations

import numpy as np


def naive_forecast(context, horizon: int) -> np.ndarray:
    """Repeat the last observed value for the whole horizon."""
    context = np.asarray(context, dtype=float)
    if len(context) == 0:
        raise ValueError("context must be non-empty")
    return np.full(horizon, context[-1], dtype=float)


def seasonal_naive_forecast(context, horizon: int, season_length: int) -> np.ndarray:
    """Repeat the last full season, cycling if horizon > season_length."""
    context = np.asarray(context, dtype=float)
    if season_length < 1:
        raise ValueError(f"season_length must be >= 1, got {season_length}")
    if len(context) < season_length:
        raise ValueError(
            f"context (len {len(context)}) shorter than season_length ({season_length})"
        )
    last_season = context[-season_length:]
    reps = int(np.ceil(horizon / season_length))
    return np.tile(last_season, reps)[:horizon]


def drift_forecast(context, horizon: int) -> np.ndarray:
    """Extrapolate the average per-step change across the whole context
    (the line through the first and last context points)."""
    context = np.asarray(context, dtype=float)
    if len(context) < 2:
        raise ValueError("drift forecast needs at least 2 context points")
    slope = (context[-1] - context[0]) / (len(context) - 1)
    steps = np.arange(1, horizon + 1)
    return context[-1] + slope * steps


def ets_forecast(context, horizon: int, seasonal_periods: int | None = None) -> np.ndarray:
    """Exponential smoothing (Holt-Winters) point forecast via statsmodels.

    Seasonal component is only fit when the context holds at least two full
    seasons; otherwise falls back to trend-only smoothing. Raises whatever
    statsmodels raises on a degenerate context (all-constant, too short) —
    the caller decides whether that counts as a skip or a hard failure, this
    function does not silently substitute the naive forecast.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    context = np.asarray(context, dtype=float)
    if len(context) < 4:
        raise ValueError(f"ETS needs at least 4 context points, got {len(context)}")
    kwargs = {}
    if seasonal_periods and len(context) >= 2 * seasonal_periods:
        kwargs = {"seasonal": "add", "seasonal_periods": seasonal_periods}
    model = ExponentialSmoothing(context, trend="add", **kwargs).fit(optimized=True)
    return np.asarray(model.forecast(horizon), dtype=float)
