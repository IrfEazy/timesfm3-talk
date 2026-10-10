"""Per-series error scaling for committed (price-free) outputs."""

from __future__ import annotations

import numpy as np


def series_error_scale(series) -> tuple[float, bool]:
    """One positive constant per series: the median of its observed values
    over the whole loaded window. Returns (scale, valid); an empty window or
    a NaN / non-positive median gives (nan, False).

    Dividing absolute errors by this constant (the same for every origin,
    cell and horizon) leaves per-series ratios of loss sums equal to the
    raw-error ratios of Preregistration v2 s1, so skill is preserved exactly.
    The constant is computed from the input data and never written to any
    output (it would reveal the price level)."""
    values = np.asarray(series.values, dtype=float)
    observed = np.asarray(series.observed, dtype=bool)
    vals = values[observed & np.isfinite(values)]
    if vals.size == 0:
        return float("nan"), False
    scale = float(np.median(vals))
    if not np.isfinite(scale) or scale <= 0:
        return float("nan"), False
    return scale, True
