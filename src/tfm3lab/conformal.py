"""Conformalized quantile regression (CQR) recalibration of P10-P90
intervals (Preregistration v2 section 4).

Split design: a threshold is fitted on CALIBRATION rows only and then
applied to EVALUATION rows; evaluation outcomes never enter the fit. The
conformity score is max(q_lo - y, y - q_hi) and the threshold is the
ceil((n+1) * coverage) / n empirical quantile of the calibration scores,
i.e. the ceil((n+1) * coverage)-th smallest score (+inf if that rank
exceeds n). Both bounds move by the threshold: a positive threshold widens
the interval, a negative one narrows it (no clamping at 0).

This is a post-hoc wrapper calibrated from the model's own past errors, not
a zero-shot result; report raw and recalibrated intervals side by side.
Regime results are empirical subgroup evaluations, not conformal
guarantees: dependent rolling forecasts break exchangeability.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

DEFAULT_COVERAGE = 0.8


def cqr_scores(actual, q_lo, q_hi) -> np.ndarray:
    """CQR conformity score per row: max(q_lo - y, y - q_hi)."""
    y = np.asarray(actual, dtype=float)
    lo = np.asarray(q_lo, dtype=float)
    hi = np.asarray(q_hi, dtype=float)
    return np.maximum(lo - y, y - hi)


def cqr_threshold(actual, q_lo, q_hi, coverage: float = DEFAULT_COVERAGE) -> float:
    """Finite-sample CQR threshold from calibration rows.

    Raises ValueError if any calibration value is non-finite (filter
    unobserved rows first) or there are no calibration rows. Returns +inf
    when ceil((n+1) * coverage) > n (too few rows for the requested level).
    """
    if not 0 < coverage < 1:
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")
    scores = cqr_scores(actual, q_lo, q_hi).ravel()
    if scores.size == 0:
        raise ValueError("no calibration rows")
    if not np.all(np.isfinite(scores)):
        raise ValueError("calibration rows contain non-finite values")
    n = scores.size
    rank = math.ceil((n + 1) * coverage)
    if rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


def apply_cqr(q_lo, q_hi, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """(q_lo - threshold, q_hi + threshold): widens for threshold > 0,
    narrows for threshold < 0. A strongly negative threshold can invert a
    narrow interval; that is reported, not repaired."""
    lo = np.asarray(q_lo, dtype=float)
    hi = np.asarray(q_hi, dtype=float)
    return lo - threshold, hi + threshold


def fit_cqr_thresholds(
    calibration: pd.DataFrame,
    group_cols: tuple[str, ...] = ("horizon_step",),
    actual_col: str = "actual",
    lo_col: str = "q10",
    hi_col: str = "q90",
    coverage: float = DEFAULT_COVERAGE,
) -> pd.DataFrame:
    """One threshold per group, from calibration rows only.

    Returns columns group_cols + ["cqr_threshold", "calibration_n"]. Pass a
    frame that already contains ONLY calibration rows (and only observed
    targets); this function never sees evaluation data.
    """
    rows = []
    for keys, grp in calibration.groupby(list(group_cols), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys, strict=True))
        row["cqr_threshold"] = cqr_threshold(
            grp[actual_col].to_numpy(), grp[lo_col].to_numpy(), grp[hi_col].to_numpy(), coverage
        )
        row["calibration_n"] = len(grp)
        rows.append(row)
    return pd.DataFrame(rows, columns=[*group_cols, "cqr_threshold", "calibration_n"])


def apply_cqr_thresholds(
    evaluation: pd.DataFrame,
    thresholds: pd.DataFrame,
    group_cols: tuple[str, ...] = ("horizon_step",),
    lo_col: str = "q10",
    hi_col: str = "q90",
) -> pd.DataFrame:
    """Returns a copy of `evaluation` with `cqr_threshold`, `calibration_n`,
    `{lo_col}_cqr` and `{hi_col}_cqr`; raw columns are untouched. Groups
    without a fitted threshold get NaN recalibrated bounds."""
    merged = evaluation.merge(thresholds, on=list(group_cols), how="left", validate="m:1")
    merged.index = evaluation.index
    lo, hi = apply_cqr(
        merged[lo_col].to_numpy(), merged[hi_col].to_numpy(), merged["cqr_threshold"].to_numpy()
    )
    merged[f"{lo_col}_cqr"] = lo
    merged[f"{hi_col}_cqr"] = hi
    return merged
