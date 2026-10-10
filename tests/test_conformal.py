"""Offline tests for tfm3lab.conformal (CQR, Preregistration v2 section 4)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tfm3lab.conformal import (
    apply_cqr,
    apply_cqr_thresholds,
    cqr_scores,
    cqr_threshold,
    fit_cqr_thresholds,
)


def test_cqr_scores_hand_computed():
    # inside (negative), below lo, above hi
    got = cqr_scores([5.0, 0.0, 12.0], [2.0, 2.0, 2.0], [10.0, 10.0, 10.0])
    np.testing.assert_allclose(got, [-3.0, 2.0, 2.0])


def test_cqr_threshold_finite_sample_rank():
    # n = 9 scores 1..9; rank = ceil(10 * 0.8) = 8 -> 8th smallest = 8
    y = np.arange(1, 10, dtype=float)
    lo = np.zeros(9)
    hi = np.zeros(9)  # score = y - 0 = y
    assert cqr_threshold(y, lo, hi, coverage=0.8) == 8.0
    # n = 3: rank = ceil(4 * 0.8) = 4 > 3 -> infinite threshold
    assert cqr_threshold(y[:3], lo[:3], hi[:3]) == math.inf


def test_cqr_threshold_rejects_empty_and_nonfinite():
    with pytest.raises(ValueError):
        cqr_threshold([], [], [])
    with pytest.raises(ValueError):
        cqr_threshold([np.nan, 1.0], [0.0, 0.0], [1.0, 1.0])


def test_apply_widens_and_narrows_both_bounds():
    lo, hi = apply_cqr([1.0, 2.0], [3.0, 6.0], 0.5)
    np.testing.assert_allclose(lo, [0.5, 1.5])
    np.testing.assert_allclose(hi, [3.5, 6.5])
    lo, hi = apply_cqr([1.0], [3.0], -0.5)
    np.testing.assert_allclose(lo, [1.5])
    np.testing.assert_allclose(hi, [2.5])


def _frame(seed, n=60):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "horizon_step": np.repeat([1, 7], n // 2),
            "actual": rng.normal(size=n),
            "q10": -1.0,
            "q90": 1.0,
        }
    )


def test_changing_evaluation_rows_cannot_change_threshold():
    cal = _frame(0)
    ev_a = _frame(1)
    ev_b = ev_a.assign(actual=ev_a["actual"] * 100.0 + 50.0)
    thr = fit_cqr_thresholds(cal)
    out_a = apply_cqr_thresholds(ev_a, thr)
    out_b = apply_cqr_thresholds(ev_b, thr)
    np.testing.assert_array_equal(out_a["cqr_threshold"], out_b["cqr_threshold"])
    np.testing.assert_array_equal(out_a["q10_cqr"], out_b["q10_cqr"])
    # refitting on calibration rows alone is unaffected by evaluation data
    pd.testing.assert_frame_equal(thr, fit_cqr_thresholds(cal.copy()))
    # and changing CALIBRATION rows does change it
    assert not thr.equals(fit_cqr_thresholds(cal.assign(actual=cal["actual"] * 3)))


def test_fit_and_apply_per_group():
    cal = _frame(2)
    thr = fit_cqr_thresholds(cal)
    assert list(thr["horizon_step"]) == [1, 7]
    assert list(thr["calibration_n"]) == [30, 30]
    for h, grp in cal.groupby("horizon_step"):
        want = cqr_threshold(grp["actual"], grp["q10"], grp["q90"])
        assert thr.loc[thr["horizon_step"] == h, "cqr_threshold"].item() == want
    ev = _frame(3).set_index(pd.RangeIndex(100, 160))
    out = apply_cqr_thresholds(ev, thr)
    assert (out.index == ev.index).all()
    np.testing.assert_array_equal(out["q10"], ev["q10"])  # raw untouched
    np.testing.assert_allclose(out["q90_cqr"], ev["q90"] + out["cqr_threshold"])
    missing = apply_cqr_thresholds(ev.assign(horizon_step=28), thr)
    assert missing["q10_cqr"].isna().all()
