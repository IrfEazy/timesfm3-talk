import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tfm3lab import shock
from tfm3lab.data.market import detect_trailing_shock_days, trailing_shock_z


def _walk(n=600, seed=0, scale=0.005):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(scale=scale, size=n)))


def test_detector_has_no_lookahead():
    v = _walk()
    v2 = v.copy()
    v2[400:] *= 1.5  # change the future after position 399
    v2[450:] *= 0.7
    z1, z2 = trailing_shock_z(v), trailing_shock_z(v2)
    assert np.array_equal(np.isnan(z1[:400]), np.isnan(z2[:400]))
    assert np.allclose(z1[:400], z2[:400], equal_nan=True)
    p1, _ = detect_trailing_shock_days(v)
    p2, _ = detect_trailing_shock_days(v2)
    assert list(p1[p1 < 400]) == list(p2[p2 < 400])


def test_detector_flags_spike_and_needs_full_window():
    v = _walk()
    v[400:] *= 1.3
    pos, z = detect_trailing_shock_days(v)
    assert 400 in pos and z[list(pos).index(400)] > 4
    assert np.isnan(trailing_shock_z(v)[:250]).all()


def _days(rows):
    return pd.DataFrame(rows, columns=["date", "index", "z"])


def test_decluster_merges_within_20_days_and_picks_onset_index():
    cal = pd.bdate_range("2020-01-01", periods=200)
    rows = [
        (cal[10], "Gold", 4.5),
        (cal[10], "SP500", -6.0),
        (cal[25], "Nikkei225", 4.2),  # 15 days after previous -> merged
        (cal[60], "Gold", 4.1),  # 35 days later -> new event
    ]
    ev = shock.decluster(_days(rows), cal)
    assert len(ev) == 2
    assert ev[0]["onset_date"] == cal[10].date()
    assert ev[0]["onset_index"] == "SP500" and ev[0]["max_abs_z"] == 6.0
    assert ev[0]["triggering_indices"] == ("Gold", "Nikkei225", "SP500")
    assert ev[1]["onset_index"] == "Gold"


def test_decluster_gap_boundary():
    cal = pd.bdate_range("2020-01-01", periods=100)
    near = shock.decluster(_days([(cal[0], "A", 4.1), (cal[20], "B", 4.1)]), cal)
    far = shock.decluster(_days([(cal[0], "A", 4.1), (cal[21], "B", 4.1)]), cal)
    assert len(near) == 1 and len(far) == 2


def test_decluster_empty():
    assert shock.decluster(_days([]), pd.bdate_range("2020-01-01", periods=5)) == []


def test_arms_and_guard_band():
    assert shock.assign_arm(dt.date(2023, 5, 31)) == "pre"
    assert shock.assign_arm(dt.date(2023, 6, 1)) == "guard"
    assert shock.assign_arm(dt.date(2024, 5, 31)) == "guard"
    assert shock.assign_arm(dt.date(2024, 6, 1)) == "post"


def test_window_fits():
    assert shock.window_fits(300, 400)
    assert not shock.window_fits(200, 400)  # context + before do not fit
    assert not shock.window_fits(380, 400)  # 25 post points do not fit
    assert shock.window_fits(375, 400)


def _errs(pre, post):
    off = np.arange(-len(pre), len(post))
    return off, np.array(list(pre) + list(post), dtype=float)


def test_lag_recovery_and_threshold():
    off, e = _errs([1, 1, 1], [9, 9, 1.4, 1.4, 1.4] + [9] * 20)
    assert shock.error_recovery_lag(off, e, 1.5) == (2.0, False)
    # at 1.25 the 1.4 values exceed the threshold
    assert shock.error_recovery_lag(off, e, 1.25) == (25.0, True)


def test_never_recovered_is_censored_not_dropped():
    off, e = _errs([1, 1, 1], [9] * 25)
    assert shock.error_recovery_lag(off, e, 1.5) == (25.0, True)


def test_lag_needs_three_consecutive_and_ignores_post_window():
    off, e = _errs([1, 1], [1, 1, 9, 1, 1, 9] + [9] * 19 + [1, 1, 1])
    assert shock.error_recovery_lag(off, e, 1.5) == (25.0, True)  # run is past the window


def test_lag_nan_cases():
    off, e = _errs([np.nan, np.nan], [1.0] * 25)
    lag, cens = shock.error_recovery_lag(off, e, 1.5)
    assert np.isnan(lag) and not cens


def test_baseline_relative_lag_arithmetic():
    off, model = _errs([1, 1], [9, 1, 1, 1] + [1] * 21)
    _, naive = _errs([1, 1], [9, 9, 9, 1, 1, 1] + [1] * 19)
    d = shock.error_recovery_lag(off, model)[0] - shock.error_recovery_lag(off, naive)[0]
    assert d == 1 - 3


def test_anticipation_hit():
    assert shock.anticipation_hit(101, 100, 0.5)
    assert shock.anticipation_hit(99, 100, -3)
    assert not shock.anticipation_hit(101, 100, -1)
    assert not shock.anticipation_hit(100, 100, 1)  # flat forecast


def test_cluster_difference_ci_deterministic_and_covers_effect():
    pre, post = [1.0, 2, 3, 4, 5, 6, 7, 8], [11.0, 12, 13, 14, 15, 16, 17, 18]
    a = shock.cluster_difference_ci(pre, post)
    b = shock.cluster_difference_ci(pre, post)
    assert a == b and a["effect"] == 10 and a["ci_low"] < 10 < a["ci_high"]
    assert a["ci_low"] > 0
    empty = shock.cluster_difference_ci([], post)
    assert np.isnan(empty["effect"]) and empty["n_pre"] == 0


def test_cluster_difference_ci_pre_minus_post_sign():
    pre, post = [0.8, 0.9, 0.7, 0.8], [0.4, 0.5, 0.5, 0.6]
    a = shock.cluster_difference_ci(pre, post, effect_definition="pre_minus_post")
    d = shock.cluster_difference_ci(pre, post)
    assert a["effect_definition"] == "pre_minus_post" and d["effect_definition"] == "post_minus_pre"
    assert a["effect"] > 0 and np.isclose(a["effect"], -d["effect"])
    assert a["ci_low"] > 0 and np.isclose(a["ci_low"], -d["ci_high"])
    with pytest.raises(ValueError):
        shock.cluster_difference_ci(pre, post, effect_definition="x")


def test_noise_sigma_and_determinism():
    ctx = _walk(300)
    sigma = 0.25 * np.std(np.diff(np.log(ctx)), ddof=1)
    draws = np.array([np.log(shock.add_log_noise(ctx, s)) - np.log(ctx) for s in range(300)])
    assert abs(draws.std() - sigma) / sigma < 0.05
    assert np.array_equal(shock.add_log_noise(ctx, 3), shock.add_log_noise(ctx, 3))
    assert not np.array_equal(shock.add_log_noise(ctx, 3), shock.add_log_noise(ctx, 4))


def test_splice_keeps_last_32_joins_continuously_and_is_deterministic():
    hist = _walk(900)
    origin, clen = 700, 256
    ctx = hist[origin - clen : origin]
    sp = shock.splice_context(hist, origin, clen, seed=1)
    assert len(sp) == clen
    assert np.array_equal(sp[-32:], ctx[-32:])
    assert sp[clen - 33] == pytest.approx(ctx[clen - 33])  # joins at the replaced block's end
    assert not np.allclose(sp[:-32], ctx[:-32])
    assert np.array_equal(sp, shock.splice_context(hist, origin, clen, seed=1))
    assert not np.array_equal(sp, shock.splice_context(hist, origin, clen, seed=2))


def test_splice_uses_only_old_enough_history():
    hist = _walk(900)
    # Poison everything within one year of the origin: result must not see it.
    origin = 700
    poisoned = hist.copy()
    poisoned[origin - 251 :] = 1e9
    a = shock.splice_context(hist, origin, 256, seed=5)
    b = shock.splice_context(poisoned, origin, 256, seed=5)
    assert np.allclose(np.diff(np.log(a[:-32])), np.diff(np.log(b[:-32])))  # level-shift invariant
    assert shock.splice_context(hist, 300, 256, seed=1) is None  # too little history


def test_promotion_rule_branches():
    ok = dict(n_pre=8, n_post=8, effect=2.0, ci_low=0.5, ci_high=3.0)
    assert shock.promotion_verdict(**ok, perturbed_effects={"noise": 1.0, "splice": 0.1}) == (
        "promoted",
        [],
    )
    v, r = shock.promotion_verdict(
        **{**ok, "n_post": 7}, perturbed_effects={"noise": 1, "splice": 1}
    )
    assert v == "appendix" and "fewer than 8" in r[0]
    v, r = shock.promotion_verdict(
        **{**ok, "ci_low": -0.1}, perturbed_effects={"noise": 1, "splice": 1}
    )
    assert v == "appendix" and any("CI" in x for x in r)
    v, r = shock.promotion_verdict(**ok, perturbed_effects={"noise": 1, "splice": -1})
    assert v == "appendix" and any("sign" in x for x in r)
    v, r = shock.promotion_verdict(**ok, perturbed_effects={"noise": 1})
    assert v == "appendix" and any("unavailable" in x for x in r)


def test_normalized_rows_have_no_raw_prices():
    df = shock.normalized_rows([10, 10], [1, 2], [105.0, 110.0], [100.0, 100.0], [100.0, 100.0], 10)
    assert df["actual_norm"].tolist() == [1.05, 1.10]
    assert df["ape"].iloc[0] == pytest.approx(abs(1.0 - 1.05) / 1.05)
    assert df["origin_offset"].tolist() == [0, 0]
