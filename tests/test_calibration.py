import numpy as np
import pandas as pd

from tfm3lab import calibration as cal


def _mtg_primary(n_days=60, start="2025-04-01", cards=("a", "b")):
    rows = []
    for c in cards:
        for h in (1, 7, 28, 56, 3):
            for d in pd.date_range(start, periods=n_days):
                rows.append(
                    {
                        "mode": "timesfm3_univariate",
                        "card": c,
                        "origin_date": d,
                        "target_date": d + pd.Timedelta(days=h),
                        "requested_horizon": h,
                        "observed": True,
                        "scale_valid": True,
                        "y_norm": 1.0,
                        "q10_norm": 0.9,
                        "q50_norm": 1.0,
                        "q90_norm": 1.1,
                    }
                )
    return pd.DataFrame(rows)


def _shock_pred(offsets=range(-5, 6), arms=(("e1", "pre"), ("e2", "post"))):
    rows = []
    for ev, arm in arms:
        for o in offsets:
            for h in (1, 7, 28, 5):
                d = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(o) + 30)
                rows.append(
                    {
                        "event_id": ev,
                        "arm": arm,
                        "forecaster": "timesfm3_univariate",
                        "perturbation": "none",
                        "origin_offset": o,
                        "horizon_step": h,
                        "actual_norm": 1.0,
                        "forecast_norm": 1.0,
                        "ape": 0.0,
                        "q10_norm": 0.9,
                        "q50_norm": 1.0,
                        "q90_norm": 1.1,
                        "origin_date": d,
                        "target_date": d + pd.Timedelta(days=h),
                    }
                )
    rows.append({**rows[0], "forecaster": "naive", "q10_norm": np.nan})
    rows.append({**rows[0], "perturbation": "noise"})
    return pd.DataFrame(rows)


def test_mtg_frame_filters_horizons_and_unobserved():
    p = _mtg_primary()
    p.loc[0, "observed"] = False
    f = cal.mtg_frame(p)
    assert set(f["horizon"]) == {1, 7, 28, 56}
    assert (f["regime"] == "mtg").all() and len(f) == len(p) * 4 // 5 - 1


def test_market_regime_assignment_and_filters():
    f = cal.market_frame(_shock_pred())
    assert set(f["horizon"]) == {1, 7, 28}
    shock = f[f["regime"] == "market_shock"]
    assert shock["origin_date"].notna().all() and len(shock) > 0
    assert shock["target_offset"].abs().max() == 3
    non = f[f["regime"] == "event_window_nonshock"]
    assert (non["target_offset"].abs() > 3).all()
    assert len(f) == 2 * 11 * 3  # naive and perturbed rows excluded


def test_regime_uses_target_offset_for_long_horizons():
    f = cal.market_frame(_shock_pred(offsets=range(-10, 11)))
    # target_offset = origin_offset + horizon_step - 1
    h7 = f[f["horizon"] == 7]
    assert (
        h7["target_offset"] == (h7["origin_date"] - pd.Timestamp("2020-01-31")).dt.days + 6
    ).all()
    at_onset = h7[h7["origin_date"] == pd.Timestamp("2020-01-31")]  # origin_offset 0
    assert (at_onset["target_offset"] == 6).all()
    assert (at_onset["regime"] == "event_window_nonshock").all()
    # origin 6 days before onset at h=7 targets the onset day
    early = h7[h7["origin_date"] == pd.Timestamp("2020-01-25")]
    assert (early["target_offset"] == 0).all() and (early["regime"] == "market_shock").all()
    h1 = f[f["horizon"] == 1]
    assert (h1["target_offset"] == (h1["origin_date"] - pd.Timestamp("2020-01-31")).dt.days).all()
    assert ((f["target_offset"].abs() <= 3) == (f["regime"] == "market_shock")).all()


def test_insufficient_calibration_is_undetermined_not_false():
    assert cal.conformal_fixes_it(float("nan"), float("nan")) is None
    t = cal.conformal_table(cal.mtg_frame(_mtg_primary(start="2025-07-01")), n_boot=50)
    t = t[t["analysis"] == "preregistered"]
    r = t[(t["regime"] == "mtg") & (t["horizon"] == 1)].iloc[0]
    assert r["status"] != "ok" and r["conformal_fixes_it"] is None


def test_verdict_functions():
    assert cal.under_covers(0.79) and not cal.under_covers(0.80)
    assert not cal.under_covers(float("nan"))
    assert cal.conformal_fixes_it(0.78, 0.83) and not cal.conformal_fixes_it(0.81, 0.9)
    assert not cal.conformal_fixes_it(float("nan"), float("nan"))


def test_coverage_arithmetic_and_ci_wiring():
    p = _mtg_primary()
    # half of the rows are outside the interval
    p.loc[p.index % 2 == 0, "y_norm"] = 2.0
    f = cal.mtg_frame(p)
    t = cal.coverage_table(f, n_boot=100)
    r = t[(t["regime"] == "mtg") & (t["horizon"] == 7)].iloc[0]
    assert abs(r["coverage"] - 0.5) < 0.02
    assert r["block_days"] == 7
    assert t[(t["regime"] == "mtg") & (t["horizon"] == 56)].iloc[0]["block_days"] == 56
    assert r["cb_low"] <= r["coverage"] <= r["cb_high"]
    assert r["under_covers"] == (r["cb_high"] < 0.8) and r["under_covers"]
    assert {"cluster_low", "wilson_low", "label"} <= set(t.columns)
    empty = t[t["regime"] == "market_shock"].iloc[0]
    assert empty["n"] == 0 and not empty["under_covers"]


def test_split_leaves_no_straddling_or_evaluation_rows():
    f = cal.mtg_frame(_mtg_primary(n_days=90))
    c, e = cal.split_chronological(f)
    assert (c["origin_date"] < cal.MTG_SPLIT_DATE).all() and (
        c["target_date"] < cal.MTG_SPLIT_DATE
    ).all()
    assert (e["origin_date"] >= cal.MTG_SPLIT_DATE).all()
    # straddling rows belong to neither
    assert len(c) + len(e) < len(f)
    m = cal.market_frame(_shock_pred())
    mc, me = cal.split_chronological(m)
    assert set(mc["arm"]) == {"pre"} and set(me["arm"]) == {"post"}


def test_thresholds_independent_of_evaluation_outcomes():
    p = _mtg_primary(n_days=90)
    base = cal.conformal_table(cal.mtg_frame(p), n_boot=50)
    q = p.copy()
    q.loc[q["origin_date"] >= cal.MTG_SPLIT_DATE, "y_norm"] = 5.0
    changed = cal.conformal_table(cal.mtg_frame(q), n_boot=50)
    a = base[(base["regime"] == "mtg") & (base["analysis"] == "preregistered")].set_index("horizon")
    b = changed[(changed["regime"] == "mtg") & (changed["analysis"] == "preregistered")].set_index(
        "horizon"
    )
    pd.testing.assert_series_equal(a["threshold"], b["threshold"])
    assert (b["coverage_before"] == 0).all()


def test_before_after_metrics():
    p = _mtg_primary(n_days=200, start="2025-02-01")
    # calibration outcomes sit 0.3 above q90 for ~20% of rows -> positive threshold
    cal_rows = p["origin_date"] < cal.MTG_SPLIT_DATE
    rng = np.random.default_rng(0)
    p.loc[cal_rows & (rng.random(len(p)) < 0.4), "y_norm"] = 1.4
    p.loc[~cal_rows, "y_norm"] = 1.3
    t = cal.conformal_table(cal.mtg_frame(p), n_boot=100)
    r = t[(t["analysis"] == "preregistered") & (t["regime"] == "mtg") & (t["horizon"] == 1)].iloc[0]
    assert r["status"] == "ok" and r["threshold"] > 0
    assert r["coverage_before"] == 0.0 and r["coverage_after"] == 1.0
    assert np.isclose(r["median_rel_width_increase"], 2 * r["threshold"] / 0.2)
    assert r["mean_interval_score_after"] < r["mean_interval_score_before"]
    assert not r["conformal_fixes_it"]  # 1.0 CI excludes 0.8


def test_interval_score():
    s = cal.interval_score([1.0, 3.0], [0.0, 0.0], [2.0, 2.0])
    assert np.allclose(s, [2.0, 2.0 + 10.0])


def _mtg_rows(origins, target_days=1):
    o = pd.to_datetime(list(origins))
    return pd.DataFrame(
        {
            "regime": "mtg",
            "family": "mtg",
            "cluster": "a",
            "horizon": 1,
            "origin_date": o,
            "target_date": o + pd.Timedelta(days=target_days),
            "y": 1.0,
            "lo": 0.9,
            "hi": 1.1,
            "arm": "",
            "target_offset": np.nan,
        }
    )


def test_exploratory_split_date_odd_and_even():
    odd = _mtg_rows(pd.date_range("2025-01-01", periods=5))  # N=5 -> rank 3
    assert cal.exploratory_split_date(odd) == pd.Timestamp("2025-01-03")
    even = _mtg_rows(pd.date_range("2025-01-01", periods=6))  # N=6 -> rank 4
    assert cal.exploratory_split_date(even) == pd.Timestamp("2025-01-04")
    dup = pd.concat([even, even], ignore_index=True)  # distinct dates only
    assert cal.exploratory_split_date(dup) == pd.Timestamp("2025-01-04")


def test_exploratory_split_date_requires_mtg_rows():
    m = cal.market_frame(_shock_pred())
    try:
        cal.exploratory_split_date(m)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_exploratory_purge_and_boundary():
    d = pd.Timestamp("2025-01-10")
    f = pd.concat(
        [
            _mtg_rows(["2025-01-09"], target_days=1),  # target == D' -> purged
            _mtg_rows(["2025-01-08"], target_days=1),  # target < D' -> calibration
            _mtg_rows(["2025-01-10"], target_days=1),  # origin == D' -> evaluation
        ],
        ignore_index=True,
    )
    c, e = cal.split_chronological(f, d)
    assert list(c["origin_date"]) == [pd.Timestamp("2025-01-08")]
    assert list(e["origin_date"]) == [d]


def test_exploratory_rows_only_for_mtg_and_prereg_unchanged():
    frame = pd.concat(
        [cal.mtg_frame(_mtg_primary(n_days=90)), cal.market_frame(_shock_pred())],
        ignore_index=True,
    )
    t = cal.conformal_table(frame, n_boot=50)
    old = cal.conformal_table(frame, n_boot=50, exploratory=False)
    pre = t[t["analysis"] == "preregistered"].reset_index(drop=True)
    pd.testing.assert_frame_equal(pre, old)
    ex = t[t["analysis"] == "exploratory_post_hoc_resplit"]
    assert set(ex["regime"]) == {"mtg"} and list(ex["horizon"]) == list(cal.MTG_HORIZONS)
    assert (ex["split_date"] == cal.exploratory_split_date(frame)).all()
    assert (pre[pre["family"] == "mtg"]["split_date"] == cal.MTG_SPLIT_DATE).all()
    assert pre[pre["family"] == "market"]["split_date"].isna().all()
    assert len(t) == len(pre) + len(cal.MTG_HORIZONS)
    # same preregistered values as the pre-amendment code path: no new columns besides two
    assert set(pre.columns) - {"analysis", "split_date"} <= set(old.columns)
