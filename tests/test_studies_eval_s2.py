"""S2 evaluation on synthetic worlds (no real panel, no real forecasts)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm3lab.studies import eval_s2 as ev

START = pd.Timestamp("2024-01-01")
N_DAYS = 120
DATES = pd.date_range(START, periods=N_DAYS)
T0 = 40  # origin column


def _decisions(n, bands=("lo",), finishes=("nonfoil",), origin_cols=None):
    return pd.DataFrame(
        {
            "decision_id": [f"d{i}" for i in range(n)],
            "product_id": [100 + i for i in range(n)],
            "subtype": [None] * n,
            "origin": [DATES[origin_cols[i] if origin_cols is not None else T0] for i in range(n)],
            "card_id": [i for i in range(n)],
            "rarity": ["R"] * n,
            "finish": [finishes[i % len(finishes)] for i in range(n)],
            "price_band": [bands[i % len(bands)] for i in range(n)],
        }
    )


def _index(n):
    return pd.Index([(100 + i, None) for i in range(n)], dtype=object, tupleize_cols=False)


def _fc(decisions, paths):
    rows = []
    for did, p in zip(decisions["decision_id"], paths, strict=True):
        if p is None:
            continue
        for d, v in enumerate(p, start=1):
            rows.append(dict(request_id=f"S2|{did}", step=d, q10=v * 0.9, q50=v, q90=v * 1.1))
    return pd.DataFrame(rows, columns=["request_id", "step", "q10", "q50", "q90"])


def _flat_world(n=1, base=10.0):
    return np.full((n, N_DAYS), base)


def _path(**kw):
    p = np.ones(28)
    for k, v in kw.items():
        p[int(k[1:]) - 1] = v
    return p


def test_choose_day_argmin_and_tie_break():
    path = np.ones(29)
    path[5] = path[9] = 0.9
    assert ev.choose_day(path) == 5
    assert ev.choose_day(np.ones(29)) == 0  # ties -> earliest (buy now)


def test_tfm5_threshold():
    p = np.ones(29)
    p[7] = 0.951
    assert ev.choose_day_tfm5(p) == 0
    p[7] = 0.95
    assert ev.choose_day_tfm5(p) == 7
    p[3] = 0.5
    assert ev.choose_day_tfm5(p) == 3


def test_execute_delay_and_failure_penalty():
    r = np.array([1.0] + [np.nan] * 28)
    r[3], r[6], r[28] = 0.8, 1.2, np.nan
    a, d, delayed, failed = ev.execute(r, 3)
    assert (a, d, delayed, failed) == (pytest.approx(0.2), 3, False, False)
    a, d, delayed, failed = ev.execute(r, 4)  # missing -> first later non-missing = day 6
    assert (a, d, delayed, failed) == (pytest.approx(-0.2), 6, True, False)
    a, d, delayed, failed = ev.execute(r, 7)  # nothing later -> fails, worst observable
    assert failed and d == -1 and a == pytest.approx(-0.2)


def test_advantage_sign_and_policy_choice():
    q = _flat_world()
    q[0, T0 + 5] = 8.0  # cheaper on day 5
    dec = _decisions(1)
    fc = _fc(dec, [_path(d5=0.8)])
    t = ev.decision_advantages(dec, DATES, _index(1), q, fc)
    r = t.iloc[0]
    assert r["dstar_tfm"] == 5 and r["A_tfm"] == pytest.approx(0.2)  # cheaper -> positive
    assert r["A_buy_now"] == 0.0 and r["A_deadline"] == 0.0
    assert r["hit_tfm"] and r["wait_tfm"]
    assert r["A_oracle"] == pytest.approx(0.2)
    assert r["A_random"] == pytest.approx(0.2 / 29)
    assert r["dstar_tfm5"] == 5  # 0.8 <= 0.95


def test_tfm5_does_not_wait_on_shallow_dip():
    q = _flat_world()
    q[0, T0 + 5] = 9.7
    dec = _decisions(1)
    t = ev.decision_advantages(dec, DATES, _index(1), q, _fc(dec, [_path(d5=0.97)]))
    assert t["dstar_tfm"].iloc[0] == 5 and t["dstar_tfm5"].iloc[0] == 0


def test_delayed_and_failed_in_table():
    q = _flat_world(2)
    # decision 0: planned day 4 missing (beyond ffill), day 5 present at 12 -> delayed, A = -0.2
    q[0, T0 + 4 : T0 + 8] = np.nan
    q[0, T0 + 8] = 12.0
    # decision 1: everything after origin missing from day 20 on, plan day 25 -> fail
    q[1, T0 + 20 :] = np.nan
    q[1, T0 + 10] = 7.0
    dec = _decisions(2)
    fc = _fc(dec, [_path(d4=0.5), _path(d25=0.5)])
    t = ev.decision_advantages(dec, DATES, _index(2), q, fc)
    a, b = t.iloc[0], t.iloc[1]
    assert a["delayed_tfm"] and not a["failed_tfm"]
    assert a["A_tfm"] == pytest.approx(-0.2)  # executed on day 8 at 12 vs 10
    assert b["failed_tfm"] and not b["delayed_tfm"]
    assert b["A_tfm"] == pytest.approx(0.0)  # min over observable advantages {0.0, 0.3}
    s = ev.summarise_policies(t).set_index("policy")
    assert s.loc["tfm", "fail_rate"] == 0.5 and s.loc["tfm", "delayed_rate"] == 0.5
    assert s.loc["tfm", "wait_share"] == 1.0


def test_failed_penalty_value():
    q = _flat_world()
    q[0, T0 + 1 : T0 + 29] = 13.0  # all later days dearer: A = -0.3
    q[0, T0 + 28] = np.nan
    dec = _decisions(1)
    t = ev.decision_advantages(dec, DATES, _index(1), q, _fc(dec, [_path(d28=0.1)]))
    r = t.iloc[0]
    assert r["failed_tfm"] and r["A_tfm"] == pytest.approx(-0.3)
    assert r["A_oracle"] == pytest.approx(0.0)


def test_baseline_failure_fallback_counted():
    q = _flat_world(2)
    q[1] = 10.0 + np.arange(N_DAYS) * 0.01
    dec = _decisions(2)
    # decision 0: context is 1 point long (origin right after a gap) -> drift/ets raise
    q[0, :T0] = np.nan
    t = ev.decision_advantages(dec, DATES, _index(2), q, _fc(dec, [_path(), _path()]))
    assert t["fit_failed_drift"].tolist() == [True, False]
    assert t["fit_failed_ets"].tolist() == [True, False]
    assert t["dstar_ets"].iloc[0] == 0 and t["dstar_drift"].iloc[0] == 0
    c = ev.counts(t)
    assert c["n_fit_failed_ets"] == 1 and c["n_fit_failed_drift"] == 1
    assert c["n_tfm_unavailable"] == 0


def test_drift_policy_follows_rule():
    q = _flat_world()
    q[0] = 10.0 - np.arange(N_DAYS) * 0.05  # falling: drift path declines -> waits to day 28
    dec = _decisions(1)
    t = ev.decision_advantages(dec, DATES, _index(1), q, _fc(dec, [_path()]))
    assert t["dstar_drift"].iloc[0] == 28
    assert t["A_drift"].iloc[0] == pytest.approx(1 - q[0, T0 + 28] / q[0, T0])
    assert t["dstar_tfm"].iloc[0] == 0


def test_missing_tfm_forecast_excluded_from_all_comparisons():
    q = _flat_world(3)
    dec = _decisions(3)
    t = ev.decision_advantages(dec, DATES, _index(3), q, _fc(dec, [None, _path(), _path()]))
    assert t["fit_failed_tfm"].tolist() == [True, False, False]
    c = ev.counts(t)
    assert c["n_tfm_unavailable"] == 1 and c["n_analysis"] == 2 and c["n_with_origin"] == 3
    assert (ev.summarise_policies(t)["n"] == 2).all()
    assert ev.primary_result(t)["n"] == 2
    e = ev.exploratory_results(t)
    for k in ("tfm_minus_ets", "tfm_minus_drift", "tfm5", "deadline", "random", "oracle"):
        assert e[k]["n"] == 2
    assert sum(v["n"] for v in e["by_finish"].values()) == 2


def test_unavailable_tfm_not_buy_now_in_primary_mean():
    # decision 0 has no TFM path; if it were bought-now it would add A=0 to the mean
    n = 3
    q = np.full((n, N_DAYS), 10.0)
    for i in range(n):
        q[i, T0 + 5] = 9.0
    dec = _decisions(n)
    t = ev.decision_advantages(
        dec, DATES, _index(n), q, _fc(dec, [None, _path(d5=0.8), _path(d5=0.8)])
    )
    assert ev.primary_result(t)["mean"] == pytest.approx(0.1)


def test_nonpositive_baseline_path_is_fit_failure(monkeypatch):
    q = _flat_world(1)
    dec = _decisions(1)
    monkeypatch.setattr(ev.baselines, "ets_forecast", lambda ctx, h: np.zeros(h))
    monkeypatch.setattr(ev.baselines, "drift_forecast", lambda ctx, h: np.full(h, -1.0))
    t = ev.decision_advantages(dec, DATES, _index(1), q, _fc(dec, [_path()]))
    assert t["fit_failed_ets"].tolist() == [True] and t["fit_failed_drift"].tolist() == [True]
    assert t["dstar_ets"].iloc[0] == 0 and t["dstar_drift"].iloc[0] == 0
    assert ev.counts(t)["n_fit_failed_ets"] == 1


def test_no_origin_excluded_and_counted():
    q = _flat_world(2)
    q[1, T0] = np.nan
    q[1, T0 - 5 : T0 + 1] = np.nan
    dec = _decisions(2)
    t = ev.decision_advantages(dec, DATES, _index(2), q, _fc(dec, [_path(), _path()]))
    assert t["has_origin"].tolist() == [True, False]
    assert ev.counts(t)["n_no_origin"] == 1
    assert ev.summarise_policies(t)["n"].iloc[0] == 1


def test_no_price_columns():
    q = _flat_world()
    dec = _decisions(1)
    t = ev.decision_advantages(dec, DATES, _index(1), q, _fc(dec, [_path()]))
    assert not [c for c in t.columns if "price" in c.lower() and c != "price_band"]
    assert not [c for c in t.columns if c.lower() in ("q", "quote", "e", "q_t")]


def _world_many(n, gain):
    """n decisions at spread origins; day-5 quote is cheaper by `gain` (ratio)."""
    q = np.full((n, N_DAYS), 10.0)
    cols = [30 + (i * 7) % 60 for i in range(n)]
    for i, c in enumerate(cols):
        q[i, c + 5] = 10.0 * (1 - gain)
    return q, cols


def _eval(gain, n=40, tfm_wait=True, seed=1):
    q, cols = _world_many(n, gain)
    dec = _decisions(n, bands=("lo", "hi"), finishes=("nonfoil", "foil"), origin_cols=cols)
    p = _path(d5=0.8) if tfm_wait else _path()
    return ev.evaluate(dec, DATES, _index(n), q, _fc(dec, [p] * n), seed)


def test_verdict_beats_worse_inconclusive():
    # day-5 cheaper by 10% (only a little noise-free): TFM waits -> positive for every decision
    assert _eval(0.1)["primary"]["verdict"] == "beats buying now"
    # day-5 dearer by 10%: TFM waits and loses
    assert _eval(-0.1)["primary"]["verdict"] == "worse than buying now"
    # TFM never waits -> A = 0 everywhere -> degenerate interval at 0 -> inconclusive
    assert _eval(0.1, tfm_wait=False)["primary"]["verdict"] == "inconclusive"


def test_primary_levels_and_exploratory_shape():
    res = _eval(0.1)
    assert res["primary"]["level"] == pytest.approx(1 - 0.05 / 3)
    e = res["exploratory"]
    assert set(e["by_price_band"]) == {"lo", "hi"} and set(e["by_finish"]) == {"foil", "nonfoil"}
    assert "verdict" not in e["tfm5"]
    assert e["tfm_minus_ets"]["n"] == 40 and e["tfm5"]["mean"] == pytest.approx(0.1)
    assert e["deadline"]["mean"] == 0.0


def test_determinism():
    a, b = _eval(0.1), _eval(0.1)
    assert a["primary"] == b["primary"] and a["exploratory"] == b["exploratory"]
    pd.testing.assert_frame_equal(a["table"], b["table"])


def test_duplicate_forecast_rejected():
    q = _flat_world()
    dec = _decisions(1)
    fc = _fc(dec, [_path()])
    with pytest.raises(ValueError, match="duplicate forecast"):
        ev.decision_advantages(dec, DATES, _index(1), q, pd.concat([fc, fc.iloc[:1]]))
