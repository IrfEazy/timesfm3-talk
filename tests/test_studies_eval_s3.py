"""S3 evaluation on synthetic worlds (no real panel, no real forecasts)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm3lab.studies import eval_s3 as ev

START = pd.Timestamp("2024-01-01")
N_DAYS = 300
DATES = pd.date_range(START, periods=N_DAYS)


def _events(cols):
    n = len(cols)
    return pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(n)],
            "product_id": [100 + i for i in range(n)],
            "subtype": [None] * n,
            "event_date": [DATES[c] for c in cols],
            "card_id": list(range(n)),
            "rarity": ["R"] * n,
            "finish": ["nonfoil"] * n,
        }
    )


def _index(n):
    return pd.Index([(100 + i, None) for i in range(n)], dtype=object, tupleize_cols=False)


def _fc(events, q50_28, q50_56=None, width=0.1):
    rows = []
    for i, eid in enumerate(events["event_id"]):
        for h, v in ((28, q50_28[i]), (56, q50_28[i] if q50_56 is None else q50_56[i])):
            if v is None:
                continue
            rows.append(
                dict(
                    request_id=f"S3|{eid}", step=h, q10=v * (1 - width), q50=v, q90=v * (1 + width)
                )
            )
    return pd.DataFrame(rows, columns=["request_id", "step", "q10", "q50", "q90"])


def _world(ratios28, prev_ratio=1.25, n_cols=None, ratios56=None):
    """Each event e at column 60+3e: q(t)=10, q(t-7)=10*prev_ratio, q(t+28)=10*r."""
    n = len(ratios28)
    cols = [60 + 3 * i for i in range(n)]
    q = np.full((n, N_DAYS), 10.0)
    for i, c in enumerate(cols):
        q[i, c - 7] = 10.0 * prev_ratio
        q[i, c + 28] = 10.0 * ratios28[i]
        q[i, c + 56] = 10.0 * (ratios28[i] if ratios56 is None else ratios56[i])
    return _events(cols), q


def test_methods_and_ale_values():
    ev_df, q = _world([0.8])
    t = ev.event_errors(ev_df, DATES, _index(1), q, _fc(ev_df, [0.9]))
    r = t.iloc[0]
    assert r["in_set_28"]
    assert r["ale_tfm_28"] == pytest.approx(abs(np.log(0.9 / 0.8)))
    assert r["ale_hold_28"] == pytest.approx(abs(np.log(1 / 0.8)))
    assert r["ale_revert_28"] == pytest.approx(abs(np.log(1.25 / 0.8)))
    assert r["ale_halfway_28"] == pytest.approx(abs(np.log(np.sqrt(1.25) / 0.8)))
    assert r["covered_80_28"] == 0.0 and r["down_28"] == 1.0


def test_coverage_and_down_share():
    ev_df, q = _world([0.8, 1.2])
    t = ev.event_errors(ev_df, DATES, _index(2), q, _fc(ev_df, [0.82, 1.0], width=0.1))
    assert t["covered_80_28"].tolist() == [1.0, 0.0]
    assert ev.exploratory_results(t)["share_down_28"] == 0.5


def test_attrition_same_set_for_all_methods():
    ev_df, q = _world([0.8, 0.9, 1.1])
    q[1, 60 + 3 + 28] = np.nan  # outcome of event 1 missing (beyond ffill: neighbours present)
    q[1, 60 + 3 + 27 : 60 + 3 + 33] = np.nan
    t = ev.event_errors(ev_df, DATES, _index(3), q, _fc(ev_df, [0.8, 0.9, 1.1]))
    assert t["attrition_28"].tolist() == [False, True, False]
    assert t["in_set_28"].tolist() == [True, False, True]
    for m in ev.METHODS:
        assert t[f"ale_{m}_28"].notna().tolist() == t["in_set_28"].tolist()
    assert ev.primary_result(t, True)["n"] == int(t["in_set_28"].sum())
    assert ev.primary_result(t, True)["n_attrition"] == 1


def test_missing_forecast_and_revert_leave_set_for_all_methods():
    ev_df, q = _world([0.8, 0.9, 1.1])
    q[2, 60 + 6 - 7 : 60 + 6 - 3] = np.nan  # q(t-7) of event 2 missing
    t = ev.event_errors(ev_df, DATES, _index(3), q, _fc(ev_df, [0.8, None, 1.1]))
    assert t["in_set_28"].tolist() == [True, False, False]
    assert not t["attrition_28"].any()
    for m in ev.METHODS:
        assert t[f"ale_{m}_28"].notna().tolist() == [True, False, False]


def _eval(ratios, f28, confirmatory=True):
    ev_df, q = _world(ratios)
    n = len(ratios)
    return ev.evaluate(ev_df, DATES, _index(n), q, _fc(ev_df, f28), confirmatory)


def test_verdict_branches():
    n = 40
    rng = np.random.default_rng(3)
    ratios = list(rng.uniform(0.7, 0.9, n))  # prices fall; hold is wrong by ~log(1/0.8)
    good = _eval(ratios, ratios)  # perfect forecasts
    assert good["primary"]["verdict"] == "beats hold"
    assert good["primary"]["mean"] < 0
    bad = _eval(ratios, [1.0 / r for r in ratios])  # forecasts far worse than hold
    assert bad["primary"]["verdict"] == "loses to hold"
    same = _eval(ratios, [1.0] * n)  # TFM == hold -> zero differences
    assert same["primary"]["verdict"] == "inconclusive"


def test_descriptive_only_when_not_confirmatory():
    n = 40
    ratios = [0.8] * n
    res = _eval(ratios, ratios, confirmatory=False)
    assert res["primary"]["verdict"] == "descriptive only"
    assert res["primary"]["mean"] < 0  # numbers are still reported


def test_exploratory_contents_and_h56():
    n = 30
    ratios = [0.8] * n
    ev_df, q = _world(ratios, ratios56=[0.6] * n)
    res = ev.evaluate(ev_df, DATES, _index(n), q, _fc(ev_df, ratios, [0.6] * n), True)
    e = res["exploratory"]
    assert set(e) == {"h28", "h56", "share_down_28"}
    assert e["h56"]["tfm_minus_hold"]["mean"] < 0
    assert e["h28"]["coverage_80"] == 1.0 and e["h56"]["coverage_80"] == 1.0
    for k in ("tfm_minus_hold", "tfm_minus_revert", "tfm_minus_halfway"):
        assert "verdict" not in e["h28"][k]
    assert e["share_down_28"] == 1.0


def test_no_price_columns():
    ev_df, q = _world([0.8])
    t = ev.event_errors(ev_df, DATES, _index(1), q, _fc(ev_df, [0.8]))
    assert not [c for c in t.columns if "price" in c.lower() or c in ("q", "q_t")]


def test_determinism():
    n = 25
    ratios = list(np.linspace(0.6, 1.3, n))
    a = _eval(ratios, [0.9] * n)
    b = _eval(ratios, [0.9] * n)
    assert a["primary"] == b["primary"] and a["exploratory"] == b["exploratory"]
    pd.testing.assert_frame_equal(a["table"], b["table"])


def test_duplicate_forecast_rejected():
    ev_df, q = _world([0.8])
    fc = _fc(ev_df, [0.8])
    with pytest.raises(ValueError, match="duplicate forecast"):
        ev.event_errors(ev_df, DATES, _index(1), q, pd.concat([fc, fc.iloc[:1]]))


def test_attrition_and_tf_unavailable_counts_do_not_overlap_no_revert():
    ev_df, q = _world([0.8, 0.9, 1.1, 1.0])
    q[0, 60 - 7 : 60 - 3] = np.nan  # event 0: no revert (and outcome also missing below)
    q[0, 60 + 26 : 60 + 33] = np.nan
    q[1, 60 + 3 + 27 : 60 + 3 + 33] = np.nan  # event 1: attrition
    res = ev.evaluate(ev_df, DATES, _index(4), q, _fc(ev_df, [0.8, 0.9, None, 1.0]), True)
    c = res["counts"]
    assert c["n_no_revert"] == 1
    assert c["n_attrition_28"] == 1  # only event 1; event 0 is counted as no-revert
    assert res["primary"]["n_attrition"] == 1
    assert c["n_tf_unavailable_28"] == 1 and c["n_tf_unavailable_56"] == 1
