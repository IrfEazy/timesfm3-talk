"""S1 evaluation on a synthetic world (no real panel, no real forecasts)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm3lab.studies import eval_s1 as ev

START = pd.Timestamp("2024-01-01")
N_DAYS = 900
DATES = pd.date_range(START, periods=N_DAYS)
SPACING = 60


def _world(n_events=12, ks=(14,), n_per=25, c_per=0, seed=0):
    """Event e (R = START + 60 e) moves by m_e (+0.1 for rarity C) linearly in
    log price from R; SKU rows are unique per (event, k)."""
    rng = np.random.default_rng(seed)
    m_event = rng.normal(0.0, 0.3, size=n_events)
    rows, keys, base = [], [], []
    pid = 1000
    for e in range(n_events):
        r = START + pd.Timedelta(days=SPACING * e)
        for k in ks:
            for i in range(n_per):
                rarity = "C" if i < c_per else "R"
                pid += 1
                rows.append(
                    dict(
                        event_date=r,
                        k=k,
                        product_id=pid,
                        subtype="Normal" if i % 2 else None,
                        group_id=e,
                        group_abbreviation=f"G{e}",
                        card_id=pid,
                        rarity=rarity,
                        finish="nonfoil",
                        first_printing=bool(i % 2),
                        floor_1=True,
                    )
                )
                keys.append((pid, "Normal" if i % 2 else None))
                m = m_event[e] + (0.1 if rarity == "C" else 0.0)
                base.append((r, m, 2.0 + 0.01 * i))
    skus = pd.DataFrame(rows)
    qmat = np.empty((len(keys), N_DAYS))
    day = np.arange(N_DAYS)
    for j, (r, m, b) in enumerate(base):
        d0 = (r - START).days
        qmat[j] = b * np.exp(m * (day - d0) / 90.0)
    return skus, pd.Index(keys, dtype=object, tupleize_cols=False), qmat


def _forecasts(skus, qmat, index, ratio_fn=None, drop=()):
    """Perfect TimesFM forecasts (q50 = realised ratio) unless ratio_fn is given."""
    row_of = {key: i for i, key in enumerate(index)}
    out = []
    for p, k, r, pid, sub in zip(
        range(len(skus)),
        skus["k"],
        skus["event_date"],
        skus["product_id"],
        skus["subtype"],
        strict=True,
    ):
        if p in drop:
            continue
        i = row_of[(int(pid), sub if isinstance(sub, str) else None)]
        d0 = (r - START).days
        ratio = qmat[i, d0 + 90] / qmat[i, d0 + k]
        if ratio_fn is not None:
            ratio = ratio_fn(ratio)
        out.append(
            dict(
                request_id=ev.request_id(k, r, pid, sub),
                step=90 - k,
                point=ratio,
                q10=ratio * 0.95,
                q50=ratio,
                q90=ratio * 1.05,
            )
        )
    return pd.DataFrame(out)


def _errors(world, forecasts=None, qmat=None, floor=1.0):
    skus, index, q = world
    q = q if qmat is None else qmat
    f = _forecasts(skus, q, index) if forecasts is None else forecasts
    return ev.sku_errors(skus, DATES, index, q, f, floor)


def test_request_id_format():
    assert ev.request_id(14, "2024-03-01", 7, None) == "S1|k14|2024-03-01|7|None"
    assert ev.request_id(7, pd.Timestamp("2024-03-01"), 7, "Foil") == "S1|k7|2024-03-01|7|Foil"


def test_no_price_columns_and_errors_are_exact():
    world = _world(n_events=3)
    out = _errors(world)
    assert not [c for c in out.columns if "price" in c.lower() or c in ("q_o", "q_t")]
    assert out["has_outcome"].all() and not out["attrition"].any()
    assert np.allclose(out["abs_err_tfm"], 0.0)
    assert np.allclose(out["abs_err_naive"], out["move_log"].abs())
    assert out["covered_80"].astype(bool).all()
    # move = m * (90 - 14) / 90, signed naive error = -move
    assert np.allclose(out["signed_err_naive"], -out["move_log"])


def test_attrition_when_target_quote_missing():
    skus, index, q = _world(n_events=3)
    q = q.copy()
    # Event-0 SKU row 0 misses q(T).
    q[0, 90] = np.nan
    out = _errors((skus, index, q))
    row = out[out["product_id"] == skus["product_id"].iat[0]].iloc[0]
    assert row["attrition"] and not row["has_outcome"]
    assert np.isnan(row["abs_err_naive"]) and np.isnan(row["abs_err_tfm"])
    assert out["attrition"].sum() == 1


def test_paired_set_drops_sku_from_all_methods_on_missing_forecast():
    world = _world(n_events=2, n_per=25)
    skus, index, q = world
    f = _forecasts(skus, q, index, drop={0, 1, 2})
    out = _errors(world, forecasts=f)
    assert (~out["tf_available"]).sum() == 3
    summ = ev.event_summary(out)
    first = summ.iloc[0]
    assert first["n_removed_missing_forecast"] == 3
    assert first["n_paired"] == 22
    ok = out[(out["event_date"] == START) & out["tf_available"]]
    assert first["mean_abs_err_naive"] == pytest.approx(ok["abs_err_naive"].mean())
    # Non-finite forecasts also count as missing.
    bad = f.copy()
    bad.loc[bad.index[5], "q50"] = np.nan
    assert (~_errors(world, forecasts=bad)["tf_available"]).sum() == 4


def test_events_qualify_with_twenty_paired_skus():
    world = _world(n_events=2, n_per=25)
    skus, index, q = world
    f = _forecasts(skus, q, index, drop={0, 1, 2, 3, 4, 5, 6})  # 18 left in event 0
    summ = ev.event_summary(_errors(world, forecasts=f))
    assert list(summ["qualifies"]) == [False, True]


def test_cohort_curve_uses_only_past_events_and_is_causal():
    world = _world(n_events=8, c_per=0)
    out = _errors(world)
    # Event e has references 0..e-2 (reference day-90 must be <= origin = R + 14).
    by_event = out.groupby("event_date")["n_ref_events"].first().to_numpy()
    assert list(by_event) == [0, 0, 1, 2, 3, 4, 5, 6]
    avail = out.groupby("event_date")["cohort_available"].all().to_numpy()
    assert list(avail) == [False] * 3 + [True] * 5
    # Expected cohort move of event e = median move over reference events 0..e-2.
    moves = out.groupby("event_date")["move_log"].first().to_numpy()
    for e in range(3, 8):
        got = out[out["event_date"] == START + pd.Timedelta(days=SPACING * e)]
        pred = got["signed_err_cohort"] + got["move_log"]
        want = np.median(np.concatenate([[moves[j]] * 25 for j in range(e - 1)]))
        assert np.allclose(pred, want, atol=1e-6)

    # Poison every quote after the origin of event 5: its cohort forecast and all
    # earlier ones must not change.
    skus, index, q = world
    q2 = q.copy()
    o5 = SPACING * 5 + 14
    q2[:, o5 + 1 :] = np.random.default_rng(1).uniform(50, 99, size=q2[:, o5 + 1 :].shape)
    out2 = _errors(world, qmat=q2, forecasts=_forecasts(skus, q, index))
    cut = out["event_date"] <= START + pd.Timedelta(days=SPACING * 5)
    p1 = (out["signed_err_cohort"] + out["move_log"])[cut]
    p2 = (out2["signed_err_cohort"] + out2["move_log"])[cut]
    assert np.allclose(p1, p2, equal_nan=True)
    assert (out2["cohort_available"][cut] == out["cohort_available"][cut]).all()


def test_cohort_excludes_reference_whose_day90_is_after_origin():
    world = _world(n_events=4)
    skus, index, q = world
    q2 = q.copy()
    # Event 2 (R=120) has event 1 (day-90 = 150) after its origin (134): poison
    # event 1's day-90 quotes; the cohort of event 2 must be unchanged (no refs).
    q2[25:50, SPACING + 90] *= 100.0
    a, b = _errors(world), _errors((skus, index, q2), forecasts=_forecasts(skus, q, index))
    e2 = a["event_date"] == START + pd.Timedelta(days=120)
    assert (a["n_ref_events"][e2] == b["n_ref_events"][e2]).all()
    assert not b["cohort_available"][e2].any()


def test_cohort_stratum_fallback_levels():
    out = _errors(_world(n_events=6, c_per=8))
    last = out[out["event_date"] == START + pd.Timedelta(days=SPACING * 3)]
    assert set(last.loc[last["rarity"] == "R", "cohort_level"]) == {"stratum"}
    assert set(last.loc[last["rarity"] == "C", "cohort_level"]) == {"all"}


def test_floor_filter_and_floor_025():
    skus, index, q = _world(n_events=2)
    s2 = skus.copy()
    s2.loc[:4, "floor_1"] = False
    out = ev.sku_errors(s2, DATES, index, q, _forecasts(skus, q, index), 1.0)
    assert len(out) == len(skus) - 5
    out025 = ev.sku_errors(s2, DATES, index, q, _forecasts(skus, q, index), 0.25)
    assert len(out025) == len(skus)  # all quotes >= 0.25 at origin


def _primary_world(n_events):
    world = _world(n_events=n_events, ks=(7, 14, 28))
    return _errors(world), world


def test_primary_result_beats_naive_with_perfect_forecasts():
    out, _ = _primary_world(12)
    res = ev.primary_result(out)
    assert res["G"] == 12 and res["k"] == 14
    assert res["mean_diff"] < 0
    assert res["t_ci_9833"][1] < 0
    assert res["sign_flip_p"] == pytest.approx(2 / 2**12)
    assert res["boot_ci_95"][1] <= 0
    assert res["verdict"] == "beats naive"


def test_primary_result_descriptive_below_ten_events():
    out, _ = _primary_world(9)
    res = ev.primary_result(out)
    assert res["G"] == 9 and res["verdict"] == "descriptive only"


def test_primary_result_counts_attrition_and_missing_forecasts():
    skus, index, q = _world(n_events=3)
    q = q.copy()
    q[0, 90] = np.nan
    f = _forecasts(skus, q, index, drop={30})
    res = ev.primary_result(_errors((skus, index, q), forecasts=f))
    assert res["n_attrition"] == 1
    assert res["n_removed_missing_forecast"] == 1


def test_exploratory_results_run_and_cohort_comparison():
    out, world = _primary_world(12)
    ex = ev.exploratory_results(out, _errors(world, floor=0.25))
    assert ex["k7"]["k"] == 7 and ex["k28"]["k"] == 28
    assert ex["floor_025"]["G"] == 12
    assert ex["cohort"]["G"] == 9  # events 3..11 have >= 2 reference events and cohort curves
    assert ex["bias_coverage"]["coverage_80"] == 1.0
    assert abs(ex["bias_coverage"]["bias_tfm"]) < 1e-9
    player = ex["player"]
    assert list(player["rarity"]) == ["R"] and player["n_events"].iat[0] == 12
    assert 0 <= player["share_down"].iat[0] <= 1
    assert (
        player["median_ci_low"].iat[0]
        <= player["median_log_move"].iat[0]
        <= player["median_ci_high"].iat[0]
    )


@pytest.mark.parametrize("kind", ["object", "string", "pyarrow"])
def test_truthy_parses_string_dtypes(kind):
    dtype = {"object": object, "string": "string", "pyarrow": "string[pyarrow]"}[kind]
    try:
        s = pd.Series(["True", "false", "0", "1", "yes"], dtype=dtype)
    except Exception:
        pytest.skip("dtype unavailable")
    assert ev._truthy(s).tolist() == [True, False, False, True, True]
    with pytest.raises(ValueError):
        ev._truthy(pd.Series(["maybe"], dtype=dtype))


def test_floor_1_output_is_parsed_boolean():
    skus, index, q = _world(n_events=3)
    skus = skus.copy()
    skus["floor_1"] = "True"
    skus.loc[0, "floor_1"] = "False"
    out = ev.sku_errors(skus, DATES, index, q, _forecasts(skus, q, index))
    assert out["floor_1"].dtype == bool and out["floor_1"].all()
    assert len(out) == len(skus) - 1


def test_no_origin_counted_in_attrition_accounting():
    skus, index, q = _world(n_events=3)
    q = q.copy()
    q[0, 14] = np.nan
    q[1, 90] = np.nan
    out = _errors((skus, index, q))
    s = ev.event_summary(out)
    first = s.iloc[0]
    assert first["n_no_origin"] == 1 and first["n_attrition"] == 1
    assert first["n_skus"] == first["n_outcome"] + first["n_attrition"] + first["n_no_origin"]
    assert ev.primary_result(out)["n_no_origin"] == 1


def test_duplicate_forecast_rows_raise():
    skus, index, q = _world(n_events=3)
    f = _forecasts(skus, q, index)
    dup = pd.concat([f, f.iloc[[0]].assign(q50=f["q50"].iloc[0] * 2)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        ev.sku_errors(skus, DATES, index, q, dup)
    with pytest.raises(ValueError, match="duplicate"):
        ev.sku_errors(skus, DATES, index, q, pd.concat([f, f.iloc[[0]]], ignore_index=True))


@pytest.mark.parametrize("missing", [np.nan, None, "None", ""])
def test_missing_subtype_round_trips_with_quote_keys(missing):
    skus, index, q = _world(n_events=3)
    f = _forecasts(skus, q, index)
    skus = skus.copy()
    skus["subtype"] = skus["subtype"].map(lambda v: missing if v is None else v)
    out = ev.sku_errors(skus, DATES, index, q, f)
    assert (
        out["subtype"].isna().sum()
        == (skus["subtype"].map(lambda v: not isinstance(v, str) or v in ("", "None"))).sum()
    )
    assert out["has_outcome"].all() and out["tf_available"].all()
