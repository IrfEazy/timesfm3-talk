"""Preregistered v3 selection rules: boundaries, determinism, future poisoning.

All synthetic; nothing here reads the real panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tfm3lab.data import panel as panel_mod
from tfm3lab.studies import quotes
from tfm3lab.studies import requests as req_mod
from tfm3lab.studies import selection as sel

NAN = np.nan
START = "2024-01-01"


def make_data(series: dict, n: int, start: str = START) -> sel.QuoteData:
    """series: {(pid, subtype): array-like of length n (NaN = missing)}."""
    skus = list(series)
    wide = np.full((len(skus), n), np.nan)
    for i, key in enumerate(skus):
        wide[i, :] = np.asarray(series[key], dtype=np.float64)
    return sel.QuoteData.from_wide(pd.date_range(start, periods=n, freq="D"), skus, wide)


def flat(n: int, value: float, *, nan_before: int = 0, nan_days=()) -> np.ndarray:
    out = np.full(n, float(value))
    out[:nan_before] = np.nan
    for d in nan_days:
        out[d] = np.nan
    return out


def rows_of(data: sel.QuoteData, *skus) -> np.ndarray:
    return np.array([data.skus.index(s) for s in skus], dtype=np.int64)


# --- primitives vs the quotes module -----------------------------------------------


def random_wide(rng, n_sku=30, n_days=200):
    wide = np.exp(rng.normal(1.0, 0.8, size=(n_sku, 1)) + 0.05 * rng.normal(size=(n_sku, n_days)))
    miss = rng.random((n_sku, n_days)) < rng.uniform(0.0, 0.5, size=(n_sku, 1))
    wide[miss] = np.nan
    wide[:, : int(rng.integers(0, 40))] = np.nan
    return wide


def test_context_length_matches_quotes_context():
    rng = np.random.default_rng(1)
    wide = random_wide(rng, 12, 700)
    skus = [(i, None) for i in range(12)]
    data = sel.QuoteData.from_wide(pd.date_range(START, periods=700), skus, wide)
    for t in (5, 80, 400, 699):
        got = sel.context_length(data, np.arange(12), t)
        want = [len(quotes.context(data.qmat, i, t)) for i in range(12)]
        assert got.tolist() == want


def test_s2_eligible_matches_a_per_sku_reference_built_on_quotes():
    rng = np.random.default_rng(2)
    wide = random_wide(rng, 60, 260)
    data = sel.QuoteData.from_wide(
        pd.date_range(START, periods=260), [(i, None) for i in range(60)], wide
    )
    n_eligible = 0
    for t in (50, 95, 150, 259):
        want = []
        for i in range(60):
            first = quotes.first_observed_idx(data.wide, i, t)
            ok = (
                data.obs[i, t]
                and data.qmat[i, t] >= 1.0
                and first is not None
                and first <= t - 90
                and t >= 90
                and quotes.n_observed_in_window(data.wide, i, t - 90, t, origin_idx=t) >= 60
                and len(quotes.context(data.qmat, i, t)) >= 60
            )
            if ok:
                want.append(i)
        assert sel.s2_eligible(data, t).tolist() == want
        n_eligible += len(want)
    assert n_eligible > 0


# --- S1 ------------------------------------------------------------------------------


def test_s1_floor_boundaries_are_inclusive_at_025_and_100():
    n, o = 40, 30
    data = make_data(
        {
            (1, "N"): flat(n, 1.0),
            (2, "N"): flat(n, 0.25),
            (3, "N"): flat(n, 0.2499),
            (4, "N"): flat(n, 0.9999),
        },
        n,
    )
    rows = np.arange(4)
    eligible, floor_1 = sel.s1_eligible(data, rows, o)
    assert eligible.tolist() == [True, True, False, True]
    assert floor_1.tolist() == [True, False, False, False]


def test_s1_needs_five_observed_days_in_the_trailing_context():
    n, o = 40, 30
    data = make_data(
        {
            (1, "N"): flat(n, 2.0, nan_before=26),  # observed 26..30 -> 5
            (2, "N"): flat(n, 2.0, nan_before=27),  # 4
            (3, "N"): flat(n, 2.0, nan_days=[29, 30]),  # origin ffilled, many observed
            (4, "N"): flat(n, 2.0, nan_days=[27, 28, 29, 30]),  # quote at origin missing
        },
        n,
    )
    eligible, _ = sel.s1_eligible(data, np.arange(4), o)
    assert eligible.tolist() == [True, False, True, False]


def test_s1_context_run_stops_at_a_gap_so_older_observations_do_not_count():
    series = flat(30, 2.0, nan_days=[10, 11, 12, 13, 14])  # 5-day hole: q missing 13, 14
    data = make_data({(1, "N"): series}, 30)
    assert not sel.s1_eligible(data, np.array([0]), 18)[0][0]  # run 15..18 = 4 observed
    assert sel.s1_eligible(data, np.array([0]), 19)[0][0]  # 15..19 = 5


def test_s1_horizon_and_select_s1_origins_and_order():
    assert [sel.s1_horizon(k) for k in sel.S1_KS] == [83, 76, 62]
    n = 60
    data = make_data(
        {
            (5, "Foil"): flat(n, 2.0),
            (3, "Normal"): flat(n, 0.5),
            (5, None): flat(n, 3.0),
            (9, "Normal"): flat(n, 3.0),  # group 7, not in the event
        },
        n,
    )
    events = pd.DataFrame({"release_date": [pd.Timestamp("2024-01-11")], "group_ids": [[1]]})
    group_of_row = np.array([1 if s[0] != 9 else 7 for s in data.skus])
    units, counts = sel.select_s1(data, events, group_of_row)
    assert units["k"].tolist() == [7] * 3 + [14] * 3 + [28] * 3
    assert units["origin_idx"].unique().tolist() == [17, 24, 38]
    first = units[units["k"] == 7]
    assert [data.skus[r] for r in first["row"]] == [(3, "Normal"), (5, None), (5, "Foil")]
    assert counts["n_candidates"].tolist() == [3, 3, 3]
    assert counts["eligible_floor_100"].tolist() == [2, 2, 2]


def test_select_s1_rejects_an_origin_outside_the_panel():
    data = make_data({(1, "N"): flat(20, 2.0)}, 20)
    events = pd.DataFrame({"release_date": [pd.Timestamp("2024-01-05")], "group_ids": [[1]]})
    with pytest.raises(ValueError, match="outside the panel"):
        sel.select_s1(data, events, np.array([1]))


# --- S2 eligibility ------------------------------------------------------------------

T = 120


def s2_set(series: dict, n: int = 130, t: int = T) -> list:
    data = make_data(series, n)
    return [data.skus[r] for r in sel.s2_eligible(data, t)]


def test_s2_price_floor_is_inclusive_and_needs_an_observation_on_t():
    got = s2_set(
        {
            (1, "N"): flat(130, 1.0),
            (2, "N"): flat(130, 0.99),
            (3, "N"): flat(130, 5.0, nan_days=[T]),  # quote ffilled, not observed on t
        }
    )
    assert got == [(1, "N")]


def test_s2_first_observed_boundary_is_t_minus_90_inclusive():
    got = s2_set(
        {(1, "N"): flat(130, 2.0, nan_before=T - 90), (2, "N"): flat(130, 2.0, nan_before=T - 89)}
    )
    assert got == [(1, "N")]


def test_s2_needs_60_observed_days_in_the_91_day_window():
    drop = list(range(T - 89, T, 2))  # isolated gaps: quotes stay valid via ffill
    assert len(drop) == 45
    got = s2_set(
        {
            (1, "N"): flat(130, 2.0, nan_days=drop[:31]),  # 91 - 31 = 60 observed
            (2, "N"): flat(130, 2.0, nan_days=drop[:32]),  # 59
        }
    )
    assert got == [(1, "N")]


def test_s2_trailing_context_length_boundary_is_60():
    got = s2_set(
        {
            (1, "N"): flat(130, 2.0, nan_days=range(T - 63, T - 59)),  # q missing at T-60: run 60
            (2, "N"): flat(130, 2.0, nan_days=range(T - 62, T - 58)),  # q missing at T-59: run 59
        }
    )
    assert got == [(1, "N")]


def test_s2_eligible_order_is_canonical_with_a_missing_subtype_first():
    got = s2_set({(5, "Foil"): flat(130, 2.0), (5, None): flat(130, 2.0), (3, "N"): flat(130, 2.0)})
    assert got == [(3, "N"), (5, None), (5, "Foil")]


def test_s2_is_empty_before_the_ninetieth_day():
    data = make_data({(1, "N"): flat(130, 2.0)}, 130)
    assert sel.s2_eligible(data, 89).size == 0
    assert sel.s2_eligible(data, 90).size == 1


# --- S2 pool and acceptance ------------------------------------------------------------


def test_pool_is_deterministic_has_m_draws_and_redraws_empty_dates():
    eligible = [np.array([], dtype=np.int64), np.array([4, 7, 9]), np.array([2])]
    window = [10, 11, 12]
    a = sel.draw_s2_pool(eligible, window, m=50, seed=sel.S2_SEED)
    b = sel.draw_s2_pool(eligible, window, m=50, seed=sel.S2_SEED)
    pd.testing.assert_frame_equal(a, b)
    assert a["draw"].tolist() == list(range(50))
    assert set(a["origin_idx"]) == {11, 12}
    assert set(a.loc[a["origin_idx"] == 12, "row"]) == {2}
    assert set(a.loc[a["origin_idx"] == 11, "row"]) <= {4, 7, 9}
    c = sel.draw_s2_pool(eligible, window, m=50, seed=1)
    assert not a.equals(c)


def test_pool_uses_the_frozen_draw_order_date_then_sku():
    eligible = [np.array([3, 5, 8]), np.array([1, 2])]
    pool = sel.draw_s2_pool(eligible, [4, 5], m=5)
    rng = np.random.default_rng(20261014)
    want = []
    for i in range(5):
        d = int(rng.integers(2))
        j = int(rng.integers(len(eligible[d])))
        want.append((i, [4, 5][d], int(eligible[d][j])))
    assert list(pool.itertuples(index=False, name=None)) == want
    assert sel.S2_POOL_SIZE == 5500 and sel.S2_SEED == 20261014


def test_pool_with_no_eligible_pair_raises():
    with pytest.raises(ValueError, match="no .* eligible"):
        sel.draw_s2_pool([np.array([], dtype=np.int64)], [3], m=2)


def pool_of(*cands):
    """cands: (draw, origin, row)."""
    return pd.DataFrame(cands, columns=["draw", "origin_idx", "row"])


CARDS = np.array(["x", "x", "y", "z"], dtype=object)  # rows 0,1 share a card


def acc(pool, cards=CARDS):
    out = sel.accept_s2(pool, cards)
    return list(zip(out["draw"], out["accepted"], strict=True))


def test_accept_spacing_boundary_28_days_is_allowed_27_is_not():
    assert acc(pool_of((0, 100, 0), (1, 128, 1))) == [(0, True), (1, True)]
    assert acc(pool_of((0, 100, 0), (1, 127, 1))) == [(0, True), (1, False)]


def test_accept_rejects_exact_duplicates_and_same_origin_same_card():
    assert acc(pool_of((0, 100, 0), (1, 100, 0))) == [(0, True), (1, False)]
    assert acc(pool_of((0, 100, 0), (1, 100, 1))) == [(0, True), (1, False)]
    assert acc(pool_of((0, 100, 0), (1, 100, 2))) == [(0, True), (1, True)]


def test_accept_is_backward_looking_the_later_draw_never_blocks_an_earlier_origin():
    # draw 0 sits LATER in time than draw 1; the earlier origin is decided first.
    assert acc(pool_of((0, 110, 0), (1, 100, 1))) == [(1, True), (0, False)]


def test_rejected_candidates_do_not_extend_the_cooldown():
    # 100 accepted; 120 rejected; 130 is 30 days after 100 -> accepted.
    got = acc(pool_of((0, 100, 0), (1, 120, 0), (2, 130, 0)))
    assert got == [(0, True), (1, False), (2, True)]


@pytest.mark.parametrize(
    ("price", "band"),
    [(1.0, "1-5"), (4.99, "1-5"), (5.0, "5-20"), (19.99, "5-20"), (20.0, "20+"), (900.0, "20+")],
)
def test_price_band_edges(price, band):
    assert sel.price_band(price) == band


def test_price_band_rejects_a_price_below_the_floor():
    with pytest.raises(ValueError):
        sel.price_band(0.99)


# --- S3 ----------------------------------------------------------------------------------

T3 = 120


def jump(n, base, high, at, *, until=None):
    out = np.full(n, float(base))
    out[at:until] = high
    return out


def crossing_rows(series: dict, t: int = T3, n: int = 140) -> list:
    data = make_data(series, n)
    rows, prior = sel.s3_crossings(data, t)
    assert len(rows) == len(prior)
    return [data.skus[r] for r in rows]


def test_s3_crossing_ratio_boundary_is_inclusive_at_1_5():
    got = crossing_rows(
        {
            (1, "N"): jump(140, 2.0, 3.0, T3),  # exactly 1.5
            (2, "N"): jump(140, 2.0, 2.99, T3),  # 1.495
        }
    )
    assert got == [(1, "N")]


def test_s3_only_the_first_day_of_a_run_above_the_ratio_is_a_crossing():
    series = {(1, "N"): jump(140, 2.0, 3.0, T3 - 1)}
    assert crossing_rows(series, T3 - 1) == [(1, "N")]
    assert crossing_rows(series, T3) == []  # ratio(t-1) = 1.5 is not < 1.5
    assert crossing_rows(series, T3 + 1) == []


def test_s3_needs_ratio_of_the_previous_day_defined_which_needs_an_observation():
    series = jump(140, 2.0, 3.0, T3)
    series[T3 - 1] = np.nan  # quote ffilled to 2.0 but not observed
    assert crossing_rows({(1, "N"): series}) == []


def test_s3_prior_quote_floor_is_inclusive_at_1_00():
    got = crossing_rows({(1, "N"): jump(140, 1.0, 1.5, T3), (2, "N"): jump(140, 0.99, 1.5, T3)})
    assert got == [(1, "N")]


def test_s3_age_and_context_requirements():
    got = crossing_rows(
        {
            (1, "N"): jump(140, 2.0, 3.0, T3),
            (2, "N"): np.where(np.arange(140) < T3 - 89, np.nan, jump(140, 2.0, 3.0, T3)),
            (3, "N"): np.where(np.arange(140) < T3 - 90, np.nan, jump(140, 2.0, 3.0, T3)),
            (4, "N"): np.where(
                np.isin(np.arange(140), range(T3 - 63, T3 - 59)), np.nan, jump(140, 2.0, 3.0, T3)
            ),
        }
    )
    assert got == [(1, "N"), (3, "N"), (4, "N")]


def test_s3_same_day_same_card_keeps_largest_prior_then_lowest_sku():
    card = np.array(["c", "c", "c", "d"], dtype=object)
    data = make_data(
        {
            (1, "N"): jump(140, 2.0, 3.0, T3),
            (2, "N"): jump(140, 4.0, 6.0, T3),  # largest q(t-7)
            (3, "N"): jump(140, 4.0, 6.0, T3),  # ties with (2,"N"); higher pid
            (4, "N"): jump(140, 2.0, 3.0, T3),
        },
        140,
    )
    events, counts = sel.select_s3(data, card, window=[T3])
    assert [data.skus[r] for r in events["row"]] == [(2, "N"), (4, "N")]
    assert counts == {"n_crossings": 4, "n_after_same_day": 2, "n_accepted": 2}


def test_s3_cooldown_blocks_up_to_55_days_and_allows_56():
    n = 260
    # A and B share card x: A crosses at 100, B at 155 (blocked); C/D card y: 100 and 156.
    data = make_data(
        {
            (1, "N"): jump(n, 2.0, 3.0, 100, until=110),
            (2, "N"): jump(n, 2.0, 3.0, 155, until=170),
            (3, "N"): jump(n, 2.0, 3.0, 100, until=110),
            (4, "N"): jump(n, 2.0, 3.0, 156, until=170),
        },
        n,
    )
    card = np.array(["x", "x", "y", "y"], dtype=object)
    events, counts = sel.select_s3(data, card, window=list(range(95, 200)))
    got = [(int(t), data.skus[r]) for t, r in zip(events["origin_idx"], events["row"], strict=True)]
    assert got == [(100, (1, "N")), (100, (3, "N")), (156, (4, "N"))]
    assert counts["n_crossings"] == 4 and counts["n_after_same_day"] == 4
    assert counts["n_accepted"] == 3


def test_s3_confirmatory_rule():
    assert sel.s3_confirmatory(100, 12)
    assert not sel.s3_confirmatory(99, 12)
    assert not sel.s3_confirmatory(100, 11)


# --- ids and requests -------------------------------------------------------------------


def test_request_ids_blocks_horizons_and_contexts():
    n = 700
    data = make_data(
        {(7, None): flat(n, 4.0) * (1 + 0.001 * np.arange(n)), (8, "Foil"): flat(n, 2.0)}, n
    )
    s1 = pd.DataFrame(
        {
            "event_date": [pd.Timestamp("2024-02-08")] * 3,
            "k": [7, 14, 28],
            "origin_idx": [600, 610, 620],
            "row": [0, 0, 1],
            "floor_1": True,
        }
    )
    s2 = pd.DataFrame({"draw": [3], "origin_idx": [650], "row": [1], "price_band": ["1-5"]})
    s3 = pd.DataFrame({"origin_idx": [660], "row": [0]})
    out = sel.build_requests(data, s1, s2, s3)
    ids = out["request_id"].tolist()
    assert ids[0] == "S1|k7|2024-02-08|7|None"
    assert ids[2] == "S1|k28|2024-02-08|8|Foil"
    day = data.dates[650].date().isoformat()
    assert ids[3] == f"S2|{day}|8|Foil"
    assert ids[4] == f"S3|{data.dates[660].date().isoformat()}|7|None"
    assert out["study"].tolist() == ["S1", "S1", "S1", "S2", "S3"]
    assert out["block"].tolist() == [4, 1, 4, 2, 3]
    assert out["horizon"].tolist() == [83, 76, 62, 28, 56]
    for ctx in out["context"]:
        assert len(ctx) == 512 and ctx[-1] == 1.0 and ctx.dtype == np.float32


def test_requests_round_trip_through_the_frozen_schema(tmp_path):
    data = make_data({(7, None): flat(100, 4.0)}, 100)
    s1 = pd.DataFrame(
        {
            "event_date": [pd.Timestamp("2024-01-01")],
            "k": [14],
            "origin_idx": [50],
            "row": [0],
            "floor_1": [True],
        }
    )
    empty2 = pd.DataFrame(columns=["draw", "origin_idx", "row", "price_band"])
    empty3 = pd.DataFrame(columns=["origin_idx", "row"])
    out = sel.build_requests(data, s1, empty2, empty3)
    path = req_mod.write_requests(out, tmp_path / "r.parquet")
    back = req_mod.read_requests(path)
    assert back["request_id"].tolist() == out["request_id"].tolist()
    assert len(back["context"].iat[0]) == 51


# --- future poisoning -------------------------------------------------------------------

N_DAYS = 360
CUT = 250


def poison_panel(seed: int = 5):
    rng = np.random.default_rng(seed)
    n_sku = 36
    steps = 0.05 * rng.normal(size=(n_sku, N_DAYS))
    steps += np.where(rng.random((n_sku, N_DAYS)) < 0.02, np.log(1.7), 0.0)
    steps -= np.where(rng.random((n_sku, N_DAYS)) < 0.02, np.log(1.7), 0.0)
    wide = np.exp(np.log(3.0) + np.cumsum(steps, axis=1))
    wide[rng.random((n_sku, N_DAYS)) < 0.08] = np.nan
    skus = [(100 + i, ("Foil" if i % 2 else None)) for i in range(n_sku)]
    return skus, wide


def poisoned(wide, seed=9):
    rng = np.random.default_rng(seed)
    out = wide.copy()
    out[:, CUT + 1 :] = np.exp(rng.normal(1.0, 1.5, size=out[:, CUT + 1 :].shape))
    out[:, CUT + 1 :][rng.random(out[:, CUT + 1 :].shape) < 0.3] = np.nan
    return out


@pytest.fixture(scope="module")
def twin():
    skus, wide = poison_panel()
    dates = pd.date_range(START, periods=N_DAYS)
    a = sel.QuoteData.from_wide(dates, skus, wide)
    b = sel.QuoteData.from_wide(dates, skus, poisoned(wide))
    assert not np.array_equal(a.qmat[:, CUT + 1 :], b.qmat[:, CUT + 1 :], equal_nan=True)
    cards = np.array([f"card{i // 3}" for i in range(len(skus))], dtype=object)
    return a, b, cards


def old_two_sided_accept(pool: pd.DataFrame, cards) -> pd.DataFrame:
    """Reference of the REPLACED rule: candidates in draw order, rejected when a
    previously accepted decision of the same card is within |dt| < 28 days,
    on either side of the origin. Kept here only to show it is not causal."""
    kept: list[tuple[object, int]] = []
    flags = {}
    for c in pool.sort_values("draw", kind="mergesort").itertuples(index=False):
        card = cards[c.row]
        ok = all(not (k == card and abs(c.origin_idx - o) < 28) for k, o in kept)
        if ok:
            kept.append((card, c.origin_idx))
        flags[c.draw] = ok
    out = pool.copy()
    out["accepted"] = out["draw"].map(flags)
    return out


def test_poisoning_leaves_everything_up_to_the_cut_unchanged(twin):
    a, b, cards = twin
    window = list(range(100, N_DAYS))

    # S1: eligibility, flags and contexts of every origin <= CUT.
    events = pd.DataFrame(
        {
            "release_date": [a.dates[100], a.dates[200], a.dates[225], a.dates[300]],
            "group_ids": [[1], [1, 2], [2], [1]],
        }
    )
    group_of_row = np.array([1 + (i % 2) for i in range(len(a.skus))])
    ua, ca = sel.select_s1(a, events, group_of_row)
    ub, cb = sel.select_s1(b, events, group_of_row)
    ua_c, ub_c = ua[ua["origin_idx"] <= CUT], ub[ub["origin_idx"] <= CUT]
    assert 0 < len(ua_c) < len(ua)
    pd.testing.assert_frame_equal(ua_c, ub_c)
    pd.testing.assert_frame_equal(ca[ca["origin_idx"] <= CUT], cb[cb["origin_idx"] <= CUT])

    # S2: eligibility sets for dates <= CUT.
    ea, eb = sel.s2_eligibility_sets(a, window), sel.s2_eligibility_sets(b, window)
    upto = CUT - window[0] + 1
    assert sum(len(e) for e in ea[:upto]) > 0
    for x, y in zip(ea[:upto], eb[:upto], strict=True):
        assert x.tolist() == y.tolist()
    assert any(x.tolist() != y.tolist() for x, y in zip(ea[upto:], eb[upto:], strict=True))

    # S2 decisions with origin <= CUT, same pool for origins <= CUT.
    pool_a = sel.draw_s2_pool(ea, window, m=400)
    pool_b_later = sel.draw_s2_pool(eb, window, m=400)
    pool_b = pd.concat(
        [pool_a[pool_a["origin_idx"] <= CUT], pool_b_later[pool_b_later["origin_idx"] > CUT]],
        ignore_index=True,
    )
    assert not pool_a.equals(pool_b)
    new_a = sel.accept_s2(pool_a, cards)
    new_b = sel.accept_s2(pool_b, cards)
    keep = lambda f: f[f["origin_idx"] <= CUT].reset_index(drop=True)  # noqa: E731
    pd.testing.assert_frame_equal(keep(new_a), keep(new_b))
    assert keep(new_a)["accepted"].any() and not keep(new_a)["accepted"].all()
    da, db = sel.s2_decisions(a, new_a), sel.s2_decisions(b, new_b)
    pd.testing.assert_frame_equal(keep(da), keep(db))

    # The same test FAILS for the old two-sided rule.
    old_a, old_b = old_two_sided_accept(pool_a, cards), old_two_sided_accept(pool_b, cards)
    sort = ["origin_idx", "draw"]
    old_a_c = keep(old_a.sort_values(sort))
    old_b_c = keep(old_b.sort_values(sort))
    assert not old_a_c.equals(old_b_c)

    # S3: accepted events with t <= CUT.
    s3a, _ = sel.select_s3(a, cards, window)
    s3b, _ = sel.select_s3(b, cards, window)
    s3a_c, s3b_c = s3a[s3a["origin_idx"] <= CUT], s3b[s3b["origin_idx"] <= CUT]
    assert len(s3a_c) > 0
    pd.testing.assert_frame_equal(s3a_c.reset_index(drop=True), s3b_c.reset_index(drop=True))

    # Scaled contexts of all those units.
    ra = sel.build_requests(a, ua_c, keep(da), s3a_c)
    rb = sel.build_requests(b, ub_c, keep(db), s3b_c)
    assert ra["request_id"].tolist() == rb["request_id"].tolist()
    for x, y in zip(ra["context"], rb["context"], strict=True):
        np.testing.assert_array_equal(x, y)


def test_selection_is_deterministic(twin):
    a, _, cards = twin
    window = list(range(100, N_DAYS))
    e = sel.s2_eligibility_sets(a, window)
    p1, p2 = sel.draw_s2_pool(e, window, m=300), sel.draw_s2_pool(e, window, m=300)
    pd.testing.assert_frame_equal(sel.accept_s2(p1, cards), sel.accept_s2(p2, cards))


def test_census_has_counts_only(twin):
    a, _, cards = twin
    window = list(range(100, N_DAYS))
    events = pd.DataFrame({"release_date": [a.dates[100]], "group_ids": [[1]]})
    u, c = sel.select_s1(a, events, np.ones(len(a.skus), dtype=int))
    e = sel.s2_eligibility_sets(a, window)
    acc_ = sel.accept_s2(sel.draw_s2_pool(e, window, m=200), cards)
    dec = sel.s2_decisions(a, acc_)
    s3, counts = sel.select_s3(a, cards, window)
    census = sel.build_census(a, c, e, window, acc_, dec, s3, counts, cards, m=200)
    assert census["s2"]["candidates_drawn"] == 200
    assert census["s2"]["accepted_N"] + census["s2"]["rejected"] == 200
    assert sum(census["s2"]["pool_candidates_per_month"].values()) == 200
    assert sum(census["s2"]["accepted_per_month"].values()) == census["s2"]["accepted_N"]
    assert census["s3"]["N_events"] == len(s3)
    assert census["s3"]["confirmatory"] is False

    def scalars(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from scalars(v)
        elif isinstance(x, list):
            for v in x:
                yield from scalars(v)
        else:
            yield x

    assert all(isinstance(v, int | bool | str | type(None)) for v in scalars(census))


# --- loading a panel directory ------------------------------------------------------------


def _write_day(directory, day: str, rows):
    cols = list(zip(*rows, strict=True)) if rows else [(), (), (), (), ()]
    table = pa.table(
        {
            "product_id": pa.array(cols[0], pa.int32()),
            "group_id": pa.array(cols[1], pa.int32()),
            "subtype": pa.array(cols[2], pa.string()),
            "market": pa.array(cols[3], pa.float32()),
            "mid": pa.array(cols[4], pa.float32()),
        },
        schema=panel_mod.PANEL_SCHEMA,
    )
    pq.write_table(table, directory / f"{day}.parquet")


def test_load_quote_data_matches_wide_prices_of_load_panel(tmp_path):
    rng = np.random.default_rng(3)
    days = pd.date_range("2024-01-01", periods=150)
    for i, d in enumerate(days):
        if i == 70:
            continue  # a missing day file becomes a NaN column
        rows = []
        for pid, st in [(1, "Normal"), (1, "Foil"), (2, None), (3, "Normal"), (99, "Normal")]:
            if pid == 3 and i > 5:
                continue  # never valid after day 5; also no valid prices at all below
            market = float(rng.uniform(1, 5)) if rng.random() > 0.2 else None
            mid = float(rng.uniform(1, 5)) if market is None and rng.random() > 0.5 else None
            if pid == 3:
                market, mid = None, None
            rows.append((pid, 7, st, market, mid))
        _write_day(tmp_path, d.date().isoformat(), rows)
    got = sel.load_quote_data(tmp_path, [1, 2, 3], end="2024-05-29")
    long = panel_mod.load_panel(tmp_path, keys=[(1, "Normal"), (1, "Foil"), (2, None)])
    want = sel.QuoteData.from_long(long)
    assert got.skus == want.skus == [(1, "Foil"), (1, "Normal"), (2, None)]
    assert got.dates.equals(want.dates)
    np.testing.assert_array_equal(got.wide, want.wide)
    np.testing.assert_array_equal(got.qmat, want.qmat)
    assert np.isnan(got.wide[:, 70]).all()
    # `end` truncates: later days are never loaded.
    short = sel.load_quote_data(tmp_path, [1, 2], end="2024-02-10")
    assert len(short.dates) == 41
