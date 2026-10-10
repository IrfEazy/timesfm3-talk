"""Offline, deterministic tests for tfm3lab.bootstrap (Preregistration v2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tfm3lab import bootstrap as bs
from tfm3lab.summarize import aggregate_leaderboard


def _panel(n_cards=5, n_days=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for i in range(n_cards):
        start = i * 3  # unequal sizes
        for d in dates[start:]:
            rows.append((f"card{i}", d, abs(rng.normal()) + 0.1))
    df = pd.DataFrame(rows, columns=["card", "date", "naive"])
    df["model"] = df["naive"] * rng.uniform(0.5, 1.5, size=len(df))
    return df


# --- defaults and basic shape -----------------------------------------------


def test_defaults_match_preregistration():
    assert bs.DEFAULT_N_BOOT == 2000
    assert bs.DEFAULT_SEED == 20261010
    assert bs.DEFAULT_CI == 0.95
    assert bs.block_days_for_horizon(1) == 7
    assert bs.block_days_for_horizon(28) == 28


def test_cluster_result_fields():
    df = _panel()
    res = bs.paired_cluster_bootstrap(df["model"], df["naive"], df["card"], n_boot=50)
    assert set(res) == {"median", "weighted_mean", "pooled"}
    r = res["median"]
    assert r.n_boot == 50 and r.seed == 20261010 and r.n_invalid == 0
    assert r.n_clusters == 5 and r.method == "cluster"
    assert r.ci_low <= r.ci_high
    assert r.replicates.shape == (50,)
    assert "replicates" not in r.to_dict()


# --- invariants ---------------------------------------------------------------


@pytest.mark.parametrize("method", ["cluster", "calendar", "two_stage"])
def test_identical_errors_give_zero_skill_in_every_replicate(method):
    df = _panel()
    res = _run(method, df["naive"], df["naive"], df)
    for r in res.values():
        assert r.estimate == 0.0
        np.testing.assert_allclose(r.replicates, 0.0, atol=1e-12)


@pytest.mark.parametrize("method", ["cluster", "calendar", "two_stage"])
def test_proportional_errors_give_one_minus_c_in_every_replicate(method):
    df = _panel()
    c = 0.8
    res = _run(method, c * df["naive"], df["naive"], df)
    for r in res.values():
        assert r.estimate == pytest.approx(1 - c)
        np.testing.assert_allclose(r.replicates, 1 - c, atol=1e-12)


def _run(method, model, naive, df, **kw):
    kw.setdefault("n_boot", 100)
    if method == "cluster":
        return bs.paired_cluster_bootstrap(model, naive, df["card"], **kw)
    if method == "calendar":
        return bs.calendar_block_bootstrap(model, naive, df["card"], df["date"], 7, **kw)
    return bs.two_stage_bootstrap(model, naive, df["card"], df["date"], 7, **kw)


def test_point_estimate_weighted_mean_matches_aggregate_leaderboard():
    df = _panel()
    g = df.groupby("card")
    per = pd.DataFrame(
        {
            "series": g.size().index,
            "n": g.size().to_numpy(),
            "relative_mae_vs_baseline": (g["model"].sum() / g["naive"].sum()).to_numpy(),
        }
    )
    per["mode"] = "m"
    agg = aggregate_leaderboard(per).iloc[0]
    res = bs.paired_cluster_bootstrap(df["model"], df["naive"], df["card"], n_boot=10)
    assert res["weighted_mean"].estimate == pytest.approx(agg["skill_mean_weighted"])
    assert res["median"].estimate == pytest.approx(agg["skill_median"])
    cal = bs.calendar_block_bootstrap(
        df["model"], df["naive"], df["card"], df["date"], 7, n_boot=10
    )
    assert cal["weighted_mean"].estimate == pytest.approx(agg["skill_mean_weighted"])
    assert cal["median"].estimate == pytest.approx(agg["skill_median"])


def test_unequal_cluster_sizes_weighted_vs_pooled_hand_computed():
    # card A: 1 row, model 1 / naive 2 -> r=0.5, n=1
    # card B: 3 rows, model sum 9 / naive sum 3 -> r=3, n=3
    model = [1.0, 3.0, 3.0, 3.0]
    naive = [2.0, 1.0, 1.0, 1.0]
    ids = ["A", "B", "B", "B"]
    res = bs.paired_cluster_bootstrap(model, naive, ids, n_boot=10)
    # weighted: 1 - (1*0.5 + 3*3)/4 = 1 - 2.375
    assert res["weighted_mean"].estimate == pytest.approx(1 - 2.375)
    # pooled: 1 - 10/5 = -1
    assert res["pooled"].estimate == pytest.approx(-1.0)
    # median of {0.5, 3} = 1.75
    assert res["median"].estimate == pytest.approx(1 - 1.75)


def test_duplicate_cluster_draws_keep_multiplicity():
    model_sum = np.array([1.0, 9.0, 4.0])
    base_sum = np.array([2.0, 3.0, 4.0])
    n = np.array([1.0, 3.0, 2.0])
    mult = np.array([[3, 0, 0], [0, 2, 1], [1, 1, 1], [2, 1, 0]])
    got = bs.skill_from_multiplicities(model_sum, base_sum, n, mult)
    r = model_sum / base_sum
    for b, row in enumerate(mult):
        idx = np.repeat(np.arange(3), row)  # explicit duplicated draws
        assert got["median"][b] == pytest.approx(1 - np.median(r[idx]))
        assert got["weighted_mean"][b] == pytest.approx(1 - np.average(r[idx], weights=n[idx]))
        assert got["pooled"][b] == pytest.approx(1 - model_sum[idx].sum() / base_sum[idx].sum())
    # [2,1,0] must differ from [1,1,0]-style collapse: median of {0.5,0.5,3}=0.5
    assert got["median"][3] == pytest.approx(0.5)


def test_cluster_multiplicities_rows_sum_to_g_and_are_uniform():
    m = bs.cluster_multiplicities(4, 20000, np.random.default_rng(1))
    assert (m.sum(axis=1) == 4).all()
    np.testing.assert_allclose(m.mean(axis=0), 1.0, atol=0.03)


def test_pairing_model_and_baseline_share_draws():
    # If model and baseline were resampled independently, proportional
    # errors could not give exactly 1-c in every replicate; additionally the
    # pooled replicate must equal the hand-computed value for the SAME draw.
    df = _panel()
    _, m_sum, b_sum, n = bs.per_cluster_sums(df["model"], df["naive"], df["card"])
    rng = np.random.default_rng(5)
    mult = bs.cluster_multiplicities(len(m_sum), 30, np.random.default_rng(5))
    res = bs.paired_cluster_bootstrap_from_sums(m_sum, b_sum, n, n_boot=30, rng=rng)
    np.testing.assert_allclose(res["pooled"].replicates, 1 - (mult @ m_sum) / (mult @ b_sum))
    assert res["pooled"].seed is None  # an explicit rng is not a recorded seed


def test_reproducible_with_seed_and_row_order_invariant():
    df = _panel()
    a = bs.paired_cluster_bootstrap(df["model"], df["naive"], df["card"], n_boot=200, seed=7)
    shuffled = df.sample(frac=1.0, random_state=3)
    b = bs.paired_cluster_bootstrap(
        shuffled["model"], shuffled["naive"], shuffled["card"], n_boot=200, seed=7
    )
    for k in a:
        # same draws; only floating-point summation order may differ
        np.testing.assert_allclose(a[k].replicates, b[k].replicates, rtol=1e-12, atol=1e-15)
    again = bs.paired_cluster_bootstrap(df["model"], df["naive"], df["card"], n_boot=200, seed=7)
    np.testing.assert_array_equal(a["median"].replicates, again["median"].replicates)
    c = bs.calendar_block_bootstrap(df["model"], df["naive"], df["card"], df["date"], 28, seed=7,
                                    n_boot=100)
    d = bs.calendar_block_bootstrap(df["model"], df["naive"], df["card"], df["date"], 28, seed=7,
                                    n_boot=100)
    np.testing.assert_array_equal(c["median"].replicates, d["median"].replicates)
    assert c["median"].block_days == 28
    e = bs.paired_cluster_bootstrap(df["model"], df["naive"], df["card"], n_boot=200, seed=8)
    assert not np.array_equal(a["median"].replicates, e["median"].replicates)


def test_percentile_interval_counts_invalid_replicates():
    low, high, n_invalid = bs.percentile_interval([0.0, 1.0, np.nan, np.inf, 2.0], ci=0.5)
    assert n_invalid == 2
    assert (low, high) == (0.5, 1.5)


def test_zero_baseline_card_is_reported_invalid_not_dropped():
    model = [1.0, 1.0, 2.0]
    naive = [0.0, 1.0, 1.0]
    res = bs.paired_cluster_bootstrap(model, naive, ["A", "B", "C"], n_boot=200, seed=1)
    wm = res["weighted_mean"]
    assert wm.estimate == -np.inf
    # replicates that drew card A are non-finite and counted
    assert wm.n_invalid > 0
    assert wm.n_invalid + np.isfinite(wm.replicates).sum() == 200


def test_rejects_nan_errors():
    with pytest.raises(ValueError, match="non-finite"):
        bs.paired_cluster_bootstrap([1.0, np.nan], [1.0, 1.0], ["a", "b"], n_boot=5)


# --- calendar blocks ------------------------------------------------------------


def test_calendar_block_multiplicities_circular_blocks():
    w = bs.calendar_block_multiplicities(10, 4, 500, np.random.default_rng(2))
    assert w.shape == (500, 10)
    assert (w.sum(axis=1) == 10).all()


def test_calendar_block_multiplicities_structure_single_replicate():
    # Reconstruct with the same generator stream: 3 blocks of length 4 on a
    # 10-day circle, truncated to 10 days.
    rng = np.random.default_rng(11)
    starts = np.random.default_rng(11).integers(0, 10, size=(1, 3))[0]
    w = bs.calendar_block_multiplicities(10, 4, 1, rng)[0]
    days = np.concatenate([(s + np.arange(4)) % 10 for s in starts])[:10]
    np.testing.assert_array_equal(w, np.bincount(days, minlength=10))


def test_common_calendar_shares_block_ids_across_cards():
    # Two cards with identical day-by-day losses must have identical
    # resampled sums in EVERY replicate: same day weights for all cards.
    dates = pd.date_range("2025-01-01", periods=40, freq="D")
    vals = np.random.default_rng(4).uniform(0.5, 2.0, size=40)
    ids = np.array(["x"] * 40 + ["y"] * 40)
    d = np.concatenate([dates, dates])
    v = np.concatenate([vals, vals])
    panel = bs.calendar_panel(v, v, ids, d)
    w = bs.calendar_block_multiplicities(40, 7, 300, np.random.default_rng(9))
    m, b, n = bs.calendar_block_resample_sums(panel, w)
    np.testing.assert_allclose(m[:, 0], m[:, 1])
    np.testing.assert_allclose(n[:, 0], n[:, 1])
    assert np.std(m[:, 0]) > 0


def test_calendar_gap_days_travel_with_the_calendar():
    # card "late" only exists on the last 5 of 20 days; a replicate that
    # draws no such day leaves it out (count 0) instead of crashing.
    dates = pd.date_range("2025-01-01", periods=20, freq="D")
    ids = ["early"] * 20 + ["late"] * 5
    d = list(dates) + list(dates[-5:])
    err = np.ones(25)
    panel = bs.calendar_panel(err, err, ids, d)
    assert panel.count.shape == (2, 20)
    assert panel.count[1, :15].sum() == 0
    res = bs.calendar_block_bootstrap(err * 0.5, err, ids, d, 7, n_boot=200)
    np.testing.assert_allclose(res["median"].replicates, 0.5)
    assert res["median"].n_invalid == 0
    # excluded card-replicates are counted: recompute from the same draws
    w = bs.calendar_block_multiplicities(20, 7, 200, np.random.default_rng(3))
    _, _, n = bs.calendar_block_resample_sums(panel, w)
    expected = int((n <= 0).sum())
    assert expected > 0
    res2 = bs.calendar_block_bootstrap(
        err * 0.5, err, ids, d, 7, n_boot=200, rng=np.random.default_rng(3)
    )
    assert res2["median"].n_excluded_card_reps == expected
    assert res2["pooled"].n_excluded_card_reps == expected
    # cluster designs never exclude cards
    assert bs.paired_cluster_bootstrap_from_sums(
        [1.0, 2.0], [2.0, 4.0], [3.0, 3.0], n_boot=50
    )["median"].n_excluded_card_reps == 0


# --- coverage ---------------------------------------------------------------------


def _coverage_data(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=50, freq="D")
    ids = np.repeat([f"c{i}" for i in range(6)], 50)
    d = np.tile(dates, 6)
    covered = rng.uniform(size=300) < 0.75
    return covered, ids, d


def test_coverage_cis_contain_estimate_region_and_are_reproducible():
    covered, ids, d = _coverage_data()
    cal = bs.coverage_calendar_block_ci(covered, ids, d, 7, n_boot=300)
    clu = bs.coverage_cluster_ci(covered, ids, n_boot=300)
    assert cal.estimate == pytest.approx(covered.mean()) == clu.estimate
    assert cal.ci_low <= cal.estimate <= cal.ci_high
    assert cal.block_days == 7 and cal.method == "calendar_block"
    assert clu.method == "cluster" and clu.n_clusters == 6
    again = bs.coverage_calendar_block_ci(covered, ids, d, 7, n_boot=300)
    np.testing.assert_array_equal(cal.replicates, again.replicates)


def test_coverage_all_covered_and_empty_policies():
    _, ids, d = _coverage_data()
    allc = np.ones(len(ids), dtype=bool)
    for r in (
        bs.coverage_calendar_block_ci(allc, ids, d, 7, n_boot=50),
        bs.coverage_cluster_ci(allc, ids, n_boot=50),
    ):
        assert (r.estimate, r.ci_low, r.ci_high, r.n_invalid) == (1.0, 1.0, 1.0, 0)
    empty = bs.coverage_cluster_ci(np.array([], dtype=bool), np.array([]), n_boot=50)
    assert np.isnan(empty.estimate) and empty.n_invalid == 50
    empty_cal = bs.coverage_calendar_block_ci(
        np.array([], dtype=bool), np.array([]), np.array([], dtype="datetime64[D]"), 7, n_boot=5
    )
    assert np.isnan(empty_cal.ci_low) and empty_cal.n_invalid == 5
    w = bs.coverage_wilson_design_adjusted(allc, ids)
    assert w.policy == "degenerate_iid" and w.ci_high == 1.0 and w.n_eff == len(ids)
    assert bs.coverage_wilson_design_adjusted([], []).policy == "empty"
    assert bs.coverage_wilson_design_adjusted([True, False], ["a", "a"]).policy == "single_cluster"


def test_wilson_design_adjusted_hand_computed():
    # clusters: k = (3, 1, 2), n = (4, 4, 2); N = 10, p = 0.6
    covered = [1, 1, 1, 0] + [1, 0, 0, 0] + [1, 1]
    ids = ["a"] * 4 + ["b"] * 4 + ["c"] * 2
    res = bs.coverage_wilson_design_adjusted(covered, ids, ci=0.95)
    p, n_total, g = 0.6, 10, 3
    resid = np.array([3 - p * 4, 1 - p * 4, 2 - p * 2])
    var = g / (g - 1) * np.sum(resid**2) / n_total**2
    n_eff = p * (1 - p) / var
    z = stats.norm.ppf(0.975)
    t = stats.t.ppf(0.975, df=g - 1)
    n_adj = n_eff * (z / t) ** 2
    center = (p + z**2 / (2 * n_adj)) / (1 + z**2 / n_adj)
    half = z * np.sqrt(p * (1 - p) / n_adj + z**2 / (4 * n_adj**2)) / (1 + z**2 / n_adj)
    assert res.policy == "ok"
    assert res.n_eff == pytest.approx(n_eff)
    assert res.n_eff_adjusted == pytest.approx(n_adj)
    assert res.ci_low == pytest.approx(max(0.0, center - half))
    assert res.ci_high == pytest.approx(min(1.0, center + half))


def test_two_stage_is_labelled_diagnostic():
    df = _panel()
    res = bs.two_stage_bootstrap(df["model"], df["naive"], df["card"], df["date"], 7, n_boot=20)
    assert res["median"].method == "two_stage_diagnostic"
    assert "DIAGNOSTIC ONLY" in bs.two_stage_bootstrap.__doc__
