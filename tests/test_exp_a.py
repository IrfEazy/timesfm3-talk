"""Offline tests of tfm3lab.exp_a with synthetic inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm3lab import exp_a

NB = 200


def make_primary(
    cards, ratio_by_card, horizons=(28, 56), n_orig=30, seed=0, scale_col="error_scale_valid"
):
    rng = np.random.default_rng(seed)
    rows = []
    for c in cards:
        for h in horizons:
            for k in range(n_orig):
                base = rng.uniform(0.5, 1.5)
                o = pd.Timestamp("2025-01-01") + pd.Timedelta(days=7 * k)
                rows.append(
                    {
                        "card": c,
                        "origin_date": o,
                        "target_date": o + pd.Timedelta(days=h),
                        "requested_horizon": h,
                        "horizon": h,
                        "observed": True,
                        scale_col: True,
                        "ae_model_scaled": base * ratio_by_card[c] * rng.uniform(0.9, 1.1),
                        "ae_baseline_naive_scaled": base,
                    }
                )
    return pd.DataFrame(rows)


def make_selfcheck(fail=False):
    rows = []
    for sc in ("baseline", "stress"):
        for g in (8, 20, 40):
            for st in ("median", "weighted_mean", "pooled"):
                cov = 0.95 if sc == "baseline" else 0.8 + g / 1000
                if fail and sc == "baseline" and g == 40:
                    cov = 0.85
                rows.append(
                    dict(
                        scenario=sc,
                        n_clusters=g,
                        statistic=st,
                        coverage=cov,
                        used_for_acceptance=(sc == "baseline" and g == 40 and st != "pooled"),
                        acceptance_low=0.92,
                        acceptance_high=0.98,
                    )
                )
    return pd.DataFrame(rows)


CARDS = [f"c{i}" for i in range(24)]


def headline(ratios, showcase=frozenset(), selfcheck=None, **kw):
    primary = make_primary(CARDS, ratios)
    return exp_a.headline_table(
        primary,
        set(showcase),
        selfcheck if selfcheck is not None else make_selfcheck(),
        n_boot=NB,
        **kw,
    )


def verdicts(df, subset="pool"):
    return df[df["subset"] == subset].groupby("horizon")["verdict"].first().to_dict()


def test_verdict_rule_branches():
    ok = {"median": (0.1, 0.2), "weighted_mean": (0.05, 0.3)}
    assert exp_a.verdict_from_cis(ok) == exp_a.BEATS
    assert (
        exp_a.verdict_from_cis({"median": (-0.3, -0.1), "weighted_mean": (-0.2, -0.01)})
        == exp_a.LOSES
    )
    assert (
        exp_a.verdict_from_cis({"median": (0.1, 0.2), "weighted_mean": (-0.1, 0.3)})
        == exp_a.INCONCLUSIVE
    )
    assert (
        exp_a.verdict_from_cis({"median": (np.nan, np.nan), "weighted_mean": (0.1, 0.2)})
        == exp_a.INCONCLUSIVE
    )


def test_headline_beats_loses_inconclusive():
    assert set(verdicts(headline({c: 0.6 for c in CARDS})).values()) == {exp_a.BEATS}
    assert set(verdicts(headline({c: 1.6 for c in CARDS})).values()) == {exp_a.LOSES}
    mixed = {c: (0.6 if i % 2 else 1.5) for i, c in enumerate(CARDS)}
    assert set(verdicts(headline(mixed)).values()) == {exp_a.INCONCLUSIVE}


def test_headline_columns_and_pooled_is_secondary():
    df = headline({c: 0.6 for c in CARDS})
    pool = df[df["subset"] == "pool"]
    assert set(pool["statistic"]) == {"median", "weighted_mean", "pooled"}
    assert set(pool["horizon"]) == {28, 56}
    pooled = pool[pool["statistic"] == "pooled"]
    assert (pooled["secondary_label"] != "").all() and not pooled["is_decision_statistic"].any()
    assert pool["cal_ci_low"].notna().all() and (pool["seed"] == 20261010).all()


def test_showcase_exclusion_and_descriptive_subset():
    showcase = {"c0", "c1", "c2"}
    ratios = {c: (1.6 if c in showcase else 0.6) for c in CARDS}
    df = headline(ratios, showcase=showcase)
    n = df.groupby(["subset", "horizon"])["n_cards"].first()
    assert (
        n[("pool", 28)] == 24 and n[("excl_showcase", 28)] == 21 and n[("showcase_only", 28)] == 3
    )
    assert verdicts(df, "excl_showcase")[28] == exp_a.BEATS
    assert verdicts(df, "showcase_only")[28] == "descriptive (no verdict)"
    # disagreement is flagged, not resolved
    shown = df[df["subset"] == "excl_showcase"]
    assert (
        shown["verdict_disagrees_with_pool"] == (shown["verdict"] != shown["pool_verdict"])
    ).all()


def test_showcase_descriptive_direction_flags_disagreement():
    showcase = {"c0", "c1", "c2"}
    unlike = {c: (1.6 if c in showcase else 0.6) for c in CARDS}
    df = headline(unlike, showcase=showcase)
    sc = df[df["subset"] == "showcase_only"]
    assert (sc["descriptive_direction"] == exp_a.LOSES).all()
    assert sc["verdict_disagrees_with_pool"].all()
    alike = headline({c: 0.6 for c in CARDS}, showcase=showcase)
    sc = alike[alike["subset"] == "showcase_only"]
    assert (sc["descriptive_direction"] == exp_a.BEATS).all()
    assert not sc["verdict_disagrees_with_pool"].any()
    assert not alike[alike["subset"] == "excl_showcase"]["verdict_disagrees_with_pool"].any()


def test_headline_requires_single_cell_and_unique_keys():
    primary = make_primary(CARDS, {c: 0.6 for c in CARDS})
    primary["context_len"] = 512
    primary.loc[primary.index[:5], "context_len"] = 256
    with pytest.raises(ValueError, match="exactly one cell"):
        exp_a.headline_table(primary, set(), make_selfcheck(), n_boot=NB)
    dup = make_primary(CARDS, {c: 0.6 for c in CARDS})
    dup = pd.concat([dup, dup.iloc[:3]])
    with pytest.raises(ValueError, match="duplicate"):
        exp_a.headline_table(dup, set(), make_selfcheck(), n_boot=NB)


def test_pool_coverage_columns():
    primary = make_primary(CARDS[:20], {c: 0.6 for c in CARDS})
    df = exp_a.headline_table(primary, set(), make_selfcheck(), pool_cards=set(CARDS), n_boot=NB)
    assert (df["n_pool_csv"] == 24).all() and (df["n_cards_in_export"] == 20).all()
    assert (df["n_pool_cards_missing"] == 4).all()
    assert exp_a.pool_coverage(primary, set(CARDS))["missing"] == CARDS[20:]


def test_invalid_scale_and_unobserved_rows_excluded():
    primary = make_primary(CARDS, {c: 0.6 for c in CARDS})
    primary.loc[primary["card"] == "c0", "error_scale_valid"] = False
    primary.loc[primary["card"] == "c1", "observed"] = False
    df = exp_a.headline_table(primary, set(), make_selfcheck(), n_boot=NB)
    assert df[df["subset"] == "pool"]["n_cards"].iloc[0] == 22


def test_legacy_scale_valid_column_accepted():
    primary = make_primary(CARDS, {c: 0.6 for c in CARDS}, scale_col="scale_valid")
    df = exp_a.headline_table(primary, set(), make_selfcheck(), n_boot=NB)
    assert (df[df["subset"] == "pool"]["verdict"] == exp_a.BEATS).all()


def test_selfcheck_lookup_closest_g_and_labels():
    sc = make_selfcheck()
    r = exp_a.selfcheck_coverage(sc, "median", 38)
    assert r["selfcheck_g"] == 40 and r["selfcheck_baseline_coverage"] == 0.95
    assert r["selfcheck_stress_coverage"] == pytest.approx(0.84)
    assert r["ci_label"] == ""
    r = exp_a.selfcheck_coverage(sc, "median", 12)  # |12-8|=4 < |12-20|=8
    assert r["selfcheck_g"] == 8 and "small-cluster" in r["ci_label"]
    assert exp_a.selfcheck_coverage(sc, "median", 14)["selfcheck_g"] == 20  # tie -> larger G
    bad = exp_a.selfcheck_coverage(make_selfcheck(fail=True), "median", 40)
    assert bad["selfcheck_acceptance_failed"] and "approximate" in bad["ci_label"]


def test_headline_rows_carry_selfcheck_and_approximate():
    df = headline({c: 0.6 for c in CARDS}, selfcheck=make_selfcheck(fail=True))
    pool = df[df["subset"] == "pool"]
    assert (pool["selfcheck_g"] == 20).all()  # 24 cards -> closest G is 20
    assert pool["selfcheck_baseline_coverage"].notna().all()
    assert pool["selfcheck_stress_coverage"].notna().all()
    assert pool["ci_label"].str.contains("approximate").all()


def make_leaderboard(cards, placebo_cards):
    rows = []
    rng = np.random.default_rng(1)
    specs = [
        ("timesfm3_univariate", "raw", True, 512),
        ("timesfm3_univariate", "raw", True, 128),
        ("timesfm3_univariate", "log1p", True, 512),
        ("timesfm3_univariate", "raw", False, 512),
        ("timesfm3_multivariate", "raw", True, 512),
        ("timesfm3_multivariate_placebo", "raw", True, 512),
    ]
    for mode, tr, mp, ctx in specs:
        cs = placebo_cards if mode.endswith("placebo") else cards
        for c in cs:
            for h in (28, 56):
                for b in ("naive", "seasonal_naive", "drift", "ets"):
                    base = 10.0 * rng.uniform(0.5, 2)
                    rows.append(
                        dict(
                            baseline_method=b,
                            mode=mode,
                            transform=tr,
                            make_positive=mp,
                            context_len=ctx,
                            requested_horizon=h,
                            series=c,
                            horizon_step=h,
                            n=20,
                            mae_model=base * rng.uniform(0.6, 1.0),
                            mae_baseline=base,
                            dm_stat=-1.0,
                            dm_pvalue=float(rng.uniform(0, 0.2)),
                        )
                    )
                rows.append(
                    dict(
                        baseline_method="naive",
                        mode=mode,
                        transform=tr,
                        make_positive=mp,
                        context_len=ctx,
                        requested_horizon=h,
                        series=c,
                        horizon_step=1,
                        n=20,
                        mae_model=1.0,
                        mae_baseline=1.0,
                        dm_stat=0.0,
                        dm_pvalue=0.5,
                    )
                )
    return pd.DataFrame(rows)


def test_secondary_pairing_and_comparisons():
    lb = make_leaderboard(CARDS, CARDS[:7])
    sec = exp_a.secondary_table(lb, n_boot=NB)
    names = set(sec["comparison"])
    for expected in (
        "primary_vs_ets",
        "primary_vs_drift",
        "primary_vs_seasonal_naive",
        "context_128_vs_primary",
        "log1p_vs_naive",
        "make_positive_false_vs_primary",
        "multivariate_vs_primary",
        "multivariate_vs_placebo",
    ):
        assert expected in names
    assert "context_64_vs_naive" not in names  # absent cells skipped
    placebo = sec[sec["comparison"] == "multivariate_vs_placebo"]
    assert (placebo["n_cards"] == 7).all()  # paired on matching cards only
    assert (placebo["n_cards_only_in_model"] == 17).all()
    assert (sec["verdict"] == "descriptive (no verdict)").all()
    assert set(sec["horizon"]) == {28, 56}
    # paired vs-baseline estimate equals 1 - weighted ratio of loss sums by construction
    row = sec[
        (sec["comparison"] == "primary_vs_ets")
        & (sec["statistic"] == "pooled")
        & (sec["horizon"] == 28)
    ].iloc[0]
    sel = lb[
        (lb["mode"] == "timesfm3_univariate")
        & (lb["transform"] == "raw")
        & lb["make_positive"]
        & (lb["context_len"] == 512)
        & (lb["baseline_method"] == "ets")
        & (lb["requested_horizon"] == 28)
        & (lb["horizon_step"] == 28)
    ]
    assert row["estimate"] == pytest.approx(
        1 - (sel.mae_model * sel.n).sum() / (sel.mae_baseline * sel.n).sum()
    )


def test_secondary_loss_sums_use_leaderboard_n():
    lb = make_leaderboard(CARDS[:5], CARDS[:5])
    sel = (
        (lb["mode"] == "timesfm3_univariate")
        & (lb["transform"] == "raw")
        & lb["make_positive"]
        & (lb["context_len"] == 512)
        & (lb["baseline_method"] == "ets")
        & (lb["horizon_step"] == lb["requested_horizon"])
    )
    lb.loc[sel & (lb["series"] == "c0"), "n"] = 200  # n drives the weighting
    sec = exp_a.secondary_table(lb, n_boot=NB)
    assert (sec["n_definition"] == exp_a.N_DEFINITION).all()
    assert "no error-scale filter" in exp_a.N_DEFINITION
    row = sec[
        (sec["comparison"] == "primary_vs_ets")
        & (sec["statistic"] == "pooled")
        & (sec["horizon"] == 28)
    ].iloc[0]
    s = lb[sel & (lb["requested_horizon"] == 28)]
    assert row["estimate"] == pytest.approx(
        1 - (s.mae_model * s.n).sum() / (s.mae_baseline * s.n).sum()
    )


def test_secondary_drops_cards_with_mismatched_n():
    lb = make_leaderboard(CARDS[:5], CARDS[:5])
    mask = (lb["mode"] == "timesfm3_multivariate_placebo") & (lb["series"] == "c0")
    lb.loc[mask, "n"] = 19
    sec = exp_a.secondary_table(lb, n_boot=NB)
    p = sec[sec["comparison"] == "multivariate_vs_placebo"]
    assert (p["n_cards"] == 4).all() and (p["n_cards_dropped_n_mismatch"] == 1).all()


def test_significance_families_and_nan_counts():
    lb = make_leaderboard(CARDS, CARDS[:7])
    lb.loc[lb.index[:3], "dm_pvalue"] = np.nan
    sig = exp_a.significance_frame(lb)
    assert (sig["requested_horizon"].isin({28, 56})).all()
    assert {
        "dm_qvalue_bh",
        "dm_qvalue_by",
        "dm_reject_bh",
        "dm_reject_by",
        "family_size",
        "family_n_nan",
    } <= set(sig.columns)
    assert sig["family_n_nan"].sum() > 0
    fam = sig.groupby(list(exp_a.SIGNIFICANCE_FAMILY))["family_size"].first()
    assert (fam == 2 * 24).any() and (fam <= 48).all()  # card x horizon per family
