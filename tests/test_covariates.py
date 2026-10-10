"""Pure covariate helpers (tfm3lab.covariates): contract, placebo, summary guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm3lab import covariates as cv


def test_availability_check_refuses_unknown_future():
    assert cv.require_future_known(cv.DAY_OF_WEEK) is cv.DAY_OF_WEEK
    with pytest.raises(cv.CovariateAvailabilityError):
        cv.require_future_known(cv.LEAKED_FUTURE)
    bad = cv.CovariateSpec("x", "unknown", False)
    with pytest.raises(cv.CovariateAvailabilityError):
        cv.require_future_known(bad)


def test_teaching_path_returns_real_past_and_future():
    v = np.arange(50.0)
    out = cv.teaching_only_leaked_covariate(v, 30, 10, 5)
    assert out.shape == (1, 15)
    np.testing.assert_array_equal(out[0], v[20:35])


def test_day_of_week_sincos():
    dates = pd.date_range("2026-01-05", periods=7)  # Monday..Sunday
    out = cv.day_of_week_covariate(dates)
    assert out.shape == (2, 7)
    np.testing.assert_allclose(out[:, 0], [0.0, 1.0], atol=1e-12)


def test_permutation_deterministic_nonidentity():
    p1, p2 = cv.dow_permutation(20261010), cv.dow_permutation(20261010)
    np.testing.assert_array_equal(p1, p2)
    assert sorted(p1) == list(range(7)) and not np.array_equal(p1, np.arange(7))
    dates = pd.date_range("2026-01-05", periods=14)
    placebo = cv.permuted_day_of_week_covariate(dates, p1)
    assert not np.allclose(placebo, cv.day_of_week_covariate(dates))
    np.testing.assert_allclose(
        placebo, cv.permuted_day_of_week_covariate(dates, cv.dow_permutation())
    )


def test_placebo_offset_deterministic_and_in_range():
    assert cv.placebo_offset_days() == cv.placebo_offset_days()
    offs = {cv.placebo_offset_days(s) for s in range(500)}
    assert min(offs) >= 60 and max(offs) <= 300
    assert len(offs) > 100


def test_shift_release_dates_circular_stays_in_span():
    rel = np.array(["2024-03-01", "2025-06-10"], dtype="datetime64[D]")
    out = cv.shift_release_dates_circular(rel, "2024-01-01", "2025-12-31", 100)
    assert out[0] == np.datetime64("2024-06-09")
    assert (out >= np.datetime64("2024-01-01")).all() and (out <= np.datetime64("2025-12-31")).all()
    wrapped = cv.shift_release_dates_circular(
        np.array(["2025-12-01"], dtype="datetime64[D]"), "2024-01-01", "2025-12-31", 100
    )
    assert wrapped[0] < np.datetime64("2025-12-01")  # wrapped around the circle


def test_days_to_release_uses_only_known_releases():
    rel = np.array(["2026-03-01", "2026-06-01"], dtype="datetime64[D]")
    dates = np.arange(np.datetime64("2026-02-20"), np.datetime64("2026-02-24"))
    # origin 2026-02-20: March release announced 60d earlier (Dec 31) -> known;
    # June release (announced ~Apr 2) -> not known yet.
    out = cv.days_to_next_release(dates, "2026-02-20", rel, announce_lead_days=60)
    np.testing.assert_array_equal(out[0], [9, 8, 7, 6])
    after = cv.days_to_next_release(
        np.array(["2026-03-02"], dtype="datetime64[D]"), "2026-02-20", rel, announce_lead_days=60
    )
    assert after[0, 0] == cv.MAX_DAYS_TO_RELEASE  # June not yet known
    later = cv.days_to_next_release(
        np.array(["2026-03-02"], dtype="datetime64[D]"), "2026-04-05", rel, announce_lead_days=60
    )
    assert later[0, 0] == 91
    explicit = cv.days_to_next_release(
        dates,
        "2026-02-20",
        rel,
        announced_at=np.array(["2026-02-21", "2026-01-01"], "datetime64[D]"),
    )
    assert explicit[0, 0] == 101  # March release not yet announced; June is


def _preds(include_leaked=False):
    rng = np.random.default_rng(0)
    rows = []
    for ser in range(6):
        for o in range(10):
            base = rng.uniform(0.5, 1.5)
            for arm, mult in (("no_covariate", 1.0), ("legit", 0.5), ("placebo", 1.0)):
                rows.append(
                    {"domain": "mtg", "series": f"c{ser}", "origin_index": o, "horizon_step": 7,
                     "arm": arm, "observed": True, "abs_err_norm": base * mult}
                )
    df = pd.DataFrame(rows)
    if include_leaked:
        df = pd.concat([df, df.assign(arm="leaked")])
    return df


def test_summary_helps_when_both_cis_positive():
    s = cv.paired_skill_summary(_preds(), primary_horizon=7, n_boot=200)
    primary = s[s["is_primary"]]
    assert len(primary) == 2
    row = primary[primary["comparison"] == "legit_vs_no_covariate"].iloc[0]
    assert row["skill_delta"] == pytest.approx(0.5)
    placebo = primary[primary["comparison"] == "legit_vs_placebo"].iloc[0]
    assert placebo["ci_low"] > 0 and row["ci_low"] > 0
    assert (s["verdict"] == "helps").all()
    assert (s["label"] == "small-cluster/exploratory").all()
    assert "leaked" not in set(s.get("arm", []))


def test_summary_raises_on_leaked_rows():
    with pytest.raises(cv.LeakedRowsError):
        cv.paired_skill_summary(_preds(include_leaked=True), primary_horizon=7, n_boot=50)
    wm = _preds().assign(watermark=cv.LEAKED_WATERMARK)
    with pytest.raises(cv.LeakedRowsError):
        cv.paired_skill_summary(wm, primary_horizon=7, n_boot=50)


def test_verdict_wording():
    assert cv.verdict_from_cis([(0.1, 0.3), (0.05, 0.2)]) == "helps"
    assert cv.verdict_from_cis([(0.1, 0.3), (-0.1, 0.2)]) == "inconclusive"
    assert cv.verdict_from_cis([(0.1, 0.3), (-0.3, -0.1)]) == "does not help"
    assert cv.verdict_from_cis([(float("nan"), 0.3), (0.1, 0.2)]) == "inconclusive"
