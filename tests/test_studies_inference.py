"""Preregistered inference primitives, on synthetic numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm3lab import bootstrap
from tfm3lab.studies import inference as inf


def test_constants():
    assert pytest.approx(0.05 / 3) == inf.ALPHA_CONFIRMATORY
    assert pytest.approx(0.98333333) == inf.CI_CONFIRMATORY
    assert inf.CI_EXPLORATORY == 0.95
    assert inf.B == 2000
    assert inf.SEED == 20261010


def test_t_interval_known_answer():
    mean, low, high = inf.t_interval([1, 2, 3, 4, 5], 0.95)
    half = 2.7764451051977987 * np.sqrt(2.5) / np.sqrt(5)
    assert mean == 3.0
    assert low == pytest.approx(3 - half)
    assert high == pytest.approx(3 + half)


def test_t_interval_needs_two_units_and_valid_level():
    with pytest.raises(ValueError):
        inf.t_interval([1.0], 0.95)
    with pytest.raises(ValueError):
        inf.t_interval([1.0, 2.0], 1.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_inputs_raise(bad):
    d = [1.0, 2.0, bad]
    with pytest.raises(ValueError):
        inf.t_interval(d, 0.95)
    with pytest.raises(ValueError):
        inf.sign_flip_test(d)
    with pytest.raises(ValueError):
        inf.cluster_bootstrap_mean(d, [0, 0, 1])
    with pytest.raises(ValueError):
        inf.calendar_block_bootstrap_mean(d, pd.date_range("2024-01-01", periods=3), 7)
    with pytest.raises(ValueError):
        inf.s1_verdict(d)


def test_sign_flip_exact_tiny_g():
    assert inf.sign_flip_test([1.0, 2.0, 3.0]) == 0.25
    assert inf.sign_flip_test([1.0, 1.0]) == 0.5
    assert inf.sign_flip_test([1.0]) == 1.0
    assert inf.sign_flip_test([0.0, 0.0]) == 1.0


def test_sign_flip_exact_ignores_seed():
    d = np.arange(1.0, 9.0)
    assert inf.sign_flip_test(d, 1) == inf.sign_flip_test(d, 2) == 2 / 2**8


def test_sign_flip_monte_carlo_branch_deterministic():
    rng = np.random.default_rng(0)
    d = rng.normal(0.3, 1.0, size=30)
    a, b = inf.sign_flip_test(d, 7), inf.sign_flip_test(d, 7)
    assert a == b
    assert 0 < a <= 1
    assert inf.sign_flip_test(d, 8) != a
    assert inf.sign_flip_test(np.full(30, 1.0), 1) == pytest.approx(1 / 20001)


def test_cluster_bootstrap_matches_manual_and_reproduces():
    values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 5.0])
    ids = np.array(["a", "a", "a", "b", "b", "c"])
    got = inf.cluster_bootstrap_mean(values, ids, 0.9, 500, 3)
    assert got == inf.cluster_bootstrap_mean(values, ids, 0.9, 500, 3)
    mult = bootstrap.cluster_multiplicities(3, 500, np.random.default_rng(3))
    sums, counts = np.array([6.0, 30.0, 5.0]), np.array([3.0, 2.0, 1.0])
    reps = (mult @ sums) / (mult @ counts)
    assert got == pytest.approx(bootstrap.percentile_interval(reps, 0.9)[:2])


def test_cluster_bootstrap_constant_values_degenerate():
    low, high = inf.cluster_bootstrap_mean([2.0] * 6, [0, 0, 1, 1, 2, 3], 0.95, 200, 1)
    assert low == high == 2.0


def test_calendar_block_bootstrap_matches_manual_and_pairs_days():
    dates = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-05", "2024-01-20"])
    values = np.array([1.0, 3.0, 5.0, 9.0])
    got = inf.calendar_block_bootstrap_mean(values, dates, 7, 0.9, 400, 5)
    assert got == inf.calendar_block_bootstrap_mean(values, dates, 7, 0.9, 400, 5)
    mult = bootstrap.calendar_block_multiplicities(20, 7, 400, np.random.default_rng(5))
    w = mult[:, [0, 0, 4, 19]].astype(float)
    with np.errstate(invalid="ignore"):
        reps = (w @ values) / w.sum(axis=1)
    assert got == pytest.approx(bootstrap.percentile_interval(reps, 0.9)[:2])


def test_calendar_block_constant_values_degenerate():
    dates = pd.date_range("2024-01-01", periods=30)
    low, high = inf.calendar_block_bootstrap_mean(np.full(30, 4.0), dates, 7, 0.95, 200, 1)
    assert low == high == 4.0


@pytest.mark.parametrize(
    ("a", "b", "lower", "want"),
    [
        ((-2, -1), (-3, -0.5), True, "beats"),
        ((-2, -1), (-3, 0.5), True, "inconclusive"),
        ((-2, 0.0), (-3, -1), True, "inconclusive"),
        ((1, 2), (0.5, 3), True, "loses"),
        ((1, 2), (-0.5, 3), True, "inconclusive"),
        ((1, 2), (0.5, 3), False, "beats"),
        ((-2, -1), (-3, -0.5), False, "loses"),
        ((-1, 1), (-1, 1), False, "inconclusive"),
        ((-2, -1), (1, 2), True, "inconclusive"),
    ],
)
def test_verdict_truth_table(a, b, lower, want):
    assert inf.verdict_two_procedures(a, b, lower) == want


def test_verdict_rejects_bad_intervals():
    with pytest.raises(ValueError):
        inf.verdict_two_procedures((np.nan, 1.0), (0.0, 1.0), True)


def _noisy(n, mean, seed=0):
    return np.random.default_rng(seed).normal(mean, 0.1, size=n)


def test_s1_descriptive_below_ten_events():
    assert inf.s1_verdict(np.full(9, -1.0) + _noisy(9, 0)) == "descriptive only"
    assert inf.s1_verdict([]) == "descriptive only"


def test_s1_verdict_branches():
    assert inf.s1_verdict(_noisy(12, -1.0)) == "beats naive"
    assert inf.s1_verdict(_noisy(12, 1.0)) == "loses to naive"
    assert inf.s1_verdict(_noisy(12, 0.0, seed=3)) == "inconclusive"
    # Same direction but the sign-flip p (2/2^10) passes; a single positive
    # outlier destroys the t interval and the verdict.
    d = np.array([-1.0] * 9 + [8.0])
    assert inf.s1_verdict(d) == "inconclusive"
