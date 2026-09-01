import numpy as np
import pytest
from scipy import stats

from tfm3lab.metrics import (
    block_bootstrap_ci,
    coverage,
    diebold_mariano,
    in_sample_scale,
    mae,
    mase,
    pinball_loss,
    pinball_loss_multi,
    pit_values,
    relative_mae,
    rmse,
    smape,
)


def test_mae_hand_computed():
    assert mae([1, 2, 3], [1, 1, 1]) == pytest.approx(1.0)


def test_rmse_hand_computed():
    assert rmse([1, 2, 3], [1, 1, 1]) == pytest.approx(np.sqrt(5 / 3))


def test_smape_maximally_wrong_when_one_side_zero():
    assert smape([10.0], [0.0]) == pytest.approx(200.0)


def test_smape_zero_when_exact():
    assert smape([10.0], [10.0]) == pytest.approx(0.0)


def test_in_sample_scale_hand_computed():
    # |2-1|,|3-2|,|4-3|,|5-4| = 1,1,1,1 -> mean 1.0
    assert in_sample_scale([1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_in_sample_scale_rejects_too_short_history():
    with pytest.raises(ValueError):
        in_sample_scale([1, 2], seasonality=3)


def test_mase_hand_computed():
    assert mase([2, 4], 2.0) == pytest.approx(1.5)


def test_relative_mae_hand_computed():
    assert relative_mae(2.0, 4.0) == pytest.approx(0.5)


def test_relative_mae_rejects_nonpositive_baseline():
    with pytest.raises(ValueError):
        relative_mae(2.0, 0.0)


def test_pinball_loss_at_median_equals_half_abs_error():
    # q=0.5 pinball loss is exactly half the absolute error.
    assert pinball_loss([10.0], [8.0], quantile=0.5) == pytest.approx(1.0)


def test_pinball_loss_asymmetric_penalty_above_and_below():
    # y=10, yhat=12 (over-forecast), q=0.9 -> (q-1)*(y-yhat) = -0.1*-2 = 0.2
    assert pinball_loss([10.0], [12.0], quantile=0.9) == pytest.approx(0.2)
    # y=12, yhat=10 (under-forecast by same amount), q=0.9 -> q*(y-yhat) = 0.9*2 = 1.8
    assert pinball_loss([12.0], [10.0], quantile=0.9) == pytest.approx(1.8)


def test_pinball_loss_multi_averages_across_levels():
    levels = [0.1, 0.5, 0.9]
    # exact forecast at every level -> zero loss regardless of level
    actual = [5.0]
    q_forecasts = np.array([[5.0, 5.0, 5.0]])
    assert pinball_loss_multi(actual, q_forecasts, levels) == pytest.approx(0.0)


def test_coverage_hand_computed():
    assert coverage([5, 15], lower=10, upper=20) == pytest.approx(0.5)


def test_coverage_all_inside():
    assert coverage([11, 12, 13], lower=10, upper=20) == pytest.approx(1.0)


def test_pit_values_exact_grid_point():
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    q_forecasts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    # actual falls exactly on the median quantile's value -> PIT == 0.5
    out = pit_values([5.0], q_forecasts, levels)
    assert out[0] == pytest.approx(0.5)


def test_pit_values_clips_outside_range():
    levels = [0.1, 0.9]
    q_forecasts = np.array([[1.0, 9.0]])
    below = pit_values([-100.0], q_forecasts, levels)
    above = pit_values([100.0], q_forecasts, levels)
    assert below[0] == pytest.approx(0.1)
    assert above[0] == pytest.approx(0.9)


# --- Diebold-Mariano ---------------------------------------------------------


def test_dm_zero_when_loss_differential_is_symmetric_and_mean_zero():
    # d = [1, -1, 1, -1], mean 0 -> dm_stat and dm_adj are both exactly 0,
    # and the t-distribution is symmetric around 0 -> p-value exactly 1.
    loss1 = np.array([3.0, 1.0, 3.0, 1.0])
    loss2 = np.array([2.0, 2.0, 2.0, 2.0])
    stat, p = diebold_mariano(loss1, loss2, horizon=1)
    assert stat == pytest.approx(0.0, abs=1e-10)
    assert p == pytest.approx(1.0, abs=1e-10)


def test_dm_degenerate_zero_variance_returns_sentinel():
    # Constant loss differential -> zero variance -> defined sentinel (0.0, 1.0),
    # not a division-by-zero crash.
    loss1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    loss2 = np.array([6.0, 6.0, 6.0, 6.0, 6.0])
    stat, p = diebold_mariano(loss1, loss2, horizon=1)
    assert (stat, p) == (0.0, 1.0)


def test_dm_rejects_too_few_observations_for_horizon():
    with pytest.raises(ValueError):
        diebold_mariano([1.0, 2.0], [1.0, 1.0], horizon=2)


def _reference_dm(d: np.ndarray, horizon: int) -> tuple[float, float]:
    """Independent from-the-paper re-implementation, used only in tests."""
    t = len(d)
    dbar = d.mean()
    gamma0 = np.sum((d - dbar) ** 2) / t
    lrv = gamma0
    for lag in range(1, horizon):
        lrv += 2 * np.sum((d[lag:] - dbar) * (d[:-lag] - dbar)) / t
    var_dbar = lrv / t
    if var_dbar <= 0:
        return 0.0, 1.0
    dm = dbar / np.sqrt(var_dbar)
    correction = np.sqrt((t + 1 - 2 * horizon + horizon * (horizon - 1) / t) / t)
    dm_adj = dm * correction
    p = 2 * (1 - stats.t.cdf(np.abs(dm_adj), df=t - 1))
    return float(dm_adj), float(p)


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_dm_matches_independent_reference_implementation(horizon):
    rng = np.random.default_rng(7)
    # AR(1)-ish positively autocorrelated series so horizon>1 lag terms are
    # well-behaved (a purely alternating series drives the small-sample
    # variance estimate negative — a known finite-sample quirk of this
    # estimator, not something this test needs to exercise).
    noise = rng.normal(size=40)
    d = np.cumsum(noise) * 0.1 + rng.normal(scale=0.05, size=40)
    loss1 = d + 5.0
    loss2 = np.full(40, 5.0)
    got_stat, got_p = diebold_mariano(loss1, loss2, horizon=horizon)
    want_stat, want_p = _reference_dm(loss1 - loss2, horizon=horizon)
    assert got_stat == pytest.approx(want_stat, rel=1e-9)
    assert got_p == pytest.approx(want_p, rel=1e-9)


# --- block bootstrap ----------------------------------------------------------


def test_block_bootstrap_ci_contains_true_mean_low_variance():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=10.0, scale=0.1, size=200)
    lower, upper = block_bootstrap_ci(values, block_size=5, n_boot=500, ci=0.9, rng=rng)
    assert lower < 10.0 < upper


def test_block_bootstrap_ci_is_deterministic_given_seeded_rng():
    values = np.arange(50, dtype=float)
    rng_kwargs = {"block_size": 4, "n_boot": 200}
    lower1, upper1 = block_bootstrap_ci(values, rng=np.random.default_rng(123), **rng_kwargs)
    lower2, upper2 = block_bootstrap_ci(values, rng=np.random.default_rng(123), **rng_kwargs)
    assert (lower1, upper1) == (lower2, upper2)


def test_block_bootstrap_ci_rejects_bad_block_size():
    with pytest.raises(ValueError):
        block_bootstrap_ci([1.0, 2.0, 3.0], block_size=0)
    with pytest.raises(ValueError):
        block_bootstrap_ci([1.0, 2.0, 3.0], block_size=10)
