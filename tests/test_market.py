import numpy as np
import pytest

from tfm3lab.data.market import detect_shock_days


def test_detect_shock_days_flags_an_obvious_spike():
    rng = np.random.default_rng(0)
    n = 200
    log_prices = np.cumsum(rng.normal(scale=0.005, size=n))
    log_prices[100:] += 0.30  # a single-day ~30% jump at index 100
    values = np.exp(log_prices) * 100.0

    detected = detect_shock_days(values, z_threshold=4.0)
    assert 100 in detected


def test_detect_shock_days_empty_on_constant_series():
    # Zero volatility -> std of log-returns is 0 -> defined empty result,
    # not a division-by-zero NaN explosion.
    values = np.full(50, 100.0)
    assert detect_shock_days(values).size == 0


def test_detect_shock_days_rejects_nonpositive_threshold():
    with pytest.raises(ValueError):
        detect_shock_days(np.array([100.0, 101.0, 102.0]), z_threshold=0.0)


def test_detect_shock_days_quiet_series_flags_nothing_at_high_threshold():
    rng = np.random.default_rng(1)
    values = 100.0 * np.exp(np.cumsum(rng.normal(scale=0.005, size=300)))
    # No engineered spike; a very high threshold should find nothing.
    assert detect_shock_days(values, z_threshold=6.0).size == 0
