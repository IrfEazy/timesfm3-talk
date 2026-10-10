import numpy as np

from tfm3lab.backtest import SeriesData
from tfm3lab.normalize import series_error_scale


def _s(values, observed=None):
    v = np.asarray(values, dtype=float)
    obs = np.ones(len(v), dtype=bool) if observed is None else np.asarray(observed)
    return SeriesData("x", v, np.arange(len(v)).astype("datetime64[D]"), obs)


def test_median_of_observed_values():
    assert series_error_scale(_s([1, 2, 100, 7], [1, 1, 1, 0])) == (2.0, True)


def test_invalid_scales_flagged():
    for vals in ([0, 0, 0], [-1, -2]):
        scale, ok = series_error_scale(_s(vals))
        assert not ok and np.isnan(scale)
    s = _s([1, 2])
    s.values[:] = np.nan  # SeriesData rejects NaN at construction; mutate after
    assert not series_error_scale(s)[1]
    assert not series_error_scale(_s([1.0], [False]))[1]
