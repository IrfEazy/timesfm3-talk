import datetime as dt

import pandas as pd
import pytest

from tfm3lab.data import macro
from tfm3lab.data.macro import compute_yoy


def test_compute_yoy_hand_computed():
    # 13 monthly levels, 100.0 .. 112.0: YoY at month 13 = (112/100 - 1)*100 = 12.0
    dates = pd.date_range("2020-01-01", periods=13, freq="MS")
    levels = [100.0 + i for i in range(13)]
    raw = pd.DataFrame({"observation_date": dates, "CPIAUCSL": levels})

    yoy = compute_yoy(raw)

    assert len(yoy) == 1  # only one point has a full 12-month lookback
    assert yoy.iloc[0] == pytest.approx(12.0)
    assert yoy.index[0] == dates[-1]


def test_compute_yoy_handles_alternate_date_column_name():
    dates = pd.date_range("2020-01-01", periods=13, freq="MS")
    levels = [200.0] * 13  # flat -> 0% YoY
    raw = pd.DataFrame({"DATE": dates, "CPIAUCSL": levels})
    yoy = compute_yoy(raw)
    assert yoy.iloc[0] == pytest.approx(0.0)


def test_compute_yoy_drops_unparseable_rows():
    dates = pd.date_range("2020-01-01", periods=14, freq="MS")
    levels = [100.0 + i for i in range(13)] + [None]  # a trailing "." from FRED, coerced to NaN
    raw = pd.DataFrame({"observation_date": dates, "CPIAUCSL": levels})
    yoy = compute_yoy(raw)
    assert len(yoy) == 1  # the unparseable last row must not produce a bogus YoY point


def test_build_cpi_series_trims_to_end_date(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=15, freq="MS")
    yoy = pd.Series([float(i) for i in range(15)], index=dates)
    monkeypatch.setattr(macro, "fetch_cpi_yoy", lambda url=macro.FRED_CPI_URL: yoy)

    series = macro.build_cpi_series(end=dt.date(2020, 10, 1))

    assert pd.Timestamp(series.dates[-1]) <= pd.Timestamp("2020-10-01")
    assert len(series.values) == 10


def test_build_cpi_series_without_end_keeps_full_series(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=5, freq="MS")
    yoy = pd.Series([1.0] * 5, index=dates)
    monkeypatch.setattr(macro, "fetch_cpi_yoy", lambda url=macro.FRED_CPI_URL: yoy)

    series = macro.build_cpi_series()

    assert len(series.values) == 5
