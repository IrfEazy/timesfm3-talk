"""Opt-in integration tests against real network services: yfinance and
FRED. Skipped by default. Run with:
    TFM3LAB_RUN_LIVE_FETCH_SMOKE=1 uv run pytest tests/test_data_live.py -v
"""

from __future__ import annotations

import os

import pytest

from tfm3lab.data.macro import build_cpi_series
from tfm3lab.data.market import DEFAULT_TICKERS, build_market_series

pytestmark = pytest.mark.skipif(
    os.environ.get("TFM3LAB_RUN_LIVE_FETCH_SMOKE") != "1",
    reason="opt-in only: hits yfinance/FRED; set TFM3LAB_RUN_LIVE_FETCH_SMOKE=1",
)


def test_build_market_series_returns_aligned_positive_series():
    series = build_market_series(start="2024-01-01", end="2024-02-01")
    assert len(series) == len(DEFAULT_TICKERS)
    lengths = {s.name: len(s.values) for s in series}
    assert len(set(lengths.values())) == 1  # all tickers share the same (intersected) calendar
    for s in series:
        assert (s.values > 0).all()
        assert s.observed.all()


def test_build_cpi_series_returns_a_long_monthly_history():
    series = build_cpi_series()
    assert len(series.values) > 100  # decades of monthly data
    assert series.observed.all()
