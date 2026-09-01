"""US CPI (FRED CPIAUCSL) — inflation YoY, the optional macro arm of the
shock experiment.

`CPIAUCSL` as downloaded today is FRED's currently revised, seasonally
adjusted series — NOT the vintage a real forecaster would have seen at the
time (plan finding #12: a "vintage" problem). Any claim about the model
reacting to the 2022 inflation shock using this series is therefore softer
evidence than the market-shock experiment on daily prices, and the talk
must disclose that, not silently treat CPI the same as SP500/VIX.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest import SeriesData

FRED_CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"


def compute_yoy(raw: pd.DataFrame, value_col: str = "CPIAUCSL") -> pd.Series:
    """Pure transform: FRED's raw (date, level) CSV -> year-over-year % change.

    Separated from the network fetch so this logic is testable with a
    small hand-built DataFrame, no HTTP involved.
    """
    date_col = "observation_date" if "observation_date" in raw.columns else "DATE"
    cpi = raw[[date_col, value_col]].copy()
    cpi[date_col] = pd.to_datetime(cpi[date_col])
    cpi[value_col] = pd.to_numeric(cpi[value_col], errors="coerce")
    cpi = cpi.dropna().set_index(date_col)[value_col].sort_index()
    return (cpi.pct_change(12) * 100).dropna()


def fetch_cpi_yoy(url: str = FRED_CPI_URL) -> pd.Series:
    return compute_yoy(pd.read_csv(url))


def build_cpi_series(url: str = FRED_CPI_URL) -> SeriesData:
    yoy = fetch_cpi_yoy(url)
    return SeriesData(
        name="CPI_YoY",
        values=yoy.to_numpy(dtype=float),
        dates=yoy.index.to_numpy(),
        observed=np.ones(len(yoy), dtype=bool),
    )
