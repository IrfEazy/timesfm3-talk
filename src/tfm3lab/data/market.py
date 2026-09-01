"""Market shock series (S&P 500, VIX, gold, oil) — Experiment B.

Unlike the MTG card prices in experiment A, every one of these tickers is
almost certainly present in TimesFM-3's pretraining corpus in some form
(the model card cites Wikipedia Pageviews and Google Trends among its
pretraining sources, alongside GIFT-Eval's own undated pretraining split).
That is exactly why Experiment B compares pre-cutoff and post-cutoff shock
events on the SAME tickers, instead of treating any single result here as
proof of zero-shot generalization — see config.PRETRAIN_CUTOFF and
docs/talk-outline.md.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

from ..backtest import SeriesData

DEFAULT_TICKERS = {
    "SP500": "^GSPC",
    "VIX": "^VIX",
    "Gold": "GC=F",
    "Oil": "CL=F",
}


def fetch_close(ticker: str, start: str, end: str | None = None) -> pd.Series:
    """Downloads one ticker's daily auto-adjusted close.

    Fetched one ticker at a time (not a multi-ticker yf.download call) to
    sidestep yfinance's MultiIndex columns for multi-symbol requests —
    simpler and less error-prone than unpacking that index correctly.
    """
    end = end or (dt.date.today() + dt.timedelta(days=1)).isoformat()
    data = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False
    )
    if data.empty:
        raise RuntimeError(f"yfinance returned no data for '{ticker}'")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):  # defensive: seen even for single-ticker requests
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    return close.rename(ticker)


def build_market_series(
    tickers: dict[str, str] = DEFAULT_TICKERS,
    start: str = "2018-01-01",
    end: str | None = None,
) -> list[SeriesData]:
    """One SeriesData per ticker, restricted to the intersection of trading
    days where every ticker has a valid positive close.

    Deliberately NOT forward-filled: for highly liquid daily instruments, a
    missing value almost always means "this market was closed," and
    filling it would manufacture a fake "no movement" observation on a day
    nothing was actually observed — the exact ffill-inflation problem
    (plan findings #9/#10) this project avoids for MTG prices. `observed`
    is therefore always True here, by construction of the intersection.
    """
    raw = {label: fetch_close(ticker, start, end) for label, ticker in tickers.items()}
    combined = pd.concat(raw, axis=1)
    combined.columns = list(tickers.keys())
    combined = combined.sort_index().dropna()
    combined = combined[(combined > 0).all(axis=1)]

    dates = combined.index.to_numpy()
    return [
        SeriesData(
            name=label,
            values=combined[label].to_numpy(dtype=float),
            dates=dates,
            observed=np.ones(len(combined), dtype=bool),
        )
        for label in tickers
    ]


def detect_shock_days(values: np.ndarray, z_threshold: float = 4.0) -> np.ndarray:
    """Indices of days whose log-return z-score exceeds `z_threshold` in
    absolute value.

    A reproducible, data-driven way to anchor "shock" events instead of
    trusting a hand-picked news date — config.KNOWN_EVENTS are used only to
    validate that this detector fires near the expected dates, not as a
    substitute for it.
    """
    if z_threshold <= 0:
        raise ValueError(f"z_threshold must be positive, got {z_threshold}")
    values = np.asarray(values, dtype=float)
    log_returns = np.diff(np.log(values))
    mean, std = np.nanmean(log_returns), np.nanstd(log_returns)
    if std <= 0:
        return np.array([], dtype=int)
    z = (log_returns - mean) / std
    return np.where(np.abs(z) >= z_threshold)[0] + 1  # +1: np.diff shifts the index by one
