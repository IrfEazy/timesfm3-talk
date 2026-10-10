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
from pathlib import Path

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

# Experiment B v2 (Preregistration v2 s3): the five detector indices. Each is
# fetched on ITS OWN trading calendar (never intersected) so a holiday in one
# market cannot hide or invent a shock in another.
INDEX_TICKERS = {
    "SP500": "^GSPC",
    "Nasdaq100": "^NDX",
    "EuroStoxx50": "^STOXX50E",
    "Nikkei225": "^N225",
    "Gold": "GC=F",
}
# Every ticker the fetch path knows (existing VIX/Oil callers are unchanged).
ALL_TICKERS = {**DEFAULT_TICKERS, **INDEX_TICKERS}

TRAILING_WINDOW = 250
SHOCK_Z_THRESHOLD = 4.0


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


def trailing_shock_z(values: np.ndarray, window: int = TRAILING_WINDOW) -> np.ndarray:
    """z-score of each day's log return against the mean and std (ddof=1) of
    the `window` log returns STRICTLY BEFORE that day. Returns an array the
    length of `values`; entry t is NaN until `window` past returns exist (and
    where the trailing std is 0). Entry t depends only on values[: t + 1], so
    changing future values can never change an earlier z (no look-ahead)."""
    values = np.asarray(values, dtype=float)
    ret = pd.Series(np.concatenate([[np.nan], np.diff(np.log(values))]))
    past = ret.shift(1)
    mean = past.rolling(window, min_periods=window).mean()
    std = past.rolling(window, min_periods=window).std(ddof=1)
    z = (ret - mean) / std.where(std > 0)
    return z.to_numpy(dtype=float)


def detect_trailing_shock_days(
    values: np.ndarray, z_threshold: float = SHOCK_Z_THRESHOLD, window: int = TRAILING_WINDOW
) -> tuple[np.ndarray, np.ndarray]:
    """(positions, z) of days with |z| >= `z_threshold` under `trailing_shock_z`.
    Positions index `values` on that index's own trading calendar."""
    if z_threshold <= 0:
        raise ValueError(f"z_threshold must be positive, got {z_threshold}")
    z = trailing_shock_z(values, window)
    pos = np.flatnonzero(np.abs(z) >= z_threshold)
    return pos, z[pos]


def fetch_index_closes(
    tickers: dict[str, str] = INDEX_TICKERS, start: str = "2018-01-01", end: str | None = None
) -> dict[str, pd.Series]:
    """label -> daily close on that index's own calendar (no intersection)."""
    return {label: fetch_close(t, start, end).dropna() for label, t in tickers.items()}


def index_cache_path(cache_dir, label: str, as_of) -> Path:
    return Path(cache_dir) / f"shock_v2_{label}_{as_of}.parquet"


def save_index_closes(closes: dict[str, pd.Series], cache_dir, as_of) -> list:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for label, s in closes.items():
        path = index_cache_path(cache_dir, label, as_of)
        pd.DataFrame(
            {"date": pd.to_datetime(s.index), "close": s.to_numpy(dtype=float)}
        ).to_parquet(path, index=False)
        paths.append(path)
    return paths


def load_index_series(cache_dir, as_of, labels=tuple(INDEX_TICKERS)) -> dict[str, SeriesData]:
    """Reads what `save_index_closes` wrote; each index keeps its own calendar."""
    out = {}
    for label in labels:
        path = index_cache_path(cache_dir, label, as_of)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run scripts/03a_select_events.py first")
        df = pd.read_parquet(path).sort_values("date")
        df = df[df["close"] > 0]
        out[label] = SeriesData(
            name=label,
            values=df["close"].to_numpy(dtype=float),
            dates=df["date"].to_numpy(),
            observed=np.ones(len(df), dtype=bool),
        )
    return out
