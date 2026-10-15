"""Frozen quote / gap / validity policy of the v3 studies.

Everything here works on a *wide* price matrix: one row per SKU
(product_id, subtype), one column per calendar day, contiguous from the first
to the last day of the panel.

Policy (frozen, do not tune):

- A price is VALID iff it is finite and > 0. Any other value (0, negative,
  NaN, inf) is treated as missing.
- The OBSERVED mask of a day is "a valid price exists on that exact day".
- The QUOTE q(d) of a day is the valid price on d if there is one, else the
  latest valid price in [d - MAX_FFILL, d - 1], else missing. It is a forward
  fill limited to MAX_FFILL = 3 days of raw observations: a quote is never
  carried across more than 3 consecutive missing days, never chains from
  another quote, and never looks backwards from the future.
- The CONTEXT of an origin is the trailing run of non-missing quotes ending at
  the origin column inclusive (it stops at the most recent missing quote),
  capped at the last MAX_CONTEXT values. It is empty when the origin quote is
  missing. The SCALED context is the context divided by the origin quote, so
  its last value is exactly 1.0.

Leakage rule: every origin-dependent function takes an explicit origin column
and slices the matrix to `[: origin + 1]` before computing anything, so no
value after the origin can influence a result.

No outcome (future price, return, label) is computed in this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_FFILL = 3
MAX_CONTEXT = 512

_LONG_COLUMNS = ("date", "product_id", "subtype", "price")


def _is_valid(values: np.ndarray) -> np.ndarray:
    """Validity rule of a price: finite and strictly positive."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(values) & (values > 0)


def _as_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (sku x day), got {arr.ndim}-D")
    return arr


def _sku(product_id: object, subtype: object) -> tuple[int, str | None]:
    """A SKU key; any missing subtype (None, NaN, NA) is None."""
    return int(product_id), subtype if isinstance(subtype, str) else None


def _check_index(value: int, size: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if not 0 <= value < size:
        raise ValueError(f"{name}={value} out of range [0, {size - 1}]")
    return int(value)


# --- long -> wide ---------------------------------------------------------------


def wide_prices(long_df: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.Index, np.ndarray]:
    """Pivots a long panel (columns date, product_id, subtype, price) into
    (dates, skus, matrix).

    - `dates`: daily DatetimeIndex, contiguous from the earliest to the latest
      date present in `long_df` (days without any row are columns of NaN).
    - `skus`: Index of (product_id, subtype) tuples, sorted by product_id then
      subtype (None last); a missing subtype is None. Row i of `matrix` is `skus[i]`.
    - `matrix`: float64, shape (n_sku, n_days); NaN where there is no row or
      the price is not valid (see `_is_valid`).

    Two rows for the same (date, product_id, subtype) raise ValueError.
    """
    absent = [c for c in _LONG_COLUMNS if c not in long_df.columns]
    if absent:
        raise ValueError(f"long frame is missing columns {absent}")
    if long_df.empty:
        return (
            pd.DatetimeIndex([], name="date"),
            pd.Index([], dtype=object, tupleize_cols=False, name="sku"),
            np.empty((0, 0), dtype=np.float64),
        )

    day = pd.to_datetime(long_df["date"])
    if day.dt.tz is not None:
        raise ValueError("date must be timezone-naive")
    if day.isna().any():
        raise ValueError("date contains missing values")
    day = day.dt.normalize()
    first_day = day.min()
    dates = pd.date_range(first_day, day.max(), freq="D", name="date")
    col = ((day - first_day) // pd.Timedelta(days=1)).to_numpy(dtype=np.int64)

    keys = pd.DataFrame(
        {
            "product_id": long_df["product_id"].to_numpy(dtype=np.int64),
            "subtype": long_df["subtype"].to_numpy(),
        }
    )
    row = keys.groupby(["product_id", "subtype"], sort=True, dropna=False).ngroup().to_numpy()
    n_sku = int(row.max()) + 1
    _, first_pos = np.unique(row, return_index=True)
    skus = pd.Index(
        [_sku(keys["product_id"].iat[i], keys["subtype"].iat[i]) for i in first_pos],
        dtype=object,
        tupleize_cols=False,
        name="sku",
    )

    flat = row.astype(np.int64) * len(dates) + col
    duplicated = pd.Series(flat).duplicated(keep=False).to_numpy()
    if duplicated.any():
        i = int(np.flatnonzero(duplicated)[0])
        raise ValueError(
            f"{int(duplicated.sum())} rows share a (date, product_id, subtype) key; "
            f"first: product_id={keys['product_id'].iat[i]}, subtype={skus[row[i]][1]!r}, "
            f"date={dates[col[i]].date().isoformat()}"
        )

    price = long_df["price"].to_numpy(dtype=np.float64)
    ok = _is_valid(price)
    matrix = np.full((n_sku, len(dates)), np.nan, dtype=np.float64)
    matrix[row[ok], col[ok]] = price[ok]
    return dates, skus, matrix


# --- quotes and observation mask -------------------------------------------------


def observed_mask(wide: np.ndarray) -> np.ndarray:
    """Boolean matrix: True where the exact day has a valid price."""
    return _is_valid(_as_matrix(wide, "wide"))


def quote_matrix(
    wide: np.ndarray, max_ffill: int = MAX_FFILL, chunk_rows: int = 16384
) -> np.ndarray:
    """The quote matrix q: the valid price of the day, else the latest valid
    price in [d - max_ffill, d - 1], else NaN. Column d depends only on
    columns <= d of `wide`. Rows are processed in chunks of `chunk_rows` with
    int32 day-index temporaries to bound peak memory."""
    if isinstance(max_ffill, bool) or not isinstance(max_ffill, int | np.integer) or max_ffill < 0:
        raise ValueError(f"max_ffill must be a non-negative integer, got {max_ffill!r}")
    if (
        isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int | np.integer)
        or chunk_rows < 1
    ):
        raise ValueError(f"chunk_rows must be a positive integer, got {chunk_rows!r}")
    wide = _as_matrix(wide, "wide")
    n_days = wide.shape[1]
    day = np.arange(n_days, dtype=np.int32)[None, :]
    out = np.empty(wide.shape, dtype=np.float64)
    for lo in range(0, wide.shape[0], chunk_rows):
        part = wide[lo : lo + chunk_rows]
        last_valid = np.maximum.accumulate(np.where(_is_valid(part), day, np.int32(-1)), axis=1)
        usable = (last_valid >= 0) & (day - last_valid <= max_ffill)
        carried = np.take_along_axis(part, np.where(usable, last_valid, 0), axis=1)
        out[lo : lo + chunk_rows] = np.where(usable, carried, np.nan)
    return out


# --- origin-time views (all slice to origin + 1 before computing) ----------------


def context(
    qmat: np.ndarray, sku_idx: int, origin_idx: int, max_len: int = MAX_CONTEXT
) -> np.ndarray:
    """The trailing run of valid quotes of SKU row `sku_idx` ending at column
    `origin_idx` inclusive, at most `max_len` values, oldest first. The run
    stops at the most recent missing quote; the result is empty when the quote
    at the origin is missing. Only columns <= origin_idx are read."""
    qmat = _as_matrix(qmat, "qmat")
    sku_idx = _check_index(sku_idx, qmat.shape[0], "sku_idx")
    origin_idx = _check_index(origin_idx, qmat.shape[1], "origin_idx")
    if isinstance(max_len, bool) or not isinstance(max_len, int | np.integer) or max_len < 1:
        raise ValueError(f"max_len must be a positive integer, got {max_len!r}")
    row = qmat[sku_idx, max(0, origin_idx + 1 - max_len) : origin_idx + 1]
    gaps = np.flatnonzero(~_is_valid(row))
    start = int(gaps[-1]) + 1 if gaps.size else 0
    return np.array(row[start:], dtype=np.float64)


def scaled_context(
    qmat: np.ndarray, sku_idx: int, origin_idx: int, max_len: int = MAX_CONTEXT
) -> np.ndarray:
    """`context` divided by the origin quote, so its last value is exactly 1.0;
    empty when the origin quote is missing."""
    ctx = context(qmat, sku_idx, origin_idx, max_len)
    return ctx / ctx[-1] if ctx.size else ctx


def first_observed_idx(wide: np.ndarray, sku_idx: int, origin_idx: int) -> int | None:
    """Column of the first valid price of SKU row `sku_idx` among columns
    <= origin_idx, or None if it has not been observed by the origin."""
    wide = _as_matrix(wide, "wide")
    sku_idx = _check_index(sku_idx, wide.shape[0], "sku_idx")
    origin_idx = _check_index(origin_idx, wide.shape[1], "origin_idx")
    hits = np.flatnonzero(_is_valid(wide[sku_idx, : origin_idx + 1]))
    return int(hits[0]) if hits.size else None


def first_observed_idx_all(wide: np.ndarray, origin_idx: int) -> np.ndarray:
    """`first_observed_idx` for every SKU as an int64 vector; -1 where the SKU
    has no valid price among columns <= origin_idx."""
    wide = _as_matrix(wide, "wide")
    origin_idx = _check_index(origin_idx, wide.shape[1], "origin_idx")
    seen = _is_valid(wide[:, : origin_idx + 1])
    return np.where(seen.any(axis=1), seen.argmax(axis=1), -1).astype(np.int64)


def n_observed_in_window(
    wide: np.ndarray,
    sku_idx: int,
    lo_idx: int,
    hi_idx: int,
    *,
    origin_idx: int | None = None,
) -> int:
    """Number of days in the inclusive window [lo_idx, hi_idx] on which SKU row
    `sku_idx` has a valid price. `lo_idx` below 0 is clipped to 0 (no data
    before the first column). `hi_idx` is the last column read; with
    `origin_idx` given, a window ending after the origin raises ValueError."""
    wide = _as_matrix(wide, "wide")
    sku_idx = _check_index(sku_idx, wide.shape[0], "sku_idx")
    hi_idx = _check_index(hi_idx, wide.shape[1], "hi_idx")
    if origin_idx is not None:
        origin_idx = _check_index(origin_idx, wide.shape[1], "origin_idx")
        if hi_idx > origin_idx:
            raise ValueError(f"window end {hi_idx} is after the origin {origin_idx}")
    lo_idx = max(0, int(lo_idx))
    if hi_idx < lo_idx:
        return 0
    return int(_is_valid(wide[sku_idx, lo_idx : hi_idx + 1]).sum())
