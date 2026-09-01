"""Rolling-origin window semantics — the single source of truth.

The Colab draft this project replaces had two different conventions for
"which index does horizon step h land on": `build_*_records` used
`origin + h` (h from 0), while `compare_before_after_event` used
`np.arange(origin + 1, origin + horizon + 1)`. That one-line disagreement
silently shifted the pre/post-shock plot by a day relative to the metrics
table. Every module in this project must go through the functions below to
turn an origin into context/target indices — never re-derive them locally.

Convention: `origin` is the index one past the last observed context point.
    context = series[origin - context_len : origin]
    target  = series[origin : origin + horizon]
So `origin` itself is the first *predicted* index, not the last observed one.
"""

from __future__ import annotations

import numpy as np


def target_indices(origin: int, horizon: int) -> np.ndarray:
    """Indices predicted from `origin`: origin, origin+1, ..., origin+horizon-1."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return np.arange(origin, origin + horizon)


def context_slice(origin: int, context_len: int) -> slice:
    """Slice of the context fed to the model: series[origin-context_len : origin].

    Clipped at 0 rather than raising, so early origins (origin < context_len)
    get a shorter-than-requested context instead of a negative start index.
    The model left-pads short contexts internally (masked, so this is safe) —
    see TimesFM3Forecaster._Query.format in the installed timesfm3 package.
    """
    if context_len < 1:
        raise ValueError(f"context_len must be >= 1, got {context_len}")
    return slice(max(0, origin - context_len), origin)


def valid_origins(
    n: int,
    context_len: int,
    horizon: int,
    max_origins: int | None = None,
) -> np.ndarray:
    """All origins with a full context window and a full target window inside [0, n).

    An origin qualifies when both its context (origin-context_len .. origin)
    and its target (origin .. origin+horizon) fit inside the series. If
    `max_origins` truncates the result, only the *last* `max_origins` are
    kept — callers must log how many were dropped (see docs/talk-outline.md's
    "no silent caps" rule); this function does not log for you.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    first = context_len
    last = n - horizon  # inclusive: target_indices(last, horizon)[-1] == n - 1
    if last < first:
        return np.array([], dtype=int)
    origins = np.arange(first, last + 1)
    if max_origins is not None and len(origins) > max_origins:
        origins = origins[-max_origins:]
    return origins


def assert_no_leakage(origin: int, context_len: int, horizon: int) -> None:
    """Defensive check: context indices must all precede target indices.

    Cheap enough to call in tests and in the backtest engine's debug mode;
    not called in the hot loop by default.
    """
    ctx = context_slice(origin, context_len)
    ctx_indices = np.arange(ctx.start, ctx.stop)
    tgt = target_indices(origin, horizon)
    if len(ctx_indices) and ctx_indices.max() >= tgt.min():
        raise AssertionError(
            f"leakage at origin={origin}: context reaches index "
            f"{ctx_indices.max()} but target starts at {tgt.min()}"
        )


def nearest_origin(dates: np.ndarray, target_date) -> int:
    """Index of the date in a sorted `dates` array closest to `target_date`.

    Replaces the draft's `dates.view("int64")`, removed in pandas >= 2.
    `dates` must be a sorted array of datetime64 (e.g. a DatetimeIndex's
    `.values`, or `.to_numpy()`).
    """
    dates64 = np.asarray(dates, dtype="datetime64[ns]")
    target64 = np.datetime64(target_date, "ns")
    pos = int(np.searchsorted(dates64, target64))
    if pos == 0:
        return 0
    if pos == len(dates64):
        return len(dates64) - 1
    before, after = dates64[pos - 1], dates64[pos]
    return pos - 1 if (target64 - before) <= (after - target64) else pos
