"""Rolling-origin backtest engine: wires windows.py + model.py +
baselines.py into one tidy long-format results table.

Two entry points mirror the project's central ablation (plan finding —
"multivariato vs univariato testato solo su MTG, va fatto anche sui
mercati"): `run_univariate_backtest` forecasts each series independently
(separate `predict_batch` list entries are never cross-attended — the API
gives this for free), `run_multivariate_backtest` stacks all series as
variates of one context per origin, exercising TimesFM-3's headline v3
feature (full cross-variate attention).

Evaluation must use the `observed` column, not every row: forward-filled
target points are not real observations, and scoring against them silently
inflates accuracy on flat stretches (plan findings #9/#10). That filtering
happens in each experiment script, not here — this module's job is only to
produce a correct, complete table.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import baselines, config
from .model import Forecaster, forecast_batch
from .windows import context_slice, target_indices


@dataclass(frozen=True)
class ValueTransform:
    """A domain transform applied to context before forecasting, and
    inverted on every output (model point/quantiles AND baseline
    forecasts) before scoring — `actual` is never transformed, so every
    transform is scored on equal footing in the original domain.

    This is what the raw-vs-log1p ablation (docs/talk-outline.md) needs:
    since TimesFM-3's point forecast is defined as the median quantile
    (verified against the installed timesfm3 source, not assumed), any
    monotonic transform's point forecast survives inversion exactly like
    its quantiles do — there is no median-vs-mean mismatch to correct for.
    """

    name: str
    forward: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]


IDENTITY_TRANSFORM = ValueTransform("identity", lambda x: x, lambda x: x)
LOG1P_TRANSFORM = ValueTransform("log1p", np.log1p, np.expm1)


@dataclass
class SeriesData:
    """One series' aligned data: values, dates, and which points are truly
    observed (vs. forward-filled)."""

    name: str
    values: np.ndarray  # shape (T,)
    dates: np.ndarray  # shape (T,), datetime64
    observed: np.ndarray  # shape (T,), bool — True where `values` is a real observation

    def __post_init__(self):
        n = len(self.values)
        if len(self.dates) != n or len(self.observed) != n:
            raise ValueError(
                f"{self.name}: values/dates/observed length mismatch "
                f"({n}, {len(self.dates)}, {len(self.observed)})"
            )
        dates64 = np.asarray(self.dates, dtype="datetime64[ns]")
        if n > 1 and np.any(np.diff(dates64) <= np.timedelta64(0, "ns")):
            raise ValueError(f"{self.name}: dates must be strictly increasing and unique")
        if not np.all(np.isfinite(self.values)):
            bad = np.flatnonzero(~np.isfinite(self.values))
            preview = bad[:10].tolist()
            suffix = f" (+{len(bad) - 10} more)" if len(bad) > 10 else ""
            raise ValueError(
                f"{self.name}: {len(bad)} non-finite value(s) at indices {preview}{suffix} — "
                "the data pipeline should forward-fill or drop these before constructing "
                "SeriesData; TimesFM-3 cannot accept NaN/inf in its context"
            )


def _assert_aligned(series_list: list[SeriesData]) -> None:
    """Multivariate stacking requires every series to share the same dates —
    otherwise `context_slice(origin, ...)` would silently pull mismatched
    calendar positions from different series into one variate stack."""
    if not series_list:
        raise ValueError("series_list must be non-empty")
    ref = series_list[0]
    for s in series_list[1:]:
        if len(s.dates) != len(ref.dates) or not np.array_equal(s.dates, ref.dates):
            raise ValueError(
                f"'{s.name}' is not date-aligned with '{ref.name}' — "
                "multivariate stacking requires a common calendar."
            )


def _baseline_forecasts(
    context: np.ndarray, horizon: int, season_length: int | None
) -> dict[str, np.ndarray]:
    out = {
        "naive": baselines.naive_forecast(context, horizon),
        "drift": baselines.drift_forecast(context, horizon),
    }
    if season_length and len(context) >= season_length:
        out["seasonal_naive"] = baselines.seasonal_naive_forecast(context, horizon, season_length)
    # ETS can fail to converge on short/degenerate contexts — skip that row's
    # ETS baseline rather than crashing the whole run.
    with contextlib.suppress(Exception):
        out["ets"] = baselines.ets_forecast(context, horizon, seasonal_periods=season_length)
    return out


def _rows_for_one_series_forecast(
    s: SeriesData,
    origin: int,
    point: np.ndarray,
    quantiles: np.ndarray,
    context_len: int,
    max_horizon: int,
    season_length: int | None,
    mode_label: str,
    transform: ValueTransform = IDENTITY_TRANSFORM,
    make_positive: bool = True,
) -> list[dict]:
    """`point`/`quantiles` are the model's raw output IN THE TRANSFORMED
    DOMAIN (i.e. what the model actually saw as context) — this function
    inverse-transforms them, and computes+inverse-transforms baselines from
    the same transformed context, before comparing anything to `actual`
    (always the original-domain observation, never transformed).
    """
    tgt = target_indices(origin, max_horizon)
    ctx_raw = s.values[context_slice(origin, context_len)]
    ctx_transformed = transform.forward(ctx_raw)
    baseline_fc = {
        name: transform.inverse(arr)
        for name, arr in _baseline_forecasts(ctx_transformed, max_horizon, season_length).items()
    }
    point = transform.inverse(point)
    quantiles = transform.inverse(quantiles)

    rows = []
    for h_step, t_idx in enumerate(tgt):
        if t_idx >= len(s.values):
            break  # defensive: only reachable with custom (non-valid_origins) origins
        row = {
            "mode": mode_label,
            "transform": transform.name,
            "make_positive": make_positive,
            "series": s.name,
            "origin_index": origin,
            "origin_date": s.dates[origin] if origin < len(s.dates) else pd.NaT,
            "target_index": int(t_idx),
            "target_date": s.dates[t_idx],
            "horizon_step": h_step + 1,
            "actual": float(s.values[t_idx]),
            "observed": bool(s.observed[t_idx]),
            "forecast": float(point[h_step]),
        }
        for level, q in zip(config.QUANTILE_LEVELS, quantiles[h_step], strict=True):
            row[f"q{round(level * 100):02d}"] = float(q)
        for name, arr in baseline_fc.items():
            row[f"baseline_{name}"] = float(arr[h_step]) if h_step < len(arr) else np.nan
        rows.append(row)
    return rows


def run_univariate_backtest(
    forecaster: Forecaster,
    series_list: list[SeriesData],
    origins: np.ndarray,
    context_len: int,
    max_horizon: int,
    season_length: int | None = None,
    use_symmetric_averaging: bool = True,
    make_positive: bool = True,
    mode_label: str = "timesfm3_univariate",
    transform: ValueTransform = IDENTITY_TRANSFORM,
) -> pd.DataFrame:
    """Forecasts every series at every origin independently, in one batched
    call (no cross-series attention). `transform` (e.g. LOG1P_TRANSFORM)
    is applied to context before forecasting and inverted on every output —
    see ValueTransform's docstring.

    Associates each forecast_batch output back to its (series, origin) by
    ts_id (`forecast_batch` guarantees the returned ts_ids are exactly the
    requested set — see model.py), never by position: nothing in the
    Forecaster protocol promises predict_batch preserves input order.
    """
    contexts, ts_ids = [], []
    meta_by_ts_id: dict[str, tuple[int, SeriesData]] = {}
    for s in series_list:
        for origin in origins:
            origin = int(origin)
            ctx = s.values[context_slice(origin, context_len)]
            contexts.append(transform.forward(ctx))
            ts_id = f"{s.name}::{origin}"
            ts_ids.append(ts_id)
            meta_by_ts_id[ts_id] = (origin, s)

    batch = forecast_batch(
        forecaster,
        contexts,
        max_horizon,
        ts_ids=ts_ids,
        use_symmetric_averaging=use_symmetric_averaging,
        make_positive=make_positive,
    )

    rows = []
    for i, ts_id in enumerate(batch.ts_ids):
        origin, s = meta_by_ts_id[ts_id]
        rows.extend(
            _rows_for_one_series_forecast(
                s,
                origin,
                batch.forecast[i],
                batch.quantiles[i],
                context_len,
                max_horizon,
                season_length,
                mode_label,
                transform,
                make_positive,
            )
        )
    return pd.DataFrame(rows)


def run_multivariate_backtest(
    forecaster: Forecaster,
    series_list: list[SeriesData],
    origins: np.ndarray,
    context_len: int,
    max_horizon: int,
    season_length: int | None = None,
    use_symmetric_averaging: bool = True,
    make_positive: bool = True,
    mode_label: str = "timesfm3_multivariate",
    transform: ValueTransform = IDENTITY_TRANSFORM,
) -> pd.DataFrame:
    """Stacks all series as variates of one context per origin — full
    cross-variate attention across them, TimesFM-3's headline v3 feature.
    Requires every series in `series_list` to share the same date index.
    `transform` behaves as in `run_univariate_backtest`.

    One ts_id per origin (the whole variate stack for that origin); results
    are re-associated to their origin by that ts_id, not by position — see
    run_univariate_backtest's docstring. Variate order *within* one origin's
    stacked context is a separate assumption (the model must not reorder
    variates inside a single call) that this function still relies on, since
    predict_batch has no per-variate id to check against.
    """
    _assert_aligned(series_list)

    contexts, ts_ids = [], []
    meta_by_ts_id: dict[str, int] = {}
    for origin in origins:
        origin = int(origin)
        stacked = np.stack(
            [transform.forward(s.values[context_slice(origin, context_len)]) for s in series_list],
            axis=0,
        )
        contexts.append(stacked)
        ts_id = f"multivariate::{origin}"
        ts_ids.append(ts_id)
        meta_by_ts_id[ts_id] = origin

    batch = forecast_batch(
        forecaster,
        contexts,
        max_horizon,
        ts_ids=ts_ids,
        use_symmetric_averaging=use_symmetric_averaging,
        make_positive=make_positive,
    )
    # batch.forecast shape: (n_origins, n_series, max_horizon)
    # batch.quantiles shape: (n_origins, n_series, max_horizon, N_QUANTILES)

    rows = []
    for i, ts_id in enumerate(batch.ts_ids):
        origin = meta_by_ts_id[ts_id]
        for j, s in enumerate(series_list):
            rows.extend(
                _rows_for_one_series_forecast(
                    s,
                    origin,
                    batch.forecast[i, j],
                    batch.quantiles[i, j],
                    context_len,
                    max_horizon,
                    season_length,
                    mode_label,
                    transform,
                    make_positive,
                )
            )
    return pd.DataFrame(rows)
