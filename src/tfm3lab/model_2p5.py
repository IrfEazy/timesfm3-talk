"""Adapter making TimesFM-2.5 satisfy the same `Forecaster` protocol
model.py's `forecast_batch` already consumes -- so it works as a drop-in
zero-shot historical baseline alongside TimesFM-3, without backtest.py
needing to know which model it's talking to.

Confirmed (see docs/superpowers/specs/2026-09-04-mtg-benchmark-preregistration-design.md):
no new/conflicting dependency -- the already-pinned `timesfm>=3.0.0`
distribution bundles TimesFM-2.5's own legacy-API implementation
(`timesfm.TimesFM_2p5_200M_torch`). Its default checkpoint
(google/timesfm-2.5-200m-pytorch) is Apache-2.0 and ungated, unlike the v3
checkpoint -- no HF login required even to load it for real.

Open, UNVERIFIED risks (no live inference run performed by this branch):
  - the exact quantile grid TimesFM-2.5 returns has not been checked
    against a real checkpoint load -- this adapter does NOT assume it
    matches config.N_QUANTILES/config.QUANTILE_LEVELS; if it differs,
    routing this adapter through model.forecast_batch raises loudly at
    assert_quantile_shape (see tests/test_model_2p5.py), it does not
    silently misalign the results table.
  - the assumed (n_series, horizon[, n_quantiles]) batch-array shape
    TimesFM_2p5_200M_torch.forecast() returns is inferred from its type
    signature (`-> tuple[np.ndarray, np.ndarray]`), not verified against a
    real call.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_REPO_ID = "google/timesfm-2.5-200m-pytorch"


@dataclasses.dataclass
class _Output:
    ts_id: str
    forecast: np.ndarray
    quantiles: np.ndarray


class TimesFM2p5Adapter:
    """Wraps a loaded TimesFM_2p5_200M_torch instance to satisfy
    model.Forecaster's predict_batch shape.

    `use_symmetric_averaging`/`use_znorm`/`sort_quantiles`/`padding_mode`
    are accepted for protocol compatibility but have no equivalent in
    TimesFM_2p5_200M_torch.forecast() and are silently unused -- they are
    implementation/performance details, not semantic content, so dropping
    them cannot silently change a forecast's meaning the way dropping
    covariates would. `past_only_covariates`/`past_future_covariates` DO
    change a forecast's meaning if silently dropped, so a non-None value
    for either raises instead.

    UNIVARIATE ONLY: TimesFM_2p5_200M_torch.forecast() has no variate
    concept, so a 2-D context (as run_multivariate_backtest stacks, one
    (n_series, context_len) array per origin) has no meaning here and
    raises NotImplementedError rather than being silently flattened or
    misread. scripts/02b_exp_mtg_benchmark.py rejects `multivariate*`
    modes for this adapter up front, before any checkpoint load.
    """

    def __init__(self, model: Any):
        self._model = model

    def predict_batch(
        self,
        contexts: Sequence[np.ndarray],
        horizon: int,
        past_only_covariates=None,
        past_future_covariates=None,
        ts_ids: list[str] | None = None,
        return_quantiles: bool = True,
        use_symmetric_averaging: bool = True,
        make_positive: bool = True,
        sort_quantiles: bool = True,
        use_znorm: bool = False,
        padding_mode: str = "none",
    ) -> list[_Output]:
        if past_only_covariates is not None or past_future_covariates is not None:
            raise NotImplementedError(
                "TimesFM2p5Adapter does not support covariates -- the underlying "
                "TimesFM_2p5_200M_torch.forecast() call has no covariate parameters"
            )
        arrays = []
        for c in contexts:
            arr = np.asarray(c)
            if arr.ndim > 1:
                raise NotImplementedError(
                    "TimesFM2p5Adapter does not support multivariate contexts -- "
                    "TimesFM_2p5_200M_torch.forecast() has no variate concept, so a "
                    f"stacked context (got shape {arr.shape}) would be misread rather "
                    "than jointly modelled; use the TimesFM-3 forecaster for "
                    "multivariate/multivariate_placebo modes"
                )
            arrays.append(np.asarray(arr, dtype=float))
        ts_ids = list(ts_ids) if ts_ids is not None else [str(i) for i in range(len(contexts))]
        point, quantiles = self._model.forecast(horizon=horizon, inputs=arrays)
        return [
            _Output(ts_id=ts_id, forecast=np.asarray(point[i]), quantiles=np.asarray(quantiles[i]))
            for i, ts_id in enumerate(ts_ids)
        ]


def load_forecaster_2p5(repo_id: str = DEFAULT_REPO_ID, **kwargs: Any) -> TimesFM2p5Adapter:
    """Loads the real TimesFM-2.5 checkpoint via the bundled legacy API.
    `timesfm` is imported inside this function, not at module load, so
    importing tfm3lab.model_2p5 (and running its unit tests against a fake
    model) never requires torch -- mirrors model.py's load_forecaster.
    Not called by this branch (no GPU/live-checkpoint run performed).
    """
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch(**kwargs)
    model.load_checkpoint(repo_id)
    return TimesFM2p5Adapter(model)
