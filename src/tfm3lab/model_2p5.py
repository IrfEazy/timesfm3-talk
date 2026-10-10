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

Model settings (differ from TimesFM-3's): 2.5 is compiled with its
README-recommended ForecastConfig -- normalize_inputs=True,
use_continuous_quantile_head=True, force_flip_invariance=True,
infer_is_positive=True, fix_quantile_crossing=True, per_core_batch_size=128,
max_context=512, max_horizon=64. TimesFM-3 uses its own evaluator settings
(model.load_forecaster, per_core_batch_size 16), so 2.5 vs 3 is a
model-as-recommended comparison, not like-for-like. Flip invariance runs each
series twice, so 2.5 latency numbers include roughly 2x compute.

Output layout (timesfm_2p5_torch.py:512-513): forecast() returns
(full_forecast[..., 5], full_forecast) with full_forecast (n, h, 10): column 0
is the mean, 1..9 are q10..q90. The adapter drops the mean column and returns
the 9 quantile columns, so shapes match config.N_QUANTILES and the point
forecast equals q50.

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

# ForecastConfig limits for compile(): v2 grids use context <= 512 and
# horizon <= 64. compile() rounds max_context up to a multiple of the input
# patch (32) and max_horizon up to the output patch (128); it raises if
# max_context + max_horizon > 16384 or, with the continuous quantile head, if
# max_horizon > 1024 (timesfm_2p5_torch.py:392-413).
MAX_CONTEXT = 512
MAX_HORIZON = 64
# per_core_batch_size: compile() sets global_batch_size = per_core_batch_size *
# device_count (timesfm_2p5_torch.py:386-388), and forecast() runs inputs in
# chunks of that size. 128 lets the largest backtest batch (up to 128 series)
# run as ONE device batch on the single Kaggle GPU; the 200M model at context
# 512 fits comfortably (flip invariance doubles the work, not the batch).
# TimesFM-3 stays at model.load_forecaster's default of 16: the two are not
# configured alike (see the module docstring).
PER_CORE_BATCH_SIZE = 128


def forecast_config_kwargs(
    max_context: int = MAX_CONTEXT, max_horizon: int = MAX_HORIZON
) -> dict[str, Any]:
    """ForecastConfig fields passed to TimesFM_2p5_200M_torch.compile (the
    upstream README settings): normalize_inputs (RevIN, robust to price scale),
    use_continuous_quantile_head (avoids quantile collapse), force_flip_invariance,
    infer_is_positive (non-negative output for non-negative input) and
    fix_quantile_crossing. Compiled output is (n, horizon, 10): column 0 = mean,
    columns 1..9 = q10..q90; the point forecast is column 5 (median, q50)."""
    return {
        "max_context": max_context,
        "max_horizon": max_horizon,
        "per_core_batch_size": PER_CORE_BATCH_SIZE,
        "normalize_inputs": True,
        "use_continuous_quantile_head": True,
        "force_flip_invariance": True,
        "infer_is_positive": True,
        "fix_quantile_crossing": True,
    }


def compile_2p5(
    model: Any, max_context: int = MAX_CONTEXT, max_horizon: int = MAX_HORIZON
) -> None:
    """compile(ForecastConfig(...)) is mandatory before forecast(): without it
    forecast() raises 'Model is not compiled. Please call compile() first.'
    (timesfm_2p5_base.py:159-160)."""
    from timesfm import ForecastConfig

    model.compile(ForecastConfig(**forecast_config_kwargs(max_context, max_horizon)))


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

    def __init__(self, model: Any, max_context: int | None = None, max_horizon: int | None = None):
        self._model = model
        self._max_context = max_context
        self._max_horizon = max_horizon

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
        if self._max_horizon is not None and horizon > self._max_horizon:
            raise ValueError(f"horizon {horizon} exceeds compiled max_horizon {self._max_horizon}")
        if self._max_context is not None:
            longest = max((len(a) for a in arrays), default=0)
            if longest > self._max_context:
                raise ValueError(
                    f"context {longest} exceeds compiled max_context {self._max_context}; "
                    "the model would silently truncate it"
                )
        ts_ids = list(ts_ids) if ts_ids is not None else [str(i) for i in range(len(contexts))]
        point, quantiles = self._model.forecast(horizon=horizon, inputs=arrays)
        return [
            _Output(
                ts_id=ts_id,
                forecast=np.asarray(point[i]),
                quantiles=np.asarray(quantiles[i])[..., 1:],
            )
            for i, ts_id in enumerate(ts_ids)
        ]


def load_forecaster_2p5(
    repo_id: str = DEFAULT_REPO_ID, revision: str | None = None, **kwargs: Any
) -> TimesFM2p5Adapter:
    """Loads the real TimesFM-2.5 checkpoint via the bundled legacy API.
    `timesfm` is imported inside this function, not at module load, so
    importing tfm3lab.model_2p5 (and running its unit tests against a fake
    model) never requires torch -- mirrors model.py's load_forecaster.
    Not called by this branch (no GPU/live-checkpoint run performed).

    `revision` (default config.CHECKPOINT_REVISION_2P5) pins the exact hub
    commit. The legacy load_checkpoint(path) takes only a local path, so a
    pinned revision is fetched with huggingface_hub.snapshot_download(repo_id,
    revision) and the resulting local directory is loaded; without a revision
    the behavior is unchanged (load_checkpoint(repo_id)).
    """
    import timesfm

    from tfm3lab import config

    revision = revision or config.CHECKPOINT_REVISION_2P5
    model = timesfm.TimesFM_2p5_200M_torch(**kwargs)
    if revision:
        from huggingface_hub import snapshot_download

        model.load_checkpoint(snapshot_download(repo_id, revision=revision))
    else:
        model.load_checkpoint(repo_id)
    compile_2p5(model)
    return TimesFM2p5Adapter(model, max_context=MAX_CONTEXT, max_horizon=MAX_HORIZON)
