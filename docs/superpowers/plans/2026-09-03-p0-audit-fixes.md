# P0 Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the five P0 methodological bugs flagged in `openai.review.md` (quantile
diagnostic mislabeling, forward-filled data treated as real in figures, positional
input/output mismatch risk in the backtest engine, missing ts_id-leakage test, missing
`SeriesData` validation) so numbers that land in `docs/talk-outline.md`/`slides/` are
trustworthy.

**Architecture:** Targeted fixes in `src/tfm3lab/{model,backtest,figdata,plots}.py`,
each with new/updated unit tests using the existing `FakeForecaster` pattern
(`tests/conftest.py`), plus one new opt-in real-checkpoint test. No new modules, no
new dependencies, no fetch/GPU/checkpoint calls during implementation.

**Tech Stack:** Python 3.12, pandas, numpy, pytest, ruff (existing stack — unchanged).

**Spec:** `docs/analysis-plan.md` (Prompt 1) and `openai.review.md`'s three "P0" sections
(quantile diagnostic, forward-fill-as-real-data, ts_id/positional association) plus its
`SeriesData` and ts_id-invariance asks folded into the same prompt.

## Global Constraints

- `uv` only — never `pip`/`uv pip`. Run tests via `uv run pytest`, lint via `uv run ruff check`.
- Never hand-edit `results/*.parquet` or type a number into `docs/talk-outline.md`/`slides/` —
  only placeholders (`[NUMERO]`) for anything that needs a real re-run.
- Tests stay offline/no-GPU by default. The one new test that needs the real checkpoint
  must be gated behind `TFM3LAB_RUN_MODEL_SMOKE=1`, matching `tests/test_model_smoke.py`'s
  existing convention.
- Every metric/aggregation must keep filtering on `observed == True` — never widen that.
- `windows.py`'s convention (`origin` = first predicted index) is untouched by this plan.
- Add a unit test for every bug fixed. No fetch, no GPU, no real TimesFM-3 checkpoint load
  during implementation or verification — `uv run pytest` (default, no opt-in env vars set).

---

## Task 1: `forecast_batch` validates ts_id uniqueness and completeness

**Files:**
- Modify: `src/tfm3lab/model.py:109-155` (`forecast_batch`)
- Modify: `tests/conftest.py` (add `ReversedFakeForecaster`, `MismatchedTsIdForecaster`)
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: nothing new — same `Forecaster.predict_batch` protocol already in `model.py`.
- Produces: `forecast_batch` now raises `ValueError` (not silently misassociates) when
  `ts_ids` has duplicates, when `len(ts_ids) != len(contexts)`, or when the forecaster's
  returned `ts_id`s don't exactly match the requested set. `BatchForecast.ts_ids` continues
  to reflect the forecaster's actual output order (unchanged field meaning) — later tasks
  rely on this to re-associate by ts_id instead of position.
- `tests/conftest.py` gains two classes later tasks also use:
  `ReversedFakeForecaster(FakeForecaster)` (yields outputs in reverse ts_id order) and
  `MismatchedTsIdForecaster(FakeForecaster)` (relabels the first output to an
  unrequested ts_id).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model.py` (after the existing `test_at_horizon_rejects_horizon_beyond_the_call`):

```python
from .conftest import MismatchedTsIdForecaster, ReversedFakeForecaster


def test_forecast_batch_rejects_duplicate_ts_ids():
    fake = _FakeForecaster()
    with pytest.raises(ValueError, match="unique"):
        forecast_batch(fake, [np.array([1.0]), np.array([2.0])], max_horizon=1, ts_ids=["a", "a"])


def test_forecast_batch_rejects_ts_id_count_mismatch():
    fake = _FakeForecaster()
    with pytest.raises(ValueError, match="got 1 ts_ids"):
        forecast_batch(fake, [np.array([1.0]), np.array([2.0])], max_horizon=1, ts_ids=["a"])


def test_forecast_batch_rejects_output_ts_ids_not_matching_request():
    fake = MismatchedTsIdForecaster()
    with pytest.raises(ValueError, match="missing="):
        forecast_batch(fake, [np.array([1.0]), np.array([2.0])], max_horizon=1, ts_ids=["a", "b"])


def test_forecast_batch_preserves_reversed_output_order_in_ts_ids():
    fake = ReversedFakeForecaster()
    result = forecast_batch(
        fake, [np.array([1.0, 2.0]), np.array([10.0, 20.0])], max_horizon=1, ts_ids=["a", "b"]
    )
    assert result.ts_ids == ["b", "a"]
    # forecast[0] must belong to "b" (last context value 20.0), not "a" — BatchForecast.ts_ids
    # and BatchForecast.forecast must stay in lockstep with whatever order the forecaster used.
    np.testing.assert_allclose(result.forecast[0], 20.0)
    np.testing.assert_allclose(result.forecast[1], 2.0)
```

Add to `tests/conftest.py`, after the existing `FakeForecaster` class:

```python
class ReversedFakeForecaster(FakeForecaster):
    """Same outputs as FakeForecaster but yielded in reverse ts_id order —
    exercises callers that must not assume predict_batch preserves input
    order (the P0 "ts_id association" fix in model.py/backtest.py)."""

    def predict_batch(self, contexts, horizon, **kwargs):
        self.last_call_kwargs = kwargs
        levels = np.linspace(0.1, 0.9, self.n_quantiles)
        ts_ids = kwargs.get("ts_ids") or [None] * len(contexts)
        items = []
        for ts_id, ctx in zip(ts_ids, contexts, strict=True):
            ctx = np.asarray(ctx, dtype=float)
            last = ctx[-1] if ctx.ndim == 1 else ctx[:, -1]
            point = np.broadcast_to(np.asarray(last)[..., None], (*np.shape(last), horizon)).copy()
            quant = point[..., None] + levels
            items.append(FakeOutput(ts_id, point, quant))
        yield from reversed(items)


class MismatchedTsIdForecaster(FakeForecaster):
    """Relabels the first output's ts_id to one the caller never requested —
    simulates a forecaster that drops/renames a ts_id, which forecast_batch
    must reject rather than silently misassociate."""

    def predict_batch(self, contexts, horizon, **kwargs):
        outputs = list(super().predict_batch(contexts, horizon, **kwargs))
        if outputs:
            outputs[0].ts_id = "unrequested_id"
        yield from outputs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model.py -k "ts_id or reversed" -v`
Expected: FAIL — `forecast_batch` currently accepts duplicate/mismatched ts_ids silently
(no `ValueError` raised), and `ReversedFakeForecaster`/`MismatchedTsIdForecaster` don't
exist yet in `conftest.py` (collection error) until Step 1's conftest edit is saved too.

- [ ] **Step 3: Implement the validation in `model.py`**

Replace `forecast_batch`'s body in `src/tfm3lab/model.py` (lines 109-155) with:

```python
def forecast_batch(
    forecaster: Forecaster,
    contexts: Sequence[np.ndarray],
    max_horizon: int,
    ts_ids: list[str] | None = None,
    past_only_covariates: list[np.ndarray | None] | None = None,
    past_future_covariates: list[np.ndarray | None] | None = None,
    use_symmetric_averaging: bool = True,
    make_positive: bool = True,
) -> BatchForecast:
    """Runs one predict_batch call at `max_horizon`.

    `use_symmetric_averaging` and `make_positive` default to
    TimesFM3Evaluator's own official benchmark defaults (True, True) so
    numbers stay comparable to what Google reports; the cost/latency
    experiment (scripts/03..05) is exactly where a caller should override
    `use_symmetric_averaging=False` to measure the ~2x compute it costs
    (each context is run once as-is and once negated, then averaged).

    Validates ts_ids before and after the call: duplicates or a
    count mismatch are rejected up front, and the forecaster's returned
    ts_ids must be exactly the requested set (no missing, no extra) —
    `BatchForecast.ts_ids` reflects the forecaster's actual output order,
    and callers (backtest.py) must re-associate results by that ts_id, never
    by raw position, since nothing in the Forecaster protocol guarantees
    predict_batch preserves input order.
    """
    ts_ids = list(ts_ids) if ts_ids is not None else [str(i) for i in range(len(contexts))]
    if len(ts_ids) != len(contexts):
        raise ValueError(f"got {len(ts_ids)} ts_ids for {len(contexts)} contexts")
    if len(set(ts_ids)) != len(ts_ids):
        seen: set[str] = set()
        dupes: set[str] = set()
        for t in ts_ids:
            (dupes if t in seen else seen).add(t)
        raise ValueError(f"ts_ids must be unique, got duplicates: {sorted(dupes)}")

    start = time.perf_counter()
    outputs = list(
        forecaster.predict_batch(
            contexts=list(contexts),
            horizon=max_horizon,
            past_only_covariates=past_only_covariates,
            past_future_covariates=past_future_covariates,
            ts_ids=ts_ids,
            return_quantiles=True,
            use_symmetric_averaging=use_symmetric_averaging,
            make_positive=make_positive,
        )
    )
    latency = time.perf_counter() - start

    output_ids = [o.ts_id for o in outputs]
    requested, returned = set(ts_ids), set(output_ids)
    if len(output_ids) != len(ts_ids) or requested != returned:
        missing = requested - returned
        extra = returned - requested
        raise ValueError(
            f"forecaster.predict_batch returned {len(output_ids)} outputs for "
            f"{len(ts_ids)} requested ts_ids — missing={sorted(missing)}, extra={sorted(extra)}"
        )

    forecasts = np.stack([o.forecast for o in outputs], axis=0)
    quantiles = np.stack([o.quantiles for o in outputs], axis=0)
    assert_quantile_shape(quantiles)

    return BatchForecast(
        ts_ids=output_ids,
        forecast=forecasts,
        quantiles=quantiles,
        latency_seconds=latency,
        n_series=len(outputs),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model.py -v`
Expected: PASS (all tests in the file, including the pre-existing ones).

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/model.py tests/conftest.py tests/test_model.py
git commit -m "fix: reject duplicate/mismatched ts_ids in forecast_batch"
```

---

## Task 2: Fix positional input/output association in `backtest.py`

**Files:**
- Modify: `src/tfm3lab/backtest.py:158-268` (`run_univariate_backtest`, `run_multivariate_backtest`)
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `model.forecast_batch` (Task 1) — specifically relies on `BatchForecast.ts_ids`
  reflecting actual output order, and on `forecast_batch` now raising on any ts_id mismatch.
- Produces: `run_univariate_backtest`/`run_multivariate_backtest` unchanged public
  signatures and returned DataFrame schema — only the internal association logic changes,
  from positional (`batch.forecast[i]` assumed to match the i-th *input*) to ts_id-keyed
  (`batch.forecast[i]` looked up by `batch.ts_ids[i]` against a `meta_by_ts_id` dict built
  from the *inputs*).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest.py`:

```python
from .conftest import ReversedFakeForecaster


def test_univariate_backtest_correct_with_reversed_output_order():
    a = _make_series("a", np.arange(30.0))
    b = _make_series("b", np.arange(30.0) * 10)
    context_len, horizon = 5, 3
    origins = valid_origins(n=30, context_len=context_len, horizon=horizon)

    df_forward = run_univariate_backtest(FakeForecaster(), [a, b], origins, context_len, horizon)
    df_reversed = run_univariate_backtest(
        ReversedFakeForecaster(), [a, b], origins, context_len, horizon
    )

    sort_cols = ["series", "origin_index", "horizon_step"]
    left = df_forward.sort_values(sort_cols).reset_index(drop=True)
    right = df_reversed.sort_values(sort_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_multivariate_backtest_correct_with_reversed_output_order():
    a = _make_series("a", np.arange(1.0, 21.0))
    b = _make_series("b", np.arange(1.0, 21.0) * 100)
    origins = valid_origins(n=20, context_len=5, horizon=3)

    df_forward = run_multivariate_backtest(
        FakeForecaster(), [a, b], origins, context_len=5, max_horizon=3
    )
    df_reversed = run_multivariate_backtest(
        ReversedFakeForecaster(), [a, b], origins, context_len=5, max_horizon=3
    )

    sort_cols = ["series", "origin_index", "horizon_step"]
    left = df_forward.sort_values(sort_cols).reset_index(drop=True)
    right = df_reversed.sort_values(sort_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backtest.py -k reversed_output_order -v`
Expected: FAIL — with the current positional association, `df_reversed`'s rows for series
"a" get series "b"'s (or the reversed origin's) forecast values, so
`pd.testing.assert_frame_equal` reports mismatched `forecast`/quantile columns.

- [ ] **Step 3: Implement ts_id-keyed association**

In `src/tfm3lab/backtest.py`, replace `run_univariate_backtest`'s body (lines 158-208) with:

```python
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
            )
        )
    return pd.DataFrame(rows)
```

Replace `run_multivariate_backtest`'s body (lines 211-268) with:

```python
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
                )
            )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: PASS (all tests in the file, including the pre-existing ones — the FakeForecaster
already preserves order, so those keep passing unchanged; the new reversed-order tests now
pass too).

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/backtest.py tests/test_backtest.py
git commit -m "fix: associate backtest outputs to inputs by ts_id, not position"
```

---

## Task 3: `SeriesData` validation (sorted/unique dates, finite values)

**Files:**
- Modify: `src/tfm3lab/backtest.py:56-72` (`SeriesData.__post_init__`)
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `SeriesData(...)` now raises `ValueError` at construction time for
  non-strictly-increasing/duplicate `dates`, or non-finite `values` — in addition to the
  existing length-mismatch check. No change to the dataclass's fields.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest.py`:

```python
def test_series_data_rejects_unsorted_dates():
    values = np.array([1.0, 2.0, 3.0])
    dates = np.array(["2024-01-03", "2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="strictly increasing"):
        SeriesData(name="bad", values=values, dates=dates, observed=np.ones(3, dtype=bool))


def test_series_data_rejects_duplicate_dates():
    values = np.array([1.0, 2.0, 3.0])
    dates = np.array(["2024-01-01", "2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="strictly increasing"):
        SeriesData(name="bad", values=values, dates=dates, observed=np.ones(3, dtype=bool))


def test_series_data_rejects_non_finite_values():
    values = np.array([1.0, np.nan, 3.0])
    dates = pd.date_range("2024-01-01", periods=3).to_numpy()
    with pytest.raises(ValueError, match="non-finite"):
        SeriesData(name="bad", values=values, dates=dates, observed=np.ones(3, dtype=bool))


def test_series_data_accepts_valid_series():
    values = np.array([1.0, 2.0, 3.0])
    dates = pd.date_range("2024-01-01", periods=3).to_numpy()
    s = SeriesData(name="ok", values=values, dates=dates, observed=np.ones(3, dtype=bool))
    assert s.name == "ok"  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backtest.py -k "series_data_rejects" -v`
Expected: FAIL — `SeriesData.__post_init__` currently only checks lengths, so unsorted
dates, duplicate dates, and NaN values all construct successfully today (no `ValueError`).

- [ ] **Step 3: Implement the validation**

Replace `SeriesData.__post_init__` in `src/tfm3lab/backtest.py` (lines 66-72) with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/backtest.py tests/test_backtest.py
git commit -m "feat: validate SeriesData dates and finiteness at construction"
```

---

## Task 4: Opt-in ts_id-invariance test against the real checkpoint

**Files:**
- Create: `tests/test_model_ts_id_invariance.py`

**Interfaces:**
- Consumes: `tfm3lab.model.forecast_batch`, `tfm3lab.model.load_forecaster` (unchanged by
  this task — Task 1 already made `forecast_batch` ts_id-safe).
- Produces: nothing consumed by later tasks — this is a standalone opt-in test file,
  skipped by default like `tests/test_model_smoke.py`.

- [ ] **Step 1: Write the test file**

Create `tests/test_model_ts_id_invariance.py`:

```python
"""Opt-in test: does the real TimesFM-3 checkpoint's forecast depend on the
ts_id label passed to predict_batch? It shouldn't — ts_id is meant to be an
opaque tracking label, not a model input. If this test ever fails, a card's
NAME (e.g. "The One Ring [LTR]") would be a channel for metadata leakage
into the forecast, which would silently invalidate every zero-shot claim in
this project's talk (the model would be reacting to the label, not learning
from the time series alone) — see openai.review.md's ts_id-invariance ask.

Skipped by default: needs network, the gated HF checkpoint, and either
`hf auth login` or an HF_TOKEN env var — reuses the same opt-in gate as
tests/test_model_smoke.py rather than inventing a second env var.

Run explicitly with:
    TFM3LAB_RUN_MODEL_SMOKE=1 uv run pytest tests/test_model_ts_id_invariance.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tfm3lab.model import forecast_batch, load_forecaster

pytestmark = pytest.mark.skipif(
    os.environ.get("TFM3LAB_RUN_MODEL_SMOKE") != "1",
    reason=(
        "opt-in only: needs network + the gated HF checkpoint + a GPU or patience; "
        "set TFM3LAB_RUN_MODEL_SMOKE=1"
    ),
)


def test_forecast_is_invariant_to_ts_id_label():
    forecaster = load_forecaster(per_core_batch_size=1)
    rng = np.random.default_rng(0)
    context = np.cumsum(rng.normal(size=64)) + 100.0

    original = forecast_batch(forecaster, [context], max_horizon=7, ts_ids=["The One Ring [LTR]"])
    anonymous = forecast_batch(forecaster, [context], max_horizon=7, ts_ids=["series_0001"])
    randomized = forecast_batch(
        forecaster, [context], max_horizon=7, ts_ids=[f"x{int(rng.integers(0, 10**9))}"]
    )

    np.testing.assert_allclose(original.forecast, anonymous.forecast, atol=1e-6)
    np.testing.assert_allclose(original.forecast, randomized.forecast, atol=1e-6)
    np.testing.assert_allclose(original.quantiles, anonymous.quantiles, atol=1e-6)
    np.testing.assert_allclose(original.quantiles, randomized.quantiles, atol=1e-6)
```

- [ ] **Step 2: Verify it's skipped by default (no network/GPU/checkpoint touched)**

Run: `uv run pytest tests/test_model_ts_id_invariance.py -v`
Expected: `1 skipped` — the `skipif` reason string is printed, confirming the gate works
without needing the real checkpoint available in this environment.

- [ ] **Step 3: Commit**

```bash
git add tests/test_model_ts_id_invariance.py
git commit -m "test: add opt-in ts_id-invariance check against the real checkpoint"
```

---

## Task 5: Rename/rebuild the quantile diagnostic (`figdata.py`)

**Files:**
- Modify: `src/tfm3lab/figdata.py:273-310` (`pit_histogram`)
- Test: `tests/test_figdata.py`

**Interfaces:**
- Consumes: `config.QUANTILE_LEVELS`, `summarize.QUANTILE_COLUMNS` (already imported).
- Produces: new `quantile_bin_calibration(preds, horizon_steps=(1,7,28)) -> pd.DataFrame`
  with columns `horizon_step, bin_index, label, count, fraction, nominal_fraction, n` — 10
  rows per horizon step. `pit_histogram` becomes a deprecated alias
  (`pit_histogram = quantile_bin_calibration`) for any leftover caller. Task 6 (`plots.py`)
  consumes this new column set (`bin_index` instead of `bin_left`/`bin_right`).

- [ ] **Step 1: Write the failing tests**

Replace the `# --- pit_histogram ---` section in `tests/test_figdata.py` (lines 229-244)
with:

```python
# --- quantile_bin_calibration ---------------------------------------------------


def test_quantile_bin_calibration_fractions_sum_to_one_per_horizon():
    preds = _synthetic_preds(n_origins=20, horizon=1, start_origin=64, drift=0.5)
    hist = figdata.quantile_bin_calibration(preds, horizon_steps=(1,))
    assert hist["fraction"].sum() == pytest.approx(1.0)


def test_quantile_bin_calibration_has_ten_bins_with_correct_labels():
    preds = _synthetic_preds(n_origins=5, horizon=1, start_origin=64)
    hist = figdata.quantile_bin_calibration(preds, horizon_steps=(1,))
    labels = hist.sort_values("bin_index")["label"].tolist()
    assert len(labels) == 10
    assert labels[0] == "≤ q10"
    assert labels[1] == "(0.1, 0.2]"
    assert labels[-2] == "(0.8, 0.9]"
    assert labels[-1] == "> q90"
    assert hist["nominal_fraction"].eq(0.1).all()


def test_quantile_bin_calibration_below_q10_lands_in_first_bin():
    # actual well below every quantile forecast -> must count in bin 0 ("<= q10"),
    # never silently merged into the (q10, q20] bin the old 8-bin histogram produced.
    preds = _synthetic_preds(n_origins=1, horizon=1, start_origin=64)
    preds.loc[preds["horizon_step"] == 1, "actual"] = -1000.0
    hist = figdata.quantile_bin_calibration(preds, horizon_steps=(1,))
    row0 = hist[hist["bin_index"] == 0].iloc[0]
    assert row0["count"] == 1
    assert hist[hist["bin_index"] != 0]["count"].sum() == 0


def test_pit_histogram_alias_still_works():
    preds = _synthetic_preds(n_origins=5, horizon=1, start_origin=64)
    pd.testing.assert_frame_equal(
        figdata.pit_histogram(preds, horizon_steps=(1,)),
        figdata.quantile_bin_calibration(preds, horizon_steps=(1,)),
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_figdata.py -k quantile_bin_calibration -v`
Expected: FAIL — `figdata.quantile_bin_calibration` doesn't exist yet (`AttributeError`).

- [ ] **Step 3: Implement the corrected diagnostic**

Replace `pit_histogram` in `src/tfm3lab/figdata.py` (lines 273-310) with:

```python
def quantile_bin_calibration(
    preds: pd.DataFrame, horizon_steps: tuple[int, ...] = (1, 7, 28)
) -> pd.DataFrame:
    """Discrete quantile-bin calibration: 10 bins built directly from the 9
    known quantile forecasts (0.1..0.9), one horizon step at a time.

    Bin i is "how many of the 9 quantile forecasts does `actual` exceed":
    bin 0 is "actual <= q10", bin k (1..8) is "q{10k} < actual <= q{10(k+1)}",
    bin 9 is "actual > q90". Each bin has nominal probability 1/10 if the
    quantiles are calibrated.

    This replaces the old `pit_histogram`, which ran metrics.pit_values'
    *interpolated* PIT through `np.histogram(pit, bins=QUANTILE_LEVELS)` —
    9 edges make 8 bins, not 9 — and then mislabeled the outer two as
    "<= q10"/">= q90" when they actually covered [0.1, 0.2) and [0.8, 0.9].
    Counting directly against the quantile columns (no interpolation step)
    gets both the bin count and the labels right by construction.
    """
    levels = config.QUANTILE_LEVELS
    n_bins = len(levels) + 1  # 10
    rows = []
    for h in horizon_steps:
        g = preds[preds["horizon_step"] == h]
        if g.empty:
            continue
        actual = g["actual"].to_numpy(dtype=float)
        quantiles = g[QUANTILE_COLUMNS].to_numpy(dtype=float)  # shape (n, 9), sorted per row
        n = len(actual)
        # bin_index[k] = count of quantile forecasts actual[k] strictly exceeds (0..9)
        bin_index = np.sum(quantiles < actual[:, None], axis=1)
        counts = np.bincount(bin_index, minlength=n_bins)[:n_bins]
        for i in range(n_bins):
            if i == 0:
                label = "≤ q10"
            elif i == n_bins - 1:
                label = "> q90"
            else:
                label = f"({levels[i - 1]:.1f}, {levels[i]:.1f}]"
            rows.append(
                {
                    "horizon_step": h,
                    "bin_index": i,
                    "label": label,
                    "count": int(counts[i]),
                    "fraction": float(counts[i]) / n if n else float("nan"),
                    "nominal_fraction": 1.0 / n_bins,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


# Deprecated alias: the name "PIT histogram" claimed continuous-PIT semantics
# this diagnostic never had. Kept only so a leftover caller doesn't hard
# break — prefer quantile_bin_calibration in new code.
pit_histogram = quantile_bin_calibration
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_figdata.py -v`
Expected: FAIL on `test_pit_histogram_fractions_sum_to_one_per_horizon` and
`test_pit_histogram_outer_bins_labeled_as_clipped` (the two pre-existing tests referencing
the old 8-bin/`bin_left`/`bin_right` schema) — fix these in Step 5.

- [ ] **Step 5: Update the two pre-existing tests that assumed the old schema**

These were already replaced by Step 1's edit (the whole `# --- pit_histogram ---` block was
replaced) — re-run to confirm:

Run: `uv run pytest tests/test_figdata.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 6: Commit**

```bash
git add src/tfm3lab/figdata.py tests/test_figdata.py
git commit -m "fix: replace mislabeled PIT histogram with discrete quantile-bin calibration"
```

---

## Task 6: Rename the plotting side (`plots.py`) and the figure script

**Files:**
- Modify: `src/tfm3lab/plots.py:188-206` (`plot_pit_histogram`)
- Modify: `scripts/06_make_figures.py:83-91,139` (`build_pit_histogram`, FIGURES list)
- Modify: `scripts/02_exp_mtg.py:147` (comment only)
- Test: `tests/test_plots_smoke.py`

**Interfaces:**
- Consumes: `figdata.quantile_bin_calibration`'s new schema (Task 5) —
  `bin_index`, `label`, `fraction`, `nominal_fraction` (no more `bin_left`/`bin_right`).
- Produces: `plots.plot_quantile_bin_calibration(hist, axes=None)`, with
  `plot_pit_histogram` kept as a deprecated alias. `scripts/06_make_figures.py` writes
  `exp_mtg_quantile_bin_calibration.png` instead of `exp_mtg_pit_histogram.png`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_plots_smoke.py`, replace `test_plot_pit_histogram_one_panel_per_horizon`
(lines 93-104) with:

```python
def test_plot_quantile_bin_calibration_one_panel_per_horizon():
    preds = pd.DataFrame(
        {
            "horizon_step": [1] * 6 + [7] * 6,
            "actual": [1, 2, 3, 4, 5, 9] * 2,
        }
    )
    for i, level in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        preds[f"q{round(level * 100):02d}"] = i + 1
    hist = figdata.quantile_bin_calibration(preds, horizon_steps=(1, 7))
    axes = plots.plot_quantile_bin_calibration(hist)
    assert len(axes) == 2


def test_plot_pit_histogram_alias_still_works():
    preds = pd.DataFrame({"horizon_step": [1] * 3, "actual": [1, 2, 9]})
    for i, level in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        preds[f"q{round(level * 100):02d}"] = i + 1
    hist = figdata.pit_histogram(preds, horizon_steps=(1,))
    axes = plots.plot_pit_histogram(hist)
    assert len(axes) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plots_smoke.py -k "quantile_bin_calibration or pit_histogram_alias" -v`
Expected: FAIL — `plots.plot_quantile_bin_calibration` doesn't exist yet.

- [ ] **Step 3: Implement in `plots.py`**

Replace `plot_pit_histogram` in `src/tfm3lab/plots.py` (lines 188-206) with:

```python
def plot_quantile_bin_calibration(hist: pd.DataFrame, axes=None):
    """One bar panel per horizon_step in `hist`. Flat at the nominal 1/10
    line = calibrated; a U-shape (mass piling into the outer two bins) =
    intervals too narrow. Bars sit at the 10 discrete quantile bins from
    figdata.quantile_bin_calibration — this is not a continuous-PIT axis."""
    steps = sorted(hist["horizon_step"].unique())
    if axes is None:
        _, axes = plt.subplots(1, len(steps), figsize=(4.2 * len(steps), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, steps, strict=True):
        g = hist[hist["horizon_step"] == h].sort_values("bin_index")
        x = np.arange(len(g))
        ax.bar(x, g["fraction"], width=0.85, color=PALETTE["model"], alpha=0.85)
        nominal = float(g["nominal_fraction"].iloc[0]) if len(g) else 0.1
        ax.axhline(nominal, color=PALETTE["baseline"], linestyle="--", label="uniforme attesa")
        ax.set_xticks(x)
        ax.set_xticklabels(g["label"], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"h={h}")
        ax.set_xlabel("quantile bin")
    axes[0].set_ylabel("frazione")
    axes[0].legend(fontsize=8)
    return axes


# Deprecated alias — see figdata.pit_histogram.
plot_pit_histogram = plot_quantile_bin_calibration
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plots_smoke.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Update `scripts/06_make_figures.py`**

Replace `build_pit_histogram` (lines 83-91) with:

```python
def build_quantile_bin_calibration() -> None:
    preds, _ = figdata.load_mtg_predictions()
    hist = figdata.quantile_bin_calibration(preds, horizon_steps=(1, 7, 28))
    if hist.empty:
        print("  skip (no rows at horizon steps 1/7/28)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4), sharey=True)
    plots.plot_quantile_bin_calibration(hist, axes=axes)
    plots.save(fig, "exp_mtg_quantile_bin_calibration")
```

Update the `FIGURES` list entry (line 139) from
`("exp_mtg_pit_histogram", build_pit_histogram),` to
`("exp_mtg_quantile_bin_calibration", build_quantile_bin_calibration),`.

- [ ] **Step 6: Update the comment in `scripts/02_exp_mtg.py`**

Change line 147 from:
```python
    # little and breaks the PIT histogram, which needs the full grid).
```
to:
```python
    # little and breaks the quantile-bin calibration diagnostic, which needs the full grid).
```

- [ ] **Step 7: Run the full test suite and lint**

Run: `uv run ruff check src tests scripts && uv run pytest`
Expected: PASS — ruff clean, all tests pass (no fetch/GPU touched, opt-in tests skipped).

- [ ] **Step 8: Commit**

```bash
git add src/tfm3lab/plots.py scripts/06_make_figures.py scripts/02_exp_mtg.py tests/test_plots_smoke.py
git commit -m "fix: rename PIT histogram plot/figure to quantile-bin calibration"
```

---

## Task 7: `rank_windows` excludes windows with unobserved targets

**Files:**
- Modify: `src/tfm3lab/figdata.py:207-248` (`rank_windows`)
- Test: `tests/test_figdata.py`

**Interfaces:**
- Consumes: the `observed` column already present in every backtest output row
  (`backtest._rows_for_one_series_forecast` writes it — no upstream change needed).
- Produces: `rank_windows(preds, *, exclude_glitches=True, min_naive_mae=1e-9,
  require_all_observed=True)` — new `observed_fraction`/`all_targets_observed` columns in
  the output, and windows with any forward-filled target dropped by default.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_figdata.py`, after `test_rank_windows_guards_near_zero_naive_mae`:

```python
def test_rank_windows_excludes_windows_with_unobserved_targets_by_default():
    rows = _make_origin_block("A", 20, base_price=100.0, horizon=3)
    df = pd.DataFrame(rows)
    df.loc[(df["origin_index"] == 20) & (df["horizon_step"] == 2), "observed"] = False

    ranked = figdata.rank_windows(df, exclude_glitches=False)
    assert 20 not in set(ranked["origin_index"])


def test_rank_windows_keeps_unobserved_windows_when_flag_disabled():
    rows = _make_origin_block("A", 20, base_price=100.0, horizon=3)
    df = pd.DataFrame(rows)
    df.loc[(df["origin_index"] == 20) & (df["horizon_step"] == 2), "observed"] = False

    ranked = figdata.rank_windows(df, exclude_glitches=False, require_all_observed=False)
    row = ranked[ranked["origin_index"] == 20].iloc[0]
    assert row["all_targets_observed"] is False
    assert row["observed_fraction"] == pytest.approx(2 / 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_figdata.py -k unobserved -v`
Expected: FAIL — `rank_windows` currently has no `observed`/`require_all_observed` handling,
so origin 20 stays in the ranked output and `observed_fraction`/`all_targets_observed`
columns don't exist (`KeyError`).

- [ ] **Step 3: Implement the filter**

Replace `rank_windows` in `src/tfm3lab/figdata.py` (lines 207-248) with:

```python
def rank_windows(
    preds: pd.DataFrame,
    *,
    exclude_glitches: bool = True,
    min_naive_mae: float = 1e-9,
    require_all_observed: bool = True,
) -> pd.DataFrame:
    """One row per (series, origin): relative MAE, P10-P90 coverage, and the
    naive-relative price move over that window, guarded against the
    near-zero-naive-MAE blowup that makes a per-window relative-MAE MEAN
    unusable (a single near-flat card can send it to 1e8+) — callers must
    use the win-rate or the median across this table, never the mean of
    `relative_mae` directly.

    Policy for unobserved targets: a window with even one forward-filled
    (non-observed) target is, by default, dropped from the ranking entirely
    (`require_all_observed=True`) — a "hero" demo window picked by relative
    MAE or coverage must not be able to win by scoring well against an
    imputed value instead of a real observation. Set
    `require_all_observed=False` to keep such windows (e.g. for a
    data-quality diagnostic); `observed_fraction`/`all_targets_observed` are
    always reported so callers can see what was dropped.
    """
    truth = reconstruct_truth(preds)
    glitches = find_glitches(truth)
    glitch_keys = set(zip(glitches["series"], glitches["index"], strict=True))

    rows = []
    for (series, origin), g in preds.groupby(["series", "origin_index"]):
        g = g.sort_values("horizon_step")
        actual = g["actual"].to_numpy(dtype=float)
        forecast = g["forecast"].to_numpy(dtype=float)
        naive_val = float(g["baseline_naive"].iloc[0])
        naive_arr = np.full_like(actual, naive_val)
        naive_mae = mae(actual, naive_arr)
        model_mae = mae(actual, forecast)
        relative = model_mae / naive_mae if naive_mae >= min_naive_mae else float("nan")
        pct_change = (actual[-1] - naive_val) / naive_val if naive_val != 0 else float("nan")
        window_indices = set(g["target_index"].astype(int).tolist()) | {int(origin)}
        contains_glitch = any((series, idx) in glitch_keys for idx in window_indices)
        rows.append(
            {
                "series": series,
                "origin_index": int(origin),
                "pct_change": pct_change,
                "relative_mae": relative,
                "coverage": coverage(actual, g["q10"].to_numpy(), g["q90"].to_numpy()),
                "contains_glitch": contains_glitch,
                "beats_naive": bool(model_mae < naive_mae),
                "observed_fraction": float(g["observed"].mean()),
                "all_targets_observed": bool(g["observed"].all()),
            }
        )
    out = pd.DataFrame(rows)
    if exclude_glitches:
        out = out[~out["contains_glitch"]].reset_index(drop=True)
    if require_all_observed:
        out = out[out["all_targets_observed"]].reset_index(drop=True)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_figdata.py -v`
Expected: PASS (all tests in the file — the pre-existing `rank_windows` tests use
`_make_origin_block`, whose rows are `observed=True` by default via `_pred_row`, so they're
unaffected by the new default filter).

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/figdata.py tests/test_figdata.py
git commit -m "fix: rank_windows excludes windows with forward-filled targets by default"
```

---

## Task 8: `build_forecast_slice` exposes `observed_mask`/`history_observed`

**Files:**
- Modify: `src/tfm3lab/figdata.py:128-204` (`ForecastSlice`, `build_forecast_slice`)
- Test: `tests/test_figdata.py`

**Interfaces:**
- Consumes: the `observed` column on `preds` rows (per-target) and on `truth` rows
  (per-history-point, via `reconstruct_truth`'s existing `observed` column).
- Produces: `ForecastSlice` gains two new required fields, `observed_mask: np.ndarray`
  (aligned with `target_dates`/`actual`) and `history_observed: np.ndarray` (aligned with
  `history_dates`/`history_values`). `build_forecast_slice` gains
  `require_observed_targets: bool = False`. Task 9 (`plots.py`) consumes both new fields.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_figdata.py`, after `test_build_forecast_slice_coverage_and_relative_mae`:

```python
def test_build_forecast_slice_exposes_observed_mask():
    rows = _make_origin_block("A", 65, base_price=50.0, horizon=3)
    preds = pd.DataFrame(rows)
    preds.loc[(preds["origin_index"] == 65) & (preds["horizon_step"] == 2), "observed"] = False

    truth = figdata.reconstruct_truth(preds)
    sl = figdata.build_forecast_slice(preds, truth, "A", origin_index=65, history_days=5)
    np.testing.assert_array_equal(sl.observed_mask, [True, False, True])
    assert sl.history_observed.dtype == bool


def test_build_forecast_slice_require_observed_targets_raises():
    rows = _make_origin_block("A", 65, base_price=50.0, horizon=3)
    preds = pd.DataFrame(rows)
    preds.loc[(preds["origin_index"] == 65) & (preds["horizon_step"] == 2), "observed"] = False
    truth = figdata.reconstruct_truth(preds)

    with pytest.raises(ValueError, match="forward-filled"):
        figdata.build_forecast_slice(
            preds, truth, "A", origin_index=65, history_days=5, require_observed_targets=True
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_figdata.py -k build_forecast_slice_exposes_observed_mask -v`
Expected: FAIL — `ForecastSlice` has no `observed_mask`/`history_observed` fields yet
(`TypeError: __init__() got an unexpected keyword argument` is not it — the test itself
will fail with `AttributeError: 'ForecastSlice' object has no attribute 'observed_mask'`).

- [ ] **Step 3: Implement**

In `src/tfm3lab/figdata.py`, add two fields to the `ForecastSlice` dataclass (lines 128-146),
right after `contains_glitch: bool`:

```python
    contains_glitch: bool
    observed_mask: np.ndarray  # bool, aligned with target_dates/actual — True where the
    #   target is a real observation, False where it was forward-filled
    history_observed: np.ndarray  # bool, aligned with history_dates/history_values
```

Replace `build_forecast_slice` (lines 149-204) with:

```python
def build_forecast_slice(
    preds: pd.DataFrame,
    truth: pd.DataFrame,
    series: str,
    origin_index: int,
    history_days: int = 120,
    glitches: pd.DataFrame | None = None,
    require_observed_targets: bool = False,
) -> ForecastSlice:
    """Builds one hero-chart window: `history_days` of real history strictly
    before `origin_index`, then the model's forecast for that origin against
    what actually happened. Raises if `origin_index` has no forecast rows —
    a silently empty chart is worse than a loud error here.

    `require_observed_targets=True` raises if any target point in this
    window is forward-filled rather than a real observation — set this for
    any window whose chart will be presented as "reale" (e.g. the hero
    slide), so a forward-filled value can never get drawn as if it were an
    actual market/TCGCSV print. Callers that only explore data (e.g. the
    demo notebook scrubbing through origins) should leave it False and
    instead use `observed_mask`/`history_observed` to render imputed points
    distinctly (see plots.plot_forecast_slice).
    """
    fc = preds[(preds["series"] == series) & (preds["origin_index"] == origin_index)]
    if fc.empty:
        raise ValueError(f"no forecast rows for series={series!r}, origin_index={origin_index}")
    fc = fc.sort_values("horizon_step")

    observed_mask = fc["observed"].to_numpy(dtype=bool)
    if require_observed_targets and not observed_mask.all():
        unobserved = fc.loc[~observed_mask, "target_index"].astype(int).tolist()
        raise ValueError(
            f"series={series!r}, origin_index={origin_index}: target index(es) "
            f"{unobserved} are forward-filled, not observed — refusing to build a "
            "slice that would plot imputed values as real (require_observed_targets=True)"
        )

    origin_date = pd.Timestamp(fc["origin_date"].iloc[0])
    hist = truth[(truth["series"] == series) & (truth["index"] < origin_index)].sort_values("index")
    hist = hist.tail(history_days)

    actual = fc["actual"].to_numpy(dtype=float)
    forecast = fc["forecast"].to_numpy(dtype=float)
    q10 = fc["q10"].to_numpy(dtype=float)
    q90 = fc["q90"].to_numpy(dtype=float)
    naive = float(fc["baseline_naive"].iloc[0])
    naive_arr = np.full_like(actual, naive)
    naive_mae = mae(actual, naive_arr)
    model_mae = mae(actual, forecast)
    relative = model_mae / naive_mae if naive_mae > 1e-9 else float("nan")

    if glitches is None:
        glitches = find_glitches(truth)
    window_indices = set(fc["target_index"].astype(int).tolist()) | set(
        hist["index"].astype(int).tolist()
    )
    series_glitch_indices = set(glitches.loc[glitches["series"] == series, "index"])
    contains_glitch = bool(window_indices & series_glitch_indices)

    return ForecastSlice(
        series=series,
        origin_index=int(origin_index),
        origin_date=origin_date,
        history_dates=hist["date"].to_numpy(),
        history_values=hist["value"].to_numpy(dtype=float),
        target_dates=fc["target_date"].to_numpy(),
        actual=actual,
        forecast=forecast,
        q10=q10,
        q90=q90,
        naive=naive,
        coverage=coverage(actual, q10, q90),
        relative_mae=relative,
        contains_glitch=contains_glitch,
        observed_mask=observed_mask,
        history_observed=hist["observed"].to_numpy(dtype=bool),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_figdata.py -v`
Expected: FAIL only on tests outside this file that construct `ForecastSlice` directly
(`tests/test_plots_smoke.py`'s `_forecast_slice()` fixture) — fix that in Task 9, Step 1.
Everything in `tests/test_figdata.py` itself must PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tfm3lab/figdata.py tests/test_figdata.py
git commit -m "feat: expose observed_mask/history_observed on ForecastSlice"
```

---

## Task 9: `plot_forecast_slice` visually distinguishes observed vs. imputed

**Files:**
- Modify: `src/tfm3lab/plots.py:49-99` (`plot_forecast_slice`)
- Modify: `scripts/06_make_figures.py:37-44` (`build_hero_slice`)
- Test: `tests/test_plots_smoke.py`

**Interfaces:**
- Consumes: `ForecastSlice.observed_mask`/`.history_observed` (Task 8).
- Produces: `plot_forecast_slice` unchanged signature; imputed points now drawn as hollow
  markers over the base line, with a "imputato (forward-fill)" legend entry when any exist.
  `scripts/06_make_figures.py`'s hero-slice builder now passes
  `require_observed_targets=True`.

- [ ] **Step 1: Fix the `_forecast_slice()` fixture and write the failing test**

In `tests/test_plots_smoke.py`, update `_forecast_slice()` (lines 25-41) to accept an
`all_observed: bool = True` parameter and add the two new fields:

```python
def _forecast_slice(all_observed: bool = True):
    return figdata.ForecastSlice(
        series="A",
        origin_index=10,
        origin_date=pd.Timestamp("2024-01-11"),
        history_dates=pd.date_range("2024-01-01", periods=10).to_numpy(),
        history_values=[float(v) for v in range(100, 110)],
        target_dates=pd.date_range("2024-01-11", periods=5).to_numpy(),
        actual=[110.0, 108.0, 112.0, 105.0, 120.0],
        forecast=[109.0, 109.0, 109.0, 109.0, 109.0],
        q10=[104.0] * 5,
        q90=[114.0] * 5,
        naive=109.0,
        coverage=0.6,
        relative_mae=1.1,
        contains_glitch=False,
        observed_mask=np.array([True, True, all_observed, True, True]),
        history_observed=np.array([True] * 9 + [all_observed]),
    )
```

Add `import numpy as np` near the top of the file (after the existing `import pandas as pd`
line), and add this test after `test_plot_forecast_slice_reveal_false_omits_target_lines`:

```python
def test_plot_forecast_slice_marks_imputed_points():
    ax_clean = plots.plot_forecast_slice(_forecast_slice(all_observed=True), reveal=True)
    ax_imputed = plots.plot_forecast_slice(_forecast_slice(all_observed=False), reveal=True)
    # the imputed case draws two extra scatter collections (one in history, one in the
    # revealed target) that the fully-observed case doesn't.
    assert len(ax_imputed.collections) > len(ax_clean.collections)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plots_smoke.py -v`
Expected: FAIL — `ForecastSlice` construction in the fixture now works (Task 8 added the
fields), but `plot_forecast_slice` doesn't yet draw anything different for imputed points,
so `test_plot_forecast_slice_marks_imputed_points` fails
(`len(ax_imputed.collections) == len(ax_clean.collections)`).

- [ ] **Step 3: Implement in `plots.py`**

Replace `plot_forecast_slice` in `src/tfm3lab/plots.py` (lines 49-99) with:

```python
def plot_forecast_slice(
    sl, ax=None, *, reveal: bool = True, show_naive: bool = True, show_band: bool = True
):
    """The hero chart: history up to the cut, then (if `reveal`) the real
    continuation against the model's forecast. `reveal=False` still fixes
    the y-limits to the revealed state, so a live demo's two cells (cut,
    then reveal) don't jump the axes between them.

    Forward-filled points (`sl.history_observed`/`sl.observed_mask` False)
    are overlaid as hollow markers on top of the solid line — never drawn as
    if they were a real print, but also never dropped from the line, so the
    chart doesn't develop a fake gap.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(
        sl.history_dates, sl.history_values, color=PALETTE["actual"], linewidth=1.8,
        label="storico reale",
    )
    history_observed = np.asarray(sl.history_observed, dtype=bool)
    history_dates = np.asarray(sl.history_dates)
    history_values = np.asarray(sl.history_values, dtype=float)
    if not history_observed.all():
        imputed = ~history_observed
        ax.scatter(
            history_dates[imputed], history_values[imputed],
            facecolors="none", edgecolors=PALETTE["baseline"], marker="o", s=30, zorder=3,
            label="imputato (forward-fill)",
        )
    ax.axvline(
        sl.origin_date, color=PALETTE["alert"], linestyle="--", linewidth=1.2, label="taglio"
    )

    all_values = list(sl.history_values)
    if reveal:
        ax.plot(sl.target_dates, sl.actual, color=PALETTE["actual"], linewidth=1.8)
        observed_mask = np.asarray(sl.observed_mask, dtype=bool)
        target_dates = np.asarray(sl.target_dates)
        actual_values = np.asarray(sl.actual, dtype=float)
        if not observed_mask.all():
            imputed = ~observed_mask
            ax.scatter(
                target_dates[imputed], actual_values[imputed],
                facecolors="none", edgecolors=PALETTE["baseline"], marker="o", s=30, zorder=3,
            )
        ax.plot(
            sl.target_dates, sl.forecast, color=PALETTE["model"], marker="o", markersize=4,
            label="mediana TimesFM-3",
        )
        if show_band:
            ax.fill_between(
                sl.target_dates, sl.q10, sl.q90, color=PALETTE["model"], alpha=0.15,
                label="P10-P90",
            )
        if show_naive:
            ax.hlines(
                sl.naive, sl.target_dates.min(), sl.target_dates.max(),
                color=PALETTE["baseline"], linestyle=":", linewidth=1.5,
                label="naive (ultimo prezzo)",
            )
        all_values += list(sl.actual) + list(sl.forecast) + list(sl.q10) + list(sl.q90)
    else:
        all_values += [sl.naive]

    pad = 0.08 * (max(all_values) - min(all_values) + 1e-9)
    ax.set_ylim(min(all_values) - pad, max(all_values) + pad)
    ax.set_title(f"{sl.series} — taglio {pd.Timestamp(sl.origin_date).date()}")
    ax.legend(loc="upper left", fontsize=9)
    if reveal:
        ax.annotate(
            f"copertura P10-P90: {sl.coverage:.2f}   relative MAE: {sl.relative_mae:.3f}",
            xy=(0.02, 0.02), xycoords="axes fraction", fontsize=9, color="#374151",
        )
    return ax
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plots_smoke.py -v`
Expected: PASS (all tests in the file, including the pre-existing reveal-true/reveal-false
line-count assertions — both use `all_observed=True` by default, so no scatter is drawn and
line counts are unchanged).

- [ ] **Step 5: Update `scripts/06_make_figures.py`'s hero-slice builder**

In `build_hero_slice` (lines 37-44), change:
```python
    sl = figdata.build_forecast_slice(preds, truth, "The One Ring [LTR]", origin_index=238)
```
to:
```python
    sl = figdata.build_forecast_slice(
        preds, truth, "The One Ring [LTR]", origin_index=238, require_observed_targets=True
    )
```

- [ ] **Step 6: Run the full test suite and lint**

Run: `uv run ruff check src tests scripts && uv run pytest`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/tfm3lab/plots.py scripts/06_make_figures.py tests/test_plots_smoke.py
git commit -m "feat: distinguish observed vs. imputed points in the hero forecast chart"
```

---

## Task 10: Update `docs/talk-outline.md`'s C4 section

**Files:**
- Modify: `docs/talk-outline.md:172-175`

**Interfaces:** none — documentation only.

- [ ] **Step 1: Edit the C4 description**

Replace the C4 bullet (lines 172-175):

```markdown
**C4 — `exp_mtg_pit_histogram` (slide Esperimento C).** Tre pannelli, h=1/7/28. Bin agli
stessi 9 livelli dei quantili — i bin estremi (`≤ q10`, `≥ q90`) sono conteggi corretti di
"oltre quel quantile", non la vera forma della coda (`pit_values` taglia, non estrapola).
A h=28 il 63.7% della massa è nei due bin estremi contro il 25% atteso.
```

with:

```markdown
**C4 — `exp_mtg_quantile_bin_calibration`, ex `exp_mtg_pit_histogram` (slide Esperimento
C).** Tre pannelli, h=1/7/28. **Rinominato**: la versione precedente costruiva 8 bin da un
istogramma dei valori PIT interpolati (`np.histogram` su 9 confini di quantile produce 8
bin, non 9) ed etichettava i due bin estremi come `≤ q10`/`≥ q90` quando in realtà erano
`[0.1, 0.2)`/`[0.8, 0.9]` — un bug di etichettatura, non solo di nome (`figdata.py`). La
versione corretta conta direttamente contro i 9 quantili previsti per riga, 10 bin
espliciti (`actual <= q10`, `(q10, q20]`, ..., `(q80, q90]`, `actual > q90`), ciascuno con
probabilità nominale 10%. La coda vera resta comunque non estrapolata (non c'è modo di
sapere quanto oltre q90 sia finito un valore, solo che lo è). Quota di massa nei due bin
estremi a h=28: `[NUMERO]` (nominale atteso 20%) — va ricalcolata con la funzione corretta,
il numero precedente (63.7% vs 25% atteso) si riferisce allo schema a 8 bin ormai rimosso.
```

- [ ] **Step 2: Verify no other reference to the old name/number survives**

Run: `grep -n "exp_mtg_pit_histogram\|63.7%" docs/talk-outline.md`
Expected: no matches.

- [ ] **Step 3: Commit**

```bash
git add docs/talk-outline.md
git commit -m "docs: rename PIT histogram references, flag stale C4 number for re-run"
```

---

## Task 11: Full verification pass

**Files:** none (verification only).

- [ ] **Step 1: Lint**

Run: `uv run ruff check src tests scripts`
Expected: `All checks passed!` (or equivalent clean output).

- [ ] **Step 2: Full test suite (default — offline, no GPU)**

Run: `uv run pytest -v`
Expected: all tests pass; the two opt-in files
(`tests/test_model_smoke.py`, `tests/test_model_ts_id_invariance.py`) report `skipped`,
not `error` or `failed`.

- [ ] **Step 3: Grep for any remaining stale references**

Run: `grep -rn "pit_histogram\|bin_left\|bin_right" src scripts tests docs/talk-outline.md`
Expected: only the intentional deprecated-alias lines in `figdata.py`/`plots.py` and the
Task 10 doc edit — no other file still assumes the old 8-bin schema.

- [ ] **Step 4: Write the completion report**

Post a short report (not a new file unless the user asks) covering: files changed, exact
test/lint commands run and their pass/fail result, residual risks (e.g. "hero slide's
origin_index=238 has never actually been re-verified against `require_observed_targets=True`
with real MTG data — only synthetic tests cover the new raise path"; "the quantile-bin
calibration numbers in `docs/talk-outline.md` are placeholders until `scripts/06_make_figures.py`
is re-run against real `results/*.parquet`"), and what migration existing results need
(none — `results/*.parquet` schemas are unchanged; only the *figure-generation* code
changed, so `uv run scripts/06_make_figures.py` needs a re-run to refresh
`exp_mtg_quantile_bin_calibration.png` and the hero slice, nothing upstream).

---
