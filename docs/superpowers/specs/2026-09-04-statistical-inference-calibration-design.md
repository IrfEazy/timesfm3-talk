# Statistical Inference & Probabilistic Calibration — Design

## Problem

The MTG forecasting evaluation (backtest.py, summarize.py, metrics.py) already
computes point-accuracy leaderboards and a single-number calibration summary
(pinball/coverage/PIT). It is missing the inferential machinery a
preregistered claim needs:

- No uncertainty on the accuracy delta itself — "model beats naive" is a
  point estimate with no CI, so `docs/analysis-plan.md`'s Claim rule ("migliora
  solo se CI del delta e' coerente e preregistrata") has nothing to point at.
- Diebold-Mariano p-values are computed per (series, horizon_step) group with
  no correction for running many such tests at once (multiple comparisons
  across cards x horizons) — and are currently the only significance signal
  available, encouraging p-value-as-headline reporting.
- The calibration curve (`scripts/04_exp_calibration.py`) reports empirical
  coverage with no uncertainty band, no normalized interval width, and no
  weighted interval score — so "coverage is 0.57 instead of 0.80" carries no
  sense of how far outside sampling noise that gap sits.
- No conformal post-processing exists — there is no way to ask "if we
  calibrate the intervals online from the model's own past errors, does
  coverage actually improve, causally, without leaking the future."

This design adds that machinery as additive modules/columns on top of the
existing raw-predictions parquet files, without touching `windows.py`'s
origin convention, `backtest.py`'s row schema, or any existing result file.

## Scope

In scope:
1. Paired moving-block bootstrap for the model-vs-baseline error delta, with
   panel (multi-series) support — `bootstrap.py` (new).
2. Benjamini-Hochberg correction of Diebold-Mariano p-values across
   card x horizon combinations within one comparison family, plus an effect
   size + CI headline — `summarize.py` additions.
3. Calibration upgrades: nominal-vs-empirical curve with binomial CI,
   normalized interval width, weighted interval score (WIS), quantile-bin
   calibration with per-bin CI — `metrics.py` additions,
   `scripts/04_exp_calibration.py` extension.
4. Causally-valid conformal post-processing (split-conformal / CQR-style),
   strictly from past-origin errors, raw and conformalized results kept
   separate — `conformal.py` (new).
5. New parquet artifacts + figures, all with `run_id` and a manifest
   reference — `scripts/08_exp_inference.py` (new).

Out of scope: re-running the model (everything here is post-hoc on already
written `exp_mtg_raw_predictions.parquet`); changing `windows.py`'s origin
convention; touching existing result files or slide numbers; a fully generic
multi-alpha conformal framework (the project's quantile grid and existing
"P10-P90 coverage" convention fix alpha=0.2 / the q10-q90 pair).

## Architecture

```
exp_mtg_raw_predictions.parquet (existing, untouched)
        |
        +--> bootstrap.py --------------------+
        |    paired_moving_block_bootstrap     |
        |    panel_paired_block_bootstrap       > summarize.py
        |                                       |  summarize_significance
        +--> metrics.py (additive) -------------+  apply_bh_correction
        |    binomial_ci
        |    weighted_interval_score
        |    interval_width_normalized
        |
        +--> conformal.py (new, standalone)
             conformalize_intervals
             evaluate_conformal_coverage

scripts/04_exp_calibration.py (extended)  -> exp_calibration_curve.parquet (+ci)
                                              exp_calibration_summary.parquet (+wis, +normalized width)

scripts/08_exp_inference.py (new)         -> exp_significance.parquet
                                              exp_significance_panel.parquet
                                              exp_conformal_predictions.parquet
                                              exp_conformal_coverage.parquet
                                              results/figures/*.png (new)
                                              results/manifests/inference-<run_id>.json
```

Nothing here changes `backtest.py`'s row schema or `windows.py`'s origin
semantics. Every new script/function reads already-written parquet files,
matching the existing `04`/`06` post-hoc pattern.

## Components

### `bootstrap.py` (new)

```python
@dataclass(frozen=True)
class BootstrapResult:
    delta_mean: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_boot: int
    block_size: int
    seed: int
    n_origins: int
```

- `paired_moving_block_bootstrap(model_abs_err, baseline_abs_err, horizon, block_size=None, n_boot=1000, ci=0.9, seed=config.SEED) -> BootstrapResult`
  - `delta = baseline_abs_err - model_abs_err`, computed once, per origin,
    BEFORE any resampling — pairing (which delta belongs to which origin) is
    fixed and never broken by resampling model/baseline independently.
  - `block_size` defaults to `horizon`; raises `ValueError` if
    `block_size < horizon` (enforces "blocchi almeno pari all'horizon" as a
    checked invariant, not just a docstring promise).
  - Moving-block resampling: same construction as the existing
    `metrics.block_bootstrap_ci` (contiguous overlapping blocks, `n_blocks =
    ceil(n/block_size)`, concatenate and truncate to `n`), applied to the
    delta array.
  - `seed` always has a value (defaults to `config.SEED`) and is always
    echoed back on the result — no unseeded, unreproducible runs.

- `panel_paired_block_bootstrap(deltas: dict[str, np.ndarray], horizon, block_size=None, n_boot=1000, ci=0.9, seed=config.SEED, weights: dict[str, int] | None = None) -> BootstrapResult`
  - Requires every series' delta array to be the same length, in the same
    origin order (the caller aligns this via `benchmark.common_origin_set`
    upstream — not re-derived here).
  - Per bootstrap replicate: draw ONE set of block-start positions, apply it
    identically to every series' delta array (preserves cross-sectional
    correlation at a shared origin, not just each series' own
    autocorrelation), then combine the per-series replicate means into one
    panel-level number via a weighted average (`weights` defaults to each
    series' `n_origins`, same convention as `summarize.aggregate_leaderboard`).
    Percentile CI is taken on that combined replicate statistic.
  - `n_origins` on the result is the shared origin count (identical for
    every series by construction).

### `metrics.py` (additive, no changes to existing functions)

- `binomial_ci(successes: int, n: int, ci: float = 0.9) -> tuple[float, float]`
  — exact Clopper-Pearson interval via `scipy.stats.beta.ppf` (scipy is
  already a dependency; no new one added). `n == 0` returns `(nan, nan)`.
- `weighted_interval_score(actual, quantile_forecasts, levels) -> float` — the
  standard Bracher et al. (2021) WIS, built from the median (`q50`, weight
  0.5) plus the 4 nested pairs available in the project's 9-level grid
  (q10/q90 alpha=0.2 w=0.1, q20/q80 alpha=0.4 w=0.2, q30/q70 alpha=0.6 w=0.3,
  q40/q60 alpha=0.8 w=0.4), normalized by `1/(K+0.5)` with `K=4`. Documented
  as "a WIS variant" — this project's requirement explicitly allows a
  documented variant rather than mandating one exact library's formula.
- `interval_width_normalized(lower, upper, scale) -> float` — mean
  `(upper - lower)` divided by `scale`. `scale` is the SAME per-series
  in-sample scale already computed by `compute_mase_scales`/
  `in_sample_scale` for MASE — reusing it keeps "normalized" consistent with
  the rest of the project's normalization instead of introducing a second,
  unrelated denominator.

### `summarize.py` (additive)

- `apply_bh_correction(df: pd.DataFrame, pvalue_col: str = "dm_pvalue", family_cols: tuple[str, ...] = ()) -> pd.DataFrame`
  — one "family" is one logical comparison whose card x horizon_step
  p-values get pooled and corrected together. `family_cols=()` (the default)
  treats the WHOLE input `df` as a single family — correct for the intended
  call site, `summarize_significance`, which is invoked once per
  (mode, transform, baseline_col) combination by the caller
  (`scripts/08_exp_inference.py`), so the family boundary is already fixed
  by what rows are IN the df, not by a column value to group on. Pass a
  non-empty `family_cols` only if a future caller stacks multiple
  comparisons into one df and needs `apply_bh_correction` to split them
  itself. Runs `statsmodels.stats.multitest.multipletests(pvals,
  method="fdr_bh")` over the family's non-NaN p-values; NaN p-values
  (already excluded upstream by `MIN_OBSERVATIONS_FOR_DM_TEST`) are excluded
  from the correction and reinserted as NaN at their original row, never
  coerced to 0 or 1 and never silently dropped. Adds one column,
  `dm_qvalue_bh`.
- `summarize_significance(df, mase_scales, group_cols=("mode", "series", "horizon_step"), baseline_col="baseline_naive", n_boot=1000, ci=0.9, seed=config.SEED) -> pd.DataFrame`
  — one row per group: `n`, `dm_stat`/`dm_pvalue` (existing DM logic, reused
  not duplicated), `delta_mean`/`ci_low`/`ci_high`/`block_size`/`seed` from
  `bootstrap.paired_moving_block_bootstrap` (block_size = that group's
  horizon_step), then a second pass over the full returned table calls
  `apply_bh_correction(result)` (default `family_cols=()`, since one
  `summarize_significance` call already covers exactly one comparison
  family — the input `df` was pre-filtered to one mode/transform/baseline
  before this function was called). Docstring states the reporting rule
  explicitly:
  effect size (`delta_mean`/`ci_low`/`ci_high`, or `skill = 1 -
  relative_mae` for the same group) is the headline; `dm_pvalue`/
  `dm_qvalue_bh` are supplementary, never the primary claim — this is the
  rule `docs/analysis-plan.md` will point to.

### `conformal.py` (new)

- `conformalize_intervals(df: pd.DataFrame, alpha: float = 0.2, min_calibration_origins: int = 20, group_cols: tuple[str, ...] = ("series", "horizon_step")) -> pd.DataFrame`
  — returns a COPY of `df` with `q10_conformal`, `q90_conformal`,
  `conformal_score_threshold`, `conformal_calibration_n`, `conformalized`
  (bool) columns added; raw `q10`/`q90` untouched on every row.
  Per `group_cols` group, sorted by `origin_index`: for row at position `i`,
  the calibration set is every STRICTLY EARLIER row in the same group
  (`origin_index` less than the current row's) with `observed == True`
  (never calibrate against a forward-filled target). Nonconformity score for
  a calibration row = `max(q10 - actual, actual - q90)` (CQR score). If
  fewer than `min_calibration_origins` such rows exist, the current row gets
  `conformalized = False` and its `q10_conformal`/`q90_conformal` equal the
  raw values unchanged. Otherwise `conformal_score_threshold` is the
  finite-sample-corrected empirical quantile of the calibration scores at
  level `ceil((n+1)*(1-alpha))/n` (split-conformal convention), and
  `q10_conformal = q10 - threshold`, `q90_conformal = q90 + threshold` — the
  threshold is NOT clamped at 0 (standard CQR allows a negative threshold,
  i.e. a valid interval shrink, and clamping would break the coverage
  guarantee the method is there to provide).
- `evaluate_conformal_coverage(df: pd.DataFrame, group_cols=("series", "horizon_step")) -> pd.DataFrame`
  — per group, restricted to `observed == True` rows: coverage and mean
  width for the RAW interval vs. the CONFORMAL interval, computed only over
  rows where `conformalized == True` (an apples-to-apples subset — comparing
  raw-everywhere against conformal-only-where-calibrated would bias the
  comparison). `n_skipped_insufficient_calibration` is reported alongside,
  never silently dropped from the picture.
- Module docstring states plainly: this output is an online post-hoc
  wrapper around the zero-shot forecasts, calibrated from the model's own
  past errors — it is NOT a zero-shot result. Report raw and conformalized
  results side by side, never blended into one number, and never claim
  "zero-shot" for the conformalized columns.

### `scripts/04_exp_calibration.py` (extended, additive columns only)

- `calibration_curve()` gains `ci_low`/`ci_high` per (regime, level) via
  `metrics.binomial_ci(round(empirical * n), n)`.
- The `summarize_calibration` call gains a `mase_scales` argument (computed
  the same way `02_exp_mtg.py` already computes MASE scales) so its output
  table gains `interval_width_normalized` and `wis` columns; existing
  columns (`pinball_avg`, `coverage_p10_p90`, `pit_mean`) are unchanged.
- `figdata.quantile_bin_calibration` gains per-bin `ci_low`/`ci_high` via the
  same `binomial_ci` helper — the bin logic itself (already corrected in a
  prior session) is untouched.

### `scripts/08_exp_inference.py` (new)

CLI script, offline-safe (reads existing parquet, no model/network call):
1. Loads `exp_mtg_raw_predictions.parquet`.
2. Computes MASE scales, calls `summarize_significance` per
   (mode, transform) combination present -> `exp_significance.parquet`.
3. For each (mode, transform), builds per-series deltas aligned on the
   showcase panel's common origins and calls `panel_paired_block_bootstrap`
   -> `exp_significance_panel.parquet`.
4. Calls `conformalize_intervals` + `evaluate_conformal_coverage` on the
   univariate/identity slice (the project's demo combination) ->
   `exp_conformal_predictions.parquet`, `exp_conformal_coverage.parquet`.
5. Renders 3 new figures via `plots.py` additions (calibration curve with CI
   band, bootstrap delta forest plot, conformal coverage before/after bar)
   into `results/figures/`.
6. Writes one `results/manifests/inference-<run_id>.json` via
   `manifest.write_manifest`, `run_id` generated the same way as
   `scripts/02b_exp_mtg_benchmark.py`
   (`f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"`), and adds
   a `run_id` column to every parquet it writes.

Everything in this script is CLI-only, dry-run-free by construction (it does
no model inference at all — pure post-hoc statistics on an already-written
file) and requires no opt-in flag, unlike the GPU-bound scripts.

## Data flow

```
exp_mtg_raw_predictions.parquet
    -> (per mode, transform) group
    -> summarize_significance(group, mase_scales)
         -> DM stat/p per (series, horizon_step)
         -> bootstrap CI per (series, horizon_step)
         -> apply_bh_correction across the (series, horizon_step) family
       -> exp_significance.parquet row
    -> panel_paired_block_bootstrap(deltas aligned on common origins)
       -> exp_significance_panel.parquet row
    -> conformalize_intervals(univariate/identity slice)
       -> exp_conformal_predictions.parquet (raw + conformal columns)
       -> evaluate_conformal_coverage(...)
          -> exp_conformal_coverage.parquet
```

## Error handling

- `paired_moving_block_bootstrap`/`panel_paired_block_bootstrap`: raise
  `ValueError` on `block_size < horizon`, on mismatched array lengths
  (panel), or on `n_boot < 1`. No silent truncation.
- `apply_bh_correction`: a family with zero non-NaN p-values is a no-op for
  that family (all-NaN output), not an exception — matches
  `summarize_accuracy`'s existing "NaN, not an invented number" policy for
  underpowered groups.
- `conformalize_intervals`: `min_calibration_origins` not met -> raw values
  passed through with `conformalized=False`, never an exception (this is the
  expected, common case for the first N origins of every series) and never a
  silently-adjusted interval computed from too little history.
- `evaluate_conformal_coverage` on a group with zero `conformalized=True`
  rows: reports `nan` coverage/width for that group plus the skipped count,
  not a crash.

## Testing strategy

All tests are local-fixture, no network/model, matching the rest of the
suite.

- `tests/test_bootstrap.py`: hand-computed delta on a small synthetic
  series; block_size < horizon raises; seed reproducibility (same seed same
  CI, different seed different CI with overwhelming probability on a fixed
  fixture); panel bootstrap requires equal-length arrays; panel CI differs
  from a naive unweighted average of per-series CIs (proof the joint
  resampling + weighting actually does something, mirroring
  `aggregate_leaderboard`'s existing weighted-vs-naive proof); an
  autocorrelated synthetic sequence produces a WIDER block-bootstrap CI than
  an i.i.d. bootstrap on the same values (direct evidence block resampling
  matters, using `metrics.block_bootstrap_ci` as the i.i.d. comparison).
- `tests/test_metrics.py` (extended): `binomial_ci` against known
  Clopper-Pearson reference values; `weighted_interval_score` on a
  perfect-quantile synthetic case (WIS ≈ 0) and an undercovered case (WIS
  clearly larger); `interval_width_normalized` hand-computed.
- `tests/test_summarize.py` (extended): `apply_bh_correction` on a
  hand-built p-value array reproduces `statsmodels`' own `fdr_bh` output;
  NaN p-values pass through untouched at their original positions;
  `summarize_significance` end-to-end on a small synthetic DataFrame.
- `tests/test_conformal.py` (new): perfect-quantile synthetic (adjustment ≈
  0, coverage already at nominal); deliberately undercovered synthetic
  (raw coverage < nominal, conformalized coverage measurably closer to
  nominal); leakage guard — a calibration row's score must never use a row
  with `origin_index >=` the current row, verified directly by checking the
  set of origins actually used; below-minimum-history rows stay raw and
  `conformalized=False`.
- `tests/test_exp_inference_cli.py` (new): argparse-only + a tiny synthetic
  raw-predictions parquet fixture proving the script writes all 4 parquet
  files with `run_id` populated and a manifest file — no real model call.

## Migration notes

No existing result file is modified. `exp_mtg_raw_predictions.parquet`,
`exp_mtg_accuracy.parquet`, `exp_mtg_calibration.parquet`,
`exp_calibration_curve.parquet`, `exp_calibration_summary.parquet` all gain
new OPTIONAL columns/rows only where this design says so (`04`'s two output
files gain columns; nothing removes or renames an existing column). The new
`exp_significance*.parquet`/`exp_conformal_*.parquet` files and
`scripts/08_exp_inference.py` do not exist until this script is actually run
— that run is a real GPU-free, network-free CLI execution this branch DOES
perform (unlike the `02b` benchmark grid, which needs a GPU and was
deliberately left unexecuted), since it only reads already-committed parquet
files.
