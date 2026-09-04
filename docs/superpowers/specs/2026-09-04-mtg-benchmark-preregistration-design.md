# MTG Benchmark Configurability & Preregistered Evaluation — Design

## Problem

Experiment A (`scripts/02_exp_mtg.py`) currently runs one fixed configuration:
context_len=64, max_horizon=28, the 7 showcase cards, naive-only headline comparison.
That is enough for the talk's narrative slide, but not enough to make the MTG
experiment a *preregistered primary evaluation*: it has no declared grid across
context/horizon, no way to test more than 7 cards without hand-editing code, no
leaderboard against the other three baselines already implemented (`baselines.py`
computes seasonal_naive/drift/ets but only naive ever reaches a headline number),
and no ablation of `make_positive` or "real vs. placebo" multivariate panels.

This work makes the MTG experiment configurable and reproducible as a
preregistered evaluation, **without touching**:
- the committed `results/exp_mtg_*.parquet` files or any number already typed
  into `docs/talk-outline.md` / `slides/` (both derive from `02_exp_mtg.py`'s
  current fixed-config run — untouched),
- `windows.py`'s origin convention,
- any GPU/live-model run — this branch adds CLI + dry-run + tests + docs only.

## Scope

**In scope:**
1. Declarative benchmark config (JSON) — context lengths, horizons, origin
   stride, a common origin set across the whole grid.
2. Card manifest separation: showcase (7, existing `DEFAULT_CARDS`, untouched)
   vs. an optional wider benchmark manifest — format + documented selection
   criteria, no invented 30+ card list.
3. Multi-baseline leaderboard (naive/seasonal_naive/drift/ets), not naive-only.
4. Ablation: raw vs. log1p (exists), `make_positive` True/False (exists as a
   backtest param, not yet recorded per-row), univariate vs. multivariate
   (exists), multivariate real panel vs. multivariate placebo panel (new).
5. Results schema with the required columns.
6. Slide-readable metrics: relative MAE, skill = 1 − relative MAE, documented
   median + weighted-mean aggregation, per-card and aggregate.
7. TimesFM-2.5 adapter as a zero-shot historical baseline, using the
   already-installed `timesfm` distribution's bundled `TimesFM_2p5_200M_torch`
   (confirmed: no new/conflicting dependency, Apache-2.0, ungated).
8. `docs/analysis-plan.md` expansion: hypotheses, primary metric, primary
   horizons, exclusion rules.

**Out of scope (explicitly deferred, not invented):**
- Actually running the benchmark grid or the 2.5 adapter against real data/GPU.
- Populating a real 30+-card benchmark manifest (criteria documented, list left
  to a future, separately-reviewed data-collection step).
- Changing `scripts/02_exp_mtg.py`'s default invocation, output files, or any
  number in `docs/talk-outline.md`.
- Changing `windows.py`, `backtest.py`'s existing row schema, or any existing
  test's expected behavior.

## Architecture

```
configs/
  benchmark_preregistered.example.json   worked example of the full grid
  benchmark_cards.example.csv            wider-manifest FORMAT + 7-row example
                                          (explicitly labeled "not a real
                                          30+ selection"), criteria documented
                                          in its header comment

src/tfm3lab/
  benchmark_config.py
    BenchmarkConfig (frozen dataclass) + load_benchmark_config(path) -> BenchmarkConfig
    validates: config_id required, context_lengths/horizons non-empty,
    modes/transforms/baselines from known sets, placebo_panel_size >= 1

  benchmark.py
    common_origin_set(n, context_lengths, horizons, stride, max_origins) -> np.ndarray
    iter_ablation_combos(cfg) -> Iterator[AblationCombo]   (context_len, horizon,
        transform, make_positive, mode)
    select_placebo_panel(pool, panel_size, seed) -> tuple[CardSpec, ...]
    dry_run_report(cfg, series_list) -> dict   (combo count, origins/combo,
        estimated predict_batch calls — no forecaster call)

  model_2p5.py
    TimesFM2p5Adapter (implements the Forecaster protocol from model.py)
    load_forecaster_2p5(repo_id=DEFAULT_REPO_ID) -> TimesFM2p5Adapter

  data/mtg.py
    + load_card_manifest(path: Path) -> tuple[CardSpec, ...]   (CSV or JSON,
      same CardSpec shape as DEFAULT_CARDS)

  backtest.py
    + "make_positive" column added to _rows_for_one_series_forecast's output
      row (currently a parameter, never recorded) — additive column, existing
      consumers unaffected (they select columns by name, never by position)

  summarize.py
    + "skill_vs_baseline" column in summarize_accuracy (= 1 - relative_mae,
      additive)
    + summarize_leaderboard(df, mase_scales, group_cols, baseline_cols) ->
      pd.DataFrame   (one row per (..., baseline_method), multi-baseline)
    + aggregate_leaderboard(leaderboard_df, weight_col="n") -> pd.DataFrame
      (collapses `series`; emits relative_mae_median, relative_mae_mean_weighted,
      skill_median, skill_mean_weighted — weighting documented in docstring)

scripts/
  02b_exp_mtg_benchmark.py
    CLI: --config PATH (required), --dry-run, --cards {showcase,benchmark,PATH},
    --adapter {timesfm3,timesfm2.5}
    Writes (only in non-dry-run mode, never invoked by this branch):
      results/exp_mtg_benchmark_raw_predictions.parquet
      results/exp_mtg_benchmark_leaderboard.parquet
      results/exp_mtg_benchmark_leaderboard_aggregate.parquet
      results/manifests/benchmark-<config_id>-<run_id>.json
    Dry-run writes only a JSON report to stdout/--dry-run-out, no results/ write.

docs/analysis-plan.md   expanded (see "Data flow" below)
```

## Data flow

1. `load_benchmark_config(path)` → `BenchmarkConfig`. `config_id` comes from the
   file (required field) — never generated from content, so two configs can
   share field values but must be named distinctly by the author.
2. `run_id` is generated at run time (UTC timestamp + short random suffix,
   analogous to `manifest.py`'s existing `write_manifest` pattern) — identifies
   *this invocation*, not the declared grid.
3. Cards resolve via `--cards`: `showcase` → `DEFAULT_CARDS` (unchanged),
   `benchmark`/a path → `load_card_manifest`. `resolve_card_specs` (existing,
   untouched) turns either into concrete TCGCSV product IDs.
4. `common_origin_set` computes ONE origin array using
   `windows.valid_origins(n, context_len=max(context_lengths), horizon=max(horizons), max_origins=cfg.max_origins)`,
   then thins by `cfg.origin_stride` (`origins[::stride]`) — done outside
   `windows.py`, which stays exactly as it is. Every grid cell reuses this same
   array: a cell's own `(context_len, horizon)` is always `<=` the values used
   to build it, so every one of `common_origin_set`'s origins is automatically
   valid for every smaller cell too (same start/end constraint, monotonic in
   both parameters) — this is the "same origins for every config" requirement.
5. `iter_ablation_combos(cfg)` enumerates the full cartesian product of
   `context_lengths x horizons x transforms x make_positive x modes`, skipping
   `multivariate_placebo` combos when the resolved card pool is smaller than
   `placebo_panel_size` (printed, not silent).
6. Each combo calls the *existing* `run_univariate_backtest` /
   `run_multivariate_backtest` (`backtest.py`, unmodified call signature) with
   that combo's `context_len`, `horizon`, `transform`, `make_positive`; for
   `multivariate_placebo`, the card list passed in is `select_placebo_panel`'s
   output instead of the showcase/benchmark set.
7. Per-combo output DataFrames are concatenated, then projected onto the
   required schema: `run_id`, `config_id`, `context_len`, `requested_horizon`
   (the combo's horizon setting — distinct from `horizon_step`, which already
   exists per-row), `mode`, `transform`, `series`, `origin` (renamed from
   `backtest.py`'s `origin_index` for this new table only — `backtest.py`
   itself keeps `origin_index`), `target_date`, `observed`, plus the existing
   forecast/quantile/baseline columns and the newly-added `make_positive`.
8. `summarize_leaderboard` + `aggregate_leaderboard` run on that projected
   table, `observed=True` filtered first (existing project-wide rule).
9. `manifest.build_fetch_manifest`-style JSON (reusing `manifest.write_manifest`,
   unmodified) records: git SHA, `config_id`, `run_id`, card specs resolved,
   grid dimensions, adapter used, hardware/inference flags
   (`use_symmetric_averaging`, `make_positive` values tested), package versions.

## TimesFM-2.5 adapter

Confirmed via the installed environment (`uv run python -c "import timesfm;
timesfm.TimesFM_2p5_200M_torch"`): the single already-pinned `timesfm>=3.0.0`
PyPI distribution bundles TimesFM-2.5's own legacy-API implementation
(`timesfm.TimesFM_2p5_200M_torch`, default repo `google/timesfm-2.5-200m-pytorch`,
confirmed **Apache-2.0, ungated** — no HF login required, unlike the v3
checkpoint). No new dependency, no version conflict — requirement 7's "if not
feasible, document the blocker" branch does not apply; the adapter is built.

`TimesFM2p5Adapter.predict_batch(...)` wraps
`TimesFM_2p5_200M_torch.forecast(horizon, inputs: list[np.ndarray]) ->
(point, quantiles)` into the same `Forecaster` protocol shape `model.py`'s
`forecast_batch` already consumes (`ts_ids`-carrying outputs, order preserved
since the underlying call takes a plain ordered list, no id concept at that
layer). **Documented open risk, not resolved by this branch**: the exact
quantile grid TimesFM-2.5 returns has not been verified against a real
inference call (no GPU/network run here). The adapter does not assume it
matches `config.QUANTILE_LEVELS` — it reports whatever shape comes back and
raises via a reused `assert_quantile_shape`-style check only when a caller's
opt-in smoke test (mirroring `test_model_smoke.py`'s
`TFM3LAB_RUN_MODEL_SMOKE=1` pattern) actually exercises it.

## Error handling

- Unknown `config_id`-less config (missing required field) → `SystemExit` at
  load time with the exact missing field.
- `multivariate_placebo` combo with too-small a card pool → skipped, printed
  reason, counted in the dry-run report's "skipped" section (never silently
  dropped).
- ETS/seasonal_naive per-row absence → already handled by existing
  `_baseline_forecasts` (unchanged); `summarize_leaderboard` filters NaN rows
  per-baseline-column before computing that baseline's metrics, and records
  the row count actually used (`n`) so a smaller `n` for `ets` than for
  `naive` in the same group is visible, not hidden.
- 2.5 adapter's quantile-shape mismatch (if it ever occurs) → raises loudly in
  the opt-in smoke test, never silently truncates/pads.

## Testing strategy

Everything below runs offline, no network, no GPU, matching the project's
existing default-pytest contract.

- `tests/test_benchmark_config.py` — load/validate `BenchmarkConfig` from
  fixture JSON (valid + each required-field-missing case).
- `tests/test_benchmark.py` — `common_origin_set` (matches `valid_origins` at
  the max cell, stride thinning, monotonic-subset property against smaller
  cells), `iter_ablation_combos` (full cartesian product, placebo skip when
  pool too small), `select_placebo_panel` (deterministic under a fixed seed).
- `tests/test_mtg.py` — extend with `load_card_manifest` (CSV + JSON fixtures).
- `tests/test_backtest.py` — extend: `make_positive` column present and
  correct in output rows for both True/False.
- `tests/test_summarize.py` — extend: `skill_vs_baseline` column;
  `summarize_leaderboard` multi-baseline rows + NaN-row filtering per baseline;
  `aggregate_leaderboard` median vs. weighted-mean on a small constructed
  case where they provably differ (documents the weighting in a test, not
  just prose).
- `tests/test_model_2p5.py` — `TimesFM2p5Adapter` against a fake underlying
  2.5 model object (mirrors `test_model.py`'s fake-`Forecaster` pattern) —
  shape translation, ts_id/order preservation, no real checkpoint load.
- `tests/test_exp_mtg_benchmark_cli.py` — mirrors
  `tests/test_fetch_data_cli.py`'s `importlib.util.spec_from_file_location`
  pattern: `--dry-run` produces a report with the expected combo count against
  a tiny fixture config + fixture series, no forecaster loaded, no network.

## Migration notes

None required for existing artifacts — `02_exp_mtg.py`, its output files, and
every number in `docs/talk-outline.md` are untouched by this branch. The new
`exp_mtg_benchmark_*.parquet` files do not exist until someone actually runs
`02b_exp_mtg_benchmark.py` (a GPU-bearing environment, e.g. Colab) — not done
as part of this branch, per its explicit "no GPU run" constraint.
