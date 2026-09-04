# tfm3lab — TimesFM-3 put to the test

Experiments backing a talk on TimesFM-3 (Google Research's zero-shot time-series foundation
model, released Aug 2026). The question isn't "is TimesFM-3 good" — Google's own launch
material only reports average rank on GIFT-Eval/FEV-Bench/TIME, no task-level numbers, and
its pretraining corpus includes public web data up to late 2023. The question is whether it
generalizes on data it could not have seen, and how it behaves when a genuine shock — one
that is *not* in its training data by construction — hits.

See `docs/talk-outline.md` for the full narrative and `docs/talk-outline.md`'s "Domande
scomode" section for the hard questions this is built to survive.

## What's here

| Experiment | Script | Question |
|---|---|---|
| A — camera pulita | `scripts/02_exp_mtg.py` | Magic: The Gathering card prices, entirely post-cutoff. Does TimesFM-3 beat naive on a domain it almost certainly never saw? |
| B — shock pre/post-cutoff | `scripts/03_exp_shock.py` | **The core result.** Same protocol on market shocks inside vs. outside the pretraining window. |
| C — calibrazione | `scripts/04_exp_calibration.py` | Are the 9 quantiles honestly calibrated, in calm vs. shock regimes? (reuses A/B's cached predictions, no new model calls) |
| D — covariate | `scripts/05_exp_covariates.py` | Legitimate future-known covariates, plus a deliberate leakage demo. |

`scripts/02b_exp_mtg_benchmark.py` runs Experiment A as a declarative,
preregistered grid (context lengths x horizons x transform x make_positive x
univariate/multivariate/placebo-panel) instead of the single fixed config
above — see `docs/analysis-plan.md` and
`configs/benchmark_preregistered.example.json`. Its outputs
(`exp_mtg_benchmark_*.parquet`) are separate from `02_exp_mtg.py`'s and never
feed the numbers already committed in `docs/talk-outline.md` or the slides.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Never `uv pip install` — everything below goes
through `uv add`/`uv sync`/`uv run`, which keeps `pyproject.toml` and `uv.lock` as the single
source of truth.

```bash
git clone <this-repo> && cd timesfm3-talk

# Pick ONE torch build for this machine (they're mutually exclusive — see
# pyproject.toml's [tool.uv] conflicts block):
uv sync --extra cpu     # no GPU (this repo's own dev machine)
uv sync --extra cuda    # Colab / a CUDA box
```

### The gated checkpoint

TimesFM-3's weights (`google/timesfm-3.0-pytorch`) are gated on Hugging Face under
`timesfm-non-commercial-license-v1.0` — non-commercial, non-production use only (the *code*
is Apache-2.0; the *weights* are not). Before running anything that loads the model:

1. Accept the license at https://huggingface.co/google/timesfm-3.0-pytorch
2. Authenticate once: `uv run hf auth login`, or set `HF_TOKEN` in the environment.

Everything that only touches data (fetching, the probe script) works without this.

## Running it

Scripts are numbered in dependency order. Each one's own docstring has the full detail;
this is the short path.

```bash
# 0. Confirm TCGCSV's undocumented archive schema still matches what mtg.py assumes
uv run scripts/00_probe_tcgcsv.py

# 1. Fetch and cache everything (MTG full history is slow on first run —
#    use --mtg-start for a quick local sanity check)
uv run scripts/01_fetch_data.py --as-of <YYYY-MM-DD>
uv run scripts/01_fetch_data.py --as-of <YYYY-MM-DD> --mtg-start 2026-08-01   # quick check instead

# 2-5. The experiments (need the real model — see "hybrid execution" below)
uv run scripts/02_exp_mtg.py
uv run scripts/03_exp_shock.py
uv run scripts/04_exp_calibration.py   # reuses 02+03's output, no GPU needed
uv run scripts/05_exp_covariates.py

# 2b. Preregistered benchmark grid (optional, separate from 2-5 above)
uv run scripts/02b_exp_mtg_benchmark.py --config configs/benchmark_preregistered.example.json --dry-run
uv run scripts/02b_exp_mtg_benchmark.py --config configs/benchmark_preregistered.example.json --as-of <YYYY-MM-DD>

# tests (fast, offline, no GPU/network required by default)
uv run pytest
```

Every script prints what it wrote and where; results land in `results/*.parquet`
(committed to the repo on purpose — see "why results/ is committed" below).

### Opt-in tests that hit real services

```bash
TFM3LAB_RUN_LIVE_FETCH_SMOKE=1 uv run pytest tests/test_mtg_live.py tests/test_data_live.py
TFM3LAB_RUN_MODEL_SMOKE=1 uv run pytest tests/test_model_smoke.py   # needs the real checkpoint
```

## Hybrid execution: Colab for compute, local for everything else

This machine has no GPU. The design: **the same scripts** run unmodified in both places —

- **Colab** (`notebooks/run_on_colab.ipynb`): clones this repo, `uv sync --extra cuda`, runs
  the GPU-hungry steps (`01_fetch_data.py`'s MTG backfill, `02`/`03`/`05`).
- **Local**: everything else — code, tests, `04_exp_calibration.py` (pure re-analysis, no
  model calls), figures, slides, the demo notebook.

The boundary is `results/*.parquet`: heavy scripts write there, light scripts (`06`, `07`,
`notebooks/demo.ipynb`) only ever read from there. Point `TFM3LAB_DATA_ROOT` at a shared
location (e.g. a mounted Google Drive folder) if you want Colab and local runs to share the
same `data/cache/` instead of re-fetching.

### Why `results/` is committed

The demo notebook must work **offline, with no Colab and no network**, on talk day. Results
are small (parquet, not raw archives) — they're checked in so the repo alone is enough to
give the talk.

## Known limitations (say these out loud, don't bury them)

- **Pretraining contamination is not provable, only bounded.** `config.PRETRAIN_CUTOFF`
  (end of Nov 2023) is a conservative "could have been seen" line built from the two *dated*
  sources the model card discloses (Wikipedia Pageviews, Google Trends) — GIFT-Eval's own
  pretraining split is undated and undisclosed. Experiment B's pre/post-cutoff comparison is
  indirect evidence, not proof either way.
- **The automatic shock detector doesn't confirm every known event.** At z=4.0 on SP500 log
  returns, only 2 of the 5 `config.KNOWN_EVENTS` fire as single-day outliers (see
  `scripts/03_exp_shock.py`'s printed validation) — some were multi-day regime shifts or hit
  VIX harder than SP500. The backtest still anchors on the known date; the mismatch is
  reported, not hidden.
- **CPI YoY uses today's revised, seasonally-adjusted FRED series**, not the vintage a
  real-time forecaster would have seen in 2022. Softer evidence than the daily market series.
- **MTG's forward-fill.** Missing TCGCSV days are forward-filled up to 3 days;
  `SeriesData.observed` marks which points are real — every metric in `summarize.py` filters
  to `observed` rows only. Market/CPI series use intersection instead (no fill at all) since
  a missing day there almost always means "market closed," not "no snapshot."
- **`scripts/00_probe_tcgcsv.py` documents an undocumented API.** TCGCSV's daily-archive URL
  scheme was reverse-engineered by probing, not found in any official docs — it can change
  without notice. Run the probe before a real ingest; `fetch_mtgjson_fallback()` in
  `tfm3lab/data/mtg.py` is the fallback if it ever breaks.
- **The committed `results/exp_mtg_*.parquet` files have no manifest.** They predate this
  branch's `--as-of`/manifest provenance work and weren't regenerated by it (no live fetch was
  run as part of this change) — getting a manifest for the currently-cached MTG results
  requires an actual `uv run scripts/01_fetch_data.py --as-of ...` re-fetch. Also, a
  `--skip-mtg` run writes no manifest at all (the manifest/data-quality logic only runs inside
  the MTG-fetch branch of `scripts/01_fetch_data.py`), so a market/CPI-only run freezes
  `as_of` but records it nowhere.

## Project layout

```
src/tfm3lab/
  config.py       paths, seeds, PRETRAIN_CUTOFF, KNOWN_EVENTS
  windows.py       rolling-origin index semantics — the ONE place origin/target math happens
  metrics.py       MAE/MASE/pinball/coverage/PIT/Diebold-Mariano/block-bootstrap
  baselines.py     naive, seasonal naive, drift, ETS
  model.py         thin wrapper around timesfm3.TimesFM3Evaluator (injectable, unit-testable)
  backtest.py      the rolling-origin engine: univariate + multivariate, log1p ablation
  summarize.py     shared post-backtest aggregation used by all 4 experiment scripts
  manifest.py      JSON reproducibility manifest for fetch/model runs
  benchmark_config.py  declarative benchmark grid (context/horizon/ablation) schema + loader
  benchmark.py          shared origin set, ablation combo enumeration, placebo panel sampling
  model_2p5.py           TimesFM-2.5 zero-shot adapter (bundled dependency, no new package)
  data/
    mtg.py          TCGCSV ingestion (+ MTGJSON fallback)
    market.py       yfinance (SP500/VIX/Gold/Oil) + shock detection
    macro.py        FRED CPI YoY
scripts/          00-07 + 02b, numbered/lettered in dependency order (see "Running it")
configs/          declarative benchmark configs + card manifests (examples, no invented 30+ card selection)
tests/            fast + offline by default; opt-in live/model tests clearly marked
docs/talk-outline.md   the actual talk, with numbered placeholders for results not yet run
notebooks/        run_on_colab.ipynb (bootstrap), demo.ipynb (offline, for the live demo)
slides/           Marp source; scripts/07_build_slides.py injects numbers from results/
results/          committed — see "why results/ is committed" above
```

## Verification

```bash
uv run ruff check src tests scripts
uv run pytest                                      # 96 tests, offline, no GPU
uv run python scripts/00_probe_tcgcsv.py            # confirms the live TCGCSV schema
uv run python scripts/01_fetch_data.py --as-of <YYYY-MM-DD> --mtg-start <30 days ago>   # quick real-data check
```

Every number that ends up on a slide should be traceable to a file in `results/` — no number
gets typed by hand into `docs/talk-outline.md` or `slides/`.
