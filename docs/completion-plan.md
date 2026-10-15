# Completion plan — tfm3lab / timesfm3-talk

Status: **Phase 0 audit, approved on 2026-10-10** (the human adopted the PM proxy's suggestions). Nothing in `src/`, `scripts/`, `results/` or the
public docs has been changed. Audit base: `main` @ `3dfd604`; this file lives on branch
`feat/completion-plan`.

Test evidence at audit time: `uv sync --extra cpu && uv run pytest` gives **249 passed, 7 skipped**
(the opt-in live and model-smoke tests). The README's "96 tests" is stale.

---

## 1. Status table

Legend: **done** means the code and tests exist and pass offline. It does not mean a GPU run has
exercised the code. **partial** and **missing** are as written. Evidence uses `path:lines`.

### P0 items and Prompts 1–2 (methodology fixes, data contract)

| Review item | Status | Evidence |
|---|---|---|
| PIT → quantile-exceedance calibration | done | `figdata.py:393-448` `quantile_bin_calibration`; `06_make_figures.py:85-93`; `tests/test_figdata.py:335-390`. Leftovers: `summarize.py:137-159` still emits `pit_mean`, and `plots.py:243` keeps a deprecated alias |
| Demo never shows imputed points as real | done | `figdata.py:209-307` (`observed_mask`, `require_observed_targets`); `figdata.py:310-368` ranks windows on observed targets only |
| Forward-fill sensitivity (0/1/3 days) | **missing** | only `data/mtg.py:401` `max_ffill_days=3`. No script or test varies it |
| Output↔input association by `ts_id`, not by order | done | `backtest.py:189-287`, `model.py:136-168`; reversed-order fake test `tests/test_backtest.py:188-224` |
| `ts_id` invariance on the real model | partial | `tests/test_model_ts_id_invariance.py` is opt-in (`TFM3LAB_RUN_MODEL_SMOKE`) and has **never run** |
| MTG price-selection policy and subtype resolution | done | `data/mtg.py:297-335`, `88-92` |
| HTTP 404 vs 429/5xx handling, atomic download | done | `data/mtg.py:117-159` |
| `--as-of` required for fetch | done | `01_fetch_data.py:16-24, 82-96`; the library default is still `today` (`mtg.py:420`) |
| Per-observation provenance | partial | only aggregate counts (`IngestReport`); no per-row archive hash or date |
| Fetch manifest | done | `manifest.py:57-105`, `results/manifests/fetch-2026-09-04-…json` |
| `--skip-mtg` writes a manifest | **missing** | `01_fetch_data.py:105-143`: the manifest is written only inside `if not args.skip_mtg` |

### P1 items and Prompt 3 (benchmark)

| Review item | Status | Evidence |
|---|---|---|
| Shared origin set across the grid | done | `benchmark.py:19-40` |
| Declarative config plus `--dry-run` | done | `benchmark_config.py`, `benchmark.py:103-150`, tests |
| Multi-baseline leaderboard | done | `summarize.py:166`; naive, seasonal_naive, drift, ets |
| Median and n-weighted aggregation | done | `summarize.py:203-240` |
| Placebo panel, ablation grid | done | `benchmark.py:56-100` |
| TimesFM-2.5 adapter | done, **never run** | `model_2p5.py`, `tests/test_model_2p5.py`; univariate only (`02b:175-190`) |
| Long horizon in the default experiment | **missing** | `02_exp_mtg.py:77-78`: context 64, max h=28. Only `02b` covers h=56/64 |
| **30+ card pool** | **missing** | `configs/benchmark_cards.example.csv` holds the 7 showcase rows and says it is not a real selection |
| **02b fits in Kaggle RAM** | **missing** | `02b:250-279` keeps every combo in a list, then calls `pd.concat` and writes once. No streaming and no resume |
| **02b ever run on a GPU** | **no** | no `results/exp_mtg_benchmark_*` files |
| Decision rules in `analysis-plan.md` | partial | the metric, primary horizons and aggregation are defined. The claim rule ("CI del delta coerente") is **not operational**: no CI level, B, seed, BH α, primary context length or showcase-vs-pool rule. H1–H4 exist only in `openai.review.md:251-266` |

### Prompt 4: inference layer (spec and 10-task plan exist; 0 of 10 tasks implemented)

| Plan task | Status | Evidence |
|---|---|---|
| T1 paired / panel block bootstrap | **missing** | `metrics.py:175-205` `block_bootstrap_ci` handles one 1-D array with time blocks only. It is not paired, has no card clusters, and **nothing calls it** except its test |
| T2 binomial CI, WIS, normalized width | missing | grep: no hits |
| T3 Benjamini–Hochberg and significance table | missing | `07_build_slides.py:63-67` counts **unadjusted** p<0.05 |
| T4 WIS in `summarize_calibration` | missing | `summarize.py:137-163` |
| T5 conformal / CQR | missing | grep `conformal`: no hits in code |
| T6–T9 figdata CIs, 04 CI band, plots, `08_exp_inference.py` | missing | — |
| Diebold–Mariano | done | `metrics.py:131-172`, called at `summarize.py:128` |

### Prompts 5–8

| Prompt | Status | Evidence and gaps |
|---|---|---|
| 5 Covariates | partial | Done: a three-arm leakage control (`05:138-207`) and wiring tests. Missing: an availability contract (`available_at`/`future_known`), a shuffled or placebo arm (so the legit result **is not falsifiable**), a LEAKED watermark, and a test that keeps the leaked arm out of summaries. The docstring (`05:7-9`) promises announcement dates but the code uses `publishedOn`. The slides show the leak MAE without a "do not present" caveat |
| 6 Shock study as red-team appendix | partial | Done: per-event dots figure, relative MAE. Missing: rename to "error-recovery lag", a censoring flag (**`03:211` silently drops events that never recover**, which biases the mean toward easy events), the `market_calm` → `event_window_nonshock` rename, tests for `estimate_adaptation_lag`, and slide caveats. `README.md:18` says "**The core result.**" `slides/talk.md:57` and `talk-outline.md:76` say "Il cuore del talk" |
| 7 Latency and evidence ledger | missing | `model.py:146-159` times each `predict_batch`, but nothing persists or summarizes the timing. No GPU sync, warmup or batch sweep. No ledger. `talk.md.template` hard-codes numbers (330M, 15.360, 64, dates) |
| 8 Hostile final review | missing | — |

### Experiment B facts

- There are 5 hand-listed events (`config.py:91-95`): 3 pre-cutoff and 2 post-cutoff. Lag is measured on SP500 only. The detector (`market.py:84-101`, |z|≥4) only *validates* the dates and confirms 2 of 5.
- The lag is self-relative: the model's error is compared with its own pre-event median error. The multipliers 1.25/1.5/2.0 are shared across events, but the absolute thresholds differ about 7× (Covid 7.55 % vs tariffs 0.99 %, per `talk-outline.md`). The tariff window contains the 9 April tariff-pause rebound, which only `talk-outline.md:105-106` mentions.
- The `03` printout pools univariate and multivariate. The slides filter to multivariate (`07:88-96`), so the two headlines differ.
- There is no baseline-relative lag, no anticipation or memorization probe, no perturbation test and no event-selection rule.

### Experiment C facts

- Calibration is measured at **h=1 only** (`04:47`), with no CIs and no recalibration. The "calm" regime means non-shock days *inside the shock windows*, not a general calm sample.

### Reproducibility facts

- The checkpoint is **unpinned**: `config.py:64` sets `CHECKPOINT_ID` with no `revision` (the same holds in `model_2p5.py`). The manifest records the git SHA, `as_of` and package versions, but not the GPU, CUDA version or checkpoint revision.
- Scripts 02–05 write **no manifest**. Only `01` and `02b` do. None of the committed `results/exp_*` files has a manifest.
- No script fails fast on a missing GPU, `HF_TOKEN` or license acceptance.

### Kaggle and Colab notebooks

- `run_on_kaggle.ipynb` runs `00`, `01 --allow-live-end` (an unpinned end date, so not reproducible), then 02–05.
- It does not run `02b`, 2.5 or latency, and has no archive step. Colab has an archive step but the same gaps.

### Presentation

- There is no LICENSE, no `.github/workflows` and no badges. A ruff config already exists (`pyproject.toml:79-85`).
- The README cites no `results/` file for any number.
- `slides/talk.html` is a standalone Marp export (only `data:` URIs), so it is suitable for GitHub Pages once it is regenerated.

### Corrections to the lead's reading

- Prompts 1–3 are mostly done, with the gaps above (ffill sensitivity, the never-run invariance test, the long-horizon default, the operational claim rule).
- Prompt 4 is **less** done than "partly": only DM and an unused 1-D bootstrap exist.
- The manifest gap is wider than "results predate provenance": scripts 02–05 still don't write one.
- Everything else the lead listed checks out.

---

## 2. Findings the plan must handle (not in the original brief)

1. **The cutoff is asserted, not certified.** `PRETRAIN_CUTOFF = 2023-11-30` is a conservative reading (`config.py:51-58`). Experiment A's "clean room" claim depends on it.
   - MTG prices after 2024-02 are post-cutoff *in time*. The same cards' pre-2024 price history may be in a public web corpus.
   - The report must state the claim as "post-cutoff data", not "an unseen domain".
   - The TimesFM-2.5-as-control idea needs a **sourced** 2.5 cutoff. Without one, 2.5 vs 3 is a model comparison, not a cutoff control.
2. **TimesFM sees no dates, only values.** Perturbing the date (shifting the calendar) cannot test memorization. If TimesFM-3 normalizes each input instance, a rescale is expected to be (near-)invariant, so it is a sanity check, not a probe. Meaningful perturbations:
   - additive noise at a fraction of local volatility
   - **context splicing**: replace the older part of the context with a different period while keeping the last *k* points.

   These test sequence recall. I will verify the normalization claim in the `timesfm` source before relying on it.
3. **Censored lags.** Events that never recover inside the window are dropped today. They must be handled as censored at the window end, using a restricted mean or a rank test, never dropped.
4. **The showcase cards were already seen** in the current results. The pool claim must be reported with and without the 7 showcase cards.
5. **Deployment framing and the license.** The checkpoint license is non-commercial and **non-production**. The latency section must present cost "per 1k forecasts in seconds on GPU X". It must not present a production deployment recommendation. A dollar cost appears only if you supply a sourced price.
6. **Data redistribution (needs your decision).** Committed results contain actual prices:
   - `exp_mtg_demo_slice.parquet`: 166k rows of TCGplayer market prices via TCGCSV
   - `exp_shock_raw_predictions.parquet`: Yahoo prices

   I could not verify either source's redistribution terms offline. Options:
   - (a) keep them, after you check the terms
   - (b) commit only errors and normalized values, and keep actuals in the Kaggle archive only.

   Raw archives and caches are already git-ignored.
7. `pyproject.toml:7` publishes your personal email in package metadata. You may prefer a no-reply address. This is your call, not part of the plan.

---

## 3. Phased plan

Each phase ends with a report: files changed, test evidence, actions for you, and open risks. Every package gets offline tests, a mechanical verify and an independent review before it is accepted. Small commits go on feature branches. Nothing is pushed.

### Phase 1: defensible conclusions (local)

Order: things that change **what the GPU computes** first, because they must be frozen before the Kaggle run. Pure offline analysis follows. An architecture consult with the external tech lead happens before 1.1 and 1.3, because each has more than one viable design.

| # | Package | Key content | Kaggle-blocking? |
|---|---|---|---|
| 1.0 | **Preregistration v2** (`docs/analysis-plan.md`) | Operational rules, see §4. Committed **before** any new run; the commit hash is cited in every manifest | yes |
| 1.1 | Inference layer (Prompt 4 T1–T4, T9), Preregistration v2 §1 | `bootstrap.py`. **Primary:** paired whole-card cluster bootstrap (resample cards with replacement, each card keeps its full history, model and baseline share draws; statistic recomputed from per-card loss sums with bootstrap multiplicities). **Sensitivity:** synchronized calendar-block bootstrap (same blocks for all cards, block length in days = max(h, 7), circular), labelled "time uncertainty, conditional on this panel". **Diagnostic only:** two-stage (cards, then within-card blocks), never a headline. Median and n-weighted skill CIs; paired deltas. DM lags in origin units: max_lag = ceil(h / origin_stride) − 1. BH at q = 0.05 primary and BY at q = 0.05 sensitivity over the card × horizon DM family per comparison. Coverage CIs primary by calendar-block bootstrap; design-adjusted Wilson as diagnostic. Synthetic coverage self-check `scripts/08_inference_selfcheck.py` (G = 8, 20, 40; acceptance at G = 40). `08_exp_inference.py` | no (analysis), but rules frozen in 1.0 |
| 1.2 | 30+ card pool (Preregistration v2 §2 pool rules) | Rule-based selection script run **locally** from the cached TCGCSV archives (no GPU needed). Eligible: single-card products of the MTG category with a Rarity attribute (no sealed product, token or art card); present in the first archive (2024-02-08); observed-day share ≥ 0.90 over the window; median market price ≥ 1.00 USD; one printing per card name (most observed days, ties by lowest productId). Strata: rarity group {mythic, rare, other} × median-price tertile (tertiles over the eligible set). 40 cards allocated proportionally, minimum 3 per non-empty stratum (largest-remainder rounding); seeded draw with seed 20261010. The 7 showcase cards are neither forced in nor out. The frozen manifest (CSV plus sha256) is committed before any model run | yes |
| 1.3 | 02b streaming and resume | Per-combo partitioned parquet (`results/benchmark/<config_id>/combo=…/part.parquet`); a done marker per combo; `--resume`. The summary is computed by streaming over the partitions. The dry-run reports projected partition sizes. **Primary cell** stays committed (small); the full grid goes to the Kaggle archive only | yes |
| 1.4 | Experiment B redesign | (a) Event list from a rule: |z|-detector on five indices (SP500, Nasdaq-100, Euro Stoxx 50, Nikkei 225, Gold; decided 2026-10-10), declustered by a minimum gap, with a guard band around the uncertain cutoff. The list is committed before the run. (b) Baseline-relative lag (model minus naive/ETS) with censoring. (c) Anticipation probe from pre-onset origins: direction hit rate, pre vs post. (d) Perturbations: noise and context splicing. (e) TimesFM-2.5 on the same events. B is **demoted to a red-team appendix now**; it is promoted only if the preregistered rule in §4 is met | yes |
| 1.5 | Experiment C fix | Calibration at all horizons from 02b/03 predictions. Causal split-conformal / CQR on q10/q90, fit on an earlier period and evaluated on a later one. Coverage before and after with CIs, plus width and interval-score cost, by regime × horizon | no (uses 1.3 outputs) |
| 1.6 | Covariates (Prompt 5) | Availability contract; a placebo arm (circular-shifted or shuffled release dates); the legit claim requires its CI to beat both no-covariate and placebo. The leaked arm is watermarked `LEAKED — teaching contrast`, and a test enforces its exclusion from summaries | yes (new arm) |
| 1.7 | Latency benchmark | `09_latency.py`: warmup, `cuda.synchronize`, batch {1,8,32,128} × context {128,512} × h {28,64}, repeated, median and p95. Records GPU name, CUDA and torch versions. CPU baselines (naive, ETS) timed on the same series | yes |
| 1.8 | Reproducibility | `revision` support in both loaders. The resolved checkpoint commit is recorded in the manifest and pinned in `config.py` after the first run. Manifest gains GPU, CUDA, `as_of` and the preregistration commit. Scripts 02–05 write manifests; `--skip-mtg` writes one. Fail-fast checks for GPU, `HF_TOKEN` and license (a gated-repo 401/403 is caught with a clear message) | yes |
| 1.9 | Small P0/P6 leftovers | Forward-fill 0/1/3 sensitivity (runs inside 02b as an ablation), the rename to error-recovery lag and `event_window_nonshock`, a single (multivariate) headline definition for B, removal of `pit_mean` | partly |

Planned routing: scouts and drafters on `haiku`. Developers on `sonnet`, with `opus` only for 1.1 (statistical core) and 1.3 (streaming plus resume correctness), if their complexity warrants it. Reviewers on `sonnet` for every logic change (all of 1.0–1.8).

### Phase 2: Kaggle run package (then stop)

- `run_on_kaggle.ipynb`:
  - A parameter cell (`EXPERIMENTS`, `AS_OF`, `DATASET_DIR`, `RESUME`).
  - Preflight: GPU, token, license, disk and RAM.
  - A timing-calibration cell that prints the projected runtime per step from measured throughput. No runtime numbers are written in advance.
  - Steps: `01 --as-of`, then 02–05, 02b, 2.5 and latency. Every step is idempotent and resumable.
  - Final cell: `tar.zst` of `results/` plus manifests, with a size report.
- `docs/kaggle-runbook.md`: secrets, accepting the gated license, GPU choice, quota planning from the calibration cell, attaching the cached dataset, and what to bring back. Expect one session, plus a second if quota or timeouts interrupt; resume makes that safe.

**→ You run it and return the archive.**

### Phase 3: conclusions (after your artifacts)

- Unpack, run `08_exp_inference.py`, figures and slides.
- `scripts/10_build_evidence_ledger.py` generates `docs/evidence-ledger.md`: claim → value → results file → script → manifest. A test fails if README, report or slides contain a number with no ledger row.
- Write `docs/report.md` (2–4 pages). Its conclusions follow the §4 rules, including "loses to naive" or "inconclusive".
- Update `talk-outline.md` and the slides. The numbers come only from the ledger, and the template's hard-coded numbers move into the ledger or get a source note.

> Slide, talk-outline and talk-material items in Phase 3 and Phase 4 were dropped on 2026-10-14 by the owner.

### Phase 4: presentation polish

- README rewrite, recruiter-first and in English, as you specified. The Italian talk material stays.
- LICENSE: **Apache-2.0**, decided 2026-10-10 (the patent grant and the same family as TimesFM's code). It states that the checkpoint and data stay under their own terms.
- CI: GitHub Actions running `uv sync --extra cpu`, `ruff check`, `ruff format --check` and `pytest`, with badges.
- GitHub Pages for `slides/talk.html` (homepage role not yet decided).
- Agent artifacts: proposal **move, don't delete**. `openai.review.md` and `docs/superpowers/**` go to `docs/dev-notes/` behind a one-line README, plus a short `docs/methodology-review-log.md` (what the external review found and what changed). Nothing is deleted without your OK.
- Suggested description and topics for you to set in the GitHub UI.
- Prompt 8 hostile review, findings fixed or documented in `docs/reviewer-report.md`.

---

## 4. Preregistration v2: summary rules

Approved 2026-10-10; Preregistration v2 in `docs/analysis-plan.md` governs where they differ. Thresholds are design choices, not results.

**General**
- 95 % CIs; B = 2000 bootstrap resamples; a fixed seed recorded in the manifest.
- *Original draft, superseded by the refinement below (now a diagnostic only):* two-stage bootstrap: cards are clusters, resampled with replacement; within each card, moving blocks over origins with block length = max(h, 7).
- BH at q = 0.05 over the card × horizon DM family, per comparison.

*Refinement adopted 2026-10-10 (PM Q14, consult arch-bootstrap); the v2 text in `analysis-plan.md` governs.*
- Primary: whole-card cluster bootstrap (cards resampled with replacement, full history kept, model and baseline paired).
- Sensitivity: synchronized calendar-block bootstrap, block length in days = max(h, 7).
- The two-stage bootstrap above is a labelled diagnostic only.
- DM lags are counted in origin units: max_lag = ceil(h / origin_stride) - 1.
- Benjamini–Yekutieli at q = 0.05 as sensitivity next to BH.
- Synthetic coverage check at G = 8, 20, 40 before real data.
- Reason: the two-stage bootstrap double-counts within-card noise, and "max(h, 7)" was ambiguous between days and origins.
- Decision thresholds are unchanged.

**Experiment A (primary)**
- Primary cell: univariate, raw, `make_positive=True`, context 512. The other cells are ablations.
- Decision at each primary h ∈ {28, 56}:
  - **beats naive** if the 95 % CI of *both* median skill and n-weighted mean skill lies above 0
  - **loses to naive** if both CIs lie below 0
  - otherwise **inconclusive**
- Reported on the frozen pool, with a sensitivity analysis excluding the 7 showcase cards. A disagreement between the two is reported, not resolved.

**Experiment B (exploratory unless promoted)**
- Primary statistic: the difference in mean baseline-relative recovery lag (model minus naive), post minus pre, as a censoring-aware restricted mean with a cluster bootstrap over events.
- Promotion out of the appendix requires at least 8 events per arm *and* a CI that excludes 0 *and* the same sign under the noise and splice perturbations. Otherwise B stays an exploratory red-team appendix, stated explicitly.
- Anticipation probe: a pre vs post difference in direction hit rate from origins −5…−1, with a CI.

**Experiment C**
- Nominal P10–P90 = 80 %.
- "Under-covers" means the coverage CI upper bound is < 0.80.
- "Conformal fixes it" means the post-recalibration CI contains 0.80 on the held-out period, reported with the median width increase.

**Covariates**
- "Helps" requires the paired skill-delta CI to be > 0 versus *both* no-covariate and placebo.

**Latency**
- Descriptive only (median and p95). No decision rule.

---

## 5. Open questions for you (answered 2026-10-10, see §6)

1. Approve this plan and the §4 thresholds, or edit them. Specifically: the pool size of 40, q = 0.05, B promotion at 8 events per arm, and context 512 as primary.
2. Committed actual prices (§2.6): option (a) or (b)?
3. Index list for the Experiment B detector. Do you have a source for TimesFM-2.5's cutoff? Without one, 2.5 is presented as a model comparison only.
4. LICENSE: Apache-2.0 or MIT?
5. Is a ~2-session Kaggle budget acceptable? The calibration cell will give real numbers before the long steps run.

---

## 6. Decisions (2026-10-10)

The human adopted the PM proxy's suggestions (record: pipeline run 20261010T104754Z-7b42a4f8, `plan.md` "PM decisions"). This answers §5.

- Pool size 40.
- BH q = 0.05.
- Experiment B promotion at >= 8 events per arm.
- Primary context 512.
- Five detector indices: SP500, Nasdaq-100, Euro Stoxx 50, Nikkei 225, gold.
- TimesFM-2.5 is a model comparison only (no sourced cutoff).
- About 2 Kaggle sessions, no paid resources.
- LICENSE: Apache-2.0.
- `pyproject` author email: the GitHub no-reply address.
- Committed prices: option (b) for new outputs (errors and normalized values only). Already-committed price files are left untouched until Phase 3; older commits still hold actual prices.
