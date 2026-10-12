# Preregistration v2

**Preregistration v2 — frozen at commit: the commit that introduces this section (see git log).**
Any later change requires a dated amendment section stating why and whether data had been seen.

All numbers below are design choices, not results.

## 0. Scope and status

- This section supersedes the v1 "Claim rule" (not operational). The v1 text is kept below, unchanged, as "v1 (historical, Italian)".
- Nothing in v2 has been run. Existing `results/` files predate v2 and are exploratory; they are never used to confirm v2 claims.
- The pretraining cutoff (2023-11-30, `config.PRETRAIN_CUTOFF`) is a conservative reading, not a certified date. Claims say "post-cutoff data", never "unseen domain".
- Later changes: a dated amendment section is required (see the header above).

## 1. General inference rules

- 95% two-sided intervals; B = 2000 resamples; seed 20261010 (numpy `default_rng` / PCG64), recorded in every manifest.
- Percentile intervals. No replicate is silently dropped; invalid replicates are counted and reported.

### Aggregate skill (Exp A, covariates)

- PRIMARY: paired whole-card cluster bootstrap.
  - Resample cards with replacement; each card keeps its full history.
  - Model and baseline errors share draws.
  - The statistic is recomputed from per-card loss sums with bootstrap multiplicities: uniform cluster sampling, no double weighting.
- SENSITIVITY: synchronized calendar-block bootstrap.
  - Same calendar blocks for all cards; block length in DAYS = max(h, 7); circular.
  - Labelled "time uncertainty, conditional on this panel".
- DIAGNOSTIC ONLY: two-stage (cards, then within-card blocks). Never a headline.
- Reason for the change from the plan: two-stage double-counts within-card noise (consult arch-bootstrap, PM Q14).

### Definitions

Per card i, horizon h, over observed rows only: s_i = 1 - sum|e_model| / sum|e_naive|; relative MAE r_i = 1 - s_i.

- median skill = 1 - median_i(r_i)
- n-weighted mean skill = 1 - sum_i n_i r_i / sum_i n_i (n_i = observed rows) — exactly `tfm3lab.summarize.aggregate_leaderboard`.
- pooled skill = 1 - sum_it|e_model| / sum_it|e_naive| — labelled SECONDARY, never a decision input.

### Per-(card, horizon) Diebold–Mariano tests

- Absolute loss. HAC long-run variance with lags in ORIGIN units: max_lag = ceil(h / origin_stride) - 1.
- HLN correction uses that effective horizon (max_lag + 1).
- Minimum 2*(max_lag+1) observations, else NaN. Zero variance handled explicitly (stat 0, p 1).
- Benjamini–Hochberg at q = 0.05 per comparison family is PRIMARY. One family = all card × horizon tests of one model-vs-baseline comparison.
- Benjamini–Yekutieli at q = 0.05 is SENSITIVITY.
- DM/BH results are secondary to the aggregate-skill CIs and never override them.

### Synthetic coverage check (offline, before real data)

- Simulate panels with known skill: independent heterogeneous clusters, AR(1) serial dependence, a common shock factor, unequal cluster sizes.
- G = 8, 20, 40; 400 simulated datasets per G.
- Acceptance at G = 40: empirical coverage of the nominal 95% primary CI within [0.92, 0.98].
- If it fails, every reported CI carries the simulated coverage next to it and is labelled "approximate".
- G = 8 and 20 results are reported, never used for acceptance. Anything with < 20 clusters is labelled small-cluster/exploratory.

## 2. Experiment A — post-cutoff MTG benchmark (primary claim)

### Data

- TCGCSV daily archives 2024-02-08 .. as_of 2026-09-04 (fetch manifest `results/manifests/fetch-2026-09-04-*.json`).
- Forward fill max 3 days; observed=False rows excluded from every metric.

### Card pool

40 cards, rule-based, frozen in a committed manifest (CSV + sha256) BEFORE any model run.

- Eligible:
  - single-card products of the MTG category (has a Rarity attribute; no sealed product, token, or art card);
  - present in the first archive (2024-02-08);
  - observed-day share ≥ 0.90 over the window;
  - median market price ≥ 1.00 USD;
  - one printing per card name (if several eligible printings, keep the one with the most observed days, ties by lowest productId).
- Strata: rarity group {mythic, rare, other} × median-price tertile (tertiles computed over the eligible set) = 9 strata.
  - Allocate 40 proportionally to stratum sizes with a minimum of 3 per non-empty stratum (largest-remainder rounding).
  - Seeded random draw within strata, seed 20261010.
- Showcase cards (7, `DEFAULT_CARDS`) are NOT forced in or out.
  - The claim is reported on the pool, plus a sensitivity excluding any showcase card that was drawn, plus showcase-only as descriptive.
  - Disagreement is reported, not resolved.

### Primary cell

- TimesFM-3, univariate, transform raw, make_positive=True, context 512, origin_stride 7.
- Horizons evaluated at h ∈ {28, 56} (primary); grid h ∈ {1, 7, 28, 56, 64}.

### Decision rule

At each primary h separately, versus naive:

- BEATS NAIVE if the 95% primary CI of BOTH median skill and n-weighted mean skill lies entirely above 0.
- LOSES TO NAIVE if both lie entirely below 0.
- Otherwise INCONCLUSIVE.
- The report states the verdict verbatim, including "loses to naive".

### Secondary (descriptive, CIs, no verdict)

- Other baselines (seasonal_naive, drift, ets).
- Context 64/128/256; log1p; make_positive False.
- Multivariate vs univariate; multivariate vs placebo panel (panel size 7, seed 42).
- TimesFM-2.5 vs TimesFM-3: model comparison ONLY. There is no sourced 2.5 cutoff, so it is never a cutoff control.
- Forward-fill 0/1/3 days sensitivity on the primary cell.

## 3. Experiment B — market shocks (EXPLORATORY red-team appendix unless promoted)

### Event list

By rule, frozen and committed before any model run.

- Indices: SP500 (^GSPC), Nasdaq-100 (^NDX), Euro Stoxx 50 (^STOXX50E), Nikkei 225 (^N225), Gold (GC=F); daily close, 2018-01-01 .. 2026-09-04.
- Shock day: |z| ≥ 4 on daily log returns, z standardized with a trailing 250-trading-day mean and std (strictly past data, so detection uses no future values).
- Declustering: shock days of any index within 20 trading days are merged into one event; onset = earliest shock day; event index set recorded.
- Arms:
  - pre = onset ≤ 2023-05-31;
  - post = onset ≥ 2024-06-01;
  - onsets in the guard band 2023-06-01 .. 2024-05-31 are excluded (cutoff uncertainty).
  - Events whose window (context + 25 trading days after) does not fit in the data are excluded.
- The five hand-listed legacy events are reported separately for continuity, never pooled into the rule-based arms.

### Error-recovery lag

- One-step (h=1) absolute percentage error on the onset index.
- Threshold = 1.5 × median pre-onset error of THAT forecaster (1.25 and 2.0 as sensitivity).
- Recovery = first run of 3 consecutive post-onset points ≤ threshold.
- Never-recovered events are CENSORED at the window end (25 trading days), never dropped.

### Primary statistic

- Baseline-relative lag D = lag_TimesFM3 − lag_naive per event (ETS as second baseline, sensitivity).
- Effect = mean D(post) − mean D(pre), censored values entering at the window end (restricted mean).
- 95% CI by event-cluster bootstrap within arms.

### Anticipation probe

- From origins at offsets −5..−1 before onset: direction hit = sign(median forecast for onset day − last observed value) equals sign(actual onset-day move).
- Effect = hit rate(pre) − hit rate(post), event-cluster bootstrap CI.

### Perturbation tests (pre and post events)

- (a) Additive Gaussian noise on log values with σ = 0.25 × context log-return std, 5 seeds.
- (b) Context splicing: replace all context points older than the last 32 with a same-length segment from the same index ending ≥ 1 year earlier, level-shifted to join continuously, 5 seeds.
- Rescaling is a sanity check only; calendar shifts are not tests because the model sees no dates.

### Other

- TimesFM-2.5 on the same events: model comparison only.
- PROMOTION out of the appendix requires ALL of:
  - ≥ 8 events per arm;
  - primary CI excludes 0;
  - same sign of the effect under both perturbations.
- Otherwise B is presented explicitly as an exploratory red-team appendix. Even when promoted it is labelled small-cluster.

## 4. Experiment C — calibration and conformal fix

- Interval: P10–P90, nominal 80%.
- Regimes:
  - `mtg` (Exp A primary cell);
  - `market_shock` (|offset| ≤ 3 trading days from onset);
  - `event_window_nonshock` (renamed from market_calm: non-shock days inside event windows — NOT a general calm sample).
- Horizons: MTG {1, 7, 28, 56}; market {1, 7, 28} (Exp B forecasts are produced up to h = 28).

### Coverage CI

- PRIMARY: synchronized calendar-block bootstrap (fixed-panel, future-time estimand), block = max(h, 7) days.
- SENSITIVITY: whole-card / whole-event cluster bootstrap.
- Design-adjusted Wilson as diagnostic.
- "Under-covers" = coverage CI upper bound < 0.80.

### Recalibration

- Conformalized quantile regression (CQR) per (regime family, horizon).
  - Conformity score max(q10 − y, y − q90).
  - Threshold = the ⌈(n+1)(0.8)⌉/n empirical quantile on the CALIBRATION split; applied symmetrically.
- Splits are chronological:
  - MTG: calibration = forecast origins before 2025-06-01 (targets of calibration rows must also precede 2025-06-01); evaluation = origins on/after 2025-06-01.
  - Market: calibration = pre-arm events; evaluation = post-arm events.
  - Thresholds never see evaluation outcomes.
- "Conformal fixes it" = post-recalibration coverage CI on the evaluation split contains 0.80. Reported with median relative width increase and interval score before/after.
- Regime results are empirical subgroup evaluations, not conformal guarantees (dependent data break exchangeability).

## 5. Covariates (Prompt 5)

- Arms:
  - no covariate;
  - legitimate covariate (calendar day-of-week for market; days-to-next-set-release for MTG using the release date known at forecast time);
  - PLACEBO (MTG: release dates circularly shifted by a seeded offset of 60–300 days; market: day-of-week labels permuted by a seeded permutation);
  - LEAKED (real future values) as a teaching contrast.
- Pool: the Exp A frozen pool (cards whose set release dates are resolvable); context 128, horizon 7, origin_stride 7.
- "Helps" iff the paired skill-delta primary CI is > 0 versus BOTH no-covariate AND placebo.
- The LEAKED arm is watermarked "LEAKED — teaching contrast, not a result" and excluded from every summary/headline by test.

## 6. Latency

- Descriptive only: median and p95 over repeats, after warmup, with GPU synchronization.
- Grid: batch {1, 8, 32, 128} × context {128, 512} × h {28, 64}; CPU naive and ETS on the same series.
- Reported as seconds per 1k forecasts on the recorded GPU. No deployment recommendation (checkpoint license is non-commercial).
- No dollar figures unless a sourced price is supplied by the human.

## 7. Provenance

- Every run manifest records: git SHA + dirty flag, this preregistration commit, as_of, checkpoint repo id + resolved revision, package versions, GPU name, CUDA and torch versions, seed, config id.
- Data redistribution: new committed outputs carry errors and normalized values only; actual prices stay in the local Kaggle archive (PM Q6/Q12).

## Amendment 2026-10-10, before any v2 data were seen

Experiment B's primary forecaster mode is univariate, so that TimesFM-3, the naive and ETS baselines and TimesFM-2.5 (univariate only) are directly comparable. The multivariate mode is a sensitivity analysis. Nothing else in section 3 changes.

## Amendment of 2026-10-10 (human decision, before any v2 result was seen)

Decision rule for Experiment A, replacing the rule in section 2 ("Decision rule"). At each primary h separately, versus naive:

- BEATS NAIVE iff the 95% primary whole-card cluster CI AND the 95% calendar-block CI, for BOTH median skill and n-weighted mean skill (four intervals), all lie entirely above 0.
- LOSES TO NAIVE iff all four lie entirely below 0.
- Otherwise INCONCLUSIVE. Comparison is strict at 0 (an interval touching 0, or with a NaN bound, is not entirely on one side).
- Reason: the synthetic stress scenario showed the primary CI under-covers under strong common shocks (see docs/inference-selfcheck.md); requiring the calendar-block CI to agree guards against that.
- Timing: the Kaggle full run had started but no output had been downloaded or inspected when the rule was adopted. The rule can only make claims harder to reach, never easier.
- The original rule's verdict (primary CI only, both statistics) is still reported as a labelled sensitivity (`verdict_original_rule`). Showcase-only rows stay descriptive; their `descriptive_direction` uses the new rule. All thresholds, horizons and other settings are unchanged. Manifests record `rule_version` (`strict-2026-10-10`).
- This supersedes the "Recommendation pending a human decision" bullet of amendment B, which the human adopted ("yes, use the stricter verdict rule").

## Amendments of 2026-10-10 (before any v2 model run)

All items are dated 2026-10-10 and were made before any v2 model run; no v2 model output exists.

- **A. Experiment A pool eligibility** (decision: PM-proxy Exchange 5, human-delegated)
  - The first literal draw admitted non-game collectibles that carry a TCGplayer Rarity attribute (World Championship Deck decklist/biography cards, an oversize card, a serial-numbered card, a LARP promo). Only item types/names were inspected; no prices, forecasts or model outputs.
  - Clarified eligibility: Rarity in {Mythic, Rare, Uncommon, Common} (rarity group "other" = Uncommon + Common); exclude the World Championship Deck group and products whose name contains (case-insensitive) Oversize, Oversized, Serial Numbered, Decklist, Biography, Checklist. No other change; same script, seed 20261010 and as_of 2026-09-04.
  - First draw CSV sha256: ed0bdd0b32cbe7e92eb7b122fe94f7ee62a468d0e8f21d78c663c0814e85817e.
  - Frozen pool: configs/benchmark_pool_v2.csv, sha256 e671a846417da9345c01d174a0b96135b4d65760c3935477f6c5397f0192d29d; manifest results/manifests/pool-v2-2026-09-04.json.
  - Interpretation recorded: "one printing per card name" folds variants whose names differ only by a trailing parenthetical (e.g. "(Extended Art)", "(Borderless)"). The catalog was fetched on 2026-10-10, after as_of (limitation).
  - No showcase card was drawn; showcase-only results therefore come from a separate primary-cell run (configs/benchmark_v2_showcase.json), descriptive only.
- **B. Self-check disclosure**
  - Baseline DGP constants were chosen after a 40-dataset preview that showed undercoverage (see docs/inference-selfcheck.md). A stress DGP was added as a sensitivity and is never used for acceptance; its coverage is reported next to each headline CI.
  - A first stress run was discarded because a Student-t common factor inside exp() has no finite mean, so the target was undefined; the factor was made Gaussian with every numeric constant unchanged. The implementer's report does not state whether that discarded run's coverage output was inspected; treat it as seen.
  - Recommendation pending a human decision (not adopted): require the primary cluster CI AND the calendar-block CI to agree for a "beats/loses" verdict.
- **C. Error scaling of committed outputs**
  - Committed absolute errors are divided by one constant per series (median of its observed values over the loaded window, never written). Per-card skill ratios are therefore exactly the raw-MAE ratios of §1; pooled skill computed from committed files is level-free and differs from raw pooled skill (secondary only).
- **D. Experiment C regime definition**
  - "|offset| <= 3 trading days from onset" is applied to the TARGET day of each forecast: target_offset = origin_offset + horizon_step - 1. When a CQR split lacks data the "fixes it" verdict is reported as not available, never as "no".
- **E. Forward-fill sensitivity mechanics**
  - For ffill k in {0, 1}: non-observed points are set to missing, re-filled forward with limit k; per card, origins whose context still contains a gap are dropped and counted (a card with no remaining origin is excluded and listed). Univariate primary cell only.
- **F. Experiment B event list outcome** (rule output, not a model result)
  - The frozen rule produced 15 events: 8 pre-cutoff, 7 post-cutoff (configs/shock_events_v2.csv). The promotion rule needs >= 8 per arm, so Experiment B is reported as an exploratory red-team appendix regardless of its results. Declustering chains from the previous shock day (gap <= 20 trading days on the union calendar).

## Amendment of 2026-10-11 (before any TimesFM-2.5 output existed)

- Kaggle session 1 showed that per-card CPU baselines dominate cell cost: 40-card cells took 4.3-7.9 min versus 0.8-1.3 min for 7-card cells.
- TimesFM-2.5 vs TimesFM-3 is a model-comparison-only descriptive secondary, so the 2.5 config (configs/benchmark_v2_timesfm2p5.json) computes only the naive baseline. ETS, seasonal naive and drift for 2.5 were never preregistered comparisons.
- The 2.5 grid runs primary-slice cells first; any cells not run are listed as not run.
- Decision: PM proxy (Exchange 7) under the human's delegation. Nothing else changes.

## Amendment of 2026-10-12 (after Kaggle runs; before any recalibrated MTG number was computed)

- Finding: the preregistered MTG calibration split (origins and targets before 2025-06-01) is empty by construction. Context 512 with data from 2024-02-08 makes the first feasible origin 2025-07-04 (53 weekly origins, 2025-07-04 to 2026-07-03, 40 cards).
  - The preregistered MTG "conformal fixes it" test is therefore reported as NOT EVALUABLE AS PREREGISTERED.
  - The market-regime split is unchanged. market_shock at h=28 has no evaluation rows and is reported as not evaluable.
- Seen when the rule below was fixed: uncalibrated MTG P10-P90 coverage on all origins per horizon, the market-regime before/after conformal results, and the Experiment A verdicts. No recalibrated MTG coverage under any alternative split had been computed.
- Exploratory rule, fixed exactly:
  - Let o_1 < ... < o_N be the distinct origin dates of the MTG primary-cell frame (N = 53). D' = o_{floor(N/2)+1} (= o_27 = 2026-01-02 in the committed primary export).
  - Calibration rows: origin_date < D' AND target_date < D' (per-horizon purge, strict). Evaluation rows: origin_date >= D'. Same D' for all cards and horizons; no rebalancing.
  - Same CQR as section 4 (pooled per horizon, same score and order-statistic convention) and same coverage CI methods (primary calendar-block, sensitivity whole-card). Before/after coverage is computed on exactly the same evaluation rows.
  - Rejected: an expanding-window variant (adds a second analyst choice and breaks the single-threshold comparison); shorter-context cells (a different forecasting configuration from the Experiment A primary cell).
- Labelling: results are "Exploratory MTG analysis - post-hoc amended split", never the preregistered verdict. The rule was "fixed before recalibration"; no claim of blindness to outcomes is made. All outcomes are reported whatever they show.
- Decision: consult conformal-split (advice) and PM proxy Exchange 8 under the human's delegation. Nothing else changes.

---

# v1 (historical, Italian) — SUPERSEDED by Preregistration v2 above

The text below is the original v1 preregistration, preserved verbatim for the record. It is not operational.

Primary target:
  prezzo marketplace giornaliero, non consiglio di investimento.

Primary horizons:
  h=28 e h=56 (dentro la griglia completa h in {1, 7, 28, 56, 64} -- vedi
  configs/benchmark_preregistered.example.json). h=28/56 sono i primary
  perche' coprono l'orizzonte "trading window" tipico per una carta
  collezionabile (mese/bimestre), non il one-step banale ne' il tail 64
  dominato dallo stitching multi-pass di TimesFM-3 oltre OUTPUT_PATCH_LENGTH.

Primary metric:
  skill = 1 - MAE_model / MAE_naive (relative MAE = MAE_model / MAE_naive;
  skill ne e' il complemento a 1 -- skill > 0 vuol dire il modello batte la
  naive, skill < 0 il contrario).

  Aggregazione fra carte: mediana (robusta a una carta outlier) E media
  pesata per numero di osservazioni per carta (n) -- mai una media ingenua
  di rapporti (una carta con 5 osservazioni peserebbe come una con 5000).
  Vedi tfm3lab.summarize.aggregate_leaderboard.

Primary comparison:
  TimesFM-3 univariato vs naive.

Secondary comparisons (leaderboard multi-baseline, non solo naive):
  vs seasonal_naive, drift, ets (quando converge) --
    tfm3lab.summarize.summarize_leaderboard;
  multivariato vs univariato;
  multivariato panel vero vs panel placebo (carte random dal pool piu'
    ampio, seed configurabile) -- tfm3lab.benchmark.select_placebo_panel;
  covariata lecita vs nessuna;
  raw vs log1p;
  make_positive True vs False;
  TimesFM-2.5 (zero-shot storico) vs TimesFM-3, quando l'adapter e'
    disponibile (tfm3lab.model_2p5) -- nessun run GPU reale eseguito in
    questo branch.

Ablation grid:
  context_lengths in {64, 128, 256, 512};
  horizons in {1, 7, 28, 56, 64};
  origin set condiviso -- calcolato UNA volta su
  (max(context_lengths)=512, max(horizons)=64), thinnato da origin_stride,
  poi riusato identico per ogni cella della griglia (vedi
  tfm3lab.benchmark.common_origin_set) -- requisito esplicito: "tutte le
  configurazioni devono essere confrontate sulle stesse origini".

Card pool:
  showcase (7 carte, DEFAULT_CARDS) per il talk;
  benchmark manifest piu' ampio (30+) OPZIONALE -- formato + criteri
  documentati in configs/benchmark_cards.example.csv, nessuna selezione
  reale di 30+ carte inventata in questo branch (richiederebbe un crawl
  TCGCSV live, fuori scope).

Exclusion rules:
  righe con observed=False escluse da ogni metrica (forward-fill, non
    un'osservazione reale -- gia' applicato in summarize.py, non toccato
    da questo lavoro);
  ETS escluso dal confronto quando non converge sul contesto dato
    (eccezione soppressa per QUELLA riga in backtest._baseline_forecasts,
    non l'intero run);
  combinazione multivariate_placebo esclusa quando il pool di carte
    disponibili e' piu' piccolo o uguale a placebo_panel_size (una
    partizione placebo richiede piu' carte del pool assegnato al panel
    reale, altrimenti campiona l'intero pool -- skip esplicito, contato
    nel dry-run report, mai un gap silenzioso);
  Diebold-Mariano non calcolato sotto MIN_OBSERVATIONS_FOR_DM_TEST
    osservazioni per gruppo (dm_stat/dm_pvalue = NaN, non un numero
    inventato).

Uncertainty:
  paired moving-block bootstrap;
  block length >= horizon;
  correzione BH per p-value multipli.

Claim rule:
  "migliora" solo se CI del delta e' coerente e preregistrata.
