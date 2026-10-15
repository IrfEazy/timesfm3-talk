# Preregistration v3: new-set cold start, buy now or wait, spike hold or revert

Dated 2026-10-15. This document freezes three new studies on the same TCGplayer
price data (via TCGCSV) and the same TimesFM-3 checkpoint used by Preregistration v2
(`docs/analysis-plan.md`). It is committed **before any v3 forecast exists and before
any v3 outcome has been computed**. The evaluation code it relies on
(`src/tfm3lab/studies/inference.py`, `eval_s1.py`, `eval_s2.py`, `eval_s3.py`) and the
selection code (`selection.py`, `scripts/13_v3_select.py`) were committed before this
document and are part of the plan.

If results contradict the narrative, the narrative changes, not the analysis.

## 1. Framing and limitations

- **Retrospective backtest.** All data (2024-02-08 .. 2026-09-03, as_of 2026-09-04)
  is after the conservative 2023-11-30 pretraining cutoff (`config.PRETRAIN_CUTOFF`).
  TimesFM-3's training coverage after that date is not certified, so claims say
  "post-cutoff data", never "unseen data", and nothing is presented as a real-time
  forecast.
- **Catalogue.** Set membership, rarity and release dates come from a catalogue
  snapshot fetched on 2026-10-10 (`data/raw/catalog-2026-09-04`), not a point-in-time
  catalogue. Groups whose `publishedOn` is after the as_of (e.g. promo groups stamped
  with the fetch time) are treated as undated for the first-printing flag.
- **Quotes are not transactions.** TCGplayer market prices are summary quotes: no
  sales volume, fees, shipping, condition or availability. S2 reports a
  **quote-based price advantage**, never "savings".

## 2. Common definitions

- SKU = (product_id, subtype); Normal and Foil are separate SKUs. Price policy
  MARKET_THEN_MID (project rule). Local per-day panel built from the 939 cached
  archives (`scripts/12_build_panel.py`, manifest
  `results/manifests/panel-2026-09-04-20261015T012257Z.json`).
- A quote is **valid** iff the selected price is finite and > 0. "Observed on d" = a
  valid price exists on day d. q(d) = valid price on d, else the latest valid price in
  [d-3, d-1], else missing (forward fill of at most 3 days, never backward).
- Single card = catalogue Rarity in {M, R, U, C}, excluding group WCD and names
  containing oversize/decklist/biography/checklist (`tfm3lab.data.pool`). Card
  identity (cluster id) = casefolded name without trailing parentheses. Finish class:
  Normal, Foil (subtype contains "Foil"), Other.
- **Model input.** The context is the trailing run of non-missing q() values ending at
  the origin (it stops at the most recent missing day), last <= 512 values, divided by
  q(origin). The forecast is the TimesFM-3 median (q50), rescaled by q(origin).
  Non-positive or non-finite model outputs are floored at 0.001 x q(origin) and
  counted. Official inference defaults (symmetric averaging, make_positive); the
  pinned checkpoint revision is stamped in every forecast part.
- **What leaves the machine.** `data/cache/v3_requests.parquet` holds only scaled
  contexts, by construction of `scripts/13_v3_select.py` (no outcomes, no absolute price
  level). It is added to the PRIVATE Kaggle cache dataset, which also holds the v2 price
  caches and is never shared. All outcomes are evaluated locally. Nothing
  price-bearing is committed: committed outputs are errors, ratios, advantages, coarse
  flags, ids, dates and counts.
- Bootstrap: B = 2000, seed 20261010, percentile intervals.

## 3. Multiplicity

The confirmatory family is {S1 primary, S2 primary, S3 primary}. Verdicts use
Bonferroni alpha = 0.05/3 (98.33% two-sided intervals), fixed regardless of S3's
status. Every other comparison (including the S1 cohort-curve comparison) is
exploratory: 95% intervals, no verdict wording.

## 4. S1: new-set cold start (confirmatory)

- **Release events**: distinct `publishedOn` dates R of non-supplemental Magic groups
  with 2024-02-08 <= R <= 2026-06-05; an event pools all groups sharing R.
- **Origins** O_k = R + k with k = 14 primary, 7 and 28 secondary. Target T = R + 90;
  horizon 90 - k.
- **Eligible SKU at O_k**: single card in the event's groups; q(O_k) not missing and
  >= $0.25; at least 5 observed days in the trailing context run. Primary floor:
  q(O_k) >= $1.00 (`floor_1`); the $0.25 floor is secondary.
- **Outcome** q(T); missing = attrition, reported per event, identical for all methods.
- **Methods**: TimesFM-3 median; naive q(O_k); cohort curve = q(O_k) x exp(median over
  reference SKUs of log(q(R'+90)/q(R'+k))), where reference events satisfy
  R' + 90 <= O_k, reference SKUs follow the same eligibility with an observed day-90
  outcome, stratum rarity x finish with fallback to pooled rarity and then to all when a
  stratum has fewer than 20 reference SKUs. Events with fewer than 2 reference events
  have no cohort forecast.
- **Metric**: absolute log error |log(f / q(T))|. Event score = mean over its eligible
  SKUs with an outcome. Events with fewer than 20 such SKUs are excluded from
  event-level averages (count reported).
- **Primary** (k = 14, floor $1): D = mean over events of [ALE_TimesFM(e) - ALE_naive(e)].
  Verdict needs both a Student-t interval over event-level paired differences
  (df G - 1, 98.33%) and a sign-flip randomization test (exact enumeration if G <= 20,
  otherwise 20000 Monte Carlo draws; two-sided p < 0.05/3) in the same direction:
  "beats naive" / "loses to naive" / "inconclusive". With G < 10 qualifying events the
  result is "descriptive only". The event percentile bootstrap is descriptive.
- **Exploratory**: cohort-curve comparison (events with >= 2 reference events, same
  SKUs); k = 7 and 28; floor $0.25; first printing vs reprint; bias (signed log error);
  q10..q90 coverage at T; player-facing descriptive of log(q(T)/q(O_14)) by rarity x
  finish (event-weighted median and share below 0, event bootstrap 95%).

## 5. S2: buy now or wait? (confirmatory)

- **Universe** single cards of any group; origin window t in [2024-08-01, 2026-08-06].
- **Eligible (SKU, t)**: observed on t, q(t) >= $1.00, first observed <= t - 90,
  >= 60 observed days in [t - 90, t], trailing context of at least 60 values.
- **Sampler (R11)**: a fixed pool of M = 5500 draws with seed 20261014: draw a date
  uniformly in the window, then one SKU uniformly among the SKUs eligible on that date
  (listed in canonical (product_id, subtype) order; a date with no eligible SKU is
  redrawn). Candidates are processed by (origin, draw index); a candidate is accepted
  only if no previously accepted decision of the same card identity has origin in
  (t - 28, t]. Every accept/reject therefore uses data <= t only. The final N is the
  accepted count, with no top-up; the target population is the one induced by this
  procedure. Price band from q(t): 1-5, 5-20, 20+ dollars.
- **Forecast** horizon 28; scaled path m_0 = 1, m_1..m_28.
- **Policies** (one execution day d* in 0..28 chosen at t; ties go to the earliest day):
  P_TFM (primary) = argmin of the TimesFM median path; P_ETS and P_drift = the same rule
  on the project's ETS and drift forecasts from the identical scaled context; P_TFM5
  waits only if min over d >= 1 of m_d <= 0.95 and then executes at the argmin.
  References: buy-now (d* = 0), deadline (d* = 28), random day (mean over d = 0..28 of
  the executed advantage), oracle (best observable price; not a policy).
- **Execution (no conditioning on fills)**: if q(t + d*) is missing, execute at the first
  later day in (d*, 28] with a quote; if none, the decision fails and receives the worst
  observable advantage in the window. Fail and delay rates are reported per policy.
- **Advantage** A = 1 - e / q(t) (positive = cheaper than buying now).
- **Analysis set**: a decision whose TimesFM path is missing or non-finite is removed
  from every TimesFM comparison (counted). A failed ETS or drift fit (exception,
  non-finite or non-positive path) makes that policy buy now for that decision
  (counted).
- **Primary**: mean A_TFM. Two procedures at 98.33%: cluster bootstrap by card identity
  and the synchronized circular calendar-block bootstrap over origin dates with block
  length 28 days. "beats buying now" only if both lower bounds > 0; "worse than buying
  now" only if both upper bounds < 0; otherwise "inconclusive".
- **Exploratory** (95%, same procedures): mean(A_TFM - A_ETS), mean(A_TFM - A_drift),
  P_TFM5, references, by price band and finish, share of waits and hit rate of waits.

## 6. S3: spike, hold or revert? (conditional confirmatory)

- **Event (SKU, t)**: ratio(d) = q(d)/q(d-7) is defined when the SKU is observed on d
  and q(d-7) is not missing. Upward crossing at t: ratio(t) >= 1.5 and ratio(t-1)
  defined and < 1.5; q(t-7) >= $1.00; first observed <= t - 90; trailing context of at
  least 60 values; t in [2024-05-08, 2026-07-09]. Crossings are processed
  chronologically; among same-day crossings of one card identity the SKU with the
  largest q(t-7) is kept (ties: lowest (product_id, subtype)); an event is accepted only
  if its card identity has no accepted event in (t - 56, t).
- **Precision target** (fixed before the census): confirmatory iff at least 100 events
  spanning at least 12 distinct calendar months; otherwise descriptive only.
- **Methods** at h = 28 (primary) and 56: TimesFM-3 median; hold q(t); revert q(t-7);
  halfway sqrt(q(t) q(t-7)). Outcome q(t + h); missing = attrition, same set for all
  methods; a missing TimesFM forecast removes the event from all methods (counted).
- **Primary**: mean(ALE_TFM - ALE_hold) at h = 28, card cluster bootstrap and
  calendar-block bootstrap (block 28 days) at 98.33%: "beats hold" / "loses to hold" /
  "inconclusive".
- **Exploratory**: vs revert, vs halfway, h = 56 (block 56 days), q10..q90 coverage,
  share of events below the spike-day quote after 28 days. The probability/Brier
  analysis of the draft was dropped (the model returns 9 quantiles, not a CDF).

## 7. Kaggle run and interruption policy

One requests file, priority blocks: 1 = S1 k = 14; 2 = S2; 3 = S3; 4 = S1 k = 7 and 28.
The runner is resumable and stamps the requests digest, checkpoint revision and
inference flags into every part. If a session ends early, completed blocks are used
and any study with an incomplete block is reported as incomplete; nothing is
resampled or cherry-picked.

## 8. Frozen samples and blind census

Produced by `scripts/13_v3_select.py --as-of 2026-09-04` before this document. Every
selection rule uses only data up to its unit's own origin; the panel was loaded only up to
2026-08-06 (`panel_end_loaded` in the manifest), so outcome days after that were not read
at all; no forecast and no outcome metric had been computed. The census contains counts only.

| File | sha256 |
|---|---|
| `configs/v3/release_events.csv` | `30faafe317416650fbca25e9cfbe8d594b7f32cb10b08234e15448f1f6b11b59` |
| `configs/v3/s1_skus.csv` | `dc1003b878583a5cbd354a546412d655c06fdc9f0d8b20a652e20c55ae33d99a` |
| `configs/v3/s2_decisions.csv` | `bf55b4d61c54747b253e5968b063fa6edad4365fe9ea0132952c0ac4842926d5` |
| `configs/v3/s3_events.csv` | `85eabe3f99631a3380f300363ba98830cfeb71b205e03d39856ff20f2883a1d1` |
| `results/v3_census.json` | `26a16ec4a1033494abbb4d5126cf1d8077b4b08bdac7447de60e333d5fa095ea` |
| `results/manifests/v3-select-2026-09-04-20261015T074243Z.json` | `d8d13393180584c2ffbf1c973a3672d475c4a6fe8c820da3f93914382bb09ef0` |
| `data/cache/v3_requests.parquet` (not committed; private Kaggle dataset only; digest stamped by the runner) | `8206e550863caa33c088681693ddd7046b848d1b79bbc439c06f670562cec2c7` |

Census counts (copied from `results/v3_census.json` and the selection summary):

- S1: 20 release events; 45823 S1 units (SKU x origin across k = 7, 14, 28, floor $0.25).
- S2: pool M = 5500 candidates, 5325 accepted decisions, 175 rejected, 3937 distinct cards.
- S3: 21028 upward crossings, 20658 after same-day deduplication, 13275 accepted events,
  7151 distinct cards, 27 distinct months; the precision target is met, so S3 is
  confirmatory.
- 64423 forecast requests in total.

## 9. Changes from the draft after independent review

Retrospective framing; one validity/gap policy for contexts and outcomes; Bonferroni
over the three primaries; small-cluster S1 inference (t + sign-flip, descriptive
below 10 events); S2 failures count against the policy; S3 crossing rule made exact and
the Brier analysis dropped; paired analysis sets; corrected upload wording; S2 sampler
made backward-looking (R11).

## 10. Implementation precisions (binding)

These make explicit what the committed code does; where a sentence above is shorter,
this section governs.

1. **First printing (S1 exploratory).** A SKU is a first printing iff its card identity
   appears in no catalogue group with `publishedOn` strictly earlier than its own group's;
   groups dated after the as_of, or undated, never count as earlier, and SKUs in such
   groups are not first printings.
2. **Exploratory intervals.** S1: 95% Student-t interval over event-level differences
   (the event percentile bootstrap is reported as descriptive). S2 and S3: card-cluster
   and calendar-block bootstrap at 95%. No exploratory result carries verdict wording.
3. **S1 sign-flip test.** Statistic |sum of event differences|; p = share of sign
   patterns with a statistic >= the observed one (relative tolerance 1e-12). With
   G <= 20 (20 release events exist, so always) the enumeration is exact; the
   plus-one-corrected Monte Carlo branch (20000 draws, seed 20261010) would apply only if
   G > 20. A verdict also requires the sign of the mean difference to agree; a negative
   mean difference favours TimesFM-3.
4. **S1 paired set.** A SKU without a valid TimesFM forecast is removed from every
   method (counted as `n_removed_missing_forecast`). The 20-SKU event qualification is
   applied after pairing. The cohort comparison additionally requires a cohort forecast
   for the SKU.
5. **S3 comparison sets per horizon.** Each horizon has its own set: quotes at t, t-7 and
   t+h, and finite positive TimesFM q10, q50 and q90 at step h; the h = 56 set can
   therefore differ from the h = 28 set. Counts are reported per horizon.
6. **S2 baselines and estimand.** ETS and drift are `tfm3lab.baselines.ets_forecast` and
   `drift_forecast`, fit on the identical scaled context (<= 512 values) at horizon 28.
   Decisions without an origin quote are counted (`n_no_origin`; none is expected by
   eligibility). The primary estimand covers every sampled decision with an origin quote
   and a valid TimesFM path.
7. **Bootstrap units.** The calendar-block bootstrap is circular over the daily calendar
   spanned by the origin dates (S2) or event dates (S3); the cluster bootstrap resamples
   whole card identities. All bootstraps use seed 20261010 and percentile intervals.
8. **Quantiles and signs.** Coverage uses q10 and q90 from the same TimesFM call as the
   q50 point forecast. All S2 comparisons, including P_TFM5, are on advantage A
   (positive = cheaper than buying now).
