# Inference self-check (synthetic coverage)

Script: `scripts/08_inference_selfcheck.py`. Output: `results/inference_selfcheck.parquet` and
`results/manifests/inference-selfcheck-<as_of>.json`. Rules come from Preregistration v2 section 1
(`docs/analysis-plan.md`).

## Purpose

Before any real data, check how often the nominal 95% PRIMARY interval (paired whole-card cluster
bootstrap, `tfm3lab.bootstrap.paired_cluster_bootstrap_from_sums`, B = 2000, seed 20261010) contains
the true value of the statistic it targets, on simulated panels with known skill.

## Estimands

The interval targets the population value of the statistic computed from per-card REALIZED ratios of
sums, `r_i = sum|e_model| / sum|e_baseline|` over the card's finite history:

- median skill = 1 - median(r_i)
- n-weighted mean skill = 1 - E[n_i r_i] / E[n_i]
- pooled skill = 1 - E[model sum] / E[baseline sum] (secondary, never a decision input)

The truth is a large Monte Carlo of these quantities under the DGP (`TRUTH_PANELS` x `TRUTH_CARDS`
cards, fresh common shock per panel; the size is stored in `truth_n_cards`). An earlier version used
`1 - E[rho]` (the expected per-observation ratio), which is a different quantity: `r_i` carries
finite-history noise and the ratio of sums is not the ratio of expectations. The Monte Carlo value
replaces it.

## Data-generating processes

Per dataset: G cards on a common calendar of 365 days; card history length uniform in [120, 365];
`rho_i = exp(mu + sigma xi_i)`; baseline error `scale_i |z_it|`; model error
`rho_i scale_i |z'_it| exp(gamma f_t)`, with `z = sqrt(lambda) f + sqrt(1-lambda) eta`,
`z' = c z + sqrt(1-c^2) u`, and `f, eta, u` unit-variance AR(1) processes.
`mu = log 0.95` in both scenarios.

| Constant | baseline | stress |
|---|---|---|
| sigma (skill log-sd) | 0.15 | 0.4 |
| scale log-sd | 0.5 | 1.0 |
| AR(1) phi | 0.6 | 0.9 |
| common-shock share lambda | 0.3 | 0.7 |
| model/baseline corr c | 0.7 | 0.3 |
| innovations | Gaussian | Student-t, df 3 (card-level noise; common factor Gaussian) |
| gamma (shock moves relative skill) | 0 | 0.3 |
| burn-in | 50 | 100 |

The baseline scenario is the only one used for acceptance. The stress scenario is a sensitivity
report: a shock that affects model and baseline differently, weak model/baseline correlation, heavy
tails, strong persistence and wide heterogeneity. Constants live in `BASELINE` and `STRESS` in the
script.

## Disclosure of design history

- Baseline constants were chosen after a 40-dataset preview (groups 8 and 40) that showed
  undercoverage under an earlier setting. They were not tuned after that. The baseline therefore is
  not a clean pre-registered DGP; the verdict line says so, and the manifest carries
  `dgp_tuned_after_preview: true`.
- Stress constants were written into the code before its first run. That first run (discarded)
  exposed a defect, not a result to tune against: a Student-t common factor inside `exp(gamma f)` has
  an infinite mean, so the truth was unstable. The fix made only the common factor Gaussian; no
  numeric constant changed.
- The implementer's report does not state whether that discarded run's coverage output was inspected; treat it as seen.
- No real data was used.

## Acceptance rule

At G = 40, baseline scenario, the empirical coverage of BOTH decision statistics (median and
n-weighted mean skill) must lie in [0.92, 0.98]. G = 8 and 20 are reported only; fewer than 20
clusters is labelled small-cluster/exploratory. Pooled skill is reported only. If the rule fails,
every reported CI carries the simulated coverage and is labelled "approximate". The script prints
`PASS`/`FAIL` for this rule and a separate `STRESS (sensitivity, not acceptance): ...` line.

## Reading the results

Read `results/inference_selfcheck.parquet` (one row per scenario x G x statistic). Columns:
`scenario`, `n_clusters`, `statistic`, `used_for_acceptance`, `coverage`, `n_covered`,
`mc_ci_low`/`mc_ci_high` (Wilson 95% interval for the Monte Carlo coverage proportion), `mc_se`,
`true_value` (Monte Carlo truth), `mean_estimate`, `mean_ci_width`, `n_invalid_replicates`,
`n_boot`, `seed`. The manifest repeats the table plus `verdict_line`, `stress_line`,
`design_provenance` and the scenario constants. Compare `coverage` with the Monte Carlo interval,
not with 0.95 alone. Numbers are not repeated here; the committed parquet and manifest are the
source. Coverage below nominal in the stress scenario indicates that, with a common shock that is
shared by all cards and moves relative skill, the cluster bootstrap understates uncertainty about a
population value; the calendar-block sensitivity is the design that targets time uncertainty.
