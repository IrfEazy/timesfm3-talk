"""Claim specs for the committed v2 results (column names only; no result values).

Defaults: 3 decimals for skill/coverage/effects, integers for counts. Latency
seconds-per-1k use 4 decimals (an override of the 2-decimal default: the fastest
cells are ~0.001 s and would otherwise render as ``0.00``).
"""

from __future__ import annotations

from .ledger import B, ClaimSpec, F, I, T, when

# Sources deliberately without a manifest match: source path -> reason.
EXEMPTIONS: dict[str, str] = {}

GA = "A: MTG preregistered inference"
GB = "B: Shock event study"
GC = "C: Calibration and conformal"
GD = "D: Covariates"
GL = "L: Latency"
GS = "S: Inference self-check"

# Row-level NaN conditions, from each table's own status / count columns.
NOT_OK = ("no_calibration_rows", "insufficient_calibration")  # calibration.py statuses
_NO_SHOCK_OBS = (when(n_pre=0), when(n_post=0))  # event study with no pre/post observations
_NO_EVAL = when(eval_n=0)  # conformal: nothing to evaluate
_NO_CONFORMAL = (when(status=NOT_OK), _NO_EVAL)  # no usable threshold, or nothing to evaluate

_CI = {"ci_low": F("ci_low"), "ci_high": F("ci_high")}


def _shock(prefix: str, source: str, expected: int) -> ClaimSpec:
    return ClaimSpec(
        id=f"b.{prefix}.{{source}}.{{forecaster}}.{{perturbation}}.{{metric}}",
        source=source,
        group=GB,
        expand=("source", "forecaster", "perturbation", "metric"),
        expected_claims=expected,
        fields={
            "value": F("effect", nan_if=_NO_SHOCK_OBS),
            "ci_low": F("ci_low", nan_if=_NO_SHOCK_OBS),
            "ci_high": F("ci_high", nan_if=_NO_SHOCK_OBS),
            "n_pre": I("n_pre"),
            "n": I("n_post"),
            "verdict": T("verdict"),
        },
    )


def _latency(prefix: str, source: str, expected: int) -> ClaimSpec:
    return ClaimSpec(
        id=f"l.{prefix}.{{method}}.{{device}}.b{{batch}}.c{{context}}.h{{horizon}}",
        source=source,
        group=GL,
        expand=("method", "device", "batch", "context", "horizon"),
        expected_claims=expected,
        fields={
            "value": F("median_s_per_1k", 4),
            "p95_s_per_1k": F("p95_s_per_1k", 4),
            "status": T("status"),
        },
    )


_SIG_KEYS = (
    "mode",
    "transform",
    "make_positive",
    "context_len",
    "baseline_method",
    "requested_horizon",
)

SPECS: tuple[ClaimSpec, ...] = (
    ClaimSpec(
        id="a.verdict.{subset}.h{horizon}.{statistic}",
        source="results/exp_mtg_v2_verdict.parquet",
        group=GA,
        expand=("subset", "horizon", "statistic"),
        expected_claims=18,
        fields={
            "value": F("estimate"),
            **_CI,
            "cal_ci_low": F("cal_ci_low"),
            "cal_ci_high": F("cal_ci_high"),
            "verdict": T("verdict"),
            "n": I("n_cards"),
            "n_rows": I("n_rows"),
            "is_decision_statistic": B("is_decision_statistic"),
        },
    ),
    ClaimSpec(
        id="a.secondary.{comparison}.h{horizon}.{statistic}",
        source="results/exp_mtg_v2_secondary.parquet",
        group=GA,
        expand=("comparison", "horizon", "statistic"),
        expected_claims=30,
        fields={
            "value": F("estimate"),
            **_CI,
            "verdict": T("verdict"),
            "n": I("n_cards"),
            "n_rows": I("n_rows", nan_if=when(comparison="log1p_vs_naive")),
        },
    ),
    ClaimSpec(
        id="a.sig.{mode}.{transform}.pos{make_positive}.c{context_len}.{baseline_method}.h{requested_horizon}",
        source="results/exp_mtg_v2_significance.parquet",
        group=GA,
        expand=_SIG_KEYS,
        expected_claims=960,
        counts={
            "bh_rejections": {"dm_reject_bh": True},
            "by_rejections": {"dm_reject_by": True},
            "n": {},
        },
        unique={"family_size": I("family_size"), "family_n_nan": I("family_n_nan")},
    ),
    ClaimSpec(
        id="c.coverage.{regime}.h{horizon}",
        source="results/exp_calibration_v2_coverage.parquet",
        group=GC,
        expand=("regime", "horizon"),
        expected_claims=10,
        fields={
            "value": F("coverage", nan_if=when(n=0)),
            "cb_low": F("cb_low", nan_if=when(n=0)),
            "cb_high": F("cb_high", nan_if=when(n=0)),
            "cluster_low": F("cluster_low", nan_if=when(n=0)),
            "cluster_high": F("cluster_high", nan_if=when(n=0)),
            "wilson_low": F("wilson_low", nan_if=when(n=0)),
            "wilson_high": F("wilson_high", nan_if=when(n=0)),
            "n": I("n"),
            "n_clusters": I("n_clusters"),
            "under_covers": B("under_covers"),
        },
    ),
    ClaimSpec(
        id="c.conformal.{analysis}.{regime}.h{horizon}",
        source="results/exp_calibration_v2_conformal.parquet",
        group=GC,
        expand=("analysis", "regime", "horizon"),
        expected_claims=14,
        fields={
            "status": T("status"),
            "value": F("coverage_after", nan_if=_NO_CONFORMAL),
            "cb_low": F("cb_low_after", nan_if=_NO_CONFORMAL),
            "cb_high": F("cb_high_after", nan_if=_NO_CONFORMAL),
            "before": F("coverage_before", nan_if=_NO_EVAL),
            "before_cb_low": F("cb_low_before", nan_if=_NO_EVAL),
            "before_cb_high": F("cb_high_before", nan_if=_NO_EVAL),
            "median_rel_width_increase": F("median_rel_width_increase", nan_if=_NO_CONFORMAL),
            "calibration_n": I("calibration_n"),
            "n": I("eval_n"),
            "conformal_fixes_it": B("conformal_fixes_it", nan_if=_NO_CONFORMAL),
        },
    ),
    _shock("tfm3", "results/exp_shock_v2_summary.parquet", 43),
    _shock("tfm25", "results/exp_shock_v2_summary_2p5.parquet", 21),
    ClaimSpec(
        id="d.cov.{domain}.{comparison}.h{horizon}.{statistic}",
        source="results/exp_covariates_v2_summary.parquet",
        group=GD,
        expand=("domain", "comparison", "horizon", "statistic"),
        expected_claims=24,
        fields={
            "value": F("skill_delta"),
            **_CI,
            "verdict": T("verdict"),
            "label": T("label"),
            "n": I("n_clusters"),
            "is_primary": B("is_primary"),
        },
    ),
    _latency("tfm3", "results/latency_v2.parquet", 48),
    _latency("tfm25", "results/latency_v2_2p5.parquet", 48),
    ClaimSpec(
        id="s.selfcheck.{scenario}.g{n_clusters}.{statistic}",
        source="results/inference_selfcheck.parquet",
        group=GS,
        expand=("scenario", "n_clusters", "statistic"),
        expected_claims=18,
        fields={
            "value": F("coverage"),
            "mc_ci_low": F("mc_ci_low"),
            "mc_ci_high": F("mc_ci_high"),
            "n": I("n_datasets"),
        },
    ),
)
