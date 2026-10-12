"""Claim specs for the committed v2 results (column names only; no result values).

Defaults: 3 decimals for skill/coverage/effects, integers for counts. Latency
seconds-per-1k use 4 decimals (an override of the 2-decimal default: the fastest
cells are ~0.001 s and would otherwise render as ``0.00``).
"""

from __future__ import annotations

from .ledger import LST, B, ClaimSpec, D, F, I, T, when

# Sources deliberately without a manifest match: source path -> reason.
_CFG = "configs/benchmark_preregistered_v2.json"
_FETCH = "results/manifests/fetch-2026-09-04-20261010T160638Z.json"
_EVENTS = "results/manifests/shock-events-v2-2026-09-04.json"
_SELFCHECK = "results/manifests/inference-selfcheck-2026-10-10.json"
_CODE_CFG = "src/tfm3lab/config.py"
_CAL = "src/tfm3lab/calibration.py"
EXEMPTIONS: dict[str, str] = {
    _CFG: "frozen committed configuration (no run manifest lists configs); sha256 pinned here",
    _FETCH: "committed run manifest used as a metadata source; sha256 pinned here",
    _EVENTS: "committed run manifest used as a metadata source; sha256 pinned here",
    _SELFCHECK: "committed run manifest used as a metadata source; sha256 pinned here",
    _CODE_CFG: "committed source constants (cutoff, checkpoint ids); sha256 pinned here",
    _CAL: "committed source constants (nominal interval level); sha256 pinned here",
}

GA = "A: MTG preregistered inference"
GB = "B: Shock event study"
GC = "C: Calibration and conformal"
GD = "D: Covariates"
GL = "L: Latency"
GS = "S: Inference self-check"
GM = "M: Declared metadata"

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
            "reasons": T("reasons"),
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
            "gpu": T("env_gpu_name"),
            "batch": I("batch"),
            "context": I("context"),
            "horizon": I("horizon"),
            "repeats": I("repeats"),
            "warmup": I("warmup"),
        },
    )


def _lb(prefix: str, source: str, h: int, expected: int, select: dict, expand: tuple) -> ClaimSpec:
    """Descriptive leaderboard point estimates at requested horizon == horizon step."""
    return ClaimSpec(
        id=prefix.format(h=h),
        source=source,
        group=GA,
        select={"requested_horizon": h, "horizon_step": h, **select},
        expand=expand,
        expected_claims=expected,
        fields={
            "value": F("skill_median"),
            "weighted": F("skill_mean_weighted"),
            "n": I("n_cards"),
            "complete": B("summary_complete"),
        },
    )


_LB3 = "results/exp_mtg_benchmark_leaderboard_aggregate.parquet"
_LB25 = "results/exp_mtg_benchmark_leaderboard_aggregate__preregistered_v2_timesfm2p5.parquet"
_LB3_ID = (
    "a.lb.{{mode}}.{{transform}}.pos{{make_positive}}.c{{context_len}}.{{baseline_method}}.h{h}"
)
_LB3_KEYS = ("mode", "transform", "make_positive", "context_len", "baseline_method")
_LB25_ID = "a.lb25.{{transform}}.pos{{make_positive}}.c{{context_len}}.h{h}"
_LB25_KEYS = ("transform", "make_positive", "context_len")


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
            "verdict_original_rule": T("verdict_original_rule"),
            "rule_version": T("rule_version"),
            "primary_vs_calendar_disagree": B("primary_vs_calendar_disagree"),
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
            "label": T("secondary_label", nan_if=when(statistic="median")),
            "ffill_days": I("ffill_days", nan_if=when(comparison=("log1p_vs_naive",))),
        },
    ),
    _lb(_LB3_ID, _LB3, 28, 192, {}, _LB3_KEYS),
    _lb(_LB3_ID, _LB3, 56, 192, {}, _LB3_KEYS),
    _lb(_LB25_ID, _LB25, 28, 16, {"baseline_method": "naive"}, _LB25_KEYS),
    _lb(_LB25_ID, _LB25, 56, 16, {"baseline_method": "naive"}, _LB25_KEYS),
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
            "block_days": I("block_days"),
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
            "under_covers_before": B("under_covers_before"),
            "label": T("label"),
            "split_date": D(
                "split_date", nan_if=when(regime=("market_shock", "event_window_nonshock"))
            ),
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
            "acceptance_low": F("acceptance_low", 2),
            "acceptance_high": F("acceptance_high", 2),
            "clusters": I("n_clusters"),
            "used_for_acceptance": B("used_for_acceptance"),
            "label": T("label"),
        },
    ),
    # ---- declared metadata: committed configs / manifests / constants -------------
    ClaimSpec(
        id="m.code",
        source=_CODE_CFG,
        group=GM,
        fields={
            "pretrain_cutoff": D("PRETRAIN_CUTOFF"),
            "checkpoint_tfm3": T("CHECKPOINT_ID"),
            "checkpoint_tfm25": T("CHECKPOINT_ID_2P5"),
        },
    ),
    ClaimSpec(
        id="m.calibration",
        source=_CAL,
        group=GM,
        fields={"nominal": F("NOMINAL", 2)},
    ),
    ClaimSpec(
        id="m.grid",
        source=_CFG,
        group=GM,
        fields={
            "config_id": T("config_id"),
            "horizons": LST("horizons"),
            "h_grid_1": I("horizons.0"),
            "h_grid_2": I("horizons.1"),
            "h_grid_3": I("horizons.2"),
            "h_grid_4": I("horizons.3"),
            "h_grid_5": I("horizons.4"),
            "primary_horizons": LST("primary_horizons"),
            "h_primary_1": I("primary_horizons.0"),
            "h_primary_2": I("primary_horizons.1"),
            "context_lengths": LST("context_lengths"),
            "context_1": I("context_lengths.0"),
            "context_2": I("context_lengths.1"),
            "context_3": I("context_lengths.2"),
            "context_4": I("context_lengths.3"),
            "origin_stride": I("origin_stride"),
            "placebo_panel_size": I("placebo_panel_size"),
            "placebo_seed": I("placebo_seed"),
            "season_length": I("season_length"),
        },
    ),
    ClaimSpec(
        id="m.fetch",
        source=_FETCH,
        group=GM,
        fields={
            "as_of": T("_meta.as_of"),
            "window_start": T("date_range.start"),
            "window_end": T("date_range.end"),
            "missing_date": T("missing_dates.0"),
            "missing_count": I("missing_dates_count"),
        },
    ),
    ClaimSpec(
        id="m.events",
        source=_EVENTS,
        group=GM,
        fields={
            "z_threshold": F("rule.z_threshold", 1),
            "trailing_window": I("rule.trailing_window"),
            "decluster_gap": I("rule.decluster_gap_trading_days"),
            "pre_end": T("rule.pre_end"),
            "post_start": T("rule.post_start"),
            "context_len": I("rule.context_len"),
            "window_before": I("rule.window_before"),
            "window_after": I("rule.window_after"),
            "n_pre": I("counts.used_by_arm.pre"),
            "n_post": I("counts.used_by_arm.post"),
        },
    ),
    ClaimSpec(
        id="m.inference",
        source="results/exp_mtg_v2_verdict.parquet",
        group=GM,
        select={"subset": "pool", "horizon": 28, "statistic": "median"},
        fields={
            "n_boot": I("n_boot"),
            "seed": I("seed"),
            "n_cards": I("n_cards_in_export"),
            "n_pool": I("n_pool_csv"),
            "n_missing": I("n_pool_cards_missing"),
            "rule_version": T("rule_version"),
            "selfcheck_g": I("selfcheck_g"),
        },
    ),
    ClaimSpec(
        id="m.covariates",
        source="results/exp_covariates_v2_summary.parquet",
        group=GM,
        select={
            "domain": "market",
            "comparison": "legit_vs_no_covariate",
            "horizon": "7",
            "statistic": "median",
        },
        fields={
            "context_len": I("context_len"),
            "origin_stride": I("origin_stride"),
            "n_boot": I("n_boot"),
            "ci_level": F("ci_level", 2),
        },
    ),
    ClaimSpec(
        id="m.selfcheck",
        source=_SELFCHECK,
        group=GM,
        fields={
            "verdict": T("verdict"),
            "n_datasets": I("settings.n_datasets"),
            "accept_low": F("acceptance.band.0", 2),
            "accept_high": F("acceptance.band.1", 2),
            "accept_clusters": I("acceptance.n_clusters"),
            "tuned_after_preview": B("dgp_tuned_after_preview"),
        },
    ),
)
