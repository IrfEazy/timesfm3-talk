"""Experiment A inference (Preregistration v2 sections 1-2), offline.

Turns the committed 02b outputs (normalized primary export, leaderboard),
the frozen pool CSV and the synthetic coverage self-check into:

- headline verdicts per primary horizon, vs naive (`headline_table`);
- descriptive secondary comparisons with paired cluster CIs
  (`secondary_table`);
- per-(card, horizon) DM p-values with BH (primary) / BY (sensitivity)
  adjustment per comparison family (`significance_frame`).

Statistics and intervals come from tfm3lab.bootstrap and
tfm3lab.summarize; nothing is re-implemented here. No real data is read
at import time and no number is typed in: coverage comes from the
self-check table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import bootstrap
from .summarize import significance_table

PRIMARY_HORIZONS = (28, 56)
BASELINES_SECONDARY = ("seasonal_naive", "drift", "ets")
CONTEXTS_SECONDARY = (64, 128, 256)
PRIMARY_CONTEXT = 512
SMALL_CLUSTER_BELOW = 20
DECISION_STATISTICS = ("median", "weighted_mean")

BEATS = "beats naive"
LOSES = "loses to naive"
INCONCLUSIVE = "inconclusive"

SUBSETS = ("pool", "excl_showcase", "showcase_only")
SUBSET_ROLE = {
    "pool": "verdict",
    "excl_showcase": "sensitivity",
    "showcase_only": "descriptive",
}

NOT_RUN = "not run"
MODEL_COMPARISON_LABEL = "model comparison only, not a cutoff control"
FFILL_REFERENCE_DAYS = 3  # the cached data's own forward-fill limit

N_DEFINITION = (
    "leaderboard n = observed rows on raw prices (no error-scale filter); "
    "may differ from the headline's valid-scale row set"
)
CELL_COLUMNS = ("mode", "context_len", "transform", "make_positive")

SIGNIFICANCE_FAMILY = ("mode", "transform", "make_positive", "context_len", "baseline_method")


# --- verdict ----------------------------------------------------------------


RULE_VERSION = "strict-2026-10-10"
ORIGINAL_RULE_VERSION = "original-v2"


def verdict_from_cis(cis: dict[str, tuple[float, float]]) -> str:
    """ORIGINAL preregistered rule (kept as a labelled sensitivity,
    `verdict_original_rule`) on the PRIMARY (cluster) CIs of the decision
    statistics: 'beats naive' iff every interval lies entirely above 0,
    'loses to naive' iff every one lies entirely below 0, else
    'inconclusive'. A NaN bound is never entirely on either side."""
    lows = [cis[s][0] for s in DECISION_STATISTICS]
    highs = [cis[s][1] for s in DECISION_STATISTICS]
    if all(np.isfinite(lo) and lo > 0 for lo in lows):
        return BEATS
    if all(np.isfinite(hi) and hi < 0 for hi in highs):
        return LOSES
    return INCONCLUSIVE


def verdict_strict(
    primary: dict[str, tuple[float, float]], calendar: dict[str, tuple[float, float]]
) -> str:
    """Rule adopted 2026-10-10 (human decision, before any v2 result): 'beats
    naive' iff the primary cluster CI AND the calendar-block CI, for BOTH
    median and n-weighted mean skill (four intervals), all lie entirely above
    0; 'loses to naive' iff all four lie entirely below 0; else
    'inconclusive'. Strict at 0; a NaN bound in any interval is never entirely
    on either side."""
    cis = [ci for d in (primary, calendar) for s in DECISION_STATISTICS for ci in (d[s],)]
    if all(np.isfinite(lo) and lo > 0 for lo, _ in cis):
        return BEATS
    if all(np.isfinite(hi) and hi < 0 for _, hi in cis):
        return LOSES
    return INCONCLUSIVE


# --- self-check lookup --------------------------------------------------------


def selfcheck_acceptance_failed(selfcheck: pd.DataFrame) -> bool:
    """True iff any acceptance row (baseline scenario at the acceptance G)
    has empirical coverage outside its recorded acceptance band."""
    acc = selfcheck[selfcheck["used_for_acceptance"].astype(bool)]
    if acc.empty:
        return True  # no acceptance evidence at all: cannot claim it passed
    outside = (acc["coverage"] < acc["acceptance_low"]) | (acc["coverage"] > acc["acceptance_high"])
    return bool(outside.any())


def selfcheck_coverage(selfcheck: pd.DataFrame, statistic: str, n_clusters: int) -> dict:
    """Simulated coverage of the primary CI for `statistic` at the G in the
    self-check closest to `n_clusters` (ties go to the larger G): baseline
    and stress scenario coverage, plus the approximate/small-cluster label."""
    rows = selfcheck[selfcheck["statistic"] == statistic]
    out = {
        "selfcheck_g": np.nan,
        "selfcheck_baseline_coverage": np.nan,
        "selfcheck_stress_coverage": np.nan,
        "selfcheck_acceptance_failed": selfcheck_acceptance_failed(selfcheck),
    }
    if rows.empty:
        out["ci_label"] = "no self-check row"
        return out
    gs = np.sort(rows["n_clusters"].unique())
    dist = np.abs(gs - n_clusters)
    g = int(gs[np.flatnonzero(dist == dist.min())[-1]])
    at_g = rows[rows["n_clusters"] == g]
    for scenario in ("baseline", "stress"):
        sel = at_g[at_g["scenario"] == scenario]
        if len(sel):
            out[f"selfcheck_{scenario}_coverage"] = float(sel["coverage"].iloc[0])
    out["selfcheck_g"] = g
    labels = []
    if out["selfcheck_acceptance_failed"]:
        labels.append("approximate")
    if n_clusters < SMALL_CLUSTER_BELOW:
        labels.append("small-cluster/exploratory")
    out["ci_label"] = "; ".join(labels)
    return out


# --- headline ----------------------------------------------------------------


def _scale_valid_column(primary: pd.DataFrame) -> str:
    for c in ("error_scale_valid", "scale_valid"):
        if c in primary.columns:
            return c
    raise ValueError("primary export has neither error_scale_valid nor scale_valid")


def usable_rows(primary: pd.DataFrame, baseline: str = "naive") -> pd.DataFrame:
    """Observed rows with a valid error scale and finite scaled errors."""
    valid_col = _scale_valid_column(primary)
    b_col = f"ae_baseline_{baseline}_scaled"
    for c in ("card", "origin_date", "requested_horizon", "observed", "ae_model_scaled", b_col):
        if c not in primary.columns:
            raise ValueError(f"primary export lacks column {c!r}")
    keep = primary["observed"].astype(bool) & primary[valid_col].astype(bool)
    keep &= np.isfinite(primary["ae_model_scaled"]) & np.isfinite(primary[b_col])
    if "horizon" in primary.columns:
        keep &= primary["horizon"] == primary["requested_horizon"]
    return primary[keep]


def check_single_cell(primary: pd.DataFrame) -> None:
    """The headline assumes the primary export is one cell: a single value of
    each of mode, context_len, transform, make_positive (when present), and
    one row per (card, origin_date, horizon)."""
    for c in CELL_COLUMNS:
        if c in primary.columns:
            vals = sorted(primary[c].dropna().astype(str).unique())
            if len(vals) > 1:
                raise ValueError(
                    f"primary export must hold exactly one cell; column {c!r} has {vals}"
                )
    h_col = "horizon" if "horizon" in primary.columns else "requested_horizon"
    keys = ["card", "origin_date", h_col]
    if "requested_horizon" in primary.columns and h_col != "requested_horizon":
        keys.append("requested_horizon")
    dup = primary.duplicated(keys, keep=False)
    if dup.any():
        raise ValueError(
            f"primary export has {int(dup.sum())} rows with duplicate "
            f"(card, origin_date, horizon) keys; expected one cell with unique rows"
        )


def pool_coverage(primary: pd.DataFrame, pool_cards: set[str] | None) -> dict:
    """Pool CSV vs export card coverage; the missing list is for the manifest."""
    in_export = set(primary["card"].astype(str))
    if pool_cards is None:
        return {"n_pool_csv": np.nan, "n_cards_in_export": len(in_export), "missing": []}
    missing = sorted(set(map(str, pool_cards)) - in_export)
    return {
        "n_pool_csv": len(pool_cards),
        "n_cards_in_export": len(in_export),
        "missing": missing,
    }


def split_subsets(rows: pd.DataFrame, showcase_cards: set[str]) -> dict[str, pd.DataFrame]:
    is_show = rows["card"].astype(str).isin(showcase_cards)
    return {
        "pool": rows,
        "excl_showcase": rows[~is_show],
        "showcase_only": rows[is_show],
    }


def headline_table(
    primary: pd.DataFrame,
    showcase_cards: set[str],
    selfcheck: pd.DataFrame,
    *,
    pool_cards: set[str] | None = None,
    horizons=PRIMARY_HORIZONS,
    baseline: str = "naive",
    n_boot: int = bootstrap.DEFAULT_N_BOOT,
    seed: int = bootstrap.DEFAULT_SEED,
) -> pd.DataFrame:
    """One row per (subset, horizon, statistic). Median and n-weighted mean
    skill carry the primary cluster CI and the calendar-block sensitivity CI;
    pooled skill is a labelled secondary. The verdict (decided on the two
    primary CIs only) is repeated on every row of its (subset, horizon).
    Showcase-only rows carry no verdict but a `descriptive_direction` from the
    same (strict) rule, used for the disagreement flag; `verdict_original_rule`
    keeps the original primary-CI-only rule as a labelled sensitivity. `pool_cards` (labels in the
    pool CSV) adds n_pool_csv / n_cards_in_export / n_pool_cards_missing."""
    check_single_cell(primary)
    cov = pool_coverage(primary, pool_cards)
    rows = usable_rows(primary, baseline)
    b_col = f"ae_baseline_{baseline}_scaled"
    out = []
    for subset, sub in split_subsets(rows, showcase_cards).items():
        for h in horizons:
            cell = sub[sub["requested_horizon"] == h]
            n_cards = int(cell["card"].nunique())
            base = {
                "subset": subset,
                "subset_role": SUBSET_ROLE[subset],
                "horizon": int(h),
                "baseline": baseline,
                "n_cards": n_cards,
                "n_rows": len(cell),
                "n_boot": n_boot,
                "seed": seed,
                "n_pool_csv": cov["n_pool_csv"],
                "n_cards_in_export": cov["n_cards_in_export"],
                "n_pool_cards_missing": len(cov["missing"]),
            }
            if n_cards == 0:
                out.append(
                    {
                        **base,
                        "statistic": "none",
                        "verdict": "no data",
                        "verdict_original_rule": "no data",
                        "rule_version": RULE_VERSION,
                    }
                )
                continue
            m = cell["ae_model_scaled"].to_numpy(float)
            b = cell[b_col].to_numpy(float)
            ids = cell["card"].astype(str).to_numpy()
            prim = bootstrap.paired_cluster_bootstrap(m, b, ids, n_boot=n_boot, seed=seed)
            cal = bootstrap.calendar_block_bootstrap(
                m, b, ids, cell["origin_date"].to_numpy(), int(h), n_boot=n_boot, seed=seed
            )
            prim_cis = {s: (prim[s].ci_low, prim[s].ci_high) for s in DECISION_STATISTICS}
            cal_cis = {s: (cal[s].ci_low, cal[s].ci_high) for s in DECISION_STATISTICS}
            direction = verdict_strict(prim_cis, cal_cis)
            direction_original = verdict_from_cis(prim_cis)
            descriptive = subset == "showcase_only"
            verdict = "descriptive (no verdict)" if descriptive else direction
            verdict_original = "descriptive (no verdict)" if descriptive else direction_original
            for stat in bootstrap.SKILL_STATISTICS:
                p, c = prim[stat], cal[stat]
                row = {
                    **base,
                    "statistic": stat,
                    "is_decision_statistic": stat in DECISION_STATISTICS,
                    "estimate": p.estimate,
                    "ci_low": p.ci_low,
                    "ci_high": p.ci_high,
                    "ci_method": p.method,
                    "n_invalid_replicates": p.n_invalid,
                    "cal_ci_low": c.ci_low,
                    "cal_ci_high": c.ci_high,
                    "cal_block_days": c.block_days,
                    "cal_n_invalid_replicates": c.n_invalid,
                    "cal_n_excluded_card_reps": c.n_excluded_card_reps,
                    "verdict": verdict,
                    "verdict_original_rule": verdict_original,
                    "rule_version": RULE_VERSION,
                    "descriptive_direction": direction if descriptive else None,
                    "descriptive_direction_original_rule": (
                        direction_original if descriptive else None
                    ),
                    "secondary_label": "secondary, not a decision input"
                    if stat == "pooled"
                    else "",
                }
                row.update(selfcheck_coverage(selfcheck, stat, n_cards))
                out.append(row)
    df = pd.DataFrame(out)
    for col in ("descriptive_direction", "descriptive_direction_original_rule"):
        if col not in df.columns:
            df[col] = None
    return flag_disagreements(df)


def flag_disagreements(df: pd.DataFrame) -> pd.DataFrame:
    """Flags (never resolves) disagreement: a subset whose verdict differs
    from the pool verdict at the same horizon (for showcase-only, its
    `descriptive_direction` is compared), and rows where the primary
    and calendar-block intervals differ in sign-significance."""
    df = df.copy()
    pool_v = df[df["subset"] == "pool"].drop_duplicates("horizon").set_index("horizon")["verdict"]
    df["pool_verdict"] = df["horizon"].map(pool_v)
    effective = df["verdict"].where(df["subset"] != "showcase_only", df["descriptive_direction"])
    df["verdict_disagrees_with_pool"] = (
        (df["subset"] != "pool")
        & df["pool_verdict"].notna()
        & (df["verdict"] != NOT_RUN)
        & (effective != df["pool_verdict"])
    )
    if "ci_low" in df.columns:

        def sig(lo, hi):
            return np.where(lo > 0, 1, np.where(hi < 0, -1, 0))

        df["primary_vs_calendar_disagree"] = sig(df["ci_low"], df["ci_high"]) != sig(
            df["cal_ci_low"], df["cal_ci_high"]
        )
    return df


def replace_showcase_rows(
    headline: pd.DataFrame,
    showcase_primary: pd.DataFrame | None,
    selfcheck: pd.DataFrame,
    *,
    horizons=PRIMARY_HORIZONS,
    baseline: str = "naive",
    n_boot: int = bootstrap.DEFAULT_N_BOOT,
    seed: int = bootstrap.DEFAULT_SEED,
) -> pd.DataFrame:
    """showcase_only rows come from the dedicated showcase run (its own
    primary export), not from the showcase cards that happen to be in the
    pool. Without that export the rows say 'not run'. Descriptive only: no
    verdict, a `descriptive_direction` for the disagreement flag."""
    keep = headline[headline["subset"] != "showcase_only"].drop(
        columns=["pool_verdict", "verdict_disagrees_with_pool", "primary_vs_calendar_disagree"],
        errors="ignore",
    )
    if showcase_primary is None:
        stub = [
            {
                "subset": "showcase_only",
                "subset_role": SUBSET_ROLE["showcase_only"],
                "horizon": int(h),
                "baseline": baseline,
                "statistic": "none",
                "verdict": NOT_RUN,
                "verdict_original_rule": NOT_RUN,
                "rule_version": RULE_VERSION,
                "descriptive_direction": NOT_RUN,
                "descriptive_direction_original_rule": NOT_RUN,
                "n_cards": 0,
                "n_rows": 0,
            }
            for h in horizons
        ]
        new = pd.DataFrame(stub)
    else:
        cards = set(showcase_primary["card"].astype(str))
        full = headline_table(
            showcase_primary, cards, selfcheck, horizons=horizons, baseline=baseline,
            n_boot=n_boot, seed=seed,
        )  # fmt: skip
        new = full[full["subset"] == "pool"].copy()
        direction = new["verdict"]
        direction_original = new["verdict_original_rule"]
        new["subset"] = "showcase_only"
        new["subset_role"] = SUBSET_ROLE["showcase_only"]
        new["descriptive_direction"] = direction
        new["descriptive_direction_original_rule"] = direction_original
        new["verdict"] = "descriptive (no verdict)"
        new["verdict_original_rule"] = "descriptive (no verdict)"
        new = new.drop(
            columns=["pool_verdict", "verdict_disagrees_with_pool", "primary_vs_calendar_disagree"],
            errors="ignore",
        )
        no_data = new["descriptive_direction"] == "no data"
        new.loc[no_data, "verdict"] = "no data"
        new.loc[no_data, "verdict_original_rule"] = "no data"
    for col in ("descriptive_direction", "descriptive_direction_original_rule"):
        if col not in keep.columns:
            keep = keep.assign(**{col: None})
    return flag_disagreements(pd.concat([keep, new], ignore_index=True))


# --- secondary ---------------------------------------------------------------


def _leaderboard_rows(lb: pd.DataFrame, model_prefix: str) -> pd.DataFrame:
    need = {
        "baseline_method",
        "mode",
        "transform",
        "make_positive",
        "context_len",
        "requested_horizon",
        "horizon_step",
        "series",
        "n",
        "mae_model",
        "mae_baseline",
    }
    missing = need - set(lb.columns)
    if missing:
        raise ValueError(f"leaderboard lacks columns {sorted(missing)}")
    out = lb[lb["horizon_step"] == lb["requested_horizon"]].copy()
    out = out[out["mode"].astype(str).str.startswith(model_prefix)]
    out["series"] = out["series"].astype(str)
    out["loss_model"] = out["mae_model"].astype(float) * out["n"].astype(float)
    out["loss_baseline"] = out["mae_baseline"].astype(float) * out["n"].astype(float)
    return out


def _mode_is(lb: pd.DataFrame, suffix: str) -> pd.Series:
    return lb["mode"].astype(str).str.endswith("_" + suffix)


def _cell(lb, *, suffix, transform, make_positive, context, h, baseline="naive") -> pd.DataFrame:
    sel = (
        _mode_is(lb, suffix)
        & (lb["transform"] == transform)
        & (lb["make_positive"].astype(bool) == make_positive)
        & (lb["context_len"] == context)
        & (lb["requested_horizon"] == h)
        & (lb["baseline_method"] == baseline)
    )
    return lb[sel].drop_duplicates("series").set_index("series")


def _paired_ci_rows(name, kind, h, model_loss, base_loss, n_obs, n_boot, seed, extra) -> list[dict]:
    res = bootstrap.paired_cluster_bootstrap_from_sums(
        model_loss, base_loss, n_obs, n_boot=n_boot, seed=seed
    )
    rows = []
    for stat in bootstrap.SKILL_STATISTICS:
        r = res[stat]
        rows.append(
            {
                "comparison": name,
                "kind": kind,
                "horizon": int(h),
                "statistic": stat,
                "estimate": r.estimate,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "ci_method": r.method,
                "n_cards": r.n_clusters,
                "n_invalid_replicates": r.n_invalid,
                "n_boot": n_boot,
                "seed": seed,
                "n_definition": N_DEFINITION,
                "verdict": "descriptive (no verdict)",
                "secondary_label": ("secondary, raw-price loss sums" if stat == "pooled" else ""),
                **extra,
            }
        )
    return rows


def _pair_cells(model_cell: pd.DataFrame, ref_cell: pd.DataFrame):
    """Cards present in both cells with the same observed count n (paired
    cluster bootstrap needs identical row sets). Returns the matched cards,
    their loss arrays, and how many cards were dropped for n mismatch."""
    common = model_cell.index.intersection(ref_cell.index)
    same_n = model_cell.loc[common, "n"].to_numpy() == ref_cell.loc[common, "n"].to_numpy()
    cards = common[same_n]
    return cards, int((~same_n).sum()), len(model_cell) - len(common), len(ref_cell) - len(common)


def secondary_table(
    leaderboard: pd.DataFrame,
    *,
    horizons=PRIMARY_HORIZONS,
    model_prefix: str = "timesfm3",
    n_boot: int = bootstrap.DEFAULT_N_BOOT,
    seed: int = bootstrap.DEFAULT_SEED,
) -> pd.DataFrame:
    """Descriptive secondary comparisons with paired cluster-bootstrap CIs
    from per-card loss sums (mae x n). No verdicts.

    - other baselines: primary cell vs seasonal_naive / drift / ets;
    - variants (context 64/128/256, log1p, make_positive False,
      multivariate): vs naive, and paired vs the primary cell (positive
      skill = the variant is better than the primary cell);
    - multivariate vs multivariate_placebo, paired on the cards both panels
      contain.
    Comparisons whose cells are absent from the leaderboard are skipped.

    Note on n: the leaderboard's n counts observed rows on raw prices with no
    error-scale filter, and loss sums are mae x n. This can differ from the
    headline's valid-scale row set; every output row records this in
    `n_definition`."""
    lb = _leaderboard_rows(leaderboard, model_prefix)
    rows: list[dict] = []

    def emit(name, kind, h, model_cell, ref_cell, loss_col_ref, extra=None):
        cards, n_mismatch, only_model, only_ref = _pair_cells(model_cell, ref_cell)
        if len(cards) == 0:
            return
        m = model_cell.loc[cards, "loss_model"].to_numpy(float)
        r = ref_cell.loc[cards, loss_col_ref].to_numpy(float)
        n = model_cell.loc[cards, "n"].to_numpy(float)
        info = {
            "n_cards_dropped_n_mismatch": n_mismatch,
            "n_cards_only_in_model": only_model,
            "n_cards_only_in_reference": only_ref,
            **(extra or {}),
        }
        rows.extend(_paired_ci_rows(name, kind, h, m, r, n, n_boot, seed, info))

    for h in horizons:
        prim = _cell(
            lb,
            suffix="univariate",
            transform="raw",
            make_positive=True,
            context=PRIMARY_CONTEXT,
            h=h,
        )
        for b in BASELINES_SECONDARY:
            ref = _cell(
                lb,
                suffix="univariate",
                transform="raw",
                make_positive=True,
                context=PRIMARY_CONTEXT,
                h=h,
                baseline=b,
            )
            if len(prim) and len(ref):
                emit(
                    f"primary_vs_{b}", "vs_baseline", h, ref, ref, "loss_baseline", {"reference": b}
                )
        variants = [
            (
                f"context_{c}",
                dict(suffix="univariate", transform="raw", make_positive=True, context=c),
            )
            for c in CONTEXTS_SECONDARY
        ]
        variants += [
            (
                "log1p",
                dict(
                    suffix="univariate",
                    transform="log1p",
                    make_positive=True,
                    context=PRIMARY_CONTEXT,
                ),
            ),
            (
                "make_positive_false",
                dict(
                    suffix="univariate",
                    transform="raw",
                    make_positive=False,
                    context=PRIMARY_CONTEXT,
                ),
            ),
            (
                "multivariate",
                dict(
                    suffix="multivariate",
                    transform="raw",
                    make_positive=True,
                    context=PRIMARY_CONTEXT,
                ),
            ),
        ]
        for name, spec in variants:
            cell = _cell(lb, h=h, **spec)
            if not len(cell):
                continue
            emit(
                f"{name}_vs_naive",
                "vs_baseline",
                h,
                cell,
                cell,
                "loss_baseline",
                {"reference": "naive"},
            )
            if len(prim):
                emit(
                    f"{name}_vs_primary",
                    "paired_model",
                    h,
                    cell,
                    prim,
                    "loss_model",
                    {"reference": "primary"},
                )
        mv = _cell(
            lb,
            h=h,
            suffix="multivariate",
            transform="raw",
            make_positive=True,
            context=PRIMARY_CONTEXT,
        )
        pl = _cell(
            lb,
            h=h,
            suffix="multivariate_placebo",
            transform="raw",
            make_positive=True,
            context=PRIMARY_CONTEXT,
        )
        if len(mv) and len(pl):
            emit(
                "multivariate_vs_placebo",
                "paired_model",
                h,
                mv,
                pl,
                "loss_model",
                {"reference": "multivariate_placebo"},
            )
    cols = ["comparison", "kind", "horizon", "statistic"]
    return (
        pd.DataFrame(rows).sort_values(cols, kind="mergesort").reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=cols)
    )


def model_comparison_table(
    lb_2p5: pd.DataFrame,
    lb_3: pd.DataFrame,
    *,
    horizons=PRIMARY_HORIZONS,
    n_boot: int = bootstrap.DEFAULT_N_BOOT,
    seed: int = bootstrap.DEFAULT_SEED,
) -> pd.DataFrame:
    """TimesFM-2.5 vs TimesFM-3 on the primary cell, paired by card (loss
    sums mae x n; cards with unequal n are dropped and counted). Positive
    skill = TimesFM-2.5 has the lower loss. Model comparison ONLY: there is
    no sourced 2.5 cutoff, so this is never a cutoff control."""
    a = _leaderboard_rows(lb_2p5, "timesfm2p5")
    b = _leaderboard_rows(lb_3, "timesfm3")
    spec = dict(suffix="univariate", transform="raw", make_positive=True, context=PRIMARY_CONTEXT)
    rows: list[dict] = []
    for h in horizons:
        ca, cb = _cell(a, h=h, **spec), _cell(b, h=h, **spec)
        cards, n_mismatch, only_a, only_b = _pair_cells(ca, cb)
        if len(cards) == 0:
            continue
        rows.extend(
            _paired_ci_rows(
                "timesfm2p5_vs_timesfm3",
                "model_comparison",
                h,
                ca.loc[cards, "loss_model"].to_numpy(float),
                cb.loc[cards, "loss_model"].to_numpy(float),
                ca.loc[cards, "n"].to_numpy(float),
                n_boot,
                seed,
                {
                    "reference": "timesfm3",
                    "comparison_label": MODEL_COMPARISON_LABEL,
                    "n_cards_dropped_n_mismatch": n_mismatch,
                    "n_cards_only_in_model": only_a,
                    "n_cards_only_in_reference": only_b,
                },
            )
        )
    return pd.DataFrame(rows)


def ffill_sensitivity_table(
    ffill_primary: dict[int, pd.DataFrame],
    primary: pd.DataFrame,
    *,
    horizons=PRIMARY_HORIZONS,
    n_boot: int = bootstrap.DEFAULT_N_BOOT,
    seed: int = bootstrap.DEFAULT_SEED,
) -> pd.DataFrame:
    """Forward-fill 0/1 vs 3 days on the primary cell (descriptive). For each
    k in `ffill_primary` (k -> primary export of the ffill_days=k run) and
    horizon: (a) skill vs naive on the run's own rows; (b) skill of the
    k-day run's model errors against the 3-day primary export's model errors
    on the (card, origin, horizon) rows usable in both (positive = fewer
    forward-filled days is better). The ffill run drops origins whose
    context keeps a gap, so (b) is restricted to the common origins."""
    rows: list[dict] = []
    ref = usable_rows(primary)
    for k, frame in sorted(ffill_primary.items()):
        own = usable_rows(frame)
        for h in horizons:
            cell = own[own["requested_horizon"] == h]
            if cell.empty:
                continue
            common = _paired_rows(cell, ref[ref["requested_horizon"] == h])
            n_ref = int((ref["requested_horizon"] == h).sum())
            for name, kind, m, b, ids, n_rows in (
                (
                    f"ffill{k}_vs_naive",
                    "vs_baseline",
                    cell["ae_model_scaled"],
                    cell["ae_baseline_naive_scaled"],
                    cell["card"],
                    len(cell),
                ),
                (
                    f"ffill{k}_vs_ffill{FFILL_REFERENCE_DAYS}",
                    "paired_ffill",
                    common[0]["ae_model_scaled"],
                    common[1]["ae_model_scaled"],
                    common[0]["card"],
                    len(common[0]),
                ),
            ):
                if n_rows == 0:
                    continue
                res = bootstrap.paired_cluster_bootstrap(
                    m.to_numpy(float),
                    b.to_numpy(float),
                    ids.astype(str).to_numpy(),
                    n_boot=n_boot,
                    seed=seed,
                )
                for stat in bootstrap.SKILL_STATISTICS:
                    r = res[stat]
                    rows.append(
                        {
                            "comparison": name,
                            "kind": kind,
                            "horizon": int(h),
                            "statistic": stat,
                            "estimate": r.estimate,
                            "ci_low": r.ci_low,
                            "ci_high": r.ci_high,
                            "ci_method": r.method,
                            "n_cards": r.n_clusters,
                            "n_rows": int(n_rows),
                            "n_rows_primary_reference": n_ref,
                            # share of the 3-day primary's usable rows that this
                            # run's (card, origin) set also covers; < 1 means the
                            # rows differ from the primary and are not directly
                            # comparable to it
                            "origin_overlap_fraction": (
                                len(common[0]) / n_ref if n_ref else np.nan
                            ),
                            "n_invalid_replicates": r.n_invalid,
                            "n_boot": n_boot,
                            "seed": seed,
                            "ffill_days": int(k),
                            "reference": "naive" if kind == "vs_baseline" else "ffill3",
                            "verdict": "descriptive (no verdict)",
                            "secondary_label": "secondary, descriptive",
                        }
                    )
    return pd.DataFrame(rows)


def _paired_rows(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows of `a` and `b` on their common (card, origin_date) keys, aligned."""
    key = ["card", "origin_date"]
    ka = a.set_index(key)
    kb = b.set_index(key)
    idx = ka.index.intersection(kb.index)
    return ka.loc[idx].reset_index(), kb.loc[idx].reset_index()


# --- significance -------------------------------------------------------------


def significance_frame(leaderboard: pd.DataFrame, q: float = 0.05) -> pd.DataFrame:
    """DM p-values per (card, horizon) with BH (primary) and BY
    (sensitivity) adjustment per comparison family: one family = all
    card x horizon tests of one model-vs-baseline comparison."""
    lb = leaderboard[leaderboard["horizon_step"] == leaderboard["requested_horizon"]]
    if "dm_pvalue" not in lb.columns:
        raise ValueError("leaderboard lacks dm_pvalue")
    keep = [
        c
        for c in (
            *SIGNIFICANCE_FAMILY,
            "series",
            "requested_horizon",
            "n",
            "dm_stat",
            "dm_pvalue",
            "relative_mae_vs_baseline",
            "skill_vs_baseline",
        )
        if c in lb.columns
    ]
    out = significance_table(lb[keep].reset_index(drop=True), family_cols=SIGNIFICANCE_FAMILY, q=q)
    out["q"] = q
    return out
