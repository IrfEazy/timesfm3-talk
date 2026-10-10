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

N_DEFINITION = (
    "leaderboard n = observed rows on raw prices (no error-scale filter); "
    "may differ from the headline's valid-scale row set"
)
CELL_COLUMNS = ("mode", "context_len", "transform", "make_positive")

SIGNIFICANCE_FAMILY = ("mode", "transform", "make_positive", "context_len", "baseline_method")


# --- verdict ----------------------------------------------------------------


def verdict_from_cis(cis: dict[str, tuple[float, float]]) -> str:
    """Preregistered rule on the PRIMARY (cluster) CIs of the decision
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
    same rule, used for the disagreement flag. `pool_cards` (labels in the
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
                out.append({**base, "statistic": "none", "verdict": "no data"})
                continue
            m = cell["ae_model_scaled"].to_numpy(float)
            b = cell[b_col].to_numpy(float)
            ids = cell["card"].astype(str).to_numpy()
            prim = bootstrap.paired_cluster_bootstrap(m, b, ids, n_boot=n_boot, seed=seed)
            cal = bootstrap.calendar_block_bootstrap(
                m, b, ids, cell["origin_date"].to_numpy(), int(h), n_boot=n_boot, seed=seed
            )
            direction = verdict_from_cis(
                {s: (prim[s].ci_low, prim[s].ci_high) for s in DECISION_STATISTICS}
            )
            verdict = "descriptive (no verdict)" if subset == "showcase_only" else direction
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
                    "descriptive_direction": direction if subset == "showcase_only" else None,
                    "secondary_label": "secondary, not a decision input"
                    if stat == "pooled"
                    else "",
                }
                row.update(selfcheck_coverage(selfcheck, stat, n_cards))
                out.append(row)
    df = pd.DataFrame(out)
    if "descriptive_direction" not in df.columns:
        df["descriptive_direction"] = None
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
        (df["subset"] != "pool") & df["pool_verdict"].notna() & (effective != df["pool_verdict"])
    )
    if "ci_low" in df.columns:

        def sig(lo, hi):
            return np.where(lo > 0, 1, np.where(hi < 0, -1, 0))

        df["primary_vs_calendar_disagree"] = sig(df["ci_low"], df["ci_high"]) != sig(
            df["cal_ci_low"], df["cal_ci_high"]
        )
    return df


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
