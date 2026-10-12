#!/usr/bin/env python3
"""Regenerates every slide figure from results/*.parquet — nothing here
reads from the model or the network. Safe to run locally, without a GPU,
right after pulling fresh results from a Colab/Kaggle run.

All data-shaping lives in tfm3lab.figdata, all rendering in tfm3lab.plots —
this script is just a registry of (name, builder) pairs plus a loop, so
notebooks/demo.ipynb can import and reuse the exact same functions.

Usage:
    uv run scripts/06_make_figures.py
    uv run scripts/06_make_figures.py --only exp_mtg_forecast_slice

Writes PNGs to results/figures/.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from tfm3lab import config, figdata, plots

plots.apply_style()


def _read(name: str) -> pd.DataFrame | None:
    path = config.RESULTS_DIR / name
    if not path.exists():
        print(f"  skip (missing {name})")
        return None
    return pd.read_parquet(path)


def build_hero_slice() -> None:
    preds, source = figdata.load_mtg_predictions()
    print(f"  (loaded MTG predictions from {source.name})")
    truth = figdata.reconstruct_truth(preds)
    sl = figdata.build_forecast_slice(
        preds, truth, "The One Ring [LTR]", origin_index=238, require_observed_targets=True
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plots.plot_forecast_slice(sl, ax=ax, reveal=True)
    plots.save(fig, "exp_mtg_forecast_slice")


def build_shock_reactions() -> None:
    df = _read("exp_shock_raw_predictions.parquet")
    if df is None:
        return
    lag_df = _read("exp_shock_adaptation_lag.parquet")
    for event in ("Crollo Covid", "Shock dazi"):
        sub = df[
            (df["series"] == "SP500")
            & (df["mode"] == "timesfm3_multivariate")
            & (df["event"] == event)
        ]
        if sub.empty:
            print(f"  skip (no rows for event={event!r})")
            continue
        threshold = None
        if lag_df is not None:
            detail = figdata.adaptation_lag_detail(df, lag_df, series="SP500")
            row = detail[detail["event"] == event]
            if not row.empty:
                threshold = float(row["threshold"].iloc[0])
        fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
        plots.plot_shock_reaction(sub, axes=axes, threshold=threshold, title=event)
        plots.save(fig, f"exp_shock_reaction_{event.replace(' ', '_')}")


def build_horizon_profile() -> None:
    accuracy = _read("exp_mtg_accuracy.parquet")
    calibration = _read("exp_mtg_calibration.parquet")
    if accuracy is None or calibration is None:
        return
    profile = figdata.horizon_profile(accuracy, calibration)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    plots.plot_horizon_profile(profile, axes=axes)
    plots.save(fig, "exp_mtg_horizon_profile")


def build_quantile_bin_calibration() -> None:
    preds, _ = figdata.load_mtg_predictions()
    hist = figdata.quantile_bin_calibration(preds, horizon_steps=(1, 7, 28))
    if hist.empty:
        print("  skip (no rows at horizon steps 1/7/28)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4), sharey=True)
    plots.plot_quantile_bin_calibration(hist, axes=axes)
    plots.save(fig, "exp_mtg_quantile_bin_calibration")


def build_calibration_curve() -> None:
    curve = _read("exp_calibration_curve.parquet")
    if curve is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    plots.plot_calibration_curve(curve, ax=ax)
    plots.save(fig, "exp_calibration_curve")


def build_adaptation_dots() -> None:
    shock = _read("exp_shock_raw_predictions.parquet")
    lag_df = _read("exp_shock_adaptation_lag.parquet")
    if shock is None or lag_df is None:
        return
    detail = figdata.adaptation_lag_detail(shock, lag_df)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5))
    plots.plot_adaptation_dots(detail, axes=axes)
    plots.save(fig, "exp_shock_adaptation_dots")


def build_card_relative_mae() -> None:
    accuracy = _read("exp_mtg_accuracy.parquet")
    if accuracy is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plots.plot_card_relative_mae(accuracy, ax=ax)
    plots.save(fig, "exp_mtg_relative_mae")


def build_glitch_vignette() -> None:
    preds, _ = figdata.load_mtg_predictions()
    truth = figdata.reconstruct_truth(preds)
    glitches = figdata.find_glitches(truth)
    if glitches.empty:
        print("  skip (no glitches detected)")
        return
    fig, axes = plt.subplots(1, len(glitches), figsize=(4.5 * len(glitches), 4))
    plots.plot_glitch_vignette(truth, glitches, ax=axes)
    plots.save(fig, "exp_mtg_data_glitch")


# --- v2 figures: committed v2 results only --------------------------------

V2_WATERMARK = "LEAKED - teaching contrast, not a result"
EXPLORATORY_LABEL = "Exploratory - post-hoc amended split"
RED_TEAM_LABEL = "Exploratory red-team appendix"
FIGURE_METADATA = {"Software": None}  # no version/date strings in the PNGs


def _save_v2(fig, name: str) -> None:
    path = config.FIGURES_DIR / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", metadata=FIGURE_METADATA)
    plt.close(fig)


def _titles(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title if subtitle is None else f"{title}\n{subtitle}", fontsize=11)


def fig_exp_a_skill(skill: pd.DataFrame, cis: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0, color="black", linewidth=1, label="no gain over naive")
    ax.plot(
        skill["horizon_step"], skill["skill_median"], color="#1f77b4", label="median across cards"
    )
    ax.plot(
        skill["horizon_step"],
        skill["skill_mean_weighted"],
        color="#d95f02",
        label="n-weighted mean",
    )
    styles = {"median": ("#1f77b4", -0.4), "weighted_mean": ("#d95f02", 0.4)}
    labelled = False
    for _, r in cis.iterrows():
        color, dx = styles.get(r["statistic"], ("gray", 0.0))
        x = r["horizon"] + dx
        ax.errorbar(
            [x],
            [r["estimate"]],
            yerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]],
            fmt="o",
            color=color,
            capsize=4,
            markeredgecolor="black",
            label=None if labelled else "verdict table (step = h)",
        )
        labelled = True
    ax.set_xlabel("horizon step (days)")
    ax.set_ylabel("skill vs naive (positive = better)")
    _titles(
        ax,
        "Primary cell: TimesFM-3 univariate skill vs naive by horizon",
        "median across cards is noisy step to step",
    )
    ax.legend(loc="lower left", fontsize=9)
    return fig


def fig_exp_c_coverage(cov: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0.80, color="black", linestyle="--", linewidth=1, label="nominal")
    for i, (regime, g) in enumerate(cov.groupby("regime")):
        g = g.sort_values("horizon")
        off = (i - (cov["regime"].nunique() - 1) / 2) * 0.4
        x = g["horizon"].to_numpy(dtype=float) + off
        ax.errorbar(
            x,
            g["coverage"],
            yerr=[g["coverage"] - g["cb_low"], g["cb_high"] - g["coverage"]],
            marker="o",
            capsize=3,
            label=str(regime),
        )
    ax.set_xlabel("horizon step (days)")
    ax.set_ylabel("P10-P90 interval coverage")
    _titles(ax, "Interval coverage by horizon and regime")
    ax.legend(fontsize=9)
    return fig


def fig_exp_c_conformal(conf: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0.80, color="black", linestyle="--", linewidth=1, label="nominal")
    for key, color, dx in (("before", "#7f7f7f", -0.3), ("after", "#1f77b4", 0.3)):
        x = conf["horizon"].to_numpy(dtype=float) + dx
        c = conf[f"coverage_{key}"]
        ax.errorbar(
            x,
            c,
            yerr=[c - conf[f"cb_low_{key}"], conf[f"cb_high_{key}"] - c],
            marker="o",
            capsize=3,
            color=color,
            label=f"{key} conformal",
        )
    ax.set_xlabel("horizon step (days)")
    ax.set_ylabel("P10-P90 interval coverage")
    _titles(ax, "MTG conformal coverage, before and after", EXPLORATORY_LABEL)
    ax.legend(fontsize=9)
    return fig


def fig_exp_b_onset(onset: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    for arm, color in (("pre", "#1f77b4"), ("post", "#d95f02")):
        g = onset[onset["arm"] == arm].sort_values("origin_offset")
        ax.plot(g["origin_offset"], g["median_ape"], color=color, marker="o", ms=3, label=arm)
    ax.axvline(0, color="black", linewidth=1, linestyle=":", label="event onset")
    ax.set_xlabel("origin offset from onset (days)")
    ax.set_ylabel("median one-step APE")
    _titles(ax, "Forecast error around event onset", RED_TEAM_LABEL)
    ax.legend(title="arm", fontsize=9)
    return fig


def fig_leakage_demo(err: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    g = err[err["observed"].astype(bool)]
    ax.plot(g["horizon_step"], g["median_err"], marker="o", ms=3, label="target observed")
    ax.set_xlabel("horizon step (days)")
    ax.set_ylabel("median scaled absolute error")
    _titles(ax, "Covariate leakage demo")
    ax.legend(fontsize=9)
    fig.text(
        0.5,
        0.5,
        V2_WATERMARK,
        fontsize=20,
        color="red",
        alpha=0.35,
        rotation=25,
        ha="center",
        va="center",
        weight="bold",
    )
    return fig


def fig_latency(table: pd.DataFrame, gpu: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, g in table.groupby("method"):
        g = g.sort_values("batch")
        ax.plot(g["batch"], g["median_s_per_1k"], marker="o", label=str(method))
    ax.set_xscale("log")
    ax.set_yscale("log")
    batches = sorted(table["batch"].unique())
    ax.set_xticks(batches, [str(b) for b in batches])
    ax.set_xlabel("batch size")
    ax.set_ylabel("median seconds per 1k forecasts")
    _titles(ax, "Forecast latency by method and batch size", f"GPU: {gpu}")
    ax.legend(title="method", fontsize=9)
    return fig


def build_v2_exp_a() -> None:
    agg = _read("exp_mtg_benchmark_leaderboard_aggregate.parquet")
    verdict = _read("exp_mtg_v2_verdict.parquet")
    if agg is None or verdict is None:
        return
    skill = figdata.exp_a_primary_skill(agg)
    if skill.empty:
        print("  skip (no primary-cell rows)")
        return
    _save_v2(
        fig_exp_a_skill(skill, figdata.exp_a_verdict_cis(verdict)), "v2_exp_a_skill_by_horizon"
    )


def build_v2_exp_c() -> None:
    cov = _read("exp_calibration_v2_coverage.parquet")
    if cov is None:
        return
    _save_v2(fig_exp_c_coverage(figdata.coverage_by_regime(cov)), "v2_exp_c_coverage")


def build_v2_exp_c_conformal() -> None:
    conf = _read("exp_calibration_v2_conformal.parquet")
    if conf is None:
        return
    rows = figdata.conformal_exploratory_mtg(conf)
    if rows.empty:
        print("  skip (no exploratory MTG rows)")
        return
    _save_v2(fig_exp_c_conformal(rows), "v2_exp_c_conformal_mtg_exploratory")


def build_v2_exp_b() -> None:
    shock = _read("exp_shock_v2_predictions.parquet")
    if shock is None:
        return
    onset = figdata.onset_error_by_offset(shock)
    if onset.empty:
        print("  skip (no onset rows)")
        return
    _save_v2(fig_exp_b_onset(onset), "v2_exp_b_error_around_onset")


def build_v2_leakage() -> None:
    demo = _read("exp_covariates_v2_leakage_demo.parquet")
    if demo is None:
        return
    _save_v2(fig_leakage_demo(figdata.leakage_error_by_horizon(demo)), "v2_covariates_leakage_demo")


def build_v2_latency() -> None:
    frames = [
        f for f in (_read("latency_v2.parquet"), _read("latency_v2_2p5.parquet")) if f is not None
    ]
    if not frames:
        return
    table, gpu = figdata.latency_summary(frames)
    if table.empty:
        print("  skip (no ok latency rows)")
        return
    _save_v2(fig_latency(table, gpu), "v2_latency")


FIGURES = [
    ("exp_mtg_forecast_slice", build_hero_slice),
    ("exp_shock_reaction", build_shock_reactions),
    ("exp_mtg_horizon_profile", build_horizon_profile),
    ("exp_mtg_quantile_bin_calibration", build_quantile_bin_calibration),
    ("exp_calibration_curve", build_calibration_curve),
    ("exp_shock_adaptation_dots", build_adaptation_dots),
    ("exp_mtg_relative_mae", build_card_relative_mae),
    ("exp_mtg_data_glitch", build_glitch_vignette),
    ("v2_exp_a_skill_by_horizon", build_v2_exp_a),
    ("v2_exp_c_coverage", build_v2_exp_c),
    ("v2_exp_c_conformal_mtg_exploratory", build_v2_exp_c_conformal),
    ("v2_exp_b_error_around_onset", build_v2_exp_b),
    ("v2_covariates_leakage_demo", build_v2_leakage),
    ("v2_latency", build_v2_latency),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", help="regenerate only the named figure (see the list below)")
    args = parser.parse_args()

    print(f"Writing figures to {config.FIGURES_DIR}")
    names = [n for n, _ in FIGURES]
    if args.only and args.only not in names:
        raise SystemExit(f"unknown figure {args.only!r}. Choices: {names}")

    for name, builder in FIGURES:
        if args.only and name != args.only:
            continue
        print(f"{name}:")
        try:
            builder()
        except Exception as e:  # one bad window must not abort the other figures
            print(f"  ERROR: {e}")
            continue


if __name__ == "__main__":
    main()
