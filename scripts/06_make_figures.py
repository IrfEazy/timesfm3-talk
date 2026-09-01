#!/usr/bin/env python3
"""Regenerates every slide figure from results/*.parquet — nothing here
reads from the model or the network. Safe to run locally, without a GPU,
right after pulling fresh results from a Colab run.

Usage: uv run scripts/06_make_figures.py

Writes PNGs to results/figures/.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from tfm3lab import config

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    }
)

COLOR_MODEL = "#2563eb"
COLOR_BASELINE = "#94a3b8"
COLOR_PRE = "#f59e0b"
COLOR_POST = "#2563eb"
COLOR_ACTUAL = "#111827"


def _save(fig, name: str) -> None:
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_mtg_relative_mae() -> None:
    path = config.RESULTS_DIR / "exp_mtg_accuracy.parquet"
    if not path.exists():
        print(f"  skip (missing {path.name})")
        return
    df = pd.read_parquet(path)
    # backtest.IDENTITY_TRANSFORM's own name is "identity" (not "raw" — that's
    # only the human-readable label scripts/02 prints to the console).
    df = df[df["transform"] == "identity"]
    if df.empty:
        print("  skip (no rows with transform == 'identity')")
        return
    summary = df.groupby(["series", "mode"])["relative_mae_vs_baseline"].mean().unstack()

    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="barh", ax=ax, color=[COLOR_MODEL, "#7c3aed"])
    ax.axvline(1.0, color=COLOR_BASELINE, linestyle="--", label="= naive")
    ax.set_xlabel("MAE relativo al naive (< 1 = TimesFM-3 vince)")
    ax.set_title("Esperimento A — Magic: The Gathering, MAE relativo per carta")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, "exp_mtg_relative_mae")


def fig_shock_adaptation_lag() -> None:
    path = config.RESULTS_DIR / "exp_shock_adaptation_lag.parquet"
    if not path.exists():
        print(f"  skip (missing {path.name})")
        return
    df = pd.read_parquet(path)
    df = df[df["mode"] == "timesfm3_multivariate"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for arm, color in (("pre_cutoff", COLOR_PRE), ("post_cutoff", COLOR_POST)):
        # mean across events sharing this arm, one point per lag-multiplier
        by_mult = df[df["arm"] == arm].groupby("multiplier")["adaptation_lag_days"].mean()
        ax.plot(by_mult.index, by_mult.to_numpy(), marker="o", color=color, label=arm)
    ax.set_xlabel("soglia di recupero (x volte l'errore pre-evento)")
    ax.set_ylabel("adaptation lag medio (giorni)")
    ax.set_title("Esperimento B — velocità di riallineamento, pre vs post-cutoff")
    ax.legend()
    _save(fig, "exp_shock_adaptation_lag")


def fig_shock_reaction(event_name: str | None = None) -> None:
    path = config.RESULTS_DIR / "exp_shock_raw_predictions.parquet"
    if not path.exists():
        print(f"  skip (missing {path.name})")
        return
    df = pd.read_parquet(path)
    df = df[(df["series"] == "SP500") & (df["mode"] == "timesfm3_multivariate")]
    if event_name is None:
        event_name = df["event"].iloc[0]
    sub = df[df["event"] == event_name].sort_values("offset")

    # Both panels share "offset" (days from the event) as x — NOT target_date:
    # mixing a date axis and an integer-offset axis under sharex=True corrupts
    # both (matplotlib reinterprets the integers as days-since-epoch dates).
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(sub["offset"], sub["actual"], color=COLOR_ACTUAL, label="reale", linewidth=2)
    axes[0].plot(
        sub["offset"],
        sub["forecast"],
        color=COLOR_MODEL,
        marker="o",
        markersize=3,
        label="TimesFM-3 (one-step)",
    )
    axes[0].fill_between(
        sub["offset"], sub["q10"], sub["q90"], color=COLOR_MODEL, alpha=0.15, label="P10-P90"
    )
    axes[0].axvline(0, color="red", linestyle="--", label="evento")
    axes[0].set_title(f"{event_name} — SP500, previsione one-step")
    axes[0].legend()

    axes[1].plot(sub["offset"], sub["abs_pct_error"], marker="o", color="#dc2626")
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_xlabel("offset rispetto all'evento (giorni)")
    axes[1].set_ylabel("errore percentuale assoluto")
    _save(fig, f"exp_shock_reaction_{event_name.replace(' ', '_')}")


def fig_calibration_curve() -> None:
    path = config.RESULTS_DIR / "exp_calibration_curve.parquet"
    if not path.exists():
        print(f"  skip (missing {path.name})")
        return
    df = pd.read_parquet(path)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], color=COLOR_BASELINE, linestyle="--", label="calibrazione perfetta")
    for regime, color in (("calm", COLOR_PRE), ("shock", "#dc2626")):
        sub = df[df["regime"] == regime].sort_values("nominal_level")
        ax.plot(
            sub["nominal_level"], sub["empirical_coverage"], marker="o", color=color, label=regime
        )
    ax.set_xlabel("livello nominale del quantile")
    ax.set_ylabel("copertura empirica")
    ax.set_title("Esperimento C — calibrazione, regime calmo vs shock")
    ax.legend()
    _save(fig, "exp_calibration_curve")


def fig_covariate_leakage_demo() -> None:
    path = config.RESULTS_DIR / "exp_covariates_leakage_demo.parquet"
    if not path.exists():
        print(f"  skip (missing {path.name})")
        return
    df = pd.read_parquet(path)
    from tfm3lab.metrics import mae

    rows = []
    for horizon_step, group in df.groupby("horizon_step"):
        clean = group[~group["leaked"]]
        leaked = group[group["leaked"]]
        rows.append(
            {
                "horizon_step": horizon_step,
                "clean": mae(clean["actual"], clean["forecast"]),
                "leaked (illecito)": mae(leaked["actual"], leaked["forecast"]),
            }
        )
    summary = pd.DataFrame(rows).set_index("horizon_step")

    fig, ax = plt.subplots(figsize=(8, 5))
    summary.plot(kind="bar", ax=ax, color=[COLOR_MODEL, "#dc2626"])
    ax.set_ylabel("MAE")
    ax.set_title("Esperimento D — cosa succede quando il futuro filtra nella covariata")
    _save(fig, "exp_covariates_leakage_demo")


def main() -> None:
    print(f"Writing figures to {config.FIGURES_DIR}")
    fig_mtg_relative_mae()
    fig_shock_adaptation_lag()
    fig_shock_reaction()
    fig_calibration_curve()
    fig_covariate_leakage_demo()


if __name__ == "__main__":
    main()
