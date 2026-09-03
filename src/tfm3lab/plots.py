"""Rendering only — every function here takes data already shaped by
figdata.py and returns a matplotlib Axes/ndarray-of-Axes. Nothing here
reads a file, calls plt.show(), or saves anything (see `save` at the
bottom, the one exception, used only by scripts/06_make_figures.py).

Both scripts/06_make_figures.py and notebooks/demo.ipynb import from here,
so the demo and the slides render the same chart from the same code —
there is exactly one place that knows what "the shock reaction chart"
looks like.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

PALETTE = {
    "actual": "#111827",
    "model": "#2563eb",
    "baseline": "#94a3b8",
    "pre": "#f59e0b",
    "post": "#2563eb",
    "alert": "#dc2626",
}


def apply_style() -> None:
    """Applies the deck's rcParams once. Idempotent — safe to call from
    both the notebook (once, in the setup cell) and the figure script."""
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


def plot_forecast_slice(
    sl, ax=None, *, reveal: bool = True, show_naive: bool = True, show_band: bool = True
):
    """The hero chart: history up to the cut, then (if `reveal`) the real
    continuation against the model's forecast. `reveal=False` still fixes
    the y-limits to the revealed state, so a live demo's two cells (cut,
    then reveal) don't jump the axes between them.

    Forward-filled points (`sl.history_observed`/`sl.observed_mask` False)
    are overlaid as hollow markers on top of the solid line — never drawn as
    if they were a real print, but also never dropped from the line, so the
    chart doesn't develop a fake gap.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(
        sl.history_dates, sl.history_values, color=PALETTE["actual"], linewidth=1.8,
        label="storico reale",
    )
    history_observed = np.asarray(sl.history_observed, dtype=bool)
    history_dates = np.asarray(sl.history_dates)
    history_values = np.asarray(sl.history_values, dtype=float)
    if not history_observed.all():
        imputed = ~history_observed
        ax.scatter(
            history_dates[imputed], history_values[imputed],
            facecolors="none", edgecolors=PALETTE["baseline"], marker="o", s=30, zorder=3,
            label="imputato (forward-fill)",
        )
    ax.axvline(
        sl.origin_date, color=PALETTE["alert"], linestyle="--", linewidth=1.2, label="taglio"
    )

    all_values = list(sl.history_values)
    if reveal:
        ax.plot(sl.target_dates, sl.actual, color=PALETTE["actual"], linewidth=1.8)
        observed_mask = np.asarray(sl.observed_mask, dtype=bool)
        target_dates = np.asarray(sl.target_dates)
        actual_values = np.asarray(sl.actual, dtype=float)
        if not observed_mask.all():
            imputed = ~observed_mask
            ax.scatter(
                target_dates[imputed], actual_values[imputed],
                facecolors="none", edgecolors=PALETTE["baseline"], marker="o", s=30, zorder=3,
                label="imputato (forward-fill)",
            )
        ax.plot(
            sl.target_dates, sl.forecast, color=PALETTE["model"], marker="o", markersize=4,
            label="mediana TimesFM-3",
        )
        if show_band:
            ax.fill_between(
                sl.target_dates, sl.q10, sl.q90, color=PALETTE["model"], alpha=0.15,
                label="P10-P90",
            )
        if show_naive:
            ax.hlines(
                sl.naive, sl.target_dates.min(), sl.target_dates.max(),
                color=PALETTE["baseline"], linestyle=":", linewidth=1.5,
                label="naive (ultimo prezzo)",
            )
        all_values += list(sl.actual) + list(sl.forecast) + list(sl.q10) + list(sl.q90)
    else:
        all_values += [sl.naive]

    pad = 0.08 * (max(all_values) - min(all_values) + 1e-9)
    ax.set_ylim(min(all_values) - pad, max(all_values) + pad)
    ax.set_title(f"{sl.series} — taglio {pd.Timestamp(sl.origin_date).date()}")
    # Both imputed-point scatters (history and target) share the label
    # "imputato (forward-fill)" — matplotlib doesn't dedupe legend entries
    # by label on its own, so collapse to one entry per unique label
    # regardless of how many artists (zero, one, or both scatters) fire.
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=9)
    if reveal:
        ax.annotate(
            f"copertura P10-P90: {sl.coverage:.2f}   relative MAE: {sl.relative_mae:.3f}",
            xy=(0.02, 0.02), xycoords="axes fraction", fontsize=9, color="#374151",
        )
    return ax


def plot_shock_reaction(
    sub: pd.DataFrame,
    axes=None,
    *,
    show_naive: bool = True,
    threshold: float | None = None,
    title: str = "",
):
    """One event's one-step-ahead reaction: real vs forecast (top), error
    (bottom). `show_naive` overlays the flat baseline — the one line that
    reveals how close the "forecast" sits to just-repeat-yesterday.
    """
    sub = sub.sort_values("offset")
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    ax0, ax1 = axes

    ax0.plot(sub["offset"], sub["actual"], color=PALETTE["actual"], linewidth=2, label="reale")
    ax0.plot(
        sub["offset"], sub["forecast"], color=PALETTE["model"], marker="o", markersize=3,
        label="TimesFM-3 (one-step)",
    )
    if "q10" in sub.columns and "q90" in sub.columns:
        ax0.fill_between(
            sub["offset"], sub["q10"], sub["q90"], color=PALETTE["model"], alpha=0.15,
            label="P10-P90",
        )
    if show_naive and "baseline_naive" in sub.columns:
        ax0.plot(
            sub["offset"], sub["baseline_naive"], color=PALETTE["baseline"], linestyle="--",
            linewidth=1.5, label="naive",
        )
    ax0.axvline(0, color=PALETTE["alert"], linestyle="--", label="evento")
    ax0.set_title(title or (str(sub["event"].iloc[0]) if "event" in sub.columns else title))
    ax0.legend(fontsize=8)

    ax1.plot(
        sub["offset"], sub["abs_pct_error"], marker="o", color=PALETTE["alert"], label="modello"
    )
    if show_naive and "baseline_naive" in sub.columns:
        naive_err = (
            (sub["actual"] - sub["baseline_naive"]).abs()
            / sub["actual"].abs().clip(lower=1e-8)
            * 100
        )
        ax1.plot(sub["offset"], naive_err, color=PALETTE["baseline"], linestyle="--", label="naive")
    if threshold is not None:
        ax1.axhline(
            threshold, color="#374151", linestyle=":", linewidth=1.2, label="soglia adaptation lag"
        )
    ax1.axvline(0, color="black", linestyle="--")
    ax1.set_xlabel("offset rispetto all'evento (giorni)")
    ax1.set_ylabel("errore % assoluto")
    ax1.legend(fontsize=8)
    return axes


def plot_horizon_profile(profile: pd.DataFrame, axes=None):
    """Two panels sharing horizon_step: relative MAE (top, ref line at 1.0)
    and P10-P90 coverage (bottom, ref line at 0.80) — the coverage panel is
    the mechanism behind why the calibration curve looks the way it does."""
    profile = profile.sort_values("horizon_step")
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax0, ax1 = axes

    ax0.plot(
        profile["horizon_step"], profile["relative_mae_mean"], color=PALETTE["model"], marker="o"
    )
    ax0.fill_between(
        profile["horizon_step"], profile["relative_mae_min"], profile["relative_mae_max"],
        color=PALETTE["model"], alpha=0.12, label="min-max fra le carte",
    )
    ax0.axhline(1.0, color=PALETTE["baseline"], linestyle="--", label="= naive")
    ax0.set_ylabel("relative MAE")
    ax0.set_title("Esperimento A — relative MAE e copertura per orizzonte")
    ax0.legend(fontsize=9)

    ax1.plot(profile["horizon_step"], profile["coverage_mean"], color=PALETTE["post"], marker="o")
    ax1.axhline(0.80, color=PALETTE["baseline"], linestyle="--", label="nominale 0.80")
    ax1.set_xlabel("horizon_step")
    ax1.set_ylabel("copertura P10-P90")
    ax1.legend(fontsize=9)
    return axes


def plot_quantile_bin_calibration(hist: pd.DataFrame, axes=None):
    """One bar panel per horizon_step in `hist`. Flat at the nominal 1/10
    line = calibrated; a U-shape (mass piling into the outer two bins) =
    intervals too narrow. Bars sit at the 10 discrete quantile bins from
    figdata.quantile_bin_calibration — this is not a continuous-PIT axis."""
    steps = sorted(hist["horizon_step"].unique())
    if axes is None:
        _, axes = plt.subplots(1, len(steps), figsize=(4.2 * len(steps), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, steps, strict=True):
        g = hist[hist["horizon_step"] == h].sort_values("bin_index")
        x = np.arange(len(g))
        ax.bar(x, g["fraction"], width=0.85, color=PALETTE["model"], alpha=0.85)
        nominal = float(g["nominal_fraction"].iloc[0]) if len(g) else 0.1
        ax.axhline(nominal, color=PALETTE["baseline"], linestyle="--", label="uniforme attesa")
        ax.set_xticks(x)
        ax.set_xticklabels(g["label"], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"h={h}")
        ax.set_xlabel("quantile bin")
    axes[0].set_ylabel("frazione")
    axes[0].legend(fontsize=8)
    return axes


# Deprecated alias — see figdata.pit_histogram.
plot_pit_histogram = plot_quantile_bin_calibration


def plot_calibration_curve(curve: pd.DataFrame, ax=None, *, group_col: str = "regime"):
    """Nominal vs empirical coverage, one line per group in `group_col`
    (expects results/exp_calibration_curve.parquet's regimes: market_calm,
    market_shock, mtg — all at the same horizon, never pooled across
    domain or horizon, see scripts/04_exp_calibration.py)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5))
    colors = {
        "market_calm": PALETTE["pre"], "market_shock": PALETTE["alert"], "mtg": PALETTE["post"]
    }
    ax.plot(
        [0, 1], [0, 1], color=PALETTE["baseline"], linestyle="--", label="calibrazione perfetta"
    )
    for key, g in curve.sort_values("nominal_level").groupby(group_col):
        n = int(g["n"].iloc[0])
        ax.plot(
            g["nominal_level"], g["empirical_coverage"], marker="o",
            color=colors.get(key), label=f"{key} (n={n})",
        )
    ax.set_xlabel("livello nominale del quantile")
    ax.set_ylabel("copertura empirica")
    ax.set_title("Esperimento C — calibrazione, stesso orizzonte (h=1)")
    ax.legend(fontsize=9)
    return ax


def plot_adaptation_dots(detail: pd.DataFrame, axes=None):
    """Per-event adaptation lag dots (top) plus the pre-event threshold each
    lag was measured against (bottom) — the second panel is what makes the
    top panel's n=3-vs-n=2 comparison honest instead of misleading."""
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax0, ax1 = axes
    colors = {"pre_cutoff": PALETTE["pre"], "post_cutoff": PALETTE["post"]}
    # Fix one shared category order (arm, then event) so both panels line
    # up under the same x position regardless of groupby iteration order.
    order = detail.sort_values(["arm", "event"])["event"].tolist()

    for arm, g in detail.groupby("arm"):
        n = int(g["n_events_in_arm"].iloc[0])
        ax0.scatter(
            g["event"], g["adaptation_lag_days"], color=colors.get(arm), s=80,
            label=f"{arm} (n={n})",
        )
        ax0.axhline(
            g["adaptation_lag_days"].mean(), color=colors.get(arm), linestyle=":", linewidth=1
        )
    ax0.set_xlim(-0.5, len(order) - 0.5)
    ax0.set_ylabel("adaptation lag (giorni)")
    ax0.set_title("Esperimento B — lag per evento, e la soglia dietro ogni numero")
    ax0.tick_params(axis="x", labelbottom=False)  # labels live on the shared-x bottom panel only
    ax0.legend(fontsize=9)

    for arm, g in detail.groupby("arm"):
        ax1.bar(g["event"], g["threshold"], color=colors.get(arm), alpha=0.7)
    ax1.set_ylabel("soglia (mediana\npre-evento × mult.)")
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels(order, rotation=25, ha="right")
    return axes


def plot_card_relative_mae(accuracy: pd.DataFrame, ax=None):
    """One dot per card: mean relative MAE across horizons (univariate,
    identity transform). Replaces the old per-card x mode bar chart, which
    spent half its ink on a mode difference (1.0610 vs 1.0601) nobody could
    see."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    by_card = (
        accuracy[
            (accuracy["mode"] == "timesfm3_univariate") & (accuracy["transform"] == "identity")
        ]
        .groupby("series")["relative_mae_vs_baseline"]
        .mean()
        .sort_values()
    )
    colors = [PALETTE["model"] if v >= 1.0 else "#16a34a" for v in by_card.to_numpy()]
    ax.scatter(by_card.to_numpy(), range(len(by_card)), color=colors, s=90, zorder=3)
    ax.set_yticks(range(len(by_card)))
    ax.set_yticklabels(by_card.index)
    ax.axvline(1.0, color=PALETTE["baseline"], linestyle="--", label="= naive")
    ax.set_xlabel("relative MAE (media sugli orizzonti)")
    ax.set_title("Esperimento A — relative MAE per carta")
    ax.legend(fontsize=9)
    return ax


def plot_glitch_vignette(truth: pd.DataFrame, glitches: pd.DataFrame, ax=None, *, window: int = 20):
    """Small multiples around each glitch: two different cards spiking on
    the same date is the evidence it's a feed artifact, not a market move."""
    n = len(glitches)
    if ax is None:
        _, axes = plt.subplots(1, max(n, 1), figsize=(4.5 * max(n, 1), 4), sharey=False)
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(ax)
    for ax_i, (_, glitch) in zip(axes, glitches.iterrows(), strict=False):
        g = truth[
            (truth["series"] == glitch["series"])
            & (truth["index"] >= glitch["index"] - window)
            & (truth["index"] <= glitch["index"] + window)
        ].sort_values("index")
        ax_i.plot(g["date"], g["value"], color=PALETTE["actual"], marker=".")
        ax_i.axvline(glitch["date"], color=PALETTE["alert"], linestyle="--")
        ax_i.set_title(f"{glitch['series']}\n{pd.Timestamp(glitch['date']).date()}", fontsize=10)
        ax_i.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
        ax_i.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax_i.tick_params(axis="x", rotation=30)
        for label in ax_i.get_xticklabels():
            label.set_ha("right")
    return axes


def save(fig, name: str, *, directory: Path = None, dpi: int = 200) -> Path:
    directory = directory if directory is not None else config.FIGURES_DIR
    path = Path(directory) / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
