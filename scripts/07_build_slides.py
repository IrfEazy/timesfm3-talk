#!/usr/bin/env python3
"""Injects real numbers from results/*.parquet into slides/talk.md.template,
writing slides/talk.md. Every placeholder is filled from a file in
results/ or left as [MANCA: ...] — never a hand-typed guess.

This script does NOT render HTML/PDF: that needs Marp CLI (`npx
@marp-team/marp-cli`) or the Marp VS Code extension, neither of which is
assumed to be available in every environment this repo runs in. Run it
yourself once slides/talk.md looks right:

    npx @marp-team/marp-cli slides/talk.md -o slides/talk.html
    npx @marp-team/marp-cli slides/talk.md -o slides/talk.pdf

Usage: uv run scripts/07_build_slides.py
"""

from __future__ import annotations

import re

import pandas as pd

from tfm3lab import config, figdata

TEMPLATE_PATH = config.REPO_ROOT / "slides" / "talk.md.template"
OUTPUT_PATH = config.REPO_ROOT / "slides" / "talk.md"


def _missing(placeholder: str) -> str:
    return f"[MANCA: {placeholder} — esegui gli script 02-05]"


def build_values() -> dict[str, str]:
    values: dict[str, str] = {}

    # --- Esperimento A -----------------------------------------------------
    # MASE used to be printed here (10.77 vs 10.21) — dropped. Its scale is
    # a one-step in-sample scale applied unchanged across 28 horizons (MASE
    # rises 1.54 -> 17.95 mechanically with h), and the cross-series mean is
    # dominated by Mishra's Factory (near-flat first 64 days, implied scale
    # 0.00032). Relative MAE per horizon is the honest, interpretable number.
    mtg_acc_path = config.RESULTS_DIR / "exp_mtg_accuracy.parquet"
    mtg_cal_path = config.RESULTS_DIR / "exp_mtg_calibration.parquet"
    if mtg_acc_path.exists() and mtg_cal_path.exists():
        accuracy = pd.read_parquet(mtg_acc_path)
        calibration = pd.read_parquet(mtg_cal_path)
        profile = figdata.horizon_profile(accuracy, calibration)
        h1 = profile[profile["horizon_step"] == profile["horizon_step"].min()].iloc[0]
        h_last = profile[profile["horizon_step"] == profile["horizon_step"].max()].iloc[0]
        values["mtg_relative_mae_h1"] = f"{h1['relative_mae_mean']:.3f}"
        values["mtg_relative_mae_hlast"] = f"{h_last['relative_mae_mean']:.3f}"
        values["mtg_hlast"] = str(int(h_last["horizon_step"]))

        # A DM p-value MEAN (the old {{mtg_dm_pvalue}}) is not a p-value and
        # pointed the wrong way (0.190 read as "no difference" while the
        # model actually loses significantly in most cells). Report the
        # fraction of significant cells in each direction instead.
        identity_uni = accuracy[
            (accuracy["transform"] == "identity") & (accuracy["mode"] == "timesfm3_univariate")
        ]
        dm_valid = identity_uni.dropna(subset=["dm_pvalue"])
        if len(dm_valid):
            significant = dm_valid[dm_valid["dm_pvalue"] < 0.05]
            worse = (significant["dm_stat"] > 0).sum() / len(dm_valid) * 100
            better = (significant["dm_stat"] < 0).sum() / len(dm_valid) * 100
            values["mtg_dm_significant_worse_pct"] = f"{worse:.0f}"
            values["mtg_dm_significant_better_pct"] = f"{better:.0f}"
        else:
            values["mtg_dm_significant_worse_pct"] = _missing("mtg_dm_significant_worse_pct")
            values["mtg_dm_significant_better_pct"] = _missing("mtg_dm_significant_better_pct")

        multi = accuracy[
            (accuracy["transform"] == "identity") & (accuracy["mode"] == "timesfm3_multivariate")
        ]
        values["mtg_relative_mae_multivariate"] = f"{multi['relative_mae_vs_baseline'].mean():.3f}"
    else:
        for key in (
            "mtg_relative_mae_h1",
            "mtg_relative_mae_hlast",
            "mtg_hlast",
            "mtg_dm_significant_worse_pct",
            "mtg_dm_significant_better_pct",
            "mtg_relative_mae_multivariate",
        ):
            values[key] = _missing(key)

    # --- Esperimento B -------------------------------------------------------
    # Filtered to mode=="timesfm3_multivariate": the unfiltered version pools
    # univariate and multivariate lags together (pre-cutoff 2.0 instead of
    # 1.67), silently disagreeing with the multivariate-only panel shown
    # alongside it in the demo and in exp_shock_adaptation_dots.png.
    lag_path = config.RESULTS_DIR / "exp_shock_adaptation_lag.parquet"
    if lag_path.exists():
        df = pd.read_parquet(lag_path)
        at_headline = df[(df["multiplier"] == 1.5) & (df["mode"] == "timesfm3_multivariate")]
        by_arm = at_headline.groupby("arm")["adaptation_lag_days"].mean()
        values["lag_pre_cutoff"] = f"{by_arm.get('pre_cutoff', float('nan')):.1f}"
        values["lag_post_cutoff"] = f"{by_arm.get('post_cutoff', float('nan')):.1f}"
    else:
        values["lag_pre_cutoff"] = _missing("lag_pre_cutoff")
        values["lag_post_cutoff"] = _missing("lag_post_cutoff")

    # --- Esperimento C ---------------------------------------------------
    # scripts/04_exp_calibration.py now writes three same-horizon (h=1)
    # regimes instead of pooling MTG's 28 horizons into "calm" against
    # market's h=1 "shock" — that pooling made shock look better calibrated
    # than calm, the opposite of the truth (see that script's docstring).
    cal_path = config.RESULTS_DIR / "exp_calibration_summary.parquet"
    if cal_path.exists():
        df = pd.read_parquet(cal_path).set_index("regime")
        for key, regime in (
            ("coverage_market_calm", "market_calm"),
            ("coverage_market_shock", "market_shock"),
        ):
            values[key] = (
                f"{df.loc[regime, 'coverage_p10_p90']:.3f}" if regime in df.index else _missing(key)
            )
    else:
        values["coverage_market_calm"] = _missing("coverage_market_calm")
        values["coverage_market_shock"] = _missing("coverage_market_shock")

    # --- Esperimento D -----------------------------------------------------
    # Reframed: the negative control (leaking the actual future price as a
    # covariate) did not visibly fire with a constant-past covariate — see
    # scripts/05_exp_covariates.py's leakage_demo for the fix and the
    # leaked_flat_past control arm kept alongside it. Report whichever
    # result the current results/ actually contains, never a hand-typed
    # assumption that the control worked.
    leak_path = config.RESULTS_DIR / "exp_covariates_leakage_demo.parquet"
    if leak_path.exists():
        from tfm3lab.metrics import mae

        df = pd.read_parquet(leak_path)
        if "arm" in df.columns:
            clean = df[df["arm"] == "clean"]
            leaked = df[df["arm"] == "leaked"]
        else:
            clean, leaked = df[~df["leaked"]], df[df["leaked"]]
        mae_clean = mae(clean["actual"], clean["forecast"])
        mae_leaked = mae(leaked["actual"], leaked["forecast"])
        values["leakage_mae_clean"] = f"{mae_clean:.4f}"
        values["leakage_mae_leaked"] = f"{mae_leaked:.4f}"
        if mae_leaked < mae_clean:
            ratio = mae_leaked / mae_clean
            values["leakage_summary"] = (
                f"controllo negativo confermato: il MAE crolla al **{ratio:.0%}** del "
                "pulito quando il prezzo futuro reale filtra nella covariata"
            )
        else:
            values["leakage_summary"] = (
                "controllo negativo **non si è ancora acceso** (MAE con leak >= MAE "
                "pulito) — risultato inconcludente, non un successo del modello"
            )
    else:
        values["leakage_mae_clean"] = _missing("leakage_mae_clean")
        values["leakage_mae_leaked"] = _missing("leakage_mae_leaked")
        values["leakage_summary"] = _missing("leakage_summary")

    return values


def main() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    values = build_values()

    missing = [k for k, v in values.items() if v.startswith("[MANCA")]
    if missing:
        print(f"Warning: {len(missing)} placeholder(s) not backed by results yet: {missing}")

    def substitute(m: re.Match) -> str:
        key = m.group(1)
        return values.get(key, _missing(key))

    rendered = re.sub(r"\{\{(\w+)\}\}", substitute, template)
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print("Render it with: npx @marp-team/marp-cli slides/talk.md -o slides/talk.html")


if __name__ == "__main__":
    main()
