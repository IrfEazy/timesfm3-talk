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

from tfm3lab import config

TEMPLATE_PATH = config.REPO_ROOT / "slides" / "talk.md.template"
OUTPUT_PATH = config.REPO_ROOT / "slides" / "talk.md"


def _missing(placeholder: str) -> str:
    return f"[MANCA: {placeholder} — esegui gli script 02-05]"


def build_values() -> dict[str, str]:
    values: dict[str, str] = {}

    mtg_path = config.RESULTS_DIR / "exp_mtg_accuracy.parquet"
    if mtg_path.exists():
        df = pd.read_parquet(mtg_path)
        identity_uni = df[(df["transform"] == "identity") & (df["mode"] == "timesfm3_univariate")]
        values["mtg_mase_model"] = f"{identity_uni['mase_model'].mean():.3f}"
        values["mtg_mase_naive"] = f"{identity_uni['mase_baseline_naive'].mean():.3f}"
        dm_valid = identity_uni["dm_pvalue"].dropna()
        values["mtg_dm_pvalue"] = (
            f"{dm_valid.mean():.3f}" if len(dm_valid) else _missing("mtg_dm_pvalue")
        )
        multi = df[(df["transform"] == "identity") & (df["mode"] == "timesfm3_multivariate")]
        rel_mae = multi["relative_mae_vs_baseline"].mean()
        values["mtg_relative_mae_multivariate"] = f"{rel_mae:.3f}"
    else:
        mtg_keys = (
            "mtg_mase_model",
            "mtg_mase_naive",
            "mtg_dm_pvalue",
            "mtg_relative_mae_multivariate",
        )
        for key in mtg_keys:
            values[key] = _missing(key)

    lag_path = config.RESULTS_DIR / "exp_shock_adaptation_lag.parquet"
    if lag_path.exists():
        df = pd.read_parquet(lag_path)
        at_headline = df[df["multiplier"] == 1.5]
        by_arm = at_headline.groupby("arm")["adaptation_lag_days"].mean()
        values["lag_pre_cutoff"] = f"{by_arm.get('pre_cutoff', float('nan')):.1f}"
        values["lag_post_cutoff"] = f"{by_arm.get('post_cutoff', float('nan')):.1f}"
    else:
        values["lag_pre_cutoff"] = _missing("lag_pre_cutoff")
        values["lag_post_cutoff"] = _missing("lag_post_cutoff")

    cal_path = config.RESULTS_DIR / "exp_calibration_summary.parquet"
    if cal_path.exists():
        df = pd.read_parquet(cal_path).set_index("regime")
        if "calm" in df.index:
            values["coverage_calm"] = f"{df.loc['calm', 'coverage_p10_p90']:.3f}"
        else:
            values["coverage_calm"] = _missing("coverage_calm")
        if "shock" in df.index:
            values["coverage_shock"] = f"{df.loc['shock', 'coverage_p10_p90']:.3f}"
        else:
            values["coverage_shock"] = _missing("coverage_shock")
    else:
        values["coverage_calm"] = _missing("coverage_calm")
        values["coverage_shock"] = _missing("coverage_shock")

    leak_path = config.RESULTS_DIR / "exp_covariates_leakage_demo.parquet"
    if leak_path.exists():
        from tfm3lab.metrics import mae

        df = pd.read_parquet(leak_path)
        clean, leaked = df[~df["leaked"]], df[df["leaked"]]
        values["leakage_mae_clean"] = f"{mae(clean['actual'], clean['forecast']):.4f}"
        values["leakage_mae_leaked"] = f"{mae(leaked['actual'], leaked['forecast']):.4f}"
    else:
        values["leakage_mae_clean"] = _missing("leakage_mae_clean")
        values["leakage_mae_leaked"] = _missing("leakage_mae_leaked")

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
