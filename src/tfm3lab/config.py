"""Paths, seeds, and the one constant the whole project's thesis rests on.

Every other module imports from here rather than hardcoding a path or a
cutoff date, so there is exactly one place to change environment (local
vs. Colab) or update a fact if Google revises the model card.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# --- Reproducibility --------------------------------------------------------

SEED = 42

# --- Paths -------------------------------------------------------------------
#
# TFM3LAB_DATA_ROOT lets the same scripts run unmodified locally (data/ under
# the repo) and on Colab (e.g. a Google Drive mount), per the plan's "one
# codebase, hybrid execution" design.

REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT_OVERRIDE = os.environ.get("TFM3LAB_DATA_ROOT")
DATA_ROOT = Path(_DATA_ROOT_OVERRIDE or REPO_ROOT / "data").resolve()
RAW_DIR = DATA_ROOT / "raw"
CACHE_DIR = DATA_ROOT / "cache"
# Same override as DATA_ROOT: on Colab with TFM3LAB_DATA_ROOT pointed at a
# Drive mount, results survive a runtime disconnect too, not just the raw
# data cache. Without the override (plain local dev), stays under the repo
# so `results/` keeps being the thing you commit.
RESULTS_DIR = (
    Path(_DATA_ROOT_OVERRIDE).resolve() / "results"
    if _DATA_ROOT_OVERRIDE
    else REPO_ROOT / "results"
)
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (RAW_DIR, CACHE_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- TimesFM-3 pretraining cutoff -------------------------------------------
#
# From the model card (huggingface.co/google/timesfm-3.0-pytorch), the
# pretraining corpus includes GiftEvalPretrain (minus series overlapping
# fev-bench), Wikipedia Pageviews (cutoff Nov 2023), Google Trends top
# queries (cutoff end of 2022), plus synthetic/augmented data. We use the
# LATER of the two dated corpora as the conservative "could have been seen"
# boundary for macro/market series: any public daily series up to this date
# is a candidate for contamination and must not be used to claim zero-shot
# generalization without the pre/post-cutoff comparison this project runs.
#
# This is deliberately conservative, not a certified exact date — GIFT-Eval's
# own pretraining split is undated and not fully disclosed. Treat it as
# "here be dragons," not a hard proof of contamination on either side.
PRETRAIN_CUTOFF = date(2023, 11, 30)

# TimesFM-3 gated checkpoint. Requires accepting the non-commercial license
# on Hugging Face and either `hf auth login` once, or an HF_TOKEN env var —
# see README.md.
CHECKPOINT_ID = "google/timesfm-3.0-pytorch"

# From timesfm3._ModelConfig / TimesFM3Forecaster.global_context: the true
# context cap is 15360 (a round "16k" in marketing copy is ceil(15360/32)*32
# rounded loosely) — cite the exact number in the talk, it's more convincing.
MAX_CONTEXT_LENGTH = 15360
INPUT_PATCH_LENGTH = 32
OUTPUT_PATCH_LENGTH = 64  # one non-autoregressive decode covers this many steps
N_QUANTILES = 9
MEDIAN_QUANTILE_INDEX = 4
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass(frozen=True)
class Event:
    """One shock event, anchoring a pre/post-cutoff comparison arm."""

    name: str
    date: date
    arm: str  # "pre_cutoff" | "post_cutoff"


# Anchors for scripts/03_exp_shock.py. Event *detection* in that script is
# automatic (z-scored log-return spikes) — these are the known reference
# dates used to validate the detector and to label the two arms, not a
# substitute for it (docs/talk-outline.md explains why).
KNOWN_EVENTS = (
    Event("Crollo Covid", date(2020, 3, 16), "pre_cutoff"),
    Event("Invasione Ucraina", date(2022, 2, 24), "pre_cutoff"),
    Event("Stretta inflazionistica", date(2022, 6, 13), "pre_cutoff"),
    Event("Unwind carry trade yen", date(2024, 8, 5), "post_cutoff"),
    Event("Shock dazi", date(2025, 4, 3), "post_cutoff"),
)
