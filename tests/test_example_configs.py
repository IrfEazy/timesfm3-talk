"""Regression guard: the shipped example config/card-manifest files must
stay loadable by the schemas they document. No network."""

from __future__ import annotations

from pathlib import Path

from tfm3lab.benchmark_config import load_benchmark_config
from tfm3lab.data.mtg import load_card_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_example_benchmark_config_loads_and_matches_the_analysis_plan():
    cfg = load_benchmark_config(REPO_ROOT / "configs" / "benchmark_preregistered.example.json")
    assert cfg.context_lengths == (64, 128, 256, 512)
    assert cfg.horizons == (1, 7, 28, 56, 64)
    assert cfg.primary_horizons == (28, 56)


def test_example_card_manifest_loads():
    cards = load_card_manifest(REPO_ROOT / "configs" / "benchmark_cards.example.csv")
    assert len(cards) == 7
    labels = {c.label for c in cards}
    assert "Ragavan [MH2]" in labels
