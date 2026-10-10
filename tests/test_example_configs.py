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


def _v2_expected_fields() -> dict:
    return {
        "context_lengths": (64, 128, 256, 512),
        "horizons": (1, 7, 28, 56, 64),
        "primary_horizons": (28, 56),
        "origin_stride": 7,
        "max_origins": None,
        "cards": "configs/benchmark_pool_v2.csv",
        "transforms": ("raw", "log1p"),
        "make_positive": (True, False),
        "placebo_panel_size": 7,
        "placebo_seed": 42,
        "season_length": 7,
        "baselines": ("naive", "seasonal_naive", "drift", "ets"),
    }


def test_preregistered_v2_config_is_frozen_as_specified():
    # Checks the JSON fields only; configs/benchmark_pool_v2.csv need not exist.
    cfg = load_benchmark_config(REPO_ROOT / "configs" / "benchmark_preregistered_v2.json")
    assert cfg.config_id == "preregistered_v2"
    assert cfg.modes == ("univariate", "multivariate", "multivariate_placebo")
    for key, value in _v2_expected_fields().items():
        assert getattr(cfg, key) == value, key
    assert "Preregistration v2" in cfg.description and "5ed53db" in cfg.description


def test_v2_timesfm2p5_config_is_univariate_twin():
    cfg = load_benchmark_config(REPO_ROOT / "configs" / "benchmark_v2_timesfm2p5.json")
    assert cfg.config_id == "preregistered_v2_timesfm2p5"
    assert cfg.modes == ("univariate",)
    for key, value in _v2_expected_fields().items():
        assert getattr(cfg, key) == value, key
    assert "Preregistration v2" in cfg.description and "5ed53db" in cfg.description
