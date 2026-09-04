"""Unit tests for tfm3lab.benchmark_config -- pure dataclass validation and
JSON loading, no network, no other tfm3lab modules involved."""

from __future__ import annotations

import json

import pytest

from tfm3lab.benchmark_config import BenchmarkConfig, load_benchmark_config


def _write(tmp_path, payload, name="config.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_payload(**overrides):
    payload = {
        "config_id": "test_grid",
        "context_lengths": [64, 128],
        "horizons": [1, 7],
    }
    payload.update(overrides)
    return payload


def test_load_benchmark_config_minimal_valid_file(tmp_path):
    path = _write(tmp_path, _valid_payload())
    cfg = load_benchmark_config(path)
    assert cfg.config_id == "test_grid"
    assert cfg.context_lengths == (64, 128)
    assert cfg.horizons == (1, 7)
    # defaults
    assert cfg.transforms == ("raw", "log1p")
    assert cfg.make_positive == (True, False)
    assert cfg.modes == ("univariate", "multivariate")
    assert cfg.origin_stride == 1


def test_load_benchmark_config_full_override(tmp_path):
    path = _write(
        tmp_path,
        _valid_payload(
            primary_horizons=[7],
            origin_stride=2,
            max_origins=50,
            cards="benchmark",
            transforms=["log1p"],
            make_positive=[False],
            modes=["multivariate_placebo"],
            placebo_panel_size=10,
            placebo_seed=7,
            season_length=7,
            baselines=["naive"],
            description="override test",
        ),
    )
    cfg = load_benchmark_config(path)
    assert cfg.primary_horizons == (7,)
    assert cfg.origin_stride == 2
    assert cfg.max_origins == 50
    assert cfg.cards == "benchmark"
    assert cfg.transforms == ("log1p",)
    assert cfg.make_positive == (False,)
    assert cfg.modes == ("multivariate_placebo",)
    assert cfg.placebo_panel_size == 10
    assert cfg.placebo_seed == 7
    assert cfg.season_length == 7
    assert cfg.baselines == ("naive",)
    assert cfg.description == "override test"


@pytest.mark.parametrize("missing_field", ["config_id", "context_lengths", "horizons"])
def test_load_benchmark_config_missing_required_field_raises(tmp_path, missing_field):
    payload = _valid_payload()
    del payload[missing_field]
    path = _write(tmp_path, payload)
    with pytest.raises(ValueError, match=missing_field):
        load_benchmark_config(path)


def test_benchmark_config_rejects_empty_config_id():
    with pytest.raises(ValueError, match="config_id"):
        BenchmarkConfig(config_id="", context_lengths=(64,), horizons=(1,))


def test_benchmark_config_rejects_empty_context_lengths():
    with pytest.raises(ValueError, match="context_lengths"):
        BenchmarkConfig(config_id="t", context_lengths=(), horizons=(1,))


def test_benchmark_config_rejects_empty_horizons():
    with pytest.raises(ValueError, match="horizons"):
        BenchmarkConfig(config_id="t", context_lengths=(64,), horizons=())


def test_benchmark_config_rejects_non_positive_horizon():
    with pytest.raises(ValueError, match="horizons"):
        BenchmarkConfig(config_id="t", context_lengths=(64,), horizons=(0,))


def test_benchmark_config_rejects_non_positive_context_length():
    with pytest.raises(ValueError, match="context_lengths"):
        BenchmarkConfig(config_id="t", context_lengths=(-1,), horizons=(1,))


def test_benchmark_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown modes"):
        BenchmarkConfig(config_id="t", context_lengths=(64,), horizons=(1,), modes=("bogus",))


def test_benchmark_config_rejects_unknown_transform():
    with pytest.raises(ValueError, match="unknown transforms"):
        BenchmarkConfig(config_id="t", context_lengths=(64,), horizons=(1,), transforms=("bogus",))


def test_benchmark_config_rejects_unknown_baseline():
    with pytest.raises(ValueError, match="unknown baselines"):
        BenchmarkConfig(config_id="t", context_lengths=(64,), horizons=(1,), baselines=("bogus",))


def test_benchmark_config_rejects_placebo_panel_size_below_one():
    with pytest.raises(ValueError, match="placebo_panel_size"):
        BenchmarkConfig(config_id="t", context_lengths=(64,), horizons=(1,), placebo_panel_size=0)


def test_benchmark_config_rejects_origin_stride_below_one():
    with pytest.raises(ValueError, match="origin_stride"):
        BenchmarkConfig(config_id="t", context_lengths=(64,), horizons=(1,), origin_stride=0)


def test_benchmark_config_rejects_primary_horizon_not_in_horizons():
    with pytest.raises(ValueError, match="primary_horizons"):
        BenchmarkConfig(
            config_id="t", context_lengths=(64,), horizons=(1, 7), primary_horizons=(28,)
        )
