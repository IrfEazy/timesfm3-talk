"""Unit tests for tfm3lab.benchmark -- pure grid/index math, no forecaster,
no network."""

from __future__ import annotations

import numpy as np
import pytest

from tfm3lab.benchmark import (
    AblationCombo,
    common_origin_set,
    dry_run_report,
    iter_ablation_combos,
    select_placebo_panel,
)
from tfm3lab.benchmark_config import BenchmarkConfig
from tfm3lab.windows import valid_origins


def _cfg(**overrides) -> BenchmarkConfig:
    fields = dict(
        config_id="t",
        context_lengths=(4, 8),
        horizons=(1, 2),
        modes=("univariate", "multivariate"),
        transforms=("raw",),
        make_positive=(True,),
    )
    fields.update(overrides)
    return BenchmarkConfig(**fields)


def test_common_origin_set_matches_valid_origins_at_widest_cell():
    got = common_origin_set(n=30, context_lengths=(4, 8), horizons=(1, 2))
    expected = valid_origins(n=30, context_len=8, horizon=2)
    np.testing.assert_array_equal(got, expected)


def test_common_origin_set_thins_by_stride():
    full = common_origin_set(n=30, context_lengths=(4, 8), horizons=(1, 2), origin_stride=1)
    thinned = common_origin_set(n=30, context_lengths=(4, 8), horizons=(1, 2), origin_stride=3)
    np.testing.assert_array_equal(thinned, full[::3])


def test_common_origin_set_is_valid_for_every_smaller_cell():
    origins = common_origin_set(n=30, context_lengths=(4, 8), horizons=(1, 2))
    origins_set = set(origins.tolist())
    for context_len in (4, 8):
        for horizon in (1, 2):
            smaller = valid_origins(n=30, context_len=context_len, horizon=horizon)
            smaller_valid = set(smaller.tolist())
            assert origins_set <= smaller_valid


def test_common_origin_set_respects_max_origins():
    got = common_origin_set(n=30, context_lengths=(4,), horizons=(1,), max_origins=3)
    assert len(got) == 3


def test_iter_ablation_combos_full_cartesian_product():
    cfg = _cfg(
        context_lengths=(4, 8),
        horizons=(1, 2),
        transforms=("raw", "log1p"),
        make_positive=(True, False),
        modes=("univariate", "multivariate"),
    )
    combos = iter_ablation_combos(cfg, card_pool_size=7)
    assert len(combos) == 2 * 2 * 2 * 2 * 2
    assert all(isinstance(c, AblationCombo) for c in combos)


def test_iter_ablation_combos_skips_placebo_when_pool_too_small():
    cfg = _cfg(modes=("univariate", "multivariate_placebo"), placebo_panel_size=7)
    combos = iter_ablation_combos(cfg, card_pool_size=3)
    assert all(c.mode != "multivariate_placebo" for c in combos)
    assert any(c.mode == "univariate" for c in combos)


def test_iter_ablation_combos_skips_placebo_when_pool_equals_panel_size():
    # Boundary case: sampling a 7-card panel from a 7-card pool returns the
    # WHOLE pool, i.e. a placebo panel identical to the real multivariate
    # panel -- an uninformative comparison, so it must be skipped too.
    cfg = _cfg(modes=("univariate", "multivariate_placebo"), placebo_panel_size=7)
    combos = iter_ablation_combos(cfg, card_pool_size=7)
    assert all(c.mode != "multivariate_placebo" for c in combos)
    assert any(c.mode == "univariate" for c in combos)


def test_dry_run_report_counts_equal_size_placebo_pool_as_skipped():
    cfg = _cfg(modes=("univariate", "multivariate_placebo"), placebo_panel_size=7)
    report = dry_run_report(cfg, n_days=30, card_pool_size=7)
    assert report["n_combos_skipped_placebo_pool_too_small"] > 0
    assert all(c["mode"] != "multivariate_placebo" for c in report["combos"])


def test_iter_ablation_combos_keeps_placebo_when_pool_large_enough():
    cfg = _cfg(modes=("multivariate_placebo",), placebo_panel_size=3)
    combos = iter_ablation_combos(cfg, card_pool_size=5)
    assert len(combos) > 0
    assert all(c.mode == "multivariate_placebo" for c in combos)


def test_select_placebo_panel_deterministic_under_same_seed():
    pool = tuple(f"card_{i}" for i in range(10))
    a = select_placebo_panel(pool, panel_size=4, seed=1)
    b = select_placebo_panel(pool, panel_size=4, seed=1)
    assert a == b
    assert len(a) == 4
    assert set(a) <= set(pool)


def test_select_placebo_panel_different_seeds_can_differ():
    pool = tuple(f"card_{i}" for i in range(20))
    a = select_placebo_panel(pool, panel_size=5, seed=1)
    b = select_placebo_panel(pool, panel_size=5, seed=2)
    assert a != b


def test_select_placebo_panel_raises_when_pool_too_small():
    with pytest.raises(ValueError, match="need >="):
        select_placebo_panel(("a", "b"), panel_size=5, seed=1)


def test_dry_run_report_zero_days_reports_zero_origins_without_raising():
    cfg = _cfg()
    report = dry_run_report(cfg, n_days=0, card_pool_size=7)
    assert report["n_origins"] == 0
    assert report["n_combos"] > 0  # combos are still enumerated, just not runnable yet


def test_dry_run_report_combo_count_and_skip_accounting():
    cfg = _cfg(modes=("univariate", "multivariate_placebo"), placebo_panel_size=7)
    report = dry_run_report(cfg, n_days=30, card_pool_size=3)  # pool too small for placebo
    assert report["n_combos_skipped_placebo_pool_too_small"] > 0
    assert all(c["mode"] != "multivariate_placebo" for c in report["combos"])


def test_dry_run_report_estimated_calls_scale_with_card_pool_for_univariate():
    cfg = _cfg(modes=("univariate",), context_lengths=(4,), horizons=(1,))
    small = dry_run_report(cfg, n_days=30, card_pool_size=2)
    large = dry_run_report(cfg, n_days=30, card_pool_size=8)
    assert large["estimated_predict_batch_calls"] == 4 * small["estimated_predict_batch_calls"]
