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


def test_cell_id_is_stable_from_fields_not_order():
    a = AblationCombo(512, 28, "raw", True, "univariate")
    b = AblationCombo(512, 28, "raw", True, "univariate")
    assert a.cell_id == b.cell_id == "ctx00512-h0028-raw-mp1-univariate"
    cfg = _cfg(context_lengths=(8, 4), horizons=(2, 1))
    reordered = _cfg(context_lengths=(4, 8), horizons=(1, 2))
    ids = {c.cell_id for c in iter_ablation_combos(cfg, 7)}
    assert ids == {c.cell_id for c in iter_ablation_combos(reordered, 7)}
    assert len(ids) == len(iter_ablation_combos(cfg, 7))
    # zero padding: lexical order == numeric order on context length
    assert AblationCombo(64, 1, "raw", True, "univariate").cell_id < a.cell_id


def test_primary_cell_flag():
    assert AblationCombo(512, 1, "raw", True, "univariate").is_primary
    assert not AblationCombo(512, 1, "log1p", True, "univariate").is_primary
    assert not AblationCombo(256, 1, "raw", True, "univariate").is_primary
    assert not AblationCombo(512, 1, "raw", False, "univariate").is_primary
    assert not AblationCombo(512, 1, "raw", True, "multivariate").is_primary


def test_dry_run_report_projects_rows_and_bytes_per_cell():
    cfg = _cfg(
        modes=("univariate", "multivariate_placebo"),
        placebo_panel_size=3,
        context_lengths=(4,),
        horizons=(1, 2),
    )
    report = dry_run_report(cfg, n_days=30, card_pool_size=5)
    n_origins = report["n_origins"]
    by_id = {c["cell_id"]: c for c in report["cells"]}
    assert by_id["ctx00004-h0002-raw-mp1-univariate"]["projected_rows"] == 5 * n_origins * 2
    assert by_id["ctx00004-h0001-raw-mp1-multivariate_placebo"]["projected_rows"] == 3 * n_origins
    per_row = report["bytes_per_row_estimate"]
    for c in report["cells"]:
        assert c["projected_bytes_estimate"] == c["projected_rows"] * per_row
    assert report["projected_rows_total"] == sum(c["projected_rows"] for c in report["cells"])


def test_canonicalize_cell_frame_sorts_and_fills_schema():
    import pandas as pd

    from tests.conftest import ReversedFakeForecaster
    from tfm3lab.backtest import SeriesData, run_univariate_backtest
    from tfm3lab.benchmark import CELL_SCHEMA, canonicalize_cell_frame

    n = 20
    dates = pd.date_range("2025-01-01", periods=n).to_numpy()
    observed = np.ones(n, dtype=bool)
    observed[7] = False  # last context point of origin 8 is unobserved
    series = [
        SeriesData("b", np.arange(1.0, n + 1), dates, observed),
        SeriesData("a", np.arange(2.0, n + 2), dates, np.ones(n, dtype=bool)),
    ]
    combo = AblationCombo(4, 2, "raw", True, "univariate")
    df = run_univariate_backtest(ReversedFakeForecaster(), series, np.array([8, 10]), 4, 2)
    table = canonicalize_cell_frame(df, combo=combo, run_id="r", config_id="c", series_list=series)
    assert table.schema.equals(CELL_SCHEMA)
    out = table.to_pandas()
    keys = list(zip(out["series"], out["origin"], out["horizon_step"], strict=True))
    assert keys == sorted(keys) and len(set(keys)) == len(keys) == 2 * 2 * 2
    assert out["baseline_ets"].dtype == float  # present even if not computed
    b8 = out[(out["series"] == "b") & (out["origin"] == 8)]["context_last_observed"].unique()
    assert b8.tolist() == [7.0]  # value at index 6, the last OBSERVED context point


def test_primary_export_frame_has_no_price_columns():
    import pandas as pd

    from tfm3lab.benchmark import PRIMARY_EXPORT_FORBIDDEN, primary_export_frame

    cell = pd.DataFrame(
        {
            "config_id": ["c"],
            "run_id": ["r"],
            "mode": ["timesfm3_univariate"],
            "context_len": [512],
            "series": ["X"],
            "origin_date": pd.to_datetime(["2025-01-01"]),
            "target_date": pd.to_datetime(["2025-01-02"]),
            "requested_horizon": [28],
            "horizon_step": [28],
            "observed": [True],
            "actual": [12.0],
            "forecast": [9.0],
            "q10": [8.0],
            "q50": [9.0],
            "q90": [11.0],
            "baseline_naive": [10.0],
            "context_last_observed": [10.0],
        }
    )
    out = primary_export_frame(cell, ("naive",), {"X": 10.0, "Y": 10.0})
    assert not (PRIMARY_EXPORT_FORBIDDEN & set(out.columns))
    assert out.loc[0, "y_norm"] == pytest.approx(1.2)
    assert out.loc[0, "ae_model_scaled"] == pytest.approx(0.3)
    assert out.loc[0, "ae_baseline_naive_scaled"] == pytest.approx(0.2)
    assert out.loc[0, "q90_norm"] == pytest.approx(1.1)


def test_primary_export_keeps_only_requested_horizon_and_flags_bad_scale():
    import numpy as np
    import pandas as pd

    from tfm3lab.benchmark import primary_export_frame

    n = 6
    cell = pd.DataFrame(
        {
            "config_id": ["c"] * n,
            "run_id": ["r"] * n,
            "mode": ["timesfm3_univariate"] * n,
            "context_len": [512] * n,
            "series": ["X"] * 3 + ["Y"] * 3,
            "origin_date": pd.to_datetime(["2025-01-01"] * n),
            "target_date": pd.to_datetime(["2025-01-02"] * n),
            "requested_horizon": [3] * n,
            "horizon_step": [1, 2, 3, 1, 2, 3],
            "observed": [True] * n,
            "actual": [12.0] * n,
            "forecast": [9.0] * n,
            "q10": [8.0] * n,
            "q50": [9.0] * n,
            "q90": [11.0] * n,
            "baseline_naive": [10.0] * n,
            "context_last_observed": [10.0, 10.0, 10.0, 0.0, 0.0, np.nan],
        }
    )
    out = primary_export_frame(cell, ("naive",), {"X": 10.0, "Y": 10.0}).reset_index(drop=True)
    assert out["horizon"].tolist() == [3, 3]  # one row per series/origin, at the requested step
    assert out["scale_valid"].tolist() == [True, False]
    norm = [c for c in out.columns if c.endswith("_norm") or c.endswith("_scaled")]
    assert not np.isinf(out[norm].to_numpy(dtype=float)).any()
    assert out.loc[1, ["y_norm", "q10_norm", "q50_norm", "q90_norm"]].isna().all()
    assert out["error_scale_valid"].tolist() == [True, True]
    bad = primary_export_frame(cell, ("naive",), {"X": 10.0, "Y": float("nan")})
    assert bad["error_scale_valid"].tolist() == [True, False]
    assert bad.loc[1, ["ae_model_scaled", "ae_baseline_naive_scaled"]].isna().all()


def test_primary_export_skill_equals_raw_ratio_per_card():
    import numpy as np
    import pandas as pd

    from tfm3lab.benchmark import PRIMARY_EXPORT_FORBIDDEN, primary_export_frame

    rng = np.random.default_rng(1)
    rows = []
    for card, level in (("A", 3.0), ("B", 40.0)):
        for _o in range(6):
            last = level * rng.uniform(0.5, 1.5)  # varies per origin
            actual = level * rng.uniform(0.5, 1.5)
            rows.append(
                {
                    "config_id": "c",
                    "run_id": "r",
                    "mode": "m",
                    "context_len": 8,
                    "series": card,
                    "origin_date": pd.Timestamp("2025-01-01"),
                    "target_date": pd.Timestamp("2025-01-02"),
                    "requested_horizon": 1,
                    "horizon_step": 1,
                    "observed": True,
                    "actual": actual,
                    "forecast": actual + rng.normal(),
                    "q10": 0.0,
                    "q50": 0.0,
                    "q90": 0.0,
                    "baseline_naive": last,
                    "context_last_observed": last,
                }
            )
    cell = pd.DataFrame(rows)
    out = primary_export_frame(cell, ("naive",), {"A": 2.5, "B": 37.0})
    assert not (PRIMARY_EXPORT_FORBIDDEN & set(out.columns))
    assert not any("price" in c or c == "error_scale" for c in out.columns)
    for card, g in out.groupby("card"):
        raw = cell[cell["series"] == card]
        want = (raw["actual"] - raw["forecast"]).abs().sum() / (
            raw["actual"] - raw["baseline_naive"]
        ).abs().sum()
        got = g["ae_model_scaled"].sum() / g["ae_baseline_naive_scaled"].sum()
        assert got == pytest.approx(want)


def test_refill_series_limits_fill_and_marks_gaps():
    import pandas as pd

    from tfm3lab.backtest import SeriesData
    from tfm3lab.benchmark import (
        filter_kept_rows,
        gap_drop_summary,
        keep_origins_per_card,
        refill_series,
    )

    n = 12
    observed = np.ones(n, dtype=bool)
    observed[[3, 4, 5, 6]] = False  # a 4-day hole
    values = np.arange(1.0, n + 1)
    values[[3, 4, 5, 6]] = 3.0  # cached 3-day fill ... then stale value
    s = SeriesData("a", values, pd.date_range("2025-01-01", periods=n).to_numpy(), observed)
    out, gap = refill_series(s, 2)
    assert gap.tolist() == [False] * 5 + [True, True] + [False] * 5
    assert np.isfinite(out.values).all() and (out.observed == observed).all()
    _, gap0 = refill_series(s, 0)
    assert gap0.sum() == 4
    origins = np.arange(4, 12)
    kept = keep_origins_per_card(origins, {"a": gap, "b": np.zeros(n, dtype=bool)}, context_len=3)
    # origin o has window [o-3, o); gaps at 5,6 -> card a loses origins 6..9; b keeps all
    assert kept["a"].tolist() == [4, 5, 10, 11] and kept["b"].tolist() == origins.tolist()
    summ = gap_drop_summary(origins, kept)
    assert summ["dropped_per_card"] == {"a": 4, "b": 0}
    assert summ["kept_fraction_per_card"] == {"a": 0.5, "b": 1.0}
    assert summ["kept_fraction_overall"] == 0.75 and summ["excluded_cards"] == []
    allgap = keep_origins_per_card(origins, {"c": np.ones(n, dtype=bool)}, context_len=3)
    assert gap_drop_summary(origins, allgap)["excluded_cards"] == ["c"]
    df = pd.DataFrame({"series": ["a", "a", "b", "c"], "origin": [4, 6, 6, 4]})
    assert filter_kept_rows(df, kept).index.tolist() == [0, 2]
