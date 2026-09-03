"""Rendering smoke tests: every plots.py function must run without a
display backend and produce the expected number of artists. These are NOT
a check on values — figdata's tests own that — only on "does it render".
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from tfm3lab import figdata, plots  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _forecast_slice(all_observed: bool = True, *, history_observed=None, target_observed=None):
    # history_observed/target_observed let a test set the two imputed-point
    # scatter calls in plot_forecast_slice independently (e.g. an imputed
    # target with a fully-observed history, or vice versa) — each is the
    # literal "is the last point observed" value (True = no imputed point,
    # False = one imputed point), defaulting to `all_observed` when not
    # given, so existing all-or-nothing callers are unaffected.
    history_observed = all_observed if history_observed is None else history_observed
    target_observed = all_observed if target_observed is None else target_observed
    return figdata.ForecastSlice(
        series="A",
        origin_index=10,
        origin_date=pd.Timestamp("2024-01-11"),
        history_dates=pd.date_range("2024-01-01", periods=10).to_numpy(),
        history_values=[float(v) for v in range(100, 110)],
        target_dates=pd.date_range("2024-01-11", periods=5).to_numpy(),
        actual=[110.0, 108.0, 112.0, 105.0, 120.0],
        forecast=[109.0, 109.0, 109.0, 109.0, 109.0],
        q10=[104.0] * 5,
        q90=[114.0] * 5,
        naive=109.0,
        coverage=0.6,
        relative_mae=1.1,
        contains_glitch=False,
        observed_mask=np.array([True, True, target_observed, True, True]),
        history_observed=np.array([True] * 9 + [history_observed]),
    )


def test_apply_style_runs_without_error():
    plots.apply_style()


def test_plot_forecast_slice_reveal_true_draws_actual_and_forecast_lines():
    sl = _forecast_slice()
    ax = plots.plot_forecast_slice(sl, reveal=True)
    assert len(ax.lines) >= 3  # history, actual continuation, forecast
    assert len(ax.collections) >= 1  # the P10-P90 band


def test_plot_forecast_slice_reveal_false_omits_target_lines():
    sl = _forecast_slice()
    ax = plots.plot_forecast_slice(sl, reveal=False)
    assert len(ax.lines) == 2  # history line + the cut axvline (also a Line2D), nothing past it


def test_plot_forecast_slice_marks_imputed_points():
    ax_clean = plots.plot_forecast_slice(_forecast_slice(all_observed=True), reveal=True)
    ax_imputed = plots.plot_forecast_slice(_forecast_slice(all_observed=False), reveal=True)
    # the imputed case draws two extra scatter collections (one in history, one in the
    # revealed target) that the fully-observed case doesn't.
    assert len(ax_imputed.collections) > len(ax_clean.collections)


def test_plot_forecast_slice_imputed_legend_entry_is_not_duplicated():
    # Both history AND target carry an imputed point in this fixture, so
    # both scatter calls fire, each with label "imputato (forward-fill)".
    # ax.get_legend_handles_labels() re-scans every artist on the axes and
    # would show the duplicate regardless of what's actually rendered — the
    # thing under test is the rendered Legend itself (ax.get_legend()),
    # which plot_forecast_slice builds from a deduplicated handle/label map.
    sl = _forecast_slice(all_observed=False)
    ax = plots.plot_forecast_slice(sl, reveal=True)
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels.count("imputato (forward-fill)") == 1


def test_plot_forecast_slice_no_imputed_legend_entry_when_fully_observed():
    sl = _forecast_slice(all_observed=True)
    ax = plots.plot_forecast_slice(sl, reveal=True)
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels.count("imputato (forward-fill)") == 0


def test_plot_forecast_slice_imputed_target_alone_still_gets_a_legend_entry():
    # History fully observed, only the TARGET has an imputed point — only
    # the target-imputed scatter call fires. Before the fix that call had no
    # `label=`, so this case rendered hollow markers with nothing in the
    # legend explaining them.
    sl = _forecast_slice(history_observed=True, target_observed=False)
    ax = plots.plot_forecast_slice(sl, reveal=True)
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels.count("imputato (forward-fill)") == 1


def test_plot_shock_reaction_draws_two_panels():
    sub = pd.DataFrame(
        {
            "offset": [-2, -1, 0, 1, 2],
            "actual": [100.0, 98.0, 80.0, 82.0, 85.0],
            "forecast": [99.0, 97.0, 96.0, 90.0, 87.0],
            "q10": [90.0] * 5,
            "q90": [105.0] * 5,
            "baseline_naive": [99.0, 98.0, 97.0, 80.0, 82.0],
            "abs_pct_error": [1.0, 1.0, 20.0, 9.0, 2.0],
            "event": ["Test Event"] * 5,
        }
    )
    axes = plots.plot_shock_reaction(sub, threshold=5.0)
    assert len(axes) == 2
    assert len(axes[0].lines) >= 3


def test_plot_horizon_profile_two_panels():
    profile = pd.DataFrame(
        {
            "horizon_step": [1, 2, 3],
            "relative_mae_mean": [1.1, 1.05, 1.02],
            "relative_mae_min": [0.9, 0.95, 0.98],
            "relative_mae_max": [1.3, 1.2, 1.1],
            "coverage_mean": [0.8, 0.7, 0.6],
        }
    )
    axes = plots.plot_horizon_profile(profile)
    assert len(axes) == 2


def test_plot_quantile_bin_calibration_one_panel_per_horizon():
    preds = pd.DataFrame(
        {
            "horizon_step": [1] * 6 + [7] * 6,
            "actual": [1, 2, 3, 4, 5, 9] * 2,
            "observed": [True] * 12,
        }
    )
    for i, level in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        preds[f"q{round(level * 100):02d}"] = i + 1
    hist = figdata.quantile_bin_calibration(preds, horizon_steps=(1, 7))
    axes = plots.plot_quantile_bin_calibration(hist)
    assert len(axes) == 2


def test_plot_pit_histogram_alias_still_works():
    preds = pd.DataFrame(
        {"horizon_step": [1] * 3, "actual": [1, 2, 9], "observed": [True] * 3}
    )
    for i, level in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        preds[f"q{round(level * 100):02d}"] = i + 1
    hist = figdata.pit_histogram(preds, horizon_steps=(1,))
    axes = plots.plot_pit_histogram(hist)
    assert len(axes) == 1


def test_plot_calibration_curve_one_line_per_regime():
    curve = pd.DataFrame(
        {
            "regime": ["market_calm"] * 3 + ["market_shock"] * 3,
            "nominal_level": [0.1, 0.5, 0.9] * 2,
            "empirical_coverage": [0.09, 0.5, 0.88, 0.15, 0.55, 0.8],
            "n": [100] * 6,
        }
    )
    ax = plots.plot_calibration_curve(curve)
    # 1 diagonal + 2 regime lines
    assert len(ax.lines) == 3


def test_plot_adaptation_dots_two_panels():
    detail = pd.DataFrame(
        {
            "event": ["E1", "E2", "E3"],
            "arm": ["pre_cutoff", "pre_cutoff", "post_cutoff"],
            "pre_event_median_error": [1.0, 2.0, 0.5],
            "threshold": [1.5, 3.0, 0.75],
            "adaptation_lag_days": [1.0, 4.0, 15.0],
            "n_events_in_arm": [2, 2, 1],
        }
    )
    axes = plots.plot_adaptation_dots(detail)
    assert len(axes) == 2


def test_plot_card_relative_mae_one_dot_per_card():
    accuracy = pd.DataFrame(
        {
            "mode": ["timesfm3_univariate"] * 3,
            "transform": ["identity"] * 3,
            "series": ["A", "B", "C"],
            "relative_mae_vs_baseline": [1.1, 0.9, 1.05],
        }
    )
    ax = plots.plot_card_relative_mae(accuracy)
    assert len(ax.get_yticklabels()) == 3


def test_plot_glitch_vignette_one_panel_per_glitch():
    truth = pd.DataFrame(
        {
            "series": ["A"] * 5,
            "index": [8, 9, 10, 11, 12],
            "date": pd.date_range("2024-01-01", periods=5),
            "value": [10.0, 10.0, 20.0, 10.0, 10.0],
        }
    )
    glitches = pd.DataFrame({"series": ["A"], "index": [10], "date": [pd.Timestamp("2024-01-03")]})
    axes = plots.plot_glitch_vignette(truth, glitches, window=3)
    assert len(axes) == 1


def test_save_writes_a_png(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    path = plots.save(fig, "smoke_test_figure", directory=tmp_path)
    assert path.exists()
    assert path.suffix == ".png"
