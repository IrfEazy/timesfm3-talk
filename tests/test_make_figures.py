import importlib.util
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "06_make_figures.py"
spec = importlib.util.spec_from_file_location("make_figures", SCRIPT)
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)

# Digits allowed only in these label forms (interval names, GPU name).
ALLOWED = re.compile(r"P\d+|per 1k|TimesFM-3|GPU: Tesla T4")


def _text_artists(fig) -> list[str]:
    out = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        out += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
        out += [t.get_text() for t in ax.texts]
        leg = ax.get_legend()
        if leg is not None:
            out += [leg.get_title().get_text()] + [t.get_text() for t in leg.get_texts()]
    return out


def _assert_no_digits(fig):
    for text in _text_artists(fig):
        assert not re.search(r"\d", ALLOWED.sub("", text)), text
    plt.close(fig)


def test_exp_a_figure_has_no_numeric_text():
    steps = np.arange(1, 6)
    skill = pd.DataFrame(
        {"horizon_step": steps, "skill_median": -0.01 * steps, "skill_mean_weighted": -0.02 * steps}
    )
    cis = pd.DataFrame(
        {
            "horizon": [3, 5],
            "statistic": ["median", "weighted_mean"],
            "estimate": [-0.03, -0.1],
            "ci_low": [-0.06, -0.15],
            "ci_high": [0.01, -0.05],
        }
    )
    _assert_no_digits(mf.fig_exp_a_skill(skill, cis))


def test_coverage_and_conformal_figures_have_no_numeric_text():
    cov = pd.DataFrame(
        {
            "regime": ["mtg", "mtg", "shock", "shock"],
            "horizon": [1, 7, 1, 7],
            "coverage": [0.8, 0.7, 0.6, 0.5],
            "cb_low": [0.75, 0.65, 0.5, 0.4],
            "cb_high": [0.85, 0.75, 0.7, 0.6],
        }
    )
    _assert_no_digits(mf.fig_exp_c_coverage(cov))
    conf = pd.DataFrame({"horizon": [1, 7]})
    for s in ("before", "after"):
        conf[f"coverage_{s}"] = [0.7, 0.6]
        conf[f"cb_low_{s}"] = [0.6, 0.5]
        conf[f"cb_high_{s}"] = [0.8, 0.7]
    fig = mf.fig_exp_c_conformal(conf)
    assert any(mf.EXPLORATORY_LABEL in t for t in _text_artists(fig))
    _assert_no_digits(fig)


def test_onset_leakage_latency_figures():
    onset = pd.DataFrame(
        {
            "arm": ["pre", "pre", "post", "post"],
            "origin_offset": [-1, 0, -1, 0],
            "median_ape": [0.01, 0.02, 0.03, 0.04],
        }
    )
    fig = mf.fig_exp_b_onset(onset)
    assert any(mf.RED_TEAM_LABEL in t for t in _text_artists(fig))
    _assert_no_digits(fig)

    err = pd.DataFrame({"horizon_step": [1, 2], "observed": [True, True], "median_err": [0.1, 0.2]})
    fig = mf.fig_leakage_demo(err)
    assert mf.V2_WATERMARK in _text_artists(fig)
    _assert_no_digits(fig)

    table = pd.DataFrame({"method": ["a", "a"], "batch": [1, 8], "median_s_per_1k": [10.0, 4.0]})
    _assert_no_digits(mf.fig_latency(table, "Tesla T4"))


def test_digit_check_catches_numbers():
    fig, ax = plt.subplots()
    ax.text(0, 0, "0.72")
    with pytest.raises(AssertionError):
        _assert_no_digits(fig)
    plt.close(fig)


def test_committed_leaderboard_matches_verdict_table():
    res = Path(__file__).resolve().parents[1] / "results"
    agg = pd.read_parquet(res / "exp_mtg_benchmark_leaderboard_aggregate.parquet")
    ver = pd.read_parquet(res / "exp_mtg_v2_verdict.parquet")
    cell = agg[
        (agg["mode"] == "timesfm3_univariate")
        & (agg["transform"] == "identity")
        & (agg["make_positive"].astype(bool))
        & (agg["context_len"] == 512)
        & (agg["baseline_method"] == "naive")
    ]
    pool = ver[(ver["subset"] == "pool") & (ver["baseline"] == "naive")]
    cols = {"median": "skill_median", "weighted_mean": "skill_mean_weighted"}
    for h in (28, 56):
        row = cell[(cell["requested_horizon"] == h) & (cell["horizon_step"] == h)]
        assert len(row) == 1
        for stat, col in cols.items():
            est = pool[(pool["horizon"] == h) & (pool["statistic"] == stat)]["estimate"]
            assert len(est) == 1
            assert abs(float(row[col].iloc[0]) - float(est.iloc[0])) < 1e-9


def test_v3_figure_preserves_interval_endpoints_and_effect_directions():
    claims = mf._v3_claims()
    fig = mf.fig_v3_primary(claims)
    for ax, study in zip(fig.axes, ("s1", "s2", "s3"), strict=True):
        c = claims[f"v3.{study}"]
        endpoints = [(c["ci_low"], c["ci_high"])]
        if study != "s1":
            endpoints.append((c["cal_ci_low"], c["cal_ci_high"]))
        for collection, bounds in zip(ax.collections, endpoints, strict=True):
            np.testing.assert_allclose(collection.get_segments()[0][:, 0], bounds)
        assert ("positive = better" if study == "s2" else "positive = worse") in ax.get_xlabel()
        assert c["verdict"] in ax.get_title()
    assert "98.33%" in fig._suptitle.get_text()
    plt.close(fig)


def test_v3_floor_figure_uses_all_requests_not_primary_counts():
    claims = mf._v3_claims()
    fig = mf.fig_v3_floor(claims)
    bars = fig.axes[0].patches
    expected = [
        claims[f"v3.floor.{s}"][field]
        for field in ("median_steps", "requests")
        for s in ("s1", "s2", "s3")
    ]
    np.testing.assert_allclose([b.get_height() for b in bars], expected)
    assert claims["v3.floor.s1"]["n"] != claims["v3.s1"]["n_skus"]
    assert "broader than the primary analysis sets" in fig.axes[0].get_title()
    plt.close(fig)
