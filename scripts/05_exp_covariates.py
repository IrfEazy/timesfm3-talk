#!/usr/bin/env python3
"""Experiment D — covariates, made falsifiable (Preregistration v2, section 5).

Arms (per series, same origins):
  no_covariate  the univariate baseline
  legit         a covariate known at the forecast origin
                  - market: day-of-week sin/cos (calendar)
                  - MTG: days to the next set release, from release dates known
                    at the origin (publishedOn minus a documented announcement
                    lead proxy — a limitation, see tfm3lab.covariates)
  placebo       same shape, no information: MTG release dates circularly shifted
                by a seeded 60-300 day offset; market weekday labels permuted
                by a seeded permutation (seed 20261010)
  leaked        TEACHING CONTRAST ONLY: the series' own real future handed in as
                a "covariate". Written ONLY to the leakage-demo file with a
                watermark; never summarized.

Pool: MTG over the frozen pool configs/benchmark_pool_v2.csv (cards whose set
release date resolves; the others are listed as excluded with a reason);
market over SP500 and the other v2 indices present in the cache. Context 128,
horizon 7, origin_stride 7. Outputs are normalized by the last observed context
value (no raw prices). "Helps" iff the paired skill-delta primary CI (cluster
bootstrap, cards/series as clusters) is > 0 versus BOTH no_covariate AND placebo.

Requires scripts/01_fetch_data.py already run and a loaded TimesFM-3 forecaster.

Usage (--as-of is REQUIRED; --allow-cpu skips the GPU preflight):
    uv run scripts/05_exp_covariates.py --as-of 2026-09-01 [--allow-cpu]

Writes to results/:
    exp_covariates_v2_predictions.parquet
    exp_covariates_v2_summary.parquet
    exp_covariates_v2_excluded.parquet
    exp_covariates_v2_leakage_demo.parquet   (watermarked, leaked arm only)
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from tfm3lab import config, covariates
from tfm3lab.backtest import SeriesData
from tfm3lab.data.mtg import MAGIC_CATEGORY_ID, fetch_groups, load_card_manifest
from tfm3lab.manifest import capture_git_state, write_run_manifest
from tfm3lab.model import forecast_batch, load_forecaster
from tfm3lab.preflight import preflight_model
from tfm3lab.windows import context_slice, valid_origins

CONTEXT_LEN = 128
HORIZON = 7
ORIGIN_STRIDE = 7
POOL_PATH = config.REPO_ROOT / "configs" / "benchmark_pool_v2.csv"
MARKET_SERIES = ("SP500", "Nasdaq100", "EuroStoxx50", "Nikkei225", "Gold")

OUT_PREDICTIONS = "exp_covariates_v2_predictions.parquet"
OUT_SUMMARY = "exp_covariates_v2_summary.parquet"
OUT_EXCLUDED = "exp_covariates_v2_excluded.parquet"
OUT_LEAKAGE = "exp_covariates_v2_leakage_demo.parquet"


def _load_series(cache_name: str) -> list[SeriesData]:
    df = pd.read_parquet(config.CACHE_DIR / cache_name)
    out = []
    for name, g in df.groupby("series"):
        g = g.sort_values("date")
        out.append(
            SeriesData(
                name=str(name),
                values=g["value"].to_numpy(dtype=float),
                dates=g["date"].to_numpy(),
                observed=g["observed"].to_numpy(dtype=bool),
            )
        )
    return out


def strided_origins(n: int, context_len: int, horizon: int, stride: int) -> np.ndarray:
    return valid_origins(n, context_len, horizon)[::stride]


def fetch_release_map(abbreviations) -> dict[str, np.datetime64]:
    """group abbreviation (upper) -> publishedOn date. Network."""
    groups = fetch_groups(MAGIC_CATEGORY_ID, requests.Session())
    wanted = {a.upper() for a in abbreviations}
    out = {}
    for _, row in groups.dropna(subset=["publishedOn"]).iterrows():
        ab = str(row["abbreviation"]).upper()
        if ab in wanted:
            out[ab] = np.datetime64(str(row["publishedOn"])[:10])
    return out


def run_arm(
    forecaster,
    s: SeriesData,
    origins: np.ndarray,
    arm: str,
    spec: covariates.CovariateSpec | None,
    cov_fn,
    *,
    context_len: int,
    horizon: int,
    domain: str,
    teaching_only: bool = False,
) -> pd.DataFrame:
    """One arm over the given origins. Legit/placebo arms go through the
    availability contract; the leaked arm must say teaching_only=True."""
    if arm == covariates.LEAKED_ARM:
        if not teaching_only:
            raise covariates.CovariateAvailabilityError(
                "the leaked arm needs the explicit teaching_only path"
            )
    elif spec is not None:
        covariates.require_future_known(spec)
    contexts, covs, ts_ids = [], [], []
    for origin in origins:
        origin = int(origin)
        ctx = s.values[context_slice(origin, context_len)]
        contexts.append(ctx)
        ts_ids.append(f"{s.name}::{origin}::{arm}")
        covs.append(None if cov_fn is None else cov_fn(s, origin))
    batch = forecast_batch(
        forecaster,
        contexts,
        horizon,
        ts_ids=ts_ids,
        past_future_covariates=covs if cov_fn is not None else None,
    )
    rows = []
    for i, origin in enumerate(origins):
        origin = int(origin)
        scale = float(contexts[i][-1])
        for h in range(horizon):
            t = origin + h
            actual = float(s.values[t])
            fc = float(batch.forecast[i, h])
            rows.append(
                {
                    "domain": domain,
                    "series": s.name,
                    "origin_index": origin,
                    "target_date": pd.Timestamp(s.dates[t]),
                    "horizon_step": h + 1,
                    "arm": arm,
                    "observed": bool(s.observed[t]) and scale != 0,
                    "actual_norm": actual / scale if scale else np.nan,
                    "forecast_norm": fc / scale if scale else np.nan,
                    "abs_err_norm": abs(fc - actual) / abs(scale) if scale else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _window_dates(s: SeriesData, origin: int, context_len: int, horizon: int):
    return s.dates[origin - context_len : origin + horizon]


def _arm_specs(domain, release_dates, perm, offset, span, lead, cl, hz):
    """{arm: (spec, cov_fn)} for one domain. cov_fn(s, origin) -> (k, ctx+h)."""
    if domain == "market":

        def legit(s, o):
            return covariates.day_of_week_covariate(_window_dates(s, o, cl, hz))

        def placebo(s, o):
            return covariates.permuted_day_of_week_covariate(
                _window_dates(s, o, cl, hz), perm
            )

        return {
            "legit": (covariates.DAY_OF_WEEK, legit),
            "placebo": (covariates.DAY_OF_WEEK_PLACEBO, placebo),
        }
    shifted = covariates.shift_release_dates_circular(release_dates, span[0], span[1], offset)

    def legit(s, o):
        return covariates.days_to_next_release(
            _window_dates(s, o, cl, hz), s.dates[o - 1], release_dates,
            announce_lead_days=lead,
        )

    def placebo(s, o):
        return covariates.days_to_next_release(
            _window_dates(s, o, cl, hz), s.dates[o - 1], shifted,
            announce_lead_days=lead,
        )

    return {
        "legit": (covariates.DAYS_TO_RELEASE, legit),
        "placebo": (covariates.DAYS_TO_RELEASE_PLACEBO, placebo),
    }


def run_experiment(
    forecaster,
    *,
    market_series: list[SeriesData],
    mtg_series: list[SeriesData],
    release_dates,
    excluded: list[dict],
    results_dir: Path,
    context_len: int = CONTEXT_LEN,
    horizon: int = HORIZON,
    stride: int = ORIGIN_STRIDE,
    seed: int = covariates.SEED,
    n_boot: int | None = None,
    announce_lead_days: int = covariates.ANNOUNCE_LEAD_DAYS_PROXY,
) -> dict[str, Path]:
    """All arms over both domains; writes the four v2 outputs."""
    perm = covariates.dow_permutation(seed)
    offset = covariates.placebo_offset_days(seed)
    all_dates = [s.dates[0] for s in mtg_series] + [s.dates[-1] for s in mtg_series]
    span = (
        (np.datetime64(min(all_dates), "D"), np.datetime64(max(all_dates), "D"))
        if all_dates
        else (np.datetime64("2024-01-01"), np.datetime64("2026-01-01"))
    )

    pred_frames, leak_frames = [], []
    for domain, series_list in (("market", market_series), ("mtg", mtg_series)):
        arms = _arm_specs(
            domain, release_dates, perm, offset, span, announce_lead_days, context_len, horizon
        )
        for s in series_list:
            origins = strided_origins(len(s.values), context_len, horizon, stride)
            if len(origins) == 0:
                excluded.append({"domain": domain, "series": s.name, "reason": "series too short"})
                continue
            kw = {"context_len": context_len, "horizon": horizon, "domain": domain}
            pred_frames.append(run_arm(forecaster, s, origins, "no_covariate", None, None, **kw))
            for arm, (spec, fn) in arms.items():
                pred_frames.append(run_arm(forecaster, s, origins, arm, spec, fn, **kw))
            if domain == "mtg":
                leak = run_arm(
                    forecaster, s, origins, covariates.LEAKED_ARM, covariates.LEAKED_FUTURE,
                    lambda ser, o: covariates.teaching_only_leaked_covariate(
                        ser.values, o, context_len, horizon
                    ),
                    teaching_only=True, **kw,
                )
                leak["watermark"] = covariates.LEAKED_WATERMARK
                leak_frames.append(leak)
    if not pred_frames:
        raise RuntimeError("no series produced any origins; nothing to evaluate")
    preds = pd.concat(pred_frames, ignore_index=True)
    covariates.assert_no_leaked(preds)  # the leaked arm never shares the main file
    summary = covariates.paired_skill_summary(
        preds, primary_horizon=horizon, n_boot=n_boot, seed=seed
    )
    summary["context_len"] = context_len
    summary["origin_stride"] = stride
    summary["placebo_offset_days"] = offset
    summary["announce_lead_days_proxy"] = announce_lead_days

    paths = {
        "predictions": results_dir / OUT_PREDICTIONS,
        "summary": results_dir / OUT_SUMMARY,
        "excluded": results_dir / OUT_EXCLUDED,
        "leakage_demo": results_dir / OUT_LEAKAGE,
    }
    leak_df = (
        pd.concat(leak_frames, ignore_index=True)
        if leak_frames
        else pd.DataFrame(columns=[*preds.columns, "watermark"])
    )
    preds.to_parquet(paths["predictions"], index=False)
    summary.to_parquet(paths["summary"], index=False)
    pd.DataFrame(excluded, columns=["domain", "series", "reason"]).to_parquet(
        paths["excluded"], index=False
    )
    leak_df.to_parquet(paths["leakage_demo"], index=False)
    return paths


def build_mtg_inputs(pool_path: Path = POOL_PATH, cache_name: str = "mtg_prices.parquet"):
    """(mtg_series, release_dates, excluded) for the frozen pool."""
    cards = load_card_manifest(pool_path)
    cached = {s.name: s for s in _load_series(cache_name)}
    release_map = fetch_release_map({c.group_abbreviation for c in cards})
    series, excluded = [], []
    for c in cards:
        if c.group_abbreviation.upper() not in release_map:
            excluded.append(
                {"domain": "mtg", "series": c.label, "reason": "set release date unresolved"}
            )
        elif c.label not in cached:
            excluded.append(
                {"domain": "mtg", "series": c.label, "reason": "no cached price series"}
            )
        else:
            series.append(cached[c.label])
    used = {
        c.group_abbreviation.upper()
        for c in cards
        if c.label in {s.name for s in series}
    }
    release_dates = np.array(sorted(release_map[a] for a in used), dtype="datetime64[D]")
    return series, release_dates, excluded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--as-of",
        required=True,
        type=dt.date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="the data freeze date used for the fetch; recorded in the run manifest (required)",
    )
    parser.add_argument(
        "--allow-cpu", action="store_true", help="skip the GPU requirement (slow smoke runs only)"
    )
    parser.add_argument(
        "--announce-lead-days",
        type=int,
        default=covariates.ANNOUNCE_LEAD_DAYS_PROXY,
        help="proxy for the unavailable set announcement date: publishedOn minus this many days",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    git_dirty = capture_git_state()
    checkpoint = preflight_model(allow_cpu=args.allow_cpu)
    forecaster = load_forecaster(revision=checkpoint["revision_resolved"])

    market = [s for s in _load_series("market_prices.parquet") if s.name in MARKET_SERIES]
    mtg, release_dates, excluded = build_mtg_inputs()
    print(f"market series: {[s.name for s in market]}; MTG cards: {len(mtg)}; "
          f"excluded: {len(excluded)}")
    for e in excluded:
        print(f"  excluded {e['series']}: {e['reason']}")
    paths = run_experiment(
        forecaster,
        market_series=market,
        mtg_series=mtg,
        release_dates=release_dates,
        excluded=excluded,
        results_dir=config.RESULTS_DIR,
        announce_lead_days=args.announce_lead_days,
    )
    summary = pd.read_parquet(paths["summary"])
    primary = summary[summary["is_primary"]]
    print(primary[["domain", "comparison", "skill_delta", "ci_low", "ci_high", "n_clusters",
                   "verdict"]].to_string(index=False))
    print("Leakage demo is watermarked and excluded from every summary.")
    manifest_path = write_run_manifest(
        "exp_covariates_v2",
        list(paths.values()),
        as_of=args.as_of,
        checkpoint=checkpoint,
        seed=covariates.SEED,
        git_dirty=git_dirty,
        extra={
            "pool": str(POOL_PATH),
            "context_len": CONTEXT_LEN,
            "horizon": HORIZON,
            "origin_stride": ORIGIN_STRIDE,
            "announce_lead_days_proxy": args.announce_lead_days,
            "excluded": excluded,
        },
    )
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
