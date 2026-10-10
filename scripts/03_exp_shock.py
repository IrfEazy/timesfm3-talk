#!/usr/bin/env python3
"""Experiment B v2 -- error-recovery lag around rule-selected market shocks.

Preregistration v2 section 3. EXPLORATORY red-team appendix unless the
promotion rule (>= 8 events per arm, primary CI excludes 0, same sign under
both perturbations) is met. The primary forecaster mode is UNIVARIATE
(comparable with naive and TimesFM-2.5); multivariate is a sensitivity.

Requires:
  - scripts/03a_select_events.py already run for the same --as-of (writes
    configs/shock_events_v2.csv and the closes cache under data/cache)
  - a GPU and the gated TimesFM-3 checkpoint (or --adapter timesfm2.5)

Usage (--as-of is REQUIRED and must match the 03a run; --allow-cpu skips the
GPU preflight for smoke runs):
    uv run scripts/03_exp_shock.py --as-of 2026-09-04
    uv run scripts/03_exp_shock.py --as-of 2026-09-04 --adapter timesfm2.5
    uv run scripts/03_exp_shock.py --as-of 2026-09-04 --legacy-events

Per event: TimesFM univariate (primary) and multivariate (sensitivity, TimesFM-3
only) rolling forecasts with max_horizon 28, naive and ETS baselines read from
the same runs, plus perturbation runs (noise, context splice; 5 seeds; h<=5).

Writes (committable: values normalized by the last observed context value,
errors as APE, no raw prices; the timesfm2.5 adapter adds a _2p5 suffix):
    results/exp_shock_v2_predictions.parquet
    results/exp_shock_v2_lags.parquet     (event x forecaster x perturbation x multiplier)
    results/exp_shock_v2_summary.parquet  (effects, CIs, promotion verdict)
Nothing is written to the legacy exp_shock_* file names. Legacy hand-listed
events (--legacy-events) are run INSTEAD of the rule events and are reported
under source="legacy"; they are never pooled with the rule-based arms.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tfm3lab import config, shock
from tfm3lab.backtest import SeriesData, run_multivariate_backtest, run_univariate_backtest
from tfm3lab.data import market
from tfm3lab.manifest import capture_git_state, write_run_manifest
from tfm3lab.model import forecast_batch
from tfm3lab.windows import context_slice, nearest_origin

EVENTS_CSV = Path(__file__).resolve().parents[1] / "configs" / "shock_events_v2.csv"
MAX_HORIZON = 28
PERTURB_HORIZON = 5
PERTURBATIONS = ("noise", "splice")

PRED_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("source", pa.string()),
        ("arm", pa.string()),
        ("forecaster", pa.string()),
        ("perturbation", pa.string()),
        ("seed", pa.int32()),
        ("origin_offset", pa.int32()),
        ("horizon_step", pa.int32()),
        ("actual_norm", pa.float64()),
        ("forecast_norm", pa.float64()),
        ("ape", pa.float64()),
    ]
)


def add_run_args(parser: argparse.ArgumentParser) -> None:
    """Flags shared by the model scripts (preregistration v2 s7)."""
    parser.add_argument(
        "--as-of",
        required=True,
        type=dt.date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="the data freeze date used for the fetch; recorded in the run manifest (required)",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="skip the GPU requirement (slow smoke runs only)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_run_args(parser)
    parser.add_argument("--adapter", choices=("timesfm3", "timesfm2.5"), default="timesfm3")
    parser.add_argument("--legacy-events", action="store_true")
    parser.add_argument("--events", type=Path, default=EVENTS_CSV)
    parser.add_argument("--context-len", type=int, default=shock.CONTEXT_LEN)
    parser.add_argument("--window-before", type=int, default=shock.WINDOW_BEFORE)
    parser.add_argument("--window-after", type=int, default=shock.WINDOW_AFTER)
    return parser


# ---------------------------------------------------------------- events


def rule_events(path: Path) -> list[shock.EventSpec]:
    df = pd.read_csv(path, keep_default_na=False)
    df = df[df["excluded_reason"] == ""]
    return [
        shock.EventSpec(
            r.event_id, "rule", r.arm, dt.date.fromisoformat(r.onset_date), r.onset_index
        )
        for r in df.itertuples()
    ]


def legacy_events() -> list[shock.EventSpec]:
    arm = {"pre_cutoff": "pre", "post_cutoff": "post"}
    return [
        shock.EventSpec(
            "legacy-" + e.name.lower().replace(" ", "-"), "legacy", arm[e.arm], e.date, "SP500"
        )
        for e in config.KNOWN_EVENTS
    ]


def onset_position(spec: shock.EventSpec, series: SeriesData) -> int:
    dates = np.asarray(series.dates, dtype="datetime64[ns]")
    if spec.source == "legacy":
        return nearest_origin(dates, np.datetime64(spec.onset_date))
    pos = int(np.searchsorted(dates, np.datetime64(spec.onset_date, "ns")))
    if pos >= len(dates) or dates[pos] != np.datetime64(spec.onset_date, "ns"):
        raise ValueError(f"{spec.event_id}: onset date missing from {series.name} calendar")
    return pos


# ---------------------------------------------------------------- forecasting


def perturbed_forecasts(forecaster, values, origins, context_len, kind, seed, horizon):
    """(n_origins, horizon) point forecasts from perturbed contexts, or None
    when the perturbation cannot be built (splice with too little history)."""
    contexts = []
    for o in origins:
        ctx = values[context_slice(int(o), context_len)]
        if kind == "noise":
            contexts.append(shock.add_log_noise(ctx, seed=seed * 1_000_003 + int(o)))
        else:
            spliced = shock.splice_context(values, int(o), context_len, seed)
            if spliced is None:
                return None
            contexts.append(spliced)
    batch = forecast_batch(forecaster, contexts, horizon, ts_ids=[f"p{int(o)}" for o in origins])
    order = {t: i for i, t in enumerate(batch.ts_ids)}
    return np.stack([batch.forecast[order[f"p{int(o)}"]] for o in origins])


def _frame(df, values, pos, spec, forecaster, perturbation="none", seed=0, column="forecast"):
    df = df[np.isfinite(df["actual"])] if len(df) else df
    last = values[df["origin_index"].to_numpy() - 1]
    out = shock.normalized_rows(
        df["origin_index"], df["horizon_step"], df["actual"], df[column], last, pos
    )
    out.insert(0, "seed", np.int32(seed))
    out.insert(0, "perturbation", perturbation)
    out.insert(0, "forecaster", forecaster)
    out.insert(0, "arm", spec.arm)
    out.insert(0, "source", spec.source)
    out.insert(0, "event_id", spec.event_id)
    return out[[f.name for f in PRED_SCHEMA]]


def _lag_rows(frame, spec, delta_baselines: dict | None):
    """Lag rows for one forecaster/perturbation/seed from its h=1 rows."""
    h1 = frame[frame["horizon_step"] == 1]
    rows = []
    for mult in shock.LAG_MULTIPLIERS:
        lag, cens = shock.error_recovery_lag(
            h1["origin_offset"].to_numpy(), h1["ape"].to_numpy(), mult
        )
        row = {
            "event_id": spec.event_id,
            "source": spec.source,
            "arm": spec.arm,
            "forecaster": frame["forecaster"].iloc[0],
            "perturbation": frame["perturbation"].iloc[0],
            "seed": int(frame["seed"].iloc[0]),
            "multiplier": mult,
            "lag": lag,
            "censored": bool(cens),
        }
        for name in ("naive", "ets"):
            base = (delta_baselines or {}).get(name, {}).get(mult)
            row[f"lag_{name}"] = base[0] if base else np.nan
            row[f"d_{name}"] = lag - base[0] if base else np.nan
        rows.append(row)
    return rows


def anticipation_row(frame, spec, values, pos):
    """Hit rate over origin offsets -4..0 (last observed at -5..-1) at the
    horizon that lands on the onset day; actual move = onset-day move."""
    move = values[pos] - values[pos - 1]
    hits = []
    for off in shock.ANTICIPATION_OFFSETS:
        rows = frame[(frame["origin_offset"] == off) & (frame["horizon_step"] == 1 - off)]
        if len(rows) != 1:
            continue
        hits.append(shock.anticipation_hit(rows["forecast_norm"].iloc[0], 1.0, move))
    return {
        "event_id": spec.event_id,
        "source": spec.source,
        "arm": spec.arm,
        "forecaster": frame["forecaster"].iloc[0],
        "perturbation": frame["perturbation"].iloc[0],
        "seed": int(frame["seed"].iloc[0]),
        "hit_rate": float(np.mean(hits)) if hits else np.nan,
        "n_origins": len(hits),
    }


def process_event(
    forecaster, adapter, spec, series, panel, context_len, window_before, window_after, perturb
):
    """Returns (prediction frames, lag rows, anticipation rows, notes) for one event."""
    s = series[spec.onset_index]
    pos = onset_position(spec, s)
    notes = []
    if not shock.window_fits(pos, len(s.values), context_len, window_before, window_after):
        return [], [], [], [f"{spec.event_id}: window does not fit; skipped"]
    origins = np.arange(pos - window_before, pos + window_after)
    uni_label = f"{adapter}_univariate"

    uni = run_univariate_backtest(
        forecaster, [s], origins, context_len, MAX_HORIZON, mode_label=uni_label
    )
    frames = {(uni_label, "none", 0): _frame(uni, s.values, pos, spec, uni_label)}
    base_frames = {
        name: _frame(uni, s.values, pos, spec, name, column=f"baseline_{name}")
        for name in ("naive", "ets")
        if f"baseline_{name}" in uni
    }
    # Baselines get unperturbed rows as their own "forecasters".
    base_lags = {}
    for name, bf in base_frames.items():
        base_lags[name] = {
            r["multiplier"]: (r["lag"], r["censored"]) for r in _lag_rows(bf, spec, None)
        }

    if panel is not None and adapter == "timesfm3":
        p_dates = np.asarray(panel[0].dates, dtype="datetime64[ns]")
        ppos = int(np.searchsorted(p_dates, np.datetime64(spec.onset_date, "ns")))
        if (
            ppos < len(p_dates)
            and p_dates[ppos] == np.datetime64(spec.onset_date, "ns")
            and shock.window_fits(ppos, len(p_dates), context_len, window_before, window_after)
        ):
            p_origins = np.arange(ppos - window_before, ppos + window_after)
            multi = run_multivariate_backtest(
                forecaster, panel, p_origins, context_len, MAX_HORIZON
            )
            multi = multi[multi["series"] == spec.onset_index]
            p_values = next(p for p in panel if p.name == spec.onset_index).values
            frames[("timesfm3_multivariate", "none", 0)] = _frame(
                multi, p_values, ppos, spec, "timesfm3_multivariate"
            )
        else:
            notes.append(f"{spec.event_id}: multivariate skipped (onset/window not in panel)")

    if perturb:
        for kind in PERTURBATIONS:
            for seed in range(1, shock.N_PERTURB_SEEDS + 1):
                fc = perturbed_forecasts(
                    forecaster, s.values, origins, context_len, kind, seed, PERTURB_HORIZON
                )
                if fc is None:
                    notes.append(f"{spec.event_id}: {kind} seed {seed} skipped (short history)")
                    continue
                recs = [
                    {
                        "origin_index": int(o),
                        "horizon_step": h + 1,
                        "actual": s.values[o + h] if o + h < len(s.values) else np.nan,
                        "forecast": fc[i, h],
                    }
                    for i, o in enumerate(origins)
                    for h in range(PERTURB_HORIZON)
                ]
                frames[(uni_label, kind, seed)] = _frame(
                    pd.DataFrame(recs), s.values, pos, spec, uni_label, kind, seed
                )

    lag_rows, antic_rows = [], []
    for (_, _, _), fr in frames.items():
        lag_rows.extend(_lag_rows(fr, spec, base_lags))
        antic_rows.append(anticipation_row(fr, spec, s.values, pos))
    for bf in base_frames.values():
        lag_rows.extend(_lag_rows(bf, spec, None))
    return [*frames.values(), *base_frames.values()], lag_rows, antic_rows, notes


# ---------------------------------------------------------------- summary


def _event_means(df, value_col):
    """Mean over perturbation seeds per event (identity when one row)."""
    return df.groupby(["event_id", "arm"], as_index=False)[value_col].mean()


def summarize(lags, antic, primary, n_boot=shock.N_BOOT, seed=shock.BOOT_SEED):
    rows = []

    def add(source, forecaster, perturbation, metric, per_event, definition="post_minus_pre"):
        pre = per_event.loc[per_event["arm"] == "pre", per_event.columns[-1]]
        post = per_event.loc[per_event["arm"] == "post", per_event.columns[-1]]
        res = shock.cluster_difference_ci(pre, post, n_boot, seed, effect_definition=definition)
        rows.append(
            {
                "source": source,
                "forecaster": forecaster,
                "perturbation": perturbation,
                "metric": metric,
                **res,
                "verdict": "",
                "reasons": "",
            }
        )
        return res

    results = {}
    for source in sorted(lags["source"].unique()) if len(lags) else []:
        ls = lags[lags["source"] == source]
        models = [f for f in ls["forecaster"].unique() if f not in ("naive", "ets")]
        for f in models:
            for pert in ls["perturbation"].unique():
                sub = ls[(ls["forecaster"] == f) & (ls["perturbation"] == pert)]
                for mult in shock.LAG_MULTIPLIERS:
                    m = sub[sub["multiplier"] == mult]
                    for base in ("naive", "ets"):
                        pe = _event_means(m[["event_id", "arm", f"d_{base}"]], f"d_{base}")
                        res = add(source, f, pert, f"lag_D_{base}@{mult}", pe)
                        results[(source, f, pert, f"lag_D_{base}@{mult}")] = res
        asub = antic[antic["source"] == source]
        for f in asub["forecaster"].unique():
            for pert in asub["perturbation"].unique():
                a = asub[(asub["forecaster"] == f) & (asub["perturbation"] == pert)]
                add(
                    source,
                    f,
                    pert,
                    "anticipation_hit_rate",
                    _event_means(a[["event_id", "arm", "hit_rate"]], "hit_rate"),
                    definition="pre_minus_post",
                )

    # Promotion applies only to the rule-based arms and the primary forecaster.
    key = ("rule", primary, "none", f"lag_D_naive@{shock.HEADLINE_MULTIPLIER}")
    if primary == "timesfm3_univariate" and key in results:
        r = results[key]
        pert = {
            p: results.get(("rule", primary, p, key[3]), {}).get("effect", float("nan"))
            for p in PERTURBATIONS
        }
        verdict, reasons = shock.promotion_verdict(
            r["n_pre"], r["n_post"], r["effect"], r["ci_low"], r["ci_high"], pert
        )
        used = {p: results.get(("rule", primary, p, key[3]), {}) for p in ("none", *PERTURBATIONS)}
        n_note = "; ".join(
            f"{p}: pre={used[p].get('n_pre', 0)}, post={used[p].get('n_post', 0)}"
            for p in ("none", *PERTURBATIONS)
        )
        rows.append(
            {
                "source": "rule",
                "forecaster": primary,
                "perturbation": "none",
                "metric": "promotion",
                "effect_definition": "post_minus_pre",
                "n_pre": r["n_pre"],
                "n_post": r["n_post"],
                "n_pre_noise": used["noise"].get("n_pre", 0),
                "n_post_noise": used["noise"].get("n_post", 0),
                "n_pre_splice": used["splice"].get("n_pre", 0),
                "n_post_splice": used["splice"].get("n_post", 0),
                "notes": (
                    f"events used per effect ({n_note}); splice skips events with too "
                    "short a history; perturbed D uses unperturbed naive/ets baseline lags"
                ),
                "effect": r["effect"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "verdict": verdict,
                "reasons": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- driver


def run_experiment(
    forecaster,
    adapter,
    specs,
    series,
    out_dir: Path,
    context_len=shock.CONTEXT_LEN,
    window_before=shock.WINDOW_BEFORE,
    window_after=shock.WINDOW_AFTER,
    perturb=True,
    n_boot=shock.N_BOOT,
):
    """Streams events to the predictions parquet one at a time; returns
    (paths, notes). `series` maps label -> SeriesData (own calendars)."""
    suffix = "" if adapter == "timesfm3" else "_2p5"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"exp_shock_v2_predictions{suffix}.parquet"
    lag_path = out_dir / f"exp_shock_v2_lags{suffix}.parquet"
    sum_path = out_dir / f"exp_shock_v2_summary{suffix}.parquet"

    panel = None
    if adapter == "timesfm3":
        frame = pd.concat(
            {k: pd.Series(v.values, index=pd.to_datetime(v.dates)) for k, v in series.items()},
            axis=1,
        ).dropna()
        if len(frame):
            panel = [
                SeriesData(
                    k, frame[k].to_numpy(float), frame.index.to_numpy(), np.ones(len(frame), bool)
                )
                for k in series
            ]

    lag_rows, antic_rows, notes = [], [], []
    with pq.ParquetWriter(pred_path, PRED_SCHEMA) as writer:
        for spec in specs:
            frames, lr, ar, nt = process_event(
                forecaster,
                adapter,
                spec,
                series,
                panel,
                context_len,
                window_before,
                window_after,
                perturb,
            )
            for fr in frames:
                writer.write_table(
                    pa.Table.from_pandas(fr, schema=PRED_SCHEMA, preserve_index=False)
                )
            lag_rows += lr
            antic_rows += ar
            notes += nt
            print(f"  {spec.event_id} ({spec.arm}, {spec.onset_index}): {len(lr)} lag rows")

    lags = pd.DataFrame(lag_rows)
    antic = pd.DataFrame(antic_rows)
    lags.to_parquet(lag_path, index=False)
    summary = summarize(lags, antic, f"{adapter}_univariate", n_boot=n_boot)
    summary.to_parquet(sum_path, index=False)
    return [pred_path, lag_path, sum_path], notes


def main() -> None:
    args = build_parser().parse_args()
    git_dirty = capture_git_state()
    if args.adapter == "timesfm3":
        from tfm3lab.model import load_forecaster
        from tfm3lab.preflight import preflight_model

        checkpoint = preflight_model(allow_cpu=args.allow_cpu)
        forecaster = load_forecaster(revision=checkpoint["revision_resolved"])
        adapter = "timesfm3"
    else:
        from tfm3lab.model_2p5 import load_forecaster_2p5
        from tfm3lab.preflight import require_gpu

        require_gpu(allow_cpu=args.allow_cpu)
        checkpoint = {
            "repo_id": config.CHECKPOINT_ID_2P5,
            "revision_requested": config.CHECKPOINT_REVISION_2P5,
            "revision_resolved": None,
        }
        forecaster = load_forecaster_2p5()
        adapter = "timesfm2p5"

    series = market.load_index_series(config.CACHE_DIR, args.as_of)
    specs = legacy_events() if args.legacy_events else rule_events(args.events)
    print(f"{len(specs)} events ({'legacy' if args.legacy_events else 'rule'}), adapter {adapter}")

    paths, notes = run_experiment(
        forecaster,
        adapter,
        specs,
        series,
        config.RESULTS_DIR,
        args.context_len,
        args.window_before,
        args.window_after,
    )
    for n in notes:
        print("note:", n)
    summary = pd.read_parquet(paths[2])
    print(
        summary[
            summary["metric"].isin(["promotion", f"lag_D_naive@{shock.HEADLINE_MULTIPLIER}"])
        ].to_string(index=False)
    )

    manifest = write_run_manifest(
        "exp_shock_v2",
        paths,
        as_of=args.as_of,
        checkpoint=checkpoint,
        seed=shock.BOOT_SEED,
        git_dirty=git_dirty,
        extra={"args": {k: str(v) for k, v in vars(args).items()}, "notes": notes},
    )
    print(f"Manifest written to {manifest}")


if __name__ == "__main__":
    main()
