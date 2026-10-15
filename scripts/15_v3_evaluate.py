#!/usr/bin/env python3
# scripts/15_v3_evaluate.py
"""Evaluate the preregistered v3 studies (S1, S2, S3) from the Kaggle forecasts.

Follows docs/analysis-plan-v3.md (sections 4-7 and 10) exactly: it only calls the
frozen evaluation functions (tfm3lab.studies.eval_s1 / eval_s2 / eval_s3) on

  * the downloaded forecasts (`--forecasts`: block=<k>/part-*.parquet plus
    block_status.json, as written by scripts/14_v3_forecast.py),
  * the local requests file (`--requests`), and
  * the quote matrix of the frozen SKUs (configs/v3/*.csv) over the full local
    panel range, outcomes included.

Checks before anything is evaluated (a failure refuses the run):
  * the requests sha256 stamped in every block record and every part equals the
    digest of the local requests file, and every part and block record carries one
    checkpoint revision (recorded in the manifest) and the official inference flags;
  * the frozen units in configs/v3 are exactly the units of the requests file;
  * the forecasts of every complete block hold exactly steps 1..horizon per request
    (requests.validate_forecasts).

Interruption policy (section 7): a block is complete iff its status is `complete`
and every request has verified forecasts. A study with ANY incomplete block (S1 spans
blocks 1 and 4, S2 block 2, S3 block 3) is reported `incomplete`; nothing of that
study is evaluated, resampled or cherry-picked.

Writes (committed; ids, dates, counts, errors, relative advantages, flags -- never
a price or a per-SKU quote), under results/:

    v3_s1_events.parquet   per event x k x floor: counts, mean ALE per method, diffs
    v3_s1_summary.json
    v3_s2_decisions.parquet  per decision: ids, dates, policy days, advantages, flags
    v3_s2_summary.json
    v3_s3_events.parquet   per event: ids, ALE per method per horizon, coverage flags
    v3_s3_summary.json
    v3_verdicts.json       one record per confirmatory primary
    manifests/exp-v3-evaluate-<as_of>-<UTC>.json  inputs and outputs with sha256

Floats are never rounded in code; JSON carries full precision (NaN -> null).
A guard refuses to write any column or key named like a raw price and any float value
that equals a quote of the row's own SKU.

Usage:
    uv run --no-sync scripts/15_v3_evaluate.py --as-of 2026-09-04 \
        --forecasts results/benchmark/v3_forecasts
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tfm3lab import config
from tfm3lab.manifest import capture_git_state, write_manifest
from tfm3lab.studies import eval_s1, eval_s2, eval_s3, inference, runner
from tfm3lab.studies import requests as rq
from tfm3lab.studies.selection import _panel_files, load_quote_data

SCRIPT = "exp-v3-evaluate"
STUDIES = ("S1", "S2", "S3")
S1_FLOORS = (1.0, 0.25)  # primary floor_1, secondary $0.25

_META_SHA = runner._META_REQUESTS_SHA
_META_REVISION = runner._META_REVISION
_META_SYMMETRIC = runner._META_SYMMETRIC
_META_POSITIVE = runner._META_POSITIVE

# Column / key names that may not appear in a committed output. `price_band` is the
# preregistered coarse band label (1-5, 5-20, 20+), not a price.
_BANNED_TOKENS = ("price", "actual", "market", "quote")
_BANNED_SUBSTRINGS = ("q_origin", "q_target")
_ALLOWED_NAMES = ("price_band", "by_price_band")


# ------------------------------------------------------------------ guards


def check_name(name: str) -> None:
    low = str(name).lower()
    for allowed in _ALLOWED_NAMES:
        low = low.replace(allowed, "")
    tokens = [t for t in re.split(r"[^a-z0-9]+", low) if t]
    if any(s in low for s in _BANNED_SUBSTRINGS) or any(
        t.startswith(b) for t in tokens for b in _BANNED_TOKENS
    ):
        raise ValueError(f"output name {name!r} looks like a raw price; refusing to write it")


def assert_no_prices(
    df: pd.DataFrame, label: str, qmat: np.ndarray | None = None, sku_index=None
) -> None:
    """Refuse a frame with a price-like column name, a currency string, or (for a
    per-SKU table) a float value that equals a quote of that row's own SKU."""
    for col in df.columns:
        check_name(col)
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str)
            if sample.str.contains(r"\$").any():
                raise ValueError(f"{label}.{col}: contains a currency string")
    if qmat is None or not {"product_id", "subtype"} <= set(df.columns):
        return
    row_of = {(int(k[0]), eval_s1._sub(k[1])): i for i, k in enumerate(sku_index)}
    floats = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    if not floats:
        return
    rows = np.array(
        [
            row_of.get((int(p), eval_s1._sub(s)), -1)
            for p, s in zip(df["product_id"], df["subtype"], strict=True)
        ]
    )
    for i in np.flatnonzero(rows >= 0):
        q = qmat[rows[i]]
        levels = np.unique(q[np.isfinite(q) & (q > 0) & (q != 1.0)])
        vals = df[floats].iloc[i].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals != 0.0) & (vals != 1.0)]
        if vals.size and np.isin(vals, levels).any():
            raise ValueError(f"{label}: row {int(i)} holds an absolute price level")


# ------------------------------------------------------------------ io helpers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_stale(path: Path) -> None:
    """An incomplete study must not leave an earlier run's table next to its summary."""
    path.unlink(missing_ok=True)


def jsonable(obj):
    """Plain-Python tree with full float precision: tuples -> lists, numpy scalars
    -> Python, DataFrames -> records, NaN/inf -> null. Nothing is rounded."""
    if isinstance(obj, pd.DataFrame):
        return jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, dict):
        for key in obj:
            check_name(key)
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.date().isoformat() if obj == obj.normalize() else obj.isoformat()
    if isinstance(obj, np.bool_ | bool):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    if obj is pd.NaT or obj is None:
        return None
    return obj


def write_json(obj, path: Path) -> Path:
    text = json.dumps(jsonable(obj), indent=2, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def write_parquet(df: pd.DataFrame, path: Path, label: str, **guard) -> Path:
    assert_no_prices(df, label, **guard)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


# ------------------------------------------------------------------ provenance and completeness


def read_status(forecasts_dir: Path) -> dict:
    path = forecasts_dir / runner.STATUS_FILE
    if not path.is_file():
        raise ValueError(f"{path} is missing: not a v3_forecasts output directory")
    return json.loads(path.read_text(encoding="utf-8"))


def check_provenance(forecasts_dir: Path, status: dict, digest: str) -> dict:
    """Refuse forecasts not made from the local requests, or mixing revisions or
    inference settings. Returns the provenance to record."""
    if not status:
        raise ValueError("block_status.json lists no block")
    revisions = set()
    for block, rec in status.items():
        if rec.get("requests_sha256") != digest:
            raise ValueError(
                f"block {block}: requests sha256 {str(rec.get('requests_sha256'))[:12]} in "
                f"block_status.json differs from the local requests digest {digest[:12]}"
            )
        revision = rec.get("checkpoint_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(f"block {block}: no checkpoint revision recorded")
        revisions.add(revision)
        for flag in ("use_symmetric_averaging", "make_positive"):
            if rec.get(flag) is not True:
                raise ValueError(f"block {block}: {flag} is {rec.get(flag)!r}, expected true")
    if len(revisions) != 1:
        raise ValueError(f"blocks were forecast at different checkpoint revisions: {revisions}")
    revision = revisions.pop()
    parts = sorted(forecasts_dir.glob("block=*/part-*.parquet"))
    for part in parts:
        meta = pq.read_schema(part).metadata or {}
        got = {
            "requests sha256": (meta.get(_META_SHA), digest),
            "checkpoint revision": (meta.get(_META_REVISION), revision),
            "use_symmetric_averaging": (meta.get(_META_SYMMETRIC), b"true"),
            "make_positive": (meta.get(_META_POSITIVE), b"true"),
        }
        for what, (found, expected) in got.items():
            if found is None:
                raise ValueError(f"{part}: no {what} recorded")
            if found.decode() != (expected if isinstance(expected, str) else expected.decode()):
                raise ValueError(f"{part}: {what} {found.decode()[:12]!r} does not match the run")
    return {
        "checkpoint_revision": revision,
        "use_symmetric_averaging": True,
        "make_positive": True,
        "parts": [
            {"path": str(p.relative_to(forecasts_dir)), "sha256": sha256_file(p)} for p in parts
        ],
    }


def block_completeness(requests_df: pd.DataFrame, status: dict) -> dict[int, bool]:
    """{block: complete} for every block of the requests file."""
    out = {}
    for block, group in requests_df.groupby("block"):
        rec = status.get(str(int(block)))
        out[int(block)] = bool(
            rec
            and rec.get("status") == "complete"
            and int(rec.get("done", -1)) == int(rec.get("requests", -2)) == len(group)
        )
    return out


def study_status(requests_df: pd.DataFrame, complete: dict[int, bool]) -> dict[str, dict]:
    out = {}
    for study in STUDIES:
        blocks = sorted({int(b) for b in requests_df.loc[requests_df["study"] == study, "block"]})
        missing = [b for b in blocks if not complete.get(b, False)]
        out[study] = {
            "blocks": blocks,
            "incomplete_blocks": missing,
            "status": "incomplete" if missing or not blocks else "complete",
        }
    return out


# ------------------------------------------------------------------ configs


def read_configs(configs_dir: Path) -> dict[str, pd.DataFrame]:
    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(configs_dir / name, keep_default_na=False)

    s1, s2, s3 = read("s1_skus.csv"), read("s2_decisions.csv"), read("s3_events.csv")
    s1["event_date"] = pd.to_datetime(s1["event_date"])
    s2["origin"] = pd.to_datetime(s2["origin"])
    s3["event_date"] = pd.to_datetime(s3["event_date"])
    for df in (s1, s2, s3):
        df["product_id"] = df["product_id"].astype(np.int64)
    return {"S1": s1, "S2": s2, "S3": s3}


def unit_request_ids(units: dict[str, pd.DataFrame]) -> dict[str, set[str]]:
    s1, s2, s3 = units["S1"], units["S2"], units["S3"]
    return {
        "S1": {
            eval_s1.request_id(k, r, p, s)
            for k, r, p, s in zip(
                s1["k"], s1["event_date"], s1["product_id"], s1["subtype"], strict=True
            )
        },
        "S2": {eval_s2.request_id(d) for d in s2["decision_id"]},
        "S3": {eval_s3.request_id(e) for e in s3["event_id"]},
    }


def check_units(requests_df: pd.DataFrame, units: dict[str, pd.DataFrame]) -> None:
    ids = unit_request_ids(units)
    for study in STUDIES:
        theirs = set(requests_df.loc[requests_df["study"] == study, "request_id"])
        if ids[study] != theirs:
            only_cfg, only_req = ids[study] - theirs, theirs - ids[study]
            raise ValueError(
                f"{study}: frozen units differ from the requests file "
                f"({len(only_cfg)} only in configs, {len(only_req)} only in requests)"
            )


# ------------------------------------------------------------------ evaluation


def s1_tables(per: dict[float, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for floor, per_sku in per.items():
        for k in sorted(per_sku["k"].unique()):
            sub = per_sku[per_sku["k"] == k]
            ev = eval_s1.event_summary(sub)
            co = eval_s1.event_summary(sub, with_cohort=True)
            ev["n_paired_cohort"] = co["n_paired"].to_numpy()
            ev["qualifies_cohort"] = co["qualifies"].to_numpy()
            ev["diff_tfm_cohort_paired"] = co["diff_tfm_cohort"].to_numpy()
            ev.insert(0, "floor", float(floor))
            frames.append(ev)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["floor", "k", "event_date"], kind="mergesort").reset_index(drop=True)


def evaluate_s1(units, fc, dates, sku_index, qmat) -> dict:
    per = {
        floor: eval_s1.sku_errors(units, dates, sku_index, qmat, fc, floor=floor)
        for floor in S1_FLOORS
    }
    f1, f025 = per[S1_FLOORS[0]], per[S1_FLOORS[1]]
    primary = eval_s1.primary_result(f1)
    return {
        "events": s1_tables(per),
        "primary": primary,
        "exploratory": eval_s1.exploratory_results(f1, f025),
        "counts": {
            "n_units": len(units),
            "n_sku_floor_1": len(f1),
            "n_sku_floor_025": len(f025),
            "n_tf_unavailable_floor_1": int((f1["has_outcome"] & ~f1["tf_available"]).sum()),
        },
    }


def verdict_records(res: dict[str, dict], statuses: dict[str, dict], s3_confirmatory) -> list:
    out = []
    for study, estimand, unit in (
        ("S1", "mean over events of ALE_TimesFM - ALE_naive (k=14, floor 1.00)", "events"),
        ("S2", "mean advantage A of P_TFM vs buying now", "decisions"),
        ("S3", "mean over events of ALE_TimesFM - ALE_hold (h=28)", "events"),
    ):
        rec = {
            "study": study,
            "estimand": estimand,
            "unit": unit,
            "status": statuses[study]["status"],
            "incomplete_blocks": statuses[study]["incomplete_blocks"],
        }
        if statuses[study]["status"] != "complete":
            rec.update(n=None, point=None, level=None, intervals=None, p=None, verdict="incomplete")
        else:
            p = res[study]["primary"]
            if study == "S1":
                rec.update(
                    n=p["G"],
                    point=p["mean_diff"],
                    level=inference.CI_CONFIRMATORY,
                    intervals={"t": p["t_ci_9833"]},
                    p=p["sign_flip_p"],
                    verdict=p["verdict"],
                )
            else:
                rec.update(
                    n=p["n"],
                    point=p["mean"],
                    level=p["level"],
                    intervals={"cluster": p["cluster_ci"], "block": p["block_ci"]},
                    p=None,
                    verdict=p["verdict"],
                )
                if study == "S3":
                    rec["confirmatory"] = bool(s3_confirmatory)
        out.append(rec)
    return out


# ------------------------------------------------------------------ cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--as-of", required=True, type=dt.date.fromisoformat, metavar="YYYY-MM-DD")
    parser.add_argument(
        "--forecasts", required=True, type=Path, help="v3_forecasts dir (block=<k>/, status)"
    )
    parser.add_argument("--panel", type=Path, default=None, help="default <cache>/panel")
    parser.add_argument(
        "--requests", type=Path, default=None, help="default <cache>/v3_requests.parquet"
    )
    parser.add_argument("--configs-dir", type=Path, default=None, help="default configs/v3")
    parser.add_argument("--census", type=Path, default=None, help="default results/v3_census.json")
    parser.add_argument(
        "--out-root", type=Path, default=None, help="write results/ under this root (tests)"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    as_of: dt.date = args.as_of
    git_dirty = capture_git_state()
    began = time.monotonic()

    def log(message: str) -> None:
        print(f"[{time.monotonic() - began:6.0f}s] {message}", file=sys.stderr)

    root = args.out_root if args.out_root is not None else config.REPO_ROOT
    results_dir = config.RESULTS_DIR if args.out_root is None else root / "results"
    panel_dir = args.panel if args.panel is not None else config.CACHE_DIR / "panel"
    requests_path = (
        args.requests if args.requests is not None else config.CACHE_DIR / "v3_requests.parquet"
    )
    configs_dir = args.configs_dir if args.configs_dir is not None else root / "configs" / "v3"
    census_path = args.census if args.census is not None else results_dir / "v3_census.json"

    # --- inputs and their validation -------------------------------------------------
    requests_df = rq.read_requests(requests_path)
    digest = rq.requests_digest(requests_df)
    status = read_status(args.forecasts)
    provenance = check_provenance(args.forecasts, status, digest)
    complete = block_completeness(requests_df, status)
    studies = study_status(requests_df, complete)
    log(f"blocks complete: {complete}; studies: { {s: v['status'] for s, v in studies.items()} }")

    units = read_configs(configs_dir)
    check_units(requests_df, units)
    census = json.loads(census_path.read_text(encoding="utf-8"))
    s3_confirmatory = bool(census["s3"]["confirmatory"])

    try:
        fc_all = rq.read_forecasts(args.forecasts)
    except FileNotFoundError:  # nothing forecast yet: every study is incomplete
        fc_all = pd.DataFrame({c: [] for c in rq.FORECAST_COLUMNS})
    block_of = requests_df.set_index("request_id")["block"]
    unknown = ~fc_all["request_id"].isin(block_of.index)
    if unknown.any():
        raise ValueError(
            f"forecasts for request_id {fc_all['request_id'][unknown].iloc[0]!r}, "
            "which is not in the requests file"
        )
    in_complete = fc_all["request_id"].map(block_of).map(complete).astype(bool)
    fc = fc_all[in_complete].reset_index(drop=True)
    done_requests = requests_df[requests_df["block"].map(complete).astype(bool)]
    if len(done_requests):
        rq.validate_forecasts(done_requests, fc)
    log(f"forecasts validated: {fc['request_id'].nunique()} requests")

    # --- quotes of the frozen SKUs over the full panel range -------------------------
    todo = [s for s in STUDIES if studies[s]["status"] == "complete"]
    results: dict[str, dict] = {}
    outputs: list[Path] = []
    guard: dict = {}
    panel_input: dict | None = None
    if todo:
        pids = sorted({int(p) for s in todo for p in units[s]["product_id"]})
        data = load_quote_data(panel_dir, pids, end=as_of)
        log(f"panel: {len(data.skus)} SKUs x {len(data.dates)} days (to {as_of})")
        guard = {"qmat": data.qmat, "sku_index": data.skus}
        day_files = _panel_files(panel_dir, as_of)
        day_digest = hashlib.sha256()
        for _, day_path in sorted(day_files, key=lambda t: t[1].name):
            day_digest.update(f"{day_path.name}\t{sha256_file(day_path)}\n".encode())
        panel_input = {
            "path": None,  # filled by rel() below
            "first_day": data.dates[0].date().isoformat(),
            "last_day": data.dates[-1].date().isoformat(),
            "n_skus": int(len(data.skus)),
            "n_days": int(len(data.dates)),
            "n_day_files": len(day_files),
            "day_files_sha256": day_digest.hexdigest(),
        }

    def by_study(study: str) -> pd.DataFrame:
        return fc[fc["request_id"].str.startswith(f"{study}|")]

    if "S1" in todo:
        results["S1"] = evaluate_s1(units["S1"], by_study("S1"), data.dates, data.skus, data.qmat)
    if "S2" in todo:
        results["S2"] = eval_s2.evaluate(
            units["S2"], data.dates, data.skus, data.qmat, by_study("S2")
        )
    if "S3" in todo:
        results["S3"] = eval_s3.evaluate(
            units["S3"], data.dates, data.skus, data.qmat, by_study("S3"), s3_confirmatory
        )

    # --- outputs ---------------------------------------------------------------------
    def summary(study: str, body: dict) -> Path:
        base = {"study": study, "as_of": as_of.isoformat(), **studies[study]}
        base["checkpoint_revision"] = provenance["checkpoint_revision"]
        return write_json({**base, **body}, results_dir / f"v3_{study.lower()}_summary.json")

    if "S1" in results:
        r = results["S1"]
        outputs.append(write_parquet(r["events"], results_dir / "v3_s1_events.parquet", "s1"))
        outputs.append(
            summary(
                "S1",
                {"primary": r["primary"], "exploratory": r["exploratory"], "counts": r["counts"]},
            )
        )
    else:
        remove_stale(results_dir / "v3_s1_events.parquet")
        outputs.append(summary("S1", {}))
    if "S2" in results:
        r = results["S2"]
        outputs.append(
            write_parquet(r["table"], results_dir / "v3_s2_decisions.parquet", "s2", **guard)
        )
        outputs.append(
            summary(
                "S2",
                {
                    "primary": r["primary"],
                    "exploratory": r["exploratory"],
                    "policies": r["policies"],
                    "counts": r["counts"],
                },
            )
        )
    else:
        remove_stale(results_dir / "v3_s2_decisions.parquet")
        outputs.append(summary("S2", {}))
    if "S3" in results:
        r = results["S3"]
        outputs.append(
            write_parquet(r["table"], results_dir / "v3_s3_events.parquet", "s3", **guard)
        )
        outputs.append(
            summary(
                "S3",
                {
                    "confirmatory": s3_confirmatory,
                    "primary": r["primary"],
                    "exploratory": r["exploratory"],
                    "counts": r["counts"],
                },
            )
        )
    else:
        remove_stale(results_dir / "v3_s3_events.parquet")
        outputs.append(summary("S3", {}))
    outputs.append(
        write_json(
            verdict_records(results, studies, s3_confirmatory), results_dir / "v3_verdicts.json"
        )
    )

    # --- manifest --------------------------------------------------------------------
    def rel(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            return path.name

    if panel_input is not None:
        panel_input["path"] = rel(panel_dir)
    config_files = sorted(configs_dir.glob("*.csv"))
    payload = {
        "script": "15_v3_evaluate",
        "parameters": {
            "as_of": as_of.isoformat(),
            "seed": inference.SEED,
            "bootstrap_B": inference.B,
            "s1_floors": list(S1_FLOORS),
            "s3_confirmatory": s3_confirmatory,
        },
        "inputs": {
            "requests": {"path": requests_path.name, "digest": digest, "n": len(requests_df)},
            "checkpoint_revision": provenance["checkpoint_revision"],
            "use_symmetric_averaging": provenance["use_symmetric_averaging"],
            "make_positive": provenance["make_positive"],
            "forecast_parts": provenance["parts"],
            "block_status": {
                "path": runner.STATUS_FILE,
                "sha256": sha256_file(args.forecasts / runner.STATUS_FILE),
            },
            "panel": panel_input,
            "configs": [{"path": p.name, "sha256": sha256_file(p)} for p in config_files],
            "census": {"path": census_path.name, "sha256": sha256_file(census_path)},
        },
        "blocks_complete": {str(b): c for b, c in sorted(complete.items())},
        "studies": studies,
        "outputs": [
            {"path": rel(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in outputs
        ],
    }
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest = write_manifest(
        payload,
        results_dir / "manifests" / f"{SCRIPT}-{as_of.isoformat()}-{ts}.json",
        as_of=as_of,
        seed=inference.SEED,
        git_dirty=git_dirty,
        checkpoint={"revision_resolved": provenance["checkpoint_revision"]},
    )
    verdicts = {s: v["status"] for s, v in studies.items()}
    print(f"v3 evaluate: {verdicts}; manifest {rel(manifest)}")


if __name__ == "__main__":
    main()
