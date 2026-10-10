#!/usr/bin/env python3
"""Select the Experiment B event list by rule (Preregistration v2 section 3).

Fetches (or reuses) daily closes for the five detector indices from
2018-01-01 to --as-of, applies the frozen rule (trailing-250-day |z| >= 4,
20-trading-day declustering, arms, window-fit exclusion) and writes

  configs/shock_events_v2.csv                      (no prices)
  results/manifests/shock-events-v2-<as_of>.json   (data sha256, rule parameters)

Declustering chains from the previous shock day: a shock day within 20
trading days (on the union calendar of the five indices) of the PREVIOUS
shock day joins that cluster, so the gap is measured from the previous shock
day, not from the cluster's onset; a cluster can therefore span more than 20
trading days. Each cluster's onset is its event.

Closes are cached under data/cache (gitignored) as shock_v2_<label>_<as_of>.parquet;
scripts/03_exp_shock.py reads the same cache. If any ticker returns no data
the script exits with an error and writes NOTHING -- it never substitutes data
or relaxes the rule.

Usage:
    uv run scripts/03a_select_events.py --as-of 2026-09-04
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from pathlib import Path

import pandas as pd

from tfm3lab import config, shock
from tfm3lab.data import market
from tfm3lab.manifest import write_manifest

START = "2018-01-01"
CSV_COLUMNS = [
    "event_id",
    "onset_date",
    "arm",
    "onset_index",
    "triggering_indices",
    "max_abs_z",
    "excluded_reason",
]
EVENTS_CSV = Path(__file__).resolve().parents[1] / "configs" / "shock_events_v2.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--as-of", required=True, type=dt.date.fromisoformat, metavar="YYYY-MM-DD")
    parser.add_argument("--refetch", action="store_true", help="ignore cached closes")
    parser.add_argument("--output", type=str, default=None, help="override the CSV path (tests)")
    return parser


def series_sha256(s: pd.Series) -> str:
    text = "\n".join(f"{d.date().isoformat()},{v:.10g}" for d, v in s.items())
    return hashlib.sha256(text.encode()).hexdigest()


def load_or_fetch(as_of: dt.date, refetch: bool) -> dict[str, pd.Series]:
    cache = config.CACHE_DIR
    paths = {label: market.index_cache_path(cache, label, as_of) for label in market.INDEX_TICKERS}
    if not refetch and all(p.exists() for p in paths.values()):
        out = {}
        for label, p in paths.items():
            df = pd.read_parquet(p)
            out[label] = pd.Series(df["close"].to_numpy(), index=pd.to_datetime(df["date"]))
        return out
    end = (as_of + dt.timedelta(days=1)).isoformat()  # yfinance end is exclusive
    closes = market.fetch_index_closes(start=START, end=end)
    closes = {k: v[v.index <= pd.Timestamp(as_of)] for k, v in closes.items()}
    empty = [k for k, v in closes.items() if len(v) == 0]
    if empty:
        raise SystemExit(f"error: no data for {empty}; nothing written")
    market.save_index_closes(closes, cache, as_of)
    return closes


def select_events(
    closes: dict[str, pd.Series], context_len: int = shock.CONTEXT_LEN
) -> pd.DataFrame:
    rows = []
    for label, s in closes.items():
        pos, z = market.detect_trailing_shock_days(s.to_numpy(dtype=float))
        for p, zz in zip(pos, z, strict=True):
            rows.append({"date": s.index[p], "index": label, "z": float(zz)})
    shock_days = pd.DataFrame(rows, columns=["date", "index", "z"])
    calendar = sorted(set().union(*[set(s.index) for s in closes.values()]))
    events = shock.decluster(shock_days, calendar)

    out = []
    for ev in events:
        s = closes[ev["onset_index"]]
        pos = s.index.get_loc(pd.Timestamp(ev["onset_date"]))
        arm = shock.assign_arm(ev["onset_date"])
        reason = ""
        if arm == "guard":
            reason = "guard_band"
        elif not shock.window_fits(pos, len(s), context_len):
            reason = "window_does_not_fit"
        out.append(
            {
                "event_id": shock.event_id_for(ev["onset_date"]),
                "onset_date": ev["onset_date"].isoformat(),
                "arm": arm,
                "onset_index": ev["onset_index"],
                "triggering_indices": ";".join(ev["triggering_indices"]),
                "max_abs_z": round(ev["max_abs_z"], 4),
                "excluded_reason": reason,
            }
        )
    return pd.DataFrame(out, columns=CSV_COLUMNS)


DECLUSTERING_NOTE = "chained from previous shock day, gap <= 20 trading days on the union calendar"


def main() -> None:
    args = build_parser().parse_args()
    closes = load_or_fetch(args.as_of, args.refetch)
    events = select_events(closes)
    target = EVENTS_CSV if args.output is None else Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(target, index=False)

    used = events[events["excluded_reason"] == ""]
    counts = {
        "candidates_by_arm": events["arm"].value_counts().to_dict(),
        "used_by_arm": used["arm"].value_counts().to_dict(),
        "excluded_by_reason": events[events["excluded_reason"] != ""]["excluded_reason"]
        .value_counts()
        .to_dict(),
    }
    payload = {
        "script": "03a_select_events",
        "events_csv": str(target),
        "events_csv_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "data_sha256": {k: series_sha256(v) for k, v in closes.items()},
        "data_ranges": {
            k: [v.index[0].date().isoformat(), v.index[-1].date().isoformat(), len(v)]
            for k, v in closes.items()
        },
        "tickers": market.INDEX_TICKERS,
        "rule": {
            "z_threshold": market.SHOCK_Z_THRESHOLD,
            "trailing_window": market.TRAILING_WINDOW,
            "decluster_gap_trading_days": shock.DECLUSTER_GAP,
            "declustering": DECLUSTERING_NOTE,
            "pre_end": shock.PRE_END.isoformat(),
            "post_start": shock.POST_START.isoformat(),
            "context_len": shock.CONTEXT_LEN,
            "window_before": shock.WINDOW_BEFORE,
            "window_after": shock.WINDOW_AFTER,
        },
        "counts": counts,
    }
    manifest = write_manifest(
        payload,
        config.RESULTS_DIR / "manifests" / f"shock-events-v2-{args.as_of.isoformat()}.json",
        as_of=args.as_of,
    )
    print(f"{len(events)} candidate events; counts: {counts}")
    print(f"Wrote {target}\nManifest {manifest}")


if __name__ == "__main__":
    main()
