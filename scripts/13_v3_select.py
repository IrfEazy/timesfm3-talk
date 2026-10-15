#!/usr/bin/env python3
"""Freeze the unit selection of the three v3 studies (S1, S2, S3) and build the
forecast requests. No outcome is ever read: every rule uses only data up to its
unit's own origin (see src/tfm3lab/studies/selection.py).

Reads the local per-day panel (scripts/12_build_panel.py) for the single-card
SKUs only, and the cached catalogue snapshot. Writes, under the repo root:

    configs/v3/s1_skus.csv          S1 units (ids and card attributes, no prices)
    configs/v3/release_events.csv   release events of the S1 window
    configs/v3/s2_decisions.csv     accepted S2 decisions
    configs/v3/s3_events.csv        accepted S3 events
    results/v3_census.json          counts only
    data/cache/v3_requests.parquet  scaled-context forecast requests (gitignored)
    results/manifests/v3-select-<as_of>-<UTC>.json  parameters, seeds, git state,
                                    sha256 of every output, counts

Usage:
    uv run --no-sync scripts/13_v3_select.py --as-of 2026-09-04
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tfm3lab import config
from tfm3lab.manifest import capture_git_state, write_manifest
from tfm3lab.studies import selection as sel
from tfm3lab.studies import universe
from tfm3lab.studies.requests import write_requests

SCRIPT = "v3-select"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--as-of",
        required=True,
        type=dt.date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="catalogue / panel snapshot date (names the default catalogue directory)",
    )
    parser.add_argument(
        "--panel", type=Path, default=None, help="panel directory (default <cache>/panel)"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="catalogue directory (default <raw>/catalog-<as-of>)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="write configs/, results/ and data/cache/ under this root instead of the repo",
    )
    return parser


def _csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sku_table(data: sel.QuoteData, cards: pd.DataFrame, first: pd.DataFrame) -> pd.DataFrame:
    """Per-row attributes of the SKUs of `data` (catalogue facts only)."""
    meta = cards.set_index("product_id")
    flag = first.set_index("product_id")["first_printing"]
    pid = np.array([p for p, _ in data.skus], dtype=np.int64)
    out = pd.DataFrame(
        {
            "product_id": pid,
            "subtype": [s for _, s in data.skus],
            "group_id": meta["group_id"].reindex(pid).to_numpy(),
            "group_abbreviation": meta["group_abbreviation"].reindex(pid).to_numpy(),
            "card_id": meta["card_name"].reindex(pid).to_numpy(),
            "rarity": meta["rarity"].reindex(pid).to_numpy(),
            "finish": [universe.finish_class(s) for _, s in data.skus],
            "first_printing": flag.reindex(pid).fillna(False).astype(bool).to_numpy(),
        }
    )
    return out


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    as_of: dt.date = args.as_of
    if as_of < sel.LAST_ORIGIN:
        raise SystemExit(
            f"--as-of {as_of} is before the last evaluation origin {sel.LAST_ORIGIN}; "
            "the panel must cover the selection windows up to that date"
        )
    git_dirty = capture_git_state()
    utc_ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    root = args.out_root if args.out_root is not None else config.REPO_ROOT
    results_dir = config.RESULTS_DIR if args.out_root is None else root / "results"
    cache_dir = config.CACHE_DIR if args.out_root is None else root / "data" / "cache"
    panel_dir = args.panel if args.panel is not None else config.CACHE_DIR / "panel"
    catalog = (
        args.catalog
        if args.catalog is not None
        else config.RAW_DIR / f"catalog-{as_of.isoformat()}"
    )
    began = time.monotonic()

    def log(message: str) -> None:
        print(f"[{time.monotonic() - began:6.0f}s] {message}", file=sys.stderr)

    products = universe.load_catalog_products(catalog)
    groups = universe.load_groups(catalog)
    cards = universe.single_cards(products)
    first = universe.first_printing(products, groups, as_of)
    events = universe.release_events(groups)
    log(f"catalogue: {len(cards)} single cards, {len(events)} release events")

    end = min(as_of, sel.LAST_ORIGIN)
    data = sel.load_quote_data(panel_dir, cards["product_id"], end=end)
    log(f"panel: {len(data.skus)} single-card SKUs x {len(data.dates)} days (to {end})")
    skus = sku_table(data, cards, first)
    group_of_row = skus["group_id"].to_numpy()
    card_of_row = skus["card_id"].to_numpy(dtype=object)

    # S1
    s1_units, s1_counts = sel.select_s1(data, events, group_of_row)
    log(f"S1: {len(s1_units)} units")
    # S2
    window = sel.s2_window(data)
    eligible = sel.s2_eligibility_sets(data, window)
    pool = sel.draw_s2_pool(eligible, window)
    accepted = sel.accept_s2(pool, card_of_row)
    s2_dec = sel.s2_decisions(data, accepted)
    log(f"S2: {sum(len(e) for e in eligible)} eligible pairs, {len(s2_dec)} accepted")
    # S3
    s3_events, s3_counts = sel.select_s3(data, card_of_row)
    log(f"S3: {s3_counts}")

    # Frames
    def attrs(rows: np.ndarray) -> pd.DataFrame:
        return skus.iloc[rows].reset_index(drop=True)

    s1_a = attrs(s1_units["row"].to_numpy(np.int64))
    s1_out = pd.DataFrame(
        {
            "event_date": s1_units["event_date"].dt.date.astype(str),
            "k": s1_units["k"],
            "product_id": s1_a["product_id"],
            "subtype": s1_a["subtype"],
            "group_id": s1_a["group_id"],
            "group_abbreviation": s1_a["group_abbreviation"],
            "card_id": s1_a["card_id"],
            "rarity": s1_a["rarity"],
            "finish": s1_a["finish"],
            "first_printing": s1_a["first_printing"],
            "floor_1": s1_units["floor_1"],
        }
    )
    events_out = pd.DataFrame(
        {
            "event_date": events["release_date"].dt.date.astype(str),
            "group_ids": [";".join(map(str, g)) for g in events["group_ids"]],
            "abbreviations": [";".join(a) for a in events["abbreviations"]],
        }
    )
    s2_a = attrs(s2_dec["row"].to_numpy(np.int64))
    s2_day = data.dates[s2_dec["origin_idx"].to_numpy(np.int64)]
    s2_out = pd.DataFrame(
        {
            "decision_id": [
                sel.unit_id(d, data.skus[r]) for d, r in zip(s2_day, s2_dec["row"], strict=True)
            ],
            "product_id": s2_a["product_id"],
            "subtype": s2_a["subtype"],
            "origin": s2_day.date.astype(str),
            "card_id": s2_a["card_id"],
            "rarity": s2_a["rarity"],
            "finish": s2_a["finish"],
            "price_band": s2_dec["price_band"],
        }
    )
    s3_a = attrs(s3_events["row"].to_numpy(np.int64))
    s3_day = data.dates[s3_events["origin_idx"].to_numpy(np.int64)]
    s3_out = pd.DataFrame(
        {
            "event_id": [
                sel.unit_id(d, data.skus[r]) for d, r in zip(s3_day, s3_events["row"], strict=True)
            ],
            "product_id": s3_a["product_id"],
            "subtype": s3_a["subtype"],
            "event_date": s3_day.date.astype(str),
            "card_id": s3_a["card_id"],
            "rarity": s3_a["rarity"],
            "finish": s3_a["finish"],
        }
    )

    census = sel.build_census(
        data, s1_counts, eligible, window, accepted, s2_dec, s3_events, s3_counts, card_of_row
    )
    req = sel.build_requests(data, s1_units, s2_dec, s3_events)

    configs_dir = root / "configs" / "v3"
    outputs = [
        _csv(s1_out, configs_dir / "s1_skus.csv"),
        _csv(events_out, configs_dir / "release_events.csv"),
        _csv(s2_out, configs_dir / "s2_decisions.csv"),
        _csv(s3_out, configs_dir / "s3_events.csv"),
    ]
    census_path = results_dir / "v3_census.json"
    census_path.parent.mkdir(parents=True, exist_ok=True)
    census_path.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs.append(census_path)
    requests_path = write_requests(req, cache_dir / "v3_requests.parquet")
    outputs.append(requests_path)
    log(f"{len(req)} requests written")

    def rel(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            return str(path)

    def portable(path: Path) -> str:
        # Repo-relative when inside the repository, else just the directory name.
        try:
            return str(path.resolve().relative_to(config.REPO_ROOT.resolve()))
        except ValueError:
            return path.resolve().name

    payload = {
        "script": "13_v3_select",
        "parameters": {
            "as_of": as_of.isoformat(),
            "panel": portable(panel_dir),
            "catalog": portable(catalog),
            "panel_end_loaded": end.isoformat(),
            "s1_ks": list(sel.S1_KS),
            "s1_floors": [sel.S1_FLOOR, sel.S1_FLOOR_1],
            "s2_window": [sel.S2_START.isoformat(), sel.S2_END.isoformat()],
            "s2_M": sel.S2_POOL_SIZE,
            "s2_seed": sel.S2_SEED,
            "s3_window": [sel.S3_START.isoformat(), sel.S3_END.isoformat()],
        },
        "outputs": [
            {"path": rel(p), "bytes": p.stat().st_size, "sha256": _sha256(p)} for p in outputs
        ],
        "counts": {
            "single_cards": len(cards),
            "skus": len(data.skus),
            "release_events": len(events),
            "s1_units": len(s1_out),
            "s2_decisions": len(s2_out),
            "s3_events": len(s3_out),
            "requests": len(req),
        },
    }
    manifest = write_manifest(
        payload,
        results_dir / "manifests" / f"{SCRIPT}-{as_of.isoformat()}-{utc_ts}.json",
        as_of=as_of,
        seed=sel.S2_SEED,
        git_dirty=git_dirty,
    )
    print(
        f"v3 selection: S1 {len(s1_out)} units, S2 {len(s2_out)} decisions, "
        f"S3 {len(s3_out)} events, {len(req)} requests; manifest {rel(manifest)}"
    )


if __name__ == "__main__":
    main()
