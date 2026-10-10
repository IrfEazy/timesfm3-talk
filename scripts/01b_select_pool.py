#!/usr/bin/env python3
"""Select the 40-card Experiment A pool exactly per Preregistration v2 section 2.

Scans the cached TCGCSV daily archives (config.RAW_DIR, category 1 only,
streaming one archive at a time) plus the TCGCSV catalog (groups + products
with their Rarity attribute, cached under RAW_DIR/catalog-<as_of>/), applies
the frozen rule in tfm3lab.data.pool, and writes

  configs/benchmark_pool_v2.csv                  (no prices; redistribution option b)
  results/manifests/pool-v2-<as_of>.json         (rule, seed, counts, sha256s)

The only network use is fetching a missing catalog file; archives must already
be cached (scripts/01_fetch_data.py). If the data cannot satisfy the rule the
script exits with an error and writes nothing -- it never relaxes the rule.

Usage:
    uv run scripts/01b_select_pool.py --as-of 2026-09-04
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from tfm3lab import config
from tfm3lab.data import pool
from tfm3lab.data.mtg import (
    _HEADERS,
    DEFAULT_CARDS,
    MAGIC_CATEGORY_ID,
    TCGCSV_ARCHIVE_START,
    TCGCSV_BASE,
    _price_from_row,
    _resolve_subtype_row,
    _session_with_retries,
    load_card_manifest,
)
from tfm3lab.manifest import write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
POOL_CSV = REPO_ROOT / "configs" / "benchmark_pool_v2.csv"
FIRST_DRAW_CSV_SHA256 = "ed0bdd0b32cbe7e92eb7b122fe94f7ee62a468d0e8f21d78c663c0814e85817e"
MANIFEST_DIR = REPO_ROOT / "results" / "manifests"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _archive_path(raw_dir: Path, day: dt.date) -> Path:
    return raw_dir / f"tcgcsv-prices-{day.isoformat()}.ppmd.7z"


# --- archive scan -------------------------------------------------------------


def _scan_archive(path: Path, day: dt.date, first_pids: np.ndarray | None):
    """Returns (pids, prices, ambiguous_pids) for one archive, category 1
    only. Price policy and subtype resolution are those of build_card_series
    (market then mid; unique Normal/None row preferred). A product whose rows
    that day cannot be resolved uniquely is reported in `ambiguous_pids`.
    If `first_pids` is given, output is restricted to those products."""
    import py7zr

    prefix = f"{day.isoformat()}/{MAGIC_CATEGORY_ID}/"
    pids: list[int] = []
    prices: list[float] = []
    ambiguous: list[int] = []
    wanted = None if first_pids is None else set(first_pids.tolist())
    with py7zr.SevenZipFile(path, mode="r") as z:
        targets = [
            i.filename for i in z.list() if i.filename.startswith(prefix) and not i.is_directory
        ]
        with tempfile.TemporaryDirectory() as tmp:
            z.extract(path=tmp, targets=targets)
            for target in targets:
                payload = json.loads((Path(tmp) / target).read_text(encoding="utf-8"))
                by_product: dict[int, list[dict]] = {}
                for row in payload.get("results", []):
                    by_product.setdefault(int(row["productId"]), []).append(row)
                for pid, rows in by_product.items():
                    if wanted is not None and pid not in wanted:
                        continue
                    try:
                        winning = _resolve_subtype_row(pid, rows, day)
                    except ValueError:
                        ambiguous.append(pid)
                        continue
                    price, _ = _price_from_row(winning)
                    if price is not None:
                        pids.append(pid)
                        prices.append(price)
    return (
        np.asarray(pids, dtype=np.int64),
        np.asarray(prices, dtype=np.float64),
        np.asarray(ambiguous, dtype=np.int64),
    )


_WORKER_FIRST: np.ndarray | None = None


def _init_worker(first_pids: np.ndarray) -> None:
    global _WORKER_FIRST
    _WORKER_FIRST = first_pids


def _worker(args: tuple[str, str]):
    path, day = args
    pids, prices, amb = _scan_archive(Path(path), dt.date.fromisoformat(day), _WORKER_FIRST)
    pos = np.searchsorted(_WORKER_FIRST, pids)
    apos = np.searchsorted(_WORKER_FIRST, amb)
    return pos, prices.astype(np.float32), apos


def scan_archives(raw_dir: Path, start: dt.date, as_of: dt.date, workers: int):
    """Streams every cached archive in [start, as_of]. Returns
    (first_pids, observed_days, median_price, ambiguous_mask, archive_days,
    missing_days). Only products observed on the first day are tracked
    (eligibility requires presence in the first archive), which bounds memory
    to n_first x n_days float32."""
    days = [start + dt.timedelta(d) for d in range((as_of - start).days + 1)]
    present = [d for d in days if _archive_path(raw_dir, d).exists()]
    missing = [d for d in days if d not in set(present)]
    if not present or present[0] != start:
        raise SystemExit(f"error: first archive {start} not cached in {raw_dir}")

    print(f"scanning {len(present)} archives ({len(missing)} calendar days missing)...", flush=True)
    pids0, _, _ = _scan_archive(_archive_path(raw_dir, start), start, None)
    first_pids = np.unique(pids0)
    n_first = len(first_pids)
    matrix = np.full((n_first, len(days)), np.nan, dtype=np.float32)
    ambiguous = np.zeros(n_first, dtype=bool)

    jobs = [(str(_archive_path(raw_dir, d)), d.isoformat()) for d in present]
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(first_pids,)
    ) as ex:
        for k, (d, (pos, prices, apos)) in enumerate(
            zip(present, ex.map(_worker, jobs), strict=True)
        ):
            matrix[pos, (d - start).days] = prices
            ambiguous[apos] = True
            if k % 50 == 0:
                print(f"  {k + 1}/{len(present)} {d}", flush=True)

    observed_days = np.isfinite(matrix).sum(axis=1)
    median_price = np.full(n_first, np.nan)
    for i in range(n_first):
        row = matrix[i]
        row = row[np.isfinite(row)]
        if row.size:
            median_price[i] = float(np.median(row.astype(np.float64)))
    return first_pids, observed_days, median_price, ambiguous, present, missing


# --- catalog ------------------------------------------------------------------


def ensure_catalog(catalog_dir: Path) -> dict[str, str]:
    """Reads (fetching only what is missing) the groups and per-group products
    JSON. Returns {filename: sha256}."""
    catalog_dir.mkdir(parents=True, exist_ok=True)
    session = _session_with_retries()

    def get(url: str, dest: Path) -> None:
        if dest.exists() and dest.stat().st_size > 0:
            return
        r = session.get(url, headers=_HEADERS, timeout=60)
        r.raise_for_status()
        part = dest.with_suffix(".part")
        part.write_bytes(r.content)
        os.replace(part, dest)

    groups_path = catalog_dir / "groups.json"
    get(f"{TCGCSV_BASE}/tcgplayer/{MAGIC_CATEGORY_ID}/groups", groups_path)
    group_ids = [int(g["groupId"]) for g in json.loads(groups_path.read_text())["results"]]
    with ThreadPoolExecutor(4) as ex:
        list(
            ex.map(
                lambda gid: get(
                    f"{TCGCSV_BASE}/tcgplayer/{MAGIC_CATEGORY_ID}/{gid}/products",
                    catalog_dir / f"products-{gid}.json",
                ),
                group_ids,
            )
        )
    return {p.name: _sha256(p) for p in sorted(catalog_dir.glob("*.json"))}


def load_catalog(catalog_dir: Path) -> pd.DataFrame:
    groups = json.loads((catalog_dir / "groups.json").read_text())["results"]
    abbr = {int(g["groupId"]): (g.get("abbreviation") or "") for g in groups}
    rows = []
    for g in groups:
        gid = int(g["groupId"])
        for p in json.loads((catalog_dir / f"products-{gid}.json").read_text())["results"]:
            ext = {e["name"]: e["value"] for e in p.get("extendedData", [])}
            rows.append(
                {
                    "product_id": int(p["productId"]),
                    "group_id": gid,
                    "group_abbreviation": abbr[gid],
                    "product_name": p["name"],
                    "rarity": ext.get("Rarity"),
                }
            )
    return pd.DataFrame(rows).drop_duplicates("product_id", keep="first")


# --- main ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--as-of", required=True, help="end of the window, YYYY-MM-DD (required)")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    return ap


def main() -> int:
    args = build_parser().parse_args()
    as_of = dt.date.fromisoformat(args.as_of)
    start = TCGCSV_ARCHIVE_START
    window_days = (as_of - start).days + 1
    raw_dir = config.RAW_DIR

    catalog_dir = raw_dir / f"catalog-{as_of.isoformat()}"
    catalog_sha = ensure_catalog(catalog_dir)
    catalog = load_catalog(catalog_dir)

    first_pids, observed, median_price, ambiguous, present, missing = scan_archives(
        raw_dir, start, as_of, args.workers
    )
    tracked = pd.DataFrame(
        {
            "product_id": first_pids,
            "observed_days": observed,
            "median_price": median_price,
            "ambiguous": ambiguous,
            "present_first_day": True,
        }
    )
    stats = tracked.merge(catalog, on="product_id", how="left")
    n_not_in_catalog = int(stats["product_name"].isna().sum())
    n_ambiguous = int(stats["ambiguous"].sum())
    stats = stats[stats["product_name"].notna() & ~stats["ambiguous"]].copy()
    stats["median_price"] = stats["median_price"].astype(float)

    showcase = frozenset(pool.card_name(c.product_name) for c in DEFAULT_CARDS)
    try:
        sel = pool.select_pool(stats, window_days, showcase_names=showcase)
    except pool.PoolRuleError as exc:
        n_elig = len(pool.eligible_printings(stats, window_days))
        print(
            f"error: rule cannot be satisfied ({exc}); eligible={n_elig}. Nothing written.",
            file=sys.stderr,
        )
        return 2

    out = sel.selected.copy()
    out["price_tertile"] = out["price_tertile"].astype(int)
    csv_df = out[list(pool.POOL_COLUMNS)]
    POOL_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(POOL_CSV, index=False, lineterminator="\n")

    loaded = load_card_manifest(POOL_CSV)
    assert [c.product_id for c in loaded] == [int(x) for x in out["product_id"]], (
        "CSV round-trip mismatch"
    )
    csv_sha = _sha256(POOL_CSV)

    funnel = {
        "tracked_first_day": int(len(first_pids)),
        "not_in_catalog": n_not_in_catalog,
        "ambiguous_subtype_excluded": n_ambiguous,
        "single_cards": int(
            sum(
                pool.is_single_card(r, n, g)
                for r, n, g in zip(
                    stats["rarity"], stats["product_name"], stats["group_abbreviation"], strict=True
                )
            )
        ),
        "eligible_after_dedup": int(len(sel.eligible)),
    }
    payload = {
        "kind": "pool-v2",
        "preregistration": "docs/analysis-plan.md section 2 (v2)",
        "rule": {
            "seed": pool.SEED,
            "pool_size": pool.POOL_SIZE,
            "min_per_stratum": pool.MIN_PER_STRATUM,
            "min_observed_share": pool.MIN_OBSERVED_SHARE,
            "min_median_price_usd": pool.MIN_MEDIAN_PRICE,
            "window": [start.isoformat(), as_of.isoformat()],
            "window_days": window_days,
            "price_policy": "market then mid; unique Normal/None subtype row",
            "rarity_groups": {"M": "mythic", "R": "rare", "other": "Uncommon + Common"},
            "eligible_rarity_codes": pool.ELIGIBLE_RARITY_CODES,
            "excluded_group_abbreviations": list(pool.EXCLUDED_GROUP_ABBREVIATIONS),
            "excluded_name_substrings": list(pool.EXCLUDED_NAME_SUBSTRINGS),
            "tertiles": "1/3 and 2/3 quantiles of median price over the eligible set",
        },
        "funnel": funnel,
        "eligible_count": int(len(sel.eligible)),
        "eligible_per_stratum": sel.eligible_per_stratum,
        "allocation": sel.allocation,
        "archives": {
            "date_range": [present[0].isoformat(), present[-1].isoformat()],
            "count": len(present),
            "missing_calendar_days": [d.isoformat() for d in missing],
        },
        "catalog": {"dir": f"data/raw/catalog-{as_of.isoformat()}", "sha256": catalog_sha},
        "pool_csv": {"path": "configs/benchmark_pool_v2.csv", "sha256": csv_sha},
        "amendment": {
            "date": "2026-10-10",
            "reason": (
                "The first literal draw admitted non-game collectibles (World Championship "
                "Deck decklist/biography cards, an oversize card, a serial-numbered card, a "
                "LARP promo). Only item types and names were inspected, no prices or "
                "forecasts, and no model run existed. Eligibility was amended before any "
                "model run: Rarity in {Mythic, Rare, Uncommon, Common}; exclude World "
                "Championship Decks and names containing Oversize, Oversized, Serial "
                "Numbered, Decklist, Biography, Checklist. Same script, seed and as_of."
            ),
            "first_draw_csv_sha256": FIRST_DRAW_CSV_SHA256,
            "amended_csv_sha256": csv_sha,
            "eligible_rarity_codes": pool.ELIGIBLE_RARITY_CODES,
            "excluded_group_abbreviations": list(pool.EXCLUDED_GROUP_ABBREVIATIONS),
            "excluded_name_substrings": list(pool.EXCLUDED_NAME_SUBSTRINGS),
            "disclosed_limitation": (
                "The TCGCSV catalog was fetched 2026-10-10, after as_of 2026-09-04."
            ),
        },
        "selected_product_ids": [int(x) for x in out["product_id"]],
        "showcase_drawn": [str(x) for x in out.loc[out["is_showcase"], "label"]],
    }
    path = write_manifest(payload, MANIFEST_DIR / f"pool-v2-{as_of.isoformat()}.json", as_of=as_of)
    print(f"eligible={len(sel.eligible)} selected={len(out)}")
    print(f"wrote {POOL_CSV} (sha256 {csv_sha})")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
