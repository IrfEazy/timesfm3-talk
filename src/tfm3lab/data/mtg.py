"""Magic: The Gathering card price ingestion — Experiment A's "camera pulita".

TCGCSV (tcgcsv.com) is the primary source: a public daily archive of
TCGplayer's own price snapshots, starting 2024-02-08 — entirely after
TimesFM-3's pretraining cutoff (config.PRETRAIN_CUTOFF). That's the whole
point of this experiment: unlike the market/macro series in experiment B,
this domain is (almost certainly) absent from the model's training data.

Verified live against the service on 2026-09-01 (see scripts/00_probe_tcgcsv.py,
which re-checks this at run time rather than trusting this comment forever):

  - groups catalog:   GET {base}/tcgplayer/{categoryId}/groups
  - products catalog: GET {base}/tcgplayer/{categoryId}/{groupId}/products
  - daily archive:    GET {base}/archive/tcgplayer/prices-{date}.ppmd.7z
      a 7z archive of entries "{date}/{categoryId}/{groupId}/prices", each
      a JSON {"results": [{"productId", "lowPrice", "midPrice", "highPrice",
      "marketPrice", "directLowPrice", "subTypeName"}, ...]}
  - Magic: The Gathering is categoryId 1
  - earliest available archive: 2024-02-08 (matches MTGJSON's own stated
    TCGCSV start date)

If this schema ever changes, `fetch_mtgjson_fallback` below covers the same
cards over MTGJSON's own rolling 90-day `AllPrices` window instead — enough
for a smaller-power version of the same experiment, not a full replacement.
"""

from __future__ import annotations

import datetime as dt
import difflib
import enum
import gzip
import json
import lzma
import tempfile
from dataclasses import dataclass
from pathlib import Path

import ijson
import pandas as pd
import requests

from .. import config
from ..backtest import SeriesData

TCGCSV_BASE = "https://tcgcsv.com"
MAGIC_CATEGORY_ID = 1
TCGCSV_ARCHIVE_START = dt.date(2024, 2, 8)
_HEADERS = {"User-Agent": "tfm3lab-research/1.0 (contact: irfeazy@gmail.com)"}
_TIMEOUT = 30


@dataclass(frozen=True)
class CardSpec:
    """One tracked printing, resolved against a specific TCGCSV group.

    `product_name` must match the TCGplayer product's exact `name` field
    (case-insensitive) — TCGplayer lists frame/border variants ("Ragavan,
    Nimble Pilferer (Borderless)") as separate products, so an inexact
    match would silently mix or miss printings.
    """

    label: str
    group_abbreviation: str
    product_name: str


class PriceSelectionPolicy(enum.Enum):
    """Which TCGplayer price field to trust for one row.

    MARKET_THEN_MID matches this project's original behavior: prefer
    marketPrice, fall back to midPrice when marketPrice is null (thin-volume
    days). MARKET_ONLY/MID_ONLY never fall back — a null preferred field
    means "no price today", not "use the other one".
    """

    MARKET_THEN_MID = "market_then_mid"
    MARKET_ONLY = "market_only"
    MID_ONLY = "mid_only"


@dataclass(frozen=True)
class PricePoint:
    """One resolved price observation: the value, which TCGplayer field it
    came from, and the subTypeName of the winning row (None if the archive
    didn't carry one)."""

    price: float
    field_used: str
    subtype: str | None


# The five headline cards from the original draft, plus two adversarial
# cases the plan calls for: a lower-liquidity legendary (thinner, noisier
# price history than the chase mythics above) and a reprinted utility land
# (typically flat, low-volatility). Verified against the live TCGCSV
# catalog on 2026-09-01 (scripts/00_probe_tcgcsv.py re-checks this) —
# `resolve_card_specs` raises with close-match suggestions if a name ever
# stops matching, rather than silently dropping a card from the experiment.
DEFAULT_CARDS = (
    CardSpec("Ragavan [MH2]", "MH2", "Ragavan, Nimble Pilferer"),
    CardSpec("Urza's Saga [MH2]", "MH2", "Urza's Saga"),
    CardSpec("Sheoldred [DMU]", "DMU", "Sheoldred, the Apocalypse"),
    CardSpec("The One Ring [LTR]", "LTR", "The One Ring"),
    CardSpec("Orcish Bowmasters [LTR]", "LTR", "Orcish Bowmasters"),
    CardSpec("Chatterfang [MH2]", "MH2", "Chatterfang, Squirrel General"),
    CardSpec("Mishra's Factory [MH2]", "MH2", "Mishra's Factory"),
)


def fetch_groups(
    category_id: int = MAGIC_CATEGORY_ID, session: requests.Session | None = None
) -> pd.DataFrame:
    session = session or requests.Session()
    url = f"{TCGCSV_BASE}/tcgplayer/{category_id}/groups"
    r = session.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    return pd.DataFrame(r.json()["results"])


def fetch_products(
    group_id: int, category_id: int = MAGIC_CATEGORY_ID, session: requests.Session | None = None
) -> pd.DataFrame:
    session = session or requests.Session()
    url = f"{TCGCSV_BASE}/tcgplayer/{category_id}/{group_id}/products"
    r = session.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    return pd.DataFrame(r.json()["results"])


def resolve_card_specs(
    cards: tuple[CardSpec, ...] = DEFAULT_CARDS,
    category_id: int = MAGIC_CATEGORY_ID,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Resolves each CardSpec to a concrete (group_id, product_id).

    Raises with close-match suggestions on any miss, rather than silently
    dropping a card — a card missing from the experiment without a logged
    reason is exactly the "no silent caps" failure mode this project's
    plan explicitly rules out.
    """
    session = session or requests.Session()
    groups = fetch_groups(category_id, session)
    products_cache: dict[int, pd.DataFrame] = {}
    rows = []
    for card in cards:
        matches = groups[groups["abbreviation"].str.upper() == card.group_abbreviation.upper()]
        if matches.empty:
            suggestions = difflib.get_close_matches(
                card.group_abbreviation, groups["abbreviation"].dropna().tolist(), n=5, cutoff=0.4
            )
            raise ValueError(
                f"no TCGCSV group with abbreviation '{card.group_abbreviation}' "
                f"for card '{card.label}'. Closest matches: {suggestions}"
            )
        group_id = int(matches.iloc[0]["groupId"])
        if group_id not in products_cache:
            products_cache[group_id] = fetch_products(group_id, category_id, session)
        products = products_cache[group_id]
        exact = products[products["name"].str.casefold() == card.product_name.casefold()]
        if exact.empty:
            suggestions = difflib.get_close_matches(
                card.product_name, products["name"].dropna().tolist(), n=5, cutoff=0.4
            )
            raise ValueError(
                f"'{card.product_name}' not found in group '{card.group_abbreviation}' "
                f"(groupId={group_id}) for card '{card.label}'. Closest matches: {suggestions}"
            )
        product_id = int(exact.iloc[0]["productId"])
        rows.append(
            {
                "label": card.label,
                "group_id": group_id,
                "product_id": product_id,
                "group_abbreviation": card.group_abbreviation,
                "product_name": card.product_name,
            }
        )
    return pd.DataFrame(rows)


def _archive_url(date: dt.date) -> str:
    return f"{TCGCSV_BASE}/archive/tcgplayer/prices-{date.isoformat()}.ppmd.7z"


def probe_archive_available(date: dt.date, session: requests.Session | None = None) -> bool:
    session = session or requests.Session()
    r = session.head(_archive_url(date), headers=_HEADERS, timeout=15)
    return r.status_code == 200


def _price_from_row(
    row: dict, policy: PriceSelectionPolicy = PriceSelectionPolicy.MARKET_THEN_MID
) -> tuple[float | None, str | None]:
    """Resolves one row to (price, field_used) per `policy`. Never
    lowPrice/highPrice — those are listing extremes, not price estimates.
    Returns (None, None) when the policy's field(s) are null.
    """
    market = row.get("marketPrice")
    mid = row.get("midPrice")
    if policy is PriceSelectionPolicy.MARKET_ONLY:
        return (float(market), "market") if market is not None else (None, None)
    if policy is PriceSelectionPolicy.MID_ONLY:
        return (float(mid), "mid") if mid is not None else (None, None)
    if market is not None:
        return float(market), "market"
    if mid is not None:
        return float(mid), "mid"
    return None, None


_PREFERRED_SUBTYPES = ("Normal", None)


def _resolve_subtype_row(product_id: int, rows: list[dict], date: dt.date) -> dict:
    """When one productId has multiple rows in a day's archive (distinct
    subTypeName — e.g. Normal vs Foil bundled together), picks the row this
    project tracks: subTypeName in ("Normal", None). Raises if that
    resolution isn't unique — no silent last-row-wins.
    """
    if len(rows) == 1:
        return rows[0]
    preferred = [r for r in rows if r.get("subTypeName") in _PREFERRED_SUBTYPES]
    if len(preferred) == 1:
        return preferred[0]
    subtypes_seen = [r.get("subTypeName") for r in rows]
    raise ValueError(
        f"ambiguous productId {product_id} on {date.isoformat()}: {len(rows)} rows, "
        f"subTypeName values {subtypes_seen} — no unique 'Normal' row to prefer"
    )


def fetch_daily_prices(
    date: dt.date,
    group_ids: set[int],
    category_id: int = MAGIC_CATEGORY_ID,
    session: requests.Session | None = None,
    raw_dir: Path = config.RAW_DIR,
) -> dict[int, dict[int, float]]:
    """Downloads one day's archive (cached under raw_dir), extracts only
    the requested groups' price files, and returns
    {group_id: {product_id: price}}.

    Price is TCGplayer's `marketPrice`, falling back to `midPrice` when
    marketPrice is null (happens on thin-volume days for less liquid
    printings) — never `lowPrice`/`highPrice`, which are listing extremes,
    not transacted-price estimates.
    """
    import py7zr

    session = session or requests.Session()
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / f"tcgcsv-prices-{date.isoformat()}.ppmd.7z"
    if not archive_path.exists():
        r = session.get(_archive_url(date), headers=_HEADERS, timeout=180)
        r.raise_for_status()
        archive_path.write_bytes(r.content)

    targets = [f"{date.isoformat()}/{category_id}/{gid}/prices" for gid in group_ids]
    out: dict[int, dict[int, float]] = {gid: {} for gid in group_ids}
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        present = [t for t in targets if t in set(z.getnames())]
        if not present:
            return out
        with tempfile.TemporaryDirectory() as tmp:
            z.extract(path=tmp, targets=present)
            for target in present:
                group_id = int(target.split("/")[2])
                payload = json.loads((Path(tmp) / target).read_text(encoding="utf-8"))
                for row in payload.get("results", []):
                    price, _ = _price_from_row(row)
                    if price is not None:
                        out[group_id][int(row["productId"])] = price
    return out


def build_card_series(
    cards: tuple[CardSpec, ...] = DEFAULT_CARDS,
    start: dt.date = TCGCSV_ARCHIVE_START,
    end: dt.date | None = None,
    raw_dir: Path = config.RAW_DIR,
    max_ffill_days: int = 3,
    session: requests.Session | None = None,
) -> list[SeriesData]:
    """Builds one SeriesData per card over [start, end], daily frequency.

    Missing days (archive gaps, or the product simply unlisted that day)
    are forward-filled up to `max_ffill_days`; `SeriesData.observed` marks
    which points are real vs. filled, per the project's rule that filled
    points must never be scored as if they were observations.
    """
    end = end or dt.date.today()
    session = session or requests.Session()
    resolved = resolve_card_specs(cards, session=session)
    group_ids = set(resolved["group_id"])

    date_range = pd.date_range(start, end, freq="D")
    raw = pd.DataFrame(index=date_range, columns=resolved["label"], dtype=float)

    for date in date_range:
        day = date.date()
        try:
            prices_by_group = fetch_daily_prices(day, group_ids, raw_dir=raw_dir, session=session)
        except requests.HTTPError:
            continue  # archive missing for this day — left as NaN, filled below
        for _, row in resolved.iterrows():
            price = prices_by_group.get(row["group_id"], {}).get(row["product_id"])
            if price is not None:
                raw.loc[date, row["label"]] = price

    observed = raw.notna()
    filled = raw.ffill(limit=max_ffill_days)

    series_list = []
    for label in resolved["label"]:
        series_list.append(
            SeriesData(
                name=label,
                values=filled[label].to_numpy(dtype=float),
                dates=filled.index.to_numpy(),
                observed=observed[label].to_numpy(),
            )
        )
    return series_list


# --- MTGJSON fallback (90-day rolling window) --------------------------------


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    r = requests.get(url, stream=True, timeout=180, headers=_HEADERS)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return dest


def _read_json_compressed(path: Path):
    path = str(path)
    if path.endswith(".xz"):
        with lzma.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fetch_mtgjson_fallback(
    cards: tuple[CardSpec, ...] = DEFAULT_CARDS,
    raw_dir: Path = config.RAW_DIR,
    max_ffill_days: int = 3,
) -> list[SeriesData]:
    """Fallback used only if TCGCSV's archive schema breaks: MTGJSON's
    `AllPrices` (rolling 90-day window), TCGplayer retail/normal prices,
    streamed with ijson so the multi-GB decompressed payload is never held
    in memory at once. Much less statistical power than `build_card_series`
    — log that loss explicitly if this path is ever actually exercised.
    """
    uuids: dict[str, str] = {}
    for card in cards:
        set_path = _download(
            f"https://mtgjson.com/api/v5/{card.group_abbreviation.upper()}.json.xz",
            raw_dir / f"mtgjson-{card.group_abbreviation.upper()}.json.xz",
        )
        payload = _read_json_compressed(set_path)
        data = payload.get("data", payload)
        cards_in_set = data.get("cards", [])
        exact = [
            c
            for c in cards_in_set
            if c.get("name", "").casefold() == card.product_name.casefold() and not c.get("isToken")
        ]
        if not exact:
            suggestions = difflib.get_close_matches(
                card.product_name, [c.get("name", "") for c in cards_in_set], n=5, cutoff=0.4
            )
            raise ValueError(
                f"'{card.product_name}' not found in MTGJSON set "
                f"'{card.group_abbreviation}'. Closest: {suggestions}"
            )
        uuids[card.label] = exact[0]["uuid"]

    all_prices_path = _download(
        "https://mtgjson.com/api/v5/AllPrices.json.xz", raw_dir / "mtgjson-AllPrices.json.xz"
    )
    wanted = set(uuids.values())
    price_series: dict[str, dict[str, float]] = {}
    with lzma.open(all_prices_path, "rb") as f:
        for uuid, entry in ijson.kvitems(f, "data"):
            if uuid in wanted:
                normal = ((entry.get("paper") or {}).get("tcgplayer") or {}).get("retail", {}).get(
                    "normal"
                ) or {}
                price_series[uuid] = {d: float(v) for d, v in normal.items() if v is not None}

    frames = {}
    for label, uuid in uuids.items():
        s = price_series.get(uuid, {})
        frames[label] = pd.Series(s, dtype=float)
        frames[label].index = pd.to_datetime(frames[label].index)

    raw = pd.DataFrame(frames).sort_index()
    raw = raw.asfreq("D")
    observed = raw.notna()
    filled = raw.ffill(limit=max_ffill_days)

    return [
        SeriesData(
            name=label,
            values=filled[label].to_numpy(dtype=float),
            dates=filled.index.to_numpy(),
            observed=observed[label].to_numpy(),
        )
        for label in uuids
    ]
