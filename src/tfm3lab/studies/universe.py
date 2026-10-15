"""Card universe of the v3 studies: single cards, card identity, finish class,
release events and the first-printing flag, all read from a cached TCGCSV
catalogue snapshot.

The catalogue is a snapshot (not point-in-time): `publishedOn` is the value at
fetch time. Groups whose `publishedOn` could not be parsed carry a missing
date (NaT) and are treated as unknown, never guessed.

Definitions (frozen):

- single card: `tfm3lab.data.pool.is_single_card(rarity, name, group_abbr)`;
- card identity: `tfm3lab.data.pool.card_name(product_name)`;
- finish class of a SKU subtype: 'Foil' if it contains 'foil' (any case),
  else 'Normal' if it is 'Normal', else 'Other' (including a missing subtype);
- release event: one calendar date R on which at least one non-supplemental
  Magic group was published, with R in [start, last_release] inclusive;
- first printing: the product's card identity does not appear in any group
  published strictly before the product's own group. Groups published on the
  same date do not count as earlier.

Nothing here reads prices.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from ..data.mtg import MAGIC_CATEGORY_ID
from ..data.pool import card_name, is_single_card

STUDY_START = dt.date(2024, 2, 8)
LAST_RELEASE = dt.date(2026, 6, 5)

FINISH_NORMAL = "Normal"
FINISH_FOIL = "Foil"
FINISH_OTHER = "Other"

PRODUCT_COLUMNS = ("product_id", "group_id", "group_abbreviation", "product_name", "rarity")
GROUP_COLUMNS = (
    "group_id",
    "abbreviation",
    "name",
    "isSupplemental",
    "published_on",
    "category_id",
)
RELEASE_COLUMNS = ("release_date", "group_ids", "abbreviations", "n_groups")
FIRST_PRINTING_COLUMNS = (
    "product_id",
    "group_id",
    "card_name",
    "published_on",
    "first_printing",
    "date_unknown",
)


# --- catalogue loading -----------------------------------------------------------


def load_catalog_products(catalog_dir: Path) -> pd.DataFrame:
    """Every product of a cached catalogue: product_id, group_id,
    group_abbreviation, product_name, rarity (TCGplayer Rarity code from
    extendedData, missing if absent). One row per product_id (first group wins).
    Same logic as `load_catalog` of scripts/01b_select_pool.py."""
    catalog_dir = Path(catalog_dir)
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
    return pd.DataFrame(rows, columns=list(PRODUCT_COLUMNS)).drop_duplicates(
        "product_id", keep="first"
    )


def _parse_published_on(value: object) -> pd.Timestamp:
    """The calendar date (UTC date part) of an ISO `publishedOn` string, at
    midnight; NaT when missing or unparseable."""
    if not isinstance(value, str):
        return pd.NaT
    try:
        return pd.Timestamp(dt.date.fromisoformat(value.strip()[:10]))
    except ValueError:
        return pd.NaT


def load_groups(catalog_dir: Path) -> pd.DataFrame:
    """Every group of a cached catalogue: group_id, abbreviation, name,
    isSupplemental (bool), published_on (datetime64, midnight, NaT if unknown)
    and category_id (TCGplayer categoryId, missing if absent)."""
    results = json.loads((Path(catalog_dir) / "groups.json").read_text())["results"]
    rows = [
        {
            "group_id": int(g["groupId"]),
            "abbreviation": g.get("abbreviation") or "",
            "name": g.get("name"),
            "isSupplemental": bool(g["isSupplemental"]),
            "published_on": _parse_published_on(g.get("publishedOn")),
            "category_id": None if g.get("categoryId") is None else int(g["categoryId"]),
        }
        for g in results
    ]
    out = pd.DataFrame(rows, columns=list(GROUP_COLUMNS))
    out["published_on"] = pd.to_datetime(out["published_on"])
    return out


# --- single cards, identity, finish ----------------------------------------------


def single_cards(products: pd.DataFrame) -> pd.DataFrame:
    """The single-card rows of a products frame (`pool.is_single_card`), with
    an added `card_name` column (`pool.card_name`), original row order."""
    keep = [
        is_single_card(r, n, a)
        for r, n, a in zip(
            products["rarity"],
            products["product_name"],
            products["group_abbreviation"],
            strict=True,
        )
    ]
    out = products.loc[keep].copy()
    out["card_name"] = [card_name(n) for n in out["product_name"]]
    return out


def finish_class(subtype: str | None) -> str:
    """'Foil' if the subtype contains 'foil' (case-insensitive, so 'Foil Etched'
    is Foil), 'Normal' for 'Normal', otherwise 'Other' (including None/NaN)."""
    if not isinstance(subtype, str):
        return FINISH_OTHER
    text = subtype.strip().casefold()
    if "foil" in text:
        return FINISH_FOIL
    return FINISH_NORMAL if text == "normal" else FINISH_OTHER


# --- release events --------------------------------------------------------------


def _as_day(value: dt.date | str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        raise ValueError(f"{value!r} must be timezone-naive")
    return stamp.normalize()


def release_events(
    groups: pd.DataFrame,
    start: dt.date | str | pd.Timestamp = STUDY_START,
    last_release: dt.date | str | pd.Timestamp = LAST_RELEASE,
) -> pd.DataFrame:
    """One row per distinct publication date R in [start, last_release]
    (inclusive) among the non-supplemental Magic groups (category_id ==
    MAGIC_CATEGORY_ID; a frame without a category_id column is taken as
    Magic-only). Groups without a date are skipped.

    Columns: release_date (midnight Timestamp), group_ids (sorted list of
    ints), abbreviations (list, same order), n_groups. Sorted by date."""
    lo, hi = _as_day(start), _as_day(last_release)
    published = groups["published_on"]
    keep = groups["isSupplemental"].eq(False).fillna(False).astype(bool)
    keep &= published.notna() & (published >= lo) & (published <= hi)
    if "category_id" in groups.columns:
        keep &= groups["category_id"].eq(MAGIC_CATEGORY_ID).fillna(False).astype(bool)

    rows = []
    for day, block in (
        groups.loc[keep]
        .sort_values(["published_on", "group_id"])
        .groupby("published_on", sort=True)
    ):
        rows.append(
            {
                "release_date": day,
                "group_ids": [int(g) for g in block["group_id"]],
                "abbreviations": [str(a) for a in block["abbreviation"]],
                "n_groups": len(block),
            }
        )
    return pd.DataFrame(rows, columns=list(RELEASE_COLUMNS))


# --- first printing --------------------------------------------------------------


def first_printing(
    products: pd.DataFrame,
    groups: pd.DataFrame,
    as_of: dt.date | str | pd.Timestamp = "2026-09-04",
) -> pd.DataFrame:
    """First-printing flag of every single-card product.

    A product is a first printing iff its card identity (`pool.card_name`)
    does not appear among the single cards of any group with `published_on`
    strictly earlier than its own group's. Pass the whole catalogue as
    `products`: earlier printings are looked up only within it. Groups on the
    same date do not count as earlier. A group dated after `as_of` (default
    2026-09-04; such dates are catalogue fetch stamps, not releases) is treated
    as undated. A product whose own group has no date
    (NaT, after `as_of`, or absent from `groups`) is unknown: first_printing False and
    date_unknown True. Undated groups never count as earlier for anyone else.

    Returns product_id, group_id, card_name, published_on, first_printing,
    date_unknown for the single cards, sorted by product_id; the number of
    unknown-date products is `result.attrs['n_unknown_date']`."""
    if products["product_id"].duplicated().any():
        raise ValueError("products has duplicated product_id values")
    if groups["group_id"].duplicated().any():
        raise ValueError("groups has duplicated group_id values")

    cards = single_cards(products)
    cards["published_on"] = pd.to_datetime(
        cards["group_id"].map(groups.set_index("group_id")["published_on"])
    )
    cutoff = _as_day(as_of)
    cards["published_on"] = cards["published_on"].where(cards["published_on"] <= cutoff)
    earliest = cards.groupby("card_name")["published_on"].min()
    cards["date_unknown"] = cards["published_on"].isna()
    cards["first_printing"] = ~cards["date_unknown"] & ~(
        cards["card_name"].map(earliest) < cards["published_on"]
    )
    out = (
        cards[list(FIRST_PRINTING_COLUMNS)]
        .sort_values("product_id", kind="stable")
        .reset_index(drop=True)
    )
    out.attrs["n_unknown_date"] = int(out["date_unknown"].sum())
    return out
