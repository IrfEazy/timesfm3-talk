"""Card universe of the v3 studies: catalogue loading, finish class, release
events and the first-printing flag, on small synthetic frames. One smoke test
reads the real catalogue snapshot when it is present (skipped otherwise); it
never touches prices."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfm3lab.data import pool
from tfm3lab.studies import universe

REPO = Path(__file__).resolve().parents[1]
REAL_CATALOG = REPO / "data" / "raw" / "catalog-2026-09-04"


def _groups(rows: list[tuple]) -> pd.DataFrame:
    """rows: (group_id, abbreviation, isSupplemental, published_on, category_id)."""
    return pd.DataFrame(
        [
            {
                "group_id": gid,
                "abbreviation": abbr,
                "name": f"Group {abbr}",
                "isSupplemental": supp,
                "published_on": pd.Timestamp(day) if day else pd.NaT,
                "category_id": cat,
            }
            for gid, abbr, supp, day, cat in rows
        ]
    )


def _products(rows: list[tuple]) -> pd.DataFrame:
    """rows: (product_id, group_id, group_abbreviation, product_name, rarity)."""
    return pd.DataFrame(rows, columns=list(universe.PRODUCT_COLUMNS))


# --- catalogue loading -----------------------------------------------------------


def _write_catalog(directory: Path) -> Path:
    groups = [
        {
            "groupId": 10,
            "name": "Old Set",
            "abbreviation": "OLD",
            "isSupplemental": False,
            "publishedOn": "2024-02-08T00:00:00",
            "categoryId": 1,
        },
        {
            "groupId": 20,
            "name": "Promos",
            "abbreviation": None,
            "isSupplemental": True,
            "publishedOn": "2026-10-09T20:00:05.5619202Z",
            "categoryId": 1,
        },
        {"groupId": 30, "name": "Undated", "abbreviation": "UND", "isSupplemental": False},
    ]
    (directory / "groups.json").write_text(json.dumps({"results": groups}))

    def product(pid: int, name: str, rarity: str | None) -> dict:
        ext = [] if rarity is None else [{"name": "Rarity", "value": rarity}]
        return {"productId": pid, "name": name, "extendedData": ext}

    products = {
        10: [product(1, "Bolt", "R"), product(2, "Sealed Box", None), product(3, "Elf", "C")],
        20: [product(3, "Elf (Promo)", "C"), product(4, "Bolt (Promo)", "R")],
        30: [],
    }
    for gid, items in products.items():
        (directory / f"products-{gid}.json").write_text(json.dumps({"results": items}))
    return directory


def test_load_catalog_products_columns_dedup_and_rarity(tmp_path):
    out = universe.load_catalog_products(_write_catalog(tmp_path))

    assert list(out.columns) == list(universe.PRODUCT_COLUMNS)
    assert out["product_id"].tolist() == [1, 2, 3, 4]  # product 3 kept once
    by_id = out.set_index("product_id")
    assert by_id.loc[3, "group_id"] == 10  # first group wins
    assert by_id.loc[3, "product_name"] == "Elf"
    assert by_id.loc[1, "rarity"] == "R" and pd.isna(by_id.loc[2, "rarity"])
    assert by_id.loc[4, "group_abbreviation"] == ""  # null abbreviation -> ""


def test_load_catalog_products_matches_the_01b_script_logic(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "select_pool_01b_for_test", REPO / "scripts" / "01b_select_pool.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    catalog = _write_catalog(tmp_path)
    pd.testing.assert_frame_equal(
        universe.load_catalog_products(catalog), script.load_catalog(catalog)
    )


def test_load_groups_parses_dates_and_keeps_unknown_ones_missing(tmp_path):
    out = universe.load_groups(_write_catalog(tmp_path))

    assert list(out.columns) == list(universe.GROUP_COLUMNS)
    assert out["group_id"].tolist() == [10, 20, 30]
    assert out["published_on"].iloc[0] == pd.Timestamp("2024-02-08")
    assert out["published_on"].iloc[1] == pd.Timestamp("2026-10-09")  # UTC date part
    assert pd.isna(out["published_on"].iloc[2])
    assert out["isSupplemental"].tolist() == [False, True, False]
    assert out["abbreviation"].tolist() == ["OLD", "", "UND"]
    assert out["category_id"].iloc[0] == 1 and pd.isna(out["category_id"].iloc[2])


# --- single cards and identity ----------------------------------------------------


def test_single_cards_uses_the_pool_filter_and_card_identity():
    products = _products(
        [
            (1, 1, "AAA", "Ragavan, Nimble Pilferer (Borderless)", "M"),
            (2, 1, "AAA", "Goblin Token", "C"),
            (3, 1, "AAA", "Booster Box", None),
            (4, 1, "WCD", "Lightning Bolt", "R"),
            (5, 1, "AAA", "Oversized Thing", "R"),
            (6, 1, "AAA", "Plains", "L"),
        ]
    )
    out = universe.single_cards(products)

    assert out["product_id"].tolist() == [1]
    assert out["card_name"].tolist() == [pool.card_name("Ragavan, Nimble Pilferer (Borderless)")]
    assert out["card_name"].iloc[0] == "ragavan, nimble pilferer"
    assert "card_name" not in products.columns  # input untouched


# --- finish class ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("subtype", "expected"),
    [
        ("Normal", "Normal"),
        ("normal", "Normal"),
        ("Foil", "Foil"),
        ("FOIL", "Foil"),
        ("Foil Etched", "Foil"),
        ("Surge Something", "Other"),
        ("", "Other"),
        (None, "Other"),
        (float("nan"), "Other"),
        (pd.NA, "Other"),
    ],
)
def test_finish_class(subtype, expected):
    assert universe.finish_class(subtype) == expected


# --- release events --------------------------------------------------------------


def test_release_events_exclude_supplemental_and_pool_same_date_groups():
    groups = _groups(
        [
            (1, "A", False, "2024-02-08", 1),  # start bound, inclusive
            (2, "B", True, "2024-02-09", 1),  # supplemental: excluded
            (4, "D", False, "2024-03-08", 1),
            (3, "C", False, "2024-03-08", 1),  # same date as D: pooled, sorted by id
            (5, "E", False, "2024-02-07", 1),  # day before start: excluded
            (6, "F", False, "2026-06-05", 1),  # last release, inclusive
            (7, "G", False, "2026-06-06", 1),  # day after: excluded
            (8, "H", False, "2024-05-01", 2),  # not Magic: excluded
            (9, "I", False, None, 1),  # undated: excluded
        ]
    )
    out = universe.release_events(groups)

    assert list(out.columns) == list(universe.RELEASE_COLUMNS)
    assert out["release_date"].tolist() == [
        pd.Timestamp("2024-02-08"),
        pd.Timestamp("2024-03-08"),
        pd.Timestamp("2026-06-05"),
    ]
    assert out["group_ids"].tolist() == [[1], [3, 4], [6]]
    assert out["abbreviations"].tolist() == [["A"], ["C", "D"], ["F"]]
    assert out["n_groups"].tolist() == [1, 2, 1]


def test_release_events_window_arguments_and_missing_category_column():
    groups = _groups(
        [
            (1, "A", False, "2024-02-08", 1),
            (2, "B", False, "2024-03-08", 1),
            (3, "C", False, "2024-04-19", 1),
        ]
    ).drop(columns=["category_id"])

    out = universe.release_events(groups, start="2024-03-08", last_release=dt.date(2024, 4, 19))
    assert out["group_ids"].tolist() == [[2], [3]]  # both bounds inclusive
    assert universe.release_events(groups, start="2025-01-01")["n_groups"].tolist() == []
    assert list(universe.release_events(groups, start="2025-01-01").columns) == list(
        universe.RELEASE_COLUMNS
    )


def test_release_events_default_window_is_the_frozen_study_window():
    window = (universe.STUDY_START, universe.LAST_RELEASE)
    assert window == (dt.date(2024, 2, 8), dt.date(2026, 6, 5))


# --- first printing --------------------------------------------------------------


def _first_printing_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _groups(
        [
            (1, "G1", False, "2024-01-01", 1),
            (2, "G2", False, "2024-06-01", 1),
            (3, "G3", True, "2024-06-01", 1),  # same date as G2
            (4, "G4", False, None, 1),  # undated
            (5, "G5", False, "2025-01-01", 1),
            (6, "WCD", False, "2020-01-01", 1),  # excluded group: not a single-card source
            (7, "G7", False, "2023-01-01", 1),
        ]
    )
    products = _products(
        [
            # Bolt: G1 first, later reprints are not first printings.
            (10, 1, "G1", "Lightning Bolt", "R"),
            (11, 2, "G2", "Lightning Bolt (Borderless)", "R"),
            (12, 5, "G5", "Lightning Bolt", "R"),
            # Counterspell: same-date groups -> both still first printing.
            (20, 2, "G2", "Counterspell", "U"),
            (21, 3, "G3", "Counterspell", "U"),
            # Unique card in one group.
            (30, 5, "G5", "Unique Card", "M"),
            # Undated own group -> unknown/False; the undated copy is never "earlier".
            (40, 4, "G4", "Orphan", "C"),
            (41, 5, "G5", "Orphan", "C"),
            # Product whose group is not in `groups` at all -> unknown.
            (50, 99, "ZZZ", "Ghost", "R"),
            # Non-singles never count: a token / sealed product / WCD copy of Bolt in
            # an earlier group does not make the G1 Bolt a reprint.
            (60, 7, "G7", "Lightning Bolt Token", "C"),
            (61, 7, "G7", "Lightning Bolt", None),
            (62, 6, "WCD", "Lightning Bolt", "R"),
        ]
    )
    return products, groups


def test_first_printing_logic():
    products, groups = _first_printing_fixture()
    out = universe.first_printing(products, groups)

    assert list(out.columns) == list(universe.FIRST_PRINTING_COLUMNS)
    assert out["product_id"].tolist() == [10, 11, 12, 20, 21, 30, 40, 41, 50]  # singles only
    flags = out.set_index("product_id")["first_printing"].to_dict()
    assert flags == {
        10: True,
        11: False,
        12: False,
        20: True,
        21: True,  # same date is not strictly earlier
        30: True,
        40: False,  # unknown date
        41: True,  # undated G4 does not count as earlier
        50: False,  # group absent from the catalogue groups
    }
    assert out.set_index("product_id")["date_unknown"].to_dict() == {
        **dict.fromkeys(flags, False),
        40: True,
        50: True,
    }
    assert out.attrs["n_unknown_date"] == 2
    assert out["first_printing"].dtype == bool
    assert out.set_index("product_id").loc[10, "card_name"] == "lightning bolt"


def test_first_printing_ignores_row_order_and_does_not_mutate_inputs():
    products, groups = _first_printing_fixture()
    before_p, before_g = products.copy(), groups.copy()
    forward = universe.first_printing(products, groups)
    shuffled = universe.first_printing(products.sample(frac=1, random_state=3), groups)

    pd.testing.assert_frame_equal(forward, shuffled)
    pd.testing.assert_frame_equal(products, before_p)
    pd.testing.assert_frame_equal(groups, before_g)


def test_first_printing_rejects_duplicate_identifiers():
    products, groups = _first_printing_fixture()
    with pytest.raises(ValueError, match="product_id"):
        universe.first_printing(pd.concat([products, products.iloc[:1]]), groups)
    with pytest.raises(ValueError, match="group_id"):
        universe.first_printing(products, pd.concat([groups, groups.iloc[:1]]))


# --- real catalogue (optional) ---------------------------------------------------


@pytest.mark.skipif(not REAL_CATALOG.is_dir(), reason="catalogue snapshot not present")
def test_real_catalogue_smoke():
    groups = universe.load_groups(REAL_CATALOG)
    products = universe.load_catalog_products(REAL_CATALOG)

    assert len(groups) > 0 and len(products) > 0
    assert set(groups["category_id"].unique()) == {1}  # the file is Magic-only
    assert set(products["group_id"]) <= set(groups["group_id"])

    events = universe.release_events(groups)
    assert len(events) > 0
    dates = events["release_date"]
    assert dates.is_monotonic_increasing and dates.is_unique
    assert dates.min() >= pd.Timestamp(universe.STUDY_START)
    assert dates.max() <= pd.Timestamp(universe.LAST_RELEASE)
    listed = {gid for ids in events["group_ids"] for gid in ids}
    supplemental = set(groups.loc[groups["isSupplemental"], "group_id"])
    assert not listed & supplemental
    assert events["n_groups"].sum() == len(listed)

    first = universe.first_printing(products, groups)
    singles = universe.single_cards(products)
    assert len(first) == len(singles) > 0
    assert set(first["product_id"]) == set(singles["product_id"])
    assert first["first_printing"].dtype == bool
    assert not (first["first_printing"] & first["date_unknown"]).any()
    assert first.attrs["n_unknown_date"] == int(first["date_unknown"].sum())
    assert 0 < first["first_printing"].mean() < 1
    assert np.isfinite(first["first_printing"].mean())


def test_first_printing_groups_after_as_of_are_unknown():
    groups = _groups(
        [
            (1, "G1", False, "2024-01-01", 1),
            (2, "PROMO", False, "2026-10-09T20:00:05", 1),  # catalogue fetch stamp
        ]
    )
    products = _products(
        [
            (10, 1, "G1", "Lightning Bolt", "R"),
            (11, 2, "PROMO", "Lightning Bolt", "R"),
            (20, 2, "PROMO", "Only Promo", "R"),
        ]
    )
    out = universe.first_printing(products, groups).set_index("product_id")
    assert out["first_printing"].to_dict() == {10: True, 11: False, 20: False}
    assert out["date_unknown"].to_dict() == {10: False, 11: True, 20: True}
    late = universe.first_printing(products, groups, as_of="2026-12-31").set_index("product_id")
    assert late.loc[11, "first_printing"] is np.False_ or not late.loc[11, "first_printing"]
    assert bool(late.loc[20, "first_printing"])
