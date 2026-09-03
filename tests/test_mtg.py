"""Unit tests for tfm3lab.data.mtg using a fake requests.Session — no
network. Live-service verification lives in test_mtg_live.py, opt-in.
"""

from __future__ import annotations

import datetime as dt

import pytest

from tfm3lab.data.mtg import (
    CardSpec,
    PriceSelectionPolicy,
    _price_from_row,
    _resolve_subtype_row,
    resolve_card_specs,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    """Routes GET requests to canned payloads by URL suffix — enough to
    drive resolve_card_specs without touching the network."""

    def __init__(self, groups, products_by_group_id):
        self._groups = groups
        self._products = products_by_group_id

    def get(self, url, **kwargs):
        if url.endswith("/groups"):
            return _FakeResponse({"results": self._groups})
        for group_id, products in self._products.items():
            if url.endswith(f"/{group_id}/products"):
                return _FakeResponse({"results": products})
        raise AssertionError(f"unexpected URL in fake session: {url}")


def _make_session():
    groups = [
        {"groupId": 2809, "abbreviation": "MH2", "name": "Modern Horizons 2", "categoryId": 1},
        {"groupId": 3102, "abbreviation": "DMU", "name": "Dominaria United", "categoryId": 1},
    ]
    products = {
        2809: [
            {"productId": 239857, "name": "Ragavan, Nimble Pilferer"},
            {"productId": 240300, "name": "Ragavan, Nimble Pilferer (Borderless)"},
        ],
        3102: [
            {"productId": 282800, "name": "Sheoldred, the Apocalypse"},
        ],
    }
    return _FakeSession(groups, products)


def test_resolve_card_specs_happy_path():
    cards = (
        CardSpec("Ragavan [MH2]", "MH2", "Ragavan, Nimble Pilferer"),
        CardSpec("Sheoldred [DMU]", "DMU", "Sheoldred, the Apocalypse"),
    )
    df = resolve_card_specs(cards, session=_make_session())
    assert list(df["label"]) == ["Ragavan [MH2]", "Sheoldred [DMU]"]
    assert list(df["product_id"]) == [239857, 282800]
    assert list(df["group_id"]) == [2809, 3102]


def test_resolve_card_specs_exact_name_match_excludes_variants():
    # "Ragavan, Nimble Pilferer" must NOT match the "(Borderless)" variant.
    cards = (CardSpec("Ragavan [MH2]", "MH2", "Ragavan, Nimble Pilferer"),)
    df = resolve_card_specs(cards, session=_make_session())
    assert df.iloc[0]["product_id"] == 239857


def test_resolve_card_specs_raises_with_suggestions_on_unknown_group():
    cards = (CardSpec("Bogus [XXX]", "XXX", "Whatever"),)
    with pytest.raises(ValueError, match="no TCGCSV group"):
        resolve_card_specs(cards, session=_make_session())


def test_resolve_card_specs_raises_with_suggestions_on_unknown_product():
    cards = (CardSpec("Bogus [MH2]", "MH2", "Definitely Not A Real Card Name"),)
    with pytest.raises(ValueError, match="not found in group"):
        resolve_card_specs(cards, session=_make_session())


def test_price_from_row_prefers_market_price():
    assert _price_from_row({"marketPrice": 12.5, "midPrice": 99.0}) == (12.5, "market")


def test_price_from_row_falls_back_to_mid_when_market_is_null():
    assert _price_from_row({"marketPrice": None, "midPrice": 8.25}) == (8.25, "mid")


def test_price_from_row_returns_none_when_both_missing():
    assert _price_from_row({"marketPrice": None, "midPrice": None}) == (None, None)


def test_price_from_row_ignores_low_and_high_price():
    # low/high are listing extremes, never a substitute for market/mid.
    row = {"marketPrice": None, "midPrice": None, "lowPrice": 1.0, "highPrice": 500.0}
    assert _price_from_row(row) == (None, None)


def test_price_from_row_market_only_policy_does_not_fall_back_to_mid():
    row = {"marketPrice": None, "midPrice": 8.25}
    assert _price_from_row(row, PriceSelectionPolicy.MARKET_ONLY) == (None, None)


def test_price_from_row_mid_only_policy_ignores_market():
    row = {"marketPrice": 12.5, "midPrice": 8.25}
    assert _price_from_row(row, PriceSelectionPolicy.MID_ONLY) == (8.25, "mid")


def test_resolve_subtype_row_single_row_returned_directly():
    rows = [{"productId": 1, "subTypeName": "Normal", "marketPrice": 5.0}]
    assert _resolve_subtype_row(1, rows, dt.date(2024, 2, 8)) is rows[0]


def test_resolve_subtype_row_prefers_normal_among_foil_variant():
    rows = [
        {"productId": 1, "subTypeName": "Foil", "marketPrice": 90.0},
        {"productId": 1, "subTypeName": "Normal", "marketPrice": 30.0},
    ]
    resolved = _resolve_subtype_row(1, rows, dt.date(2024, 2, 8))
    assert resolved["subTypeName"] == "Normal"


def test_resolve_subtype_row_treats_missing_subtype_as_normal():
    rows = [
        {"productId": 1, "subTypeName": "Foil", "marketPrice": 90.0},
        {"productId": 1, "marketPrice": 30.0},  # no subTypeName key at all
    ]
    resolved = _resolve_subtype_row(1, rows, dt.date(2024, 2, 8))
    assert resolved["marketPrice"] == 30.0


def test_resolve_subtype_row_raises_on_two_normal_rows():
    rows = [
        {"productId": 1, "subTypeName": "Normal", "marketPrice": 5.0},
        {"productId": 1, "subTypeName": "Normal", "marketPrice": 6.0},
    ]
    with pytest.raises(ValueError, match="ambiguous productId"):
        _resolve_subtype_row(1, rows, dt.date(2024, 2, 8))


def test_resolve_subtype_row_raises_when_no_normal_among_multiple_non_normal():
    rows = [
        {"productId": 1, "subTypeName": "Foil", "marketPrice": 90.0},
        {"productId": 1, "subTypeName": "Foil Etched", "marketPrice": 120.0},
    ]
    with pytest.raises(ValueError, match="ambiguous productId"):
        _resolve_subtype_row(1, rows, dt.date(2024, 2, 8))
