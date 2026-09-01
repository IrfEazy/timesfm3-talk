"""Opt-in integration test against the REAL TCGCSV service — confirms the
archive URL scheme, groups/products catalog shape, and DEFAULT_CARDS
resolution documented in tfm3lab/data/mtg.py still hold.

Skipped by default (network + a live third-party service). Run with:
    TFM3LAB_RUN_LIVE_FETCH_SMOKE=1 uv run pytest tests/test_mtg_live.py -v

This is the same check scripts/00_probe_tcgcsv.py runs before a real
ingest — see that script if this test starts failing.
"""

from __future__ import annotations

import datetime as dt
import os

import pytest

from tfm3lab.data.mtg import DEFAULT_CARDS, build_card_series, resolve_card_specs

pytestmark = pytest.mark.skipif(
    os.environ.get("TFM3LAB_RUN_LIVE_FETCH_SMOKE") != "1",
    reason="opt-in only: hits the live tcgcsv.com service; set TFM3LAB_RUN_LIVE_FETCH_SMOKE=1",
)


def test_all_default_cards_resolve_against_live_catalog():
    df = resolve_card_specs(DEFAULT_CARDS)
    assert len(df) == len(DEFAULT_CARDS)
    assert df["product_id"].is_unique


def test_build_card_series_returns_observed_data_for_a_small_window():
    series = build_card_series(
        cards=DEFAULT_CARDS[:2],
        start=dt.date(2024, 2, 8),
        end=dt.date(2024, 2, 14),
    )
    assert len(series) == 2
    for s in series:
        assert len(s.values) == 7
        assert s.observed.all()  # this window is known-good, no archive gaps
