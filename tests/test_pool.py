"""Offline tests for tfm3lab.data.pool (synthetic inputs, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm3lab.data import pool

WINDOW = 100


def _row(pid, name, *, rarity="R", obs=100, price=5.0, first=True, group="AAA", gid=1):
    return {
        "product_id": pid,
        "group_id": gid,
        "group_abbreviation": group,
        "product_name": name,
        "rarity": rarity,
        "present_first_day": first,
        "observed_days": obs,
        "median_price": price,
    }


def _stats(rows):
    return pd.DataFrame(rows)


def test_rarity_group():
    assert pool.rarity_group("M") == "mythic"
    assert pool.rarity_group("R") == "rare"
    for r in ("U", "C"):
        assert pool.rarity_group(r) == "other"


def test_card_name_strips_trailing_parentheticals():
    assert pool.card_name("Ragavan, Nimble Pilferer (Borderless)") == "ragavan, nimble pilferer"
    assert pool.card_name("Plains (277) (Foil Etched)") == "plains"
    assert pool.card_name("Urza's Saga") == "urza's saga"


@pytest.mark.parametrize(
    "rarity,name,ok",
    [
        ("R", "Card", True),
        (None, "Kaldheim - Draft Booster Box", False),
        ("T", "Goblin Token", False),
        ("S", "Some Art Card (Gold-Stamped)", False),
        ("C", "Elf Token (Surge Foil)", False),
        ("S", "Helper Card - 1/9", False),
        ("U", "Tokyo Hero", True),
    ],
)
def test_is_single_card(rarity, name, ok):
    assert pool.is_single_card(rarity, name) is ok


def test_eligibility_filters():
    rows = [
        _row(1, "Keep"),
        _row(2, "Sealed", rarity=None),
        _row(3, "Goblin Token", rarity="T"),
        _row(4, "Late Arrival", first=False),
        _row(5, "Sparse", obs=89),
        _row(6, "Edge Share", obs=90),
        _row(7, "Cheap", price=0.99),
        _row(8, "Edge Price", price=1.00),
    ]
    out = pool.eligible_printings(_stats(rows), WINDOW)
    assert sorted(out["product_id"]) == [1, 6, 8]


def test_dedup_most_observed_days_then_lowest_product_id():
    rows = [
        _row(10, "Dup (Borderless)", obs=95),
        _row(11, "Dup", obs=99),
        _row(12, "Dup (Extended Art)", obs=99),
        _row(20, "Tie A", obs=95),
        _row(21, "Tie A (Showcase)", obs=95),
    ]
    out = pool.eligible_printings(_stats(rows), WINDOW)
    assert sorted(out["product_id"]) == [11, 20]


def test_tertiles_equal_thirds_and_ties():
    prices = pd.Series(np.arange(1.0, 10.0))
    t = pool.assign_tertiles(prices)
    assert list(t) == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    same = pool.assign_tertiles(pd.Series([2.0] * 6))
    assert set(same) == {1}


def _sizes(**kw):
    out = {}
    for g, t in pool.stratum_order():
        out[(g, t)] = kw.get(f"{g}{t}", 0)
    return out


def test_allocate_proportional_sums_to_total_largest_remainder():
    sizes = _sizes(
        mythic1=10,
        mythic2=10,
        mythic3=10,
        rare1=10,
        rare2=10,
        rare3=10,
        other1=10,
        other2=10,
        other3=11,
    )
    alloc = pool.allocate(sizes, total=40)
    assert sum(alloc.values()) == 40
    # 101 cards -> quotas ~4.0-4.4; the 4 largest remainders... all 4 or 5
    assert all(4 <= v <= 5 for v in alloc.values())
    # the larger stratum has the largest quota
    assert alloc[("other", 3)] >= 4


def test_allocate_min_three_and_remainder():
    sizes = _sizes(mythic1=2 + 0, rare1=300, other1=300)
    sizes[("mythic", 1)] = 3  # tiny but non-empty -> floor of 3
    alloc = pool.allocate(sizes, total=40)
    assert alloc[("mythic", 1)] == 3
    assert alloc[("rare", 1)] + alloc[("other", 1)] == 37
    assert sum(alloc.values()) == 40
    assert alloc[("mythic", 2)] == 0  # empty stays empty


def test_allocate_remainder_tie_breaks_by_stratum_order():
    sizes = _sizes(mythic1=50, rare1=50, other1=50)  # quota 13.33 each, 1 left over
    alloc = pool.allocate(sizes, total=40)
    assert [alloc[("mythic", 1)], alloc[("rare", 1)], alloc[("other", 1)]] == [14, 13, 13]


def test_allocate_infeasible_raises():
    with pytest.raises(pool.PoolRuleError):
        pool.allocate(_sizes(mythic1=2, rare1=100), total=40)
    with pytest.raises(pool.PoolRuleError):
        pool.allocate(_sizes(mythic1=10, rare1=20), total=40)


def _synthetic(n_per=12):
    rows = []
    pid = 100
    for rarity in ("M", "R", "U"):
        for k in range(3 * n_per):
            pid += 1
            rows.append(_row(pid, f"Card {pid}", rarity=rarity, price=1.0 + k + (pid % 7) * 0.01))
    return _stats(rows)


def test_select_pool_deterministic_and_row_order_independent():
    df = _synthetic()
    a = pool.select_pool(df, WINDOW)
    b = pool.select_pool(df.sample(frac=1.0, random_state=3).reset_index(drop=True), WINDOW)
    assert list(a.selected["product_id"]) == list(b.selected["product_id"])
    assert len(a.selected) == 40
    assert sum(a.allocation.values()) == 40
    assert a.selected["product_id"].is_unique
    c = pool.select_pool(df, WINDOW, seed=1)
    assert list(c.selected["product_id"]) != list(a.selected["product_id"])
    # every selected card sits in the stratum it claims
    for _, r in a.selected.iterrows():
        assert r["stratum"] == f"{r['rarity_group']}-t{r['price_tertile']}"


def test_select_pool_flags_showcase_and_does_not_force():
    df = _synthetic()
    df.loc[0, "product_name"] = "Urza's Saga (Borderless)"
    sel = pool.select_pool(df, WINDOW, showcase_names=frozenset({"urza's saga"}))
    flagged = sel.selected[sel.selected["is_showcase"]]
    assert set(flagged["card_name"]) <= {"urza's saga"}
    forced_out = df.drop(index=0)
    assert len(pool.select_pool(forced_out, WINDOW).selected) == 40


def test_select_pool_too_few_eligible_raises():
    with pytest.raises(pool.PoolRuleError):
        pool.select_pool(_synthetic(n_per=3).head(39), WINDOW)


def test_amendment_constants():
    assert pool.ELIGIBLE_RARITY_CODES == {
        "M": "Mythic",
        "R": "Rare",
        "U": "Uncommon",
        "C": "Common",
    }
    assert pool.EXCLUDED_GROUP_ABBREVIATIONS == ("WCD",)
    assert pool.EXCLUDED_NAME_SUBSTRINGS == (
        "oversize",
        "oversized",
        "serial numbered",
        "decklist",
        "biography",
        "checklist",
    )


@pytest.mark.parametrize("rarity", ["M", "R", "U", "C"])
def test_admitted_rarities(rarity):
    assert pool.is_single_card(rarity, "Some Card", "AAA") is True


@pytest.mark.parametrize("rarity", ["L", "S", "P", "T", None])
def test_other_rarities_rejected(rarity):
    assert pool.is_single_card(rarity, "Some Card", "AAA") is False


def test_wcd_group_excluded():
    assert pool.is_single_card("R", "Feldon's Cane", "WCD") is False


@pytest.mark.parametrize(
    "name",
    [
        "Sidar Kondo (OVERSIZE)",
        "Foo (Oversized)",
        "Aegar (Serial Numbered)",
        "2004 Aeo Paquette DeckList Card",
        "1999 Kai Budde BIOGRAPHY Card",
        "Checklist Card",
    ],
)
def test_excluded_names(name):
    assert pool.is_single_card("R", name, "AAA") is False


def test_eligibility_applies_amendment():
    rows = [
        _row(1, "Good Card"),
        _row(2, "Deck Thing", group="WCD"),
        _row(3, "Big Oversize Card"),
        _row(4, "Land Thing", rarity="L"),
        _row(5, "Uncommon Card", rarity="U"),
    ]
    out = pool.eligible_printings(_stats(rows), WINDOW)
    assert list(out["product_id"]) == [1, 5]
