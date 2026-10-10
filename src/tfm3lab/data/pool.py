"""Rule-based selection of the Experiment A card pool (Preregistration v2 §2).

Pure logic only: no network, no archive I/O. scripts/01b_select_pool.py scans
the cached TCGCSV archives and catalog into a per-product statistics table and
hands it to `select_pool`. The rule is frozen; if the data cannot satisfy it
(too few eligible cards, a stratum smaller than the minimum) this module
raises `PoolRuleError` instead of relaxing anything.

Input table (`stats`), one row per product, columns:
  product_id, group_id, group_abbreviation, product_name, rarity (raw TCGplayer
  code or None), present_first_day (bool), observed_days (int), median_price
  (float, USD, median over observed days).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 20261010
POOL_SIZE = 40
MIN_PER_STRATUM = 3
MIN_OBSERVED_SHARE = 0.90
MIN_MEDIAN_PRICE = 1.00

# Amendment 2026-10-10 (eligibility): TCGplayer Rarity codes admitted, and
# non-game items excluded. Frozen; do not tune.
ELIGIBLE_RARITY_CODES = {"M": "Mythic", "R": "Rare", "U": "Uncommon", "C": "Common"}
EXCLUDED_GROUP_ABBREVIATIONS = ("WCD",)  # World Championship Decks
EXCLUDED_NAME_SUBSTRINGS = (
    "oversize",
    "oversized",
    "serial numbered",
    "decklist",
    "biography",
    "checklist",
)

RARITY_GROUPS = ("mythic", "rare", "other")
TERTILES = (1, 2, 3)

_NON_CARD_NAME = re.compile(r"\b(token|art card|helper card)\b", re.IGNORECASE)
_TRAILING_PAREN = re.compile(r"\s*\([^()]*\)\s*$")


class PoolRuleError(RuntimeError):
    """The frozen selection rule cannot be satisfied by the data."""


def rarity_group(rarity: str | None) -> str:
    if rarity == "M":
        return "mythic"
    if rarity == "R":
        return "rare"
    return "other"  # Uncommon + Common (only codes admitted by is_single_card)


def card_name(product_name: str) -> str:
    """Card identity for the one-printing rule: casefolded product name with
    trailing parenthetical printing/variant suffixes removed, e.g.
    'Ragavan, Nimble Pilferer (Borderless)' -> 'ragavan, nimble pilferer'."""
    name = str(product_name).strip()
    while True:
        stripped = _TRAILING_PAREN.sub("", name)
        if stripped == name or not stripped:
            break
        name = stripped
    return name.casefold()


def is_single_card(
    rarity: str | None, product_name: str, group_abbreviation: str | None = None
) -> bool:
    """A game card: Rarity is Mythic, Rare, Uncommon or Common, not a token or
    art card, not in an excluded group (World Championship Decks), and the
    name contains none of EXCLUDED_NAME_SUBSTRINGS (case-insensitive)."""
    if rarity not in ELIGIBLE_RARITY_CODES:
        return False
    if group_abbreviation in EXCLUDED_GROUP_ABBREVIATIONS:
        return False
    name = str(product_name)
    lowered = name.casefold()
    if any(sub in lowered for sub in EXCLUDED_NAME_SUBSTRINGS):
        return False
    return _NON_CARD_NAME.search(name) is None


def eligible_printings(
    stats: pd.DataFrame,
    window_days: int,
    *,
    min_share: float = MIN_OBSERVED_SHARE,
    min_price: float = MIN_MEDIAN_PRICE,
) -> pd.DataFrame:
    """Applies every eligibility filter, then keeps one printing per card
    name (most observed days, ties by lowest productId). Returns rows sorted
    by product_id with added `card_name` and `rarity_group` columns."""
    df = stats.copy()
    single = [
        is_single_card(r, n, g)
        for r, n, g in zip(df["rarity"], df["product_name"], df["group_abbreviation"], strict=True)
    ]
    df = df[np.array(single, dtype=bool)]
    df = df[df["present_first_day"].astype(bool)]
    df = df[df["observed_days"] / window_days >= min_share]
    df = df[df["median_price"] >= min_price]
    df = df.assign(
        card_name=[card_name(n) for n in df["product_name"]],
        rarity_group=[rarity_group(r) for r in df["rarity"]],
    )
    df = df.sort_values(
        ["card_name", "observed_days", "product_id"], ascending=[True, False, True], kind="stable"
    )
    df = df.drop_duplicates("card_name", keep="first")
    return df.sort_values("product_id").reset_index(drop=True)


def assign_tertiles(median_prices: pd.Series) -> pd.Series:
    """Median-price tertile (1 lowest .. 3 highest) over the given set:
    cut points are the 1/3 and 2/3 quantiles (linear interpolation); a price
    equal to a cut point falls in the lower tertile."""
    q1, q2 = np.quantile(median_prices.to_numpy(dtype=float), [1 / 3, 2 / 3])
    out = np.where(median_prices <= q1, 1, np.where(median_prices <= q2, 2, 3))
    return pd.Series(out, index=median_prices.index, dtype=int)


def stratum_label(group: str, tertile: int) -> str:
    return f"{group}-t{tertile}"


def stratum_order() -> list[tuple[str, int]]:
    return [(g, t) for g in RARITY_GROUPS for t in TERTILES]


def allocate(
    sizes: dict[tuple[str, int], int],
    total: int = POOL_SIZE,
    min_per: int = MIN_PER_STRATUM,
) -> dict[tuple[str, int], int]:
    """Proportional allocation of `total` with a floor of `min_per` per
    non-empty stratum and largest-remainder rounding.

    Strata whose proportional quota falls below the floor are fixed at the
    floor; the rest is re-apportioned among the others (repeated until
    stable). A stratum whose quota exceeds its size is fixed at its size.
    Remainder ties break by stratum order. Raises PoolRuleError when the rule
    is infeasible (a non-empty stratum smaller than the floor, or fewer
    candidates than `total`).
    """
    nonempty = {s: n for s, n in sizes.items() if n > 0}
    if sum(nonempty.values()) < total:
        raise PoolRuleError(f"only {sum(nonempty.values())} eligible cards, need {total}")
    small = {s: n for s, n in nonempty.items() if n < min_per}
    if small:
        raise PoolRuleError(f"strata smaller than the minimum of {min_per}: {small}")
    if min_per * len(nonempty) > total:
        raise PoolRuleError("minimum allocation exceeds pool size")

    order = [s for s in stratum_order() if s in nonempty]
    fixed: dict[tuple[str, int], int] = {}
    while True:
        free = [s for s in order if s not in fixed]
        budget = total - sum(fixed.values())
        if not free:
            raise PoolRuleError("allocation infeasible: every stratum is pinned")
        free_total = sum(nonempty[s] for s in free)
        quotas = {s: budget * nonempty[s] / free_total for s in free}
        changed = False
        for s in free:
            if quotas[s] < min_per:
                fixed[s] = min_per
                changed = True
            elif quotas[s] > nonempty[s]:
                fixed[s] = nonempty[s]
                changed = True
        if changed:
            continue
        alloc = {s: math.floor(quotas[s]) for s in free}
        leftover = budget - sum(alloc.values())
        by_remainder = sorted(free, key=lambda s: (-(quotas[s] - alloc[s]), order.index(s)))
        for s in by_remainder[:leftover]:
            alloc[s] += 1
        alloc.update(fixed)
        if any(alloc[s] < min_per or alloc[s] > nonempty[s] for s in alloc):
            # Rounding pushed a stratum out of bounds: fix it and re-apportion.
            for s in free:
                if alloc[s] < min_per:
                    fixed[s] = min_per
                elif alloc[s] > nonempty[s]:
                    fixed[s] = nonempty[s]
            continue
        return {s: alloc.get(s, 0) for s in sizes}


@dataclass(frozen=True)
class PoolSelection:
    eligible: pd.DataFrame  # with price_tertile and stratum
    selected: pd.DataFrame
    eligible_per_stratum: dict[str, int]
    allocation: dict[str, int]


def select_pool(
    stats: pd.DataFrame,
    window_days: int,
    *,
    showcase_names: frozenset[str] = frozenset(),
    seed: int = SEED,
    total: int = POOL_SIZE,
) -> PoolSelection:
    """Full Preregistration v2 §2 selection. Deterministic given
    (stats content, window_days, seed): the result does not depend on input
    row order because candidates are sorted by product_id within each stratum
    and strata are visited in a fixed order."""
    elig = eligible_printings(stats, window_days)
    if len(elig) < total:
        raise PoolRuleError(f"only {len(elig)} eligible cards, need {total}")
    elig["price_tertile"] = assign_tertiles(elig["median_price"])
    elig["stratum"] = [
        stratum_label(g, int(t))
        for g, t in zip(elig["rarity_group"], elig["price_tertile"], strict=True)
    ]
    sizes = {
        (g, t): int(((elig["rarity_group"] == g) & (elig["price_tertile"] == t)).sum())
        for g, t in stratum_order()
    }
    alloc = allocate(sizes, total=total)

    rng = np.random.default_rng(seed)
    picks = []
    for g, t in stratum_order():
        k = alloc[(g, t)]
        if k == 0:
            continue
        cand = elig[(elig["rarity_group"] == g) & (elig["price_tertile"] == t)]
        cand = cand.sort_values("product_id")
        idx = rng.choice(len(cand), size=k, replace=False)
        picks.append(cand.iloc[np.sort(idx)])
    selected = pd.concat(picks).reset_index(drop=True)
    selected["is_showcase"] = [n in showcase_names for n in selected["card_name"]]
    selected["label"] = [
        f"{n} [{a}]"
        for n, a in zip(selected["product_name"], selected["group_abbreviation"], strict=True)
    ]
    return PoolSelection(
        eligible=elig,
        selected=selected,
        eligible_per_stratum={stratum_label(g, t): n for (g, t), n in sizes.items()},
        allocation={stratum_label(g, t): n for (g, t), n in alloc.items()},
    )


POOL_COLUMNS = (
    "label",
    "group_abbreviation",
    "product_name",
    "product_id",
    "rarity_group",
    "price_tertile",
    "stratum",
    "is_showcase",
)
