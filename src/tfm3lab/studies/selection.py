"""Leakage-safe, preregistered selection rules of the three v3 studies.

Everything works on one `QuoteData`: the wide price matrix (`wide`), the quote
matrix `q` (`quotes.quote_matrix`, frozen policy) and the observed mask, one
row per single-card SKU (product_id, subtype), one column per calendar day.
"Observed on d" means a valid price on that exact day; every price used by a
rule is the quote q(d) (never a raw outlier, never an interpolation).

Leakage rule: every rule below is a function of one origin (or date) t and
first slices `[:, : t + 1]` of the matrices; nothing after t is read, and
nothing in this module computes an outcome. The only forward-looking quantity
is the S2 candidate pool: its draw walks the per-date eligibility sets, which
are themselves origin-only. The S2 accept/reject pass is backward-looking only.

Rules (frozen; see the work package for the full text):

- S1: release event R (window 2024-02-08..2026-06-05), k in (7, 14, 28),
  origin O = R + k, candidates = SKUs of the event's groups; eligible iff
  q(O) >= 0.25 and the trailing context at O (the run of non-missing quotes
  ending at O, <= 512) holds >= 5 observed days; floor_1 = q(O) >= 1.00.
  Horizon 90 - k.
- S2: dates t in [2024-08-01, 2026-08-06]; (SKU, t) eligible iff observed on
  t, q(t) >= 1.00, first observed <= t - 90, >= 60 observed days in
  [t - 90, t], trailing context length >= 60. M = 5500 successful draws
  (date uniform among window dates, then SKU uniform among the date's
  eligible SKUs in canonical order; an empty date is redrawn). Candidates are
  then sorted by (origin, draw index) and accepted iff no earlier ACCEPTED
  decision of the same card has origin in (t - 28, t]. Horizon 28.
- S3: ratio(d) = q(d) / q(d - 7), defined only when observed on d and q(d - 7)
  exists. Upward crossing at t iff ratio(t) >= 1.5, ratio(t - 1) defined and
  < 1.5, q(t - 7) >= 1.00, first observed <= t - 90, trailing context length
  >= 60, t in [2024-05-08, 2026-07-09]. Same-day crossings of one card keep
  the largest q(t - 7) (ties: lowest canonical SKU); chronologically accepted
  iff the card has no accepted event with date in (t - 56, t). Horizon 56.
"""

from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..data.panel import load_panel
from . import quotes, requests

# --- frozen constants ------------------------------------------------------------

S1_KS: tuple[int, ...] = (7, 14, 28)
S1_FLOOR = 0.25
S1_FLOOR_1 = 1.00
S1_MIN_OBSERVED = 5
S1_HORIZON_BASE = 90

S2_START = dt.date(2024, 8, 1)
S2_END = dt.date(2026, 8, 6)
S2_MIN_PRICE = 1.00
S2_MIN_AGE = 90
S2_MIN_OBSERVED = 60
S2_MIN_CONTEXT = 60
S2_POOL_SIZE = 5500  # M
S2_SEED = 20261014
S2_SPACING = 28
S2_HORIZON = 28
S2_BANDS: tuple[tuple[float, str], ...] = ((5.0, "1-5"), (20.0, "5-20"))
S2_TOP_BAND = "20+"

S3_START = dt.date(2024, 5, 8)
S3_END = dt.date(2026, 7, 9)
S3_LAG = 7
S3_RATIO = 1.5
S3_MIN_PRIOR = 1.00
S3_MIN_AGE = 90
S3_MIN_CONTEXT = 60
S3_COOLDOWN = 56
S3_HORIZON = 56
S3_MIN_EVENTS = 100
S3_MIN_MONTHS = 12

BLOCK_S1 = {14: 1, 7: 4, 28: 4}
BLOCK_S2 = 2
BLOCK_S3 = 3

# The last origin any rule reads; the panel is never loaded beyond it.
LAST_ORIGIN = max(S2_END, S3_END)


# --- the quote matrix bundle -----------------------------------------------------


def canonical_key(sku: tuple[int, str | None]) -> tuple[int, bool, str]:
    """Canonical SKU order: (product_id, subtype) with the missing subtype first."""
    return int(sku[0]), sku[1] is not None, sku[1] or ""


@dataclass(frozen=True)
class QuoteData:
    """dates (daily, contiguous), skus (row labels), wide prices, quotes, observed
    mask and the canonical rank of each row (rank 0 sorts first)."""

    dates: pd.DatetimeIndex
    skus: list[tuple[int, str | None]]
    wide: np.ndarray
    qmat: np.ndarray
    obs: np.ndarray
    rank: np.ndarray

    @classmethod
    def from_wide(cls, dates, skus, wide: np.ndarray) -> QuoteData:
        skus = [quotes._sku(p, s) for p, s in skus]
        wide = np.asarray(wide, dtype=np.float64)
        order = sorted(range(len(skus)), key=lambda i: canonical_key(skus[i]))
        rank = np.empty(len(skus), dtype=np.int64)
        rank[order] = np.arange(len(skus))
        return cls(
            dates=pd.DatetimeIndex(dates),
            skus=skus,
            wide=wide,
            qmat=quotes.quote_matrix(wide),
            obs=quotes.observed_mask(wide),
            rank=rank,
        )

    @classmethod
    def from_long(cls, long_df: pd.DataFrame) -> QuoteData:
        dates, skus, wide = quotes.wide_prices(long_df)
        return cls.from_wide(dates, list(skus), wide)

    def day_index(self, day) -> int:
        stamp = pd.Timestamp(day).normalize()
        if stamp < self.dates[0] or stamp > self.dates[-1]:
            raise ValueError(
                f"{stamp.date()} is outside the panel [{self.dates[0].date()}, "
                f"{self.dates[-1].date()}]"
            )
        return int((stamp - self.dates[0]) // pd.Timedelta(days=1))


# --- origin-time primitives (each slices to the origin first) ---------------------


def _valid(values: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.isfinite(values) & (values > 0)


def _ge(values: np.ndarray, floor: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.isfinite(values) & (values >= floor)


def trailing_run(valid: np.ndarray) -> np.ndarray:
    """Per row, the number of trailing True values of a 2-D boolean array."""
    width = valid.shape[1]
    if width == 0:
        return np.zeros(valid.shape[0], dtype=np.int64)
    gap = ~valid[:, ::-1]
    return np.where(gap.any(axis=1), gap.argmax(axis=1), width).astype(np.int64)


def context_length(data: QuoteData, rows: np.ndarray, t: int) -> np.ndarray:
    """Length of `quotes.context` (<= 512) of `rows` at origin column t."""
    q = data.qmat[:, : t + 1]
    lo = max(0, t + 1 - quotes.MAX_CONTEXT)
    return trailing_run(_valid(q[rows, lo:]))


def _ever_observed_by(data: QuoteData, rows: np.ndarray, last_col: int) -> np.ndarray:
    """True where the row has a valid price in a column <= last_col."""
    if last_col < 0:
        return np.zeros(len(rows), dtype=bool)
    return data.obs[:, : last_col + 1][rows].any(axis=1)


def _by_rank(data: QuoteData, rows: np.ndarray) -> np.ndarray:
    return rows[np.argsort(data.rank[rows], kind="stable")]


# --- S1 --------------------------------------------------------------------------


def s1_eligible(data: QuoteData, rows: np.ndarray, origin: int) -> tuple[np.ndarray, np.ndarray]:
    """For candidate `rows` at origin column `origin`: (eligible, floor_1)
    boolean arrays aligned with `rows`. Reads columns <= origin only."""
    rows = np.asarray(rows, dtype=np.int64)
    q = data.qmat[:, : origin + 1]
    obs = data.obs[:, : origin + 1]
    price = q[rows, origin] if len(rows) else np.empty(0)
    ok = _ge(price, S1_FLOOR)
    eligible = np.zeros(len(rows), dtype=bool)
    if ok.any():
        sel = rows[ok]
        lo = max(0, origin + 1 - quotes.MAX_CONTEXT)
        run = trailing_run(_valid(q[sel, lo:]))
        width = origin + 1 - lo
        in_run = np.arange(width)[None, :] >= (width - run)[:, None]
        n_obs = (obs[sel, lo:] & in_run).sum(axis=1)
        eligible[np.flatnonzero(ok)] = n_obs >= S1_MIN_OBSERVED
    floor_1 = eligible & _ge(price, S1_FLOOR_1)
    return eligible, floor_1


def s1_horizon(k: int) -> int:
    return S1_HORIZON_BASE - k


def select_s1(
    data: QuoteData, events: pd.DataFrame, group_of_row: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S1 units and per-(event, k) counts.

    `events`: release_events frame (release_date, group_ids). `group_of_row`:
    group_id of each row of `data`. Returns (units, counts); units has event_date,
    k, origin_idx, row, floor_1 (canonical SKU order within an event and k)."""
    unit_parts, counts = [], []
    group_of_row = np.asarray(group_of_row)
    for ev in events.itertuples(index=False):
        rows = _by_rank(data, np.flatnonzero(np.isin(group_of_row, list(ev.group_ids))))
        day = pd.Timestamp(ev.release_date).normalize()
        for k in S1_KS:
            origin = data.day_index(day + pd.Timedelta(days=k))
            eligible, floor_1 = s1_eligible(data, rows, origin)
            counts.append(
                {
                    "event_date": day,
                    "k": k,
                    "origin_idx": origin,
                    "n_candidates": len(rows),
                    "eligible_floor_025": int(eligible.sum()),
                    "eligible_floor_100": int(floor_1.sum()),
                }
            )
            unit_parts.append(
                pd.DataFrame(
                    {
                        "event_date": day,
                        "k": k,
                        "origin_idx": origin,
                        "row": rows[eligible],
                        "floor_1": floor_1[eligible],
                    }
                )
            )
    cols = ["event_date", "k", "origin_idx", "row", "floor_1"]
    units = pd.concat(unit_parts, ignore_index=True) if unit_parts else pd.DataFrame(columns=cols)
    count_cols = [
        "event_date",
        "k",
        "origin_idx",
        "n_candidates",
        "eligible_floor_025",
        "eligible_floor_100",
    ]
    return units, pd.DataFrame(counts, columns=count_cols)


# --- S2 --------------------------------------------------------------------------


def s2_eligible(data: QuoteData, t: int) -> np.ndarray:
    """Rows eligible at date column t, in canonical order. Reads columns <= t."""
    q = data.qmat[:, : t + 1]
    obs = data.obs[:, : t + 1]
    if t < S2_MIN_AGE:
        return np.empty(0, dtype=np.int64)
    rows = np.flatnonzero(obs[:, t] & _ge(q[:, t], S2_MIN_PRICE))
    rows = rows[_ever_observed_by(data, rows, t - S2_MIN_AGE)]
    rows = rows[obs[rows, t - S2_MIN_AGE : t + 1].sum(axis=1) >= S2_MIN_OBSERVED]
    # Trailing context length >= 60 <=> the last 60 quotes are all valid.
    rows = rows[_valid(q[rows, t - S2_MIN_CONTEXT + 1 :]).all(axis=1)]
    return _by_rank(data, rows)


def s2_window(data: QuoteData) -> list[int]:
    lo, hi = data.day_index(S2_START), data.day_index(S2_END)
    return list(range(lo, hi + 1))


def s2_eligibility_sets(data: QuoteData, window: list[int]) -> list[np.ndarray]:
    return [s2_eligible(data, t) for t in window]


def draw_s2_pool(
    eligible: list[np.ndarray],
    window: list[int],
    m: int = S2_POOL_SIZE,
    seed: int = S2_SEED,
) -> pd.DataFrame:
    """M successful draws: date uniform among the window dates, then a SKU
    uniform among the date's eligible SKUs (canonical order). A date without
    eligible SKUs is redrawn. Columns draw, origin_idx, row."""
    if not any(len(e) for e in eligible):
        raise ValueError("S2: no (SKU, date) pair is eligible in the window")
    rng = np.random.default_rng(seed)
    draws = []
    while len(draws) < m:
        d = int(rng.integers(len(window)))
        pool = eligible[d]
        if len(pool) == 0:
            continue
        draws.append((len(draws), window[d], int(pool[int(rng.integers(len(pool)))])))
    return pd.DataFrame(draws, columns=["draw", "origin_idx", "row"]).astype(
        {"draw": np.int64, "origin_idx": np.int64, "row": np.int64}
    )


def accept_s2(
    pool: pd.DataFrame, card_of_row: np.ndarray, spacing: int = S2_SPACING
) -> pd.DataFrame:
    """Backward-looking de-duplication: candidates sorted by (origin, draw) and
    accepted iff no previously accepted decision of the same card has origin in
    (t - spacing, t]. Adds `accepted`; returns the sorted pool."""
    ordered = pool.sort_values(["origin_idx", "draw"], kind="mergesort").reset_index(drop=True)
    last: dict[object, int] = {}
    accepted = np.zeros(len(ordered), dtype=bool)
    for i, (t, row) in enumerate(zip(ordered["origin_idx"], ordered["row"], strict=True)):
        card = card_of_row[row]
        if card in last and last[card] > t - spacing:
            continue
        last[card] = int(t)
        accepted[i] = True
    ordered["accepted"] = accepted
    return ordered


def price_band(price: float) -> str:
    if not np.isfinite(price) or price < S2_MIN_PRICE:
        raise ValueError(f"price {price!r} is below the S2 floor")
    for upper, label in S2_BANDS:
        if price < upper:
            return label
    return S2_TOP_BAND


def s2_decisions(data: QuoteData, accepted: pd.DataFrame) -> pd.DataFrame:
    """Accepted S2 decisions with their price band (q at the origin only)."""
    acc = accepted[accepted["accepted"]].reset_index(drop=True)
    bands = [
        price_band(float(data.qmat[:, : t + 1][row, t]))
        for t, row in zip(acc["origin_idx"], acc["row"], strict=True)
    ]
    out = acc[["draw", "origin_idx", "row"]].copy()
    out["price_band"] = bands
    return out


# --- S3 --------------------------------------------------------------------------


def s3_crossings(data: QuoteData, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Upward crossings at date column t: (rows, q(t - 7)). Reads columns <= t."""
    empty = (np.empty(0, dtype=np.int64), np.empty(0))
    if t < S3_LAG + 1:
        return empty
    q = data.qmat[:, : t + 1]
    obs = data.obs[:, : t + 1]
    prior = q[:, t - S3_LAG]
    prior_ok = np.isfinite(prior)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_t = q[:, t] / prior
        prev_prior = q[:, t - 1 - S3_LAG]
        ratio_prev = q[:, t - 1] / prev_prior
    defined_t = obs[:, t] & prior_ok
    defined_prev = obs[:, t - 1] & np.isfinite(prev_prior)
    with np.errstate(invalid="ignore"):
        cross = defined_t & (ratio_t >= S3_RATIO) & defined_prev & (ratio_prev < S3_RATIO)
        cross &= prior >= S3_MIN_PRIOR
    rows = np.flatnonzero(cross)
    if t - S3_MIN_AGE < 0 or t - S3_MIN_CONTEXT + 1 < 0:
        return empty
    rows = rows[_ever_observed_by(data, rows, t - S3_MIN_AGE)]
    rows = rows[_valid(q[rows, t - S3_MIN_CONTEXT + 1 :]).all(axis=1)]
    return rows, prior[rows]


def s3_window(data: QuoteData) -> list[int]:
    return list(range(data.day_index(S3_START), data.day_index(S3_END) + 1))


def select_s3(
    data: QuoteData, card_of_row: np.ndarray, window: list[int] | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Chronological S3 selection. Returns (events, counts); events has
    origin_idx, row (accepted only, date order) and counts has n_crossings
    (SKU level), n_after_same_day, n_accepted."""
    window = s3_window(data) if window is None else window
    last: dict[object, int] = {}
    kept: list[tuple[int, int]] = []
    n_raw = n_dedup = 0
    for t in window:
        rows, prior = s3_crossings(data, t)
        n_raw += len(rows)
        best: dict[object, tuple[float, int, int]] = {}
        for row, p in zip(rows.tolist(), prior.tolist(), strict=True):
            card = card_of_row[row]
            cand = (-p, int(data.rank[row]), row)  # largest q(t-7), then lowest SKU
            if card not in best or cand < best[card]:
                best[card] = cand
        n_dedup += len(best)
        for card, (_, _rank, row) in sorted(best.items(), key=lambda kv: kv[1][1]):
            if card in last and last[card] > t - S3_COOLDOWN:
                continue
            last[card] = t
            kept.append((t, row))
    events = pd.DataFrame(kept, columns=["origin_idx", "row"]).astype(
        {"origin_idx": np.int64, "row": np.int64}
    )
    counts = {"n_crossings": n_raw, "n_after_same_day": n_dedup, "n_accepted": len(events)}
    return events, counts


def s3_confirmatory(n_events: int, n_months: int) -> bool:
    return n_events >= S3_MIN_EVENTS and n_months >= S3_MIN_MONTHS


# --- ids -------------------------------------------------------------------------


def _sub(subtype: str | None) -> str:
    return "None" if subtype is None else str(subtype)


def unit_id(day: pd.Timestamp, sku: tuple[int, str | None]) -> str:
    return f"{pd.Timestamp(day).date().isoformat()}|{sku[0]}|{_sub(sku[1])}"


def s1_request_id(k: int, release: pd.Timestamp, sku: tuple[int, str | None]) -> str:
    return f"S1|k{k}|{pd.Timestamp(release).date().isoformat()}|{sku[0]}|{_sub(sku[1])}"


# --- requests --------------------------------------------------------------------


def build_requests(
    data: QuoteData,
    s1_units: pd.DataFrame,
    s2_dec: pd.DataFrame,
    s3_events: pd.DataFrame,
) -> pd.DataFrame:
    """The forecast requests of every selected unit (scaled contexts at the
    unit's origin; nothing after the origin is read)."""
    rid, study, block, horizon, ctx = [], [], [], [], []

    def add(request_id, name, blk, hor, row, origin):
        scaled = quotes.scaled_context(data.qmat, int(row), int(origin))
        rid.append(request_id)
        study.append(name)
        block.append(blk)
        horizon.append(hor)
        ctx.append(scaled.astype(np.float32))

    for u in s1_units.itertuples(index=False):
        k = int(u.k)
        add(
            s1_request_id(k, u.event_date, data.skus[u.row]),
            "S1",
            BLOCK_S1[k],
            s1_horizon(k),
            u.row,
            u.origin_idx,
        )
    for d in s2_dec.itertuples(index=False):
        day = data.dates[int(d.origin_idx)]
        add("S2|" + unit_id(day, data.skus[d.row]), "S2", BLOCK_S2, S2_HORIZON, d.row, d.origin_idx)
    for e in s3_events.itertuples(index=False):
        day = data.dates[int(e.origin_idx)]
        add("S3|" + unit_id(day, data.skus[e.row]), "S3", BLOCK_S3, S3_HORIZON, e.row, e.origin_idx)
    frame = pd.DataFrame(
        {
            "request_id": rid,
            "study": study,
            "block": np.asarray(block, dtype=np.int8),
            "horizon": np.asarray(horizon, dtype=np.int16),
            "context": ctx,
        },
        columns=list(requests.REQUEST_COLUMNS),
    )
    return frame


# --- census (counts only) --------------------------------------------------------


def _month(day: pd.Timestamp) -> str:
    return pd.Timestamp(day).strftime("%Y-%m")


def month_counts(days) -> dict[str, int]:
    return dict(sorted(Counter(_month(d) for d in days).items()))


def build_census(
    data: QuoteData,
    s1_counts: pd.DataFrame,
    eligible: list[np.ndarray],
    window: list[int],
    accepted: pd.DataFrame,
    s2_dec: pd.DataFrame,
    s3_events: pd.DataFrame,
    s3_counts: dict[str, int],
    card_of_row: np.ndarray,
    m: int = S2_POOL_SIZE,
    seed: int = S2_SEED,
) -> dict:
    """Counts only: no price, ratio or outcome."""
    s1 = {
        "n_events": int(s1_counts["event_date"].nunique()),
        "per_event": [
            {
                "event_date": r.event_date.date().isoformat(),
                "k": int(r.k),
                "n_candidates": int(r.n_candidates),
                "eligible_floor_0.25": int(r.eligible_floor_025),
                "eligible_floor_1.00": int(r.eligible_floor_100),
            }
            for r in s1_counts.itertuples(index=False)
        ],
        "totals_by_k": {
            str(k): {
                "eligible_floor_0.25": int(g["eligible_floor_025"].sum()),
                "eligible_floor_1.00": int(g["eligible_floor_100"].sum()),
            }
            for k, g in s1_counts.groupby("k")
        },
    }
    window_days = [data.dates[t] for t in window]
    eligible_per_month: Counter = Counter()
    for day, e in zip(window_days, eligible, strict=True):
        eligible_per_month[_month(day)] += len(e)
    pool_days = [data.dates[int(t)] for t in accepted["origin_idx"]]
    acc_days = [data.dates[int(t)] for t in s2_dec["origin_idx"]]
    s2 = {
        "window": [S2_START.isoformat(), S2_END.isoformat()],
        "M": int(m),
        "seed": int(seed),
        "eligible_pairs": int(sum(len(e) for e in eligible)),
        "dates_with_eligible": int(sum(1 for e in eligible if len(e))),
        "eligible_per_month": dict(sorted(eligible_per_month.items())),
        "candidates_drawn": int(len(accepted)),
        "accepted_N": int(len(s2_dec)),
        "rejected": int(len(accepted) - len(s2_dec)),
        "distinct_cards": len({card_of_row[r] for r in s2_dec["row"]}),
        "pool_candidates_per_month": month_counts(pool_days),
        "accepted_per_month": month_counts(acc_days),
        "price_band_counts": dict(sorted(Counter(s2_dec["price_band"]).items())),
    }
    s3_days = [data.dates[int(t)] for t in s3_events["origin_idx"]]
    n_months = len({_month(d) for d in s3_days})
    s3 = {
        "window": [S3_START.isoformat(), S3_END.isoformat()],
        "crossings_before_cooldown": int(s3_counts["n_crossings"]),
        "crossings_after_same_day_dedup": int(s3_counts["n_after_same_day"]),
        "crossings_after_cooldown": int(s3_counts["n_accepted"]),
        "N_events": int(len(s3_events)),
        "distinct_months": n_months,
        "distinct_cards": len({card_of_row[r] for r in s3_events["row"]}),
        "events_per_month": month_counts(s3_days),
        "confirmatory": s3_confirmatory(len(s3_events), n_months),
    }
    return {"s1": s1, "s2": s2, "s3": s3}


# --- loading the real panel ------------------------------------------------------

_DAY_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.parquet$")
_BLOCK_DAYS = 60


def _panel_files(panel_dir: Path, end: dt.date) -> list[tuple[dt.date, Path]]:
    files = []
    for path in sorted(Path(panel_dir).iterdir()):
        m = _DAY_FILE.match(path.name)
        if m and dt.date.fromisoformat(m.group(1)) <= end:
            files.append((dt.date.fromisoformat(m.group(1)), path))
    if not files:
        raise ValueError(f"no day files <= {end} in {panel_dir}")
    return files


def load_quote_data(
    panel_dir: Path, product_ids, end: dt.date | str | pd.Timestamp = LAST_ORIGIN
) -> QuoteData:
    """The quote data of the given products (all their subtypes) from the day
    files up to `end` inclusive. Days are read in blocks straight into a
    preallocated wide matrix (no giant long frame); SKUs that never have a
    valid price are dropped. Day columns are contiguous from the first to the
    last file date (days without a file are NaN)."""
    end = pd.Timestamp(end).date()
    files = _panel_files(panel_dir, end)
    ids = np.unique(np.asarray(list(product_ids), dtype=np.int64))
    id_filter = pc.field("product_id").isin(pa.array(ids, type=pa.int32()))

    # Pass 1: which (product_id, subtype) pairs exist (two light columns).
    pairs: set[tuple[int, str | None]] = set()
    for _, path in files:
        tbl = pq.read_table(path, columns=["product_id", "subtype"], filters=id_filter)
        if tbl.num_rows:
            df = tbl.to_pandas().drop_duplicates()
            pairs.update(
                (int(p), s if isinstance(s, str) else None)
                for p, s in zip(df["product_id"], df["subtype"], strict=True)
            )
    subtypes = sorted({s for _, s in pairs if s is not None})
    code = {s: i for i, s in enumerate(subtypes)}
    none_code = len(subtypes)
    width = none_code + 1
    ordered = sorted(pairs, key=lambda ps: (ps[0], none_code if ps[1] is None else code[ps[1]]))
    keys = np.array(
        [p * width + (none_code if s is None else code[s]) for p, s in ordered], np.int64
    )

    first, last = files[0][0], files[-1][0]
    dates = pd.date_range(first, last, freq="D", name="date")
    wide = np.full((len(ordered), len(dates)), np.nan, dtype=np.float64)
    for lo in range(0, len(files), _BLOCK_DAYS):
        chunk = files[lo : lo + _BLOCK_DAYS]
        long = load_panel(
            Path(panel_dir),
            keys=ordered,
            start=chunk[0][0],
            end=chunk[-1][0],
        )
        if long.empty:
            continue
        sub = pd.Categorical(long["subtype"], categories=subtypes).codes.astype(np.int64)
        sub = np.where(sub < 0, none_code, sub)
        row = np.searchsorted(keys, long["product_id"].to_numpy(np.int64) * width + sub)
        col = ((pd.to_datetime(long["date"]) - dates[0]) // pd.Timedelta(days=1)).to_numpy()
        price = long["price"].to_numpy(np.float64)
        ok = _valid(price)
        wide[row[ok], col[ok]] = price[ok]
        del long
    # Dropping SKUs with no valid price up to the load end looks at data through
    # `end`, but cannot change any decision: such SKUs have no valid quote at any
    # origin <= end and are therefore ineligible at every origin.
    keep = np.isfinite(wide).any(axis=1)
    skus = [s for s, k in zip(ordered, keep, strict=True) if k]
    return QuoteData.from_wide(dates, skus, wide[keep])
