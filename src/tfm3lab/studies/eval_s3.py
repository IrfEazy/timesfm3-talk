"""S3 (spike hold or revert) evaluation: pure functions on synthetic-testable
inputs. Nothing here reads files.

Inputs: S3 events (event_id, product_id, subtype, event_date, card_id, rarity,
finish), the quote matrix (`quotes.quote_matrix` with its `dates` and `skus`
index) and TimesFM forecasts (request_id 'S3|<event_id>', steps 1..56, in units
of q(t)); t = event_date.

Methods at horizon h in {28 (primary), 56}, as forecast ratios f / q(t):
TimesFM = q50 at step h; hold = 1; revert = q(t-7)/q(t); halfway =
sqrt(q(t-7)/q(t)) (the geometric mean of hold and revert). Outcome
q(t+h) / q(t) with the frozen quote policy. Absolute log error
ALE = |log(f / q(t+h))|.

Comparison set at each h (identical for all methods): events with a quote at
t, at t-7 and at t+h, and a finite positive TimesFM q10/q50/q90 at step h.
Events with a missing q(t+h) are attrition (`n_attrition`), counted separately
from events lacking the other inputs. The per-event table never holds prices.

Primary: mean(ALE_TFM - ALE_hold) at h = 28, card cluster bootstrap and
28-day calendar-block bootstrap at 98.33%; verdict via
`verdict_two_procedures(lower_is_better=True)`. If the census says the study
is not confirmatory the result is labelled 'descriptive only' and no verdict
is emitted. Exploratory (95%): vs revert, vs halfway, h = 56 (block_days 56),
q10..q90 coverage, share of events with q(t+28) < q(t).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import inference

HORIZONS = (28, 56)
PRIMARY_H = 28
LAG = 7
METHODS = ("tfm", "hold", "revert", "halfway")
ID_COLUMNS = ("event_id", "product_id", "subtype", "event_date", "card_id", "rarity", "finish")
VERDICTS = {"beats": "beats hold", "loses": "loses to hold", "inconclusive": "inconclusive"}
DESCRIPTIVE = "descriptive only"


def _sub(x) -> str | None:
    if not isinstance(x, str) or x.strip() in ("", "None", "nan", "NaN"):
        return None
    return x


def request_id(event_id) -> str:
    return f"S3|{event_id}"


def _quote_at(qmat: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    out = np.full(rows.shape, np.nan)
    ok = (rows >= 0) & (cols >= 0) & (cols < qmat.shape[1])
    v = qmat[rows[ok], cols[ok]]
    out[ok] = np.where(np.isfinite(v) & (v > 0), v, np.nan)
    return out


def event_errors(
    events: pd.DataFrame,
    dates: pd.DatetimeIndex,
    sku_index: pd.Index,
    qmat: np.ndarray,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Per-event table: ids, flags, absolute log errors per method and horizon,
    80% coverage flags and the q(t+28) < q(t) indicator (no prices)."""
    missing = [c for c in ID_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"events table is missing columns {missing}")
    qmat = np.asarray(qmat, dtype=float)
    df = events[list(ID_COLUMNS)].copy().reset_index(drop=True)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.normalize()
    df["subtype"] = df["subtype"].map(_sub).astype(object)
    if df["event_id"].duplicated().any():
        raise ValueError("duplicate event_id")
    row_of = {(int(k[0]), _sub(k[1])): i for i, k in enumerate(sku_index)}
    rows = np.array(
        [
            row_of.get((int(p), _sub(s)), -1)
            for p, s in zip(df["product_id"], df["subtype"], strict=True)
        ],
        dtype=np.int64,
    )
    col_t = dates.get_indexer(pd.DatetimeIndex(df["event_date"]))
    q_t = _quote_at(qmat, rows, col_t)
    q_prev = _quote_at(qmat, rows, np.where(col_t >= 0, col_t - LAG, -1))
    df["has_origin"] = np.isfinite(q_t)
    df["has_revert"] = df["has_origin"] & np.isfinite(q_prev)
    with np.errstate(invalid="ignore", divide="ignore"):
        rev = q_prev / q_t

    f = forecasts[["request_id", "step", "q10", "q50", "q90"]].copy()
    f["step"] = f["step"].astype(int)
    dup = f.duplicated(["request_id", "step"])
    if dup.any():
        d = f[dup].iloc[0]
        raise ValueError(f"duplicate forecast row for {d['request_id']} step {int(d['step'])}")
    ids = [request_id(e) for e in df["event_id"]]

    ratio28 = None
    for h in HORIZONS:
        q_h = _quote_at(qmat, rows, np.where(col_t >= 0, col_t + h, -1))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = q_h / q_t
        key = pd.DataFrame({"request_id": ids, "step": h})
        fq = key.merge(f, on=["request_id", "step"], how="left")[["q10", "q50", "q90"]].to_numpy(
            float
        )
        with np.errstate(invalid="ignore"):
            f_ok = np.isfinite(fq).all(axis=1) & (fq > 0).all(axis=1)
        has_out = df["has_origin"].to_numpy() & np.isfinite(ratio)
        df[f"tf_available_{h}"] = f_ok
        df[f"has_outcome_{h}"] = has_out
        # attrition is measured on the has-origin-and-has-revert set so it does
        # not overlap with n_no_origin / n_no_revert
        base = df["has_revert"].to_numpy()  # has_revert implies has_origin
        df[f"attrition_{h}"] = base & ~np.isfinite(ratio)
        df[f"tf_unavailable_{h}"] = base & ~f_ok
        in_set = has_out & f_ok & df["has_revert"].to_numpy()
        df[f"in_set_{h}"] = in_set
        ratios = {
            "tfm": np.where(f_ok, fq[:, 1], np.nan),
            "hold": np.ones(len(df)),
            "revert": rev,
            "halfway": np.sqrt(rev),
        }
        with np.errstate(invalid="ignore", divide="ignore"):
            for m, r in ratios.items():
                df[f"ale_{m}_{h}"] = np.where(in_set, np.abs(np.log(r / ratio)), np.nan)
            df[f"covered_80_{h}"] = np.where(
                in_set, (fq[:, 0] <= ratio) & (ratio <= fq[:, 2]), np.nan
            )
        if h == PRIMARY_H:
            ratio28 = ratio
    df["down_28"] = np.where(df[f"in_set_{PRIMARY_H}"], ratio28 < 1.0, np.nan)
    return df


def _cis(values, sub, block_days: int, level: float, seed: int) -> dict:
    v = np.asarray(values, dtype=float)
    out = {
        "n": int(v.size),
        "mean": float(v.mean()) if v.size else float("nan"),
        "cluster_ci": (float("nan"), float("nan")),
        "block_ci": (float("nan"), float("nan")),
    }
    if v.size:
        out["cluster_ci"] = inference.cluster_bootstrap_mean(
            v, sub["card_id"].to_numpy(), level, inference.B, seed
        )
        out["block_ci"] = inference.calendar_block_bootstrap_mean(
            v, sub["event_date"].to_numpy(), block_days, level, inference.B, seed
        )
    return out


def primary_result(table: pd.DataFrame, confirmatory: bool, seed: int = inference.SEED) -> dict:
    """Mean(ALE_TFM - ALE_hold) at h = 28 at 98.33%. `confirmatory` is the
    census flag; if False the result is 'descriptive only', no verdict."""
    s = table[table[f"in_set_{PRIMARY_H}"]]
    diff = (s[f"ale_tfm_{PRIMARY_H}"] - s[f"ale_hold_{PRIMARY_H}"]).to_numpy(float)
    out = _cis(diff, s, PRIMARY_H, inference.CI_CONFIRMATORY, seed)
    out["level"] = inference.CI_CONFIRMATORY
    out["n_attrition"] = int(table[f"attrition_{PRIMARY_H}"].sum())
    if not confirmatory:
        out["verdict"] = DESCRIPTIVE
    elif not out["n"]:
        out["verdict"] = VERDICTS["inconclusive"]
    else:
        out["verdict"] = VERDICTS[
            inference.verdict_two_procedures(
                out["cluster_ci"], out["block_ci"], lower_is_better=True
            )
        ]
    return out


def exploratory_results(table: pd.DataFrame, seed: int = inference.SEED) -> dict:
    """95% intervals, same two procedures, no verdict wording."""
    out: dict = {}
    for h in HORIZONS:
        s = table[table[f"in_set_{h}"]]
        bd = h
        out[f"h{h}"] = {
            "n_attrition": int(table[f"attrition_{h}"].sum()),
            "tfm_minus_hold": _cis(
                s[f"ale_tfm_{h}"] - s[f"ale_hold_{h}"], s, bd, inference.CI_EXPLORATORY, seed
            ),
            "tfm_minus_revert": _cis(
                s[f"ale_tfm_{h}"] - s[f"ale_revert_{h}"], s, bd, inference.CI_EXPLORATORY, seed
            ),
            "tfm_minus_halfway": _cis(
                s[f"ale_tfm_{h}"] - s[f"ale_halfway_{h}"], s, bd, inference.CI_EXPLORATORY, seed
            ),
            "coverage_80": float(s[f"covered_80_{h}"].astype(float).mean())
            if len(s)
            else float("nan"),
        }
    s = table[table[f"in_set_{PRIMARY_H}"]]
    out["share_down_28"] = float(s["down_28"].astype(float).mean()) if len(s) else float("nan")
    return out


def evaluate(
    events, dates, sku_index, qmat, forecasts, confirmatory: bool, seed: int = inference.SEED
) -> dict:
    table = event_errors(events, dates, sku_index, qmat, forecasts)
    return {
        "table": table,
        "primary": primary_result(table, confirmatory, seed),
        "exploratory": exploratory_results(table, seed),
        "counts": {
            "n_events": len(table),
            "n_no_origin": int((~table["has_origin"]).sum()),
            "n_no_revert": int((table["has_origin"] & ~table["has_revert"]).sum()),
            **{f"n_tf_unavailable_{h}": int(table[f"tf_unavailable_{h}"].sum()) for h in HORIZONS},
            **{f"n_attrition_{h}": int(table[f"attrition_{h}"].sum()) for h in HORIZONS},
        },
    }
