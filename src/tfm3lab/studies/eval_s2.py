"""S2 (buy now or wait) evaluation: pure functions on synthetic-testable
inputs. Nothing here reads files.

Inputs: the S2 decisions table (decision_id, product_id, subtype, origin,
card_id, rarity, finish, price_band), the quote matrix (`quotes.quote_matrix`
with its `dates` and `skus` index) and the TimesFM forecasts (request_id
'S2|<decision_id>', steps 1..28, already in units of q(t), median = q50).

Definitions, with t the origin and q the frozen quote:
- Path: m_0 = 1, m_d for d = 1..28 (forecast median / baseline forecast of the
  scaled context, last value 1). Baselines (ETS, drift) are computed HERE from
  `quotes.scaled_context(qmat, sku, t)` (<= 512 values), horizon 28.
- Policies choose d* in 0..28 (ties -> earliest day): P_TFM argmin of the
  TimesFM path (primary); P_ETS, P_drift the same rule on their paths; P_TFM5
  waits only if min_{d>=1} m_d <= 0.95 and then executes at the argmin, else
  d* = 0. References: buy-now (0), deadline (28), random-day (mean over
  d = 0..28 of the executed advantage), oracle (best observable advantage,
  not a policy).
- Execution (R6): if q(t+d*) is missing, execute at the first later d' in
  (d*, 28] with a non-missing quote; if none, the decision FAILS and gets
  A = min over d in 0..28 of the advantage at non-missing quotes.
  Advantage A = 1 - e / q(t) (positive = cheaper than buying now).
- FROZEN FALLBACK RULE (baselines only): if an ETS/drift fit fails
  (exception, non-finite or non-positive path) the policy executes buy-now
  (d* = 0) for that decision and the count is reported
  (`n_fit_failed_<policy>`).
- TimesFM unavailability (design R9): a decision whose TimesFM path is missing
  or non-finite does NOT fall back to buy-now; it is removed from every
  TimesFM comparison (primary, tfm5, tfm-ets, tfm-drift and the reference rows
  reported alongside them) so all comparisons are paired on one analysis set
  (`n_tfm_unavailable`; sizes in `n_analysis`, `n_with_origin`).
- Every sampled decision with an origin quote is in the primary estimand;
  decisions without one (no context to scale) are counted (`n_no_origin`).

The per-decision table holds ids, dates, flags and advantages, never prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import baselines
from . import inference, quotes

HORIZON = 28
WAIT_THRESHOLD = 0.95
BLOCK_DAYS = 28
POLICIES = ("tfm", "ets", "drift", "tfm5", "buy_now", "deadline")
ID_COLUMNS = (
    "decision_id",
    "product_id",
    "subtype",
    "origin",
    "card_id",
    "rarity",
    "finish",
    "price_band",
)
VERDICTS = {
    "beats": "beats buying now",
    "loses": "worse than buying now",
    "inconclusive": "inconclusive",
}


def _sub(x) -> str | None:
    """Missing subtype: NaN/None/''/'None'/'nan' all map to None."""
    if not isinstance(x, str) or x.strip() in ("", "None", "nan", "NaN"):
        return None
    return x


def request_id(decision_id) -> str:
    return f"S2|{decision_id}"


def choose_day(path: np.ndarray) -> int:
    """argmin of the 29-point path m_0..m_28; ties -> earliest day."""
    return int(np.argmin(path))


def choose_day_tfm5(path: np.ndarray, threshold: float = WAIT_THRESHOLD) -> int:
    """Wait only if min_{d>=1} m_d <= threshold, then execute at the argmin."""
    return choose_day(path) if float(np.min(path[1:])) <= threshold else 0


def execute(ratios: np.ndarray, d_star: int) -> tuple[float, int, bool, bool]:
    """(advantage, executed day, delayed, failed) for planned day `d_star` and
    realised quote ratios q(t+d)/q(t), d = 0..28 (NaN = missing quote)."""
    adv = 1.0 - ratios
    if np.isfinite(ratios[d_star]):
        return float(adv[d_star]), d_star, False, False
    later = np.flatnonzero(np.isfinite(ratios[d_star + 1 :]))
    if later.size:
        d = d_star + 1 + int(later[0])
        return float(adv[d]), d, True, False
    return float(np.nanmin(adv)), -1, False, True


def _valid_path(p) -> np.ndarray | None:
    try:
        arr = np.asarray(p, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.shape != (HORIZON,) or not np.all(np.isfinite(arr)) or not np.all(arr > 0):
        return None
    return arr


def _baseline_path(fn, ctx: np.ndarray) -> np.ndarray | None:
    try:
        return _valid_path(fn(ctx, HORIZON))
    except Exception:  # frozen rule: any fit failure -> buy-now fallback
        return None


def _tfm_paths(forecasts: pd.DataFrame) -> dict[str, np.ndarray]:
    f = forecasts[["request_id", "step", "q50"]].copy()
    f["step"] = f["step"].astype(int)
    dup = f.duplicated(["request_id", "step"])
    if dup.any():
        d = f[dup].iloc[0]
        raise ValueError(f"duplicate forecast row for {d['request_id']} step {int(d['step'])}")
    f = f[(f["step"] >= 1) & (f["step"] <= HORIZON)]
    out: dict[str, np.ndarray] = {}
    for rid, g in f.groupby("request_id", sort=False):
        arr = np.full(HORIZON, np.nan)
        arr[g["step"].to_numpy() - 1] = g["q50"].to_numpy(dtype=float)
        out[str(rid)] = arr
    return out


def decision_advantages(
    decisions: pd.DataFrame,
    dates: pd.DatetimeIndex,
    sku_index: pd.Index,
    qmat: np.ndarray,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Per-decision table: ids, flags, planned/executed days and advantages
    for each policy and reference (ids, flags and advantages only)."""
    missing = [c for c in ID_COLUMNS if c not in decisions.columns]
    if missing:
        raise ValueError(f"decisions table is missing columns {missing}")
    qmat = np.asarray(qmat, dtype=float)
    df = decisions[list(ID_COLUMNS)].copy().reset_index(drop=True)
    df["origin"] = pd.to_datetime(df["origin"]).dt.normalize()
    df["subtype"] = df["subtype"].map(_sub).astype(object)
    if df["decision_id"].duplicated().any():
        raise ValueError("duplicate decision_id")
    row_of = {(int(k[0]), _sub(k[1])): i for i, k in enumerate(sku_index)}
    tpaths = _tfm_paths(forecasts)
    cols = dates.get_indexer(pd.DatetimeIndex(df["origin"]))

    recs = []
    for i in range(len(df)):
        row = row_of.get((int(df["product_id"].iat[i]), df["subtype"].iat[i]), -1)
        col = int(cols[i])
        rec: dict = {"has_origin": False}
        ctx = np.array([])
        if row >= 0 and col >= 0:
            ctx = quotes.scaled_context(qmat, row, col)
        if ctx.size == 0:
            recs.append(rec)
            continue
        rec["has_origin"] = True
        q0 = qmat[row, col]
        ratios = np.full(HORIZON + 1, np.nan)
        for d in range(HORIZON + 1):
            if col + d < qmat.shape[1]:
                v = qmat[row, col + d] / q0
                ratios[d] = v if np.isfinite(v) and v > 0 else np.nan
        paths = {
            "tfm": _valid_path(tpaths.get(request_id(df["decision_id"].iat[i]))),
            "ets": _baseline_path(baselines.ets_forecast, ctx),
            "drift": _baseline_path(baselines.drift_forecast, ctx),
        }
        planned: dict[str, int] = {}
        for name, p in paths.items():
            rec[f"fit_failed_{name}"] = p is None
            full = None if p is None else np.concatenate([[1.0], p])
            planned[name] = 0 if full is None else choose_day(full)
            if name == "tfm":
                planned["tfm5"] = 0 if full is None else choose_day_tfm5(full)
        planned["buy_now"], planned["deadline"] = 0, HORIZON
        for name in POLICIES:
            a, d_exec, delayed, failed = execute(ratios, planned[name])
            rec[f"dstar_{name}"] = planned[name]
            rec[f"A_{name}"] = a
            rec[f"delayed_{name}"] = delayed
            rec[f"failed_{name}"] = failed
            rec[f"wait_{name}"] = planned[name] >= 1
            rec[f"hit_{name}"] = bool(a > 0) if planned[name] >= 1 else False
        per_day = [execute(ratios, d)[0] for d in range(HORIZON + 1)]
        rec["A_random"] = float(np.mean(per_day))
        rec["A_oracle"] = float(np.nanmax(1.0 - ratios))
        recs.append(rec)

    out = pd.DataFrame(recs)
    for c in out.columns:
        if c.startswith(("fit_failed_", "delayed_", "failed_", "wait_", "hit_")):
            out[c] = out[c].fillna(False).astype(bool)
    return pd.concat([df, out], axis=1)


def analysis_set(table: pd.DataFrame) -> pd.DataFrame:
    """Decisions with an origin quote AND an available TimesFM path (R9)."""
    return table[table["has_origin"] & ~table["fit_failed_tfm"]]


def summarise_policies(table: pd.DataFrame) -> pd.DataFrame:
    """Per policy: mean advantage and fail/delayed/wait/hit rates over the
    decisions of the analysis set (origin quote and TimesFM path)."""
    t = analysis_set(table)
    rows = []
    for name in (*POLICIES, "random", "oracle"):
        row: dict = {"policy": name, "n": len(t), "mean_A": float(t[f"A_{name}"].mean())}
        if name in POLICIES:
            waits = t[f"wait_{name}"]
            row.update(
                fail_rate=float(t[f"failed_{name}"].mean()),
                delayed_rate=float(t[f"delayed_{name}"].mean()),
                wait_share=float(waits.mean()),
                hit_rate=float(t.loc[waits, f"hit_{name}"].mean()) if waits.any() else np.nan,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_cis(values: np.ndarray, clusters, origins, level: float, seed: int) -> dict:
    v = np.asarray(values, dtype=float)
    out = {
        "n": int(v.size),
        "mean": float(v.mean()) if v.size else float("nan"),
        "cluster_ci": (float("nan"), float("nan")),
        "block_ci": (float("nan"), float("nan")),
    }
    if v.size:
        out["cluster_ci"] = inference.cluster_bootstrap_mean(v, clusters, level, inference.B, seed)
        out["block_ci"] = inference.calendar_block_bootstrap_mean(
            v, origins, BLOCK_DAYS, level, inference.B, seed
        )
    return out


def primary_result(table: pd.DataFrame, seed: int = inference.SEED) -> dict:
    """Confirmatory: mean A_TFM at 98.33% (card cluster bootstrap and 28-day
    calendar-block bootstrap); verdict from `verdict_two_procedures`.
    Computed on the analysis set (TimesFM path available)."""
    t = analysis_set(table)
    out = _mean_cis(
        t["A_tfm"].to_numpy(float),
        t["card_id"].to_numpy(),
        t["origin"].to_numpy(),
        inference.CI_CONFIRMATORY,
        seed,
    )
    out["level"] = inference.CI_CONFIRMATORY
    out["verdict"] = (
        VERDICTS["inconclusive"]
        if not out["n"]
        else VERDICTS[
            inference.verdict_two_procedures(
                out["cluster_ci"], out["block_ci"], lower_is_better=False
            )
        ]
    )
    return out


def exploratory_results(table: pd.DataFrame, seed: int = inference.SEED) -> dict:
    """95% intervals, same two procedures, no verdict wording, all on the
    analysis set (TimesFM path available)."""
    t = analysis_set(table)

    def ci(values, sub):
        return _mean_cis(
            np.asarray(values, float),
            sub["card_id"].to_numpy(),
            sub["origin"].to_numpy(),
            inference.CI_EXPLORATORY,
            seed,
        )

    out: dict = {
        "tfm_minus_ets": ci(t["A_tfm"] - t["A_ets"], t),
        "tfm_minus_drift": ci(t["A_tfm"] - t["A_drift"], t),
        "tfm5": ci(t["A_tfm5"], t),
    }
    for name in ("deadline", "random", "oracle", "buy_now"):
        out[name] = ci(t[f"A_{name}"], t)
    for by in ("price_band", "finish"):
        out[f"by_{by}"] = {
            str(k): ci(g["A_tfm"], g) for k, g in t.groupby(by, sort=True, dropna=False)
        }
    return out


def counts(table: pd.DataFrame) -> dict:
    """Fallback and exclusion counts to report."""
    t = table[table["has_origin"]]
    return {
        "n_decisions": len(table),
        "n_no_origin": int((~table["has_origin"]).sum()),
        "n_with_origin": len(t),
        "n_tfm_unavailable": int(t["fit_failed_tfm"].sum()),
        "n_analysis": len(analysis_set(table)),
        "n_fit_failed_ets": int(t["fit_failed_ets"].sum()),
        "n_fit_failed_drift": int(t["fit_failed_drift"].sum()),
    }


def evaluate(decisions, dates, sku_index, qmat, forecasts, seed: int = inference.SEED) -> dict:
    table = decision_advantages(decisions, dates, sku_index, qmat, forecasts)
    return {
        "table": table,
        "policies": summarise_policies(table),
        "counts": counts(table),
        "primary": primary_result(table, seed),
        "exploratory": exploratory_results(table, seed),
    }
