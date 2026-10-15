"""S1 (new-set cold start) evaluation: pure functions on synthetic-testable
inputs. Nothing here reads files.

Inputs: the S1 SKU table (event_date, k, product_id, subtype, group_id,
group_abbreviation, card_id, rarity, finish, first_printing, floor_1), the
quote matrix (`quotes.quote_matrix`, with its `dates` and `skus` index) and the
TimesFM forecasts (`requests` forecast schema, scaled by the origin quote).

Definitions (R = event date, O = R + k origin, T = R + 90 target):
- q(O), q(T): frozen quotes; q(T) missing -> attrition (no outcome).
- move = log(q(T) / q(O)); signed error = log(forecast ratio) - move;
  absolute error = |signed error|.
- TimesFM forecast ratio = q50 at step 90 - k; naive ratio = 1 (q(O));
  cohort ratio = exp(median of log(q(R'+90) / q(R'+k))) over reference SKUs of
  reference events with R' + 90 <= O, same k and floor, stratum rarity x
  finish, falling back to pooled rarity, then to all, while < 20 reference
  SKUs. Fewer than 2 reference events -> no cohort forecast.
- Floor: floor 1.0 uses the table's `floor_1` flag; any other floor uses
  q(O) >= floor.
- Coverage: q10 <= q(T)/q(O) <= q90 (the 80% interval).

The per-SKU table holds ids, flags, ratios and errors only, never prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import bootstrap
from . import inference
from . import requests as rq

HORIZON_DAYS = 90
MIN_REF_SKUS = 20
MIN_REF_EVENTS = 2
MIN_EVENT_SKUS = 20
PRIMARY_K = 14
METHODS = ("tfm", "naive", "cohort")
SKU_COLUMNS = (
    "event_date",
    "k",
    "product_id",
    "subtype",
    "group_id",
    "group_abbreviation",
    "card_id",
    "rarity",
    "finish",
    "first_printing",
    "floor_1",
)
ID_COLUMNS = SKU_COLUMNS[:10]


def _sub(x) -> str | None:
    """Missing subtype: NaN/None/''/'None'/'nan' all map to None."""
    if not isinstance(x, str) or x.strip() in ("", "None", "nan", "NaN"):
        return None
    return x


def request_id(k: int, event_date, product_id: int, subtype) -> str:
    """S1|k<k>|<R ISO>|<product_id>|<subtype> (a missing subtype is 'None')."""
    r = pd.Timestamp(event_date).date().isoformat()
    return f"S1|k{int(k)}|{r}|{int(product_id)}|{_sub(subtype)}"


def _quote_at(qmat: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    out = np.full(rows.shape, np.nan)
    ok = (rows >= 0) & (cols >= 0) & (cols < qmat.shape[1])
    out[ok] = qmat[rows[ok], cols[ok]]
    return out


def _truthy(s: pd.Series) -> np.ndarray:
    if not (pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s)):
        yes, no = ("true", "1", "1.0", "yes"), ("false", "0", "0.0", "no", "")

        def parse(v):
            if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
                return False
            if isinstance(v, bool | np.bool_):
                return bool(v)
            t = str(v).strip().lower()
            if t in yes:
                return True
            if t in no:
                return False
            raise ValueError(f"unrecognised boolean value {v!r}")

        return np.array([parse(v) for v in s], dtype=bool)
    return s.fillna(False).astype(bool).to_numpy()


def _cohort_median(pool: pd.DataFrame, rarity, finish) -> tuple[float, str]:
    for level, sub in (
        ("stratum", pool[(pool["rarity"] == rarity) & (pool["finish"] == finish)]),
        ("rarity", pool[pool["rarity"] == rarity]),
        ("all", pool),
    ):
        if len(sub) >= MIN_REF_SKUS or level == "all":
            if sub.empty:
                return float("nan"), level
            return float(np.median(sub["move"].to_numpy())), level
    raise AssertionError("unreachable")


def sku_errors(
    skus: pd.DataFrame,
    dates: pd.DatetimeIndex,
    sku_index: pd.Index,
    qmat: np.ndarray,
    forecasts: pd.DataFrame,
    floor: float = 1.0,
) -> pd.DataFrame:
    """Per-SKU error table for SKUs passing `floor` (ids, flags, errors only)."""
    missing = [c for c in SKU_COLUMNS if c not in skus.columns]
    if missing:
        raise ValueError(f"SKU table is missing columns {missing}")
    qmat = np.asarray(qmat, dtype=float)
    df = skus[list(SKU_COLUMNS)].copy().reset_index(drop=True)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.normalize()
    df["k"] = df["k"].astype(int)
    df["product_id"] = df["product_id"].astype(np.int64)
    df["subtype"] = df["subtype"].map(_sub).astype(object)
    df["first_printing"] = _truthy(df["first_printing"])
    df["floor_1"] = _truthy(df["floor_1"])

    row_of = {(int(key[0]), _sub(key[1])): i for i, key in enumerate(sku_index)}
    rows = np.array(
        [
            row_of.get((int(p), _sub(s)), -1)
            for p, s in zip(df["product_id"], df["subtype"], strict=True)
        ],
        dtype=np.int64,
    )
    r_date = df["event_date"]
    o_date = r_date + pd.to_timedelta(df["k"], unit="D")
    t_date = r_date + pd.Timedelta(days=HORIZON_DAYS)
    col_o = dates.get_indexer(pd.DatetimeIndex(o_date))
    col_t = dates.get_indexer(pd.DatetimeIndex(t_date))
    q_o = _quote_at(qmat, rows, col_o)
    q_t = _quote_at(qmat, rows, col_t)

    if floor == 1.0:
        in_floor = df["floor_1"].to_numpy()
    else:
        with np.errstate(invalid="ignore"):
            in_floor = q_o >= floor
    has_origin = np.isfinite(q_o) & (q_o > 0)
    has_outcome = has_origin & np.isfinite(q_t) & (q_t > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        move = np.where(has_outcome, np.log(q_t / q_o), np.nan)

    df["_o_date"], df["_t_date"] = o_date, t_date
    df["move"] = move
    df["has_origin"] = has_origin
    df["has_outcome"] = has_outcome
    df["attrition"] = has_origin & ~has_outcome
    df["no_origin"] = ~has_origin
    df["_in_floor"] = in_floor

    # TimesFM forecast at step 90 - k, scaled by q(O).
    ids = [
        request_id(k, r, p, s)
        for k, r, p, s in zip(
            df["k"], df["event_date"], df["product_id"], df["subtype"], strict=True
        )
    ]
    f = forecasts[["request_id", "step", "q10", "q50", "q90"]].copy()
    f["step"] = f["step"].astype(int)
    dup = f.duplicated(["request_id", "step"])
    if dup.any():
        d = f[dup].iloc[0]
        raise ValueError(f"duplicate forecast row for {d['request_id']} step {int(d['step'])}")
    key = pd.DataFrame({"request_id": ids, "step": HORIZON_DAYS - df["k"].to_numpy()})
    fm = key.merge(f, on=["request_id", "step"], how="left")
    fq = fm[["q10", "q50", "q90"]].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        f_ok = np.isfinite(fq).all(axis=1) & (fq > 0).all(axis=1)
    df["tf_available"] = f_ok
    fq = np.where(f_ok[:, None], fq, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        tf_signed = np.log(fq[:, 1]) - move
    df["move_log"] = move
    df["signed_err_tfm"] = tf_signed
    df["signed_err_naive"] = -move
    with np.errstate(invalid="ignore"):
        ratio = np.exp(move)
        df["covered_80"] = np.where(
            f_ok & has_outcome, (fq[:, 0] <= ratio) & (ratio <= fq[:, 2]), np.nan
        )

    # Cohort curve: only reference SKUs with day-90 date <= origin of the target.
    ref = df[df["_in_floor"] & df["has_outcome"]]
    c_move = np.full(len(df), np.nan)
    n_ref_events = np.zeros(len(df), dtype=int)
    level = np.full(len(df), "", dtype=object)
    cache: dict[tuple, tuple[float, str]] = {}
    for (k, r), idx in df.groupby(["k", "event_date"], sort=True).groups.items():
        pos = df.index.get_indexer(idx)
        o_d = df["_o_date"].iat[pos[0]]
        pool = ref[(ref["k"] == k) & (ref["_t_date"] <= o_d)]
        n_ev = pool["event_date"].nunique()
        n_ref_events[pos] = n_ev
        if n_ev < MIN_REF_EVENTS:
            continue
        for p in pos:
            rar, fin = df["rarity"].iat[p], df["finish"].iat[p]
            ck = (k, r, rar, fin)
            if ck not in cache:
                cache[ck] = _cohort_median(pool, rar, fin)
            c_move[p], level[p] = cache[ck]
    df["n_ref_events"] = n_ref_events
    df["cohort_level"] = level
    df["cohort_available"] = np.isfinite(c_move)
    df["signed_err_cohort"] = c_move - move
    for m in METHODS:
        df[f"abs_err_{m}"] = df[f"signed_err_{m}"].abs()

    out_cols = [
        *ID_COLUMNS,
        "floor_1",
        "has_origin",
        "has_outcome",
        "attrition",
        "no_origin",
        "tf_available",
        "cohort_available",
        "n_ref_events",
        "cohort_level",
        "move_log",
        "signed_err_tfm",
        "signed_err_naive",
        "signed_err_cohort",
        "abs_err_tfm",
        "abs_err_naive",
        "abs_err_cohort",
        "covered_80",
    ]
    return df.loc[df["_in_floor"], out_cols].reset_index(drop=True)


# --- aggregation -------------------------------------------------------------


def _set_mask(per_sku: pd.DataFrame, with_cohort: bool) -> pd.Series:
    m = per_sku["has_outcome"] & per_sku["tf_available"]
    return m & per_sku["cohort_available"] if with_cohort else m


def event_summary(
    per_sku: pd.DataFrame, with_cohort: bool = False, min_skus: int = MIN_EVENT_SKUS
) -> pd.DataFrame:
    """One row per (k, event_date). The comparison set is the paired set: SKUs
    with an outcome and a TimesFM forecast (and a cohort forecast if
    `with_cohort`); SKUs lacking a forecast are removed from every method and
    counted (n_removed_missing_forecast)."""
    mask = _set_mask(per_sku, with_cohort)
    rows = []
    for (k, r), g in per_sku.groupby(["k", "event_date"], sort=True):
        p = g[mask.loc[g.index]]
        no_cohort = g["has_outcome"] & g["tf_available"] & ~g["cohort_available"]
        row = {
            "k": k,
            "event_date": r,
            "n_skus": len(g),
            "n_outcome": int(g["has_outcome"].sum()),
            "n_attrition": int(g["attrition"].sum()),
            "n_no_origin": int(g["no_origin"].sum()),
            "n_removed_missing_forecast": int((g["has_outcome"] & ~g["tf_available"]).sum()),
            "n_removed_no_cohort": int(no_cohort.sum()) if with_cohort else 0,
            "n_paired": len(p),
            "qualifies": len(p) >= min_skus,
        }
        for m in METHODS:
            row[f"mean_abs_err_{m}"] = p[f"abs_err_{m}"].mean() if len(p) else np.nan
            row[f"bias_{m}"] = p[f"signed_err_{m}"].mean() if len(p) else np.nan
        row["coverage_80"] = p["covered_80"].astype(float).mean() if len(p) else np.nan
        row["diff_tfm_naive"] = row["mean_abs_err_tfm"] - row["mean_abs_err_naive"]
        row["diff_tfm_cohort"] = row["mean_abs_err_tfm"] - row["mean_abs_err_cohort"]
        row["n_ref_events"] = int(g["n_ref_events"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _nan_interval() -> tuple[float, float]:
    return float("nan"), float("nan")


def _diff_result(diffs: np.ndarray, seed: int) -> dict:
    g = int(diffs.size)
    out: dict = {"G": g, "mean_diff": float(diffs.mean()) if g else float("nan")}
    out["t_ci_9833"] = out["t_ci_95"] = out["boot_ci_95"] = _nan_interval()
    out["sign_flip_p"] = float("nan")
    if g >= 2:
        out["t_ci_9833"] = inference.t_interval(diffs, inference.CI_CONFIRMATORY)[1:]
        out["t_ci_95"] = inference.t_interval(diffs, inference.CI_EXPLORATORY)[1:]
        out["sign_flip_p"] = inference.sign_flip_test(diffs, seed)
        out["boot_ci_95"] = inference.cluster_bootstrap_mean(
            diffs, np.arange(g), inference.CI_EXPLORATORY, inference.B, seed
        )
    return out


def compare_to_naive(
    per_sku: pd.DataFrame,
    k: int = PRIMARY_K,
    seed: int = inference.SEED,
    mask: pd.Series | None = None,
) -> dict:
    """TimesFM vs naive on events at `k` (negative mean_diff favours TimesFM).
    `mask` optionally restricts SKUs (e.g. first printings) before pairing."""
    sub = per_sku[per_sku["k"] == k]
    if mask is not None:
        sub = sub[mask.loc[sub.index]]
    ev = event_summary(sub)
    q = ev[ev["qualifies"]]
    diffs = q["diff_tfm_naive"].to_numpy(dtype=float)
    out = _diff_result(diffs, seed)
    out.update(
        k=int(k),
        n_events_total=len(ev),
        n_skus_paired=int(q["n_paired"].sum()),
        n_attrition=int(ev["n_attrition"].sum()),
        n_no_origin=int(ev["n_no_origin"].sum()),
        n_removed_missing_forecast=int(ev["n_removed_missing_forecast"].sum()),
        verdict=inference.s1_verdict(diffs, seed),
    )
    return out


def median_above_floor_mask(per_sku: pd.DataFrame, forecasts: pd.DataFrame) -> pd.Series:
    """Amendment A1 sensitivity mask, aligned to `per_sku`: True for SKUs whose
    TimesFM q50 path (after the floor) is above the floor at every step 1..90-k.
    A SKU without a complete path is False."""
    ids = pd.Series(
        [
            request_id(k, r, p, s)
            for k, r, p, s in zip(
                per_sku["k"],
                per_sku["event_date"],
                per_sku["product_id"],
                per_sku["subtype"],
                strict=True,
            )
        ]
    )
    need = (HORIZON_DAYS - per_sku["k"]).to_numpy()
    f = forecasts[["request_id", "step", "q50"]]
    f = f[f["request_id"].isin(set(ids))]
    f = f[f["step"].astype(int) <= f["request_id"].map(dict(zip(ids, need, strict=True)))]
    g = f.groupby("request_id")["q50"].agg(n="size", lo="min")
    n = ids.map(g["n"]).fillna(0).to_numpy()
    lo = ids.map(g["lo"]).to_numpy(dtype=float)
    floor = float(np.float32(rq.FLOOR_VALUE)) * (1.0 + rq.FLOOR_REL_TOL)
    with np.errstate(invalid="ignore"):
        ok = (n == need) & (lo > floor)
    return pd.Series(ok, index=per_sku.index)


def floor_sensitivity(
    per_sku_floor1: pd.DataFrame, forecasts: pd.DataFrame, seed: int = inference.SEED
) -> dict:
    """Exploratory (Amendment A1): the k = 14 floor_1 comparison on SKUs whose q50
    never hit the floor. 95% Student-t interval over events; no verdict."""
    mask = median_above_floor_mask(per_sku_floor1, forecasts)
    res = compare_to_naive(per_sku_floor1, PRIMARY_K, seed, mask)
    at_k = per_sku_floor1["k"] == PRIMARY_K
    return {
        "k": res["k"],
        "n_events": res["G"],
        "n_events_total": res["n_events_total"],
        "mean_diff": res["mean_diff"],
        "t_ci_95": res["t_ci_95"],
        "n_skus_paired": res["n_skus_paired"],
        "n_skus_median_above_floor": int((at_k & mask).sum()),
        "n_skus_floor_1": int(at_k.sum()),
    }


def primary_result(per_sku_floor1: pd.DataFrame, seed: int = inference.SEED) -> dict:
    """Primary S1 result: k = 14, floor_1."""
    return compare_to_naive(per_sku_floor1, PRIMARY_K, seed)


def cohort_comparison(
    per_sku: pd.DataFrame, k: int = PRIMARY_K, seed: int = inference.SEED
) -> dict:
    """Exploratory TimesFM vs cohort curve on the set where all three methods
    exist (events need >= 2 reference events, implied by a cohort forecast)."""
    ev = event_summary(per_sku[per_sku["k"] == k], with_cohort=True)
    q = ev[ev["qualifies"]]
    out = _diff_result(q["diff_tfm_cohort"].to_numpy(dtype=float), seed)
    out.update(k=int(k), n_skus_paired=int(q["n_paired"].sum()))
    return out


def bias_and_coverage(per_sku: pd.DataFrame, k: int = PRIMARY_K) -> dict:
    """Event-averaged signed error per method and 80% coverage (paired set)."""
    ev = event_summary(per_sku[per_sku["k"] == k])
    q = ev[ev["qualifies"]]
    out = {f"bias_{m}": float(q[f"bias_{m}"].mean()) if len(q) else float("nan") for m in METHODS}
    out["coverage_80"] = float(q["coverage_80"].mean()) if len(q) else float("nan")
    out["G"] = len(q)
    return out


def player_descriptive(
    per_sku: pd.DataFrame, k: int = PRIMARY_K, seed: int = inference.SEED
) -> pd.DataFrame:
    """Per rarity x finish: event-weighted median of log(q(T)/q(O)) and the
    event-weighted share of SKUs with a negative move, each with a 95% event
    bootstrap CI. Per-event statistics are taken over SKUs with an outcome."""
    sub = per_sku[(per_sku["k"] == k) & per_sku["has_outcome"]]
    rows = []
    for (rar, fin), g in sub.groupby(["rarity", "finish"], sort=True):
        ev = g.groupby("event_date")["move_log"].agg(
            med="median", share=lambda s: float((s < 0).mean()), n="size"
        )
        n_ev = len(ev)
        mult = bootstrap.cluster_multiplicities(n_ev, inference.B, np.random.default_rng(seed))
        med_reps = bootstrap.weighted_median(ev["med"].to_numpy()[None, :], mult)
        share_reps = (mult @ ev["share"].to_numpy()) / mult.sum(axis=1)
        m_lo, m_hi, _ = bootstrap.percentile_interval(med_reps, inference.CI_EXPLORATORY)
        s_lo, s_hi, _ = bootstrap.percentile_interval(share_reps, inference.CI_EXPLORATORY)
        rows.append(
            {
                "rarity": rar,
                "finish": fin,
                "n_events": n_ev,
                "n_skus": int(ev["n"].sum()),
                "median_log_move": float(np.median(ev["med"])),
                "median_ci_low": m_lo,
                "median_ci_high": m_hi,
                "share_down": float(ev["share"].mean()),
                "share_ci_low": s_lo,
                "share_ci_high": s_hi,
            }
        )
    return pd.DataFrame(rows)


def exploratory_results(
    per_sku_floor1: pd.DataFrame,
    per_sku_floor025: pd.DataFrame | None = None,
    seed: int = inference.SEED,
    forecasts: pd.DataFrame | None = None,
) -> dict:
    """All exploratory S1 analyses (95% intervals; never decision inputs).
    With `forecasts`, adds the Amendment A1 floor sensitivity."""
    fp = per_sku_floor1["first_printing"].astype(bool)
    out = {
        "cohort": cohort_comparison(per_sku_floor1, PRIMARY_K, seed),
        "k7": compare_to_naive(per_sku_floor1, 7, seed),
        "k28": compare_to_naive(per_sku_floor1, 28, seed),
        "first_printing": compare_to_naive(per_sku_floor1, PRIMARY_K, seed, fp),
        "reprint": compare_to_naive(per_sku_floor1, PRIMARY_K, seed, ~fp),
        "bias_coverage": bias_and_coverage(per_sku_floor1, PRIMARY_K),
        "player": player_descriptive(per_sku_floor1, PRIMARY_K, seed),
    }
    if per_sku_floor025 is not None:
        out["floor_025"] = compare_to_naive(per_sku_floor025, PRIMARY_K, seed)
    if forecasts is not None:
        out["exploratory_q50_above_floor"] = floor_sensitivity(per_sku_floor1, forecasts, seed)
    return out
