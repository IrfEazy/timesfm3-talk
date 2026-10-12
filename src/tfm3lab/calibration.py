"""Experiment C: calibration of the P10-P90 interval by regime x horizon,
with chronological split-conformal (CQR) recalibration
(Preregistration v2 section 4). Pure functions over normalized frames.

Regimes: `mtg` (Exp A primary export), `market_shock` (|target_offset| <=
3 trading days from onset) and `event_window_nonshock` (other offsets
inside event windows; NOT a general calm sample). Coverage CIs: PRIMARY
synchronized calendar-block bootstrap on target dates (block = max(h, 7)
days), SENSITIVITY cluster bootstrap (cards / events), DIAGNOSTIC
design-adjusted Wilson. Thresholds are fitted on the calibration split only.
Results are empirical subgroup evaluations, not conformal guarantees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tfm3lab import bootstrap, conformal

NOMINAL = 0.8
ALPHA = 1 - NOMINAL
MTG_HORIZONS = (1, 7, 28, 56)
MARKET_HORIZONS = (1, 7, 28)
SHOCK_OFFSET = 3
MTG_SPLIT_DATE = pd.Timestamp("2025-06-01")
REGIMES = ("mtg", "market_shock", "event_window_nonshock")
FAMILY = {"mtg": "mtg", "market_shock": "market", "event_window_nonshock": "market"}
HORIZONS = {"mtg": MTG_HORIZONS, "market": MARKET_HORIZONS}
LABEL = "empirical subgroup evaluation, not a conformal guarantee"


def mtg_frame(primary: pd.DataFrame) -> pd.DataFrame:
    """Standard columns (regime, family, cluster, horizon, origin_date,
    target_date, y, lo, hi) from the Exp A primary export; observed rows
    with finite values only, requested horizons only."""
    df = primary
    if "mode" in df:
        df = df[df["mode"] == "timesfm3_univariate"]
    out = pd.DataFrame(
        {
            "regime": "mtg",
            "family": "mtg",
            "cluster": df["card"].astype(str).to_numpy(),
            "horizon": df["requested_horizon"].to_numpy(),
            "origin_date": pd.to_datetime(df["origin_date"]).to_numpy(),
            "target_date": pd.to_datetime(df["target_date"]).to_numpy(),
            "y": df["y_norm"].to_numpy(float),
            "lo": df["q10_norm"].to_numpy(float),
            "hi": df["q90_norm"].to_numpy(float),
            "arm": "",
            "target_offset": np.nan,
        }
    )
    keep = np.array(df["observed"], dtype=bool)
    if "scale_valid" in df:
        keep &= df["scale_valid"].to_numpy(bool)
    return _finalize(out[keep], MTG_HORIZONS)


def market_frame(pred: pd.DataFrame, shock_offset: int = SHOCK_OFFSET) -> pd.DataFrame:
    """Same columns from the Exp B v2 predictions (timesfm3_univariate,
    unperturbed). Regime from |target_offset|.

    Derivation: in scripts/03, `origin_offset` is the origin position minus
    the onset position, and `horizon_step` h (>= 1) forecasts position
    origin + h - 1 (anticipation rows: offset o with step 1 - o lands on the
    onset, o + (1 - o) - 1 = 0). So the day whose outcome the interval
    covers is at `target_offset = origin_offset + horizon_step - 1` trading
    days from onset; the preregistered "|offset| <= 3" applies to that day.
    """
    df = pred[(pred["forecaster"] == "timesfm3_univariate") & (pred["perturbation"] == "none")]
    target_offset = (df["origin_offset"] + df["horizon_step"] - 1).to_numpy()
    shocked = np.abs(target_offset) <= shock_offset
    out = pd.DataFrame(
        {
            "regime": np.where(shocked, "market_shock", "event_window_nonshock"),
            "family": "market",
            "cluster": df["event_id"].astype(str).to_numpy(),
            "horizon": df["horizon_step"].to_numpy(),
            "origin_date": pd.to_datetime(df["origin_date"]).to_numpy(),
            "target_date": pd.to_datetime(df["target_date"]).to_numpy(),
            "y": df["actual_norm"].to_numpy(float),
            "lo": df["q10_norm"].to_numpy(float),
            "hi": df["q90_norm"].to_numpy(float),
            "arm": df["arm"].astype(str).to_numpy(),
            "target_offset": target_offset,
        }
    )
    return _finalize(out, MARKET_HORIZONS)


def _finalize(out: pd.DataFrame, horizons) -> pd.DataFrame:
    ok = np.isfinite(out[["y", "lo", "hi"]].to_numpy()).all(axis=1) & out["target_date"].notna()
    out = out[ok & out["horizon"].isin(horizons)]
    return out.reset_index(drop=True)


def covered(df: pd.DataFrame, lo=None, hi=None) -> np.ndarray:
    lo = df["lo"].to_numpy() if lo is None else lo
    hi = df["hi"].to_numpy() if hi is None else hi
    y = df["y"].to_numpy()
    return (y >= lo) & (y <= hi)


def under_covers(ci_high: float) -> bool:
    """Preregistered verdict: primary CI upper bound < 0.80."""
    return bool(np.isfinite(ci_high) and ci_high < NOMINAL)


def conformal_fixes_it(ci_low: float, ci_high: float) -> bool | None:
    """Preregistered verdict: post-recalibration primary CI contains 0.80.
    None (undetermined) when the CI is not finite."""
    if not (np.isfinite(ci_low) and np.isfinite(ci_high)):
        return None
    return bool(ci_low <= NOMINAL <= ci_high)


def interval_score(y, lo, hi, alpha: float = ALPHA) -> np.ndarray:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)


def _cis(cov: np.ndarray, sub: pd.DataFrame, h: int, n_boot: int, seed: int) -> dict:
    cb = bootstrap.coverage_calendar_block_ci(
        cov, sub["cluster"].to_numpy(), sub["target_date"].to_numpy(), h, n_boot=n_boot, seed=seed
    )
    cl = bootstrap.coverage_cluster_ci(cov, sub["cluster"].to_numpy(), n_boot=n_boot, seed=seed)
    wi = bootstrap.coverage_wilson_design_adjusted(cov, sub["cluster"].to_numpy())
    return {
        "coverage": cb.estimate,
        "cb_low": cb.ci_low,
        "cb_high": cb.ci_high,
        "block_days": cb.block_days,
        "cb_n_invalid": cb.n_invalid,
        "cluster_low": cl.ci_low,
        "cluster_high": cl.ci_high,
        "wilson_low": wi.ci_low,
        "wilson_high": wi.ci_high,
        "wilson_policy": wi.policy,
    }


def coverage_table(
    frame: pd.DataFrame, n_boot: int = bootstrap.DEFAULT_N_BOOT, seed: int = bootstrap.DEFAULT_SEED
) -> pd.DataFrame:
    """Coverage of [q10, q90] by regime x horizon with the three CIs."""
    rows = []
    for regime in REGIMES:
        for h in HORIZONS[FAMILY[regime]]:
            sub = frame[(frame["regime"] == regime) & (frame["horizon"] == h)]
            row = {
                "regime": regime,
                "horizon": h,
                "n": len(sub),
                "n_clusters": sub["cluster"].nunique(),
            }
            row.update(_cis(covered(sub), sub, h, n_boot, seed))
            row["under_covers"] = under_covers(row["cb_high"])
            row["under_covers_cluster"] = under_covers(row["cluster_high"])
            row["label"] = LABEL
            rows.append(row)
    return pd.DataFrame(rows)


def split_chronological(
    frame: pd.DataFrame, split_date: pd.Timestamp = MTG_SPLIT_DATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(calibration, evaluation). MTG: calibration = origin AND target
    before split_date, evaluation = origin on/after it (rows straddling the
    date are in neither). Market: calibration = pre-arm events, evaluation
    = post-arm events (other arms dropped)."""
    mtg = frame["family"] == "mtg"
    cal = (mtg & (frame["origin_date"] < split_date) & (frame["target_date"] < split_date)) | (
        ~mtg & (frame["arm"] == "pre")
    )
    ev = (mtg & (frame["origin_date"] >= split_date)) | (~mtg & (frame["arm"] == "post"))
    return frame[cal], frame[ev]


PREREGISTERED = "preregistered"
EXPLORATORY = "exploratory_post_hoc_resplit"


def exploratory_split_date(frame: pd.DataFrame) -> pd.Timestamp:
    """Exploratory MTG split D' = o_{floor(N/2)+1}: the element at 1-based
    rank floor(N/2)+1 of the sorted distinct MTG origin dates (post hoc,
    amendment of 2026-10-12). Raises ValueError without MTG rows."""
    mtg = frame[frame["family"] == "mtg"]
    if len(mtg) == 0:
        raise ValueError("no MTG rows: cannot choose an exploratory split date")
    dates = np.sort(pd.to_datetime(mtg["origin_date"]).unique())
    return pd.Timestamp(dates[len(dates) // 2])


def conformal_table(
    frame: pd.DataFrame,
    n_boot: int = bootstrap.DEFAULT_N_BOOT,
    seed: int = bootstrap.DEFAULT_SEED,
    split_date: pd.Timestamp = MTG_SPLIT_DATE,
    exploratory: bool = True,
) -> pd.DataFrame:
    """CQR per (family, horizon); evaluation reported per (regime, horizon).

    Column `analysis` is 'preregistered' for the rows computed with
    `split_date`; `split_date` is the MTG split used and NaT for market rows
    (their split is by arm). With `exploratory` and MTG rows present, extra
    MTG-only rows with analysis='exploratory_post_hoc_resplit' are appended,
    split at exploratory_split_date(frame); market rows are not duplicated."""
    pre = _conformal_rows(frame, REGIMES, split_date, n_boot, seed, PREREGISTERED)
    parts = [pre]
    if exploratory and (frame["family"] == "mtg").any():
        d2 = exploratory_split_date(frame)
        parts.append(_conformal_rows(frame, ("mtg",), d2, n_boot, seed, EXPLORATORY))
    return pd.concat(parts, ignore_index=True)


def _conformal_rows(
    frame: pd.DataFrame,
    regimes,
    split_date: pd.Timestamp,
    n_boot: int,
    seed: int,
    analysis: str,
) -> pd.DataFrame:
    cal, ev = split_chronological(frame, split_date)
    rows = []
    for regime in regimes:
        fam = FAMILY[regime]
        for h in HORIZONS[fam]:
            c = cal[(cal["family"] == fam) & (cal["horizon"] == h)]
            e = ev[(ev["regime"] == regime) & (ev["horizon"] == h)]
            row = {
                "analysis": analysis,
                "family": fam,
                "regime": regime,
                "split_date": pd.Timestamp(split_date) if fam == "mtg" else pd.NaT,
                "horizon": h,
                "calibration_n": len(c),
                "eval_n": len(e),
            }
            if len(c) == 0:
                thr, status = float("nan"), "no_calibration_rows"
            else:
                thr = conformal.cqr_threshold(c["y"], c["lo"], c["hi"], NOMINAL)
                status = "ok" if np.isfinite(thr) else "insufficient_calibration"
            row["threshold"], row["status"] = thr, status
            cov_b = covered(e)
            before = _cis(cov_b, e, h, n_boot, seed)
            row.update(
                {
                    f"{k}_before": v
                    for k, v in before.items()
                    if k in ("coverage", "cb_low", "cb_high")
                }
            )
            row["under_covers_before"] = under_covers(before["cb_high"])
            if status == "ok" and len(e):
                lo2, hi2 = conformal.apply_cqr(e["lo"], e["hi"], thr)
                after = _cis(covered(e, lo2, hi2), e, h, n_boot, seed)
                w0, w1 = (e["hi"] - e["lo"]).to_numpy(), hi2 - lo2
                pos = w0 > 0
                row.update(
                    {
                        f"{k}_after": v
                        for k, v in after.items()
                        if k in ("coverage", "cb_low", "cb_high")
                    }
                )
                row["median_rel_width_increase"] = (
                    float(np.median((w1[pos] - w0[pos]) / w0[pos])) if pos.any() else float("nan")
                )
                row["mean_interval_score_before"] = float(
                    interval_score(e["y"], e["lo"], e["hi"]).mean()
                )
                row["mean_interval_score_after"] = float(interval_score(e["y"], lo2, hi2).mean())
                row["conformal_fixes_it"] = conformal_fixes_it(after["cb_low"], after["cb_high"])
            else:
                for k in (
                    "coverage_after",
                    "cb_low_after",
                    "cb_high_after",
                    "median_rel_width_increase",
                    "mean_interval_score_before",
                    "mean_interval_score_after",
                ):
                    row[k] = float("nan")
                row["conformal_fixes_it"] = None  # undetermined, not False
            row["label"] = LABEL
            rows.append(row)
    return pd.DataFrame(rows)
