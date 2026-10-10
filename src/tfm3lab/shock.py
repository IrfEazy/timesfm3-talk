"""Experiment B (Preregistration v2 section 3): pure event, lag, anticipation,
perturbation and promotion logic. No network, no model: the forecasting loop
lives in scripts/03_exp_shock.py, the detector in data/market.py.

Conventions
-----------
- `origin_offset` = origin - onset_position, i.e. the TARGET offset of the
  h=1 forecast made at that origin. Post-onset points are offsets
  0 .. window_after-1 (25 points); an event that never recovers is censored
  at `window_after` (the window end), never dropped.
- Declustering chains: a shock day joins the running event when it is within
  `gap` trading days (union calendar of all indices) of the PREVIOUS shock day
  of that event. Onset = earliest shock day; `onset_index` = index with the
  largest |z| on the onset day (ties: alphabetical label).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

PRE_END = dt.date(2023, 5, 31)
POST_START = dt.date(2024, 6, 1)
DECLUSTER_GAP = 20
LAG_MULTIPLIERS = (1.25, 1.5, 2.0)
HEADLINE_MULTIPLIER = 1.5
CONSECUTIVE_POINTS = 3
WINDOW_AFTER = 25
WINDOW_BEFORE = 12
CONTEXT_LEN = 256
KEEP_RECENT = 32
NOISE_FRACTION = 0.25
SPLICE_MIN_GAP = 252  # one year of trading days
N_BOOT = 2000
BOOT_SEED = 20261010
N_PERTURB_SEEDS = 5
MIN_EVENTS_PER_ARM = 8
ANTICIPATION_OFFSETS = (-4, -3, -2, -1, 0)  # origin offsets <=> last observed at -5..-1


@dataclass(frozen=True)
class EventSpec:
    event_id: str
    source: str  # "rule" | "legacy"
    arm: str  # "pre" | "post"
    onset_date: dt.date
    onset_index: str


def assign_arm(onset_date: dt.date) -> str:
    if onset_date <= PRE_END:
        return "pre"
    if onset_date >= POST_START:
        return "post"
    return "guard"


def window_fits(
    pos: int,
    n: int,
    context_len: int = CONTEXT_LEN,
    window_before: int = WINDOW_BEFORE,
    window_after: int = WINDOW_AFTER,
) -> bool:
    """Context before the first origin, and window_after post-onset points."""
    return pos - window_before - context_len >= 0 and pos + window_after - 1 <= n - 1


def decluster(shock_days: pd.DataFrame, calendar, gap: int = DECLUSTER_GAP) -> list[dict]:
    """shock_days: columns date, index, z (one row per index per shock day).
    calendar: sorted union of trading dates. Returns one dict per event with
    onset_date, onset_index, triggering_indices (sorted tuple), max_abs_z (|z|
    of the onset index on the onset day), n_shock_days."""
    if len(shock_days) == 0:
        return []
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()
    df = shock_days.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["cal_pos"] = cal.get_indexer(df["date"])
    if (df["cal_pos"] < 0).any():
        raise ValueError("shock day not on the supplied calendar")
    df = df.sort_values(["cal_pos", "index"]).reset_index(drop=True)

    clusters, current, last_pos = [], [], None
    for row in df.itertuples():
        if last_pos is not None and row.cal_pos - last_pos > gap:
            clusters.append(current)
            current = []
        current.append(row)
        last_pos = row.cal_pos
    clusters.append(current)

    events = []
    for rows in clusters:
        onset_pos = rows[0].cal_pos
        day = [r for r in rows if r.cal_pos == onset_pos]
        best = sorted(day, key=lambda r: (-abs(r.z), r.index))[0]
        events.append(
            {
                "onset_date": pd.Timestamp(rows[0].date).date(),
                "onset_index": best.index,
                "triggering_indices": tuple(sorted({r.index for r in rows})),
                "max_abs_z": float(abs(best.z)),
                "n_shock_days": len(rows),
            }
        )
    return events


def event_id_for(onset_date: dt.date) -> str:
    return f"ev-{onset_date.isoformat()}"


# ---------------------------------------------------------------- lag


def error_recovery_lag(
    offsets,
    apes,
    multiplier: float = HEADLINE_MULTIPLIER,
    window_after: int = WINDOW_AFTER,
    consecutive: int = CONSECUTIVE_POINTS,
) -> tuple[float, bool]:
    """(lag, censored). threshold = multiplier x median pre-onset (offset < 0)
    APE of this forecaster; lag = first post-onset offset starting a run of
    `consecutive` points <= threshold, within offsets 0..window_after-1. No
    such run -> (window_after, True). NaN APE never counts as recovered;
    NaN threshold (no usable pre-onset errors) -> (nan, False)."""
    offsets = np.asarray(offsets)
    apes = np.asarray(apes, dtype=float)
    pre = apes[(offsets < 0) & np.isfinite(apes)]
    if len(pre) == 0:
        return float("nan"), False
    threshold = float(np.median(pre)) * multiplier
    order = np.argsort(offsets)
    off, err = offsets[order], apes[order]
    keep = (off >= 0) & (off < window_after)
    off, err = off[keep], err[keep]
    ok = np.isfinite(err) & (err <= threshold)
    for i in range(len(off) - consecutive + 1):
        if ok[i : i + consecutive].all() and off[i + consecutive - 1] - off[i] == consecutive - 1:
            return float(off[i]), False
    return float(window_after), True


def anticipation_hit(forecast_onset: float, last_observed: float, actual_move: float) -> bool:
    """sign(forecast for the onset day - last observed) == sign(actual onset move).
    A zero forecast move (sign 0) never counts as a hit unless the actual move is 0."""
    return bool(np.sign(forecast_onset - last_observed) == np.sign(actual_move))


# ---------------------------------------------------------------- inference


EFFECT_DEFINITIONS = ("post_minus_pre", "pre_minus_post")


def cluster_difference_ci(
    pre,
    post,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    ci: float = 0.95,
    effect_definition: str = "post_minus_pre",
) -> dict:
    """effect = mean(post) - mean(pre) ("post_minus_pre", default) or
    mean(pre) - mean(post) ("pre_minus_post", the preregistered anticipation
    direction); recorded in the `effect_definition` field. Percentile CI from an event-cluster
    bootstrap resampling events with replacement WITHIN each arm. NaN events
    are dropped (counted in n_pre/n_post as used)."""
    if effect_definition not in EFFECT_DEFINITIONS:
        raise ValueError(f"effect_definition must be one of {EFFECT_DEFINITIONS}")
    sign = 1.0 if effect_definition == "post_minus_pre" else -1.0
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    pre, post = pre[np.isfinite(pre)], post[np.isfinite(post)]
    out = {
        "n_pre": len(pre),
        "n_post": len(post),
        "mean_pre": float(pre.mean()) if len(pre) else float("nan"),
        "mean_post": float(post.mean()) if len(post) else float("nan"),
        "effect": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "n_boot": n_boot,
        "effect_definition": effect_definition,
    }
    if len(pre) == 0 or len(post) == 0:
        return out
    out["effect"] = sign * (out["mean_post"] - out["mean_pre"])
    rng = np.random.default_rng(seed)
    pre_draw = pre[rng.integers(0, len(pre), size=(n_boot, len(pre)))].mean(axis=1)
    post_draw = post[rng.integers(0, len(post), size=(n_boot, len(post)))].mean(axis=1)
    diff = sign * (post_draw - pre_draw)
    alpha = (1 - ci) / 2
    out["ci_low"], out["ci_high"] = (float(q) for q in np.quantile(diff, [alpha, 1 - alpha]))
    return out


def promotion_verdict(
    n_pre: int,
    n_post: int,
    effect: float,
    ci_low: float,
    ci_high: float,
    perturbed_effects: dict[str, float],
    min_events: int = MIN_EVENTS_PER_ARM,
) -> tuple[str, list[str]]:
    """("promoted" | "appendix", reasons). All three conditions must hold:
    >= min_events per arm, primary CI excludes 0, and both perturbation
    effects have the same (non-zero) sign as the primary effect. `reasons`
    lists every failed condition (empty when promoted)."""
    reasons = []
    if n_pre < min_events or n_post < min_events:
        reasons.append(f"fewer than {min_events} events per arm (pre={n_pre}, post={n_post})")
    if not (np.isfinite(ci_low) and np.isfinite(ci_high)) or ci_low <= 0 <= ci_high:
        reasons.append("primary CI does not exclude 0")
    names = ("noise", "splice")
    missing = [k for k in names if not np.isfinite(perturbed_effects.get(k, float("nan")))]
    if missing:
        reasons.append(f"perturbation effect unavailable: {', '.join(missing)}")
    elif not np.isfinite(effect) or not all(
        np.sign(perturbed_effects[k]) == np.sign(effect) != 0 for k in names
    ):
        reasons.append("perturbation effects do not share the primary effect's sign")
    return ("appendix" if reasons else "promoted"), reasons


# ---------------------------------------------------------------- perturbations


def add_log_noise(context, seed: int, fraction: float = NOISE_FRACTION) -> np.ndarray:
    """Additive Gaussian noise on log values, sigma = fraction x std of the
    context's log returns. Deterministic given `seed`."""
    context = np.asarray(context, dtype=float)
    logs = np.log(context)
    sigma = fraction * float(np.std(np.diff(logs), ddof=1)) if len(context) > 2 else 0.0
    rng = np.random.default_rng(seed)
    return np.exp(logs + rng.normal(0.0, sigma, size=len(logs)))


def splice_context(
    history,
    origin: int,
    context_len: int,
    seed: int,
    keep: int = KEEP_RECENT,
    min_gap: int = SPLICE_MIN_GAP,
) -> np.ndarray | None:
    """Context for `origin` with everything older than the last `keep` points
    replaced by a same-length segment of the same series that ends at least
    `min_gap` points before the origin (so only past data is used), level-
    shifted in log space so the segment's last value equals the replaced
    block's last value (continuous join). Returns None when the history is
    too short to hold such a segment. The segment end is drawn from `seed`
    and the origin."""
    history = np.asarray(history, dtype=float)
    ctx = history[max(0, origin - context_len) : origin]
    n_old = len(ctx) - keep
    if n_old < 1:
        return None
    hi = origin - min_gap  # exclusive end index of the segment, at most this
    if hi < n_old:
        return None
    rng = np.random.default_rng([int(seed), int(origin)])
    end = int(rng.integers(n_old, hi + 1))
    seg_log = np.log(history[end - n_old : end])
    seg_log = seg_log + (np.log(ctx[n_old - 1]) - seg_log[-1])
    return np.concatenate([np.exp(seg_log), ctx[n_old:]])


# ---------------------------------------------------------------- tables


def normalized_rows(
    origin_index,
    horizon_step,
    actual,
    forecast,
    last_observed,
    onset_pos: int,
) -> pd.DataFrame:
    """Committable rows: forecast and actual divided by the last observed
    context value; APE is scale-free. No raw prices are kept."""
    origin_index = np.asarray(origin_index)
    horizon_step = np.asarray(horizon_step)
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    last = np.asarray(last_observed, dtype=float)
    a, f = actual / last, forecast / last
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(f - a) / np.abs(a)
    return pd.DataFrame(
        {
            "origin_offset": (origin_index - onset_pos).astype("int32"),
            "horizon_step": horizon_step.astype("int32"),
            "actual_norm": a,
            "forecast_norm": f,
            "ape": ape,
        }
    )
