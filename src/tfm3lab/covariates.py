"""Covariates for Experiment D (Preregistration v2, section 5).

Pure helpers, no I/O and no model: the availability contract, the legitimate
covariates, the placebo constructions and the leaked-arm guard.

Availability contract. Every covariate is declared with a `CovariateSpec`
(name, available_at rule, future_known flag). A covariate enters a legitimate
arm only through `require_future_known`, which refuses any spec whose future
values are not known at the forecast origin. The ONLY way to feed the model
the real future is `teaching_only_leaked_covariate`, a deliberately named
bypass used for the watermarked teaching contrast.

Release-date limitation. The data source (TCGCSV groups) exposes `publishedOn`
only, not the date a set was announced. A release is treated as known at the
origin when `publishedOn - ANNOUNCE_LEAD_DAYS_PROXY <= origin date`. The lead
is an ASSUMPTION (a proxy for the real announcement date, which is not in the
data); the legitimate MTG covariate is only as honest as that proxy. When a
real announcement date is available, pass it via `announced_at`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 20261010
PLACEBO_OFFSET_RANGE_DAYS = (60, 300)  # inclusive, uniform
ANNOUNCE_LEAD_DAYS_PROXY = 60
MAX_DAYS_TO_RELEASE = 365.0

LEAKED_ARM = "leaked"
LEAKED_WATERMARK = "LEAKED — teaching contrast, not a result"


class CovariateAvailabilityError(ValueError):
    """A covariate whose future values are unknown at the forecast origin."""


class LeakedRowsError(ValueError):
    """Leaked-arm rows reached a summary/aggregation path."""


@dataclass(frozen=True)
class CovariateSpec:
    name: str
    available_at: str  # human-readable rule: when the value for date t is known
    future_known: bool  # True iff values for future dates are known at the origin


DAY_OF_WEEK = CovariateSpec(
    "day_of_week_sincos", "calendar: any date is known in advance", True
)
DAY_OF_WEEK_PLACEBO = CovariateSpec(
    "day_of_week_permuted", "calendar labels, seeded permutation (placebo)", True
)
DAYS_TO_RELEASE = CovariateSpec(
    "days_to_next_set_release",
    "release known once announced (proxy: publishedOn minus ANNOUNCE_LEAD_DAYS_PROXY)",
    True,
)
DAYS_TO_RELEASE_PLACEBO = CovariateSpec(
    "days_to_next_set_release_shifted",
    "release dates circularly shifted by a seeded 60-300 day offset (placebo)",
    True,
)
LEAKED_FUTURE = CovariateSpec(
    "own_future_values", "NOT known at the origin (the series' own future)", False
)


def require_future_known(spec: CovariateSpec) -> CovariateSpec:
    """Availability check: refuse covariates unknown at the forecast origin."""
    if not spec.future_known:
        raise CovariateAvailabilityError(
            f"covariate '{spec.name}' is not known at the forecast origin "
            f"({spec.available_at}); only the explicit teaching-only leaked path may use it"
        )
    return spec


# --- market: day of week -------------------------------------------------


def _sincos(dow: np.ndarray) -> np.ndarray:
    angle = 2 * np.pi * np.asarray(dow, dtype=float) / 7.0
    return np.stack([np.sin(angle), np.cos(angle)], axis=0)


def day_of_week_covariate(dates) -> np.ndarray:
    """Cyclical day-of-week encoding, shape (2, len(dates))."""
    return _sincos(pd.DatetimeIndex(dates).dayofweek.to_numpy())


def dow_permutation(seed: int = SEED) -> np.ndarray:
    """Seeded permutation of weekday labels 0..6; never the identity."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(7)
        if not np.array_equal(perm, np.arange(7)):
            return perm


def permuted_day_of_week_covariate(dates, perm: np.ndarray | None = None) -> np.ndarray:
    """PLACEBO: day-of-week labels replaced via a fixed permutation."""
    perm = dow_permutation() if perm is None else np.asarray(perm)
    return _sincos(perm[pd.DatetimeIndex(dates).dayofweek.to_numpy()])


# --- MTG: days to next set release ---------------------------------------


def days_to_next_release(
    dates,
    origin_date,
    release_dates,
    *,
    announced_at=None,
    announce_lead_days: int = ANNOUNCE_LEAD_DAYS_PROXY,
) -> np.ndarray:
    """Days from each date to the next release on or after it, using ONLY
    releases known at `origin_date`; shape (1, len(dates)), capped at 365
    (also used when no known release lies ahead).

    A release is known if its announcement date (`announced_at`, aligned with
    `release_dates`) is <= origin_date; without announcement dates the proxy
    publishedOn - announce_lead_days is used (see module docstring)."""
    dates64 = np.asarray(dates, dtype="datetime64[D]")
    rel = np.asarray(release_dates, dtype="datetime64[D]")
    if announced_at is None:
        ann = rel - np.timedelta64(int(announce_lead_days), "D")
    else:
        ann = np.asarray(announced_at, dtype="datetime64[D]")
        if len(ann) != len(rel):
            raise ValueError("announced_at must align with release_dates")
    known = np.sort(rel[ann <= np.datetime64(origin_date, "D")])
    out = np.full(len(dates64), MAX_DAYS_TO_RELEASE)
    for i, d in enumerate(dates64):
        j = np.searchsorted(known, d, side="left")
        if j < len(known):
            out[i] = float((known[j] - d).astype(int))
    return np.clip(out, 0, MAX_DAYS_TO_RELEASE)[None, :]


def placebo_offset_days(seed: int = SEED) -> int:
    """Seeded offset drawn uniformly from 60..300 days (inclusive)."""
    lo, hi = PLACEBO_OFFSET_RANGE_DAYS
    return int(np.random.default_rng(seed).integers(lo, hi + 1))


def shift_release_dates_circular(release_dates, start, end, offset_days: int) -> np.ndarray:
    """PLACEBO: shift release dates by `offset_days` on the circle of calendar
    days [start, end] (period = days in the span), so every date stays inside
    the span the series covers."""
    start = np.datetime64(start, "D")
    period = int((np.datetime64(end, "D") - start).astype(int)) + 1
    if period <= 0:
        raise ValueError("end must not precede start")
    rel = np.asarray(release_dates, dtype="datetime64[D]")
    pos = (rel - start).astype(int)
    return start + ((pos + int(offset_days)) % period).astype("timedelta64[D]")


# --- leaked arm (teaching only) ------------------------------------------


def teaching_only_leaked_covariate(values, origin: int, context_len: int, horizon: int):
    """TEACHING-ONLY leak: the series' own real past plus its real future,
    shape (1, context_len + horizon). Bypasses the availability contract on
    purpose; its output must never reach a summary (see `assert_no_leaked`)."""
    values = np.asarray(values, dtype=float)
    ctx = values[max(0, origin - context_len) : origin]
    return np.concatenate([ctx, values[origin : origin + horizon]])[None, :]


def assert_no_leaked(df: pd.DataFrame) -> None:
    """Raise if any row belongs to the leaked arm or carries its watermark."""
    if "arm" in df.columns and (df["arm"] == LEAKED_ARM).any():
        raise LeakedRowsError("leaked-arm rows must never enter a summary or headline")
    if "watermark" in df.columns and df["watermark"].astype(str).str.contains("LEAKED").any():
        raise LeakedRowsError("watermarked (LEAKED) rows must never enter a summary or headline")


# --- summary ----------------------------------------------------------------

COMPARISONS = (("legit", "no_covariate"), ("legit", "placebo"))
PRIMARY_STATISTIC = "weighted_mean"
SMALL_CLUSTER_N = 20


def verdict_from_cis(cis: list[tuple[float, float]]) -> str:
    """'helps' only if every CI lies strictly above 0 (legit vs BOTH
    no-covariate and placebo); 'does not help' if any CI's upper bound is <= 0;
    otherwise 'inconclusive'."""
    if not cis or any(not (np.isfinite(lo) and np.isfinite(hi)) for lo, hi in cis):
        return "inconclusive"
    if all(lo > 0 for lo, _ in cis):
        return "helps"
    if any(hi <= 0 for _, hi in cis):
        return "does not help"
    return "inconclusive"


def paired_skill_summary(
    preds: pd.DataFrame,
    *,
    primary_horizon: int,
    n_boot: int | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Paired skill delta (skill of the first arm relative to the second:
    1 - sum|e_a| / sum|e_b|) with the PRIMARY paired whole-series cluster
    bootstrap (tfm3lab.bootstrap), per domain x comparison x horizon cell
    ("<primary_horizon>" and the all-step pool "all"). Raises on leaked rows.

    `preds` needs: domain, series, origin_index, horizon_step, arm, observed,
    abs_err_norm. The verdict sits on every row of its domain, computed from
    the PRIMARY_STATISTIC at the primary horizon."""
    from . import bootstrap

    assert_no_leaked(preds)
    kw = {"seed": seed}
    if n_boot is not None:
        kw["n_boot"] = n_boot
    key = ["domain", "series", "origin_index", "horizon_step"]
    wide = (
        preds[preds["observed"]]
        .pivot_table(index=key, columns="arm", values="abs_err_norm", aggfunc="first")
        .dropna()
        .reset_index()
    )
    rows = []
    for domain, dwide in wide.groupby("domain"):
        cells = {
            str(primary_horizon): dwide[dwide["horizon_step"] == primary_horizon],
            "all": dwide,
        }
        verdict_cis = []
        domain_rows = []
        for horizon, cell in cells.items():
            for a, b in COMPARISONS:
                if cell.empty or a not in cell or b not in cell:
                    continue
                res = bootstrap.paired_cluster_bootstrap(
                    cell[a].to_numpy(), cell[b].to_numpy(), cell["series"].to_numpy(), **kw
                )
                for stat, ci in res.items():
                    if horizon == str(primary_horizon) and stat == PRIMARY_STATISTIC:
                        verdict_cis.append((ci.ci_low, ci.ci_high))
                    domain_rows.append(
                        {
                            "domain": domain,
                            "comparison": f"{a}_vs_{b}",
                            "horizon": horizon,
                            "statistic": stat,
                            "is_primary": horizon == str(primary_horizon)
                            and stat == PRIMARY_STATISTIC,
                            "skill_delta": ci.estimate,
                            "ci_low": ci.ci_low,
                            "ci_high": ci.ci_high,
                            "ci_level": ci.ci_level,
                            "method": ci.method,
                            "n_boot": ci.n_boot,
                            "seed": seed,
                            "n_invalid": ci.n_invalid,
                            "n_clusters": ci.n_clusters,
                            "label": "small-cluster/exploratory"
                            if ci.n_clusters < SMALL_CLUSTER_N
                            else "",
                        }
                    )
        verdict = verdict_from_cis(verdict_cis) if len(verdict_cis) == len(COMPARISONS) else (
            "inconclusive"
        )
        for r in domain_rows:
            r["verdict"] = verdict
        rows.extend(domain_rows)
    out = pd.DataFrame(rows)
    assert_no_leaked(out)
    return out
