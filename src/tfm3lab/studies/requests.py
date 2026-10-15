"""Frozen file schemas of the v3 studies, their IO and validators.

Two parquet files cross the local/GPU boundary:

requests (local -> Kaggle, written by the request builder):
    request_id  string, unique
    study       string in {S1, S2, S3}
    block       int8, priority 1..4 (the runner processes block 1 first)
    horizon     int16 >= 1
    context     list<float32>, length 1..512, every value finite and > 0

    A context is a scaled price trajectory (the origin quote is ~1). The
    schema has no column that could carry an outcome, absolute price or label:
    `read_requests` rejects any extra column, so a requests file cannot leak
    the future by accident.

forecasts (Kaggle -> local, written by the runner, one part file per batch):
    request_id  string
    step        int16, 1..horizon
    point       float32, the model point forecast
    q10..q90    float32, the nine model quantiles (config.QUANTILE_LEVELS)

`validate_forecasts` checks a forecasts frame against its requests. Every
validator raises ValueError naming the first offending request_id.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import cellstore, config

STUDIES: tuple[str, ...] = ("S1", "S2", "S3")
BLOCK_MIN, BLOCK_MAX = 1, 4
MAX_CONTEXT_LENGTH = 512
MAX_HORIZON = int(np.iinfo(np.int16).max)

REQUEST_COLUMNS: tuple[str, ...] = ("request_id", "study", "block", "horizon", "context")
QUANTILE_COLUMNS: tuple[str, ...] = tuple(f"q{round(lv * 100)}" for lv in config.QUANTILE_LEVELS)
FORECAST_COLUMNS: tuple[str, ...] = ("request_id", "step", "point", *QUANTILE_COLUMNS)

# Same tolerance as model.assert_quantile_shape: the checkpoint's own quantile
# check lets a crossing of up to 1e-6 through, so the file validator does too.
MONOTONE_TOLERANCE = 1e-6

REQUESTS_SCHEMA = pa.schema(
    [
        ("request_id", pa.string()),
        ("study", pa.string()),
        ("block", pa.int8()),
        ("horizon", pa.int16()),
        ("context", pa.list_(pa.float32())),
    ]
)
FORECASTS_SCHEMA = pa.schema(
    [
        ("request_id", pa.string()),
        ("step", pa.int16()),
        ("point", pa.float32()),
        *[(name, pa.float32()) for name in QUANTILE_COLUMNS],
    ]
)


def _fail(request_id: object, message: str) -> ValueError:
    return ValueError(f"request_id {request_id!r}: {message}")


# ---------------------------------------------------------------- requests


def _int_column(df: pd.DataFrame, name: str, lo: int, hi: int, ids: list[str]) -> np.ndarray:
    """Integer column within [lo, hi] as int64; the first offender is named."""
    series = df[name]
    if not pd.api.types.is_integer_dtype(series):
        raise ValueError(f"column {name!r} must have an integer dtype, got {series.dtype}")
    null = series.isna().to_numpy()
    if null.any():
        raise _fail(ids[int(np.argmax(null))], f"{name} is null")
    values = series.to_numpy(dtype=np.int64)
    bad = (values < lo) | (values > hi)
    if bad.any():
        i = int(np.argmax(bad))
        raise _fail(ids[i], f"{name}={int(values[i])} outside {lo}..{hi}")
    return values


def validate_requests_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a requests frame and return it normalised (int8 block, int16
    horizon, float32 1-D numpy contexts, fresh RangeIndex, input order kept).
    Raises ValueError with the offending request_id."""
    missing = [c for c in REQUEST_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUEST_COLUMNS]
    if missing or extra:
        raise ValueError(
            f"requests columns must be exactly {list(REQUEST_COLUMNS)}; "
            f"missing {missing}, unexpected {extra}"
        )

    raw_ids = df["request_id"].tolist()
    for i, rid in enumerate(raw_ids):
        if not isinstance(rid, str) or not rid:
            raise ValueError(f"row {i}: request_id must be a non-empty string, got {rid!r}")
    ids: list[str] = raw_ids
    if len(set(ids)) != len(ids):
        dupes = sorted(rid for rid, n in Counter(ids).items() if n > 1)
        raise _fail(dupes[0], f"request_id is not unique ({len(dupes)} duplicated ids)")

    study = df["study"]
    bad_study = ~study.isin(STUDIES).to_numpy()
    if bad_study.any():
        i = int(np.argmax(bad_study))
        raise _fail(ids[i], f"study={study.iloc[i]!r} not in {list(STUDIES)}")
    block = _int_column(df, "block", BLOCK_MIN, BLOCK_MAX, ids)
    horizon = _int_column(df, "horizon", 1, MAX_HORIZON, ids)

    contexts = np.empty(len(ids), dtype=object)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for i, raw in enumerate(df["context"].tolist()):
            if raw is None:
                raise _fail(ids[i], "context is null")
            try:
                ctx = np.asarray(raw, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise _fail(ids[i], f"context is not a numeric sequence ({exc})") from exc
            if ctx.ndim != 1 or not 1 <= ctx.size <= MAX_CONTEXT_LENGTH:
                raise _fail(
                    ids[i],
                    f"context must be 1-D with length 1..{MAX_CONTEXT_LENGTH}, "
                    f"got shape {ctx.shape}",
                )
            # min() is NaN for a NaN anywhere, so `> 0` also rejects NaN; -inf
            # fails `> 0` and +inf fails the finiteness test of the max.
            if not (ctx.min() > 0 and np.isfinite(ctx.max())):
                raise _fail(ids[i], "context must be finite and > 0 everywhere")
            contexts[i] = ctx

    return pd.DataFrame(
        {
            "request_id": ids,
            "study": study.astype(str).tolist(),
            "block": block.astype(np.int8),
            "horizon": horizon.astype(np.int16),
            "context": contexts,
        }
    )


def requests_digest(df: pd.DataFrame) -> str:
    """sha256 (hex) of a canonical serialisation of a requests frame.

    Independent of row order and of how the parquet file happened to be
    encoded: rows are taken in request_id order and each contributes its
    request_id, study, block, horizon and context length (JSON header line)
    followed by its context as little-endian float32. The runner stamps this
    digest on every forecast part, so a resume against different requests is
    refused. Expects a frame that passed `validate_requests_frame`."""
    digest = hashlib.sha256(b"tfm3lab.requests.v1\n")
    for row in df.sort_values("request_id", kind="mergesort").itertuples(index=False):
        context = np.asarray(row.context, dtype="<f4")
        header = [row.request_id, row.study, int(row.block), int(row.horizon), int(context.size)]
        digest.update(json.dumps(header).encode() + b"\n")
        digest.update(context.tobytes())
    return digest.hexdigest()


def _requests_table(clean: pd.DataFrame) -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array(clean["request_id"].tolist(), pa.string()),
            pa.array(clean["study"].tolist(), pa.string()),
            pa.array(clean["block"].to_numpy(), pa.int8()),
            pa.array(clean["horizon"].to_numpy(), pa.int16()),
            pa.array(clean["context"].tolist(), pa.list_(pa.float32())),
        ],
        schema=REQUESTS_SCHEMA,
    )


def write_requests(df: pd.DataFrame, path: str | Path) -> Path:
    """Validate `df` and write it atomically as the frozen requests parquet."""
    path = Path(path)
    cellstore.atomic_write_table(_requests_table(validate_requests_frame(df)), path)
    return path


def _type_matches(actual: pa.DataType, expected: pa.DataType) -> bool:
    if pa.types.is_list(expected):
        return pa.types.is_list(actual) and actual.value_type == expected.value_type
    return actual == expected


def _check_schema(actual: pa.Schema, expected: pa.Schema, what: str) -> None:
    missing = [n for n in expected.names if n not in actual.names]
    extra = [n for n in actual.names if n not in expected.names]
    if missing or extra:
        raise ValueError(
            f"{what} columns must be exactly {expected.names}; missing {missing}, "
            f"unexpected {extra}"
        )
    for field in expected:
        got = actual.field(field.name).type
        if not _type_matches(got, field.type):
            raise ValueError(f"{what} column {field.name!r} must be {field.type}, got {got}")


def read_requests(path: str | Path) -> pd.DataFrame:
    """Read and fully validate a requests parquet: exact column set and types,
    no nulls, unique request_id, study/block/horizon ranges, context length cap,
    finite and positive context values. Raises ValueError naming the offending
    request_id."""
    table = pq.read_table(Path(path))
    _check_schema(table.schema, REQUESTS_SCHEMA, "requests")
    ids = table.column("request_id")
    for name in REQUEST_COLUMNS:
        column = table.column(name)
        if column.null_count:
            null = np.asarray(column.is_null().to_numpy(zero_copy_only=False), dtype=bool)
            i = int(np.argmax(null))
            who = ids[i].as_py() if name != "request_id" else f"row {i}"
            raise ValueError(f"request_id {who!r}: {name} is null")
    return validate_requests_frame(table.to_pandas())


# ---------------------------------------------------------------- forecasts


def _forecast_table(df: pd.DataFrame) -> pa.Table:
    missing = [c for c in FORECAST_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in FORECAST_COLUMNS]
    if missing or extra:
        raise ValueError(
            f"forecast columns must be exactly {list(FORECAST_COLUMNS)}; "
            f"missing {missing}, unexpected {extra}"
        )
    arrays = []
    for field in FORECASTS_SCHEMA:
        values = df[field.name]
        if field.type == pa.string():
            arrays.append(pa.array(values.tolist(), pa.string()))
        else:
            arrays.append(pa.array(values.to_numpy(), field.type))
    return pa.Table.from_arrays(arrays, schema=FORECASTS_SCHEMA)


def write_forecasts(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a forecasts frame atomically with the frozen forecast schema."""
    path = Path(path)
    cellstore.atomic_write_table(_forecast_table(df), path)
    return path


def read_forecasts(path: str | Path) -> pd.DataFrame:
    """Read one forecast parquet, or every `block=*/part-*.parquet` below a
    directory, as one frame sorted by (request_id, step)."""
    path = Path(path)
    files = sorted(path.glob("block=*/part-*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no forecast parts under {path}")
    frames = []
    for file in files:
        table = pq.read_table(file)
        _check_schema(table.schema, FORECASTS_SCHEMA, f"forecasts ({file.name})")
        frames.append(table.to_pandas())
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["request_id", "step"], kind="mergesort").reset_index(drop=True)


# Preregistered floor (analysis-plan-v3 section 2, scaled: 0.001 x q(origin) ~ 0.001) and
# the relative tolerance used to recognise a floored value (Amendment A1).
FLOOR_VALUE = 0.001
FLOOR_REL_TOL = 1e-9


def floored_mask(values) -> np.ndarray:
    """True where a value equals the floor (float32 0.001, as the runner writes it)
    within FLOOR_REL_TOL relative."""
    ref = float(np.float32(FLOOR_VALUE))
    v = np.asarray(values, dtype=np.float64)
    return np.abs(v - ref) <= FLOOR_REL_TOL * ref


def rearrange_quantiles(forecasts_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Amendment A1: sort the nine quantiles of every row ascending and set the
    point forecast to the sorted q50 in changed rows. Rows already non-decreasing
    are left untouched (point included). Returns (new frame, number of changed rows)."""
    out = forecasts_df.copy()
    if not len(out):
        return out, 0
    cols = list(QUANTILE_COLUMNS)
    before = out[cols].to_numpy()
    after = np.sort(before, axis=1)
    changed = (after != before).any(axis=1)
    n = int(changed.sum())
    if n:
        for j, name in enumerate(cols):
            out.loc[changed, name] = after[changed, j].astype(out[name].dtype)
        out.loc[changed, "point"] = out.loc[changed, "q50"].astype(out["point"].dtype)
    return out, n


def _first_id(ids: pd.Series, mask: np.ndarray) -> str:
    return str(ids.to_numpy()[int(np.argmax(mask))])


def validate_forecasts(requests_df: pd.DataFrame, forecasts_df: pd.DataFrame) -> None:
    """Every request has exactly steps 1..horizon (no duplicates, no unknown
    request_id), all values finite, quantiles non-decreasing (within
    MONOTONE_TOLERANCE). Raises ValueError naming the first offending
    request_id; pass the matching subset of requests to validate a partial run."""
    missing = [c for c in FORECAST_COLUMNS if c not in forecasts_df.columns]
    if missing:
        raise ValueError(f"forecasts lack columns {missing}")
    horizons = requests_df.set_index("request_id")["horizon"]
    if not horizons.index.is_unique:
        raise ValueError("requests_df request_id is not unique")

    f = forecasts_df.reset_index(drop=True)
    ids = f["request_id"]
    horizon = ids.map(horizons)
    unknown = horizon.isna().to_numpy()
    if unknown.any():
        raise _fail(_first_id(ids, unknown), "forecast rows for a request that is not in requests")
    h = horizon.to_numpy(dtype=np.int64)
    step = f["step"].to_numpy(dtype=np.int64)

    dup = f.duplicated(["request_id", "step"]).to_numpy()
    if dup.any():
        raise _fail(_first_id(ids, dup), f"duplicate forecast step {int(step[np.argmax(dup)])}")
    out_of_range = (step < 1) | (step > h)
    if out_of_range.any():
        i = int(np.argmax(out_of_range))
        raise _fail(ids.iloc[i], f"step {int(step[i])} outside 1..{int(h[i])}")
    counts = f.groupby("request_id").size()
    have = horizons.index.to_series().map(counts).fillna(0).astype(np.int64)
    short = have.to_numpy() != horizons.to_numpy(dtype=np.int64)
    if short.any():
        # no duplicates and every step in range, so a short count means gaps
        rid = sorted(horizons.index[short])[0]
        raise _fail(rid, f"has {int(have[rid])} forecast steps, expected 1..{int(horizons[rid])}")

    values = f[["point", *QUANTILE_COLUMNS]].to_numpy(dtype=np.float64)
    nonfinite = ~np.isfinite(values).all(axis=1)
    if nonfinite.any():
        raise _fail(_first_id(ids, nonfinite), "non-finite point or quantile value")
    crossing = (np.diff(values[:, 1:], axis=1) < -MONOTONE_TOLERANCE).any(axis=1)
    if crossing.any():
        i = int(np.argmax(crossing))
        raise _fail(ids.iloc[i], f"quantiles not non-decreasing at step {int(step[i])}")
