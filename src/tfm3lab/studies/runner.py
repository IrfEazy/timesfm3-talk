"""Resumable, block-prioritised forecast runner for the v3 studies.

`run_requests` turns a validated requests frame (scaled contexts, no
outcomes) into forecast parquet parts:

    <out_dir>/block=<k>/part-<n>.parquet     one file per batch, written atomically
    <out_dir>/block_status.json              per block: requests, done,
                                             floored_values, seconds, status,
                                             requests_sha256

ORDER. Blocks run in ascending order (block 1 is the highest priority, so a
session that ends early still leaves the most valuable block complete). Within
a block, requests are grouped by horizon (ascending) and taken in request_id
order inside a group, then split into batches of at most `batch_size`. Every
batch is one `forecast_batch` call at its own max horizon, so no forward pass
decodes more steps than the batch needs. `use_symmetric_averaging=True` and
`make_positive=True` (the official defaults) are passed explicitly.

FLOORING. The contexts are scaled so the origin quote is ~1; a non-finite or
non-positive point/quantile value is meaningless as a price multiple and is
replaced by FLOOR_VALUE (0.001). The number of replaced values is stored in
the part and summed into the block's `floored_values`.

RESUME. A part file is the commit unit: the bookkeeping that belongs to it
(floored values, seconds) is stored in the part's own parquet schema metadata,
so a crash can never leave counts and data disagreeing, and `block_status.json`
is always re-derivable from the parts. On resume every existing part is read
back and verified (request_ids known to the requests frame, exactly steps
1..horizon, no request in two parts); requests already present are skipped and
never written twice. A part that contradicts the requests (foreign request_id,
partial or duplicated steps) raises ValueError instead of being guessed around.
`resume=False` refuses to touch a directory that already holds parts.

PROVENANCE. Every part carries the sha256 of the requests it was forecast from
(`requests_digest`: a canonical serialisation of the validated frame, so row
order and parquet encoding do not matter), and `block_status.json` repeats it
in each block record. A resume checks the metadata of every part under out_dir,
including blocks the current frame does not contain: parts written from other
requests, or by a runner that recorded no digest, raise ValueError telling the
caller to use a fresh output directory. The read-only helpers
(`current_status`, `pending_counts`) apply the same check.
"""

from __future__ import annotations

import dataclasses
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import cellstore, config
from ..model import Forecaster, forecast_batch
from .requests import (
    BLOCK_MAX,
    BLOCK_MIN,
    FORECASTS_SCHEMA,
    QUANTILE_COLUMNS,
    requests_digest,
    validate_requests_frame,
)

FLOOR_VALUE = 0.001
STATUS_FILE = "block_status.json"
PART_GLOB = "part-*.parquet"

_PART_RE = re.compile(r"^part-(\d{6,})\.parquet$")
_META_FLOORED = b"tfm3lab.floored_values"
_META_SECONDS = b"tfm3lab.seconds"
_META_REQUESTS = b"tfm3lab.requests"
_META_REQUESTS_SHA = b"tfm3lab.requests_sha256"


@dataclasses.dataclass
class BlockScan:
    """What the parts already on disk say about one block."""

    done: set[str] = dataclasses.field(default_factory=set)
    floored: int = 0
    seconds: float = 0.0
    n_parts: int = 0
    next_part: int = 0


def block_dir(out_dir: str | Path, block: int) -> Path:
    return Path(out_dir) / f"block={block}"


def _existing_parts(out_dir: Path) -> list[Path]:
    return sorted(p for p in out_dir.glob(f"block=*/{PART_GLOB}") if _PART_RE.match(p.name))


def _check_provenance(out_dir: Path, requests_sha: str) -> None:
    """Refuse parts that were not forecast from exactly these requests. Only
    the parquet footers are read."""
    for path in _existing_parts(out_dir):
        meta = pq.read_schema(path).metadata or {}
        found = meta.get(_META_REQUESTS_SHA)
        if found is None:
            raise ValueError(
                f"{path}: no requests sha256 recorded, so it cannot be tied to this requests "
                "file; use a fresh output directory (--out) instead of resuming it"
            )
        if found.decode() != requests_sha:
            raise ValueError(
                f"{path}: forecast from other requests (requests sha256 {found.decode()[:12]}, "
                f"these requests {requests_sha[:12]}); resuming would mix two request sets. "
                "Use a fresh output directory (--out) for this requests file"
            )


def _scan_block(directory: Path, horizons: dict[str, int]) -> BlockScan:
    """Read back every part of a block and verify it against `horizons`
    ({request_id: horizon} of this block's requests)."""
    scan = BlockScan()
    if not directory.is_dir():
        return scan
    for path in sorted(directory.glob(PART_GLOB)):
        match = _PART_RE.match(path.name)
        if not match:
            continue
        frame = pq.read_table(path, columns=["request_id", "step"]).to_pandas()
        ids = frame["request_id"]
        expected = ids.map(horizons)
        unknown = expected.isna().to_numpy()
        if unknown.any():
            raise ValueError(
                f"{path}: request_id {ids.iloc[int(np.argmax(unknown))]!r} is not a request of "
                f"this block; the forecast directory does not belong to this requests file"
            )
        repeated = ids.isin(scan.done).to_numpy()
        if repeated.any():
            raise ValueError(
                f"{path}: request_id {ids.iloc[int(np.argmax(repeated))]!r} already has "
                f"forecasts in an earlier part"
            )
        step = frame["step"].to_numpy(dtype=np.int64)
        h = expected.to_numpy(dtype=np.int64)
        broken = (step < 1) | (step > h) | frame.duplicated(["request_id", "step"]).to_numpy()
        counts = frame.groupby("request_id").size()
        short = counts.to_numpy() != counts.index.map(horizons).to_numpy(dtype=np.int64)
        if broken.any() or short.any():
            rid = (
                ids.iloc[int(np.argmax(broken))]
                if broken.any()
                else counts.index[int(np.argmax(short))]
            )
            raise ValueError(f"{path}: request_id {rid!r} does not hold exactly steps 1..horizon")
        scan.done.update(counts.index)
        meta = pq.read_schema(path).metadata or {}
        scan.floored += int(meta.get(_META_FLOORED, 0))
        scan.seconds += float(meta.get(_META_SECONDS, 0.0))
        scan.n_parts += 1
        scan.next_part = max(scan.next_part, int(match.group(1)) + 1)
    return scan


def _floor(values: np.ndarray) -> tuple[np.ndarray, int]:
    bad = ~np.isfinite(values) | (values <= 0)
    n = int(bad.sum())
    if n:
        values = np.where(bad, np.float32(FLOOR_VALUE), values).astype(np.float32)
    return values, n


def _forecast_chunk(
    forecaster: Forecaster, chunk: pd.DataFrame, requests_sha: str
) -> tuple[pa.Table, int]:
    """One forecast_batch call -> a part table (schema metadata set, including
    the requests sha256) and the number of floored values. Results are
    re-associated by ts_id, never by position."""
    ids = chunk["request_id"].tolist()
    horizons = chunk["horizon"].to_numpy(dtype=np.int64)
    max_horizon = int(horizons.max())
    start = time.perf_counter()
    result = forecast_batch(
        forecaster,
        list(chunk["context"]),
        max_horizon,
        ts_ids=ids,
        use_symmetric_averaging=True,
        make_positive=True,
    )
    position = {ts_id: i for i, ts_id in enumerate(result.ts_ids)}
    order = np.fromiter((position[i] for i in ids), dtype=np.int64, count=len(ids))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        point = np.asarray(result.forecast, dtype=np.float32)[order]
        quantiles = np.asarray(result.quantiles, dtype=np.float32)[order]
    if point.shape != (len(ids), max_horizon) or quantiles.shape != (
        len(ids),
        max_horizon,
        config.N_QUANTILES,
    ):
        raise ValueError(
            f"forecaster returned point {point.shape} / quantiles {quantiles.shape}; expected "
            f"({len(ids)}, {max_horizon}) / ({len(ids)}, {max_horizon}, {config.N_QUANTILES})"
        )

    steps = np.concatenate([np.arange(1, h + 1, dtype=np.int16) for h in horizons])
    point_flat, floored = _floor(np.concatenate([point[i, :h] for i, h in enumerate(horizons)]))
    quantile_flat, floored_q = _floor(
        np.concatenate([quantiles[i, :h, :] for i, h in enumerate(horizons)], axis=0)
    )
    floored += floored_q
    columns = {
        "request_id": pa.array(np.repeat(np.asarray(ids, dtype=object), horizons), pa.string()),
        "step": pa.array(steps, pa.int16()),
        "point": pa.array(point_flat, pa.float32()),
    }
    for j, name in enumerate(QUANTILE_COLUMNS):
        columns[name] = pa.array(np.ascontiguousarray(quantile_flat[:, j]), pa.float32())
    seconds = time.perf_counter() - start
    table = pa.Table.from_pydict(columns, schema=FORECASTS_SCHEMA).replace_schema_metadata(
        {
            _META_FLOORED: str(floored).encode(),
            _META_SECONDS: repr(seconds).encode(),
            _META_REQUESTS: str(len(ids)).encode(),
            _META_REQUESTS_SHA: requests_sha.encode(),
        }
    )
    return table, floored


def _select_blocks(
    df: pd.DataFrame, blocks: Iterable[int] | None, echo: Callable[[str], None]
) -> list[int]:
    present = sorted(int(b) for b in df["block"].unique())
    if blocks is None:
        return present
    wanted = sorted({int(b) for b in blocks})
    bad = [b for b in wanted if not BLOCK_MIN <= b <= BLOCK_MAX]
    if bad:
        raise ValueError(f"blocks {bad} outside {BLOCK_MIN}..{BLOCK_MAX}")
    for b in wanted:
        if b not in present:
            echo(f"block {b}: no requests in this file, skipped")
    return [b for b in wanted if b in present]


def _block_horizons(group: pd.DataFrame) -> dict[str, int]:
    return dict(zip(group["request_id"], group["horizon"].astype(int), strict=True))


def _scan_all(
    df: pd.DataFrame, out_dir: Path, requests_sha: str
) -> tuple[dict[int, pd.DataFrame], dict[int, BlockScan]]:
    _check_provenance(out_dir, requests_sha)
    per_block = {int(b): g for b, g in df.groupby("block", sort=True)}
    scans = {
        b: _scan_block(block_dir(out_dir, b), _block_horizons(g)) for b, g in per_block.items()
    }
    return per_block, scans


def _status_record(n_requests: int, scan: BlockScan, requests_sha: str) -> dict:
    done = len(scan.done)
    status = "complete" if done == n_requests else "partial" if done else "pending"
    return {
        "requests": n_requests,
        "done": done,
        "floored_values": scan.floored,
        "seconds": round(scan.seconds, 3),
        "status": status,
        "requests_sha256": requests_sha,
    }


def block_status(
    per_block: dict[int, pd.DataFrame], scans: dict[int, BlockScan], requests_sha: str
) -> dict:
    return {
        str(b): _status_record(len(per_block[b]), scans[b], requests_sha) for b in sorted(per_block)
    }


def has_forecast_parts(out_dir: str | Path) -> bool:
    """True when out_dir already holds forecast part files of any block."""
    return bool(_existing_parts(Path(out_dir)))


def current_status(requests_df: pd.DataFrame, out_dir: str | Path) -> dict:
    """The per-block status of what is on disk (same shape as run_requests'
    return value). Read-only: nothing is written and no model is needed."""
    df = validate_requests_frame(requests_df)
    sha = requests_digest(df)
    per_block, scans = _scan_all(df, Path(out_dir), sha)
    return block_status(per_block, scans, sha)


def pending_counts(
    requests_df: pd.DataFrame, out_dir: str | Path, blocks: Iterable[int] | None = None
) -> dict[int, int]:
    """{block: requests without verified forecasts yet} for the selected
    blocks. Read-only: nothing is written and no model is needed."""
    df = validate_requests_frame(requests_df)
    out_dir = Path(out_dir)
    selected = _select_blocks(df, blocks, lambda _msg: None)
    per_block, scans = _scan_all(df, out_dir, requests_digest(df))
    return {b: len(per_block[b]) - len(scans[b].done) for b in selected}


def run_requests(
    requests_df: pd.DataFrame,
    forecaster: Forecaster,
    out_dir: str | Path,
    batch_size: int = 128,
    resume: bool = True,
    blocks: Iterable[int] | None = None,
    echo: Callable[[str], None] = print,
) -> dict:
    """Forecast every pending request of the selected blocks (default: all)
    and return the per-block status (also kept in out_dir/block_status.json).

    Raises FileExistsError when `resume` is False and out_dir already holds
    parts, and ValueError when existing parts contradict the requests or were
    forecast from other requests (different sha256)."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    df = validate_requests_frame(requests_df)
    sha = requests_digest(df)
    out_dir = Path(out_dir)
    selected = _select_blocks(df, blocks, echo)
    if not resume and _existing_parts(out_dir):
        raise FileExistsError(
            f"{out_dir} already holds forecast parts; pass resume=True to continue them "
            "(never overwritten, never duplicated)"
        )
    per_block, scans = _scan_all(df, out_dir, sha)
    status_path = out_dir / STATUS_FILE
    cellstore.atomic_write_json(status_path, block_status(per_block, scans, sha))

    for b in selected:
        group, scan = per_block[b], scans[b]
        directory = block_dir(out_dir, b)
        directory.mkdir(parents=True, exist_ok=True)
        for debris in directory.glob("part-*.parquet.tmp-*"):  # interrupted atomic writes
            debris.unlink()
        todo = group[~group["request_id"].isin(scan.done)].sort_values(
            ["horizon", "request_id"], kind="mergesort"
        )
        if todo.empty:
            echo(f"block {b}: {len(group)}/{len(group)} requests already done")
            continue
        echo(f"block {b}: {len(todo)} of {len(group)} requests to forecast")
        for horizon, same_horizon in todo.groupby("horizon", sort=True):
            for first in range(0, len(same_horizon), batch_size):
                chunk = same_horizon.iloc[first : first + batch_size]
                table, floored = _forecast_chunk(forecaster, chunk, sha)
                cellstore.atomic_write_table(
                    table, directory / f"part-{scan.next_part:06d}.parquet"
                )
                scan.next_part += 1
                scan.n_parts += 1
                scan.floored += floored
                scan.seconds += float(table.schema.metadata[_META_SECONDS])
                scan.done.update(chunk["request_id"])
                cellstore.atomic_write_json(status_path, block_status(per_block, scans, sha))
                echo(
                    f"block {b}: {len(scan.done)}/{len(group)} done "
                    f"(horizon {int(horizon)}, batch of {len(chunk)}"
                    f"{f', {floored} floored values' if floored else ''})"
                )
        echo(f"block {b}: {_status_record(len(group), scan, sha)['status']}")
    return block_status(per_block, scans, sha)
