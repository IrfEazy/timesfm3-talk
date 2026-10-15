"""Resumable, block-prioritised forecast runner for the v3 studies.

`run_requests` turns a validated requests frame (scaled contexts, no
outcomes) into forecast parquet parts:

    <out_dir>/block=<k>/part-<n>.parquet     one file per batch, written atomically
    <out_dir>/block_status.json              per block: requests, done,
                                             floored_values, seconds, status,
                                             requests_sha256, checkpoint_revision,
                                             use_symmetric_averaging, make_positive

ORDER. Blocks run in ascending order (block 1 is the highest priority, so a
session that ends early still leaves the most valuable block complete). Within
a block, requests are grouped by horizon (ascending) and taken in request_id
order inside a group, then split into batches of at most `batch_size`. Every
batch is one `forecast_batch` call at its own max horizon, so no forward pass
decodes more steps than the batch needs. `use_symmetric_averaging` and
`make_positive` (default True, the official defaults) are passed explicitly to
every forward pass and recorded as provenance (see PROVENANCE).

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
order and parquet encoding do not matter), the resolved checkpoint revision
the forecaster was loaded at, and the two inference flags; `block_status.json`
repeats all of it in each block record. A resume checks the metadata of every
part under out_dir, including blocks the current frame does not contain: parts
written from other requests, at another checkpoint revision, with other
inference flags, or by a runner that recorded none of these, raise ValueError
telling the caller to use a fresh output directory, so one directory can never
mix two request sets, two revisions or two inference settings. The read-only
helpers (`current_status`, `pending_counts`) apply the same checks; they check
the revision only when given one (a caller that has not resolved it yet), and
then record `checkpoint_revision: null`.
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
# the official TimesFM inference defaults, passed explicitly and recorded
USE_SYMMETRIC_AVERAGING = True
MAKE_POSITIVE = True
_FULL_SHA = re.compile(r"[0-9a-fA-F]{40}")

_PART_RE = re.compile(r"^part-(\d{6,})\.parquet$")
_META_FLOORED = b"tfm3lab.floored_values"
_META_SECONDS = b"tfm3lab.seconds"
_META_REQUESTS = b"tfm3lab.requests"
_META_REQUESTS_SHA = b"tfm3lab.requests_sha256"
_META_REVISION = b"tfm3lab.checkpoint_revision"
_META_SYMMETRIC = b"tfm3lab.use_symmetric_averaging"
_META_POSITIVE = b"tfm3lab.make_positive"


@dataclasses.dataclass(frozen=True)
class Provenance:
    """What a part must have been forecast with to belong to this run.

    `checkpoint_revision` is the resolved hub commit the forecaster was loaded
    at; None means "not asserted" and is only meaningful for the read-only
    helpers (run_requests always supplies one)."""

    requests_sha: str
    checkpoint_revision: str | None
    use_symmetric_averaging: bool = USE_SYMMETRIC_AVERAGING
    make_positive: bool = MAKE_POSITIVE

    def part_metadata(self) -> dict[bytes, bytes]:
        meta = {
            _META_REQUESTS_SHA: self.requests_sha.encode(),
            _META_SYMMETRIC: _flag_bytes(self.use_symmetric_averaging),
            _META_POSITIVE: _flag_bytes(self.make_positive),
        }
        if self.checkpoint_revision is not None:
            meta[_META_REVISION] = self.checkpoint_revision.encode()
        return meta

    def status_fields(self) -> dict:
        return {
            "requests_sha256": self.requests_sha,
            "checkpoint_revision": self.checkpoint_revision,
            "use_symmetric_averaging": self.use_symmetric_averaging,
            "make_positive": self.make_positive,
        }


def _flag_bytes(value: bool) -> bytes:
    return b"true" if value else b"false"


def _provenance(
    df: pd.DataFrame,
    checkpoint_revision: str | None,
    use_symmetric_averaging: bool,
    make_positive: bool,
) -> Provenance:
    if checkpoint_revision is not None and not str(checkpoint_revision).strip():
        raise ValueError("checkpoint_revision must be a non-empty string (the resolved revision)")
    return Provenance(
        requests_sha=requests_digest(df),
        checkpoint_revision=None if checkpoint_revision is None else str(checkpoint_revision),
        use_symmetric_averaging=bool(use_symmetric_averaging),
        make_positive=bool(make_positive),
    )


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


def _check_provenance(out_dir: Path, prov: Provenance) -> None:
    """Refuse parts that were not forecast from exactly these requests, at this
    checkpoint revision and with these inference flags. Only the parquet
    footers are read."""
    revision = prov.checkpoint_revision
    flags = (
        (_META_SYMMETRIC, "use_symmetric_averaging", _flag_bytes(prov.use_symmetric_averaging)),
        (_META_POSITIVE, "make_positive", _flag_bytes(prov.make_positive)),
    )
    for path in _existing_parts(out_dir):
        meta = pq.read_schema(path).metadata or {}
        found = meta.get(_META_REQUESTS_SHA)
        if found is None:
            raise ValueError(
                f"{path}: no requests sha256 recorded, so it cannot be tied to this requests "
                "file; use a fresh output directory (--out) instead of resuming it"
            )
        if found.decode() != prov.requests_sha:
            raise ValueError(
                f"{path}: forecast from other requests (requests sha256 {found.decode()[:12]}, "
                f"these requests {prov.requests_sha[:12]}); resuming would mix two request sets. "
                "Use a fresh output directory (--out) for this requests file"
            )
        if revision is not None:
            found = meta.get(_META_REVISION)
            if found is None:
                raise ValueError(
                    f"{path}: no checkpoint revision recorded, so it cannot be tied to "
                    f"revision {revision}; use a fresh output directory (--out) instead of "
                    "resuming it"
                )
            if found.decode() != revision:
                raise ValueError(
                    f"{path}: forecast with checkpoint revision {found.decode()}, this run uses "
                    f"{revision}; resuming would mix two checkpoint revisions. Use a fresh "
                    "output directory (--out) for this revision"
                )
        for key, name, expected in flags:
            found = meta.get(key)
            if found is None:
                raise ValueError(
                    f"{path}: no {name} recorded, so it cannot be tied to "
                    f"{name}={expected.decode()}; use a fresh output directory (--out) instead "
                    "of resuming it"
                )
            if found != expected:
                raise ValueError(
                    f"{path}: forecast with {name}={found.decode()}, this run uses "
                    f"{name}={expected.decode()}; resuming would mix two inference settings. "
                    "Use a fresh output directory (--out) for these settings"
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
    forecaster: Forecaster, chunk: pd.DataFrame, prov: Provenance
) -> tuple[pa.Table, int]:
    """One forecast_batch call -> a part table (schema metadata set, including
    the provenance) and the number of floored values. Results are
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
        use_symmetric_averaging=prov.use_symmetric_averaging,
        make_positive=prov.make_positive,
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
            **prov.part_metadata(),
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
    df: pd.DataFrame, out_dir: Path, prov: Provenance
) -> tuple[dict[int, pd.DataFrame], dict[int, BlockScan]]:
    _check_provenance(out_dir, prov)
    per_block = {int(b): g for b, g in df.groupby("block", sort=True)}
    scans = {
        b: _scan_block(block_dir(out_dir, b), _block_horizons(g)) for b, g in per_block.items()
    }
    return per_block, scans


def _status_record(n_requests: int, scan: BlockScan, prov: Provenance) -> dict:
    done = len(scan.done)
    status = "complete" if done == n_requests else "partial" if done else "pending"
    return {
        "requests": n_requests,
        "done": done,
        "floored_values": scan.floored,
        "seconds": round(scan.seconds, 3),
        "status": status,
        **prov.status_fields(),
    }


def block_status(
    per_block: dict[int, pd.DataFrame], scans: dict[int, BlockScan], prov: Provenance
) -> dict:
    return {str(b): _status_record(len(per_block[b]), scans[b], prov) for b in sorted(per_block)}


def has_forecast_parts(out_dir: str | Path) -> bool:
    """True when out_dir already holds forecast part files of any block."""
    return bool(_existing_parts(Path(out_dir)))


def current_status(
    requests_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    checkpoint_revision: str | None = None,
    use_symmetric_averaging: bool = USE_SYMMETRIC_AVERAGING,
    make_positive: bool = MAKE_POSITIVE,
) -> dict:
    """The per-block status of what is on disk (same shape as run_requests'
    return value). Read-only: nothing is written and no model is needed. The
    checkpoint revision is checked (and recorded) only when given."""
    df = validate_requests_frame(requests_df)
    prov = _provenance(df, checkpoint_revision, use_symmetric_averaging, make_positive)
    per_block, scans = _scan_all(df, Path(out_dir), prov)
    return block_status(per_block, scans, prov)


def pending_counts(
    requests_df: pd.DataFrame,
    out_dir: str | Path,
    blocks: Iterable[int] | None = None,
    *,
    checkpoint_revision: str | None = None,
    use_symmetric_averaging: bool = USE_SYMMETRIC_AVERAGING,
    make_positive: bool = MAKE_POSITIVE,
) -> dict[int, int]:
    """{block: requests without verified forecasts yet} for the selected
    blocks. Read-only: nothing is written and no model is needed. The
    checkpoint revision is checked only when given."""
    df = validate_requests_frame(requests_df)
    prov = _provenance(df, checkpoint_revision, use_symmetric_averaging, make_positive)
    out_dir = Path(out_dir)
    selected = _select_blocks(df, blocks, lambda _msg: None)
    per_block, scans = _scan_all(df, out_dir, prov)
    return {b: len(per_block[b]) - len(scans[b].done) for b in selected}


def run_requests(
    requests_df: pd.DataFrame,
    forecaster: Forecaster,
    out_dir: str | Path,
    batch_size: int = 128,
    resume: bool = True,
    blocks: Iterable[int] | None = None,
    echo: Callable[[str], None] = print,
    *,
    checkpoint_revision: str,
    use_symmetric_averaging: bool = USE_SYMMETRIC_AVERAGING,
    make_positive: bool = MAKE_POSITIVE,
) -> dict:
    """Forecast every pending request of the selected blocks (default: all)
    and return the per-block status (also kept in out_dir/block_status.json).

    `checkpoint_revision` is the resolved hub commit `forecaster` was loaded
    at; it and the two inference flags (passed to every forward pass) are
    stamped into every part and status record as provenance.

    Raises FileExistsError when `resume` is False and out_dir already holds
    parts, and ValueError when existing parts contradict the requests or were
    forecast from other requests (different sha256), at another checkpoint
    revision or with other inference flags."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if checkpoint_revision is None:
        raise ValueError("checkpoint_revision is required (the resolved revision of the model)")
    if not _FULL_SHA.fullmatch(str(checkpoint_revision).strip()):
        raise ValueError(
            f"checkpoint_revision must be a full 40-hex commit sha, got {checkpoint_revision!r}"
        )
    df = validate_requests_frame(requests_df)
    prov = _provenance(df, checkpoint_revision, use_symmetric_averaging, make_positive)
    out_dir = Path(out_dir)
    selected = _select_blocks(df, blocks, echo)
    if not resume and _existing_parts(out_dir):
        raise FileExistsError(
            f"{out_dir} already holds forecast parts; pass resume=True to continue them "
            "(never overwritten, never duplicated)"
        )
    per_block, scans = _scan_all(df, out_dir, prov)
    status_path = out_dir / STATUS_FILE
    cellstore.atomic_write_json(status_path, block_status(per_block, scans, prov))

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
                table, floored = _forecast_chunk(forecaster, chunk, prov)
                cellstore.atomic_write_table(
                    table, directory / f"part-{scan.next_part:06d}.parquet"
                )
                scan.next_part += 1
                scan.n_parts += 1
                scan.floored += floored
                scan.seconds += float(table.schema.metadata[_META_SECONDS])
                scan.done.update(chunk["request_id"])
                cellstore.atomic_write_json(status_path, block_status(per_block, scans, prov))
                echo(
                    f"block {b}: {len(scan.done)}/{len(group)} done "
                    f"(horizon {int(horizon)}, batch of {len(chunk)}"
                    f"{f', {floored} floored values' if floored else ''})"
                )
        echo(f"block {b}: {_status_record(len(group), scan, prov)['status']}")
    return block_status(per_block, scans, prov)
