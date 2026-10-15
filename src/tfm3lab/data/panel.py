"""Local per-day price panel for every Magic product, built from the cached
TCGCSV daily archives (data/raw/tcgcsv-prices-<date>.ppmd.7z).

mtg.fetch_daily_prices serves the showcase cards: it re-opens one archive per
call and keeps one row per product. The studies built on top of the full
catalog (new-set cold start, buy-timing, spike census) need the daily history
of arbitrary SKUs -- Normal and Foil are separate SKUs there -- without
re-parsing 939 archives each time. build_panel() parses every archive exactly
once into `<out_dir>/<YYYY-MM-DD>.parquet` (one row per product x subtype),
and load_panel() reads any slice back as a long frame.

Only marketPrice and midPrice are stored: lowPrice, highPrice and
directLowPrice are listing extremes, never price estimates (the same rule as
mtg._price_from_row). Which of the two is *used* is decided at load time by a
PriceSelectionPolicy, so the stored panel stays policy-free.

The builder never touches the network: a date without a cached archive is
recorded as `missing_archive`, not an error. A day is parsed entirely in
memory by one worker (one ~80 MB archive -> ~25 MB of Magic JSON), so memory
is bounded by `workers`, not by the number of dates. Each day's parquet is
written to a temp file and renamed, so a killed run leaves only complete
days behind and the next run resumes by skipping the days already written.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import py7zr
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from py7zr.io import BytesIOFactory

from .mtg import MAGIC_CATEGORY_ID, PriceSelectionPolicy, _sha256_file

PANEL_SCHEMA = pa.schema(
    [
        ("product_id", pa.int32()),
        ("group_id", pa.int32()),
        ("subtype", pa.string()),
        ("market", pa.float32()),
        ("mid", pa.float32()),
    ]
)

STATUS_WRITTEN = "written"
STATUS_SKIPPED = "skipped_existing"
STATUS_MISSING = "missing_archive"

# Metadata stamped into every day file (no prices): lets a resumed run check
# that an existing file still matches the archive on disk.
_META_DATE = b"tfm3lab.date"
_META_CATEGORY = b"tfm3lab.category_id"
_META_ARCHIVE_SHA = b"tfm3lab.archive_sha256"

# Small row groups over product_id-sorted rows let the loader's filter skip
# most of a day file via the row-group min/max statistics.
_ROW_GROUP_SIZE = 16_384
# Py7zBytesIO silently drops writes once it holds `limit` bytes. The cap is
# far above any real member (a few MB), and extract_day still compares every
# member's size with the archive header, so a short read is an error rather
# than a silently truncated day.
_MEMBER_LIMIT = 512 << 20
# A parallel build runs the dates in chunks of `workers * _CHUNK_DAYS_PER_WORKER`
# with a fresh spawn-context pool per chunk, so no worker lives for more than
# about this many days and an allocator that does not return freed pages to the
# OS cannot grow it across the ~940 days of a full build. (Not
# ProcessPoolExecutor(max_tasks_per_child=...): on CPython 3.12 it deadlocks when
# a worker is recycled while all the tasks are already queued.)
_CHUNK_DAYS_PER_WORKER = 50

_TMP_MARKER = ".tmp-"
_DAY_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.parquet$")


def archive_file(raw_dir: Path, date: dt.date) -> Path:
    """Cached archive location -- same naming as mtg.fetch_daily_prices."""
    return Path(raw_dir) / f"tcgcsv-prices-{date.isoformat()}.ppmd.7z"


def panel_path(out_dir: Path, date: dt.date) -> Path:
    return Path(out_dir) / f"{date.isoformat()}.parquet"


# --- one archive -> one table -------------------------------------------------


def extract_day(
    archive_path: Path, date: dt.date, category_id: int = MAGIC_CATEGORY_ID
) -> pa.Table:
    """Reads ONLY the `<date>/<category_id>/<group>/prices` members of one
    archive (the other categories are decompressed past but never kept) and
    returns every row as a PANEL_SCHEMA table sorted by (product_id, subtype).

    Every row is kept: a product's Normal and Foil rows (any subTypeName,
    verbatim, None allowed) are separate SKUs. A duplicate (product_id,
    subtype) within the day raises ValueError naming the product, subtype
    and date -- no silent last-row-wins. lowPrice, highPrice and
    directLowPrice are never read into the table.
    """
    member_re = re.compile(rf"^{re.escape(date.isoformat())}/{category_id}/(\d+)/prices$")
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        members: list[tuple[int, str, int]] = []
        for info in z.list():
            match = member_re.match(info.filename)
            if match and not info.is_directory:
                members.append((int(match.group(1)), info.filename, info.uncompressed))
        members.sort()
        factory = BytesIOFactory(limit=_MEMBER_LIMIT)
        if members:
            z.extract(targets=[name for _, name, _ in members], factory=factory)

        columns: dict[str, list] = {name: [] for name in PANEL_SCHEMA.names}
        seen: set[tuple[int, str | None]] = set()
        for group_id, name, declared_size in members:
            buf = factory.get(name)
            buf.seek(0)
            raw = buf.read()
            if len(raw) != declared_size:
                raise ValueError(
                    f"{name} on {date.isoformat()}: extracted {len(raw)} bytes, archive header "
                    f"declares {declared_size}"
                )
            payload = json.loads(raw)
            for row in payload.get("results", []):
                try:
                    product_id = int(row["productId"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{name} on {date.isoformat()}: row without productId"
                    ) from exc
                subtype = row.get("subTypeName")
                if (product_id, subtype) in seen:
                    raise ValueError(
                        f"duplicate (product_id={product_id}, subtype={subtype!r}) on "
                        f"{date.isoformat()} (second occurrence in {name})"
                    )
                seen.add((product_id, subtype))
                columns["product_id"].append(product_id)
                columns["group_id"].append(group_id)
                columns["subtype"].append(subtype)
                columns["market"].append(row.get("marketPrice"))
                columns["mid"].append(row.get("midPrice"))
            factory.products.pop(name)  # release the member once parsed

    table = pa.table(
        {f.name: pa.array(columns[f.name], type=f.type) for f in PANEL_SCHEMA},
        schema=PANEL_SCHEMA,
    )
    return table.sort_by([("product_id", "ascending"), ("subtype", "ascending")])


# --- one day -> one parquet file ------------------------------------------------


def _write_parquet_atomic(table: pa.Table, path: Path) -> tuple[str, int]:
    """Writes `table` to a temp file beside `path`, fsyncs, hashes it and
    renames it into place; the temp file is removed on any failure, so `path`
    is either absent or complete. Returns (output sha256, row count)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}{_TMP_MARKER}{os.getpid()}")
    try:
        with tmp.open("wb") as fh:
            pq.write_table(table, fh, compression="zstd", row_group_size=_ROW_GROUP_SIZE)
            fh.flush()
            os.fsync(fh.fileno())
        digest = _sha256_file(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return digest, table.num_rows


def _validated_existing(
    out_path: Path, date: dt.date, category_id: int, archive_sha: str | None
) -> tuple[str, int]:
    """Checks that an existing day file can be resumed from and returns its
    (recorded archive sha256, row count). It counts as valid only if its footer
    is readable, its schema is PANEL_SCHEMA, its stamped date (and category) are
    this day's, and its stamped archive sha256 is present and -- when the archive
    is still cached -- equal to the cached archive's. Anything else raises a
    RuntimeError asking for --force; nothing is rebuilt silently."""

    def refuse(reason: str) -> RuntimeError:
        return RuntimeError(
            f"{out_path} cannot be resumed for {date.isoformat()}: {reason}; "
            "re-run with --force to rebuild it"
        )

    try:
        meta = pq.read_metadata(out_path)
        schema = pq.read_schema(out_path)
        stored = {
            key: (meta.metadata or {}).get(key)
            for key in (_META_DATE, _META_CATEGORY, _META_ARCHIVE_SHA)
        }
        stamped = {k: v.decode() if v is not None else None for k, v in stored.items()}
    except (pa.ArrowException, OSError, ValueError) as exc:
        raise refuse(f"unreadable parquet footer ({type(exc).__name__}: {exc})") from exc

    if not schema.remove_metadata().equals(PANEL_SCHEMA):
        raise refuse("its schema differs from the panel schema")
    if stamped[_META_DATE] is None:
        raise refuse("it carries no date metadata")
    if stamped[_META_DATE] != date.isoformat():
        raise refuse(f"it is stamped for date {stamped[_META_DATE]!r}")
    if stamped[_META_CATEGORY] not in (None, str(category_id)):
        raise refuse(f"it is stamped for category {stamped[_META_CATEGORY]!r}")
    recorded = stamped[_META_ARCHIVE_SHA]
    if not recorded:
        raise refuse("it carries no archive_sha256 metadata")
    if archive_sha is not None and recorded != archive_sha:
        raise refuse("it was built from a different archive than the cached one")
    return recorded, meta.num_rows


def build_day(
    raw_dir: Path,
    out_dir: Path,
    date: dt.date,
    *,
    force: bool = False,
    category_id: int = MAGIC_CATEGORY_ID,
) -> dict:
    """Builds (or skips, or reports missing) one day. Returns the per-date
    record {date, archive_sha256, output_sha256, rows, status}; never any
    price. Runs inside a pool worker, so everything it holds is one day.

    An existing day file is skipped only if _validated_existing accepts it
    (else RuntimeError, --force rebuilds); an archive without a single row for
    the category raises ValueError instead of writing an empty day.
    """
    out_path = panel_path(out_dir, date)
    archive = archive_file(raw_dir, date)
    archive_sha = _sha256_file(archive) if archive.exists() else None

    if out_path.exists() and not force:
        recorded, rows = _validated_existing(out_path, date, category_id, archive_sha)
        return {
            "date": date.isoformat(),
            "archive_sha256": archive_sha or recorded,
            "output_sha256": _sha256_file(out_path),
            "rows": rows,
            "status": STATUS_SKIPPED,
        }

    if archive_sha is None:
        return {
            "date": date.isoformat(),
            "archive_sha256": None,
            "output_sha256": None,
            "rows": None,
            "status": STATUS_MISSING,
        }

    table = extract_day(archive, date, category_id)
    if table.num_rows == 0:
        # Never persist an empty day: a 0-row file would be skipped as "built"
        # on every resume. Nothing is written, so a rerun re-checks the archive.
        raise ValueError(
            f"{archive.name}: no category {category_id} price rows for {date.isoformat()}; "
            "refusing to write an empty day file"
        )
    table = table.replace_schema_metadata(
        {
            _META_DATE: date.isoformat().encode(),
            _META_CATEGORY: str(category_id).encode(),
            _META_ARCHIVE_SHA: archive_sha.encode(),
        }
    )
    output_sha, rows = _write_parquet_atomic(table, out_path)
    return {
        "date": date.isoformat(),
        "archive_sha256": archive_sha,
        "output_sha256": output_sha,
        "rows": rows,
        "status": STATUS_WRITTEN,
    }


def build_panel(
    raw_dir: Path,
    out_dir: Path,
    start: dt.date,
    end: dt.date,
    workers: int = 1,
    force: bool = False,
    progress: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Builds `<out_dir>/<date>.parquet` for every date in [start, end]
    (inclusive) from the archives cached in `raw_dir`; no network access.

    Existing day files are skipped unless `force` (resume). A date without a
    cached archive is recorded as `missing_archive`. With `workers` > 1 the
    dates are processed in chunks of `workers * 50`, each chunk by its own
    spawn-context process pool (one day per worker at a time), so no worker
    outlives its chunk; `workers` == 1 runs in this process. A failure (e.g. a
    duplicate SKU in an archive) cancels the rest of its chunk, skips the later
    chunks and propagates; days already written stay valid.
    Not safe for two concurrent builds into the same `out_dir`.

    Returns the per-date records, ordered by date: {date, archive_sha256,
    output_sha256, rows, status}, status in {written, skipped_existing,
    missing_archive}. `progress` is called with each record as it finishes.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if end < start:
        raise ValueError(f"end {end.isoformat()} is before start {start.isoformat()}")
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"*.parquet{_TMP_MARKER}*"):  # left by a killed run
        stale.unlink(missing_ok=True)

    dates = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    records: list[dict] = []
    if workers == 1:
        for date in dates:
            records.append(build_day(raw_dir, out_dir, date, force=force))
            if progress:
                progress(records[-1])
    else:
        chunk = workers * _CHUNK_DAYS_PER_WORKER
        for lo in range(0, len(dates), chunk):
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
                futures = [
                    pool.submit(build_day, raw_dir, out_dir, d, force=force)
                    for d in dates[lo : lo + chunk]
                ]
                try:
                    for future in as_completed(futures):
                        records.append(future.result())
                        if progress:
                            progress(records[-1])
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
    return sorted(records, key=lambda r: r["date"])


# --- loading ------------------------------------------------------------------


def _as_date(value: dt.date | str | pd.Timestamp) -> dt.date:
    return pd.Timestamp(value).date()


def _wanted_by_subtype(keys: Iterable[tuple[int, str | None]]) -> dict[str | None, np.ndarray]:
    grouped: dict[str | None, set[int]] = defaultdict(set)
    for product_id, subtype in keys:
        grouped[None if pd.isna(subtype) else str(subtype)].add(int(product_id))
    return {st: np.fromiter(ids, dtype=np.int64, count=len(ids)) for st, ids in grouped.items()}


def _select_price(
    market: np.ndarray, mid: np.ndarray, policy: PriceSelectionPolicy
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised mtg._price_from_row: returns (price, field) with
    field == '' where the policy yields no price."""
    has_market, has_mid = ~np.isnan(market), ~np.isnan(mid)
    if policy is PriceSelectionPolicy.MARKET_ONLY:
        use_market, use_mid = has_market, np.zeros_like(has_mid)
    elif policy is PriceSelectionPolicy.MID_ONLY:
        use_market, use_mid = np.zeros_like(has_market), has_mid
    else:
        use_market, use_mid = has_market, ~has_market & has_mid
    price = np.where(use_market, market, np.where(use_mid, mid, np.nan))
    field = np.where(use_market, "market", np.where(use_mid, "mid", ""))
    return price, field


def _empty_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series([], dtype="datetime64[ns]"),
            "product_id": pd.Series([], dtype="int32"),
            "subtype": pd.Series([], dtype="str"),
            "price": pd.Series([], dtype="float64"),
            "field": pd.Series([], dtype="str"),
        }
    )


def load_panel(
    out_dir: Path,
    keys: Iterable[tuple[int, str | None]] | None = None,
    start: dt.date | str | pd.Timestamp | None = None,
    end: dt.date | str | pd.Timestamp | None = None,
    policy: PriceSelectionPolicy = PriceSelectionPolicy.MARKET_THEN_MID,
) -> pd.DataFrame:
    """Reads the day files of `out_dir` (optionally within [start, end],
    inclusive) into one long frame (date, product_id, subtype, price, field),
    ordered by date then (product_id, subtype).

    `price` is resolved per row by `policy` -- by default the project's
    MARKET_THEN_MID: marketPrice when not null (field 'market'), else
    midPrice (field 'mid'); rows with neither are dropped. Prices carry the
    float32 precision they are stored at. `date` is datetime64[ns] at
    midnight; days without a file are simply absent.

    `keys` -- an iterable of (product_id, subtype) pairs, subtype None
    matching a null subtype -- restricts the result to exactly those SKUs:
    the product_id filter is pushed into the parquet read (row-group
    statistics skip most of each file), then the pair is matched exactly.
    keys=None loads every SKU of every day (~100k rows per day): pass keys
    or a narrow date range unless that is really what you want.
    """
    wanted = None if keys is None else _wanted_by_subtype(list(keys))
    lo = _as_date(start) if start is not None else None
    hi = _as_date(end) if end is not None else None

    files: list[tuple[dt.date, Path]] = []
    for path in sorted(Path(out_dir).iterdir()):
        match = _DAY_FILE.match(path.name)
        if not match:
            continue
        date = dt.date.fromisoformat(match.group(1))
        if (lo is None or date >= lo) and (hi is None or date <= hi):
            files.append((date, path))

    read_filter = None
    if wanted is not None:
        ids = np.unique(np.concatenate([*wanted.values(), np.empty(0, dtype=np.int64)]))
        ids = ids[(ids >= np.iinfo(np.int32).min) & (ids <= np.iinfo(np.int32).max)]
        read_filter = pc.field("product_id").isin(pa.array(ids, type=pa.int32()))

    frames: list[pd.DataFrame] = []
    frame_dates: list[dt.date] = []
    for date, path in files:
        table = pq.read_table(
            path, columns=["product_id", "subtype", "market", "mid"], filters=read_filter
        )
        if table.num_rows == 0:
            continue
        df = table.to_pandas()
        if wanted is not None:
            mask = np.zeros(len(df), dtype=bool)
            for subtype, ids_for_subtype in wanted.items():
                same_subtype = df["subtype"].isna() if subtype is None else df["subtype"] == subtype
                mask |= same_subtype.to_numpy(dtype=bool) & df["product_id"].isin(
                    ids_for_subtype
                ).to_numpy(dtype=bool)
            df = df[mask]
        price, field = _select_price(
            df["market"].to_numpy(dtype=np.float64),
            df["mid"].to_numpy(dtype=np.float64),
            policy,
        )
        keep = field != ""
        if not keep.any():
            continue
        frames.append(
            pd.DataFrame(
                {
                    "product_id": df["product_id"].to_numpy()[keep],
                    "subtype": df["subtype"].to_numpy()[keep],
                    "price": price[keep],
                    "field": field[keep],
                }
            )
        )
        frame_dates.append(date)

    if not frames:
        return _empty_panel()
    out = pd.concat(frames, ignore_index=True)
    day_values = np.array([np.datetime64(d, "ns") for d in frame_dates], dtype="datetime64[ns]")
    out.insert(0, "date", np.repeat(day_values, [len(f) for f in frames]))
    out["subtype"] = out["subtype"].astype("str")
    out["field"] = out["field"].astype("str")
    return out
