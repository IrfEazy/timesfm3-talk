"""Offline tests for the per-day price panel (src/tfm3lab/data/panel.py) and
its CLI (scripts/12_build_panel.py): tiny synthetic 7z archives built in
tmp_path, no network, no real data.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import signal
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tfm3lab.data import panel
from tfm3lab.data.mtg import PriceSelectionPolicy, _price_from_row

D1, D2, D3 = dt.date(2024, 2, 8), dt.date(2024, 2, 9), dt.date(2024, 2, 10)
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "12_build_panel.py"


def row(product_id, subtype="Normal", market=1.0, mid=2.0):
    """One TCGCSV price row; low/high/directLow carry a sentinel that must
    never reach the panel."""
    return {
        "productId": product_id,
        "lowPrice": 777.0,
        "midPrice": mid,
        "highPrice": 888.0,
        "marketPrice": market,
        "directLowPrice": 999.0,
        "subTypeName": subtype,
    }


def make_archive(raw_dir: Path, date: dt.date, groups: dict[tuple[int, int], list[dict]]) -> Path:
    """groups maps (category_id, group_id) -> rows of that group's prices file."""
    import py7zr

    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"tcgcsv-prices-{date.isoformat()}.ppmd.7z"
    with py7zr.SevenZipFile(path, mode="w") as z:
        for (category_id, group_id), rows in groups.items():
            payload = json.dumps({"success": True, "errors": [], "results": rows})
            z.writestr(payload, f"{date.isoformat()}/{category_id}/{group_id}/prices")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def raw_dir(tmp_path):
    return tmp_path / "raw"


@pytest.fixture
def out_dir(tmp_path):
    return tmp_path / "panel"


def standard_day(date: dt.date, raw_dir: Path) -> Path:
    return make_archive(
        raw_dir,
        date,
        {
            (1, 100): [row(1, "Normal", 1.5, 1.4), row(1, "Foil", 4.0, 3.9)],
            (1, 101): [row(2, "Normal", None, 0.25), row(3, None, 10.0, None)],
            (2, 200): [row(50, "Normal", 5.0, 5.0)],  # another category, never read
        },
    )


# --- extract_day ---------------------------------------------------------------


def test_extract_day_reads_only_the_requested_category(raw_dir):
    archive = standard_day(D1, raw_dir)
    table = panel.extract_day(archive, D1)
    assert table.schema == panel.PANEL_SCHEMA
    assert table.column("product_id").to_pylist() == [1, 1, 2, 3]
    assert table.column("group_id").to_pylist() == [100, 100, 101, 101]
    other = panel.extract_day(archive, D1, category_id=2)
    assert other.column("product_id").to_pylist() == [50]
    assert other.column("group_id").to_pylist() == [200]


def test_extract_day_keeps_every_subtype_and_never_stores_low_high_direct(raw_dir):
    archive = make_archive(
        raw_dir,
        D1,
        {
            (1, 100): [
                row(1, "Foil", 4.0, 3.5),
                row(1, "Normal", 1.5, 1.25),
                row(1, "1st Edition Holofoil", 9.0, 8.0),
                row(4, None, None, None),
            ]
        },
    )
    table = panel.extract_day(archive, D1)
    # sorted by (product_id, subtype); subtypes verbatim; null rows kept
    assert table.column("subtype").to_pylist() == [
        "1st Edition Holofoil",
        "Foil",
        "Normal",
        None,
    ]
    assert table.column("market").to_pylist() == pytest.approx([9.0, 4.0, 1.5, None], nan_ok=True)
    assert table.column("mid").to_pylist() == pytest.approx([8.0, 3.5, 1.25, None], nan_ok=True)
    assert set(table.column_names) == {"product_id", "group_id", "subtype", "market", "mid"}
    for sentinel in (777.0, 888.0, 999.0):
        assert sentinel not in table.column("market").to_pylist()
        assert sentinel not in table.column("mid").to_pylist()


def test_extract_day_duplicate_within_a_group_raises_naming_product_and_date(raw_dir):
    archive = make_archive(raw_dir, D1, {(1, 100): [row(7, "Normal"), row(7, "Normal", 3.0)]})
    with pytest.raises(ValueError, match=r"product_id=7.*Normal.*2024-02-08"):
        panel.extract_day(archive, D1)


def test_extract_day_duplicate_across_groups_raises(raw_dir):
    archive = make_archive(
        raw_dir, D1, {(1, 100): [row(7, None)], (1, 101): [row(8, "Foil"), row(7, None)]}
    )
    with pytest.raises(ValueError, match=r"product_id=7.*None.*2024-02-08"):
        panel.extract_day(archive, D1)


def test_extract_day_same_product_different_subtypes_is_not_a_duplicate(raw_dir):
    archive = make_archive(raw_dir, D1, {(1, 100): [row(7, "Normal"), row(7, "Foil")]})
    assert panel.extract_day(archive, D1).num_rows == 2


def test_extract_day_without_the_category_is_an_empty_table(raw_dir):
    archive = make_archive(raw_dir, D1, {(2, 200): [row(50)]})
    table = panel.extract_day(archive, D1)
    assert table.num_rows == 0 and table.schema == panel.PANEL_SCHEMA


def test_extract_day_detects_a_truncated_member(raw_dir, monkeypatch):
    """py7zr's in-memory writer silently drops bytes past its limit; extract_day must
    compare each member with the size declared in the archive header."""
    archive = make_archive(raw_dir, D1, {(1, 100): [row(1), row(2)]})

    class TruncatingFactory(panel.BytesIOFactory):
        def create(self, filename):
            product = super().create(filename)
            write = product.write
            product.write = lambda data: write(data[:-1])
            return product

    monkeypatch.setattr(panel, "BytesIOFactory", TruncatingFactory)
    with pytest.raises(ValueError, match="archive header"):
        panel.extract_day(archive, D1)


# --- build_panel -----------------------------------------------------------------


def test_build_panel_writes_one_file_per_date_and_records_missing(raw_dir, out_dir):
    a1 = standard_day(D1, raw_dir)
    a3 = standard_day(D3, raw_dir)
    records = panel.build_panel(raw_dir, out_dir, D1, D3, workers=1)

    assert [r["date"] for r in records] == ["2024-02-08", "2024-02-09", "2024-02-10"]
    assert [r["status"] for r in records] == ["written", "missing_archive", "written"]
    assert sorted(p.name for p in out_dir.iterdir()) == ["2024-02-08.parquet", "2024-02-10.parquet"]
    first, missing, third = records
    assert first["archive_sha256"] == sha256(a1) and third["archive_sha256"] == sha256(a3)
    assert first["output_sha256"] == sha256(out_dir / "2024-02-08.parquet")
    assert first["rows"] == 4 == pq.read_metadata(out_dir / "2024-02-08.parquet").num_rows
    assert missing == {
        "date": "2024-02-09",
        "archive_sha256": None,
        "output_sha256": None,
        "rows": None,
        "status": "missing_archive",
    }
    schema = pq.read_schema(out_dir / "2024-02-08.parquet")
    assert [f.name for f in schema] == panel.PANEL_SCHEMA.names
    assert schema.metadata[b"tfm3lab.archive_sha256"].decode() == sha256(a1)


def test_build_panel_resumes_by_skipping_existing_days_and_force_rewrites(raw_dir, out_dir):
    standard_day(D1, raw_dir)
    standard_day(D2, raw_dir)
    first = panel.build_panel(raw_dir, out_dir, D1, D2, workers=1)
    before = {p.name: p.stat() for p in out_dir.iterdir()}

    again = panel.build_panel(raw_dir, out_dir, D1, D2, workers=1)
    assert [r["status"] for r in again] == ["skipped_existing"] * 2
    for a, b in zip(first, again, strict=True):
        assert (a["archive_sha256"], a["output_sha256"], a["rows"]) == (
            b["archive_sha256"],
            b["output_sha256"],
            b["rows"],
        )
    after = {p.name: p.stat() for p in out_dir.iterdir()}
    assert {n: (s.st_ino, s.st_mtime_ns) for n, s in before.items()} == {
        n: (s.st_ino, s.st_mtime_ns) for n, s in after.items()
    }

    forced = panel.build_panel(raw_dir, out_dir, D1, D2, workers=1, force=True)
    assert [r["status"] for r in forced] == ["written"] * 2
    assert [r["output_sha256"] for r in forced] == [r["output_sha256"] for r in first]
    assert all(p.stat().st_ino != before[p.name].st_ino for p in out_dir.iterdir())


def test_build_panel_skip_keeps_the_recorded_archive_when_the_archive_is_gone(raw_dir, out_dir):
    archive = standard_day(D1, raw_dir)
    built = panel.build_panel(raw_dir, out_dir, D1, D1)[0]
    archive.unlink()
    (skipped,) = panel.build_panel(raw_dir, out_dir, D1, D1)
    assert skipped["status"] == "skipped_existing"
    assert skipped["archive_sha256"] == built["archive_sha256"]


def test_build_panel_refuses_to_skip_an_output_built_from_a_different_archive(raw_dir, out_dir):
    standard_day(D1, raw_dir)
    panel.build_panel(raw_dir, out_dir, D1, D1)
    make_archive(raw_dir, D1, {(1, 100): [row(99)]})  # same date, different bytes
    with pytest.raises(RuntimeError, match="--force"):
        panel.build_panel(raw_dir, out_dir, D1, D1)
    (rebuilt,) = panel.build_panel(raw_dir, out_dir, D1, D1, force=True)
    assert rebuilt["status"] == "written" and rebuilt["rows"] == 1


def restamp(path: Path, **overrides) -> None:
    """Rewrites a day file in place with some tfm3lab.* metadata replaced (None
    drops the key), keeping the data and schema."""
    table = pq.read_table(path)
    meta = dict(table.schema.metadata or {})
    for key, value in overrides.items():
        name = f"tfm3lab.{key}".encode()
        if value is None:
            meta.pop(name, None)
        else:
            meta[name] = value.encode()
    pq.write_table(table.replace_schema_metadata(meta), path)


def _drop_all_metadata(path):
    pq.write_table(pq.read_table(path).replace_schema_metadata(None), path)


def _wrong_schema(path):
    table = pq.read_table(path)  # keeps the tfm3lab.* stamps
    pq.write_table(table.set_column(3, "market", table.column("market").cast(pa.float64())), path)


def _extra_column(path):
    table = pq.read_table(path)
    pq.write_table(table.append_column("low", table.column("market")), path)


def _garbage(path):
    path.write_bytes(b"this is not a parquet file")


def _truncated(path):
    path.write_bytes(path.read_bytes()[:-12])


def _empty_file(path):
    path.write_bytes(b"")


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(_drop_all_metadata, id="no-metadata"),
        pytest.param(lambda p: restamp(p, archive_sha256=None), id="no-archive-sha"),
        pytest.param(lambda p: restamp(p, archive_sha256=""), id="blank-archive-sha"),
        pytest.param(lambda p: restamp(p, archive_sha256="0" * 64), id="other-archive-sha"),
        pytest.param(lambda p: restamp(p, date=None), id="no-date"),
        pytest.param(lambda p: restamp(p, date="2024-02-09"), id="other-date"),
        pytest.param(lambda p: restamp(p, category_id="2"), id="other-category"),
        pytest.param(_wrong_schema, id="schema-type"),
        pytest.param(_extra_column, id="schema-extra-column"),
        pytest.param(_garbage, id="unreadable-footer"),
        pytest.param(_truncated, id="truncated-footer"),
        pytest.param(_empty_file, id="empty-file"),
    ],
)
def test_resume_refuses_an_existing_file_that_is_not_provably_this_days_build(
    raw_dir, out_dir, damage
):
    standard_day(D1, raw_dir)
    panel.build_panel(raw_dir, out_dir, D1, D1)
    path = out_dir / "2024-02-08.parquet"
    damage(path)
    damaged = path.read_bytes()

    for kwargs in ({}, {"workers": 2}):
        with pytest.raises(RuntimeError, match=r"--force"):
            panel.build_panel(raw_dir, out_dir, D1, D1, **kwargs)
        assert path.read_bytes() == damaged  # refused, not touched
    with pytest.raises(RuntimeError, match=r"2024-02-08\.parquet"):
        panel.build_day(raw_dir, out_dir, D1)

    (rebuilt,) = panel.build_panel(raw_dir, out_dir, D1, D1, force=True)
    assert rebuilt["status"] == "written" and rebuilt["rows"] == 4
    (skipped,) = panel.build_panel(raw_dir, out_dir, D1, D1)
    assert skipped["status"] == "skipped_existing"
    assert skipped["output_sha256"] == rebuilt["output_sha256"]


def test_resume_without_metadata_is_refused_even_when_the_archive_is_gone(raw_dir, out_dir):
    archive = standard_day(D1, raw_dir)
    panel.build_panel(raw_dir, out_dir, D1, D1)
    _drop_all_metadata(out_dir / "2024-02-08.parquet")
    archive.unlink()
    with pytest.raises(RuntimeError, match="--force"):
        panel.build_panel(raw_dir, out_dir, D1, D1)


def test_resume_accepts_a_file_without_a_stored_category(raw_dir, out_dir):
    """The category stamp is checked only when present."""
    standard_day(D1, raw_dir)
    (built,) = panel.build_panel(raw_dir, out_dir, D1, D1)
    restamp(out_dir / "2024-02-08.parquet", category_id=None)
    (skipped,) = panel.build_panel(raw_dir, out_dir, D1, D1)
    assert skipped["status"] == "skipped_existing"
    assert skipped["archive_sha256"] == built["archive_sha256"] and skipped["rows"] == 4


def test_build_panel_makes_no_network_request(raw_dir, out_dir, monkeypatch):
    import requests

    def boom(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(requests.Session, "request", boom)
    standard_day(D1, raw_dir)
    records = panel.build_panel(raw_dir, out_dir, D1, D2, workers=1)
    assert [r["status"] for r in records] == ["written", "missing_archive"]


def test_build_panel_process_pool_matches_the_in_process_build(raw_dir, tmp_path):
    for date in (D1, D2, D3):
        standard_day(date, raw_dir)
    serial = panel.build_panel(raw_dir, tmp_path / "serial", D1, D3, workers=1)
    pooled = panel.build_panel(raw_dir, tmp_path / "pooled", D1, D3, workers=2)
    assert pooled == serial
    assert [r["status"] for r in pooled] == ["written"] * 3


# build_panel runs in a child interpreter so that a pool deadlock fails the test after
# a hard timeout instead of hanging it (a stuck ProcessPoolExecutor also blocks
# interpreter exit, so an in-process thread/timeout cannot recover). The child
# reports the records, the error (if any) and what each pool it created was given.
_ISOLATED_BUILD = """
import datetime as dt, json, sys
from pathlib import Path
from tfm3lab.data import panel

raw, out, start, end, workers, per_worker = sys.argv[1:7]
if per_worker != "default":
    panel._CHUNK_DAYS_PER_WORKER = int(per_worker)
pools = []

class SpyPool(panel.ProcessPoolExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs, self.submitted = kwargs, 0
        pools.append(self)

    def submit(self, *args, **kwargs):
        self.submitted += 1
        return super().submit(*args, **kwargs)

panel.ProcessPoolExecutor = SpyPool
outcome, progress = {}, []
try:
    outcome["records"] = panel.build_panel(
        Path(raw), Path(out), dt.date.fromisoformat(start), dt.date.fromisoformat(end),
        workers=int(workers), progress=progress.append,
    )
except Exception as exc:
    outcome["error"] = f"{type(exc).__name__}: {exc}"
outcome["progress_dates"] = [r["date"] for r in progress]
outcome["pools"] = [
    {
        "submitted": pool.submitted,
        "max_workers": pool.kwargs["max_workers"],
        "start_method": pool.kwargs["mp_context"].get_start_method(),
        "kwargs": sorted(pool.kwargs),
    }
    for pool in pools
]
print(json.dumps(outcome))
"""


def isolated_build(raw_dir, out_dir, start, end, *, workers, days_per_worker="default", timeout=90):
    """Runs build_panel(workers=...) in a fresh interpreter (the module's
    chunk size patched to `days_per_worker` days per worker) and returns its
    outcome dict. After `timeout` seconds the whole process group is killed
    and the test fails."""
    cmd = [sys.executable, "-c", _ISOLATED_BUILD]
    cmd += [str(raw_dir), str(out_dir), start.isoformat(), end.isoformat()]
    cmd += [str(workers), str(days_per_worker)]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)  # the child and its pool workers
        proc.communicate()
        pytest.fail(f"build_panel did not finish within {timeout}s (process pool deadlock?)")
    assert proc.returncode == 0, stderr
    return json.loads(stdout)


def test_build_panel_runs_a_multi_chunk_pool_build_to_completion(raw_dir, tmp_path):
    """2 workers x 2 days per worker = chunks of 4 dates: 11 dates are 3 chunks,
    each with its own fresh spawn pool. Must finish (not deadlock) and match the
    in-process build exactly."""
    days = [D1 + dt.timedelta(days=i) for i in range(11)]
    for date in days:
        if date != days[6]:  # one missing archive in the middle of the second chunk
            standard_day(date, raw_dir)
    serial = panel.build_panel(raw_dir, tmp_path / "serial", days[0], days[-1], workers=1)

    pooled_dir = tmp_path / "pooled"
    outcome = isolated_build(raw_dir, pooled_dir, days[0], days[-1], workers=2, days_per_worker=2)

    assert "error" not in outcome
    pooled = outcome["records"]
    assert pooled == serial
    assert [r["date"] for r in pooled] == [d.isoformat() for d in days]
    assert [r["status"] for r in pooled] == ["written"] * 6 + ["missing_archive"] + ["written"] * 4
    assert sorted(outcome["progress_dates"]) == [d.isoformat() for d in days]  # one per date
    assert sorted(p.name for p in pooled_dir.iterdir()) == [
        f"{d.isoformat()}.parquet" for d in days if d != days[6]
    ]
    assert all(pq.read_metadata(p).num_rows == 4 for p in pooled_dir.iterdir())

    # three fresh spawn pools of 2 workers given 4 + 4 + 3 tasks, none recycling workers
    assert [pool["submitted"] for pool in outcome["pools"]] == [4, 4, 3]
    for pool in outcome["pools"]:
        assert pool["max_workers"] == 2 and pool["start_method"] == "spawn"
        assert pool["kwargs"] == ["max_workers", "mp_context"]


def test_build_panel_default_chunk_is_50_days_per_worker(raw_dir, tmp_path):
    assert panel._CHUNK_DAYS_PER_WORKER == 50
    for date in (D1, D2, D3):
        standard_day(date, raw_dir)
    outcome = isolated_build(raw_dir, tmp_path / "pooled", D1, D3, workers=2)
    assert [r["status"] for r in outcome["records"]] == ["written"] * 3
    assert [pool["submitted"] for pool in outcome["pools"]] == [3]  # 3 dates < 2 x 50: one pool


def test_a_failing_day_stops_a_two_worker_build_with_the_error_and_no_partial_file(
    raw_dir, out_dir
):
    standard_day(D1, raw_dir)
    make_archive(raw_dir, D2, {(1, 100): [row(7, "Normal"), row(7, "Normal")]})  # duplicate
    standard_day(D3, raw_dir)
    with pytest.raises(ValueError, match=r"duplicate .*product_id=7.*2024-02-09"):
        panel.build_panel(raw_dir, out_dir, D1, D3, workers=2)
    names = sorted(p.name for p in out_dir.iterdir())
    assert "2024-02-09.parquet" not in names  # the failed day was never written
    assert not [n for n in names if panel._TMP_MARKER in n]  # no temp file left behind
    assert set(names) <= {"2024-02-08.parquet", "2024-02-10.parquet"}  # only complete days
    for name in names:
        assert pq.read_metadata(out_dir / name).num_rows == 4  # and each of them is readable


def test_a_failure_in_the_second_chunk_stops_the_build_before_the_third(raw_dir, out_dir):
    days = [D1 + dt.timedelta(days=i) for i in range(9)]  # chunks of 4: days 0-3, 4-7, 8
    for date in days:
        standard_day(date, raw_dir)
    bad = days[5]  # second chunk
    make_archive(raw_dir, bad, {(1, 100): [row(7, "Normal"), row(7, "Normal")]})
    outcome = isolated_build(raw_dir, out_dir, days[0], days[-1], workers=2, days_per_worker=2)

    assert "records" not in outcome
    assert re.search(rf"ValueError: duplicate .*product_id=7.*{bad.isoformat()}", outcome["error"])
    assert len(outcome["pools"]) == 2  # the third chunk never got a pool
    names = {p.name for p in out_dir.iterdir()}
    assert not [n for n in names if panel._TMP_MARKER in n]  # no temp file left behind
    assert {f"{d.isoformat()}.parquet" for d in days[:4]} <= names  # first chunk complete
    assert f"{bad.isoformat()}.parquet" not in names  # the failed day was never written
    assert f"{days[8].isoformat()}.parquet" not in names  # the third chunk never started
    assert all(pq.read_metadata(out_dir / n).num_rows == 4 for n in names)


def test_an_empty_archive_stops_a_two_worker_build_without_a_file_for_that_day(raw_dir, out_dir):
    standard_day(D1, raw_dir)
    make_archive(raw_dir, D2, {(2, 200): [row(50)]})
    with pytest.raises(ValueError, match="no category 1 price rows"):
        panel.build_panel(raw_dir, out_dir, D1, D2, workers=2)
    assert not (out_dir / "2024-02-09.parquet").exists()
    assert not [p for p in out_dir.iterdir() if panel._TMP_MARKER in p.name]


def test_build_day_creates_the_output_directory(raw_dir, out_dir):
    standard_day(D1, raw_dir)
    record = panel.build_day(raw_dir, out_dir / "nested" / "panel", D1)
    assert record["status"] == "written"
    assert (out_dir / "nested" / "panel" / "2024-02-08.parquet").exists()


def test_build_panel_reports_progress_per_date(raw_dir, out_dir):
    standard_day(D1, raw_dir)
    seen = []
    panel.build_panel(raw_dir, out_dir, D1, D2, progress=seen.append)
    assert [r["date"] for r in seen] == ["2024-02-08", "2024-02-09"]


def test_build_panel_rejects_bad_arguments(raw_dir, out_dir):
    with pytest.raises(ValueError, match="workers"):
        panel.build_panel(raw_dir, out_dir, D1, D1, workers=0)
    with pytest.raises(ValueError, match="before start"):
        panel.build_panel(raw_dir, out_dir, D2, D1)


def test_atomic_write_leaves_no_partial_file_when_the_write_fails(raw_dir, out_dir, monkeypatch):
    standard_day(D1, raw_dir)

    def failing_write(table, where, **kwargs):
        where.write(b"PAR1 half a parquet fi")
        raise OSError("disk full")

    monkeypatch.setattr(panel.pq, "write_table", failing_write)
    with pytest.raises(OSError, match="disk full"):
        panel.build_panel(raw_dir, out_dir, D1, D1)
    assert list(out_dir.iterdir()) == []


def test_failed_forced_rewrite_keeps_the_previous_complete_file(raw_dir, out_dir, monkeypatch):
    standard_day(D1, raw_dir)
    panel.build_panel(raw_dir, out_dir, D1, D1)
    path = out_dir / "2024-02-08.parquet"
    original = path.read_bytes()

    def failing_write(table, where, **kwargs):
        where.write(b"junk")
        raise OSError("disk full")

    monkeypatch.setattr(panel.pq, "write_table", failing_write)
    with pytest.raises(OSError):
        panel.build_panel(raw_dir, out_dir, D1, D1, force=True)
    assert path.read_bytes() == original
    assert [p.name for p in out_dir.iterdir()] == ["2024-02-08.parquet"]


def test_build_panel_clears_stale_temp_files_from_a_killed_run(raw_dir, out_dir):
    standard_day(D1, raw_dir)
    out_dir.mkdir()
    stale = out_dir / "2024-02-08.parquet.tmp-12345"
    stale.write_bytes(b"partial")
    panel.build_panel(raw_dir, out_dir, D1, D1)
    assert sorted(p.name for p in out_dir.iterdir()) == ["2024-02-08.parquet"]


def test_a_duplicate_stops_the_build_without_writing_that_day(raw_dir, out_dir):
    standard_day(D1, raw_dir)
    make_archive(raw_dir, D2, {(1, 100): [row(7, "Normal"), row(7, "Normal")]})
    with pytest.raises(ValueError, match="duplicate"):
        panel.build_panel(raw_dir, out_dir, D1, D2, workers=1)
    assert sorted(p.name for p in out_dir.iterdir()) == ["2024-02-08.parquet"]


def test_build_day_does_not_persist_an_archive_without_the_category(raw_dir, out_dir):
    make_archive(raw_dir, D1, {(2, 200): [row(50)]})  # other categories only
    with pytest.raises(ValueError, match=r"no category 1 price rows.*2024-02-08"):
        panel.build_day(raw_dir, out_dir, D1)
    assert not out_dir.exists() or list(out_dir.iterdir()) == []


def test_build_day_does_not_persist_a_category_whose_groups_have_no_rows(raw_dir, out_dir):
    make_archive(raw_dir, D1, {(1, 100): [], (1, 101): []})
    with pytest.raises(ValueError, match="no category 1 price rows"):
        panel.build_day(raw_dir, out_dir, D1)
    assert not out_dir.exists() or list(out_dir.iterdir()) == []


def test_an_empty_archive_stops_the_build_and_a_rerun_rechecks_it(raw_dir, out_dir):
    standard_day(D1, raw_dir)
    make_archive(raw_dir, D2, {(2, 200): [row(50)]})
    standard_day(D3, raw_dir)
    with pytest.raises(ValueError, match="no category 1 price rows"):
        panel.build_panel(raw_dir, out_dir, D1, D3, workers=1)
    assert sorted(p.name for p in out_dir.iterdir()) == ["2024-02-08.parquet"]

    # the empty day left no file behind, so once its archive is repaired the
    # resumed build writes it (it is not "skipped_existing" as an empty file)
    make_archive(raw_dir, D2, {(1, 100): [row(9)]})
    records = panel.build_panel(raw_dir, out_dir, D1, D3, workers=1)
    assert [r["status"] for r in records] == ["skipped_existing", "written", "written"]
    assert records[1]["rows"] == 1


def test_a_missing_archive_is_still_recorded_not_an_error(raw_dir, out_dir):
    """missing_archive (no file) stays distinct from an archive with no category rows."""
    standard_day(D1, raw_dir)
    records = panel.build_panel(raw_dir, out_dir, D1, D3, workers=1)
    assert [r["status"] for r in records] == ["written", "missing_archive", "missing_archive"]


# --- load_panel -------------------------------------------------------------------


def price_grid_archive(raw_dir, date):
    return make_archive(
        raw_dir,
        date,
        {
            (1, 100): [
                row(1, "Normal", 1.5, 9.0),  # market preferred
                row(2, "Normal", None, 0.25),  # mid fallback
                row(3, "Normal", None, None),  # neither: dropped
                row(4, "Normal", 0.0, 7.0),  # a market of 0.0 is a price, not a null
                row(5, "Foil", 6.0, None),  # market only
            ]
        },
    )


def test_load_panel_price_policy_market_then_mid(raw_dir, out_dir):
    price_grid_archive(raw_dir, D1)
    panel.build_panel(raw_dir, out_dir, D1, D1)
    df = panel.load_panel(out_dir)
    assert list(df.columns) == ["date", "product_id", "subtype", "price", "field"]
    assert df["product_id"].tolist() == [1, 2, 4, 5]  # product 3 has neither field
    assert df["field"].tolist() == ["market", "mid", "market", "market"]
    assert df["price"].tolist() == pytest.approx([1.5, 0.25, 0.0, 6.0])
    assert df["subtype"].tolist() == ["Normal", "Normal", "Normal", "Foil"]
    assert str(df["date"].dtype) == "datetime64[ns]"
    assert set(df["date"]) == {pd.Timestamp("2024-02-08")}


@pytest.mark.parametrize("policy", list(PriceSelectionPolicy))
def test_load_panel_agrees_with_price_from_row_for_every_policy(raw_dir, out_dir, policy):
    values = [None, 0.0, 1.5]
    cases = [(m, d) for m in values for d in values]
    rows = [row(100 + i, "Normal", m, d) for i, (m, d) in enumerate(cases)]
    make_archive(raw_dir, D1, {(1, 100): rows})
    panel.build_panel(raw_dir, out_dir, D1, D1)
    got = panel.load_panel(out_dir, policy=policy).set_index("product_id")
    for i, (m, d) in enumerate(cases):
        expected_price, expected_field = _price_from_row({"marketPrice": m, "midPrice": d}, policy)
        if expected_price is None:
            assert 100 + i not in got.index
        else:
            assert got.loc[100 + i, "field"] == expected_field
            assert got.loc[100 + i, "price"] == pytest.approx(expected_price)


def test_load_panel_default_policy_is_market_then_mid():
    import inspect

    default = inspect.signature(panel.load_panel).parameters["policy"].default
    assert default is PriceSelectionPolicy.MARKET_THEN_MID


def test_load_panel_filters_by_exact_product_subtype_pairs(raw_dir, out_dir):
    for date in (D1, D2):
        make_archive(
            raw_dir,
            date,
            {
                (1, 100): [
                    row(1, "Normal", 1.0),
                    row(1, "Foil", 2.0),
                    row(2, "Normal", 3.0),
                    row(2, "Foil", 4.0),
                    row(3, None, 5.0),
                    row(3, "Foil", 6.0),
                    row(4, "Normal", 7.0),
                ]
            },
        )
    panel.build_panel(raw_dir, out_dir, D1, D2)

    # (1, Foil) and (2, Normal) -- not the cross product (1, Normal) / (2, Foil)
    df = panel.load_panel(out_dir, keys=[(1, "Foil"), (2, "Normal")])
    pairs = list(zip(df["product_id"], df["subtype"], strict=True))
    assert pairs == [(1, "Foil"), (2, "Normal")] * 2
    assert df.groupby("product_id")["price"].first().to_dict() == {1: 2.0, 2: 3.0}
    assert [d.date() for d in df["date"]] == [D1, D1, D2, D2]  # date-major order

    # a null subtype is matched only by None
    df = panel.load_panel(out_dir, keys=[(3, None)])
    assert df["product_id"].tolist() == [3, 3] and df["subtype"].isna().all()
    assert panel.load_panel(out_dir, keys=[(3, "Normal")]).empty

    # generators work (consumed once), numpy ints work, unknown keys give an empty frame
    df = panel.load_panel(out_dir, keys=((np.int64(p), s) for p, s in [(4, "Normal")]))
    assert df["product_id"].tolist() == [4, 4]
    assert panel.load_panel(out_dir, keys=[(999, "Normal")]).empty
    assert panel.load_panel(out_dir, keys=[(2**40, "Normal")]).empty
    empty = panel.load_panel(out_dir, keys=[])
    assert empty.empty and list(empty.columns) == [
        "date",
        "product_id",
        "subtype",
        "price",
        "field",
    ]
    assert str(empty["date"].dtype) == "datetime64[ns]"


def test_load_panel_date_window_is_inclusive_and_ignores_foreign_files(raw_dir, out_dir):
    for date in (D1, D2, D3):
        standard_day(date, raw_dir)
    panel.build_panel(raw_dir, out_dir, D1, D3)
    (out_dir / "notes.txt").write_text("x")
    (out_dir / "2024-02-09.parquet.tmp-5").write_bytes(b"partial")

    all_days = panel.load_panel(out_dir, keys=[(1, "Normal")])
    assert [d.date() for d in all_days["date"]] == [D1, D2, D3]
    window = panel.load_panel(out_dir, keys=[(1, "Normal")], start=D2, end=D3)
    assert [d.date() for d in window["date"]] == [D2, D3]
    window = panel.load_panel(out_dir, keys=[(1, "Normal")], start="2024-02-09", end="2024-02-09")
    assert [d.date() for d in window["date"]] == [D2]
    assert panel.load_panel(out_dir, start=dt.date(2025, 1, 1)).empty


def test_load_panel_on_an_empty_directory(out_dir):
    out_dir.mkdir()
    df = panel.load_panel(out_dir, keys=[(1, "Normal")])
    assert df.empty and list(df.columns) == ["date", "product_id", "subtype", "price", "field"]


# --- CLI ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("build_panel_12", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_help_lists_the_flags(cli, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for flag in ("--as-of", "--start", "--workers", "--out", "--force"):
        assert flag in text


def test_cli_defaults_and_required_as_of(cli):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])
    args = cli.build_parser().parse_args(["--as-of", "2026-09-04"])
    assert args.as_of == dt.date(2026, 9, 4)
    assert args.start == dt.date(2024, 2, 8)
    assert 1 <= args.workers <= 4
    assert args.out is None and args.force is False
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--as-of", "2026-09-04", "--workers", "0"])


def test_cli_rejects_start_after_as_of(cli):
    with pytest.raises(SystemExit):
        cli.main(["--as-of", "2024-02-01", "--start", "2024-02-08"])


def test_cli_end_to_end_writes_panel_and_a_price_free_manifest(cli, tmp_path, monkeypatch, capsys):
    raw = tmp_path / "raw"
    secret = 4242.4242
    make_archive(raw, D1, {(1, 100): [row(1, "Normal", secret, 1.0), row(1, "Foil", 8123.5, 8.0)]})
    make_archive(raw, D2, {(1, 100): [row(1, "Normal", 2.5, 2.0)]})
    monkeypatch.setattr(cli.config, "RAW_DIR", raw)
    monkeypatch.setattr(cli.config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cli.config, "RESULTS_DIR", tmp_path / "results")

    # as-of is inclusive (as in 01_fetch_data): 2024-02-10 has no archive -> recorded missing
    cli.main(["--as-of", "2024-02-10", "--start", "2024-02-08", "--workers", "1"])
    out = capsys.readouterr().out
    assert out.count("\n") == 1  # one-line summary on stdout
    assert "written 2" in out and "missing_archive 1" in out

    panel_dir = tmp_path / "cache" / "panel"
    assert sorted(p.name for p in panel_dir.iterdir()) == [
        "2024-02-08.parquet",
        "2024-02-09.parquet",
    ]
    (manifest_path,) = (tmp_path / "results" / "manifests").glob("panel-2024-02-10-*.json")
    text = manifest_path.read_text()
    manifest = json.loads(text)

    assert manifest["_meta"]["as_of"] == "2024-02-10" and manifest["_meta"]["git_sha"]
    assert "git_dirty" in manifest["_meta"]
    params = manifest["parameters"]
    assert (params["start"], params["end"], params["workers"], params["force"]) == (
        "2024-02-08",
        "2024-02-10",
        1,
        False,
    )
    assert manifest["counts"] == {
        "written": 2,
        "skipped_existing": 0,
        "missing_archive": 1,
        "dates": 3,
        "rows": 3,
    }
    assert manifest["missing_dates"] == ["2024-02-10"]
    by_date = {r["date"]: r for r in manifest["records"]}
    assert by_date["2024-02-08"]["archive_sha256"] == sha256(
        raw / "tcgcsv-prices-2024-02-08.ppmd.7z"
    )
    assert by_date["2024-02-08"]["output_sha256"] == sha256(panel_dir / "2024-02-08.parquet")
    assert by_date["2024-02-08"]["rows"] == 2 and by_date["2024-02-08"]["status"] == "written"
    assert by_date["2024-02-10"]["status"] == "missing_archive"
    for needle in (str(secret), "8123.5"):
        assert needle not in text
    assert all(
        set(r) == {"date", "archive_sha256", "output_sha256", "rows", "status"}
        for r in manifest["records"]
    )

    # a rerun resumes: everything already built is skipped
    cli.main(["--as-of", "2024-02-10", "--start", "2024-02-08", "--workers", "1"])
    assert "skipped_existing 2" in capsys.readouterr().out


def test_cli_out_flag_redirects_the_panel(cli, tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    make_archive(raw, D1, {(1, 100): [row(1)]})
    monkeypatch.setattr(cli.config, "RAW_DIR", raw)
    monkeypatch.setattr(cli.config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cli.config, "RESULTS_DIR", tmp_path / "results")
    cli.main(
        ["--as-of", "2024-02-08", "--start", "2024-02-08", "--out", str(tmp_path / "elsewhere")]
    )
    assert (tmp_path / "elsewhere" / "2024-02-08.parquet").exists()
    assert not (tmp_path / "cache" / "panel").exists()


@pytest.mark.skipif(not (REPO_ROOT / ".git").exists(), reason="code snapshot without .git")
def test_default_panel_location_is_gitignored():
    result = subprocess.run(
        ["git", "check-ignore", "-q", "data/cache/panel/2024-02-08.parquet"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    assert result.returncode == 0
