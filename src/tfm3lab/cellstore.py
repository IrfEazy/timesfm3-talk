"""Crash-safe per-cell result store for long benchmark grids.

A run is identified by an *execution fingerprint*: the sha256 of a canonical
JSON payload holding everything that determines the scientific result (the
resolved config including defaults, git sha + dirty flag + digest of the
result-determining source files (RESULT_DETERMINING_PATHS only),
checkpoint repo/revision, card-manifest and input-cache sha256, schema and
metric versions, seed). Its directory is

    <root>/<config_id>/<fingerprint[:12]>/
        manifest.json                 (written once, atomically, at creation)
        cell=<cell_id>/part.parquet   (one grid cell's rows)
        cell=<cell_id>/done.json      (commit marker: THE commit point)
        _quarantine/                  (uncommitted debris moved aside on resume)

Commit protocol for one cell (single writer per run):

    1. write the rows to a uniquely named temp file in the cell directory;
       fsync it;
    2. validate the temp file: schema, strictly increasing canonical keys
       (detects duplicates without holding a key set), every key inside the
       cell's expected domain, row count == size of that domain (so the cell
       covers its domain exactly);
    3. os.replace(temp, part.parquet); fsync the directory;
    4. write done.json (protocol_version, run_id, cell_id, schema_version,
       filename, byte_size, row_count, sha256, coverage_digest) the same way:
       temp + fsync + os.replace + directory fsync.

Resume invariants (see RunStore.open):

    | cell state                           | action                         |
    |--------------------------------------|--------------------------------|
    | no directory                         | compute                        |
    | temp file(s) only / file, no marker  | quarantine, recompute          |
    | marker, missing file                 | fail loudly                    |
    | marker + hash/size/count/schema/key  | fail loudly (repair=True:      |
    |   mismatch, unreadable marker        |   quarantine and recompute)    |
    | fully valid pair                     | skip, never overwrite          |

Every check is done read-only over the whole run BEFORE anything is moved,
and a fingerprint mismatch refuses before any mutation. Readers
(RunStore.iter_committed) re-verify each marker and only ever read
marker-validated files, in sorted cell_id order -- never a recursive glob.

This relies on a local filesystem with POSIX rename/fsync semantics: the run
directory must NOT live on a FUSE/network/Drive mount (rename and fsync are not
atomic or durable there). Compute on local disk, copy the finished run out at
the end.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import secrets
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROTOCOL_VERSION = 1
CANONICALIZATION_VERSION = 1
PART_NAME = "part.parquet"
MARKER_NAME = "done.json"
MANIFEST_NAME = "manifest.json"
QUARANTINE_DIR = "_quarantine"
CELL_DIR_PREFIX = "cell="

_CELL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_MARKER_FIELDS = (
    "protocol_version",
    "run_id",
    "cell_id",
    "schema_version",
    "filename",
    "byte_size",
    "row_count",
    "sha256",
    "coverage_digest",
)

# Paths (relative to the repo root) whose content determines results. Only
# these enter the code digest in the fingerprint: uv.lock, docs/, tests/,
# notebooks and logs can change without invalidating a run.
RESULT_DETERMINING_PATHS = (
    "src/tfm3lab",
    "scripts/02b_exp_mtg_benchmark.py",
    "configs",
    "pyproject.toml",
)

# Test-only fault injection: called as hook(stage, cell_id) at each commit
# stage ("after_temp_write", "during_validation", "after_part_rename",
# "during_marker_write", "after_marker_rename"). None in production.
_FAULT_HOOK: Callable[[str, str], None] | None = None


class CellStoreError(RuntimeError):
    """Base class: the store refuses to proceed."""


class FingerprintMismatchError(CellStoreError):
    """--resume against a run whose execution fingerprint differs."""


class CorruptCellError(CellStoreError):
    """A committed cell (marker present) failed verification."""


class CellValidationError(CellStoreError):
    """Rows about to be committed violate schema, keys or coverage."""


def _fault(stage: str, cell_id: str) -> None:
    if _FAULT_HOOK is not None:
        _FAULT_HOOK(stage, cell_id)


# --------------------------------------------------------------------------
# canonical JSON, hashing, fingerprint
# --------------------------------------------------------------------------


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (tuple, set, frozenset)):
        return list(obj)
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"not canonically serializable: {type(obj).__name__}")


def canonical_json(obj) -> str:
    """Deterministic JSON: sorted keys, no whitespace, tuples as lists."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default
    )


def _normalize(obj):
    """Round-trip through canonical JSON so in-memory payloads compare equal
    to payloads read back from manifest.json (tuples -> lists, etc.)."""
    return json.loads(canonical_json(obj))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fingerprint_payload(
    *,
    resolved_config: dict,
    git_sha: str,
    git_dirty: bool | None,
    git_dirty_digest: str | None,
    checkpoint: dict | None,
    card_manifest_sha256: str,
    input_cache_sha256: str,
    schema_version: int,
    metric_version: int,
    seed: int | None,
    extra: dict | None = None,
) -> dict:
    """Everything that determines the scientific result of a run. Operational
    settings (output location, verbosity, --resume itself) are deliberately
    NOT part of it; `extra` carries caller-specific scientific inputs."""
    payload = {
        "canonicalization_version": CANONICALIZATION_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "resolved_config": resolved_config,
        "git": {"sha": git_sha, "dirty": git_dirty, "dirty_digest": git_dirty_digest},
        "checkpoint": checkpoint,
        "card_manifest_sha256": card_manifest_sha256,
        "input_cache_sha256": input_cache_sha256,
        "schema_version": schema_version,
        "metric_version": metric_version,
        "seed": seed,
        "extra": extra or {},
    }
    return _normalize(payload)


def fingerprint_of(payload: dict) -> str:
    return sha256_text(canonical_json(payload))


def _snapshot_digest(root: Path, paths: tuple[str, ...]) -> str:
    files: set[Path] = set()
    for rel in paths:
        target = root / rel
        if target.is_file():
            files.add(target)
        elif target.is_dir():
            files.update(p for p in target.rglob("*") if p.is_file())
    digest = hashlib.sha256(b"snapshot")
    for path in sorted(files):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest.update(b"\0file\0" + path.relative_to(root).as_posix().encode() + b"\0")
        try:
            digest.update(bytes.fromhex(file_sha256(path)))
        except OSError:
            digest.update(b"<unreadable>")
    return digest.hexdigest()


def git_worktree_digest(
    repo_root: str | Path, paths: tuple[str, ...] = RESULT_DETERMINING_PATHS
) -> str | None:
    """sha256 over `git diff HEAD` plus the contents of untracked files,
    restricted to `paths` (the result-determining sources). Everything else
    (uv.lock, docs, tests, notebooks, logs, results) is ignored. None when
    git is unavailable. A clean scope yields the digest of empty input.

    A code snapshot has no .git: the digest is then taken over the snapshot's own
    files under `paths` (names and contents, __pycache__ skipped)."""
    if not (Path(repo_root) / ".git").exists():
        return _snapshot_digest(Path(repo_root), paths)
    pathspec = ["--", *paths]
    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--binary", *pathspec],
            capture_output=True,
            check=True,
            timeout=30,
            cwd=repo_root,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z", *pathspec],
            capture_output=True,
            check=True,
            timeout=30,
            cwd=repo_root,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    digest = hashlib.sha256(diff)
    for name in sorted(n for n in untracked.split(b"\0") if n):
        if b"__pycache__" in name or name.endswith(b".pyc"):
            continue
        digest.update(b"\0untracked\0" + name + b"\0")
        path = Path(repo_root) / os.fsdecode(name)
        try:
            digest.update(bytes.fromhex(file_sha256(path)))
        except OSError:
            digest.update(b"<unreadable>")
    return digest.hexdigest()


# --------------------------------------------------------------------------
# durable file primitives
# --------------------------------------------------------------------------


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # platforms without directory fds
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _tmp_name(final: Path) -> Path:
    return final.with_name(f"{final.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")


def _atomic_write_bytes(
    path: Path, data: bytes, *, stage: str | None = None, cell_id: str = ""
) -> None:
    tmp = _tmp_name(path)
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    if stage is not None:
        _fault(stage, cell_id)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def atomic_write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n"
    _atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_table(table: pa.Table, path: str | Path, compression: str = "zstd") -> None:
    """Write a parquet file via temp + fsync + os.replace + dir fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_name(path)
    try:
        pq.write_table(table, tmp, compression=compression)
        _fsync_file(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    _fsync_dir(path.parent)


# --------------------------------------------------------------------------
# cell specification and validation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSpec:
    """One grid cell's identity and expected coverage: the rows must be
    exactly the Cartesian product of `key_domains` (in column order), sorted
    strictly increasing by those key columns."""

    cell_id: str
    key_domains: tuple[tuple[str, tuple], ...]
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        if not _CELL_ID_RE.match(self.cell_id):
            raise ValueError(f"cell_id {self.cell_id!r} must match {_CELL_ID_RE.pattern}")
        for col, domain in self.key_domains:
            if list(domain) != sorted(set(domain)):
                raise ValueError(f"{self.cell_id}: domain of {col!r} must be sorted and unique")

    @property
    def key_columns(self) -> tuple[str, ...]:
        return tuple(col for col, _ in self.key_domains)

    @property
    def expected_rows(self) -> int:
        n = 1
        for _, domain in self.key_domains:
            n *= len(domain)
        return n

    @property
    def coverage_digest(self) -> str:
        return sha256_text(canonical_json([[c, list(d)] for c, d in self.key_domains]))

    def describe(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "expected_rows": self.expected_rows,
            "coverage_digest": self.coverage_digest,
            **self.meta,
        }


def _lex_strictly_increasing(arrays: list[np.ndarray]) -> bool:
    if len(arrays[0]) < 2:
        return True
    greater = np.zeros(len(arrays[0]) - 1, dtype=bool)
    equal = np.ones(len(arrays[0]) - 1, dtype=bool)
    for a in arrays:
        lo, hi = a[:-1], a[1:]
        greater |= equal & np.asarray(hi > lo, dtype=bool)
        equal &= np.asarray(hi == lo, dtype=bool)
    return bool(greater.all())


def validate_parquet(
    path: Path,
    spec: CellSpec,
    schema: pa.Schema,
    batch_size: int = 65536,
    *,
    inject_fault: bool = False,
) -> int:
    """Raise CellValidationError unless `path` has exactly `schema`, its key
    columns are non-null, inside the cell's domain and strictly increasing
    (across batch boundaries too), and the row count equals the domain size.
    Returns the row count. Streams only the key columns."""
    try:
        pf = pq.ParquetFile(path)
        file_schema = pf.schema_arrow
    except Exception as exc:  # noqa: BLE001 - any read failure is a validation failure
        raise CellValidationError(f"{spec.cell_id}: unreadable parquet ({exc})") from exc
    if not file_schema.equals(schema, check_metadata=False):
        raise CellValidationError(f"{spec.cell_id}: schema mismatch")
    n_rows = pf.metadata.num_rows
    if n_rows != spec.expected_rows:
        raise CellValidationError(
            f"{spec.cell_id}: {n_rows} rows, expected exactly {spec.expected_rows}"
        )
    keys = list(spec.key_columns)
    value_sets = {col: pa.array(list(domain)) for col, domain in spec.key_domains}
    prev: list[np.ndarray] | None = None
    seen = 0
    try:
        for batch in pf.iter_batches(batch_size=batch_size, columns=keys):
            cols = []
            for col in keys:
                arr = batch.column(col)
                if arr.null_count:
                    raise CellValidationError(f"{spec.cell_id}: null key in {col!r}")
                inside = pc.all(pc.is_in(arr, value_set=value_sets[col])).as_py()
                if not inside:
                    raise CellValidationError(f"{spec.cell_id}: {col!r} value outside domain")
                cols.append(arr.to_numpy(zero_copy_only=False))
            if prev is not None:
                cols = [np.concatenate([p, c]) for p, c in zip(prev, cols, strict=True)]
            if not _lex_strictly_increasing(cols):
                raise CellValidationError(
                    f"{spec.cell_id}: keys {keys} not strictly increasing (duplicate or unsorted)"
                )
            prev = [c[-1:] for c in cols]
            seen += batch.num_rows
    except CellValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - corrupt pages surface here
        raise CellValidationError(f"{spec.cell_id}: unreadable data ({exc})") from exc
    if inject_fault:  # test hook: crash after keys were streamed, before the verdict
        _fault("during_validation", spec.cell_id)
    if seen != n_rows:
        raise CellValidationError(f"{spec.cell_id}: decoded {seen} rows, metadata says {n_rows}")
    return n_rows


# --------------------------------------------------------------------------
# run store
# --------------------------------------------------------------------------


@dataclass
class OpenReport:
    created: bool
    committed: list[str]
    pending: list[str]
    quarantined: list[str]


def _payload_diff(a: dict, b: dict, prefix: str = "") -> list[str]:
    keys = sorted(set(a) | set(b))
    out = []
    for k in keys:
        path = f"{prefix}{k}"
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if isinstance(va, dict) and isinstance(vb, dict):
            out.extend(_payload_diff(va, vb, prefix=f"{path}."))
        elif va != vb:
            out.append(path)
    return out


class RunStore:
    """One execution's run directory. Construct with RunStore.open()."""

    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        schema: pa.Schema,
        schema_version: int,
        cells: dict[str, CellSpec],
    ):
        self.run_dir = run_dir
        self.run_id = run_id
        self.schema = schema
        self.schema_version = schema_version
        self.cells = cells
        self.committed: dict[str, dict] = {}

    # ---- opening ---------------------------------------------------------

    @classmethod
    def open(
        cls,
        root: str | Path,
        config_id: str,
        payload: dict,
        schema: pa.Schema,
        schema_version: int,
        cells: list[CellSpec],
        *,
        resume: bool,
        repair: bool = False,
    ) -> tuple[RunStore, OpenReport]:
        if not _CELL_ID_RE.match(config_id):
            raise CellStoreError(f"config_id {config_id!r} is not a safe directory name")
        by_id = {c.cell_id: c for c in cells}
        if len(by_id) != len(cells):
            raise CellStoreError("duplicate cell_id in the cell list")
        payload = _normalize(payload)
        fingerprint = fingerprint_of(payload)
        run_id = fingerprint[:12]
        config_dir = Path(root) / config_id
        run_dir = config_dir / run_id
        store = cls(run_dir, run_id, schema, schema_version, by_id)
        expected_manifest = store._manifest_body(fingerprint, payload)

        created = False
        if run_dir.exists():
            if not resume:
                raise CellStoreError(
                    f"{run_dir} already exists; pass --resume to continue it "
                    "(committed cells are never overwritten)"
                )
            store._check_manifest(expected_manifest)
        else:
            if resume:
                siblings = (
                    sorted(p for p in config_dir.iterdir() if p.is_dir())
                    if config_dir.exists()
                    else []
                )
                if siblings:
                    details = []
                    for sib in siblings:
                        try:
                            old = json.loads((sib / MANIFEST_NAME).read_text(encoding="utf-8"))
                            diff = _payload_diff(old.get("fingerprint_payload", {}), payload)
                            details.append(f"{sib.name}: differs in {diff}")
                        except (OSError, ValueError):
                            details.append(f"{sib.name}: unreadable manifest")
                    raise FingerprintMismatchError(
                        f"--resume: no run with execution fingerprint {run_id} under "
                        f"{config_dir}; existing run(s) have a different fingerprint "
                        f"({'; '.join(details)}). Refusing to resume; nothing was modified. "
                        "Start a fresh run (omit --resume) or restore the original inputs."
                    )
            created = True

        # Read-only verification of every cell before any mutation.
        states = store._scan()
        corrupt = {cid: why for cid, (state, why) in states.items() if state == "corrupt"}
        if corrupt and not repair:
            listing = "; ".join(f"{cid}: {why}" for cid, why in sorted(corrupt.items()))
            raise CorruptCellError(
                f"committed cell(s) failed verification in {run_dir}: {listing}. Nothing was "
                "modified. Investigate, or rerun with --repair to quarantine and recompute them."
            )

        if created:
            run_dir.mkdir(parents=True, exist_ok=True)
            _fsync_dir(config_dir)
            _fsync_dir(config_dir.parent)
            atomic_write_json(run_dir / MANIFEST_NAME, expected_manifest)
        elif not (run_dir / MANIFEST_NAME).exists():
            atomic_write_json(run_dir / MANIFEST_NAME, expected_manifest)

        quarantined = []
        for cid, (state, _why) in sorted(states.items()):
            if state in ("uncommitted", "corrupt"):
                store._quarantine(cid)
                quarantined.append(cid)
        pending = sorted(cid for cid in by_id if cid not in store.committed)
        report = OpenReport(
            created=created,
            committed=sorted(store.committed),
            pending=pending,
            quarantined=quarantined,
        )
        return store, report

    def _manifest_body(self, fingerprint: str, payload: dict) -> dict:
        return _normalize(
            {
                "protocol_version": PROTOCOL_VERSION,
                "run_id": self.run_id,
                "fingerprint": fingerprint,
                "fingerprint_payload": payload,
                "schema_version": self.schema_version,
                "schema": self.schema.remove_metadata().to_string(show_schema_metadata=False),
                "cells": [self.cells[c].describe() for c in sorted(self.cells)],
            }
        )

    def _check_manifest(self, expected: dict) -> None:
        path = self.run_dir / MANIFEST_NAME
        if not path.exists():
            if any(p.name.startswith(CELL_DIR_PREFIX) for p in self.run_dir.iterdir()):
                raise CellStoreError(f"{path} is missing but cell directories exist; refusing")
            return  # crashed between mkdir and manifest write: re-initialized by open()
        try:
            found = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CellStoreError(f"{path} is unreadable ({exc}); refusing to resume") from exc
        if found.get("fingerprint") != expected["fingerprint"]:
            diff = _payload_diff(
                found.get("fingerprint_payload", {}), expected["fingerprint_payload"]
            )
            raise FingerprintMismatchError(
                f"{path}: execution fingerprint differs (fields {diff}); refusing to resume"
            )
        for key in (
            "protocol_version",
            "run_id",
            "fingerprint_payload",
            "schema_version",
            "schema",
            "cells",
        ):
            if found.get(key) != expected[key]:
                raise FingerprintMismatchError(
                    f"{path}: manifest field {key!r} differs from this execution; refusing"
                )

    # ---- scanning --------------------------------------------------------

    def cell_dir(self, cell_id: str) -> Path:
        return self.run_dir / f"{CELL_DIR_PREFIX}{cell_id}"

    def _scan(self) -> dict[str, tuple[str, str]]:
        """cell_id -> (state, reason) for every cell directory present;
        populates self.committed. Read-only."""
        states: dict[str, tuple[str, str]] = {}
        if not self.run_dir.exists():
            return states
        for entry in sorted(self.run_dir.iterdir()):
            if not entry.name.startswith(CELL_DIR_PREFIX):
                continue
            cid = entry.name[len(CELL_DIR_PREFIX) :]
            if cid not in self.cells:
                raise CorruptCellError(
                    f"{entry}: unknown cell (not in this run's cell list); refusing"
                )
            if not entry.is_dir():
                raise CorruptCellError(f"{entry}: expected a directory; refusing")
            marker_path = entry / MARKER_NAME
            if not marker_path.exists():
                states[cid] = ("uncommitted", "no commit marker")
                continue
            try:
                marker = self.verify_cell(cid)
            except CellStoreError as exc:
                states[cid] = ("corrupt", str(exc))
                continue
            extras = sorted(
                p.name for p in entry.iterdir() if p.name not in (PART_NAME, MARKER_NAME)
            )
            if extras:
                states[cid] = ("corrupt", f"unexpected artifacts next to a commit: {extras}")
                continue
            self.committed[cid] = marker
            states[cid] = ("committed", "")
        return states

    def verify_cell(self, cell_id: str) -> dict:
        """Full verification of a committed cell; returns its marker or raises
        CorruptCellError."""
        spec = self.cells[cell_id]
        cdir = self.cell_dir(cell_id)
        marker_path = cdir / MARKER_NAME
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CorruptCellError(f"{cell_id}: unreadable commit marker ({exc})") from exc
        if not isinstance(marker, dict) or any(f not in marker for f in _MARKER_FIELDS):
            raise CorruptCellError(f"{cell_id}: commit marker missing required fields")
        expect = {
            "protocol_version": PROTOCOL_VERSION,
            "run_id": self.run_id,
            "cell_id": cell_id,
            "schema_version": self.schema_version,
            "filename": PART_NAME,
            "row_count": spec.expected_rows,
            "coverage_digest": spec.coverage_digest,
        }
        for key, value in expect.items():
            if marker[key] != value:
                raise CorruptCellError(
                    f"{cell_id}: marker {key}={marker[key]!r}, expected {value!r}"
                )
        part = cdir / PART_NAME
        if not part.exists():
            raise CorruptCellError(f"{cell_id}: commit marker present but {PART_NAME} is missing")
        size = part.stat().st_size
        if size != marker["byte_size"]:
            raise CorruptCellError(f"{cell_id}: {size} bytes, marker says {marker['byte_size']}")
        if file_sha256(part) != marker["sha256"]:
            raise CorruptCellError(f"{cell_id}: sha256 mismatch")
        try:
            validate_parquet(part, spec, self.schema)
        except CellValidationError as exc:
            raise CorruptCellError(str(exc)) from exc
        return marker

    def _quarantine(self, cell_id: str) -> Path:
        qdir = self.run_dir / QUARANTINE_DIR
        qdir.mkdir(exist_ok=True)
        ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        target = qdir / f"{CELL_DIR_PREFIX}{cell_id}.{ts}-{secrets.token_hex(3)}"
        os.replace(self.cell_dir(cell_id), target)
        _fsync_dir(self.run_dir)
        _fsync_dir(qdir)
        self.committed.pop(cell_id, None)
        return target

    # ---- committing ------------------------------------------------------

    def commit_cell(self, cell_id: str, table: pa.Table) -> dict:
        """Publish one cell with the temp/validate/rename/marker protocol."""
        if cell_id not in self.cells:
            raise CellStoreError(f"unknown cell {cell_id!r}")
        if cell_id in self.committed:
            raise CellStoreError(f"{cell_id} is already committed; never overwritten")
        spec = self.cells[cell_id]
        if not table.schema.equals(self.schema, check_metadata=False):
            raise CellValidationError(f"{cell_id}: table schema does not match the run schema")
        cdir = self.cell_dir(cell_id)
        if cdir.exists():
            raise CellStoreError(f"{cdir} exists without a commit; reopen the store to quarantine")
        cdir.mkdir()
        _fsync_dir(self.run_dir)

        part = cdir / PART_NAME
        tmp = _tmp_name(part)
        pq.write_table(table, tmp, compression="zstd")
        _fsync_file(tmp)
        _fault("after_temp_write", cell_id)
        try:
            row_count = validate_parquet(tmp, spec, self.schema, inject_fault=True)
        except CellValidationError:
            tmp.unlink(missing_ok=True)
            raise
        byte_size = tmp.stat().st_size
        sha = file_sha256(tmp)
        os.replace(tmp, part)
        _fsync_dir(cdir)
        _fault("after_part_rename", cell_id)

        marker = {
            "protocol_version": PROTOCOL_VERSION,
            "run_id": self.run_id,
            "cell_id": cell_id,
            "schema_version": self.schema_version,
            "filename": PART_NAME,
            "byte_size": byte_size,
            "row_count": row_count,
            "sha256": sha,
            "coverage_digest": spec.coverage_digest,
            "committed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        }
        data = (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _atomic_write_bytes(cdir / MARKER_NAME, data, stage="during_marker_write", cell_id=cell_id)
        _fault("after_marker_rename", cell_id)
        self.committed[cell_id] = marker
        return marker

    # ---- reading ---------------------------------------------------------

    @property
    def complete(self) -> bool:
        return set(self.committed) == set(self.cells)

    def committed_hashes(self) -> dict[str, dict]:
        return {
            cid: {k: self.committed[cid][k] for k in ("sha256", "row_count", "byte_size")}
            for cid in sorted(self.committed)
        }

    def iter_committed(
        self, columns: list[str] | None = None, cell_ids: list[str] | None = None
    ) -> Iterator[tuple[str, pa.Table]]:
        """Yield (cell_id, table) for committed cells in sorted cell_id order,
        re-verifying each marker+file pair immediately before reading it."""
        wanted = sorted(self.committed if cell_ids is None else cell_ids)
        for cid in wanted:
            if cid not in self.committed:
                raise CellStoreError(f"{cid} is not committed")
            marker = self.verify_cell(cid)
            if marker["sha256"] != self.committed[cid]["sha256"]:
                raise CorruptCellError(f"{cid}: committed file changed since it was opened")
            yield cid, pq.read_table(self.cell_dir(cid) / PART_NAME, columns=columns)
