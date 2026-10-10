"""Unit tests for tfm3lab.cellstore: commit protocol, validation, resume
classification and fingerprint refusal. Offline, tiny synthetic tables."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tfm3lab import cellstore
from tfm3lab.cellstore import (
    CellSpec,
    CellValidationError,
    CorruptCellError,
    FingerprintMismatchError,
    RunStore,
)

SCHEMA = pa.schema([("s", pa.string()), ("o", pa.int64()), ("v", pa.float64())])
DOMAIN_S = ("a", "b")
DOMAIN_O = (1, 2, 3)


def _spec(cell_id="c1"):
    return CellSpec(cell_id=cell_id, key_domains=(("s", DOMAIN_S), ("o", DOMAIN_O)))


def _rows(cell_value=0.0):
    keys = [(s, o) for s in DOMAIN_S for o in DOMAIN_O]
    return pa.table(
        {
            "s": [k[0] for k in keys],
            "o": [k[1] for k in keys],
            "v": [cell_value + i for i in range(len(keys))],
        },
        schema=SCHEMA,
    )


def _payload(**over):
    fields = dict(
        resolved_config={"config_id": "t", "horizons": (1, 2)},
        git_sha="abc",
        git_dirty=False,
        git_dirty_digest=None,
        checkpoint={"repo_id": "r", "revision": "deadbeef"},
        card_manifest_sha256="0" * 64,
        input_cache_sha256="1" * 64,
        schema_version=1,
        metric_version=1,
        seed=42,
    )
    fields.update(over)
    return cellstore.build_fingerprint_payload(**fields)


def _open(root, *, resume=False, repair=False, payload=None, cells=("c1", "c2")):
    return RunStore.open(
        root,
        "t",
        payload or _payload(),
        SCHEMA,
        1,
        [_spec(c) for c in cells],
        resume=resume,
        repair=repair,
    )


def _snapshot(root):
    return {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }


def test_fingerprint_is_canonical_and_sensitive():
    a = _payload()
    b = _payload(resolved_config={"horizons": [1, 2], "config_id": "t"})
    assert cellstore.fingerprint_of(a) == cellstore.fingerprint_of(b)
    assert cellstore.fingerprint_of(a) != cellstore.fingerprint_of(_payload(git_dirty=True))
    assert cellstore.fingerprint_of(a) != cellstore.fingerprint_of(
        _payload(checkpoint={"repo_id": "r", "revision": "other"})
    )


def test_cell_spec_counts_and_rejects_unsafe_ids():
    assert _spec().expected_rows == 6
    with pytest.raises(ValueError):
        CellSpec(cell_id="../evil", key_domains=(("s", DOMAIN_S),))
    with pytest.raises(ValueError):
        CellSpec(cell_id="ok", key_domains=(("o", (2, 1)),))


def test_commit_then_resume_skips_and_reads_back(tmp_path):
    store, rep = _open(tmp_path)
    assert rep.created and rep.pending == ["c1", "c2"]
    marker = store.commit_cell("c1", _rows())
    for key in (
        "protocol_version",
        "run_id",
        "cell_id",
        "schema_version",
        "filename",
        "byte_size",
        "row_count",
        "sha256",
    ):
        assert key in marker
    assert marker["row_count"] == 6
    on_disk = json.loads((store.cell_dir("c1") / "done.json").read_text())
    assert on_disk["sha256"] == cellstore.file_sha256(store.cell_dir("c1") / "part.parquet")

    store2, rep2 = _open(tmp_path, resume=True)
    assert rep2.committed == ["c1"] and rep2.pending == ["c2"]
    ((cid, table),) = list(store2.iter_committed())
    assert cid == "c1" and table.equals(_rows())
    with pytest.raises(cellstore.CellStoreError, match="never overwritten"):
        store2.commit_cell("c1", _rows())


def test_existing_run_without_resume_refuses(tmp_path):
    _open(tmp_path)
    with pytest.raises(cellstore.CellStoreError, match="--resume"):
        _open(tmp_path)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda t: t.slice(0, 5), "rows"),  # missing row
        (lambda t: pa.concat_tables([t.slice(0, 5), t.slice(4, 1)]), "strictly increasing"),
        (lambda t: t.take([1, 0, 2, 3, 4, 5]), "strictly increasing"),
        (lambda t: t.set_column(1, "o", pa.array([1, 2, 9, 1, 2, 3])), "outside domain"),
    ],
    ids=["short", "duplicate_same_count", "unsorted", "out_of_domain"],
)
def test_commit_rejects_bad_rows(tmp_path, mutate, match):
    store, _ = _open(tmp_path)
    with pytest.raises(CellValidationError, match=match):
        store.commit_cell("c1", mutate(_rows()))
    assert not (store.cell_dir("c1") / "done.json").exists()


def test_commit_rejects_schema_mismatch(tmp_path):
    store, _ = _open(tmp_path)
    bad = _rows().cast(pa.schema([("s", pa.string()), ("o", pa.int32()), ("v", pa.float64())]))
    with pytest.raises(CellValidationError, match="schema"):
        store.commit_cell("c1", bad)


def test_duplicate_across_batch_boundary_is_detected(tmp_path):
    rows = _rows()
    # same count as the domain, but (b, 1) duplicated at the batch seam and
    # (a, 3) missing -- only cross-batch comparison catches this.
    dup = pa.concat_tables([rows.slice(0, 2), rows.slice(3, 1), rows.slice(3, 3)])
    path = tmp_path / "x.parquet"
    pq.write_table(dup, path)
    with pytest.raises(CellValidationError):
        cellstore.validate_parquet(path, _spec(), SCHEMA, batch_size=3)


def test_temp_only_and_file_without_marker_are_quarantined(tmp_path):
    store, _ = _open(tmp_path)
    for cid in ("c1", "c2"):
        store.cell_dir(cid).mkdir()
    (store.cell_dir("c1") / "part.parquet.tmp-1-ab").write_bytes(b"partial")
    pq.write_table(_rows(), store.cell_dir("c2") / "part.parquet")
    store2, rep = _open(tmp_path, resume=True)
    assert rep.quarantined == ["c1", "c2"] and rep.pending == ["c1", "c2"]
    assert not store2.cell_dir("c1").exists()
    assert len(list((store2.run_dir / "_quarantine").iterdir())) == 2
    store2.commit_cell("c1", _rows())


@pytest.mark.parametrize("damage", ["delete_file", "truncate", "flip", "bad_json", "wrong_sha"])
def test_damaged_committed_cell_fails_loudly_without_mutation(tmp_path, damage):
    store, _ = _open(tmp_path)
    store.commit_cell("c1", _rows())
    part = store.cell_dir("c1") / "part.parquet"
    marker = store.cell_dir("c1") / "done.json"
    if damage == "delete_file":
        part.unlink()
    elif damage == "truncate":
        part.write_bytes(part.read_bytes()[:-10])
    elif damage == "flip":
        data = bytearray(part.read_bytes())
        data[len(data) // 2] ^= 0xFF
        part.write_bytes(bytes(data))
    elif damage == "bad_json":
        marker.write_text("{not json")
    else:
        m = json.loads(marker.read_text())
        m["sha256"] = "0" * 64
        marker.write_text(json.dumps(m))
    before = _snapshot(tmp_path)
    with pytest.raises(CorruptCellError):
        _open(tmp_path, resume=True)
    assert _snapshot(tmp_path) == before
    store2, rep = _open(tmp_path, resume=True, repair=True)
    assert rep.quarantined == ["c1"] and "c1" in rep.pending


def test_unknown_cell_directory_refuses(tmp_path):
    store, _ = _open(tmp_path)
    (store.run_dir / "cell=zzz").mkdir()
    with pytest.raises(CorruptCellError, match="unknown cell"):
        _open(tmp_path, resume=True)


def test_resume_with_different_fingerprint_refuses_before_mutation(tmp_path):
    store, _ = _open(tmp_path)
    store.cell_dir("c2").mkdir()
    (store.cell_dir("c2") / "part.parquet.tmp-1-ab").write_bytes(b"x")
    before = _snapshot(tmp_path)
    with pytest.raises(FingerprintMismatchError, match="git.sha"):
        _open(tmp_path, resume=True, payload=_payload(git_sha="other"))
    assert _snapshot(tmp_path) == before


def test_tampered_manifest_refuses(tmp_path):
    store, _ = _open(tmp_path)
    path = store.run_dir / "manifest.json"
    m = json.loads(path.read_text())
    m["fingerprint_payload"]["seed"] = 7
    path.write_text(json.dumps(m))
    with pytest.raises(FingerprintMismatchError):
        _open(tmp_path, resume=True)


def test_fault_during_marker_write_leaves_no_commit(tmp_path, monkeypatch):
    store, _ = _open(tmp_path)

    def hook(stage, cell_id):
        if stage == "during_marker_write":
            raise RuntimeError("boom")

    monkeypatch.setattr(cellstore, "_FAULT_HOOK", hook)
    with pytest.raises(RuntimeError):
        store.commit_cell("c1", _rows())
    monkeypatch.setattr(cellstore, "_FAULT_HOOK", None)
    assert (store.cell_dir("c1") / "part.parquet").exists()
    assert not (store.cell_dir("c1") / "done.json").exists()
    _, rep = _open(tmp_path, resume=True)
    assert rep.quarantined == ["c1"]


def test_atomic_write_table_replaces(tmp_path):
    path = tmp_path / "out.parquet"
    cellstore.atomic_write_table(_rows(), path)
    cellstore.atomic_write_table(_rows(1.0), path)
    assert pq.read_table(path).equals(_rows(1.0))
    assert [p.name for p in tmp_path.iterdir()] == ["out.parquet"]


def _git_repo(tmp_path):
    import subprocess

    root = tmp_path / "repo"
    for rel, text in {
        "src/tfm3lab/a.py": "x = 1\n",
        "scripts/02b_exp_mtg_benchmark.py": "y = 1\n",
        "configs/c.json": "{}\n",
        "pyproject.toml": "[project]\n",
        "uv.lock": "lock 1\n",
        "docs/guide.md": "doc\n",
        "notebooks/n.ipynb": "{}\n",
        "tests/test_x.py": "t = 1\n",
    }.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)

    def git(*a):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *a],
            cwd=root,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    return root


def test_code_digest_ignores_lock_docs_notebooks_tests(tmp_path):
    root = _git_repo(tmp_path)
    base = cellstore.git_worktree_digest(root)
    assert base is not None
    (root / "uv.lock").write_text("lock 2\n")
    (root / "docs/guide.md").write_text("changed\n")
    (root / "notebooks/n.ipynb").write_text('{"cells": []}\n')
    (root / "tests/test_x.py").write_text("t = 2\n")
    (root / "docs/new.md").write_text("untracked doc\n")
    (root / "run.log").write_text("log\n")
    (root / "src/tfm3lab/__pycache__").mkdir()
    (root / "src/tfm3lab/__pycache__/a.pyc").write_bytes(b"\0")
    assert cellstore.git_worktree_digest(root) == base


@pytest.mark.parametrize(
    "rel",
    ["src/tfm3lab/a.py", "configs/c.json", "scripts/02b_exp_mtg_benchmark.py", "pyproject.toml"],
)
def test_code_digest_changes_for_result_determining_edits(tmp_path, rel):
    root = _git_repo(tmp_path)
    base = cellstore.git_worktree_digest(root)
    (root / rel).write_text("edited = True\n")
    assert cellstore.git_worktree_digest(root) != base


@pytest.mark.parametrize("rel", ["src/tfm3lab/new.py", "configs/new.json"])
def test_code_digest_changes_for_new_untracked_files(tmp_path, rel):
    root = _git_repo(tmp_path)
    base = cellstore.git_worktree_digest(root)
    (root / rel).write_text("new = 1\n")
    assert cellstore.git_worktree_digest(root) != base
