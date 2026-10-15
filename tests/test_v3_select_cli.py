"""End-to-end test of scripts/13_v3_select.py on a synthetic catalogue and panel
(no real data, no network). Loaded via importlib like the other script tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tfm3lab.data import panel as panel_mod
from tfm3lab.studies import requests as req_mod

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "13_v3_select.py"
AS_OF = "2026-09-04"
FIRST_DAY, LAST_DAY = "2024-02-08", "2026-08-06"
GROUPS = {  # group_id -> (abbreviation, publishedOn)
    11: ("AAA", "2024-03-01T00:00:00"),
    12: ("BBB", "2025-01-10T00:00:00"),
    13: ("CCC", "2026-05-01T00:00:00"),
}


def load_script():
    spec = importlib.util.spec_from_file_location("v3_select_13", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_catalog(directory: Path) -> Path:
    directory.mkdir(parents=True)
    groups = [
        {
            "groupId": gid,
            "name": f"Set {abbr}",
            "abbreviation": abbr,
            "isSupplemental": False,
            "publishedOn": published,
            "categoryId": 1,
        }
        for gid, (abbr, published) in GROUPS.items()
    ]
    (directory / "groups.json").write_text(json.dumps({"results": groups}))
    for gid in GROUPS:
        items = []
        for j in range(8):
            pid = gid * 100 + j
            name = f"Card {j % 5}" if j < 5 else f"Card {j} of {gid}"  # reprints: Card 0..4
            rarity = "R" if j % 2 else "U"
            items.append(
                {
                    "productId": pid,
                    "name": name,
                    "extendedData": [{"name": "Rarity", "value": rarity}],
                }
            )
        items.append({"productId": gid * 100 + 90, "name": "Booster Box", "extendedData": []})
        (directory / f"products-{gid}.json").write_text(json.dumps({"results": items}))
    return directory


def write_panel(directory: Path) -> Path:
    directory.mkdir(parents=True)
    days = pd.date_range(FIRST_DAY, LAST_DAY)
    rng = np.random.default_rng(11)
    skus = [(gid * 100 + j, st) for gid in GROUPS for j in range(8) for st in ("Normal", "Foil")]
    skus += [(gid * 100 + 90, "Normal") for gid in GROUPS]  # sealed: must be ignored
    steps = 0.04 * rng.normal(size=(len(skus), len(days)))
    steps += np.where(rng.random(steps.shape) < 0.01, np.log(1.8), 0.0)
    steps -= np.where(rng.random(steps.shape) < 0.01, np.log(1.8), 0.0)
    price = np.exp(np.log(3.0) + np.cumsum(steps, axis=1)).astype(np.float32)
    price[rng.random(price.shape) < 0.05] = np.nan
    for i, day in enumerate(days):
        if day == pd.Timestamp("2025-03-03"):
            continue
        table = pa.table(
            {
                "product_id": pa.array([p for p, _ in skus], pa.int32()),
                "group_id": pa.array([p // 100 for p, _ in skus], pa.int32()),
                "subtype": pa.array([s for _, s in skus], pa.string()),
                "market": pa.array(price[:, i], pa.float32()),
                "mid": pa.array([None] * len(skus), pa.float32()),
            },
            schema=panel_mod.PANEL_SCHEMA,
        )
        pq.write_table(table, directory / f"{day.date().isoformat()}.parquet")
    return directory


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    base = tmp_path_factory.mktemp("v3in")
    return write_catalog(base / "catalog"), write_panel(base / "panel")


def run(inputs, out_root: Path) -> Path:
    catalog, panel = inputs
    load_script().main(
        [
            "--as-of",
            AS_OF,
            "--panel",
            str(panel),
            "--catalog",
            str(catalog),
            "--out-root",
            str(out_root),
        ]
    )
    return out_root


COMMITTED = [
    "configs/v3/s1_skus.csv",
    "configs/v3/release_events.csv",
    "configs/v3/s2_decisions.csv",
    "configs/v3/s3_events.csv",
    "results/v3_census.json",
]


@pytest.fixture(scope="module")
def first_run(inputs, tmp_path_factory):
    return run(inputs, tmp_path_factory.mktemp("out1"))


def test_writes_every_output_with_the_frozen_columns(first_run):
    root = first_run
    s1 = pd.read_csv(root / "configs/v3/s1_skus.csv")
    assert list(s1.columns) == [
        "event_date",
        "k",
        "product_id",
        "subtype",
        "group_id",
        "group_abbreviation",
        "card_id",
        "rarity",
        "finish",
        "first_printing",
        "floor_1",
    ]
    assert set(s1["k"]) == {7, 14, 28} and set(s1["event_date"]) == {
        "2024-03-01",
        "2025-01-10",
        "2026-05-01",
    }
    assert set(s1["product_id"]) <= {g * 100 + j for g in GROUPS for j in range(8)}
    assert set(s1["finish"]) == {"Normal", "Foil"}
    events = pd.read_csv(root / "configs/v3/release_events.csv")
    assert list(events.columns) == ["event_date", "group_ids", "abbreviations"]
    assert events["abbreviations"].tolist() == ["AAA", "BBB", "CCC"]
    s2 = pd.read_csv(root / "configs/v3/s2_decisions.csv")
    assert list(s2.columns) == [
        "decision_id",
        "product_id",
        "subtype",
        "origin",
        "card_id",
        "rarity",
        "finish",
        "price_band",
    ]
    assert len(s2) > 0 and set(s2["price_band"]) <= {"1-5", "5-20", "20+"}
    assert s2["origin"].between("2024-08-01", "2026-08-06").all()
    s3 = pd.read_csv(root / "configs/v3/s3_events.csv")
    assert list(s3.columns) == [
        "event_id",
        "product_id",
        "subtype",
        "event_date",
        "card_id",
        "rarity",
        "finish",
    ]
    assert len(s3) > 0


def test_nothing_numeric_beyond_ids_and_counts_is_committed(first_run):
    root = first_run
    for name in COMMITTED[:4]:
        cols = pd.read_csv(root / name).columns
        assert not {"price", "ratio", "q", "return", "outcome"} & set(cols)
    census = json.loads((root / "results/v3_census.json").read_text())
    assert set(census) == {"s1", "s2", "s3"}
    text = (root / "results/v3_census.json").read_text()
    assert 'price"' not in text and "ratio" not in text
    assert census["s2"]["M"] == 5500 and census["s2"]["candidates_drawn"] == 5500
    assert census["s2"]["accepted_N"] + census["s2"]["rejected"] == 5500
    assert census["s1"]["n_events"] == 3 and len(census["s1"]["per_event"]) == 9
    assert census["s3"]["N_events"] == census["s3"]["crossings_after_cooldown"]


def test_requests_parquet_matches_the_selected_units(first_run):
    root = first_run
    req = req_mod.read_requests(root / "data/cache/v3_requests.parquet")
    s1 = pd.read_csv(root / "configs/v3/s1_skus.csv")
    s2 = pd.read_csv(root / "configs/v3/s2_decisions.csv")
    s3 = pd.read_csv(root / "configs/v3/s3_events.csv")
    assert (req["study"] == "S1").sum() == len(s1)
    assert (req["study"] == "S2").sum() == len(s2)
    assert (req["study"] == "S3").sum() == len(s3)
    assert set(req.loc[req["study"] == "S2", "request_id"]) == {
        "S2|" + d for d in s2["decision_id"]
    }
    assert set(req.loc[req["study"] == "S3", "request_id"]) == {"S3|" + d for d in s3["event_id"]}
    blocks = req.groupby("study")["block"].agg(set)
    assert blocks["S2"] == {2} and blocks["S3"] == {3} and blocks["S1"] == {1, 4}
    s1_req = req[req["study"] == "S1"].set_index("request_id")
    some = s1_req.index[0]
    k = int(some.split("|")[1][1:])
    assert s1_req.loc[some, "horizon"] == 90 - k
    assert all(c[-1] == 1.0 and 1 <= len(c) <= 512 for c in req["context"])


def test_manifest_records_hashes_seed_and_counts(first_run):
    root = first_run
    (path,) = list((root / "results/manifests").glob("v3-select-2026-09-04-*.json"))
    manifest = json.loads(path.read_text())
    assert manifest["_meta"]["seed"] == 20261014
    assert "git_sha" in manifest["_meta"] and "git_dirty" in manifest["_meta"]
    assert manifest["parameters"]["s2_M"] == 5500
    by_path = {o["path"]: o for o in manifest["outputs"]}
    assert set(by_path) == {*COMMITTED, "data/cache/v3_requests.parquet"}
    for rel, entry in by_path.items():
        assert entry["sha256"] == hashlib.sha256((root / rel).read_bytes()).hexdigest()
    assert manifest["counts"]["requests"] == len(
        req_mod.read_requests(root / "data/cache/v3_requests.parquet")
    )


def test_manifest_records_no_absolute_paths(first_run):
    root = first_run
    (path,) = list((root / "results/manifests").glob("v3-select-2026-09-04-*.json"))
    params = json.loads(path.read_text())["parameters"]
    assert params["panel"] == "panel" and params["catalog"] == "catalog"
    assert not any(str(v).startswith("/") for v in (params["panel"], params["catalog"]))


def test_early_as_of_fails_with_clear_message(inputs, tmp_path):
    catalog, panel = inputs
    with pytest.raises(SystemExit) as exc:
        load_script().main(
            ["--as-of", "2026-08-05", "--panel", str(panel), "--catalog", str(catalog)]
            + ["--out-root", str(tmp_path)]
        )
    assert "last evaluation origin" in str(exc.value)


def test_two_runs_give_byte_identical_outputs(inputs, first_run, tmp_path):
    second = run(inputs, tmp_path)
    for rel in [*COMMITTED, "data/cache/v3_requests.parquet"]:
        assert (first_run / rel).read_bytes() == (second / rel).read_bytes(), rel


def test_requests_file_is_gitignored():
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    done = subprocess.run(
        ["git", "check-ignore", "-q", "data/cache/v3_requests.parquet"], cwd=REPO, check=False
    )
    assert done.returncode == 0
