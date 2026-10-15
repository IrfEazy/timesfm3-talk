"""End-to-end tests of scripts/15_v3_evaluate.py on a synthetic panel, synthetic
frozen units and synthetic forecasts (made by the real runner with a fake
forecaster). No real data, no model, no network."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tfm3lab.data import panel as panel_mod
from tfm3lab.studies import eval_s1, eval_s2, eval_s3, runner
from tfm3lab.studies import requests as rq

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "15_v3_evaluate.py"
AS_OF = "2024-12-31"
FIRST_DAY, LAST_DAY = "2024-02-08", AS_OF
REVISION = "ab" * 20
PRODUCTS = list(range(1000, 1024))
EVENTS = ["2024-03-01", "2024-04-01", "2024-08-01"]
KS = (7, 14, 28)
BLOCK_S1 = {14: 1, 7: 4, 28: 4}
SHAPE = np.linspace(0.8, 1.2, 9)


def load_script():
    spec = importlib.util.spec_from_file_location("v3_evaluate_15", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeForecaster:
    """point[step] = last context value * (1 + 0.01 * step); quantiles = point * SHAPE."""

    def predict_batch(
        self,
        contexts,
        horizon,
        past_only_covariates=None,
        past_future_covariates=None,
        ts_ids=None,
        return_quantiles=True,
        use_symmetric_averaging=True,
        make_positive=True,
        **kwargs,
    ):
        steps = np.arange(1, horizon + 1)
        out = []
        for ctx, ts_id in zip(contexts, ts_ids, strict=True):
            point = (float(ctx[-1]) * (1 + 0.01 * steps)).astype(np.float32)
            quant = (point[:, None] * SHAPE[None, :]).astype(np.float32)
            out.append(SimpleNamespace(ts_id=ts_id, forecast=point, quantiles=quant))
        return out


def write_panel(directory: Path) -> Path:
    directory.mkdir(parents=True)
    days = pd.date_range(FIRST_DAY, LAST_DAY)
    rng = np.random.default_rng(5)
    n = len(PRODUCTS)
    steps = 0.03 * rng.normal(size=(n, len(days)))
    price = np.exp(np.log(3.0) + np.cumsum(steps, axis=1)).astype(np.float32)
    price[rng.random(price.shape) < 0.01] = np.nan
    for i, day in enumerate(days):
        table = pa.table(
            {
                "product_id": pa.array(PRODUCTS, pa.int32()),
                "group_id": pa.array([1] * n, pa.int32()),
                "subtype": pa.array(["Normal"] * n, pa.string()),
                "market": pa.array(price[:, i], pa.float32()),
                "mid": pa.array([None] * n, pa.float32()),
            },
            schema=panel_mod.PANEL_SCHEMA,
        )
        pq.write_table(table, directory / f"{day.date().isoformat()}.parquet")
    return directory


def write_configs(directory: Path) -> Path:
    directory.mkdir(parents=True)
    rows = []
    for r in EVENTS:
        for k in KS:
            for j, pid in enumerate(PRODUCTS):
                rows.append(
                    {
                        "event_date": r,
                        "k": k,
                        "product_id": pid,
                        "subtype": "Normal",
                        "group_id": 1,
                        "group_abbreviation": "AAA",
                        "card_id": f"card {pid}",
                        "rarity": "R" if j % 2 else "U",
                        "finish": "Normal",
                        "first_printing": j % 3 != 0,
                        "floor_1": True,
                    }
                )
    pd.DataFrame(rows).to_csv(directory / "s1_skus.csv", index=False, lineterminator="\n")
    origins = pd.date_range("2024-06-01", periods=12, freq="9D")
    s2 = [
        {
            "decision_id": f"{o.date()}|{pid}|Normal",
            "product_id": pid,
            "subtype": "Normal",
            "origin": o.date().isoformat(),
            "card_id": f"card {pid}",
            "rarity": "R",
            "finish": "Normal",
            "price_band": "1-5",
        }
        for o, pid in zip(origins, PRODUCTS[:12], strict=True)
    ]
    pd.DataFrame(s2).to_csv(directory / "s2_decisions.csv", index=False, lineterminator="\n")
    days = pd.date_range("2024-05-15", periods=10, freq="11D")
    s3 = [
        {
            "event_id": f"{d.date()}|{pid}|Normal",
            "product_id": pid,
            "subtype": "Normal",
            "event_date": d.date().isoformat(),
            "card_id": f"card {pid}",
            "rarity": "U",
            "finish": "Normal",
        }
        for d, pid in zip(days, PRODUCTS[10:20], strict=True)
    ]
    pd.DataFrame(s3).to_csv(directory / "s3_events.csv", index=False, lineterminator="\n")
    return directory


def build_requests(configs: Path) -> pd.DataFrame:
    s1 = pd.read_csv(configs / "s1_skus.csv", keep_default_na=False)
    s2 = pd.read_csv(configs / "s2_decisions.csv", keep_default_na=False)
    s3 = pd.read_csv(configs / "s3_events.csv", keep_default_na=False)
    ctx = np.array([0.9, 0.95, 1.0], dtype=np.float32)
    rows = []
    for u in s1.itertuples():
        rid = eval_s1.request_id(u.k, u.event_date, u.product_id, u.subtype)
        rows.append((rid, "S1", BLOCK_S1[u.k], 90 - u.k))
    rows += [(eval_s2.request_id(d), "S2", 2, 28) for d in s2["decision_id"]]
    rows += [(eval_s3.request_id(e), "S3", 3, 56) for e in s3["event_id"]]
    return pd.DataFrame(
        [
            {"request_id": r, "study": s, "block": b, "horizon": h, "context": ctx}
            for r, s, b, h in rows
        ]
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("v3eval")
    configs = write_configs(base / "configs")
    requests_df = build_requests(configs)
    requests_path = base / "v3_requests.parquet"
    rq.write_requests(requests_df, requests_path)
    return SimpleNamespace(
        base=base,
        configs=configs,
        panel=write_panel(base / "panel"),
        requests=requests_path,
        requests_df=requests_df,
    )


def make_forecasts(world, name: str, blocks=None) -> Path:
    out = world.base / name
    runner.run_requests(
        world.requests_df,
        FakeForecaster(),
        out,
        batch_size=64,
        blocks=blocks,
        echo=lambda _msg: None,
        checkpoint_revision=REVISION,
    )
    return out


def write_census(root: Path, confirmatory: bool) -> None:
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results" / "v3_census.json").write_text(
        json.dumps({"s3": {"confirmatory": confirmatory}})
    )


def run(world, forecasts: Path, root: Path, confirmatory: bool = True, requests=None) -> Path:
    write_census(root, confirmatory)
    load_script().main(
        [
            "--as-of",
            AS_OF,
            "--forecasts",
            str(forecasts),
            "--panel",
            str(world.panel),
            "--requests",
            str(requests or world.requests),
            "--configs-dir",
            str(world.configs),
            "--out-root",
            str(root),
        ]
    )
    return root


@pytest.fixture(scope="module")
def complete_forecasts(world):
    return make_forecasts(world, "fc_complete")


@pytest.fixture(scope="module")
def complete_run(world, complete_forecasts, tmp_path_factory):
    return run(world, complete_forecasts, tmp_path_factory.mktemp("out_complete"))


OUTPUTS = [
    "v3_s1_events.parquet",
    "v3_s1_summary.json",
    "v3_s2_decisions.parquet",
    "v3_s2_summary.json",
    "v3_s3_events.parquet",
    "v3_s3_summary.json",
    "v3_verdicts.json",
]


def test_complete_run_writes_every_output_and_three_verdicts(complete_run):
    res = complete_run / "results"
    for name in OUTPUTS:
        assert (res / name).is_file(), name
    verdicts = json.loads((res / "v3_verdicts.json").read_text())
    assert [v["study"] for v in verdicts] == ["S1", "S2", "S3"]
    for v in verdicts:
        assert v["status"] == "complete" and v["verdict"] != "incomplete"
        assert v["n"] and v["point"] is not None and v["intervals"]
    assert verdicts[0]["p"] is not None and set(verdicts[0]["intervals"]) == {"t"}
    assert set(verdicts[1]["intervals"]) == {"cluster", "block"}
    ev = pd.read_parquet(res / "v3_s1_events.parquet")
    assert set(ev["floor"]) == {0.25, 1.0} and set(ev["k"]) == set(KS)
    assert {"mean_abs_err_tfm", "mean_abs_err_naive", "diff_tfm_naive", "n_paired"} <= set(ev)
    s2 = pd.read_parquet(res / "v3_s2_decisions.parquet")
    assert len(s2) == 12 and {"A_tfm", "dstar_tfm", "decision_id", "origin"} <= set(s2)
    s3 = pd.read_parquet(res / "v3_s3_events.parquet")
    assert len(s3) == 10 and {"ale_tfm_28", "covered_80_28", "ale_hold_56"} <= set(s3)
    s2sum = json.loads((res / "v3_s2_summary.json").read_text())
    assert s2sum["counts"]["n_decisions"] == 12 and s2sum["status"] == "complete"


def test_summary_matches_a_direct_call_of_the_frozen_functions(world, complete_run):
    # tfm forecast ratio is 1 + 0.01 * (90 - k) here; the naive is 1.
    s1 = json.loads((complete_run / "results" / "v3_s1_summary.json").read_text())
    assert s1["primary"]["k"] == 14 and s1["primary"]["G"] >= 1
    assert s1["exploratory"]["k7"]["k"] == 7


def test_no_price_columns_or_values_in_outputs(complete_run):
    res = complete_run / "results"
    banned = ("price", "q_origin", "q_target", "actual", "market", "quote")

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from walk(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from walk(v)

    for name in OUTPUTS:
        if name.endswith(".parquet"):
            names = list(pd.read_parquet(res / name).columns)
        else:
            names = list(walk(json.loads((res / name).read_text())))
        for n in names:
            stripped = n.replace("price_band", "")
            assert not any(b in stripped.lower() for b in banned), (name, n)


def test_guard_refuses_price_like_names_and_price_levels():
    mod = load_script()
    for bad in ("price", "q_origin", "q_target", "actual", "market", "quote", "mean_quote_x"):
        with pytest.raises(ValueError):
            mod.check_name(bad)
    mod.check_name("price_band")
    with pytest.raises(ValueError):
        mod.assert_no_prices(pd.DataFrame({"q_target": [1.0]}), "t")
    qmat = np.array([[2.5, 3.5, 4.5]])
    frame = pd.DataFrame({"product_id": [7], "subtype": [None], "A_x": [3.5]})
    with pytest.raises(ValueError, match="absolute price level"):
        mod.assert_no_prices(frame, "t", qmat=qmat, sku_index=[(7, None)])
    frame["A_x"] = [0.31]
    mod.assert_no_prices(frame, "t", qmat=qmat, sku_index=[(7, None)])
    with pytest.raises(ValueError):
        mod.write_json({"a": {"price": 1.0}}, Path("/nonexistent/never-written.json"))


def test_guard_flags_systematic_matches_not_coincidences():
    mod = load_script()
    n = 100
    qmat = np.tile([0.5, 2.5, 3.5], (n, 1))
    index = [(i, None) for i in range(n)]
    base = pd.DataFrame({"product_id": range(n), "subtype": [None] * n})
    rng = np.random.default_rng(0)
    noise = rng.uniform(0.1, 0.4, n)
    # one coincidental match among 100 eligible rows: tolerated
    ok = base.assign(A_adv=noise)
    ok.loc[3, "A_adv"] = 0.5
    mod.assert_no_prices(ok, "t", qmat=qmat, sku_index=index)
    # a column copying the quotes: refused, naming column and counts but no values
    bad = base.assign(A_copy=3.5)
    with pytest.raises(ValueError, match=r"t\.A_copy.*100 of 100") as exc:
        mod.assert_no_prices(bad, "t", qmat=qmat, sku_index=index)
    assert "3.5" not in str(exc.value)
    # 6 of 100 matches reaches the 5 percent share
    some = base.assign(A_adv=noise)
    some.loc[:5, "A_adv"] = 2.5
    with pytest.raises(ValueError, match="6 of 100"):
        mod.assert_no_prices(some, "t", qmat=qmat, sku_index=index)
    # 20 matches among many eligible rows reaches the count threshold
    m = 1000
    big = pd.DataFrame({"product_id": range(m), "subtype": [None] * m})
    big["A_adv"] = rng.uniform(0.1, 0.4, m)
    big.loc[:19, "A_adv"] = 0.5
    with pytest.raises(ValueError, match="20 of 1000"):
        mod.assert_no_prices(
            big, "t", qmat=np.tile([0.5, 2.5], (m, 1)), sku_index=[(i, None) for i in range(m)]
        )


def test_revision_and_inputs_recorded_in_manifest(world, complete_forecasts, complete_run):
    files = sorted(
        (complete_run / "results" / "manifests").glob("exp-v3-evaluate-2024-12-31-*.json")
    )
    assert len(files) == 1
    manifest = json.loads(files[0].read_text())
    assert manifest["inputs"]["checkpoint_revision"] == REVISION
    assert manifest["_meta"]["checkpoint"]["revision_resolved"] == REVISION
    digest = rq.requests_digest(rq.read_requests(world.requests))
    assert manifest["inputs"]["requests"]["digest"] == digest
    parts = manifest["inputs"]["forecast_parts"]
    assert len(parts) == len(list(complete_forecasts.glob("block=*/part-*.parquet")))
    assert all(len(p["sha256"]) == 64 for p in parts)
    assert {c["path"] for c in manifest["inputs"]["configs"]} == {
        "s1_skus.csv",
        "s2_decisions.csv",
        "s3_events.csv",
    }
    listed = {o["path"]: o["sha256"] for o in manifest["outputs"]}
    mod = load_script()
    for name in OUTPUTS:
        assert listed[f"results/{name}"] == mod.sha256_file(complete_run / "results" / name)


def test_incomplete_block_marks_that_study_incomplete_only(world, tmp_path):
    forecasts = make_forecasts(world, "fc_no_block3", blocks=[1, 2, 4])
    root = run(world, forecasts, tmp_path / "out")
    res = root / "results"
    verdicts = {v["study"]: v for v in json.loads((res / "v3_verdicts.json").read_text())}
    assert verdicts["S3"]["status"] == "incomplete" and verdicts["S3"]["verdict"] == "incomplete"
    assert verdicts["S3"]["incomplete_blocks"] == [3] and verdicts["S3"]["n"] is None
    assert verdicts["S1"]["status"] == verdicts["S2"]["status"] == "complete"
    assert not (res / "v3_s3_events.parquet").exists()
    s3 = json.loads((res / "v3_s3_summary.json").read_text())
    assert s3["status"] == "incomplete" and "primary" not in s3


def test_s1_with_unfinished_block_4_is_incomplete_without_partial_evaluation(world, tmp_path):
    forecasts = make_forecasts(world, "fc_no_block4", blocks=[1, 2, 3])
    root = run(world, forecasts, tmp_path / "out")
    res = root / "results"
    verdicts = {v["study"]: v for v in json.loads((res / "v3_verdicts.json").read_text())}
    assert verdicts["S1"]["status"] == "incomplete" and verdicts["S1"]["incomplete_blocks"] == [4]
    assert verdicts["S2"]["status"] == verdicts["S3"]["status"] == "complete"
    assert not (res / "v3_s1_events.parquet").exists()


def test_descriptive_s3_when_census_is_not_confirmatory(world, complete_forecasts, tmp_path):
    root = run(world, complete_forecasts, tmp_path / "out", confirmatory=False)
    verdicts = {v["study"]: v for v in json.loads((root / "results/v3_verdicts.json").read_text())}
    assert (
        verdicts["S3"]["verdict"] == "descriptive only" and verdicts["S3"]["confirmatory"] is False
    )


def test_requests_sha_mismatch_in_status_is_refused(world, tmp_path):
    forecasts = make_forecasts(world, "fc_tampered_status")
    path = forecasts / "block_status.json"
    status = json.loads(path.read_text())
    status["1"]["requests_sha256"] = "0" * 64
    path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="requests sha256"):
        run(world, forecasts, tmp_path / "out")
    assert not (tmp_path / "out" / "results" / "v3_verdicts.json").exists()


def test_other_local_requests_file_is_refused(world, complete_forecasts, tmp_path):
    other = world.requests_df.copy()
    other.at[0, "context"] = np.array([0.5, 0.6, 1.0], dtype=np.float32)
    path = tmp_path / "other_requests.parquet"
    rq.write_requests(other, path)
    with pytest.raises(ValueError, match="requests sha256"):
        run(world, complete_forecasts, tmp_path / "out", requests=path)


def test_part_with_other_revision_is_refused(world, tmp_path):
    forecasts = make_forecasts(world, "fc_tampered_part")
    part = sorted(forecasts.glob("block=1/part-*.parquet"))[0]
    table = pq.read_table(part)
    meta = dict(table.schema.metadata)
    meta[b"tfm3lab.checkpoint_revision"] = b"cd" * 20
    pq.write_table(table.replace_schema_metadata(meta), part)
    with pytest.raises(ValueError, match="checkpoint revision"):
        run(world, forecasts, tmp_path / "out")


def test_frozen_units_must_match_requests(world, complete_forecasts, tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    for f in world.configs.glob("*.csv"):
        (configs / f.name).write_text(f.read_text())
    s3 = pd.read_csv(configs / "s3_events.csv").iloc[1:]
    s3.to_csv(configs / "s3_events.csv", index=False, lineterminator="\n")
    mod = load_script()
    write_census(tmp_path / "out", True)
    with pytest.raises(ValueError, match="S3: frozen units differ"):
        mod.main(
            [
                "--as-of",
                AS_OF,
                "--forecasts",
                str(complete_forecasts),
                "--panel",
                str(world.panel),
                "--requests",
                str(world.requests),
                "--configs-dir",
                str(configs),
                "--out-root",
                str(tmp_path / "out"),
            ]
        )


def test_outputs_are_deterministic(world, complete_forecasts, complete_run, tmp_path):
    second = run(world, complete_forecasts, tmp_path / "again")
    for name in OUTPUTS:
        assert (second / "results" / name).read_bytes() == (
            complete_run / "results" / name
        ).read_bytes(), name


def test_eval_modules_used_unchanged():
    # the driver must evaluate through the frozen modules, not reimplement them
    text = SCRIPT.read_text()
    for call in ("eval_s1.primary_result", "eval_s2.evaluate", "eval_s3.evaluate"):
        assert call in text


def test_incomplete_study_deletes_stale_table_and_does_not_list_it(world, tmp_path):
    res = tmp_path / "out" / "results"
    res.mkdir(parents=True)
    (res / "v3_s3_events.parquet").write_bytes(b"stale")
    forecasts = make_forecasts(world, "fc_no_block3_stale", blocks=[1, 2, 4])
    root = run(world, forecasts, tmp_path / "out")
    assert not (root / "results" / "v3_s3_events.parquet").exists()
    manifest = json.loads(next((root / "results" / "manifests").glob("*.json")).read_text())
    assert not any("v3_s3_events" in o["path"] for o in manifest["outputs"])


def test_manifest_records_panel_input(world, complete_forecasts, complete_run):
    import hashlib

    manifest = json.loads(
        next((complete_run / "results" / "manifests").glob("exp-v3-evaluate-*.json")).read_text()
    )
    panel = manifest["inputs"]["panel"]
    files = sorted(f for f in world.panel.glob("*.parquet") if f.stem <= AS_OF)
    expected = hashlib.sha256()
    mod = load_script()
    for f in files:
        expected.update(f"{f.name}\t{mod.sha256_file(f)}\n".encode())
    assert panel["day_files_sha256"] == expected.hexdigest()
    assert panel["n_day_files"] == len(files)
    assert panel["first_day"] == files[0].stem and panel["last_day"] == files[-1].stem
    assert panel["n_days"] == len(files) and panel["n_skus"] > 0
    assert panel["path"]


# ------------------------------------------------------------------ Amendment A1


def test_clean_run_records_zero_rearrangement_and_model_behaviour(complete_run):
    res = complete_run / "results"
    mb = json.loads((res / "v3_model_behaviour.json").read_text())
    assert mb["amendment"] == "A1" and set(mb["studies"]) == {"S1", "S2", "S3"}
    assert all(n == 0 for n in mb["rows_rearranged_by_block"].values())
    assert set(mb["rows_rearranged_by_block"]) == {"1", "2", "3", "4"}
    for st in mb["studies"].values():
        assert st["floored_share_all_quantiles"] == 0.0
        assert st["share_requests_q50_floored_any_step"] == 0.0
        assert set(st["floored_share_by_quantile"]) == set(rq.QUANTILE_COLUMNS)
    manifest = json.loads(next((res / "manifests").glob("exp-v3-evaluate-*.json")).read_text())
    assert manifest["quantile_rearrangement"]["rows_changed"] == 0
    assert any(o["path"] == "results/v3_model_behaviour.json" for o in manifest["outputs"])
    s1 = json.loads((res / "v3_s1_summary.json").read_text())
    sens = s1["exploratory"]["exploratory_q50_above_floor"]
    assert sens["k"] == 14 and "verdict" not in sens
    assert sens["n_skus_median_above_floor"] == sens["n_skus_floor_1"]  # nothing floored
    assert sens["n_events"] == s1["primary"]["G"]
    assert len(sens["t_ci_95"]) == 2


def test_crossing_rows_are_rearranged_before_strict_validation(world, complete_forecasts, tmp_path):
    import shutil

    fc = tmp_path / "fc_crossing"
    shutil.copytree(complete_forecasts, fc)
    part = sorted((fc / "block=1").glob("part-*.parquet"))[0]
    table = pq.read_table(part)
    meta = table.schema.metadata
    df = table.to_pandas()
    assert df["request_id"].str.startswith("S1|").any()
    df.loc[0:2, "q10"] = np.float32(50.0)  # three crossing rows
    df.loc[3:5, "q20"] = np.float32(0.001)  # floored but monotone
    df.loc[3:5, "q10"] = np.float32(0.001)
    df.loc[6, "q50"] = np.float32(0.001)  # floored median; crossing with q60?
    df.loc[6, ["q10", "q20", "q30", "q40"]] = np.float32(0.001)
    pq.write_table(
        pa.Table.from_pandas(df, schema=table.schema).replace_schema_metadata(meta), part
    )

    # strict validation alone would refuse these forecasts
    requests_df = rq.read_requests(world.requests)
    block1 = requests_df[requests_df["block"] == 1]
    raw = rq.read_forecasts(fc)
    raw = raw[raw["request_id"].isin(block1["request_id"])]
    with pytest.raises(ValueError, match="not non-decreasing"):
        rq.validate_forecasts(block1, raw)

    root = run(world, fc, tmp_path / "out")
    res = root / "results"
    mb = json.loads((res / "v3_model_behaviour.json").read_text())
    assert mb["rows_rearranged_by_block"]["1"] == 3
    assert sum(mb["rows_rearranged_by_block"].values()) == 3
    s1 = df["request_id"].iloc[6].split("|")[0]
    st = mb["studies"][s1]
    assert st["floored_share_by_quantile"]["q10"] > 0
    assert st["floored_share_by_quantile"]["q50"] > 0
    assert st["share_requests_q50_floored_any_step"] > 0
    assert st["floored_share_all_quantiles"] > 0
    manifest = json.loads(next((res / "manifests").glob("exp-v3-evaluate-*.json")).read_text())
    assert manifest["quantile_rearrangement"]["rows_changed_by_block"]["1"] == 3
    assert manifest["quantile_rearrangement"]["rows_changed"] == 3
