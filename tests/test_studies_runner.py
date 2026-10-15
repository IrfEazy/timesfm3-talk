"""Offline tests of the v3 forecast runner (tfm3lab.studies.runner) and of
scripts/14_v3_forecast.py. No GPU, no model download: every forecaster here is
a fake implementing the Forecaster protocol."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tfm3lab import config
from tfm3lab.studies import requests as rq
from tfm3lab.studies import runner

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "14_v3_forecast.py"
QUANTILE_SHAPE = np.linspace(0.8, 1.2, 9)  # monotone multipliers of the point forecast
REVISION = "ab" * 20  # a resolved checkpoint commit sha (stands in for the pinned one)
OTHER_REVISION = "cd" * 20


def run_requests(*args, **kwargs):
    """runner.run_requests at REVISION unless a test names another revision."""
    kwargs.setdefault("checkpoint_revision", REVISION)
    return runner.run_requests(*args, **kwargs)


# ------------------------------------------------------------------ fakes and fixtures


class FakeForecaster:
    """Deterministic Forecaster: point[step] = last context value * (1 + 0.01 * step),
    quantiles = point * QUANTILE_SHAPE. `corrupt(ts_id, point, quantiles)` may damage
    the arrays; `fail_on_call=n` raises on the n-th call (a simulated interruption)."""

    def __init__(self, *, fail_on_call: int | None = None, reverse: bool = False, corrupt=None):
        self.calls: list[dict] = []
        self.fail_on_call = fail_on_call
        self.reverse = reverse
        self.corrupt = corrupt

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
        self.calls.append(
            {
                "horizon": horizon,
                "ts_ids": list(ts_ids),
                "n_contexts": len(contexts),
                "return_quantiles": return_quantiles,
                "use_symmetric_averaging": use_symmetric_averaging,
                "make_positive": make_positive,
            }
        )
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("simulated interruption")
        outputs = []
        steps = np.arange(1, horizon + 1)
        for ctx, ts_id in zip(contexts, ts_ids, strict=True):
            point = (float(ctx[-1]) * (1 + 0.01 * steps)).astype(np.float32)
            quantiles = (point[:, None] * QUANTILE_SHAPE[None, :]).astype(np.float32)
            if self.corrupt is not None:
                self.corrupt(ts_id, point, quantiles)
            outputs.append(SimpleNamespace(ts_id=ts_id, forecast=point, quantiles=quantiles))
        if self.reverse:
            outputs.reverse()
        return outputs

    @property
    def all_ids(self) -> list[str]:
        return [i for call in self.calls for i in call["ts_ids"]]


def make_requests(spec: list[tuple[str, int, int]]) -> pd.DataFrame:
    """spec = [(request_id, block, horizon)]; the last context value is 1 + index / 100
    so every request has its own recognisable forecast."""
    rows = []
    for i, (rid, block, horizon) in enumerate(spec):
        ctx = np.concatenate([np.linspace(0.9, 1.0, 5), [1.0 + i / 100]]).astype(np.float32)
        rows.append(
            {
                "request_id": rid,
                "study": ("S1", "S2", "S3")[i % 3],
                "block": block,
                "horizon": horizon,
                "context": ctx,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def quiet():
    lines: list[str] = []
    return lines.append


@pytest.fixture
def requests_df() -> pd.DataFrame:
    # deliberately shuffled across blocks and horizons
    return make_requests(
        [
            ("c3", 2, 7),
            ("a1", 3, 3),
            ("b2", 1, 7),
            ("d4", 1, 3),
            ("e5", 2, 3),
            ("f6", 1, 3),
            ("g7", 3, 7),
            ("h8", 1, 7),
        ]
    )


def parts_of(out: Path, block: int) -> list[Path]:
    return sorted((out / f"block={block}").glob("part-*.parquet"))


# ------------------------------------------------------------------ ordering, grouping, defaults


def test_blocks_run_in_ascending_order_and_batches_are_horizon_grouped(
    requests_df, tmp_path, quiet
):
    fake = FakeForecaster()
    status = run_requests(requests_df, fake, tmp_path, batch_size=2, echo=quiet)

    block_of = dict(zip(requests_df["request_id"], requests_df["block"], strict=True))
    horizon_of = dict(zip(requests_df["request_id"], requests_df["horizon"], strict=True))
    call_blocks = [{block_of[i] for i in call["ts_ids"]} for call in fake.calls]
    assert all(len(b) == 1 for b in call_blocks)  # a batch never straddles blocks
    flat = [next(iter(b)) for b in call_blocks]
    assert flat == sorted(flat) and set(flat) == {1, 2, 3}

    for call in fake.calls:
        horizons = {horizon_of[i] for i in call["ts_ids"]}
        assert horizons == {call["horizon"]}  # no batch decodes more steps than it needs
        assert call["ts_ids"] == sorted(call["ts_ids"])  # request_id order inside a batch
        assert call["n_contexts"] == len(call["ts_ids"]) <= 2

    # block 1 holds b2,h8 (h=7) and d4,f6 (h=3): horizon groups ascending, ids sorted within
    block1 = [c["ts_ids"] for c in fake.calls[:2]]
    assert block1 == [["d4", "f6"], ["b2", "h8"]]
    assert sorted(fake.all_ids) == sorted(requests_df["request_id"])
    assert len(fake.all_ids) == len(set(fake.all_ids))
    assert {k: v["status"] for k, v in status.items()} == dict.fromkeys("123", "complete")


def test_batch_size_splits_a_horizon_group(tmp_path, quiet):
    df = make_requests([(f"r{i}", 1, 4) for i in range(5)])
    fake = FakeForecaster()
    run_requests(df, fake, tmp_path, batch_size=2, echo=quiet)
    assert [len(c["ts_ids"]) for c in fake.calls] == [2, 2, 1]
    assert [c["ts_ids"] for c in fake.calls] == [["r0", "r1"], ["r2", "r3"], ["r4"]]
    assert [p.name for p in parts_of(tmp_path, 1)] == [f"part-{n:06d}.parquet" for n in range(3)]


def test_official_defaults_are_passed_to_the_model(requests_df, tmp_path, quiet):
    fake = FakeForecaster()
    run_requests(requests_df, fake, tmp_path, echo=quiet)
    assert fake.calls
    for call in fake.calls:
        assert call["use_symmetric_averaging"] is True
        assert call["make_positive"] is True
        assert call["return_quantiles"] is True


def test_blocks_argument_selects_blocks_and_reports_pending_ones(requests_df, tmp_path):
    fake, lines = FakeForecaster(), []
    status = run_requests(requests_df, fake, tmp_path, blocks=[3, 1], echo=lines.append)
    block_of = dict(zip(requests_df["request_id"], requests_df["block"], strict=True))
    assert {block_of[i] for i in fake.all_ids} == {1, 3}  # ascending whatever the order given
    assert [block_of[c["ts_ids"][0]] for c in fake.calls] == sorted(
        block_of[c["ts_ids"][0]] for c in fake.calls
    )
    assert status["2"] == {
        "requests": 2,
        "done": 0,
        "floored_values": 0,
        "seconds": 0.0,
        "status": "pending",
        "requests_sha256": rq.requests_digest(requests_df),
        "checkpoint_revision": REVISION,
        "use_symmetric_averaging": True,
        "make_positive": True,
    }
    assert not (tmp_path / "block=2").exists()

    lines.clear()
    run_requests(requests_df, FakeForecaster(), tmp_path, blocks=[4], echo=lines.append)
    assert any("block 4: no requests" in line for line in lines)
    with pytest.raises(ValueError, match="outside 1..4"):
        run_requests(requests_df, FakeForecaster(), tmp_path, blocks=[5])


def test_results_are_matched_by_ts_id_not_by_position(requests_df, tmp_path, quiet):
    run_requests(requests_df, FakeForecaster(reverse=True), tmp_path, batch_size=3, echo=quiet)
    forecasts = rq.read_forecasts(tmp_path)
    last = {
        rid: float(ctx[-1])
        for rid, ctx in zip(requests_df["request_id"], requests_df["context"], strict=True)
    }
    assert len(forecasts) == int(requests_df["horizon"].sum())
    assert tuple(f"q{k}" for k in range(10, 100, 10)) == rq.QUANTILE_COLUMNS
    for row in forecasts.itertuples():
        assert row.point == pytest.approx(last[row.request_id] * (1 + 0.01 * row.step), rel=1e-6)
    # every one of the nine columns q10..q90 carries its own level of the fake's grid
    point = forecasts["point"].to_numpy(dtype=np.float64)
    for name, shape in zip(rq.QUANTILE_COLUMNS, QUANTILE_SHAPE, strict=True):
        np.testing.assert_allclose(
            forecasts[name].to_numpy(dtype=np.float64), shape * point, rtol=1e-6, err_msg=name
        )


def test_forecast_files_follow_the_frozen_schema_and_validate(requests_df, tmp_path, quiet):
    run_requests(requests_df, FakeForecaster(), tmp_path, batch_size=3, echo=quiet)
    for part in tmp_path.glob("block=*/part-*.parquet"):
        schema = pq.read_schema(part)
        assert schema.names == list(rq.FORECAST_COLUMNS)
        assert [str(t) for t in schema.types] == ["string", "int16"] + ["float"] * 10
    rq.validate_forecasts(requests_df, rq.read_forecasts(tmp_path))
    assert not list(tmp_path.rglob("*.tmp-*"))  # every write was atomic


def test_runner_rejects_bad_arguments(requests_df, tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        run_requests(requests_df, FakeForecaster(), tmp_path, batch_size=0)
    bad = requests_df.assign(block=[1, 1, 1, 1, 1, 1, 1, 9])
    with pytest.raises(ValueError, match="h8"):
        run_requests(bad, FakeForecaster(), tmp_path)


def test_wrong_model_output_shape_fails_loudly(tmp_path, quiet):
    class Short(FakeForecaster):
        def predict_batch(self, contexts, horizon, **kw):
            return super().predict_batch(contexts, horizon - 1, **kw)

    df = make_requests([("r0", 1, 3)])
    with pytest.raises(ValueError, match="expected"):
        run_requests(df, Short(), tmp_path, echo=quiet)
    assert not parts_of(tmp_path, 1)


# ------------------------------------------------------------------ flooring


def damage(ts_id, point, quantiles):
    """Only for request 'bad': NaN at step 1's point, non-positive q10/q20 at step 1."""
    if ts_id == "bad":
        point[0] = np.nan
        quantiles[0, 0] = -1.0
        quantiles[0, 1] = 0.0


def test_non_finite_and_non_positive_values_are_floored_and_counted(tmp_path, quiet):
    df = make_requests([("bad", 1, 3), ("good", 1, 3)])
    status = run_requests(df, FakeForecaster(corrupt=damage), tmp_path, echo=quiet)
    assert status["1"]["floored_values"] == 3  # NaN point + two quantiles
    assert status["1"]["status"] == "complete"

    forecasts = rq.read_forecasts(tmp_path)
    bad = forecasts[forecasts["request_id"] == "bad"].set_index("step")
    assert bad.loc[1, "point"] == pytest.approx(runner.FLOOR_VALUE)
    assert bad.loc[1, "q10"] == pytest.approx(runner.FLOOR_VALUE)
    assert bad.loc[1, "q20"] == pytest.approx(runner.FLOOR_VALUE)
    assert bad.loc[1, "q30"] > 0.5  # untouched
    assert np.isfinite(forecasts.drop(columns="request_id").to_numpy()).all()
    assert (forecasts[["point", *rq.QUANTILE_COLUMNS]].to_numpy() > 0).all()
    good = forecasts[forecasts["request_id"] == "good"]
    assert (good[["point", *rq.QUANTILE_COLUMNS]].to_numpy() >= 0.8).all()  # no flooring
    rq.validate_forecasts(df, forecasts)

    saved = json.loads((tmp_path / runner.STATUS_FILE).read_text())
    assert saved["1"]["floored_values"] == 3


def test_floored_count_survives_a_resume(tmp_path, quiet):
    df = make_requests([("bad", 1, 3), ("r1", 1, 3), ("r2", 1, 3)])
    fake = FakeForecaster(corrupt=damage, fail_on_call=2)
    with pytest.raises(RuntimeError, match="simulated"):
        run_requests(df, fake, tmp_path, batch_size=2, echo=quiet)  # batch 1 = bad, r1
    status = run_requests(df, FakeForecaster(corrupt=damage), tmp_path, echo=quiet)
    assert status["1"]["floored_values"] == 3 and status["1"]["done"] == 3


# ------------------------------------------------------------------ resume


def test_resume_after_an_interruption_does_not_duplicate_rows(requests_df, tmp_path, quiet):
    first = FakeForecaster(fail_on_call=2)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_requests(requests_df, first, tmp_path, batch_size=2, echo=quiet)
    assert len(first.calls) == 2  # one batch committed, the second died mid-block 1
    done_first = first.calls[0]["ts_ids"]
    assert done_first == ["d4", "f6"]

    interrupted = json.loads((tmp_path / runner.STATUS_FILE).read_text())
    assert interrupted["1"]["status"] == "partial" and interrupted["1"]["done"] == 2
    assert interrupted["2"]["status"] == interrupted["3"]["status"] == "pending"
    assert len(list(tmp_path.glob("block=*/part-*.parquet"))) == 1

    second = FakeForecaster()
    status = run_requests(requests_df, second, tmp_path, batch_size=2, echo=quiet)
    assert not set(second.all_ids) & set(done_first)  # committed requests are never redone
    assert sorted(done_first + second.all_ids) == sorted(requests_df["request_id"])
    assert len(second.all_ids) == len(set(second.all_ids))
    assert {v["status"] for v in status.values()} == {"complete"}

    forecasts = rq.read_forecasts(tmp_path)
    assert not forecasts.duplicated(["request_id", "step"]).any()
    rq.validate_forecasts(requests_df, forecasts)
    # the new part continues the numbering of block 1 instead of overwriting part 0
    assert [p.name for p in parts_of(tmp_path, 1)] == [f"part-{n:06d}.parquet" for n in range(2)]
    assert second.calls[0]["ts_ids"] == ["b2", "h8"]  # resumes exactly where it stopped


def test_resume_mid_block_continues_the_part_numbering(tmp_path, quiet):
    df = make_requests([(f"r{i}", 1, 2) for i in range(6)])
    with pytest.raises(RuntimeError):
        run_requests(df, FakeForecaster(fail_on_call=2), tmp_path, batch_size=2, echo=quiet)
    assert len(parts_of(tmp_path, 1)) == 1
    run_requests(df, FakeForecaster(), tmp_path, batch_size=2, echo=quiet)
    assert [p.name for p in parts_of(tmp_path, 1)] == [f"part-{n:06d}.parquet" for n in range(3)]


def test_resume_with_everything_done_runs_no_inference(requests_df, tmp_path, quiet):
    run_requests(requests_df, FakeForecaster(), tmp_path, echo=quiet)
    before = sorted((p.name, p.stat().st_mtime_ns) for p in tmp_path.rglob("part-*.parquet"))
    again = FakeForecaster()
    status = run_requests(requests_df, again, tmp_path, echo=quiet)
    assert again.calls == []
    assert {v["status"] for v in status.values()} == {"complete"}
    assert (
        sorted((p.name, p.stat().st_mtime_ns) for p in tmp_path.rglob("part-*.parquet")) == before
    )


def test_fresh_run_refuses_a_directory_with_parts(requests_df, tmp_path, quiet):
    run_requests(requests_df, FakeForecaster(), tmp_path, resume=False, echo=quiet)
    assert runner.has_forecast_parts(tmp_path)
    again = FakeForecaster()
    with pytest.raises(FileExistsError, match="resume=True"):
        run_requests(requests_df, again, tmp_path, resume=False, echo=quiet)
    assert again.calls == []
    assert not runner.has_forecast_parts(tmp_path / "elsewhere")


def scale_first_context(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """A copy of `df` whose first request has a different context (same ids, shapes)."""
    contexts = list(df["context"])
    contexts[0] = (contexts[0] * np.float32(factor)).astype(np.float32)
    return df.assign(context=pd.Series(contexts, index=df.index, dtype=object))


def restamp_key(out: Path, key: bytes, value: str | None) -> None:
    """Rewrite one provenance entry recorded in every part (None removes it)."""
    for part in out.glob("block=*/part-*.parquet"):
        table = pq.read_table(part)
        meta = dict(table.schema.metadata or {})
        meta.pop(key, None)
        if value is not None:
            meta[key] = value.encode()
        pq.write_table(table.replace_schema_metadata(meta), part)


def restamp(out: Path, sha: str | None) -> None:
    """Rewrite the requests sha256 recorded in every part (None removes it)."""
    restamp_key(out, runner._META_REQUESTS_SHA, sha)


def test_resume_refuses_parts_that_do_not_belong_to_the_requests(requests_df, tmp_path, quiet):
    run_requests(requests_df, FakeForecaster(), tmp_path, echo=quiet)
    other = make_requests([("zz1", 1, 3), ("zz2", 1, 3)])
    # the digest of the requests catches it first ...
    with pytest.raises(ValueError, match="other requests"):
        run_requests(other, FakeForecaster(), tmp_path, echo=quiet)
    # ... and the content check still holds when the digest happens to match
    restamp(tmp_path, rq.requests_digest(other))
    with pytest.raises(ValueError, match="not a request of this block"):
        run_requests(other, FakeForecaster(), tmp_path, echo=quiet)


def test_resume_refuses_a_part_with_missing_steps(tmp_path, quiet):
    df = make_requests([("r0", 1, 3), ("r1", 1, 3)])
    run_requests(df, FakeForecaster(), tmp_path, batch_size=1, echo=quiet)
    victim = parts_of(tmp_path, 1)[0]
    table = pq.read_table(victim)
    pq.write_table(table.slice(0, 2), victim)  # r0 now has steps 1..2 only
    with pytest.raises(ValueError, match="r0.*exactly steps 1..horizon"):
        run_requests(df, FakeForecaster(), tmp_path, echo=quiet)


def test_resume_refuses_a_request_in_two_parts(tmp_path, quiet):
    df = make_requests([("r0", 1, 2), ("r1", 1, 2)])
    run_requests(df, FakeForecaster(), tmp_path, batch_size=1, echo=quiet)
    parts = parts_of(tmp_path, 1)
    shutil.copy(parts[0], tmp_path / "block=1" / "part-000009.parquet")
    with pytest.raises(ValueError, match="already has forecasts in an earlier part"):
        run_requests(df, FakeForecaster(), tmp_path, echo=quiet)


def test_parts_and_status_record_the_requests_sha256(requests_df, tmp_path, quiet):
    run_requests(requests_df, FakeForecaster(), tmp_path, batch_size=3, echo=quiet)
    sha = rq.requests_digest(requests_df)
    assert len(sha) == 64
    parts = list(tmp_path.glob("block=*/part-*.parquet"))
    assert parts
    for part in parts:
        assert pq.read_schema(part).metadata[runner._META_REQUESTS_SHA] == sha.encode()
    saved = json.loads((tmp_path / runner.STATUS_FILE).read_text())
    assert {rec["requests_sha256"] for rec in saved.values()} == {sha}


def test_parts_and_status_record_the_checkpoint_revision_and_flags(requests_df, tmp_path, quiet):
    returned = run_requests(requests_df, FakeForecaster(), tmp_path, batch_size=3, echo=quiet)
    parts = list(tmp_path.glob("block=*/part-*.parquet"))
    assert parts
    for part in parts:
        meta = pq.read_schema(part).metadata
        assert meta[runner._META_REVISION] == REVISION.encode()
        assert meta[runner._META_SYMMETRIC] == b"true"
        assert meta[runner._META_POSITIVE] == b"true"
    saved = json.loads((tmp_path / runner.STATUS_FILE).read_text())
    assert saved == returned
    for rec in saved.values():
        assert rec["checkpoint_revision"] == REVISION
        assert rec["use_symmetric_averaging"] is True and rec["make_positive"] is True
    # the read-only helper reports the same provenance when given the revision
    assert runner.current_status(requests_df, tmp_path, checkpoint_revision=REVISION) == saved


def test_inference_flags_reach_the_model_and_are_recorded(requests_df, tmp_path, quiet):
    fake = FakeForecaster()
    status = run_requests(
        requests_df,
        fake,
        tmp_path,
        echo=quiet,
        use_symmetric_averaging=False,
        make_positive=False,
    )
    assert fake.calls
    for call in fake.calls:
        assert call["use_symmetric_averaging"] is False and call["make_positive"] is False
    for part in tmp_path.glob("block=*/part-*.parquet"):
        meta = pq.read_schema(part).metadata
        assert meta[runner._META_SYMMETRIC] == b"false" and meta[runner._META_POSITIVE] == b"false"
    for rec in status.values():
        assert rec["use_symmetric_averaging"] is False and rec["make_positive"] is False


def interrupted_dir(requests_df, out, **kwargs):
    """Run until the second batch dies; returns the names of the committed parts."""
    fake = FakeForecaster(fail_on_call=2)
    with pytest.raises(RuntimeError, match="simulated"):
        run_requests(requests_df, fake, out, batch_size=2, echo=lambda _line: None, **kwargs)
    return sorted(p.name for p in out.rglob("part-*.parquet"))


def test_resume_refuses_parts_from_another_checkpoint_revision_and_runs_nothing(
    requests_df, tmp_path, quiet
):
    before = interrupted_dir(requests_df, tmp_path)
    assert before
    status_before = (tmp_path / runner.STATUS_FILE).read_text()

    again = FakeForecaster()
    with pytest.raises(ValueError, match="checkpoint revision.*mix.*fresh output directory"):
        run_requests(requests_df, again, tmp_path, echo=quiet, checkpoint_revision=OTHER_REVISION)
    assert again.calls == []
    assert sorted(p.name for p in tmp_path.rglob("part-*.parquet")) == before
    assert (tmp_path / runner.STATUS_FILE).read_text() == status_before  # nothing rewritten
    assert not list(tmp_path.rglob("*.tmp-*"))
    with pytest.raises(ValueError) as exc:  # the message names both revisions
        run_requests(requests_df, again, tmp_path, echo=quiet, checkpoint_revision=OTHER_REVISION)
    assert REVISION in str(exc.value) and OTHER_REVISION in str(exc.value)
    # the read-only helpers refuse as well ...
    with pytest.raises(ValueError, match="checkpoint revision"):
        runner.current_status(requests_df, tmp_path, checkpoint_revision=OTHER_REVISION)
    with pytest.raises(ValueError, match="checkpoint revision"):
        runner.pending_counts(requests_df, tmp_path, checkpoint_revision=OTHER_REVISION)
    # ... but a caller that has not resolved a revision yet is not refused
    assert runner.pending_counts(requests_df, tmp_path) == {1: 2, 2: 2, 3: 2}
    assert runner.current_status(requests_df, tmp_path)["1"]["checkpoint_revision"] is None

    # the same revision resumes cleanly
    status = run_requests(requests_df, FakeForecaster(), tmp_path, echo=quiet)
    assert {v["status"] for v in status.values()} == {"complete"}
    assert {v["checkpoint_revision"] for v in status.values()} == {REVISION}


@pytest.mark.parametrize(
    ("flag", "key"),
    [
        ("use_symmetric_averaging", runner._META_SYMMETRIC),
        ("make_positive", runner._META_POSITIVE),
    ],
)
def test_resume_refuses_parts_made_with_other_inference_flags(
    requests_df, tmp_path, quiet, flag, key
):
    before = interrupted_dir(requests_df, tmp_path)
    again = FakeForecaster()
    with pytest.raises(ValueError, match=f"{flag}=true.*{flag}=false.*fresh output directory"):
        run_requests(requests_df, again, tmp_path, echo=quiet, **{flag: False})
    assert again.calls == []
    assert sorted(p.name for p in tmp_path.rglob("part-*.parquet")) == before
    with pytest.raises(ValueError, match=flag):
        runner.current_status(requests_df, tmp_path, **{flag: False})
    with pytest.raises(ValueError, match=flag):
        runner.pending_counts(requests_df, tmp_path, **{flag: False})

    # parts made with the flag off refuse a resume with the default ...
    other = tmp_path / "other"
    interrupted_dir(requests_df, other, **{flag: False})
    with pytest.raises(ValueError, match=f"{flag}=false.*{flag}=true"):
        run_requests(requests_df, FakeForecaster(), other, echo=quiet)
    # ... and resume when the flag matches
    status = run_requests(requests_df, FakeForecaster(), other, echo=quiet, **{flag: False})
    assert {v["status"] for v in status.values()} == {"complete"}
    assert pq.read_schema(next(other.rglob("part-*.parquet"))).metadata[key] == b"false"


def test_resume_refuses_parts_without_a_recorded_revision_or_flags(requests_df, tmp_path, quiet):
    run_requests(requests_df, FakeForecaster(), tmp_path / "good", echo=quiet)
    for key, text in (
        (runner._META_REVISION, "no checkpoint revision recorded"),
        (runner._META_SYMMETRIC, "no use_symmetric_averaging recorded"),
        (runner._META_POSITIVE, "no make_positive recorded"),
    ):
        victim = tmp_path / key.decode()
        shutil.copytree(tmp_path / "good", victim)
        restamp_key(victim, key, None)  # e.g. written by an older runner
        with pytest.raises(ValueError, match=f"{text}.*fresh output directory"):
            run_requests(requests_df, FakeForecaster(), victim, echo=quiet)


def test_run_requests_needs_a_non_empty_checkpoint_revision(requests_df, tmp_path, quiet):
    with pytest.raises(TypeError, match="checkpoint_revision"):
        runner.run_requests(requests_df, FakeForecaster(), tmp_path, echo=quiet)
    for bad in (None, "", "  ", "main", "ab" * 19, "zz" * 20, "ab" * 21):
        with pytest.raises(ValueError, match="checkpoint_revision"):
            runner.run_requests(
                requests_df, FakeForecaster(), tmp_path, echo=quiet, checkpoint_revision=bad
            )
    assert not runner.has_forecast_parts(tmp_path)


def test_run_requests_accepts_an_uppercase_sha(requests_df, tmp_path, quiet):
    runner.run_requests(
        requests_df, FakeForecaster(), tmp_path, echo=quiet, checkpoint_revision=REVISION.upper()
    )
    assert runner.has_forecast_parts(tmp_path)


def test_resume_refuses_parts_from_different_requests_and_runs_nothing(
    requests_df, tmp_path, quiet
):
    with pytest.raises(RuntimeError, match="simulated"):
        run_requests(
            requests_df, FakeForecaster(fail_on_call=2), tmp_path, batch_size=2, echo=quiet
        )
    before = sorted(p.name for p in tmp_path.rglob("part-*.parquet"))
    assert before

    # same ids and shapes, one context value differs: a different requests file
    changed = scale_first_context(requests_df, 1.5)
    assert rq.requests_digest(changed) != rq.requests_digest(requests_df)

    again = FakeForecaster()
    with pytest.raises(ValueError, match="other requests.*fresh output directory"):
        run_requests(changed, again, tmp_path, echo=quiet)
    assert again.calls == []
    assert sorted(p.name for p in tmp_path.rglob("part-*.parquet")) == before
    assert not list(tmp_path.rglob("*.tmp-*"))
    # the read-only helpers refuse as well
    with pytest.raises(ValueError, match="other requests"):
        runner.current_status(changed, tmp_path)
    with pytest.raises(ValueError, match="other requests"):
        runner.pending_counts(changed, tmp_path)

    # the original requests still resume cleanly
    status = run_requests(requests_df, FakeForecaster(), tmp_path, echo=quiet)
    assert {v["status"] for v in status.values()} == {"complete"}


def test_resume_refuses_a_subset_of_the_requests_that_made_the_parts(requests_df, tmp_path, quiet):
    run_requests(requests_df, FakeForecaster(), tmp_path, echo=quiet)
    only_block_1 = requests_df[requests_df["block"] == 1]  # every id is known to the parts
    with pytest.raises(ValueError, match="other requests"):
        run_requests(only_block_1, FakeForecaster(), tmp_path, echo=quiet)


def test_resume_accepts_the_same_requests_in_another_row_order(requests_df, tmp_path, quiet):
    with pytest.raises(RuntimeError, match="simulated"):
        run_requests(
            requests_df, FakeForecaster(fail_on_call=2), tmp_path, batch_size=2, echo=quiet
        )
    shuffled = requests_df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert rq.requests_digest(shuffled) == rq.requests_digest(requests_df)
    status = run_requests(shuffled, FakeForecaster(), tmp_path, batch_size=2, echo=quiet)
    assert {v["status"] for v in status.values()} == {"complete"}
    rq.validate_forecasts(requests_df, rq.read_forecasts(tmp_path))


def test_resume_refuses_a_part_without_a_recorded_requests_sha256(requests_df, tmp_path, quiet):
    run_requests(requests_df, FakeForecaster(), tmp_path, echo=quiet)
    restamp(tmp_path, None)  # e.g. written by an older runner
    with pytest.raises(ValueError, match="no requests sha256.*fresh output directory"):
        run_requests(requests_df, FakeForecaster(), tmp_path, echo=quiet)


def test_interrupted_atomic_write_debris_is_removed(requests_df, tmp_path, quiet):
    debris = tmp_path / "block=1" / "part-000000.parquet.tmp-123-abcd"
    debris.parent.mkdir(parents=True)
    debris.write_bytes(b"half a file")
    run_requests(requests_df, FakeForecaster(), tmp_path, echo=quiet)
    assert not debris.exists()


# ------------------------------------------------------------------ block_status.json


def test_block_status_contents(requests_df, tmp_path, quiet):
    returned = run_requests(requests_df, FakeForecaster(), tmp_path, blocks=[1, 2], echo=quiet)
    saved = json.loads((tmp_path / runner.STATUS_FILE).read_text())
    assert saved == returned
    assert sorted(saved) == ["1", "2", "3"]
    for rec in saved.values():
        assert set(rec) == {
            "requests",
            "done",
            "floored_values",
            "seconds",
            "status",
            "requests_sha256",
            "checkpoint_revision",
            "use_symmetric_averaging",
            "make_positive",
        }
        assert rec["requests_sha256"] == rq.requests_digest(requests_df)
        assert rec["checkpoint_revision"] == REVISION
    assert saved["1"]["requests"] == 4 and saved["1"]["done"] == 4
    assert saved["2"]["requests"] == 2 and saved["2"]["done"] == 2
    assert saved["1"]["status"] == saved["2"]["status"] == "complete"
    assert saved["1"]["seconds"] >= 0 and saved["1"]["floored_values"] == 0
    assert saved["3"] == {
        "requests": 2,
        "done": 0,
        "floored_values": 0,
        "seconds": 0.0,
        "status": "pending",
        "requests_sha256": rq.requests_digest(requests_df),
        "checkpoint_revision": REVISION,
        "use_symmetric_averaging": True,
        "make_positive": True,
    }


def test_status_is_derived_from_the_parts_and_is_read_only(requests_df, tmp_path, quiet):
    out = tmp_path / "out"
    assert runner.pending_counts(requests_df, out) == {1: 4, 2: 2, 3: 2}
    assert not out.exists()  # looking never creates anything
    with pytest.raises(RuntimeError):
        run_requests(requests_df, FakeForecaster(fail_on_call=2), out, batch_size=3, echo=quiet)
    (out / runner.STATUS_FILE).unlink()  # the json is a cache: the parts are the truth
    status = runner.current_status(requests_df, out)
    assert status["1"]["status"] == "partial" and status["1"]["done"] == 2
    assert runner.pending_counts(requests_df, out) == {1: 2, 2: 2, 3: 2}
    assert runner.pending_counts(requests_df, out, blocks=[2]) == {2: 2}
    assert not (out / runner.STATUS_FILE).exists()


# ------------------------------------------------------------------ the CLI


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("script_14_v3_forecast", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_help_lists_every_flag(script, capsys):
    with pytest.raises(SystemExit) as exc:
        script.build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for flag in (
        "--requests",
        "--out",
        "--adapter",
        "--batch-size",
        "--blocks",
        "--resume",
        "--allow-cpu",
        "--as-of",
        "--dry-run",
    ):
        assert flag in text


def test_cli_help_runs_as_a_real_subprocess():
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )
    assert done.returncode == 0 and "--requests" in done.stdout


def test_cli_defaults_and_argument_validation(script):
    args = script.build_parser().parse_args(["--as-of", "2026-09-04"])
    assert args.requests == config.CACHE_DIR / "v3_requests.parquet"
    assert args.out == config.RESULTS_DIR / "benchmark" / "v3_forecasts"
    assert (args.adapter, args.batch_size, args.blocks) == ("timesfm3", 128, None)
    assert args.resume is False and args.allow_cpu is False and args.dry_run is False

    args = script.build_parser().parse_args(
        ["--blocks", "3,1,1", "--batch-size", "16", "--resume", "--allow-cpu"]
    )
    assert args.blocks == [1, 3] and args.batch_size == 16 and args.resume and args.allow_cpu
    for bad in (
        ["--blocks", "0"],
        ["--blocks", "5"],
        ["--blocks", "x"],
        ["--blocks", ""],
        ["--batch-size", "0"],
        ["--adapter", "timesfm2.5"],
    ):
        with pytest.raises(SystemExit):
            script.build_parser().parse_args(bad)


def test_default_output_dir_is_gitignored(script):
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    probe = (
        script.default_out_dir().relative_to(config.RESULTS_DIR.parent)
        / "block=1/part-000000.parquet"
    )
    done = subprocess.run(
        ["git", "check-ignore", "-q", str(probe)], cwd=ROOT, capture_output=True, text=True
    )
    assert done.returncode == 0, "results/benchmark/v3_forecasts must stay out of Git"


@pytest.fixture
def cli_env(script, tmp_path, monkeypatch, requests_df):
    """The script with preflight and model loading faked, manifests under tmp_path."""
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "CHECKPOINT_REVISION", REVISION)
    monkeypatch.setattr(script, "capture_git_state", lambda: False)
    fake = FakeForecaster()
    loaded: list[str | None] = []
    preflights: list[bool] = []
    resolve: dict[str, str] = {}  # revision requested -> resolved; a full sha resolves to itself

    def preflight(repo_id, revision, *, allow_cpu=False):
        preflights.append(allow_cpu)
        return {
            "repo_id": repo_id,
            "revision_requested": revision,
            "revision_resolved": resolve.get(revision, revision),
        }

    def load(revision=None):
        loaded.append(revision)
        return fake

    monkeypatch.setattr(script, "preflight_model", preflight)
    monkeypatch.setattr(script, "load_forecaster", load)
    path = rq.write_requests(requests_df, tmp_path / "v3_requests.parquet")
    argv = [
        "--requests",
        str(path),
        "--out",
        str(tmp_path / "out"),
        "--as-of",
        "2026-09-04",
        "--batch-size",
        "3",
    ]
    return SimpleNamespace(
        fake=fake,
        loaded=loaded,
        preflights=preflights,
        resolve=resolve,
        argv=argv,
        path=path,
        tmp=tmp_path,
    )


def manifests(env) -> list[Path]:
    return sorted((env.tmp / "results" / "manifests").glob("v3_forecast-2026-09-04-*.json"))


def test_cli_run_writes_forecasts_and_a_value_free_manifest(script, cli_env, requests_df, capsys):
    assert script.main([*cli_env.argv, "--allow-cpu"]) == 0
    out = cli_env.tmp / "out"
    assert cli_env.preflights == [True] and cli_env.loaded == [REVISION]
    rq.validate_forecasts(requests_df, rq.read_forecasts(out))
    assert {v["status"] for v in json.loads((out / runner.STATUS_FILE).read_text()).values()} == {
        "complete"
    }

    (manifest_path,) = manifests(cli_env)
    text = manifest_path.read_text()
    manifest = json.loads(text)
    assert manifest["requests"]["sha256"] == hashlib.sha256(cli_env.path.read_bytes()).hexdigest()
    assert manifest["requests"]["content_sha256"] == rq.requests_digest(requests_df)
    assert manifest["requests"]["n_requests"] == 8
    assert manifest["requests"]["by_block"] == {"1": 4, "2": 2, "3": 2}
    assert sum(manifest["requests"]["by_study"].values()) == 8
    assert manifest["model"]["revision"] == config.CHECKPOINT_REVISION == REVISION
    assert manifest["model"]["revision_resolved"] == REVISION
    assert manifest["model"]["use_symmetric_averaging"] is True
    assert manifest["model"]["make_positive"] is True
    assert manifest["_meta"]["checkpoint"]["revision_resolved"] == REVISION
    # the resolved revision and the flags are stamped into every part and status record
    for part in out.glob("block=*/part-*.parquet"):
        meta = pq.read_schema(part).metadata
        assert meta[runner._META_REVISION] == REVISION.encode()
        assert meta[runner._META_SYMMETRIC] == b"true" and meta[runner._META_POSITIVE] == b"true"
    for rec in json.loads((out / runner.STATUS_FILE).read_text()).values():
        assert rec["checkpoint_revision"] == REVISION
        assert rec["use_symmetric_averaging"] is True and rec["make_positive"] is True
    assert manifest["device"] == manifest["_meta"]["environment"]["gpu_name"]
    assert manifest["batch_size"] == 3 and manifest["blocks_selected"] == [1, 2, 3]
    assert {k: v["status"] for k, v in manifest["block_status"].items()} == dict.fromkeys(
        "123", "complete"
    )
    assert manifest["counts"] == {"requests": 8, "done": 8, "floored_values": 0, "parts": 6}
    assert manifest["session"]["forecast_this_session"] == {"1": 4, "2": 2, "3": 2}
    assert manifest["validation"] == {"blocks": [1, 2, 3], "ok": True, "error": None}
    assert manifest["error"] is None
    assert len(manifest["outputs"]) == 7  # six parts (one per batch) and block_status.json
    assert all(len(o["sha256"]) == 64 for o in manifest["outputs"])
    # no forecast values and no contexts anywhere in the manifest
    forecasts = rq.read_forecasts(out)
    assert not any(f"{v:.6f}"[:7] in text for v in forecasts["point"].head(3))
    for key in ("context", "point", "q10", "q90"):
        assert f'"{key}"' not in text
    assert "manifest" in capsys.readouterr().out


def test_cli_second_resume_has_nothing_to_do_and_needs_no_gpu(script, cli_env):
    assert script.main([*cli_env.argv, "--allow-cpu"]) == 0
    calls_after_first = len(cli_env.fake.calls)
    cli_env.preflights.clear()
    cli_env.loaded.clear()
    assert script.main([*cli_env.argv, "--resume"]) == 0
    assert cli_env.preflights == [] and cli_env.loaded == []  # no preflight, no model load
    assert len(cli_env.fake.calls) == calls_after_first
    assert len(manifests(cli_env)) in (1, 2)  # a second manifest only if the clock ticked
    last = json.loads(manifests(cli_env)[-1].read_text())
    assert last["session"]["forecast_this_session"] == {"1": 0, "2": 0, "3": 0}
    assert last["model"]["revision_resolved"] is None


def test_cli_fresh_run_refuses_existing_parts(script, cli_env):
    assert script.main([*cli_env.argv, "--allow-cpu"]) == 0
    with pytest.raises(SystemExit, match="--resume"):
        script.main(cli_env.argv)


def test_cli_interrupted_run_keeps_progress_and_records_the_error(script, cli_env, requests_df):
    cli_env.fake.fail_on_call = 2  # block 1: the horizon-3 batch is committed, horizon 7 dies
    with pytest.raises(RuntimeError, match="simulated interruption"):
        script.main([*cli_env.argv, "--allow-cpu"])
    (manifest_path,) = manifests(cli_env)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["error"].startswith("RuntimeError: simulated interruption")
    assert manifest["validation"] is None
    assert manifest["block_status"]["1"]["status"] == "partial"
    assert manifest["counts"]["done"] == 2

    cli_env.fake.fail_on_call = None
    assert script.main([*cli_env.argv, "--resume"]) == 0
    rq.validate_forecasts(requests_df, rq.read_forecasts(cli_env.tmp / "out"))


def test_cli_blocks_flag_runs_only_those_blocks(script, cli_env):
    assert script.main([*cli_env.argv, "--allow-cpu", "--blocks", "2"]) == 0
    status = json.loads((cli_env.tmp / "out" / runner.STATUS_FILE).read_text())
    assert {k: v["status"] for k, v in status.items()} == {
        "1": "pending",
        "2": "complete",
        "3": "pending",
    }
    manifest = json.loads(manifests(cli_env)[-1].read_text())
    assert manifest["blocks_selected"] == [2]
    assert manifest["validation"]["blocks"] == [2]


def test_cli_dry_run_reports_without_gpu_or_writes(script, cli_env, capsys):
    argv = [a for a in cli_env.argv if a not in ("--as-of", "2026-09-04")]
    assert script.main([*argv, "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "block 1: 4 pending of 4" in text and "dry run" in text
    assert cli_env.preflights == [] and cli_env.loaded == []
    assert not (cli_env.tmp / "out").exists() and manifests(cli_env) == []


def test_cli_dry_run_reports_even_when_out_holds_parts_and_resume_is_absent(
    script, cli_env, capsys
):
    cli_env.fake.fail_on_call = 2  # block 1: one batch committed, the second dies
    with pytest.raises(RuntimeError, match="simulated interruption"):
        script.main([*cli_env.argv, "--allow-cpu"])
    out = cli_env.tmp / "out"
    parts_before = sorted((p.name, p.stat().st_mtime_ns) for p in out.rglob("*.parquet"))
    status_before = (out / runner.STATUS_FILE).read_text()
    manifests_before = manifests(cli_env)
    cli_env.preflights.clear()
    cli_env.loaded.clear()
    capsys.readouterr()

    argv = [a for a in cli_env.argv if a not in ("--as-of", "2026-09-04")]
    assert "--resume" not in argv
    assert script.main([*argv, "--dry-run"]) == 0  # the real-run guard does not apply
    text = capsys.readouterr().out
    assert "block 1: 2 pending of 4" in text and "block 2: 2 pending of 2" in text
    assert "already holds forecast parts" in text and "needs --resume" in text
    assert "dry run" in text
    assert cli_env.preflights == [] and cli_env.loaded == []
    assert sorted((p.name, p.stat().st_mtime_ns) for p in out.rglob("*.parquet")) == parts_before
    assert (out / runner.STATUS_FILE).read_text() == status_before
    assert manifests(cli_env) == manifests_before

    # without --dry-run the same invocation is still refused
    with pytest.raises(SystemExit, match="--resume"):
        script.main(cli_env.argv)


def test_cli_resume_refuses_parts_from_different_requests(script, cli_env, requests_df):
    assert script.main([*cli_env.argv, "--allow-cpu", "--blocks", "2"]) == 0
    changed = scale_first_context(requests_df, 2.0)
    other = rq.write_requests(changed, cli_env.tmp / "other.parquet")
    argv = cli_env.argv.copy()
    argv[1] = str(other)
    calls = len(cli_env.fake.calls)
    with pytest.raises(SystemExit, match="other requests.*fresh output directory"):
        script.main([*argv, "--resume"])
    with pytest.raises(SystemExit, match="other requests"):
        script.main([*argv, "--dry-run"])
    assert len(cli_env.fake.calls) == calls and cli_env.loaded == [REVISION]  # no new load


def test_cli_resume_refuses_another_pinned_revision_before_any_preflight(
    script, cli_env, monkeypatch
):
    assert script.main([*cli_env.argv, "--allow-cpu", "--blocks", "2"]) == 0
    calls = len(cli_env.fake.calls)
    cli_env.preflights.clear()
    cli_env.loaded.clear()
    monkeypatch.setattr(config, "CHECKPOINT_REVISION", OTHER_REVISION.upper())  # case is ignored
    for extra in (["--resume"], ["--dry-run"]):
        with pytest.raises(SystemExit, match="checkpoint revision.*fresh output directory"):
            script.main([*cli_env.argv, *extra])
    assert cli_env.preflights == [] and cli_env.loaded == []  # refused before any GPU time
    assert len(cli_env.fake.calls) == calls

    # a fresh --out is the documented way forward
    fresh = [*cli_env.argv]
    fresh[fresh.index("--out") + 1] = str(cli_env.tmp / "fresh")
    assert script.main([*fresh, "--allow-cpu"]) == 0
    assert [r.lower() for r in cli_env.loaded] == [OTHER_REVISION]  # the hub answers in lowercase


@pytest.mark.parametrize("bad", ["main", "v1.0", "abc123", "", "ab" * 19])
@pytest.mark.parametrize("extra", [[], ["--resume"], ["--dry-run"]])
def test_cli_refuses_a_revision_that_is_not_a_full_sha(script, cli_env, monkeypatch, bad, extra):
    monkeypatch.setattr(config, "CHECKPOINT_REVISION", bad)
    with pytest.raises(SystemExit, match="40-hex commit sha"):
        script.main([*cli_env.argv, "--allow-cpu", *extra])
    assert cli_env.preflights == [] and cli_env.loaded == []
    assert not cli_env.fake.calls


def test_cli_requires_as_of_for_a_real_run(script, cli_env):
    argv = [a for a in cli_env.argv if a not in ("--as-of", "2026-09-04")]
    with pytest.raises(SystemExit):
        script.main(argv)


def test_cli_rejects_missing_and_invalid_requests_files(script, cli_env):
    missing = cli_env.argv.copy()
    missing[1] = str(cli_env.tmp / "nope.parquet")
    with pytest.raises(SystemExit, match="not found"):
        script.main(missing)

    # a file that write_requests would never produce: one horizon is 0
    table = pq.read_table(cli_env.path)
    horizon = table.column("horizon").to_pylist()
    horizon[2] = 0
    table = table.set_column(3, rq.REQUESTS_SCHEMA.field("horizon"), pa.array(horizon, pa.int16()))
    bad_path = cli_env.tmp / "bad.parquet"
    pq.write_table(table, bad_path)
    invalid = cli_env.argv.copy()
    invalid[1] = str(bad_path)
    with pytest.raises(SystemExit, match="horizon=0"):
        script.main(invalid)
    assert cli_env.preflights == []


def test_cli_gpu_is_required_unless_allow_cpu(script, cli_env, monkeypatch):
    """The real preflight_model: the GPU check comes first, --allow-cpu skips it."""
    from tfm3lab import preflight

    monkeypatch.setattr(script, "preflight_model", preflight.preflight_model)
    monkeypatch.setattr(preflight, "_cuda_available", lambda: False)
    monkeypatch.setattr(
        preflight, "require_checkpoint_access", lambda repo, revision=None: REVISION
    )
    with pytest.raises(SystemExit, match="no CUDA GPU"):
        script.main(cli_env.argv)
    assert cli_env.loaded == [] and not (cli_env.tmp / "out").exists()

    assert script.main([*cli_env.argv, "--allow-cpu"]) == 0
    assert cli_env.loaded == [REVISION]
