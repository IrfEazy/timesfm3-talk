"""Offline tests of the v3 studies file contract (tfm3lab.studies.requests):
the frozen requests/forecasts schemas, their IO and the validators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tfm3lab.studies import requests as rq

# ------------------------------------------------------------------ helpers


def make_requests(n: int = 4, *, length: int = 8, horizon: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n):
        ctx = (1.0 + 0.01 * np.arange(length) + 0.001 * i).astype(np.float32)
        rows.append(
            {
                "request_id": f"r{i:03d}",
                "study": ("S1", "S2", "S3")[i % 3],
                "block": 1 + i % 4,
                "horizon": horizon,
                "context": ctx,
            }
        )
    return pd.DataFrame(rows)


def set_context(df: pd.DataFrame, i: int, value) -> pd.DataFrame:
    """Replace one context (DataFrame.at would unwrap a 1-element array)."""
    column = df["context"].tolist()
    column[i] = value
    return df.assign(context=pd.Series(column, index=df.index, dtype=object))


def with_row(df: pd.DataFrame, i: int, **changes) -> pd.DataFrame:
    out = df.copy()
    for column, value in changes.items():
        if column == "context":
            out = set_context(out, i, value)
        else:
            out.at[i, column] = value
    return out


def make_forecasts(requests_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, h in zip(requests_df["request_id"], requests_df["horizon"], strict=True):
        for step in range(1, int(h) + 1):
            point = 1.0 + 0.01 * step
            row = {"request_id": rid, "step": step, "point": point}
            for j, name in enumerate(rq.QUANTILE_COLUMNS):
                row[name] = point * (0.8 + 0.05 * j)
            rows.append(row)
    return pd.DataFrame(rows).astype(
        {"step": "int16", **{c: "float32" for c in ("point", *rq.QUANTILE_COLUMNS)}}
    )


# ------------------------------------------------------------------ schema constants


def test_frozen_column_names_and_types():
    assert rq.REQUEST_COLUMNS == ("request_id", "study", "block", "horizon", "context")
    expected_quantiles = tuple(f"q{p}" for p in range(10, 100, 10))
    assert expected_quantiles == rq.QUANTILE_COLUMNS
    expected_forecast = ("request_id", "step", "point", *expected_quantiles)
    assert expected_forecast == rq.FORECAST_COLUMNS
    assert rq.STUDIES == ("S1", "S2", "S3")
    schema = {f.name: f.type for f in rq.REQUESTS_SCHEMA}
    assert schema == {
        "request_id": pa.string(),
        "study": pa.string(),
        "block": pa.int8(),
        "horizon": pa.int16(),
        "context": pa.list_(pa.float32()),
    }
    forecast = {f.name: f.type for f in rq.FORECASTS_SCHEMA}
    assert forecast["request_id"] == pa.string() and forecast["step"] == pa.int16()
    assert all(forecast[c] == pa.float32() for c in ("point", *rq.QUANTILE_COLUMNS))
    assert rq.MAX_CONTEXT_LENGTH == 512 and (rq.BLOCK_MIN, rq.BLOCK_MAX) == (1, 4)


# ------------------------------------------------------------------ requests round trip


def test_write_read_round_trip(tmp_path):
    df = make_requests(6)
    path = rq.write_requests(df, tmp_path / "sub" / "v3_requests.parquet")
    assert path.is_file()
    assert [p.name for p in path.parent.iterdir()] == ["v3_requests.parquet"]  # no temp debris

    schema = pq.read_schema(path)
    assert schema.field("block").type == pa.int8()
    assert schema.field("horizon").type == pa.int16()
    assert schema.field("context").type == pa.list_(pa.float32())

    back = rq.read_requests(path)
    assert list(back.columns) == list(rq.REQUEST_COLUMNS)
    assert back["request_id"].tolist() == df["request_id"].tolist()  # input order kept
    assert back["study"].tolist() == df["study"].tolist()
    assert back["block"].dtype == np.int8 and back["horizon"].dtype == np.int16
    for got, want in zip(back["context"], df["context"], strict=True):
        assert got.dtype == np.float32 and got.ndim == 1
        np.testing.assert_array_equal(got, want)


def test_context_length_bounds_are_inclusive(tmp_path):
    df = set_context(make_requests(2), 0, np.array([1.0], dtype=np.float32))
    df = set_context(df, 1, np.full(512, 1.5, dtype=np.float32))
    back = rq.read_requests(rq.write_requests(df, tmp_path / "r.parquet"))
    assert [len(c) for c in back["context"]] == [1, 512]


def test_requests_digest_is_canonical(tmp_path):
    df = make_requests(5)
    digest = rq.requests_digest(df)
    assert len(digest) == 64 and int(digest, 16) >= 0
    # independent of row order, index and of the parquet round trip
    assert rq.requests_digest(df.iloc[::-1].reset_index(drop=True)) == digest
    back = rq.read_requests(rq.write_requests(df, tmp_path / "r.parquet"))
    assert rq.requests_digest(back) == digest
    # float64 / list inputs normalised the way write_requests does
    assert rq.requests_digest(rq.validate_requests_frame(df)) == digest


@pytest.mark.parametrize(
    "changes",
    [
        {"request_id": "other"},
        {"study": "S3"},
        {"block": 4},
        {"horizon": 9},
        {"context": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)},
        {"context": np.array([1.0, 1.01], dtype=np.float32)},
    ],
)
def test_requests_digest_changes_with_any_field(changes):
    df = make_requests(3)
    assert rq.requests_digest(with_row(df, 1, **changes)) != rq.requests_digest(df)


def test_requests_digest_depends_on_the_set_of_requests():
    df = make_requests(3)
    assert rq.requests_digest(df.iloc[:2]) != rq.requests_digest(df)
    assert rq.requests_digest(df.iloc[:0]) != rq.requests_digest(df)


def test_write_accepts_lists_and_float64_contexts(tmp_path):
    df = set_context(make_requests(2), 0, [1.0, 1.5, 2.0])
    df = set_context(df, 1, np.array([1.0, 2.0], dtype=np.float64))
    back = rq.read_requests(rq.write_requests(df, tmp_path / "r.parquet"))
    assert [c.tolist() for c in back["context"]] == [[1.0, 1.5, 2.0], [1.0, 2.0]]


# ------------------------------------------------------------------ requests validation


@pytest.mark.parametrize(
    ("changes", "needle"),
    [
        ({"study": "S4"}, "study"),
        ({"study": "s1"}, "study"),
        ({"block": 0}, "block"),
        ({"block": 5}, "block"),
        ({"horizon": 0}, "horizon"),
        ({"horizon": -3}, "horizon"),
        ({"horizon": 40000}, "horizon"),
        ({"context": np.array([], dtype=np.float32)}, "length"),
        ({"context": np.ones(513, dtype=np.float32)}, "length"),
        ({"context": np.ones((2, 3), dtype=np.float32)}, "1-D"),
        ({"context": np.array([1.0, np.nan, 1.0])}, "finite"),
        ({"context": np.array([1.0, np.inf, 1.0])}, "finite"),
        ({"context": np.array([1.0, -np.inf, 1.0])}, "finite"),
        ({"context": np.array([1.0, 0.0, 1.0])}, "> 0"),
        ({"context": np.array([1.0, -0.5, 1.0])}, "> 0"),
        ({"context": np.array([1.0, 1e39, 1.0])}, "finite"),  # overflows float32
        ({"context": np.array([1.0, 1e-50, 1.0])}, "> 0"),  # underflows to 0 in float32
        ({"context": None}, "null"),
        ({"context": ["a", "b"]}, "numeric"),
    ],
)
def test_invalid_row_names_the_offending_request_id(changes, needle):
    df = with_row(make_requests(4), 2, **changes)
    with pytest.raises(ValueError, match="r002") as err:
        rq.validate_requests_frame(df)
    assert needle in str(err.value)
    with pytest.raises(ValueError, match="r002"):
        rq.write_requests(df, "/nonexistent-dir-never-created/x.parquet")


def test_first_offender_is_reported_when_several_rows_are_bad():
    df = make_requests(5)
    df = with_row(df, 1, block=9)
    df = with_row(df, 3, block=0)
    with pytest.raises(ValueError, match="r001"):
        rq.validate_requests_frame(df)


def test_duplicate_request_id_is_rejected_and_named():
    df = make_requests(4)
    df.at[3, "request_id"] = "r001"
    with pytest.raises(ValueError, match="r001.*not unique"):
        rq.validate_requests_frame(df)


@pytest.mark.parametrize("bad", ["", None, 7])
def test_request_id_must_be_a_non_empty_string(bad):
    df = make_requests(3)
    df["request_id"] = df["request_id"].astype(object)
    df.at[1, "request_id"] = bad
    with pytest.raises(ValueError, match="request_id must be a non-empty string"):
        rq.validate_requests_frame(df)


def test_extra_and_missing_columns_are_rejected():
    df = make_requests(2)
    with pytest.raises(ValueError, match="unexpected.*outcome"):
        rq.validate_requests_frame(df.assign(outcome=[1.0, 2.0]))  # no way to smuggle a label
    with pytest.raises(ValueError, match="missing.*study"):
        rq.validate_requests_frame(df.drop(columns=["study"]))


def test_non_integer_block_dtype_is_rejected():
    df = make_requests(2)
    df["block"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="block.*integer"):
        rq.validate_requests_frame(df)


# ------------------------------------------------------------------ read_requests on files


def _write_raw(path, **columns):
    pq.write_table(pa.table(columns), path)
    return path


def _valid_columns(n: int = 3) -> dict:
    return {
        "request_id": pa.array([f"r{i}" for i in range(n)], pa.string()),
        "study": pa.array(["S1"] * n, pa.string()),
        "block": pa.array([1] * n, pa.int8()),
        "horizon": pa.array([2] * n, pa.int16()),
        "context": pa.array([[1.0, 1.1]] * n, pa.list_(pa.float32())),
    }


def test_read_requests_accepts_a_hand_built_valid_file(tmp_path):
    df = rq.read_requests(_write_raw(tmp_path / "ok.parquet", **_valid_columns()))
    assert len(df) == 3 and df["context"].iloc[0].tolist() == pytest.approx([1.0, 1.1])


@pytest.mark.parametrize(
    ("column", "wrong", "needle"),
    [
        ("block", pa.array([1, 2, 3], pa.int64()), "block"),
        ("horizon", pa.array([2, 2, 2], pa.int32()), "horizon"),
        ("context", pa.array([[1.0, 1.1]] * 3, pa.list_(pa.float64())), "context"),
        ("study", pa.array([1, 1, 1], pa.int64()), "study"),
    ],
)
def test_read_requests_rejects_wrong_column_types(tmp_path, column, wrong, needle):
    cols = _valid_columns()
    cols[column] = wrong
    path = _write_raw(tmp_path / "bad.parquet", **cols)
    with pytest.raises(ValueError, match=f"{needle}.*must be"):
        rq.read_requests(path)


def test_read_requests_rejects_extra_columns(tmp_path):
    cols = _valid_columns()
    cols["y_true"] = pa.array([1.0, 2.0, 3.0], pa.float32())
    with pytest.raises(ValueError, match="unexpected.*y_true"):
        rq.read_requests(_write_raw(tmp_path / "bad.parquet", **cols))


def test_read_requests_names_the_request_with_a_null_field(tmp_path):
    cols = _valid_columns()
    cols["horizon"] = pa.array([2, None, 2], pa.int16())
    with pytest.raises(ValueError, match="r1.*horizon is null"):
        rq.read_requests(_write_raw(tmp_path / "bad.parquet", **cols))
    cols = _valid_columns()
    cols["context"] = pa.array([[1.0, 1.1], None, [1.0, 1.1]], pa.list_(pa.float32()))
    with pytest.raises(ValueError, match="r1.*context is null"):
        rq.read_requests(_write_raw(tmp_path / "bad2.parquet", **cols))


def test_read_requests_validates_values_in_the_file(tmp_path):
    cols = _valid_columns()
    cols["context"] = pa.array(
        [[1.0, 1.1], [1.0, float("nan")], [1.0, 1.1]], pa.list_(pa.float32())
    )
    with pytest.raises(ValueError, match="r1.*finite"):
        rq.read_requests(_write_raw(tmp_path / "nan.parquet", **cols))
    cols = _valid_columns()
    cols["block"] = pa.array([1, 1, 7], pa.int8())
    with pytest.raises(ValueError, match="r2.*block"):
        rq.read_requests(_write_raw(tmp_path / "block.parquet", **cols))
    cols = _valid_columns()
    cols["request_id"] = pa.array(["a", "b", "a"], pa.string())
    with pytest.raises(ValueError, match="'a'.*not unique"):
        rq.read_requests(_write_raw(tmp_path / "dupe.parquet", **cols))
    cols = _valid_columns()
    cols["context"] = pa.array([[1.0] * 600] * 3, pa.list_(pa.float32()))
    with pytest.raises(ValueError, match="r0.*length"):
        rq.read_requests(_write_raw(tmp_path / "long.parquet", **cols))


# ------------------------------------------------------------------ forecasts IO


def test_forecast_round_trip_and_directory_read(tmp_path):
    requests_df = make_requests(4)
    forecasts = make_forecasts(requests_df)
    first, second = (
        forecasts[forecasts["request_id"] < "r002"],
        forecasts[forecasts["request_id"] >= "r002"],
    )
    # written out of order on purpose: the directory read sorts by (request_id, step)
    rq.write_forecasts(second, tmp_path / "block=2" / "part-000000.parquet")
    rq.write_forecasts(first, tmp_path / "block=1" / "part-000000.parquet")

    schema = pq.read_schema(tmp_path / "block=1" / "part-000000.parquet")
    assert schema.names == list(rq.FORECAST_COLUMNS)
    assert schema.field("step").type == pa.int16() and schema.field("q50").type == pa.float32()

    back = rq.read_forecasts(tmp_path)
    assert back["request_id"].tolist() == forecasts["request_id"].tolist()
    assert back["step"].tolist() == forecasts["step"].tolist()
    np.testing.assert_allclose(back["q90"], forecasts["q90"])
    single = rq.read_forecasts(tmp_path / "block=1" / "part-000000.parquet")
    assert len(single) == len(first)
    rq.validate_forecasts(requests_df, back)


def test_read_forecasts_without_parts_and_wrong_schema(tmp_path):
    with pytest.raises(FileNotFoundError):
        rq.read_forecasts(tmp_path)
    bad = tmp_path / "block=1" / "part-000000.parquet"
    bad.parent.mkdir()
    pq.write_table(pa.table({"request_id": ["r"], "step": pa.array([1], pa.int16())}), bad)
    with pytest.raises(ValueError, match="forecasts.*columns"):
        rq.read_forecasts(tmp_path)
    with pytest.raises(ValueError, match="forecast columns"):
        rq.write_forecasts(pd.DataFrame({"request_id": ["r"]}), tmp_path / "x.parquet")


# ------------------------------------------------------------------ validate_forecasts


def test_validate_forecasts_accepts_a_complete_monotone_file():
    requests_df = make_requests(5)
    requests_df["horizon"] = [1, 2, 3, 4, 5]
    rq.validate_forecasts(requests_df, make_forecasts(requests_df))
    # the row order of the forecasts is irrelevant
    rq.validate_forecasts(requests_df, make_forecasts(requests_df).sample(frac=1, random_state=0))


def test_validate_forecasts_tolerates_a_float_noise_crossing_only():
    requests_df = make_requests(1, horizon=2)
    forecasts = make_forecasts(requests_df)
    forecasts.loc[0, "q60"] = forecasts.loc[0, "q50"] - 5e-7  # within the checkpoint's tolerance
    rq.validate_forecasts(requests_df, forecasts)
    forecasts.loc[0, "q60"] = forecasts.loc[0, "q50"] - 1e-3
    with pytest.raises(ValueError, match="r000.*not non-decreasing at step 1"):
        rq.validate_forecasts(requests_df, forecasts)


def test_validate_forecasts_flags_a_missing_step():
    requests_df = make_requests(3, horizon=4)
    forecasts = make_forecasts(requests_df)
    forecasts = forecasts[~((forecasts["request_id"] == "r001") & (forecasts["step"] == 3))]
    with pytest.raises(ValueError, match="r001.*3 forecast steps, expected 1..4"):
        rq.validate_forecasts(requests_df, forecasts)


def test_validate_forecasts_flags_a_request_without_forecasts():
    requests_df = make_requests(3)
    forecasts = make_forecasts(requests_df)
    with pytest.raises(ValueError, match="r002.*0 forecast steps"):
        rq.validate_forecasts(requests_df, forecasts[forecasts["request_id"] != "r002"])


def test_validate_forecasts_flags_duplicate_and_out_of_range_steps():
    requests_df = make_requests(2, horizon=3)
    forecasts = make_forecasts(requests_df)
    dup = pd.concat([forecasts, forecasts.iloc[[4]]], ignore_index=True)
    with pytest.raises(ValueError, match="r001.*duplicate forecast step 2"):
        rq.validate_forecasts(requests_df, dup)
    shifted = forecasts.copy()
    shifted.loc[(shifted["request_id"] == "r001") & (shifted["step"] == 3), "step"] = 4
    with pytest.raises(ValueError, match="r001.*step 4 outside 1..3"):
        rq.validate_forecasts(requests_df, shifted)
    zero = forecasts.copy()
    zero.loc[0, "step"] = 0
    with pytest.raises(ValueError, match="r000.*step 0 outside 1..3"):
        rq.validate_forecasts(requests_df, zero)


def test_validate_forecasts_flags_unknown_request_ids():
    requests_df = make_requests(2)
    forecasts = make_forecasts(requests_df)
    stray = forecasts.iloc[[0]].assign(request_id="zzz")
    with pytest.raises(ValueError, match="zzz.*not in requests"):
        rq.validate_forecasts(requests_df, pd.concat([forecasts, stray], ignore_index=True))


@pytest.mark.parametrize("column", ["point", "q10", "q50", "q90"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_validate_forecasts_flags_non_finite_values(column, bad):
    requests_df = make_requests(2)
    forecasts = make_forecasts(requests_df)
    forecasts.loc[forecasts.index[4], column] = bad  # first step of request r001
    with pytest.raises(ValueError, match="r001.*non-finite"):
        rq.validate_forecasts(requests_df, forecasts)


def test_validate_forecasts_checks_columns_and_unique_requests():
    requests_df = make_requests(2)
    forecasts = make_forecasts(requests_df)
    with pytest.raises(ValueError, match="lack columns.*q10"):
        rq.validate_forecasts(requests_df, forecasts.drop(columns=["q10"]))
    both = pd.concat([requests_df, requests_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="not unique"):
        rq.validate_forecasts(both, forecasts)


def test_validate_forecasts_on_a_partial_run_needs_the_matching_request_subset():
    requests_df = make_requests(4)
    forecasts = make_forecasts(requests_df)
    done = forecasts[forecasts["request_id"].isin(["r000", "r001"])]
    with pytest.raises(ValueError, match="r002"):
        rq.validate_forecasts(requests_df, done)
    rq.validate_forecasts(requests_df.iloc[:2], done)
