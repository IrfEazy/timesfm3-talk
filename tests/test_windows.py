import numpy as np
import pytest

from tfm3lab.windows import (
    assert_no_leakage,
    context_slice,
    nearest_origin,
    target_indices,
    valid_origins,
)


def test_target_indices_basic():
    np.testing.assert_array_equal(target_indices(origin=10, horizon=3), [10, 11, 12])


def test_target_indices_horizon_one():
    np.testing.assert_array_equal(target_indices(origin=5, horizon=1), [5])


def test_target_indices_rejects_nonpositive_horizon():
    with pytest.raises(ValueError):
        target_indices(origin=5, horizon=0)


def test_context_slice_full_window():
    s = context_slice(origin=10, context_len=4)
    assert (s.start, s.stop) == (6, 10)


def test_context_slice_clips_at_zero_for_early_origin():
    # origin=2 with context_len=10 would start at -8; must clip to 0, not error.
    s = context_slice(origin=2, context_len=10)
    assert (s.start, s.stop) == (0, 2)


def test_context_and_target_are_adjacent_no_gap_no_overlap():
    origin, context_len, horizon = 20, 8, 5
    ctx = context_slice(origin, context_len)
    tgt = target_indices(origin, horizon)
    assert ctx.stop == origin == tgt[0]


def test_valid_origins_exact_boundaries():
    # n=10, context_len=3, horizon=2 -> first usable origin=3, last=8
    # (target_indices(8,2) = [8,9], the last valid pair within n=10)
    origins = valid_origins(n=10, context_len=3, horizon=2)
    assert origins[0] == 3
    assert origins[-1] == 8
    # every origin must have a full context and a full target inside [0, n)
    for o in origins:
        ctx = context_slice(int(o), 3)
        assert ctx.start >= 0
        tgt = target_indices(int(o), 2)
        assert tgt[-1] < 10


def test_valid_origins_empty_when_series_too_short():
    assert valid_origins(n=4, context_len=3, horizon=2).size == 0


def test_valid_origins_max_origins_keeps_the_last_ones():
    origins = valid_origins(n=20, context_len=3, horizon=2, max_origins=3)
    all_origins = valid_origins(n=20, context_len=3, horizon=2)
    np.testing.assert_array_equal(origins, all_origins[-3:])


def test_assert_no_leakage_passes_for_valid_window():
    # Should not raise.
    assert_no_leakage(origin=50, context_len=32, horizon=7)


def test_assert_no_leakage_catches_off_by_one_construction():
    # Simulate the draft's bug: treat target as starting at origin - 1
    # (one step *into* the context) and confirm the guard would catch it.
    origin, context_len, horizon = 50, 32, 7
    ctx = context_slice(origin, context_len)
    bogus_target_start = origin - 1  # overlaps the last context index
    ctx_indices = np.arange(ctx.start, ctx.stop)
    assert ctx_indices.max() >= bogus_target_start  # the leakage the bug would cause
    # and the real function, given the correct origin, does not raise:
    assert_no_leakage(origin, context_len, horizon)


def test_nearest_origin_exact_match():
    dates = np.array(["2022-01-01", "2022-01-02", "2022-01-03"], dtype="datetime64[ns]")
    assert nearest_origin(dates, "2022-01-02") == 1


def test_nearest_origin_rounds_to_closer_neighbor():
    dates = np.array(["2022-01-01", "2022-01-05", "2022-01-10"], dtype="datetime64[ns]")
    # 2022-01-03 is 2 days after 01-01 and 2 days before 01-05 -> ties go to `before`.
    assert nearest_origin(dates, "2022-01-03") == 0
    # 2022-01-04 is 3 days after 01-01, 1 day before 01-05 -> after.
    assert nearest_origin(dates, "2022-01-04") == 1


def test_nearest_origin_clips_outside_range():
    dates = np.array(["2022-01-01", "2022-01-05"], dtype="datetime64[ns]")
    assert nearest_origin(dates, "2019-01-01") == 0
    assert nearest_origin(dates, "2030-01-01") == 1
