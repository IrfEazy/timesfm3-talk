"""Quote / gap / validity policy of the v3 studies, on small synthetic frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm3lab.studies import quotes

NAN = np.nan


def _long(rows: list[tuple[str, int, str | None, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "product_id", "subtype", "price"]).assign(
        date=lambda d: pd.to_datetime(d["date"])
    )


def _wide(*rows: list[float]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


# --- wide_prices -----------------------------------------------------------------


def test_wide_prices_is_daily_contiguous_with_nan_for_missing_rows():
    df = _long(
        [
            ("2024-02-08", 2, "Normal", 5.0),
            ("2024-02-08", 1, "Foil", 9.0),
            ("2024-02-11", 1, "Foil", 10.0),
            ("2024-02-10", 2, "Normal", 6.0),
        ]
    )
    dates, skus, mat = quotes.wide_prices(df)

    assert list(dates) == list(pd.date_range("2024-02-08", "2024-02-11"))
    assert dates.freq is not None
    assert list(skus) == [(1, "Foil"), (2, "Normal")]
    assert mat.dtype == np.float64 and mat.shape == (2, 4)
    np.testing.assert_array_equal(mat[0], [9.0, NAN, NAN, 10.0])
    np.testing.assert_array_equal(mat[1], [5.0, NAN, 6.0, NAN])


def test_wide_prices_maps_invalid_prices_to_missing():
    df = _long(
        [
            ("2024-02-08", 1, "Normal", 0.0),
            ("2024-02-09", 1, "Normal", -3.0),
            ("2024-02-10", 1, "Normal", float("nan")),
            ("2024-02-11", 1, "Normal", float("inf")),
            ("2024-02-12", 1, "Normal", -float("inf")),
            ("2024-02-13", 1, "Normal", 0.25),
        ]
    )
    dates, _, mat = quotes.wide_prices(df)

    assert len(dates) == 6  # invalid rows still define the calendar
    np.testing.assert_array_equal(mat[0], [NAN, NAN, NAN, NAN, NAN, 0.25])


def test_wide_prices_none_subtype_is_a_sku_and_float32_input_is_float64():
    df = _long([("2024-02-08", 7, None, 1.5), ("2024-02-08", 7, "Normal", 2.5)])
    df["price"] = df["price"].astype("float32")
    _, skus, mat = quotes.wide_prices(df)

    assert (7, None) in skus and (7, "Normal") in skus
    assert skus.get_loc((7, None)) != skus.get_loc((7, "Normal"))
    assert mat.dtype == np.float64
    assert mat[skus.get_loc((7, None)), 0] == 1.5


def test_wide_prices_object_dtype_missing_subtypes_all_become_none_and_sort_last():
    df = _long([("2024-02-08", 5, "Normal", 1.0), ("2024-02-08", 5, "Foil", 2.0)])
    df["subtype"] = pd.Series(["Normal", None], dtype=object)
    df = pd.concat([df, df.iloc[:1].assign(product_id=4, subtype=np.nan)], ignore_index=True)
    _, skus, mat = quotes.wide_prices(df)

    assert list(skus) == [(4, None), (5, "Normal"), (5, None)]
    assert mat[:, 0].tolist() == [1.0, 1.0, 2.0]


def test_wide_prices_rejects_duplicate_keys_and_missing_columns():
    dup = _long([("2024-02-08", 1, "Normal", 1.0), ("2024-02-08", 1, "Normal", 2.0)])
    with pytest.raises(ValueError, match="share a"):
        quotes.wide_prices(dup)
    with pytest.raises(ValueError, match="missing columns"):
        quotes.wide_prices(dup.drop(columns=["price"]))


def test_wide_prices_empty_frame():
    dates, skus, mat = quotes.wide_prices(_long([]))
    assert len(dates) == 0 and len(skus) == 0 and mat.shape == (0, 0)


# --- quote matrix and observed mask ----------------------------------------------


def test_quote_forward_fills_exactly_three_days_not_four():
    # price on day 0, then missing: days 1-3 are filled, day 4 is not.
    wide = _wide([2.0, NAN, NAN, NAN, NAN, NAN])
    q = quotes.quote_matrix(wide)
    np.testing.assert_array_equal(q[0], [2.0, 2.0, 2.0, 2.0, NAN, NAN])


def test_quote_never_fills_backwards():
    wide = _wide([NAN, NAN, 3.0, NAN])
    q = quotes.quote_matrix(wide)
    np.testing.assert_array_equal(q[0], [NAN, NAN, 3.0, 3.0])


def test_quote_fill_does_not_chain_across_a_long_gap():
    # A new valid price resets the fill; an old one is not revived.
    wide = _wide([1.0, NAN, NAN, NAN, NAN, 5.0, NAN])
    q = quotes.quote_matrix(wide)
    np.testing.assert_array_equal(q[0], [1.0, 1.0, 1.0, 1.0, NAN, 5.0, 5.0])


def test_quote_prefers_the_exact_day_price_over_a_fill():
    wide = _wide([1.0, 4.0, NAN])
    np.testing.assert_array_equal(quotes.quote_matrix(wide)[0], [1.0, 4.0, 4.0])


def test_quote_matrix_respects_max_ffill_and_revalidates_prices():
    wide = _wide([1.0, 0.0, -2.0, np.inf, NAN])
    # zero / negative / inf are not prices: nothing is observed besides day 0
    np.testing.assert_array_equal(
        quotes.quote_matrix(wide, max_ffill=1)[0], [1.0, 1.0, NAN, NAN, NAN]
    )
    np.testing.assert_array_equal(
        quotes.quote_matrix(wide, max_ffill=0)[0], [1.0, NAN, NAN, NAN, NAN]
    )
    with pytest.raises(ValueError):
        quotes.quote_matrix(wide, max_ffill=-1)


def test_quote_matrix_is_causal_per_column():
    rng = np.random.default_rng(0)
    wide = rng.uniform(1, 9, size=(5, 40))
    wide[rng.random(wide.shape) < 0.5] = NAN
    base = quotes.quote_matrix(wide)
    poisoned = wide.copy()
    poisoned[:, 20:] = rng.uniform(1, 9, size=(5, 20))
    np.testing.assert_array_equal(quotes.quote_matrix(poisoned)[:, :20], base[:, :20])


def test_observed_mask_is_the_exact_day_valid_price_not_the_quote():
    wide = _wide([1.0, NAN, 0.0, 2.0])
    obs = quotes.observed_mask(wide)
    q = quotes.quote_matrix(wide)
    assert obs[0].tolist() == [True, False, False, True]
    assert not np.isnan(q[0, 1])  # filled quote on an unobserved day


# --- context ---------------------------------------------------------------------


def test_context_is_the_trailing_run_and_stops_at_a_gap_longer_than_three_days():
    wide = _wide([9.0, 8.0, NAN, NAN, NAN, NAN, 1.0, 2.0, 4.0])
    q = quotes.quote_matrix(wide)
    # day 6 has no valid price within 3 days before it: the q run starts there.
    np.testing.assert_array_equal(quotes.context(q, 0, 8), [1.0, 2.0, 4.0])
    # a gap of exactly 3 days is bridged by the fill
    wide3 = _wide([9.0, 8.0, NAN, NAN, NAN, 1.0, 2.0])
    np.testing.assert_array_equal(
        quotes.context(quotes.quote_matrix(wide3), 0, 6), [9.0, 8.0, 8.0, 8.0, 8.0, 1.0, 2.0]
    )


def test_context_includes_the_origin_and_ignores_later_columns():
    q = _wide([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(quotes.context(q, 0, 2), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(quotes.context(q, 0, 0), [1.0])


def test_context_max_len_keeps_the_most_recent_values():
    q = _wide(np.arange(1.0, 11.0).tolist())
    np.testing.assert_array_equal(quotes.context(q, 0, 9, max_len=4), [7.0, 8.0, 9.0, 10.0])
    assert quotes.context(q, 0, 9).shape == (10,)
    assert quotes.context(np.ones((1, 600)), 0, 599).shape == (quotes.MAX_CONTEXT,)
    assert quotes.context(q, 0, 9, max_len=1).tolist() == [10.0]


def test_context_is_empty_when_the_origin_quote_is_missing():
    q = _wide([1.0, 2.0, NAN, 3.0])
    out = quotes.context(q, 0, 2)
    assert out.shape == (0,) and out.dtype == np.float64
    assert quotes.scaled_context(q, 0, 2).shape == (0,)


def test_context_returns_a_copy():
    q = _wide([1.0, 2.0, 3.0])
    out = quotes.context(q, 0, 2)
    out[:] = 99.0
    assert q.tolist() == [[1.0, 2.0, 3.0]]


def test_scaled_context_ends_at_exactly_one():
    q = _wide([3.0, 7.5, 0.3, 0.7])
    out = quotes.scaled_context(q, 0, 3)
    assert out[-1] == 1.0
    np.testing.assert_allclose(out, np.array([3.0, 7.5, 0.3, 0.7]) / 0.7)
    # origin-relative, not series-relative
    np.testing.assert_allclose(quotes.scaled_context(q, 0, 1), [3.0 / 7.5, 1.0])


@pytest.mark.parametrize(("sku", "origin"), [(-1, 0), (1, 0), (0, -1), (0, 3), (0, 99)])
def test_out_of_range_indices_raise_instead_of_wrapping(sku, origin):
    q = _wide([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="out of range"):
        quotes.context(q, sku, origin)


def test_context_rejects_bad_max_len_and_non_integer_origin():
    q = _wide([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        quotes.context(q, 0, 2, max_len=0)
    with pytest.raises(TypeError):
        quotes.context(q, 0, 1.5)


# --- first observed / observed count ---------------------------------------------


def test_first_observed_idx_is_as_of_the_origin():
    wide = _wide([NAN, NAN, 4.0, NAN, 5.0], [1.0, NAN, NAN, NAN, NAN], [NAN, NAN, NAN, NAN, 6.0])
    assert quotes.first_observed_idx(wide, 0, 4) == 2
    assert quotes.first_observed_idx(wide, 1, 4) == 0
    assert quotes.first_observed_idx(wide, 2, 4) == 4
    assert quotes.first_observed_idx(wide, 0, 1) is None  # first price is in the future
    assert quotes.first_observed_idx(wide, 2, 3) is None
    assert quotes.first_observed_idx_all(wide, 4).tolist() == [2, 0, 4]
    assert quotes.first_observed_idx_all(wide, 1).tolist() == [-1, 0, -1]


def test_first_observed_ignores_non_prices():
    wide = _wide([0.0, -1.0, NAN, np.inf, 2.0])
    assert quotes.first_observed_idx(wide, 0, 4) == 4


def test_n_observed_in_window_counts_valid_exact_day_prices_inclusively():
    wide = _wide([1.0, NAN, 2.0, 0.0, 3.0, 4.0])
    assert quotes.n_observed_in_window(wide, 0, 0, 5) == 4
    assert quotes.n_observed_in_window(wide, 0, 1, 3) == 1
    assert quotes.n_observed_in_window(wide, 0, 2, 2) == 1
    assert quotes.n_observed_in_window(wide, 0, 4, 2) == 0  # empty window
    assert quotes.n_observed_in_window(wide, 0, -10, 2) == 2  # lo clipped to day 0


def test_n_observed_in_window_refuses_a_window_past_the_origin():
    wide = _wide([1.0, 2.0, 3.0, 4.0])
    assert quotes.n_observed_in_window(wide, 0, 0, 2, origin_idx=2) == 3
    with pytest.raises(ValueError, match="after the origin"):
        quotes.n_observed_in_window(wide, 0, 0, 3, origin_idx=2)
    with pytest.raises(ValueError, match="out of range"):
        quotes.n_observed_in_window(wide, 0, 0, 4)


# --- leakage ---------------------------------------------------------------------


def test_future_poisoning_leaves_every_origin_result_unchanged():
    rng = np.random.default_rng(7)
    n_sku, n_days, origin = 6, 80, 50
    wide = rng.uniform(0.5, 50.0, size=(n_sku, n_days))
    wide[rng.random(wide.shape) < 0.4] = NAN
    wide[1, 30 : origin + 1] = NAN  # a SKU with a long gap before the origin
    wide[2, : origin + 1] = NAN  # never observed by the origin
    wide[2, origin + 1 :] = 3.0  # ... but observed afterwards
    poisoned = wide.copy()
    poisoned[:, origin + 1 :] = rng.choice(
        [NAN, 0.0, -1.0, np.inf, 1e9, 0.01], size=(n_sku, n_days - origin - 1)
    )
    assert not np.array_equal(wide[:, origin + 1 :], poisoned[:, origin + 1 :], equal_nan=True)

    q, q_poisoned = quotes.quote_matrix(wide), quotes.quote_matrix(poisoned)
    for sku in range(n_sku):
        for view in (quotes.context, quotes.scaled_context):
            np.testing.assert_array_equal(view(q, sku, origin), view(q_poisoned, sku, origin))
            # poisoning the quote matrix itself after the origin changes nothing either
            q_direct = q.copy()
            q_direct[:, origin + 1 :] = 123.0
            np.testing.assert_array_equal(view(q, sku, origin), view(q_direct, sku, origin))
        assert quotes.first_observed_idx(wide, sku, origin) == quotes.first_observed_idx(
            poisoned, sku, origin
        )
        assert quotes.n_observed_in_window(
            wide, sku, origin - 20, origin, origin_idx=origin
        ) == quotes.n_observed_in_window(poisoned, sku, origin - 20, origin, origin_idx=origin)
    np.testing.assert_array_equal(
        quotes.first_observed_idx_all(wide, origin), quotes.first_observed_idx_all(poisoned, origin)
    )
    assert quotes.first_observed_idx(wide, 2, origin) is None
    assert quotes.context(q, 1, origin).size == 0  # the long gap before the origin


def test_quote_matrix_chunked_equals_unchunked():
    rng = np.random.default_rng(0)
    wide = rng.uniform(0.5, 9, size=(37, 60))
    wide[rng.random(wide.shape) < 0.5] = np.nan
    np.testing.assert_array_equal(
        quotes.quote_matrix(wide, chunk_rows=5), quotes.quote_matrix(wide, chunk_rows=10**6)
    )
