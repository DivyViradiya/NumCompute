"""Unit tests for numcompute.sort_search."""
import bisect

import numpy as np
import pytest

from numcompute.sort_search import (
    binary_search,
    binary_search_many,
    multi_key_sort,
    quickselect,
    stable_argsort,
    stable_sort,
    topk,
)


# ---------------------------------------------------------------- stable sort
def test_stable_argsort_ascending_keeps_tie_order():
    assert stable_argsort([3, 1, 2, 1, 3]).tolist() == [1, 3, 2, 0, 4]


def test_stable_argsort_descending_keeps_tie_order():
    assert stable_argsort([3, 1, 2, 1, 3], descending=True).tolist() == [0, 4, 2, 1, 3]


def test_stable_argsort_descending_puts_nan_first():
    assert stable_argsort([1, np.nan, 2], descending=True).tolist() == [1, 2, 0]


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_stable_argsort_matches_numpy_on_2d(axis):
    M = np.random.default_rng(0).integers(0, 4, size=(5, 6))
    assert np.array_equal(stable_argsort(M, axis=axis), np.argsort(M, axis=axis, kind="stable"))


def test_stable_sort_returns_values_and_indices():
    values, idx = stable_sort([3, 1, 2, 1, 3], return_indices=True)
    assert values.tolist() == [1, 1, 2, 3, 3]
    assert idx.tolist() == [1, 3, 2, 0, 4]


def test_stable_sort_empty_and_noncontiguous():
    assert stable_sort([]).size == 0
    M = np.arange(30).reshape(5, 6)[::2, ::-1]
    assert np.array_equal(stable_sort(M, axis=1), np.sort(M, axis=1))


def test_stable_sort_invalid_input():
    with pytest.raises(ValueError):
        stable_argsort(5)
    with pytest.raises(ValueError):
        stable_argsort([1, 2], axis=3)


# -------------------------------------------------------------- multi-key sort
def test_multi_key_sort_two_keys_ascending():
    T = np.array([[2, 1], [1, 5], [2, 0], [1, 5]])
    assert multi_key_sort(T).tolist() == [[1, 5], [1, 5], [2, 0], [2, 1]]


def test_multi_key_sort_mixed_directions():
    T = np.array([[2, 1], [1, 5], [2, 0], [1, 5]])
    out = multi_key_sort(T, keys=[0, 1], ascending=[True, False])
    assert out.tolist() == [[1, 5], [1, 5], [2, 1], [2, 0]]


def test_multi_key_sort_is_stable():
    T = np.array([[2, 1], [1, 5], [2, 0], [1, 5]])
    _, idx = multi_key_sort(T, keys=[0], return_indices=True)
    assert idx.tolist() == [1, 3, 0, 2]


def test_multi_key_sort_descending_strings():
    S = np.array([["b", "1"], ["a", "2"], ["b", "0"]], dtype=object)
    out = multi_key_sort(S, keys=[0, 1], ascending=[False, True])
    assert out.tolist() == [["b", "0"], ["b", "1"], ["a", "2"]]


def test_multi_key_sort_empty_table():
    assert multi_key_sort(np.empty((0, 2))).shape == (0, 2)


def test_multi_key_sort_bad_arguments():
    T = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        multi_key_sort([1, 2])  # not 2-D
    with pytest.raises(ValueError):
        multi_key_sort(T, keys=[5])
    with pytest.raises(ValueError):
        multi_key_sort(T, keys=[])
    with pytest.raises(ValueError):
        multi_key_sort(T, ascending=[True])


# ------------------------------------------------------------------------ topk
def test_topk_basic_values_and_indices():
    values, idx = topk([3, 1, 4, 1, 5], 2)
    assert values.tolist() == [5, 4]
    assert idx.tolist() == [4, 2]


def test_topk_smallest():
    values, idx = topk([3, 1, 4, 1, 5], 2, largest=False)
    assert values.tolist() == [1, 1]
    assert idx.tolist() == [1, 3]


def test_topk_k_zero_and_k_larger_than_n():
    assert topk([1, 2, 3], 0)[0].size == 0
    assert topk([1, 2, 3], 10)[0].tolist() == [3, 2, 1]


def test_topk_tie_at_boundary_uses_lowest_index():
    assert topk([5, 5, 5, 5], 2)[1].tolist() == [0, 1]


def test_topk_nan_is_never_preferred():
    values, idx = topk([np.nan, 2.0, np.nan, 1.0], 2)
    assert values.tolist() == [2.0, 1.0]
    assert idx.tolist() == [1, 3]
    # NaN is only returned when there are not enough real numbers
    assert topk([np.nan, np.nan, 1.0], 2)[1].tolist() == [2, 0]


def test_topk_return_indices_false():
    assert topk([1, 2, 3], 2, return_indices=False).tolist() == [3, 2]


def test_topk_infinities_and_integer_extremes():
    assert topk([-np.inf, np.inf, 0.0], 1)[0].tolist() == [np.inf]
    big = np.array([-(2**63), 5], dtype=np.int64)
    assert topk(big, 1, largest=False)[0].tolist() == [-(2**63)]
    assert topk(np.array([3, 1, 2], dtype=np.uint8), 2)[0].tolist() == [3, 2]


@pytest.mark.parametrize("largest", [True, False])
def test_topk_matches_full_stable_sort_on_random_data(largest):
    rng = np.random.default_rng(1)
    for _ in range(100):
        x = rng.integers(0, 6, size=rng.integers(1, 12)).astype(float)
        k = int(rng.integers(0, len(x) + 3))
        values, idx = topk(x, k, largest=largest)
        expected = stable_argsort(-x if largest else x)[: min(k, len(x))]
        assert idx.tolist() == expected.tolist()
        assert np.array_equal(values, x[expected])


def test_topk_2d_axes():
    M = np.random.default_rng(2).normal(size=(4, 7))
    values, idx = topk(M, 3, axis=1)
    assert np.array_equal(values, -np.sort(-M, axis=1)[:, :3])
    assert np.array_equal(np.take_along_axis(M, idx, axis=1), values)
    values0, _ = topk(M, 2, axis=0, largest=False)
    assert values0.shape == (2, 7)
    assert np.array_equal(values0, np.sort(M, axis=0)[:2])


def test_topk_unsorted_output_is_in_input_order():
    M = np.random.default_rng(3).normal(size=(4, 7))
    _, idx = topk(M, 3, axis=1, sort=False)
    assert (np.diff(idx, axis=1) > 0).all()


def test_topk_noncontiguous_and_empty_inputs():
    M = np.random.default_rng(4).normal(size=(6, 6))[::2, ::-1]
    assert topk(M, 2)[0].shape == (3, 2)
    assert topk(np.empty((3, 0)), 2)[0].shape == (3, 0)
    assert topk(np.empty((0, 5)), 2)[0].shape == (0, 2)


def test_topk_invalid_arguments():
    with pytest.raises(ValueError):
        topk([1], -1)
    with pytest.raises(TypeError):
        topk([1], 1.5)
    with pytest.raises(TypeError):
        topk([1], True)
    with pytest.raises(TypeError):
        topk(["a", "b"], 1)
    with pytest.raises(ValueError):
        topk(5, 1)


# ----------------------------------------------------------------- quickselect
def test_quickselect_matches_sorted_value():
    rng = np.random.default_rng(5)
    for _ in range(100):
        x = rng.integers(0, 8, size=rng.integers(1, 30))
        k = int(rng.integers(0, len(x)))
        assert quickselect(x, k, seed=1) == np.sort(x)[k]


def test_quickselect_all_equal_and_single():
    assert quickselect(np.full(1000, 7), 500) == 7
    assert quickselect([2.5], 0) == 2.5


def test_quickselect_min_and_max():
    x = [9, 4, 7, 1, 8]
    assert quickselect(x, 0) == 1
    assert quickselect(x, 4) == 9


def test_quickselect_invalid_arguments():
    with pytest.raises(ValueError):
        quickselect([], 0)
    with pytest.raises(ValueError):
        quickselect([1, 2], 2)
    with pytest.raises(ValueError):
        quickselect([1, 2], -1)
    with pytest.raises(ValueError):
        quickselect([1.0, np.nan], 0)
    with pytest.raises(ValueError):
        quickselect([[1, 2]], 0)
    with pytest.raises(TypeError):
        quickselect([1, 2], 0.5)


# --------------------------------------------------------------- binary search
def test_binary_search_found_and_missing():
    assert binary_search([1, 3, 3, 5], 3) == (1, True)
    assert binary_search([1, 3, 3, 5], 4) == (3, False)


def test_binary_search_duplicates_left_vs_right():
    arr = [1, 3, 3, 3, 5]
    assert binary_search(arr, 3, side="left") == (1, True)
    assert binary_search(arr, 3, side="right") == (4, True)


def test_binary_search_below_above_and_empty():
    assert binary_search([1, 2, 3], 0) == (0, False)
    assert binary_search([1, 2, 3], 9) == (3, False)
    assert binary_search([], 1) == (0, False)


def test_binary_search_matches_bisect_on_random_data():
    rng = np.random.default_rng(6)
    for _ in range(200):
        s = sorted(rng.integers(0, 8, size=rng.integers(0, 10)).tolist())
        x = int(rng.integers(-1, 9))
        assert binary_search(s, x, "left") == (bisect.bisect_left(s, x), x in s)
        idx, found = binary_search(s, x, "right")
        assert idx == bisect.bisect_right(s, x) and found == (x in s)


def test_binary_search_return_types():
    idx, found = binary_search([1, 2, 3], 2)
    assert isinstance(idx, int) and isinstance(found, bool)


def test_binary_search_nan_query():
    assert binary_search([1.0, 2.0, np.nan, np.nan], np.nan) == (2, False)


def test_binary_search_invalid_arguments():
    with pytest.raises(ValueError):
        binary_search([[1, 2]], 1)
    with pytest.raises(ValueError):
        binary_search([1, 2], 1, side="middle")


def test_binary_search_many_matches_scalar_version():
    arr = np.array([1, 3, 3, 5])
    queries = np.array([0, 1, 3, 4, 9])
    for side in ("left", "right"):
        idx, found = binary_search_many(arr, queries, side=side)
        for q, i, f in zip(queries, idx, found):
            assert (int(i), bool(f)) == binary_search(arr, q, side)


def test_binary_search_many_empty_array_and_bad_shape():
    idx, found = binary_search_many([], [1, 2])
    assert idx.tolist() == [0, 0] and not found.any()
    with pytest.raises(ValueError):
        binary_search_many([1], [[1]])
