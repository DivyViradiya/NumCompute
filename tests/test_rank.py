"""Unit tests for numcompute.rank."""
import numpy as np
import pytest

from numcompute.rank import percentile, rank


def reference_rank(x, method):
    """Slow, obviously-correct ranking used to check the vectorised code."""
    x = np.asarray(x, dtype=float)
    out = np.zeros(len(x))
    uniq = np.unique(x)
    for j, v in enumerate(x):
        less, eq = (x < v).sum(), (x == v).sum()
        out[j] = {
            "min": less + 1,
            "max": less + eq,
            "average": less + (eq + 1) / 2,
            "dense": np.searchsorted(uniq, v) + 1,
        }.get(method, 0)
    if method == "ordinal":
        out[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
    return out


# ------------------------------------------------------------------------ rank
@pytest.mark.parametrize(
    "method, expected",
    [
        ("average", [1, 2.5, 2.5, 4]),
        ("min", [1, 2, 2, 4]),
        ("max", [1, 3, 3, 4]),
        ("dense", [1, 2, 2, 3]),
        ("ordinal", [1, 2, 3, 4]),
    ],
)
def test_rank_methods_on_known_example(method, expected):
    assert rank([10, 20, 20, 30], method=method).tolist() == expected


def test_rank_unsorted_input_keeps_positions():
    assert rank([30, 10, 20]).tolist() == [3, 1, 2]


def test_rank_ordinal_follows_order_of_appearance():
    assert rank([5, 5, 5], method="ordinal").tolist() == [1, 2, 3]


def test_rank_all_equal_values():
    assert rank([2, 2, 2]).tolist() == [2, 2, 2]
    assert rank([2, 2, 2], method="dense").tolist() == [1, 1, 1]


def test_rank_single_and_empty():
    assert rank([5]).tolist() == [1]
    assert rank([]).size == 0


def test_rank_matches_reference_on_random_data():
    rng = np.random.default_rng(0)
    for _ in range(100):
        x = rng.integers(0, 5, size=rng.integers(1, 15)).astype(float)
        for method in ("average", "min", "max", "dense", "ordinal"):
            assert np.array_equal(rank(x, method), reference_rank(x, method))


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_rank_along_each_axis_of_3d_array(axis):
    X = np.random.default_rng(1).integers(0, 4, size=(3, 4, 5)).astype(float)
    expected = np.apply_along_axis(lambda v: reference_rank(v, "average"), axis, X)
    assert np.array_equal(rank(X, axis=axis), expected)


def test_rank_noncontiguous_input():
    X = np.random.default_rng(2).integers(0, 4, size=(3, 4, 5)).astype(float)[::2, ::-1, ::2]
    assert np.array_equal(rank(X, "min", axis=1), rank(np.ascontiguousarray(X), "min", axis=1))


def test_rank_infinities_are_ordinary_values():
    assert rank([np.inf, -np.inf, 0, np.inf]).tolist() == [3.5, 1, 2, 3.5]


def test_rank_nan_omit_gives_nan_rank_and_ignores_it():
    out = rank([3, np.nan, 1, 3, np.nan])
    assert out[0] == 2.5 and out[2] == 1 and out[3] == 2.5
    assert np.isnan(out[1]) and np.isnan(out[4])


def test_rank_nan_propagate_and_raise():
    assert np.isnan(rank([3, np.nan, 1], nan_policy="propagate")).all()
    with pytest.raises(ValueError):
        rank([1, np.nan], nan_policy="raise")


def test_rank_invalid_arguments():
    with pytest.raises(ValueError):
        rank([1], method="median")
    with pytest.raises(ValueError):
        rank([1], nan_policy="ignore")
    with pytest.raises(ValueError):
        rank(3)  # 0-d
    with pytest.raises(ValueError):
        rank([[1, 2]], axis=2)
    with pytest.raises(TypeError):
        rank(["a", "b"])


# ------------------------------------------------------------------ percentile
@pytest.mark.parametrize("interpolation", ["linear", "lower", "higher", "midpoint", "nearest"])
def test_percentile_matches_numpy_for_every_interpolation(interpolation):
    rng = np.random.default_rng(3)
    for _ in range(100):
        x = rng.normal(size=rng.integers(1, 20)).round(1)
        q = rng.uniform(0, 100, size=3)
        assert np.allclose(
            percentile(x, q, interpolation=interpolation),
            np.percentile(x, q, method=interpolation),
        )


def test_percentile_known_values():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 50, interpolation="lower") == 2
    assert percentile([1, 2, 3, 4], 50, interpolation="higher") == 3
    assert percentile([1, 2, 3, 4], 50, interpolation="midpoint") == 2.5


def test_percentile_extremes_and_single_value():
    assert percentile([1, 2, 3, 4], 0) == 1
    assert percentile([1, 2, 3, 4], 100) == 4
    assert percentile([7], 33) == 7


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_percentile_along_axis_matches_numpy(axis):
    Z = np.random.default_rng(4).normal(size=(4, 5, 6))
    assert np.allclose(percentile(Z, [10, 50, 90], axis=axis), np.percentile(Z, [10, 50, 90], axis=axis))
    assert np.allclose(percentile(Z, 37.5, axis=axis), np.percentile(Z, 37.5, axis=axis))


def test_percentile_axis_none_flattens_and_q_dimension_comes_first():
    Z = np.random.default_rng(5).normal(size=(4, 5))
    assert np.allclose(percentile(Z, [25, 75]), np.percentile(Z, [25, 75]))
    assert percentile(Z, [10, 50], axis=0).shape == (2, 5)


def test_percentile_noncontiguous_input():
    Z = np.random.default_rng(6).normal(size=(4, 6))[::2, ::-1]
    assert percentile(Z, 50, axis=1).shape == (2,)


def test_percentile_nan_policies():
    W = np.array([[1, np.nan, 3, 5], [np.nan] * 4])
    assert np.isnan(percentile(W, 50, axis=1)).all()  # propagate
    omitted = percentile(W, 50, axis=1, nan_policy="omit")
    assert omitted[0] == 3 and np.isnan(omitted[1])  # all-NaN row stays NaN
    with pytest.raises(ValueError):
        percentile(W, 50, nan_policy="raise")


def test_percentile_infinite_values():
    assert percentile([1, np.inf], 50) == np.inf
    assert percentile([np.inf, np.inf], 50) == np.inf


def test_percentile_empty_q_list():
    assert percentile([1, 2, 3], []).shape == (0,)


def test_percentile_invalid_arguments():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1], 101)
    with pytest.raises(ValueError):
        percentile([1], -1)
    with pytest.raises(ValueError):
        percentile([1], np.nan)
    with pytest.raises(ValueError):
        percentile([1], [[5]])
    with pytest.raises(ValueError):
        percentile([1], 5, interpolation="cubic")
    with pytest.raises(ValueError):
        percentile(5, 50, axis=0)
    with pytest.raises(TypeError):
        percentile(["a"], 5)
