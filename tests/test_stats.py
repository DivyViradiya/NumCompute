"""Unit tests for numcompute.stats."""
import warnings

import numpy as np
import pytest

from numcompute.stats import (
    RunningStats,
    describe,
    histogram,
    maximum,
    mean,
    median,
    minimum,
    quantile,
    std,
    var,
)


# ------------------------------------------------------ basic descriptive stats
def test_mean_median_std_var_known_values():
    x = [2, 4, 4, 4, 5, 5, 7, 9]
    assert mean(x) == 5.0
    assert median(x) == 4.5
    assert var(x) == 4.0
    assert std(x) == 2.0
    assert minimum(x) == 2 and maximum(x) == 9


def test_ddof_changes_divisor():
    assert np.isclose(var([1, 2, 3, 4], ddof=1), 5 / 3)
    assert np.isclose(std([1, 2, 3, 4], ddof=1), np.sqrt(5 / 3))


def test_ddof_too_large_gives_nan_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(var([5.0], ddof=1))
        assert np.isnan(var([1.0, 2.0], ddof=2))


@pytest.mark.parametrize("axis", [None, 0, 1, 2])
def test_matches_numpy_for_every_axis(axis):
    X = np.random.default_rng(0).normal(size=(4, 5, 6))
    assert np.allclose(mean(X, axis), np.mean(X, axis))
    assert np.allclose(median(X, axis), np.median(X, axis))
    assert np.allclose(std(X, axis), np.std(X, axis))
    assert np.allclose(var(X, axis, ddof=1), np.var(X, axis, ddof=1))
    assert np.allclose(minimum(X, axis), np.min(X, axis))
    assert np.allclose(maximum(X, axis), np.max(X, axis))


def test_output_shapes_follow_axis():
    X = np.ones((4, 5))
    assert np.shape(mean(X)) == ()
    assert mean(X, axis=0).shape == (5,)
    assert mean(X, axis=1).shape == (4,)
    assert mean(X, axis=-1).shape == (4,)


def test_noncontiguous_input():
    X = np.random.default_rng(1).normal(size=(6, 6))[::2, ::-1]
    assert np.allclose(mean(X, axis=1), np.mean(np.ascontiguousarray(X), axis=1))


def test_single_element_and_all_equal():
    assert mean([7.0]) == 7.0 and std([7.0]) == 0.0
    assert std([3.0, 3.0, 3.0]) == 0.0
    assert median([5.0, 5.0, 5.0, 5.0]) == 5.0


def test_nan_propagate_is_default():
    assert np.isnan(mean([1, np.nan, 3]))
    assert np.isnan(median([1, np.nan, 3]))
    assert np.isnan(std([1, np.nan, 3]))
    assert np.isnan(minimum([1, np.nan, 3]))


def test_nan_omit_ignores_nan():
    x = [1, np.nan, 3, 5]
    assert mean(x, nan_policy="omit") == 3
    assert median(x, nan_policy="omit") == 3
    assert np.isclose(std(x, nan_policy="omit"), np.std([1, 3, 5]))
    assert minimum(x, nan_policy="omit") == 1 and maximum(x, nan_policy="omit") == 5


def test_nan_omit_all_nan_slice_is_nan_without_warning():
    X = np.array([[1.0, np.nan], [3.0, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = mean(X, axis=0, nan_policy="omit")
    assert out[0] == 2.0 and np.isnan(out[1])


def test_nan_raise():
    with pytest.raises(ValueError):
        mean([1, np.nan], nan_policy="raise")


def test_infinite_values():
    assert mean([1, np.inf]) == np.inf
    assert maximum([1, np.inf]) == np.inf


def test_invalid_arguments():
    with pytest.raises(ValueError):
        mean([])
    with pytest.raises(ValueError):
        mean([1, 2], axis=1)
    with pytest.raises(ValueError):
        mean([1, 2], nan_policy="ignore")
    with pytest.raises(ValueError):
        var([1, 2], ddof=-1)
    with pytest.raises(ValueError):
        mean(5.0, axis=0)
    with pytest.raises(TypeError):
        mean(["a", "b"])


def test_describe_keys_and_values():
    d = describe([1, 2, 3, 4, 5])
    assert d["count"] == 5 and d["mean"] == 3 and d["median"] == 3
    assert d["min"] == 1 and d["max"] == 5 and d["q25"] == 2 and d["q75"] == 4


def test_describe_omit_counts_non_nan_values():
    d = describe([1, np.nan, 3], nan_policy="omit")
    assert d["count"] == 2 and d["mean"] == 2


def test_describe_with_axis():
    X = np.arange(12.0).reshape(3, 4)
    d = describe(X, axis=0)
    assert d["mean"].shape == (4,) and d["q25"].shape == (4,)


# ---------------------------------------------------------------- RunningStats
def test_running_stats_matches_numpy_in_batches():
    X = np.random.default_rng(2).normal(5, 3, size=(1000, 4))
    rs = RunningStats()
    for chunk in np.array_split(X, 7):
        rs.update(chunk)
    assert np.allclose(rs.mean, X.mean(axis=0))
    assert np.allclose(rs.var(), X.var(axis=0))
    assert np.allclose(rs.var(ddof=1), X.var(axis=0, ddof=1))
    assert np.allclose(rs.std(), X.std(axis=0))
    assert np.array_equal(rs.count, [1000] * 4)


def test_running_stats_one_row_at_a_time_is_welford():
    x = np.random.default_rng(3).normal(size=(200, 2))
    rs = RunningStats()
    for row in x:
        rs.update(row.reshape(1, -1))
    assert np.allclose(rs.mean, x.mean(axis=0)) and np.allclose(rs.var(), x.var(axis=0))


def test_running_stats_is_stable_with_large_offset():
    x = 1e9 + np.random.default_rng(4).normal(size=(5000, 1))
    rs = RunningStats()
    for chunk in np.array_split(x, 10):
        rs.update(chunk)
    assert np.isclose(rs.var()[0], x.var(), rtol=1e-6)


def test_running_stats_skips_nan_per_feature():
    X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]])
    rs = RunningStats().update(X)
    assert np.array_equal(rs.count, [2, 2])
    assert np.allclose(rs.mean, [2.0, 5.0])


def test_running_stats_1d_input_is_one_feature():
    rs = RunningStats().update([1.0, 2.0, 3.0])
    assert rs.mean.shape == (1,) and rs.mean[0] == 2.0


def test_running_stats_merge_equals_single_stream():
    X = np.random.default_rng(5).normal(size=(300, 3))
    a, b = RunningStats().update(X[:120]), RunningStats().update(X[120:])
    a.merge(b)
    assert np.allclose(a.mean, X.mean(axis=0)) and np.allclose(a.var(), X.var(axis=0))


def test_running_stats_empty_and_unseen():
    rs = RunningStats()
    rs.update(np.empty((0, 3)))
    assert rs.count.tolist() == [0, 0, 0]
    assert np.isnan(rs.mean).all() and np.isnan(rs.var()).all()


def test_running_stats_errors():
    rs = RunningStats().update(np.ones((2, 3)))
    with pytest.raises(ValueError):
        rs.update(np.ones((2, 4)))
    with pytest.raises(ValueError):
        rs.update(np.ones((2, 2, 2)))
    with pytest.raises(ValueError):
        rs.var(ddof=-1)
    with pytest.raises(TypeError):
        rs.merge("not stats")
    with pytest.raises(ValueError):
        rs.merge(RunningStats().update(np.ones((1, 5))))


def test_running_stats_works_with_chunked_csv(tmp_path):
    from numcompute.io import iter_csv_chunks

    data = np.random.default_rng(6).normal(size=(50, 2))
    path = tmp_path / "d.csv"
    np.savetxt(path, data, delimiter=",")
    rs = RunningStats()
    for chunk in iter_csv_chunks(path, chunksize=8):
        rs.update(chunk)
    assert np.allclose(rs.mean, data.mean(axis=0))


# ------------------------------------------------------------------- histogram
def test_histogram_matches_numpy():
    x = np.random.default_rng(7).normal(size=500)
    c, e = histogram(x, bins=12)
    c_ref, e_ref = np.histogram(x, bins=12)
    assert np.array_equal(c, c_ref) and np.allclose(e, e_ref)


def test_histogram_known_example_and_right_edge_included():
    c, e = histogram([1, 2, 2, 3, 3, 3], bins=3)
    assert c.tolist() == [1, 2, 3]
    assert e[0] == 1 and e[-1] == 3


def test_histogram_explicit_edges_and_range():
    c, _ = histogram([0.5, 1.5, 2.5, 9], bins=[0, 1, 2, 3])
    assert c.tolist() == [1, 1, 1]  # 9 is outside the edges
    c, e = histogram([1, 2, 3, 10], bins=2, range=(0, 4))
    assert c.tolist() == [1, 2] and e.tolist() == [0, 2, 4]


def test_histogram_density_integrates_to_one():
    x = np.random.default_rng(8).normal(size=1000)
    d, e = histogram(x, bins=20, density=True)
    assert np.isclose((d * np.diff(e)).sum(), 1.0)


def test_histogram_all_equal_values_and_empty():
    c, e = histogram([4, 4, 4], bins=2)
    assert c.sum() == 3 and e[0] == 3.5 and e[-1] == 4.5
    c, e = histogram([], bins=3)
    assert c.tolist() == [0, 0, 0]


def test_histogram_nan_policies():
    c, _ = histogram([1, np.nan, 2, 3], bins=2)
    assert c.sum() == 3
    with pytest.raises(ValueError):
        histogram([1, np.nan], nan_policy="raise")
    with pytest.raises(ValueError):
        histogram([1, 2], nan_policy="propagate")


def test_histogram_invalid_arguments():
    with pytest.raises(ValueError):
        histogram([1, 2], bins=0)
    with pytest.raises(ValueError):
        histogram([1, 2], bins=[1])
    with pytest.raises(ValueError):
        histogram([1, 2], bins=[0, 2, 1])
    with pytest.raises(ValueError):
        histogram([1, np.inf], bins=3)
    with pytest.raises(ValueError):
        histogram([1, 2], bins=2, range=(3, 1))


# -------------------------------------------------------------------- quantile
def test_quantile_matches_numpy():
    x = np.random.default_rng(9).normal(size=(6, 7))
    for axis in (None, 0, 1):
        assert np.allclose(quantile(x, [0.1, 0.5, 0.9], axis=axis), np.quantile(x, [0.1, 0.5, 0.9], axis=axis))


@pytest.mark.parametrize("interpolation", ["linear", "lower", "higher", "midpoint", "nearest"])
def test_quantile_interpolations(interpolation):
    x = [1.0, 2.0, 4.0, 8.0]
    assert np.isclose(quantile(x, 0.4, interpolation=interpolation), np.quantile(x, 0.4, method=interpolation))


def test_quantile_nan_handling():
    x = [1, np.nan, 3, 5]
    assert np.isnan(quantile(x, 0.5))
    assert quantile(x, 0.5, nan_policy="omit") == 3
    with pytest.raises(ValueError):
        quantile(x, 0.5, nan_policy="raise")


def test_quantile_bounds():
    assert quantile([1, 2, 3], 0) == 1 and quantile([1, 2, 3], 1) == 3
    with pytest.raises(ValueError):
        quantile([1, 2, 3], 1.5)
    with pytest.raises(ValueError):
        quantile([1, 2, 3], -0.1)
    with pytest.raises(ValueError):
        quantile([], 0.5)
