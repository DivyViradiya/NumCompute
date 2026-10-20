"""Unit tests for numcompute.utils."""
import numpy as np
import pytest

from numcompute.utils import (
    NotFittedError,
    as_float_array,
    check_2d,
    check_is_fitted,
    check_nan_policy,
    from_2d_along_axis,
    iter_batches,
    leaky_relu,
    log_softmax,
    logsumexp,
    normalize_axis,
    one_hot,
    pairwise_distances,
    relu,
    sigmoid,
    softmax,
    softplus,
    tanh,
    to_2d_along_axis,
)


# ------------------------------------------------------------------ logsumexp
def test_logsumexp_matches_naive_formula_on_small_values():
    a = np.random.default_rng(0).normal(size=(4, 5))
    assert np.allclose(logsumexp(a, axis=1), np.log(np.exp(a).sum(axis=1)))
    assert np.isclose(logsumexp(a), np.log(np.exp(a).sum()))


def test_logsumexp_survives_overflow_and_underflow():
    assert np.isclose(logsumexp([1000.0, 1000.0]), 1000 + np.log(2))
    assert np.isclose(logsumexp([-1000.0, -1000.0]), -1000 + np.log(2))


def test_logsumexp_special_values():
    assert logsumexp([-np.inf, -np.inf]) == -np.inf
    assert logsumexp([np.inf, 1.0]) == np.inf
    assert np.isnan(logsumexp([np.nan, 1.0]))


def test_logsumexp_keepdims_and_shapes():
    a = np.ones((2, 3, 4))
    assert logsumexp(a, axis=1).shape == (2, 4)
    assert logsumexp(a, axis=1, keepdims=True).shape == (2, 1, 4)


def test_logsumexp_empty_raises():
    with pytest.raises(ValueError):
        logsumexp([])


# -------------------------------------------------------------------- softmax
def test_softmax_rows_sum_to_one_and_ordering():
    s = softmax([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    assert np.allclose(s.sum(axis=1), 1.0)
    assert s[0, 2] > s[0, 1] > s[0, 0]
    assert np.allclose(s[1], 1 / 3)


def test_softmax_is_stable_for_huge_logits():
    s = softmax([[1000.0, 1001.0, 1002.0], [-1000.0, 0.0, 1000.0]])
    assert not np.isnan(s).any() and np.allclose(s.sum(axis=1), 1.0)
    assert np.allclose(s[1], [0.0, 0.0, 1.0])


def test_softmax_axis_and_shift_invariance():
    X = np.random.default_rng(1).normal(size=(3, 4))
    assert np.allclose(softmax(X, axis=0).sum(axis=0), 1.0)
    assert np.allclose(softmax(X), softmax(X + 100.0))


def test_log_softmax_matches_log_of_softmax():
    X = np.random.default_rng(2).normal(size=(3, 4))
    assert np.allclose(log_softmax(X), np.log(softmax(X)))
    assert np.isfinite(log_softmax([[1000.0, 0.0]])).all()


def test_softmax_errors():
    with pytest.raises(ValueError):
        softmax([])
    with pytest.raises(ValueError):
        softmax(5.0)
    with pytest.raises(ValueError):
        softmax([[1.0, 2.0]], axis=3)


# ---------------------------------------------------------------- activations
def test_sigmoid_values_and_stability():
    assert np.allclose(sigmoid([-1000.0, 0.0, 1000.0]), [0.0, 0.5, 1.0])
    assert np.allclose(sigmoid([-2.0, 2.0]), 1 / (1 + np.exp([2.0, -2.0])))


def test_relu_leaky_relu_tanh_softplus():
    assert relu([-1.0, 0.0, 2.0]).tolist() == [0.0, 0.0, 2.0]
    assert np.isnan(relu([np.nan])[0])
    assert np.allclose(leaky_relu([-10.0, 5.0], alpha=0.1), [-1.0, 5.0])
    assert np.allclose(tanh([0.0, 1000.0]), [0.0, 1.0])
    assert np.allclose(softplus([1000.0, -1000.0, 0.0]), [1000.0, 0.0, np.log(2)])


def test_activation_rejects_non_numeric():
    with pytest.raises(TypeError):
        sigmoid("abc")


# ------------------------------------------------------------------- one_hot
def test_one_hot_basic_and_explicit_classes():
    assert one_hot([0, 2, 1]).tolist() == [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    assert one_hot([0, 1], n_classes=4).shape == (2, 4)
    assert one_hot([]).shape == (0, 0)


def test_one_hot_errors():
    with pytest.raises(ValueError):
        one_hot([-1, 0])
    with pytest.raises(ValueError):
        one_hot([0, 5], n_classes=3)
    with pytest.raises(ValueError):
        one_hot([[0, 1]])
    with pytest.raises(ValueError):
        one_hot([0.5, 1.0])


# ------------------------------------------------------- pairwise_distances
X_PTS = np.random.default_rng(3).normal(size=(6, 3))
Y_PTS = np.random.default_rng(4).normal(size=(4, 3))


def test_euclidean_matches_brute_force():
    ref = np.sqrt(((X_PTS[:, None] - Y_PTS[None]) ** 2).sum(axis=2))
    assert np.allclose(pairwise_distances(X_PTS, Y_PTS), ref)


def test_self_distances_have_exact_zero_diagonal_and_are_symmetric():
    D = pairwise_distances(X_PTS)
    assert (np.diag(D) == 0).all() and np.allclose(D, D.T)


@pytest.mark.parametrize("metric", ["sqeuclidean", "manhattan", "chebyshev", "cosine"])
def test_other_metrics_against_definitions(metric):
    D = pairwise_distances(X_PTS, Y_PTS, metric=metric)
    diff = X_PTS[:, None] - Y_PTS[None]
    ref = {
        "sqeuclidean": (diff ** 2).sum(axis=2),
        "manhattan": np.abs(diff).sum(axis=2),
        "chebyshev": np.abs(diff).max(axis=2),
        "cosine": 1 - (X_PTS @ Y_PTS.T) / np.outer(np.linalg.norm(X_PTS, axis=1), np.linalg.norm(Y_PTS, axis=1)),
    }[metric]
    assert np.allclose(D, ref)


def test_cosine_zero_vector_has_distance_one():
    assert pairwise_distances([[0.0, 0.0], [1.0, 0.0]], metric="cosine")[0, 1] == 1.0
    assert pairwise_distances([[0.0, 0.0]], metric="cosine")[0, 0] == 1.0


def test_distances_never_negative_for_close_points():
    P = np.full((3, 5), 1e8) + np.random.default_rng(5).normal(size=(3, 5)) * 1e-4
    assert (pairwise_distances(P) >= 0).all()


def test_pairwise_distances_errors():
    with pytest.raises(ValueError):
        pairwise_distances(X_PTS, Y_PTS[:, :2])
    with pytest.raises(ValueError):
        pairwise_distances(X_PTS, metric="hamming")
    with pytest.raises(ValueError):
        pairwise_distances([1.0, 2.0])


# ------------------------------------------------------------------ batching
def test_iter_batches_sizes_and_drop_last():
    assert [len(b) for b in iter_batches(np.arange(10), 4)] == [4, 4, 2]
    assert [len(b) for b in iter_batches(np.arange(10), 4, drop_last=True)] == [4, 4]
    assert list(iter_batches(np.arange(0), 3)) == []


def test_iter_batches_with_targets_and_shuffle_is_reproducible():
    X, y = np.arange(20).reshape(10, 2), np.arange(10)
    batches = list(iter_batches(X, 3, y=y, shuffle=True, seed=1))
    assert np.array_equal(np.sort(np.concatenate([b[1] for b in batches])), y)
    assert all(np.array_equal(bx[:, 0] // 2, by) for bx, by in batches)
    again = list(iter_batches(X, 3, y=y, shuffle=True, seed=1))
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(batches, again))


def test_iter_batches_errors():
    with pytest.raises(ValueError):
        list(iter_batches(np.arange(3), 0))
    with pytest.raises(TypeError):
        list(iter_batches(np.arange(3), 1.5))
    with pytest.raises(ValueError):
        list(iter_batches(np.arange(3), 2, y=np.arange(4)))


# --------------------------------------------------------------- validation
def test_as_float_array_and_check_2d():
    assert as_float_array([1, 2]).dtype == float
    assert as_float_array([1, 2], copy=True) is not None
    with pytest.raises(TypeError):
        as_float_array("abc")
    with pytest.raises(TypeError):
        as_float_array(["a", "b"])
    with pytest.raises(ValueError):
        as_float_array([], allow_empty=False)
    with pytest.raises(ValueError):
        as_float_array([[1]], ndim=1)
    assert check_2d([[1, 2]]).shape == (1, 2)
    with pytest.raises(ValueError):
        check_2d([1, 2])


def test_check_nan_policy_and_normalize_axis():
    assert check_nan_policy("omit") == "omit"
    with pytest.raises(ValueError):
        check_nan_policy("skip")
    assert normalize_axis(-1, 3) == 2 and normalize_axis(0, 3) == 0
    with pytest.raises(ValueError):
        normalize_axis(3, 3)
    with pytest.raises(TypeError):
        normalize_axis(1.0, 3)


def test_check_is_fitted():
    class Model:
        pass

    m = Model()
    with pytest.raises(NotFittedError):
        check_is_fitted(m, "coef_")
    m.coef_ = 1
    check_is_fitted(m, ["coef_"])
    assert issubclass(NotFittedError, ValueError)


def test_to_2d_and_back_round_trip():
    a = np.arange(24).reshape(2, 3, 4)
    a2, shape = to_2d_along_axis(a, 1)
    assert a2.shape == (8, 3)
    assert np.array_equal(from_2d_along_axis(a2, shape, 1), a)
    with pytest.raises(ValueError):
        to_2d_along_axis(np.float64(3), 0)
