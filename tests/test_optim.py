"""Unit tests for numcompute.optim."""
import numpy as np
import pytest

from numcompute.optim import backtracking_line_search, gradient_check, grad, jacobian


def quad(v):
    return float(np.sum(np.asarray(v) ** 2))


def test_grad_of_quadratic_central():
    assert np.allclose(grad(quad, [1.0, -2.0, 3.0]), [2.0, -4.0, 6.0], atol=1e-8)


def test_grad_known_function_with_cross_terms():
    f = lambda v: v[0] ** 2 + 3 * v[1] + v[0] * v[1]
    assert np.allclose(grad(f, [2.0, 1.0]), [5.0, 5.0], atol=1e-7)


@pytest.mark.parametrize("method, tol", [("central", 1e-8), ("forward", 1e-4), ("backward", 1e-4)])
def test_grad_methods_and_their_accuracy(method, tol):
    x = np.array([0.5, 1.0])
    f = lambda v: np.sin(v[0]) * np.exp(v[1])
    exact = [np.cos(0.5) * np.exp(1.0), np.sin(0.5) * np.exp(1.0)]
    assert np.allclose(grad(f, x, method=method), exact, atol=tol)


def test_central_difference_is_more_accurate_than_forward():
    f = lambda v: np.exp(v[0])
    err_c = abs(grad(f, [1.0], h=1e-4)[0] - np.e)
    err_f = abs(grad(f, [1.0], h=1e-4, method="forward")[0] - np.e)
    assert err_c < err_f


def test_grad_per_coordinate_step_and_relative_step():
    assert np.allclose(grad(quad, [1.0, 2.0], h=np.array([1e-5, 1e-4])), [2.0, 4.0], atol=1e-6)
    f = lambda v: v[0] ** 2
    exact = 2e6
    assert abs(grad(f, [1e6], relative=True)[0] - exact) <= abs(grad(f, [1e6])[0] - exact)
    assert np.isclose(grad(f, [1e6], relative=True)[0], exact, rtol=1e-9)


def test_grad_vectorized_matches_scalar_calls():
    def batched(V):
        V = np.asarray(V)
        return np.sum(V ** 2, axis=-1) if V.ndim == 2 else np.sum(V ** 2)
    x = np.random.default_rng(0).normal(size=6)
    assert np.allclose(grad(batched, x, vectorized=True), grad(quad, x))


def test_grad_does_not_modify_x():
    x = np.array([1.0, 2.0])
    grad(quad, x)
    assert x.tolist() == [1.0, 2.0]


def test_grad_single_variable_and_zero_gradient():
    assert np.allclose(grad(lambda v: v[0] ** 3, [2.0]), [12.0], atol=1e-6)
    assert np.allclose(grad(quad, [0.0, 0.0]), [0.0, 0.0], atol=1e-9)


def test_grad_errors():
    with pytest.raises(ValueError):
        grad(quad, [])
    with pytest.raises(ValueError):
        grad(quad, [[1.0, 2.0]])
    with pytest.raises(ValueError):
        grad(quad, [np.nan, 1.0])
    with pytest.raises(ValueError):
        grad(quad, [1.0], h=0)
    with pytest.raises(ValueError):
        grad(quad, [1.0], h=-1e-5)
    with pytest.raises(ValueError):
        grad(quad, [1.0], method="spline")
    with pytest.raises(ValueError):
        grad(lambda v: v, [1.0, 2.0])  # not scalar valued
    def log_first(v):
        with np.errstate(all="ignore"):
            return float(np.log(v[0]))

    with pytest.raises(ValueError):
        grad(log_first, [0.0])  # nan / -inf at the perturbed points
    with pytest.raises(TypeError):
        grad("not callable", [1.0])
    with pytest.raises(TypeError):
        grad(quad, ["a"])


def test_jacobian_known_function():
    F = lambda v: np.array([v[0] * v[1], v[0] + v[1], v[1] ** 2])
    J = jacobian(F, [2.0, 3.0])
    assert J.shape == (3, 2)
    assert np.allclose(J, [[3, 2], [1, 1], [0, 6]], atol=1e-7)


def test_jacobian_linear_map_recovers_matrix():
    A = np.random.default_rng(1).normal(size=(4, 3))
    assert np.allclose(jacobian(lambda v: A @ v, np.ones(3)), A, atol=1e-7)


@pytest.mark.parametrize("method", ["central", "forward", "backward"])
def test_jacobian_methods(method):
    F = lambda v: np.array([np.sin(v[0]), v[0] * v[1]])
    assert np.allclose(jacobian(F, [0.3, 2.0], method=method), [[np.cos(0.3), 0], [2.0, 0.3]], atol=1e-4)


def test_jacobian_vectorized_matches_loop():
    A = np.random.default_rng(2).normal(size=(3, 4))
    x = np.random.default_rng(3).normal(size=4)
    assert np.allclose(jacobian(lambda V: np.atleast_2d(V) @ A.T if np.ndim(V) == 2 else A @ V, x, vectorized=True), A, atol=1e-7)


def test_jacobian_scalar_function_gives_one_row():
    J = jacobian(quad, [1.0, 2.0])
    assert J.shape == (1, 2) and np.allclose(J, [[2.0, 4.0]], atol=1e-7)


def test_jacobian_rejects_matrix_valued_function():
    with pytest.raises(ValueError):
        jacobian(lambda v: np.outer(v, v), [1.0, 2.0])


def test_gradient_check_accepts_correct_and_rejects_wrong_gradient():
    assert gradient_check(quad, lambda v: 2 * v, [1.0, -2.0]) < 1e-8
    assert gradient_check(quad, lambda v: 3 * v, [1.0, -2.0]) > 0.1
    with pytest.raises(ValueError):
        gradient_check(quad, lambda v: np.array([1.0]), [1.0, 2.0])


def test_line_search_accepts_full_step_when_valid():
    assert backtracking_line_search(quad, [1.0, 1.0], [-1.0, -1.0], grad_x=[2.0, 2.0], alpha0=0.5) == 0.5


def test_line_search_shrinks_step_and_satisfies_armijo():
    f = lambda v: float(np.sum(v ** 2))
    x, d = np.array([1.0, 1.0]), np.array([-4.0, -4.0])
    g = np.array([2.0, 2.0])
    alpha = backtracking_line_search(f, x, d, grad_x=g)
    assert alpha < 1.0
    assert f(x + alpha * d) <= f(x) + 1e-4 * alpha * (g @ d)


def test_line_search_estimates_gradient_when_missing():
    alpha = backtracking_line_search(quad, [1.0, 1.0], [-2.0, -2.0])
    assert alpha == 0.5


def test_line_search_errors():
    with pytest.raises(ValueError):
        backtracking_line_search(quad, [1.0], [1.0], grad_x=[2.0])  # ascent direction
    with pytest.raises(ValueError):
        backtracking_line_search(quad, [1.0], [-1.0], c=2.0)
    with pytest.raises(ValueError):
        backtracking_line_search(quad, [1.0, 1.0], [-1.0], grad_x=[2.0, 2.0])
    with pytest.raises(RuntimeError):
        # a constant function can never show the required decrease
        backtracking_line_search(lambda v: 0.0, [1.0], [-1.0], grad_x=[1.0], max_iter=3)
