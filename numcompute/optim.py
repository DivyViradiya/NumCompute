"""Finite-difference gradients and Jacobians, gradient checking, line search.

Functions take a point ``x`` of shape ``(n,)``. A *scalar function* maps it to
a number; a *vector function* maps it to an array of shape ``(m,)``.

Vectorised evaluation
---------------------
By default ``f`` is called once per perturbed point (2n calls for central
differences). If ``f`` can evaluate a whole batch of points at once, pass
``vectorized=True``: ``f`` then receives an array of shape ``(k, n)`` and must
return shape ``(k,)`` (scalar function) or ``(k, m)`` (vector function), and
only one or two calls are made.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

from .utils import as_float_array

__all__ = ["grad", "jacobian", "gradient_check", "backtracking_line_search"]

_METHODS = ("central", "forward", "backward")


def _check_point(x) -> np.ndarray:
    arr = as_float_array(x, name="x", ndim=1, allow_empty=False)
    if not np.isfinite(arr).all():
        raise ValueError("x must contain only finite values")
    return arr


def _step_vector(h, x: np.ndarray, relative: bool) -> np.ndarray:
    hv = np.broadcast_to(as_float_array(h, name="h"), x.shape).astype(float)
    if not (np.isfinite(hv).all() and (hv > 0).all()):
        raise ValueError("h must be positive and finite")
    return hv * np.maximum(1.0, np.abs(x)) if relative else hv.copy()


def _evaluate(func: Callable, points: np.ndarray, vectorized: bool) -> np.ndarray:
    if not callable(func):
        raise TypeError("func must be callable")
    if vectorized:
        out = np.asarray(func(points), dtype=float)
        if out.ndim == 0 or out.shape[0] != points.shape[0]:
            raise ValueError(
                f"vectorized func must return one row per input point "
                f"({points.shape[0]}), got shape {out.shape}"
            )
    else:
        out = np.stack([np.asarray(func(p), dtype=float) for p in points])
    if not np.isfinite(out).all():
        raise ValueError("func returned a non-finite value (nan/inf) near x")
    return out


def _finite_differences(func, x, h, method, vectorized, relative) -> np.ndarray:
    """Return derivative array of shape (n, *output_shape)."""
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    x = _check_point(x)
    hv = _step_vector(h, x, relative)
    H = np.diag(hv)
    if method == "central":
        fp = _evaluate(func, x + H, vectorized)
        fm = _evaluate(func, x - H, vectorized)
        num, denom = fp - fm, 2.0 * hv
    else:
        f0 = _evaluate(func, x[None, :], vectorized)[0]
        if method == "forward":
            num, denom = _evaluate(func, x + H, vectorized) - f0, hv
        else:
            num, denom = f0 - _evaluate(func, x - H, vectorized), hv
    return num / denom.reshape((-1,) + (1,) * (num.ndim - 1))


def grad(
    f: Callable,
    x,
    h: Union[float, np.ndarray] = 1e-5,
    method: str = "central",
    vectorized: bool = False,
    relative: bool = False,
) -> np.ndarray:
    """Finite-difference gradient of a scalar function.

    Parameters
    ----------
    f : callable
        ``f(x) -> float`` for ``x`` of shape ``(n,)`` (or the batched form
        when ``vectorized=True``).
    x : array_like, shape (n,)
        Point at which to differentiate; finite values only.
    h : float or array_like of shape (n,), optional
        Step size(s), positive. Default 1e-5.
    method : {'central', 'forward', 'backward'}, optional
        ``central``: ``(f(x+h) - f(x-h)) / 2h``, error O(h^2).
        ``forward``: ``(f(x+h) - f(x)) / h``, error O(h).
        ``backward``: ``(f(x) - f(x-h)) / h``, error O(h).
    vectorized : bool, optional
        ``f`` accepts a batch of points (see module docstring).
    relative : bool, optional
        Scale each step by ``max(1, |x_i|)``, useful for large coordinates.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Approximate gradient.

    Raises
    ------
    TypeError
        If ``f`` is not callable or ``x`` is not numeric.
    ValueError
        If ``x`` is empty/non-finite/not 1-D, ``h`` is not positive, ``method``
        is unknown, or ``f`` does not return a finite scalar.

    Notes
    -----
    Too small an ``h`` amplifies rounding error and too large an ``h`` adds
    truncation error; 1e-5 (central) is a good default for smooth functions.
    If ``f`` is dominated by one huge term, small steps in the other
    coordinates can be lost to rounding; use ``relative=True`` or a larger
    ``h`` for such problems.
    Cost: 2n evaluations of ``f`` (central) or n + 1 (forward/backward);
    1-2 calls if vectorised. Time O(n * cost(f)), space O(n^2) for the
    perturbation matrix.

    Examples
    --------
    >>> np.round(grad(lambda v: v[0] ** 2 + 3 * v[1], [2.0, 1.0]), 4).tolist()
    [4.0, 3.0]
    """
    d = _finite_differences(f, x, h, method, vectorized, relative)
    if d.ndim != 1:
        raise ValueError(
            f"f must return a scalar for grad; got output of shape {d.shape[1:]}. "
            "Use jacobian for vector functions"
        )
    return d


def jacobian(
    F: Callable,
    x,
    h: Union[float, np.ndarray] = 1e-5,
    method: str = "central",
    vectorized: bool = False,
    relative: bool = False,
) -> np.ndarray:
    """Finite-difference Jacobian of a vector function.

    Parameters
    ----------
    F : callable
        ``F(x) -> array of shape (m,)`` for ``x`` of shape ``(n,)`` (or the
        batched form when ``vectorized=True``). A scalar result is treated
        as ``m = 1``.
    x : array_like, shape (n,)
        Point at which to differentiate.
    h, method, vectorized, relative
        As in :func:`grad`.

    Returns
    -------
    numpy.ndarray, shape (m, n)
        ``J[i, j] = d F_i / d x_j``.

    Raises
    ------
    TypeError, ValueError
        As in :func:`grad`; also if ``F`` returns more than one dimension.

    Notes
    -----
    Cost: 2n evaluations (central) or n + 1; time O(n * cost(F)),
    space O(n^2 + n m).

    Examples
    --------
    >>> J = jacobian(lambda v: np.array([v[0] * v[1], v[0] + v[1]]), [2.0, 3.0])
    >>> np.round(J, 4).tolist()
    [[3.0, 2.0], [1.0, 1.0]]
    """
    d = _finite_differences(F, x, h, method, vectorized, relative)
    if d.ndim == 1:
        return d.reshape(1, -1)
    if d.ndim != 2:
        raise ValueError(f"F must return a 1-D array of shape (m,); got output shape {d.shape[1:]}")
    return d.T


def gradient_check(
    f: Callable,
    grad_f: Callable,
    x,
    h: float = 1e-5,
    method: str = "central",
) -> float:
    """Compare an analytic gradient with a finite-difference estimate.

    Parameters
    ----------
    f : callable
        Scalar function.
    grad_f : callable
        ``grad_f(x) -> array of shape (n,)``, the analytic gradient.
    x : array_like, shape (n,)
        Test point.
    h, method
        As in :func:`grad`.

    Returns
    -------
    float
        ``||g_num - g_ana|| / max(||g_num|| + ||g_ana||, 1e-12)``; values
        below about 1e-6 indicate a correct gradient.

    Raises
    ------
    ValueError
        If ``grad_f`` returns the wrong shape (plus errors from :func:`grad`).

    Notes
    -----
    Time O(n * cost(f)), space O(n^2).

    Examples
    --------
    >>> gradient_check(lambda v: (v ** 2).sum(), lambda v: 2 * v, [1.0, -2.0]) < 1e-8
    True
    """
    g_num = grad(f, x, h=h, method=method)
    g_ana = as_float_array(grad_f(_check_point(x)), name="grad_f(x)", ndim=1)
    if g_ana.shape != g_num.shape:
        raise ValueError(f"grad_f returned shape {g_ana.shape}, expected {g_num.shape}")
    scale = max(np.linalg.norm(g_num) + np.linalg.norm(g_ana), 1e-12)
    return float(np.linalg.norm(g_num - g_ana) / scale)


def backtracking_line_search(
    f: Callable,
    x,
    direction,
    grad_x: Optional[np.ndarray] = None,
    alpha0: float = 1.0,
    c: float = 1e-4,
    rho: float = 0.5,
    max_iter: int = 50,
) -> float:
    """Armijo backtracking line search along a descent direction.

    Finds ``alpha`` with ``f(x + alpha d) <= f(x) + c * alpha * (g . d)`` by
    starting at ``alpha0`` and multiplying by ``rho`` until the condition holds.

    Parameters
    ----------
    f : callable
        Scalar function of ``x`` with shape ``(n,)``.
    x : array_like, shape (n,)
        Current point.
    direction : array_like, shape (n,)
        Search direction ``d``; must satisfy ``g . d < 0``.
    grad_x : array_like, shape (n,), optional
        Gradient at ``x``; estimated with :func:`grad` if omitted.
    alpha0 : float, optional
        Initial step length, positive. Default 1.
    c : float, optional
        Sufficient-decrease constant in (0, 1). Default 1e-4.
    rho : float, optional
        Shrink factor in (0, 1). Default 0.5.
    max_iter : int, optional
        Maximum number of shrinks. Default 50.

    Returns
    -------
    float
        Accepted step length.

    Raises
    ------
    ValueError
        If a parameter is out of range, shapes differ, or ``direction`` is not
        a descent direction.
    RuntimeError
        If no step satisfies the condition within ``max_iter`` shrinks.

    Notes
    -----
    Time O(max_iter * cost(f)), space O(n).

    Examples
    --------
    >>> f = lambda v: float((v ** 2).sum())
    >>> backtracking_line_search(f, [1.0, 1.0], [-2.0, -2.0], grad_x=[2.0, 2.0])
    0.5
    """
    if not (alpha0 > 0 and 0 < c < 1 and 0 < rho < 1):
        raise ValueError("require alpha0 > 0, 0 < c < 1 and 0 < rho < 1")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"max_iter must be a positive int, got {max_iter!r}")
    x = _check_point(x)
    d = as_float_array(direction, name="direction", ndim=1)
    g = grad(f, x) if grad_x is None else as_float_array(grad_x, name="grad_x", ndim=1)
    if d.shape != x.shape or g.shape != x.shape:
        raise ValueError(f"x, direction and grad_x must share shape {x.shape}")
    slope = float(g @ d)
    if not slope < 0:
        raise ValueError(f"direction is not a descent direction (g.d = {slope:g} >= 0)")
    f0 = float(f(x))
    alpha = float(alpha0)
    for _ in range(int(max_iter)):  # sequential by nature: each trial depends on the last
        if float(f(x + alpha * d)) <= f0 + c * alpha * slope:
            return alpha
        alpha *= rho
    raise RuntimeError(f"line search did not find a sufficient decrease in {max_iter} iterations")
