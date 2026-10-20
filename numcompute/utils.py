"""Shared helpers: validation, stable numerics, distances, batching.

Everything here is plain NumPy. Other modules import from this one, so it
must not import any other ``numcompute`` module.
"""
from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "NotFittedError",
    "NAN_POLICIES",
    "check_nan_policy",
    "normalize_axis",
    "as_float_array",
    "check_2d",
    "check_is_fitted",
    "to_2d_along_axis",
    "from_2d_along_axis",
    "logsumexp",
    "softmax",
    "log_softmax",
    "sigmoid",
    "relu",
    "leaky_relu",
    "tanh",
    "softplus",
    "one_hot",
    "pairwise_distances",
    "iter_batches",
]

NAN_POLICIES = ("propagate", "omit", "raise")


class NotFittedError(ValueError, AttributeError):
    """Raised when an estimator is used before ``fit`` has been called."""


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------
def check_nan_policy(nan_policy: str, allowed: Sequence[str] = NAN_POLICIES) -> str:
    """Validate a ``nan_policy`` string.

    Parameters
    ----------
    nan_policy : str
        Value supplied by the caller.
    allowed : sequence of str, optional
        Accepted values. Default ``('propagate', 'omit', 'raise')``.

    Returns
    -------
    str
        ``nan_policy`` unchanged, if valid.

    Raises
    ------
    ValueError
        If ``nan_policy`` is not in ``allowed``.

    Notes
    -----
    Time O(1), space O(1).
    """
    if nan_policy not in allowed:
        raise ValueError(
            f"nan_policy must be one of {tuple(allowed)}, got {nan_policy!r}"
        )
    return nan_policy


def normalize_axis(axis: int, ndim: int) -> int:
    """Convert a possibly negative ``axis`` to ``0 <= axis < ndim``.

    Parameters
    ----------
    axis : int
        Axis supplied by the caller (negative values count from the end).
    ndim : int
        Number of dimensions of the array.

    Returns
    -------
    int
        Non-negative axis.

    Raises
    ------
    TypeError
        If ``axis`` is not an integer.
    ValueError
        If ``axis`` is out of range for ``ndim``.

    Notes
    -----
    Time O(1), space O(1).
    """
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise TypeError(f"axis must be an int, got {type(axis).__name__}")
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of ndim {ndim}")
    return int(axis) % ndim if ndim else 0


def as_float_array(
    x,
    name: str = "x",
    ndim: Union[int, Tuple[int, ...], None] = None,
    allow_empty: bool = True,
    copy: bool = False,
) -> np.ndarray:
    """Convert ``x`` to a float64 ndarray with validation.

    Parameters
    ----------
    x : array_like
        Input data. Strings are rejected.
    name : str, optional
        Name used in error messages.
    ndim : int or tuple of int, optional
        Allowed number of dimensions. ``None`` allows any.
    allow_empty : bool, optional
        If False, an array with zero elements raises ``ValueError``.
    copy : bool, optional
        Always return a copy.

    Returns
    -------
    numpy.ndarray
        Float64 array (same shape as ``x``).

    Raises
    ------
    TypeError
        If ``x`` cannot be converted to numbers.
    ValueError
        If the number of dimensions is not allowed or the array is empty
        when ``allow_empty=False``.

    Notes
    -----
    Time O(n) when a conversion or copy happens, otherwise O(1); space O(n)
    when copying.
    """
    if isinstance(x, (str, bytes)):
        raise TypeError(f"{name} must be a numeric array-like, got {type(x).__name__}")
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array-like: {exc}") from exc
    if ndim is not None:
        allowed = (ndim,) if isinstance(ndim, int) else tuple(ndim)
        if arr.ndim not in allowed:
            raise ValueError(
                f"{name} must have ndim in {allowed}, got ndim={arr.ndim} "
                f"(shape {arr.shape})"
            )
    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} must not be empty (shape {arr.shape})")
    return arr.copy() if copy else arr


def check_2d(X, name: str = "X", allow_empty: bool = True) -> np.ndarray:
    """Return ``X`` as a 2-D float array of shape ``(n_samples, n_features)``.

    Parameters
    ----------
    X : array_like
        Input data.
    name : str, optional
        Name used in error messages.
    allow_empty : bool, optional
        If False, zero rows or zero columns raise ``ValueError``.

    Returns
    -------
    numpy.ndarray
        Float64 array with ``ndim == 2``.

    Raises
    ------
    TypeError
        If ``X`` is not numeric.
    ValueError
        If ``X`` is not 2-D (reshape 1-D data with ``X.reshape(-1, 1)``).

    Notes
    -----
    Time O(1) to O(n); space O(1) to O(n).
    """
    arr = as_float_array(X, name=name, ndim=2, allow_empty=allow_empty)
    return arr


def check_is_fitted(estimator, attributes: Union[str, Sequence[str]]) -> None:
    """Raise :class:`NotFittedError` if ``estimator`` lacks fitted attributes.

    Parameters
    ----------
    estimator : object
        Estimator or transformer instance.
    attributes : str or sequence of str
        Attribute names that ``fit`` is expected to create.

    Raises
    ------
    NotFittedError
        If any attribute is missing.

    Notes
    -----
    Time O(len(attributes)), space O(1).
    """
    names = [attributes] if isinstance(attributes, str) else list(attributes)
    missing = [a for a in names if not hasattr(estimator, a)]
    if missing:
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet; "
            f"call 'fit' before using it (missing: {missing})."
        )


def to_2d_along_axis(a: np.ndarray, axis: int) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Reshape ``a`` to 2-D so that ``axis`` becomes the last axis.

    Parameters
    ----------
    a : numpy.ndarray
        Array with ``ndim >= 1``.
    axis : int
        Axis to move to the end.

    Returns
    -------
    a2 : numpy.ndarray
        Shape ``(prod(other dims), a.shape[axis])``.
    moved_shape : tuple of int
        Shape of ``a`` after ``np.moveaxis(a, axis, -1)``; pass to
        :func:`from_2d_along_axis` to undo.

    Raises
    ------
    ValueError
        If ``a`` is 0-d or ``axis`` is out of range.

    Notes
    -----
    Time O(n) (copy if non-contiguous), space O(n).
    """
    a = np.asarray(a)
    if a.ndim == 0:
        raise ValueError("expected an array with at least one dimension")
    ax = normalize_axis(axis, a.ndim)
    moved = np.moveaxis(a, ax, -1)
    n = moved.shape[-1]
    return moved.reshape(-1, n), moved.shape


def from_2d_along_axis(a2: np.ndarray, moved_shape: Tuple[int, ...], axis: int) -> np.ndarray:
    """Inverse of :func:`to_2d_along_axis`.

    Parameters
    ----------
    a2 : numpy.ndarray
        2-D array whose last axis has length ``moved_shape[-1]``.
    moved_shape : tuple of int
        Shape returned by :func:`to_2d_along_axis`.
    axis : int
        The original axis.

    Returns
    -------
    numpy.ndarray
        Array of the original shape.

    Notes
    -----
    Time O(1) (view) or O(n); space O(1) or O(n).
    """
    out = a2.reshape(moved_shape)
    return np.moveaxis(out, -1, normalize_axis(axis, len(moved_shape)))


# --------------------------------------------------------------------------
# Stable numerics
# --------------------------------------------------------------------------
def logsumexp(a, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False):
    """Compute ``log(sum(exp(a)))`` without overflow or underflow.

    Uses the max-shift identity ``max + log(sum(exp(a - max)))``.

    Parameters
    ----------
    a : array_like
        Input values, any shape.
    axis : int or tuple of int, optional
        Axis or axes to reduce. ``None`` reduces over all elements.
    keepdims : bool, optional
        Keep reduced axes with size 1.

    Returns
    -------
    float or numpy.ndarray
        Reduced array (scalar if everything is reduced).

    Raises
    ------
    TypeError
        If ``a`` is not numeric.
    ValueError
        If ``a`` is empty.

    Notes
    -----
    ``-inf`` entries are handled (all ``-inf`` gives ``-inf``); NaN propagates.
    Time O(n), space O(n).

    Examples
    --------
    >>> float(logsumexp([1000.0, 1000.0]))  # naive exp() would overflow
    1000.6931471805599
    """
    arr = as_float_array(a, name="a", allow_empty=False)
    amax = np.max(arr, axis=axis, keepdims=True)
    amax_safe = np.where(np.isfinite(amax), amax, 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        total = np.sum(np.exp(arr - amax_safe), axis=axis, keepdims=True)
        out = np.log(total) + amax_safe
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out[()] if out.ndim == 0 else out


def softmax(x, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Parameters
    ----------
    x : array_like
        Logits, any shape with ``ndim >= 1``.
    axis : int, optional
        Axis along which probabilities sum to 1. Default -1.

    Returns
    -------
    numpy.ndarray
        Same shape as ``x``; entries in [0, 1] summing to 1 along ``axis``.

    Raises
    ------
    TypeError
        If ``x`` is not numeric.
    ValueError
        If ``x`` is 0-d, empty, or ``axis`` is invalid.

    Notes
    -----
    Computed as ``exp(x - logsumexp(x))`` so large logits never overflow.
    Time O(n), space O(n).
    """
    arr = as_float_array(x, name="x", allow_empty=False)
    if arr.ndim == 0:
        raise ValueError("softmax requires at least a 1-D input")
    ax = normalize_axis(axis, arr.ndim)
    with np.errstate(invalid="ignore", over="ignore"):
        return np.exp(arr - logsumexp(arr, axis=ax, keepdims=True))


def log_softmax(x, axis: int = -1) -> np.ndarray:
    """Numerically stable ``log(softmax(x))``.

    Parameters
    ----------
    x : array_like
        Logits, ``ndim >= 1``.
    axis : int, optional
        Axis to normalise over. Default -1.

    Returns
    -------
    numpy.ndarray
        Same shape as ``x``.

    Raises
    ------
    TypeError, ValueError
        As for :func:`softmax`.

    Notes
    -----
    Time O(n), space O(n).
    """
    arr = as_float_array(x, name="x", allow_empty=False)
    if arr.ndim == 0:
        raise ValueError("log_softmax requires at least a 1-D input")
    ax = normalize_axis(axis, arr.ndim)
    return arr - logsumexp(arr, axis=ax, keepdims=True)


def sigmoid(x) -> np.ndarray:
    """Stable logistic function ``1 / (1 + exp(-x))``.

    Parameters
    ----------
    x : array_like
        Any shape.

    Returns
    -------
    numpy.ndarray
        Same shape as ``x``; values in [0, 1], no overflow warnings.

    Raises
    ------
    TypeError
        If ``x`` is not numeric.

    Notes
    -----
    Uses ``exp(-|x|)`` so the exponent is never positive. Time O(n), space O(n).
    """
    arr = as_float_array(x, name="x")
    e = np.exp(-np.abs(arr))
    return np.where(arr >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def relu(x) -> np.ndarray:
    """Rectified linear unit ``max(0, x)``.

    Parameters
    ----------
    x : array_like
        Any shape.

    Returns
    -------
    numpy.ndarray
        Same shape as ``x``. NaN stays NaN.

    Raises
    ------
    TypeError
        If ``x`` is not numeric.

    Notes
    -----
    Time O(n), space O(n).
    """
    arr = as_float_array(x, name="x")
    return np.where(np.isnan(arr), np.nan, np.maximum(arr, 0.0))


def leaky_relu(x, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU: ``x`` if ``x > 0`` else ``alpha * x``.

    Parameters
    ----------
    x : array_like
        Any shape.
    alpha : float, optional
        Slope for negative inputs. Default 0.01.

    Returns
    -------
    numpy.ndarray
        Same shape as ``x``.

    Raises
    ------
    TypeError
        If ``x`` is not numeric.

    Notes
    -----
    Time O(n), space O(n).
    """
    arr = as_float_array(x, name="x")
    return np.where(arr > 0, arr, alpha * arr)


def tanh(x) -> np.ndarray:
    """Hyperbolic tangent (overflow-safe wrapper around ``numpy.tanh``).

    Parameters
    ----------
    x : array_like
        Any shape.

    Returns
    -------
    numpy.ndarray
        Same shape as ``x``; values in [-1, 1].

    Raises
    ------
    TypeError
        If ``x`` is not numeric.

    Notes
    -----
    Time O(n), space O(n).
    """
    return np.tanh(as_float_array(x, name="x"))


def softplus(x) -> np.ndarray:
    """Stable ``log(1 + exp(x))``.

    Parameters
    ----------
    x : array_like
        Any shape.

    Returns
    -------
    numpy.ndarray
        Same shape as ``x``.

    Raises
    ------
    TypeError
        If ``x`` is not numeric.

    Notes
    -----
    Implemented as ``logaddexp(0, x)``. Time O(n), space O(n).
    """
    return np.logaddexp(0.0, as_float_array(x, name="x"))


def one_hot(labels, n_classes: Optional[int] = None) -> np.ndarray:
    """One-hot encode non-negative integer labels.

    Parameters
    ----------
    labels : array_like of int, shape (n,)
        Class indices in ``[0, n_classes)``.
    n_classes : int, optional
        Number of classes. Defaults to ``max(labels) + 1``.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(n, n_classes)``.

    Raises
    ------
    ValueError
        If labels are not 1-D, negative, non-integer, or ``>= n_classes``.

    Notes
    -----
    Time O(n * n_classes), space O(n * n_classes).
    """
    lab = np.asarray(labels)
    if lab.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {lab.shape}")
    if lab.size and (lab.dtype.kind not in "iu"):
        if lab.dtype.kind == "f" and np.all(lab == np.floor(lab)):
            lab = lab.astype(int)
        else:
            raise ValueError("labels must be integers")
    if lab.size and lab.min() < 0:
        raise ValueError("labels must be non-negative")
    k = int(lab.max()) + 1 if n_classes is None and lab.size else (n_classes or 0)
    if lab.size and lab.max() >= k:
        raise ValueError(f"label {int(lab.max())} is out of range for n_classes={k}")
    lab = lab.astype(np.intp, copy=False)  # empty input arrives as float64
    out = np.zeros((lab.size, k))
    out[np.arange(lab.size), lab] = 1.0
    return out


# --------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------
_METRICS = ("euclidean", "sqeuclidean", "manhattan", "chebyshev", "cosine")


def pairwise_distances(X, Y=None, metric: str = "euclidean") -> np.ndarray:
    """Distance matrix between rows of ``X`` and rows of ``Y``.

    Parameters
    ----------
    X : array_like, shape (n, d)
        First set of points.
    Y : array_like, shape (m, d), optional
        Second set of points. Defaults to ``X``.
    metric : {'euclidean', 'sqeuclidean', 'manhattan', 'chebyshev', 'cosine'}
        Distance type. ``cosine`` is ``1 - cos_similarity``; a zero vector
        has cosine distance 1 to everything.

    Returns
    -------
    numpy.ndarray, shape (n, m)
        Non-negative distances. When ``Y`` is None the diagonal is exactly 0.

    Raises
    ------
    TypeError
        If inputs are not numeric.
    ValueError
        If inputs are not 2-D, feature counts differ, or ``metric`` is unknown.

    Notes
    -----
    Euclidean uses ``|x|^2 + |y|^2 - 2 x.y`` (BLAS, fast) and clips tiny
    negatives caused by rounding. Time O(n m d), space O(n m) for euclidean/
    cosine; manhattan and chebyshev build an O(n m d) intermediate.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
    A = check_2d(X, "X")
    B = A if Y is None else check_2d(Y, "Y")
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"X and Y must have the same number of features, got "
            f"{A.shape[1]} and {B.shape[1]}"
        )
    if metric in ("euclidean", "sqeuclidean"):
        sq = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
        np.maximum(sq, 0.0, out=sq)
        if Y is None:
            np.fill_diagonal(sq, 0.0)
        return np.sqrt(sq) if metric == "euclidean" else sq
    if metric == "manhattan":
        return np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)
    if metric == "chebyshev":
        return np.abs(A[:, None, :] - B[None, :, :]).max(axis=2) if A.shape[1] else np.zeros((len(A), len(B)))
    # cosine
    na = np.linalg.norm(A, axis=1)[:, None]
    nb = np.linalg.norm(B, axis=1)[None, :]
    denom = na * nb
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(denom > 0, (A @ B.T) / np.where(denom > 0, denom, 1.0), 0.0)
    out = 1.0 - sim
    np.clip(out, 0.0, 2.0, out=out)
    if Y is None:
        d = np.diag_indices(min(out.shape))
        out[d] = np.where(np.diag(denom) > 0, 0.0, 1.0)
    return out


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------
def iter_batches(
    X,
    batch_size: int,
    y=None,
    shuffle: bool = False,
    seed: Optional[int] = None,
    drop_last: bool = False,
) -> Iterator:
    """Yield mini-batches of rows from ``X`` (and ``y``).

    Parameters
    ----------
    X : array_like, shape (n, ...)
        Data; batches are taken along axis 0.
    batch_size : int
        Rows per batch, ``>= 1``.
    y : array_like, shape (n, ...), optional
        Targets, batched in step with ``X``.
    shuffle : bool, optional
        Shuffle row order first (reproducible with ``seed``).
    seed : int, optional
        Seed for the shuffle.
    drop_last : bool, optional
        Drop a final batch smaller than ``batch_size``.

    Yields
    ------
    numpy.ndarray or tuple of numpy.ndarray
        ``X_batch`` if ``y`` is None, else ``(X_batch, y_batch)``.

    Raises
    ------
    TypeError
        If ``batch_size`` is not an integer.
    ValueError
        If ``batch_size < 1``, ``X`` is 0-d, or lengths of ``X`` and ``y`` differ.

    Notes
    -----
    The Python-level loop runs once per batch (n / batch_size iterations), not
    per element. Time O(n) total, space O(batch_size) per batch (O(n) index
    array when shuffling).
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise TypeError(f"batch_size must be an int, got {type(batch_size).__name__}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    Xa = np.asarray(X)
    if Xa.ndim == 0:
        raise ValueError("X must have at least one dimension")
    n = Xa.shape[0]
    ya = None if y is None else np.asarray(y)
    if ya is not None and ya.shape[0] != n:
        raise ValueError(f"X and y have different lengths: {n} and {ya.shape[0]}")
    order = np.random.default_rng(seed).permutation(n) if shuffle else None
    for start in range(0, n, batch_size):
        stop = start + batch_size
        if drop_last and stop > n:
            break
        idx = slice(start, stop) if order is None else order[start:stop]
        yield (Xa[idx] if ya is None else (Xa[idx], ya[idx]))
