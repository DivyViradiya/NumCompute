"""Ranking with ties and percentiles.

NaN policy
----------
``rank`` : ``'omit'`` (default) gives NaN elements a NaN rank and ranks the
rest among themselves; ``'propagate'`` makes a whole slice NaN if it holds a
NaN; ``'raise'`` raises ``ValueError``.
``percentile`` : ``'propagate'`` (default), ``'omit'`` or ``'raise'``.
"""
from __future__ import annotations

from typing import Union

import numpy as np

from .utils import (
    as_float_array,
    check_nan_policy,
    from_2d_along_axis,
    normalize_axis,
    to_2d_along_axis,
)

__all__ = ["rank", "percentile"]

_RANK_METHODS = ("average", "min", "max", "dense", "ordinal")
_INTERPOLATIONS = ("linear", "lower", "higher", "midpoint", "nearest")


def rank(data, method: str = "average", axis: int = -1, nan_policy: str = "omit") -> np.ndarray:
    """Rank values (1 = smallest), handling ties.

    Parameters
    ----------
    data : array_like of numbers, ndim >= 1
        Values to rank.
    method : {'average', 'min', 'max', 'dense', 'ordinal'}, optional
        How tied values are ranked:

        * ``average`` - mean of the positions they occupy (1, 2.5, 2.5, 4).
        * ``min``     - lowest position (1, 2, 2, 4).
        * ``max``     - highest position (1, 3, 3, 4).
        * ``dense``   - consecutive ranks without gaps (1, 2, 2, 3).
        * ``ordinal`` - distinct ranks in order of appearance (1, 2, 3, 4).
    axis : int, optional
        Axis along which to rank. Default -1.
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        See module docstring. Default ``'omit'``.

    Returns
    -------
    numpy.ndarray of float
        Same shape as ``data``.

    Raises
    ------
    TypeError
        If ``data`` is not numeric.
    ValueError
        If ``method``/``nan_policy`` is invalid, ``axis`` is out of range,
        ``data`` is 0-d, or NaN is present with ``nan_policy='raise'``.

    Notes
    -----
    Fully vectorised: one stable argsort, tie groups found with a cumulative
    maximum/minimum scan. Time O(n log n) per slice, space O(n).

    Examples
    --------
    >>> rank([10, 20, 20, 30]).tolist()
    [1.0, 2.5, 2.5, 4.0]
    >>> rank([10, 20, 20, 30], method="dense").tolist()
    [1.0, 2.0, 2.0, 3.0]
    """
    if method not in _RANK_METHODS:
        raise ValueError(f"method must be one of {_RANK_METHODS}, got {method!r}")
    check_nan_policy(nan_policy)
    a = as_float_array(data, name="data")
    if a.ndim == 0:
        raise ValueError("data must have at least one dimension")
    normalize_axis(axis, a.ndim)
    if a.size == 0:
        return np.empty(a.shape, dtype=float)
    a2, moved_shape = to_2d_along_axis(a, axis)
    nan_mask = np.isnan(a2)
    if nan_policy == "raise" and nan_mask.any():
        raise ValueError("data contains NaN")
    m, n = a2.shape
    order = np.argsort(a2, axis=1, kind="stable")  # NaN sorted to the end
    s = np.take_along_axis(a2, order, axis=1)
    new_group = np.ones((m, n), dtype=bool)
    if n > 1:
        both_nan = np.isnan(s[:, 1:]) & np.isnan(s[:, :-1])
        new_group[:, 1:] = (s[:, 1:] != s[:, :-1]) & ~both_nan
    pos = np.arange(n)
    if method == "ordinal":
        r = np.broadcast_to(pos + 1.0, (m, n))
    elif method == "dense":
        r = np.cumsum(new_group, axis=1).astype(float)
    else:
        start = np.maximum.accumulate(np.where(new_group, pos, 0), axis=1)
        if method == "min":
            r = start + 1.0
        else:
            ends_here = np.ones((m, n), dtype=bool)
            ends_here[:, :-1] = new_group[:, 1:]
            end = np.minimum.accumulate(np.where(ends_here, pos, n)[:, ::-1], axis=1)[:, ::-1]
            r = end + 1.0 if method == "max" else (start + end) / 2.0 + 1.0
    ranks = np.empty((m, n), dtype=float)
    np.put_along_axis(ranks, order, np.asarray(r, dtype=float), axis=1)
    if nan_policy == "propagate":
        ranks[nan_mask.any(axis=1)] = np.nan
    else:
        ranks[nan_mask] = np.nan
    return from_2d_along_axis(ranks, moved_shape, axis)


def percentile(
    data,
    q,
    axis: Union[int, None] = None,
    interpolation: str = "linear",
    nan_policy: str = "propagate",
):
    """Percentiles of ``data`` with selectable interpolation.

    Parameters
    ----------
    data : array_like of numbers
        Input values; must not be empty.
    q : float or array_like of float, 1-D
        Percentile(s) in ``[0, 100]``.
    axis : int or None, optional
        Axis to reduce. ``None`` (default) uses the flattened array.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, optional
        How to pick a value between two neighbours ``a <= b`` at fractional
        position ``i + f``: ``linear`` -> ``a + (b - a) f``; ``lower`` -> ``a``;
        ``higher`` -> ``b``; ``midpoint`` -> ``(a + b) / 2``; ``nearest`` ->
        the closer one (ties to even index).
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        ``propagate``: a slice containing NaN gives NaN. ``omit``: NaNs are
        ignored (an all-NaN slice gives NaN). ``raise``: ValueError.

    Returns
    -------
    float or numpy.ndarray
        Scalar if ``q`` is scalar and ``axis`` is None. Otherwise shape is
        ``q.shape + reduced_shape`` (``q`` dimension first, like NumPy).

    Raises
    ------
    TypeError
        If ``data`` or ``q`` is not numeric.
    ValueError
        If ``data`` is empty, ``q`` is outside [0, 100] or NaN or has more
        than one dimension, an option is invalid, or NaN is present with
        ``nan_policy='raise'``.

    Notes
    -----
    Position of percentile ``q`` among ``n`` valid values is
    ``q / 100 * (n - 1)`` (NumPy's default definition). One sort per slice:
    time O(n log n), space O(n).

    Examples
    --------
    >>> float(percentile([1, 2, 3, 4], 50))
    2.5
    >>> float(percentile([1, 2, 3, 4], 50, interpolation="lower"))
    2.0
    """
    if interpolation not in _INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {_INTERPOLATIONS}, got {interpolation!r}")
    check_nan_policy(nan_policy)
    a = as_float_array(data, name="data", allow_empty=False)
    q_arr = as_float_array(q, name="q")
    if q_arr.ndim > 1:
        raise ValueError(f"q must be a scalar or 1-D, got shape {q_arr.shape}")
    if np.isnan(q_arr).any() or (q_arr < 0).any() or (q_arr > 100).any():
        raise ValueError("q must lie in [0, 100]")
    scalar_q = q_arr.ndim == 0
    q1 = np.atleast_1d(q_arr)

    if axis is None:
        a2, out_shape = a.reshape(1, -1), ()
    else:
        if a.ndim == 0:
            raise ValueError("cannot use axis with a 0-d array")
        a2, moved_shape = to_2d_along_axis(a, axis)
        out_shape = moved_shape[:-1]
    m, n = a2.shape
    nan_mask = np.isnan(a2)
    n_nan = nan_mask.sum(axis=1)
    if nan_policy == "raise" and n_nan.any():
        raise ValueError("data contains NaN")

    srt = np.sort(a2, axis=1)  # NaN at the end
    n_valid = n - n_nan
    pos = (q1[:, None] / 100.0) * (np.maximum(n_valid, 1) - 1)[None, :]  # (Q, m)
    lo = np.floor(pos).astype(np.intp)
    hi = np.minimum(np.ceil(pos).astype(np.intp), np.maximum(n_valid - 1, 0)[None, :])
    v_lo = np.take_along_axis(srt, lo.T, axis=1).T
    v_hi = np.take_along_axis(srt, hi.T, axis=1).T
    same = (lo == hi) | (v_lo == v_hi)
    if interpolation == "lower":
        res = v_lo
    elif interpolation == "higher":
        res = v_hi
    elif interpolation == "nearest":
        near = np.minimum(np.rint(pos).astype(np.intp), np.maximum(n_valid - 1, 0)[None, :])
        res = np.take_along_axis(srt, near.T, axis=1).T
    else:
        with np.errstate(invalid="ignore"):
            if interpolation == "midpoint":
                mixed = (v_lo + v_hi) / 2.0
            else:
                mixed = v_lo + (v_hi - v_lo) * (pos - lo)
        res = np.where(same, v_lo, mixed)
    bad = (n_valid == 0)[None, :]
    if nan_policy == "propagate":
        bad = bad | (n_nan > 0)[None, :]
    res = np.where(bad, np.nan, res)
    res = res.reshape((q1.size,) + out_shape)
    if scalar_q:
        res = res[0]
    return res[()] if res.ndim == 0 else res
