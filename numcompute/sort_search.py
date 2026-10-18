"""Sorting, top-k selection, quickselect and binary search.

Tie and NaN conventions
-----------------------
* Sorts are stable (equal elements keep their original order).
* NaN sorts last in ascending order and is never preferred by :func:`topk`.
* :func:`topk` resolves ties at the k-th boundary to the lowest index.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .utils import normalize_axis

__all__ = [
    "stable_argsort",
    "stable_sort",
    "multi_key_sort",
    "topk",
    "quickselect",
    "binary_search",
    "binary_search_many",
]


def stable_argsort(a, axis: int = -1, descending: bool = False) -> np.ndarray:
    """Indices that stably sort ``a`` along ``axis``.

    Parameters
    ----------
    a : array_like
        Input of any shape with ``ndim >= 1``.
    axis : int, optional
        Axis to sort along. Default -1.
    descending : bool, optional
        Sort from largest to smallest. Ties still keep their original order
        (first occurrence first). NaN comes last when ascending and first
        when descending.

    Returns
    -------
    numpy.ndarray of int
        Same shape as ``a``.

    Raises
    ------
    ValueError
        If ``a`` is 0-d or ``axis`` is out of range.

    Notes
    -----
    Time O(n log n) along the axis, space O(n).

    Examples
    --------
    >>> stable_argsort([2, 1, 2], descending=True).tolist()
    [0, 2, 1]
    """
    arr = np.asarray(a)
    if arr.ndim == 0:
        raise ValueError("a must have at least one dimension")
    ax = normalize_axis(axis, arr.ndim)
    if not descending:
        return np.argsort(arr, axis=ax, kind="stable")
    n = arr.shape[ax]
    rev = np.flip(arr, axis=ax)
    idx = np.flip(np.argsort(rev, axis=ax, kind="stable"), axis=ax)
    return (n - 1) - idx


def stable_sort(a, axis: int = -1, descending: bool = False, return_indices: bool = False):
    """Stable sort along ``axis``.

    Parameters
    ----------
    a : array_like
        Input with ``ndim >= 1``.
    axis : int, optional
        Axis to sort along. Default -1.
    descending : bool, optional
        Largest first. Default False.
    return_indices : bool, optional
        Also return the permutation.

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, numpy.ndarray)
        Sorted copy (same shape as ``a``), plus indices if requested.

    Raises
    ------
    ValueError
        If ``a`` is 0-d or ``axis`` is invalid.

    Notes
    -----
    Time O(n log n), space O(n).
    """
    arr = np.asarray(a)
    idx = stable_argsort(arr, axis=axis, descending=descending)
    out = np.take_along_axis(arr, idx, axis=normalize_axis(axis, arr.ndim))
    return (out, idx) if return_indices else out


def multi_key_sort(
    data,
    keys: Optional[Sequence[int]] = None,
    ascending: Union[bool, Sequence[bool]] = True,
    return_indices: bool = False,
):
    """Sort the rows of a 2-D array by several columns.

    Parameters
    ----------
    data : array_like, shape (n_rows, n_cols)
        Table to sort; any dtype that NumPy can order.
    keys : sequence of int, optional
        Column indices in priority order (first = primary key). Default: all
        columns left to right.
    ascending : bool or sequence of bool, optional
        One flag for all keys or one per key.
    return_indices : bool, optional
        Also return the row permutation.

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, numpy.ndarray)
        Rows reordered, shape ``(n_rows, n_cols)``, plus indices if requested.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, a key column is out of range, ``keys`` is
        empty, or ``ascending`` has the wrong length.

    Notes
    -----
    The sort is stable, so rows equal on every key keep their order.
    Descending string/object keys are handled through their rank codes.
    Time O(k n log n) for k keys, space O(n k).

    Examples
    --------
    >>> multi_key_sort([[2, 1], [1, 5], [2, 0]], keys=[0, 1]).tolist()
    [[1, 5], [2, 0], [2, 1]]
    """
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"data must be 2-D, got shape {arr.shape}")
    n_cols = arr.shape[1]
    key_list = list(range(n_cols)) if keys is None else [int(k) for k in keys]
    if not key_list:
        raise ValueError("keys must not be empty")
    for k in key_list:
        if not -n_cols <= k < n_cols:
            raise ValueError(f"key column {k} out of range for {n_cols} columns")
    if isinstance(ascending, (bool, np.bool_)):
        asc = [bool(ascending)] * len(key_list)
    else:
        asc = [bool(x) for x in ascending]
        if len(asc) != len(key_list):
            raise ValueError(f"ascending has {len(asc)} flags for {len(key_list)} keys")
    if arr.shape[0] == 0:
        idx = np.empty(0, dtype=np.intp)
    else:
        columns = []
        for k, up in zip(key_list, asc):
            col = arr[:, k]
            if not up:
                if col.dtype.kind == "f":
                    col = -col
                else:  # exact for ints, strings, bools: negate the rank codes
                    col = -np.unique(col, return_inverse=True)[1].ravel()
            columns.append(col)
        idx = np.lexsort(columns[::-1])  # lexsort: last key is primary
    out = arr[idx]
    return (out, idx) if return_indices else out


def topk(
    values,
    k: int,
    largest: bool = True,
    return_indices: bool = True,
    axis: int = -1,
    sort: bool = True,
):
    """Select the ``k`` largest (or smallest) entries along ``axis``.

    Uses ``numpy.partition`` (introselect), so selection is linear rather than
    a full sort.

    Parameters
    ----------
    values : array_like of numbers, ndim >= 1
        Input array.
    k : int
        Number of entries to return. ``k = 0`` returns an empty result and
        ``k > n`` is clamped to ``n``.
    largest : bool, optional
        True for largest, False for smallest. Default True.
    return_indices : bool, optional
        Return ``(values, indices)`` if True, else just ``values``.
    axis : int, optional
        Axis to select along. Default -1.
    sort : bool, optional
        Order the ``k`` results best-first. If False they appear in input order.

    Returns
    -------
    values : numpy.ndarray
        Shape of the input with ``axis`` replaced by ``k``.
    indices : numpy.ndarray of int
        Same shape as ``values``; positions along ``axis``. Only if
        ``return_indices``.

    Raises
    ------
    TypeError
        If ``k`` is not an integer or ``values`` is not numeric.
    ValueError
        If ``k < 0``, ``values`` is 0-d, or ``axis`` is invalid.

    Notes
    -----
    Ties at the boundary go to the lowest index, so output is deterministic.
    NaN is treated as the worst value and is returned only if fewer than
    ``k`` real numbers exist. Time O(n + k log k) per slice, space O(n).

    Examples
    --------
    >>> v, i = topk([3, 1, 4, 1, 5], 2)
    >>> v.tolist(), i.tolist()
    ([5, 4], [4, 2])
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    a = np.asarray(values)
    if a.dtype.kind not in "biuf":
        raise TypeError(f"values must be numeric, got dtype {a.dtype}")
    if a.ndim == 0:
        raise ValueError("values must have at least one dimension")
    ax = normalize_axis(axis, a.ndim)
    moved = np.moveaxis(a, ax, -1)
    n = moved.shape[-1]
    k = min(int(k), n)
    lead = moved.shape[:-1]
    if k == 0 or moved.size == 0:
        vals = np.empty(lead + (k,), dtype=a.dtype)
        idx = np.empty(lead + (k,), dtype=np.intp)
    else:
        flat = moved.reshape(-1, n)
        if a.dtype.kind == "f":
            key = -flat if largest else flat
        else:  # ~x reverses integer/bool order without overflow
            key = ~flat if largest else flat
        kth = np.partition(key, k - 1, axis=1)[:, k - 1:k]
        if a.dtype.kind == "f":
            key_nan, kth_nan = np.isnan(key), np.isnan(kth)
            less = np.where(kth_nan, ~key_nan, key < kth)
            equal = np.where(kth_nan, key_nan, key == kth)
        else:
            less, equal = key < kth, key == kth
        need = k - less.sum(axis=1, keepdims=True)
        mask = less | (equal & (np.cumsum(equal, axis=1) <= need))
        idx = np.nonzero(mask)[1].reshape(-1, k)  # row-major: ascending index per row
        if sort:
            order = np.argsort(np.take_along_axis(key, idx, axis=1), axis=1, kind="stable")
            idx = np.take_along_axis(idx, order, axis=1)
        vals = np.take_along_axis(flat, idx, axis=1)
        vals = vals.reshape(lead + (k,))
        idx = idx.reshape(lead + (k,))
    vals = np.moveaxis(vals, -1, ax)
    idx = np.moveaxis(idx, -1, ax)
    return (vals, idx) if return_indices else vals


def quickselect(a, k: int, seed: Optional[int] = None):
    """Return the ``k``-th smallest element (0-based) without sorting.

    Educational randomised quickselect with a three-way partition, so many
    duplicates are handled in linear time.

    Parameters
    ----------
    a : array_like, shape (n,)
        1-D numeric data without NaN.
    k : int
        Rank to select, ``0 <= k < n``. ``k = 0`` is the minimum.
    seed : int, optional
        Seed for the random pivot (result does not depend on it).

    Returns
    -------
    scalar
        The element that would be at index ``k`` after sorting.

    Raises
    ------
    TypeError
        If ``k`` is not an integer or ``a`` is not numeric.
    ValueError
        If ``a`` is not 1-D, empty, contains NaN, or ``k`` is out of range.

    Notes
    -----
    Expected time O(n), worst case O(n^2); space O(n) for the working copies.

    Examples
    --------
    >>> int(quickselect([7, 1, 5, 3], 2))
    5
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    arr = np.asarray(a)
    if arr.dtype.kind not in "biuf":
        raise TypeError(f"a must be numeric, got dtype {arr.dtype}")
    if arr.ndim != 1:
        raise ValueError(f"a must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("a must not be empty")
    if arr.dtype.kind == "f" and np.isnan(arr).any():
        raise ValueError("a contains NaN; remove or impute it first")
    if not 0 <= k < arr.size:
        raise ValueError(f"k must satisfy 0 <= k < {arr.size}, got {k}")
    rng = np.random.default_rng(seed)
    cur, k = arr, int(k)
    while True:
        pivot = cur[rng.integers(cur.size)]
        lows = cur[cur < pivot]
        if k < lows.size:
            cur = lows
            continue
        n_equal = int(np.count_nonzero(cur == pivot))
        if k < lows.size + n_equal:
            return pivot
        k -= lows.size + n_equal
        cur = cur[cur > pivot]


def binary_search(sorted_array, x, side: str = "left") -> Tuple[int, bool]:
    """Binary search in a sorted 1-D array.

    Parameters
    ----------
    sorted_array : array_like, shape (n,)
        Ascending data (NaN, if any, must be at the end). Sortedness is not
        checked.
    x : scalar
        Value to look for.
    side : {'left', 'right'}, optional
        ``'left'`` returns the first position where ``x`` could be inserted
        (``bisect_left``); ``'right'`` the last (``bisect_right``).

    Returns
    -------
    index : int
        Insertion index in ``[0, n]`` that keeps the array sorted.
    found : bool
        Whether ``x`` is present.

    Raises
    ------
    ValueError
        If ``sorted_array`` is not 1-D or ``side`` is invalid.

    Notes
    -----
    With duplicates, ``side='left'`` gives the first occurrence and
    ``side='right'`` one past the last. A NaN ``x`` is inserted before the
    trailing NaNs and is never ``found``. Time O(log n), space O(1).

    Examples
    --------
    >>> binary_search([1, 3, 3, 5], 3)
    (1, True)
    >>> binary_search([1, 3, 3, 5], 4)
    (3, False)
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    a = np.asarray(sorted_array)
    if a.ndim != 1:
        raise ValueError(f"sorted_array must be 1-D, got shape {a.shape}")
    n = a.size
    if a.dtype.kind == "f" and isinstance(x, (float, np.floating)) and np.isnan(x):
        return int(n - np.count_nonzero(np.isnan(a))), False
    lo, hi = 0, n
    if side == "left":
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return int(lo), bool(lo < n and a[lo] == x)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return int(lo), bool(lo > 0 and a[lo - 1] == x)


def binary_search_many(sorted_array, xs, side: str = "left") -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised :func:`binary_search` for many query values.

    Parameters
    ----------
    sorted_array : array_like, shape (n,)
        Ascending 1-D data.
    xs : array_like, shape (m,)
        Query values.
    side : {'left', 'right'}, optional
        As in :func:`binary_search`.

    Returns
    -------
    indices : numpy.ndarray of int, shape (m,)
    found : numpy.ndarray of bool, shape (m,)

    Raises
    ------
    ValueError
        If ``sorted_array`` or ``xs`` is not 1-D or ``side`` is invalid.

    Notes
    -----
    Time O(m log n), space O(m).
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    a, q = np.asarray(sorted_array), np.asarray(xs)
    if a.ndim != 1 or q.ndim != 1:
        raise ValueError(f"sorted_array and xs must be 1-D, got {a.shape} and {q.shape}")
    idx = np.searchsorted(a, q, side=side)
    if side == "left":
        found = (idx < a.size) & (a[np.minimum(idx, max(a.size - 1, 0))] == q) if a.size else np.zeros(q.shape, bool)
    else:
        found = (idx > 0) & (a[np.maximum(idx - 1, 0)] == q) if a.size else np.zeros(q.shape, bool)
    return idx, found
