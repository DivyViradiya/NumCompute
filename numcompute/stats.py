"""Descriptive statistics, streaming (Welford/Chan) statistics, histogram, quantiles.

NaN policy (``nan_policy`` argument)
------------------------------------
``'propagate'`` (default) - a NaN in a slice makes that slice's result NaN.
``'omit'``                - NaNs are ignored; an all-NaN slice gives NaN.
``'raise'``               - ``ValueError`` if any NaN is present.
``histogram`` accepts only ``'omit'`` (default) and ``'raise'``.

Axis semantics: ``axis=None`` reduces over all elements and returns a scalar;
an integer axis removes that axis from the output shape.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .rank import percentile
from .utils import as_float_array, check_nan_policy, normalize_axis

__all__ = [
    "mean",
    "median",
    "var",
    "std",
    "minimum",
    "maximum",
    "describe",
    "RunningStats",
    "histogram",
    "quantile",
]

Axis = Union[int, None]


def _prepare(a, axis: Axis, nan_policy: str) -> np.ndarray:
    check_nan_policy(nan_policy)
    arr = as_float_array(a, name="a", allow_empty=False)
    if axis is not None:
        if arr.ndim == 0:
            raise ValueError("cannot use axis with a 0-d array")
        normalize_axis(axis, arr.ndim)
    if nan_policy == "raise" and np.isnan(arr).any():
        raise ValueError("a contains NaN")
    return arr


def mean(a, axis: Axis = None, nan_policy: str = "propagate"):
    """Arithmetic mean.

    Parameters
    ----------
    a : array_like of numbers
        Non-empty input.
    axis : int or None, optional
        Axis to reduce; ``None`` for all elements.
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        See module docstring.

    Returns
    -------
    float or numpy.ndarray
        Scalar if ``axis`` is None, otherwise ``a.shape`` without ``axis``.

    Raises
    ------
    TypeError
        If ``a`` is not numeric.
    ValueError
        If ``a`` is empty, ``axis``/``nan_policy`` is invalid, or NaN is
        present with ``nan_policy='raise'``.

    Notes
    -----
    Time O(n), space O(1) extra.

    Examples
    --------
    >>> float(mean([1, 2, 3, 4]))
    2.5
    """
    arr = _prepare(a, axis, nan_policy)
    if nan_policy == "omit":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(arr, axis=axis)
    return np.mean(arr, axis=axis)


def median(a, axis: Axis = None, nan_policy: str = "propagate"):
    """Median (50th percentile).

    Parameters
    ----------
    a, axis, nan_policy
        As in :func:`mean`.

    Returns
    -------
    float or numpy.ndarray
        Same shape rules as :func:`mean`.

    Raises
    ------
    TypeError, ValueError
        As in :func:`mean`.

    Notes
    -----
    Implemented with :func:`numcompute.rank.percentile`. Time O(n log n),
    space O(n).

    Examples
    --------
    >>> float(median([3, 1, 2, 10]))
    2.5
    """
    arr = _prepare(a, axis, nan_policy)
    return percentile(arr, 50, axis=axis, nan_policy=nan_policy)


def var(a, axis: Axis = None, ddof: int = 0, nan_policy: str = "propagate"):
    """Variance.

    Parameters
    ----------
    a : array_like of numbers
        Non-empty input.
    axis : int or None, optional
        Axis to reduce.
    ddof : int, optional
        Delta degrees of freedom: divisor is ``n - ddof``. Default 0.
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        See module docstring.

    Returns
    -------
    float or numpy.ndarray
        NaN where ``n - ddof <= 0``.

    Raises
    ------
    TypeError
        If ``a`` is not numeric.
    ValueError
        If ``ddof < 0`` or an argument is invalid, or the input is empty.

    Notes
    -----
    Two-pass algorithm (mean first), which is numerically stable. Time O(n),
    space O(n).

    Examples
    --------
    >>> float(var([1, 2, 3, 4], ddof=1))
    1.6666666666666667
    """
    if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError(f"ddof must be a non-negative int, got {ddof!r}")
    arr = _prepare(a, axis, nan_policy)
    fn = np.nanvar if nan_policy == "omit" else np.var
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            out = fn(arr, axis=axis, ddof=ddof)
    count = np.sum(~np.isnan(arr), axis=axis) if nan_policy == "omit" else (
        arr.size if axis is None else arr.shape[axis]
    )
    out = np.where(np.asarray(count) - ddof <= 0, np.nan, out)
    return out[()] if out.ndim == 0 else out


def std(a, axis: Axis = None, ddof: int = 0, nan_policy: str = "propagate"):
    """Standard deviation (square root of :func:`var`).

    Parameters
    ----------
    a, axis, ddof, nan_policy
        As in :func:`var`.

    Returns
    -------
    float or numpy.ndarray
        Same shape rules as :func:`var`.

    Raises
    ------
    TypeError, ValueError
        As in :func:`var`.

    Notes
    -----
    Time O(n), space O(n).

    Examples
    --------
    >>> float(std([2, 4, 4, 4, 5, 5, 7, 9]))
    2.0
    """
    return np.sqrt(var(a, axis=axis, ddof=ddof, nan_policy=nan_policy))


def minimum(a, axis: Axis = None, nan_policy: str = "propagate"):
    """Smallest value.

    Parameters
    ----------
    a, axis, nan_policy
        As in :func:`mean`.

    Returns
    -------
    float or numpy.ndarray
        All-NaN slices give NaN under ``'omit'``.

    Raises
    ------
    TypeError, ValueError
        As in :func:`mean`.

    Notes
    -----
    Time O(n), space O(1) extra.
    """
    arr = _prepare(a, axis, nan_policy)
    if nan_policy == "omit":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmin(arr, axis=axis)
    return np.min(arr, axis=axis)


def maximum(a, axis: Axis = None, nan_policy: str = "propagate"):
    """Largest value.

    Parameters
    ----------
    a, axis, nan_policy
        As in :func:`mean`.

    Returns
    -------
    float or numpy.ndarray
        All-NaN slices give NaN under ``'omit'``.

    Raises
    ------
    TypeError, ValueError
        As in :func:`mean`.

    Notes
    -----
    Time O(n), space O(1) extra.
    """
    arr = _prepare(a, axis, nan_policy)
    if nan_policy == "omit":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmax(arr, axis=axis)
    return np.max(arr, axis=axis)


def describe(a, axis: Axis = None, nan_policy: str = "propagate") -> Dict[str, object]:
    """Summary statistics in one call.

    Parameters
    ----------
    a : array_like of numbers
        Non-empty input.
    axis : int or None, optional
        Axis to summarise.
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        See module docstring.

    Returns
    -------
    dict
        Keys ``count`` (non-NaN values), ``mean``, ``std`` (ddof=0), ``min``,
        ``q25``, ``median``, ``q75``, ``max``. Values are scalars or arrays
        shaped like the reduced output.

    Raises
    ------
    TypeError, ValueError
        As in :func:`mean`.

    Notes
    -----
    Time O(n log n), space O(n).
    """
    arr = _prepare(a, axis, nan_policy)
    q25, med, q75 = percentile(arr, [25, 50, 75], axis=axis, nan_policy=nan_policy)
    count = np.sum(~np.isnan(arr), axis=axis)
    return {
        "count": count,
        "mean": mean(arr, axis, nan_policy),
        "std": std(arr, axis, 0, nan_policy),
        "min": minimum(arr, axis, nan_policy),
        "q25": q25,
        "median": med,
        "q75": q75,
        "max": maximum(arr, axis, nan_policy),
    }


class RunningStats:
    """Streaming per-feature count, mean and variance (Welford / Chan).

    Feed data in batches with :meth:`update`; memory stays O(n_features) no
    matter how many rows are seen. Batches are merged with Chan et al.'s
    parallel formula, which is numerically stable and equals Welford's
    update when a batch has one row. NaN entries are skipped per feature.

    Parameters
    ----------
    None

    Attributes
    ----------
    n_features_ : int or None
        Number of features (set by the first update).
    count : numpy.ndarray, shape (n_features,)
        Non-NaN observations per feature.
    mean : numpy.ndarray, shape (n_features,)
        Running mean (NaN for features with no observations).

    Examples
    --------
    >>> rs = RunningStats().update([[1.0], [2.0]]).update([[3.0], [4.0]])
    >>> float(rs.mean[0]), float(rs.var(ddof=1)[0])
    (2.5, 1.6666666666666667)
    """

    def __init__(self) -> None:
        self.n_features_: Optional[int] = None
        self._n = np.zeros(0)
        self._mean = np.zeros(0)
        self._m2 = np.zeros(0)

    def _init(self, n_features: int) -> None:
        self.n_features_ = n_features
        self._n = np.zeros(n_features)
        self._mean = np.zeros(n_features)
        self._m2 = np.zeros(n_features)

    def _merge(self, n_b, mean_b, m2_b) -> None:
        total = self._n + n_b
        delta = mean_b - self._mean
        frac = n_b / np.maximum(total, 1.0)
        self._mean = self._mean + delta * frac
        self._m2 = self._m2 + m2_b + delta**2 * self._n * frac
        self._n = total

    def update(self, x) -> "RunningStats":
        """Add a batch of observations.

        Parameters
        ----------
        x : array_like, shape (n_rows, n_features) or (n_rows,)
            New data. A 1-D input is treated as one feature.

        Returns
        -------
        RunningStats
            ``self`` (so calls can be chained).

        Raises
        ------
        TypeError
            If ``x`` is not numeric.
        ValueError
            If ``x`` is not 1-D/2-D or its feature count differs from earlier batches.

        Notes
        -----
        Time O(n_rows * n_features), space O(n_rows * n_features) for the
        batch (state is O(n_features)).
        """
        arr = as_float_array(x, name="x", ndim=(1, 2))
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if self.n_features_ is None:
            self._init(arr.shape[1])
        elif arr.shape[1] != self.n_features_:
            raise ValueError(
                f"x has {arr.shape[1]} features, but earlier batches had {self.n_features_}"
            )
        if arr.shape[0] == 0:
            return self
        valid = ~np.isnan(arr)
        n_b = valid.sum(axis=0).astype(float)
        filled = np.where(valid, arr, 0.0)
        mean_b = filled.sum(axis=0) / np.maximum(n_b, 1.0)
        m2_b = (((filled - mean_b) ** 2) * valid).sum(axis=0)
        self._merge(n_b, mean_b, m2_b)
        return self

    def merge(self, other: "RunningStats") -> "RunningStats":
        """Combine another ``RunningStats`` into this one.

        Parameters
        ----------
        other : RunningStats
            Statistics of another stream with the same number of features.

        Returns
        -------
        RunningStats
            ``self``.

        Raises
        ------
        TypeError
            If ``other`` is not a ``RunningStats``.
        ValueError
            If the feature counts differ.

        Notes
        -----
        Time O(n_features), space O(n_features).
        """
        if not isinstance(other, RunningStats):
            raise TypeError("other must be a RunningStats")
        if other.n_features_ is None:
            return self
        if self.n_features_ is None:
            self._init(other.n_features_)
        elif self.n_features_ != other.n_features_:
            raise ValueError(
                f"feature counts differ: {self.n_features_} and {other.n_features_}"
            )
        self._merge(other._n, other._mean, other._m2)
        return self

    @property
    def count(self) -> np.ndarray:
        """Non-NaN observation count per feature."""
        return self._n.copy()

    @property
    def mean(self) -> np.ndarray:
        """Running mean per feature (NaN where nothing was observed)."""
        return np.where(self._n > 0, self._mean, np.nan)

    def var(self, ddof: int = 0) -> np.ndarray:
        """Running variance per feature.

        Parameters
        ----------
        ddof : int, optional
            Divisor is ``count - ddof``. Default 0.

        Returns
        -------
        numpy.ndarray, shape (n_features,)
            NaN where ``count - ddof <= 0``.

        Raises
        ------
        ValueError
            If ``ddof < 0``.
        """
        if ddof < 0:
            raise ValueError(f"ddof must be non-negative, got {ddof}")
        denom = self._n - ddof
        return np.where(denom > 0, self._m2 / np.where(denom > 0, denom, 1.0), np.nan)

    def std(self, ddof: int = 0) -> np.ndarray:
        """Running standard deviation (square root of :meth:`var`)."""
        return np.sqrt(self.var(ddof))


def histogram(
    a,
    bins: Union[int, np.ndarray] = 10,
    range: Optional[Tuple[float, float]] = None,
    density: bool = False,
    nan_policy: str = "omit",
) -> Tuple[np.ndarray, np.ndarray]:
    """Histogram of a flattened array.

    Parameters
    ----------
    a : array_like of numbers
        Input values (flattened).
    bins : int or array_like, optional
        Number of equal-width bins, or strictly increasing bin edges
        (``len(bins) - 1`` bins). Default 10.
    range : (float, float), optional
        Lower and upper limit for integer ``bins``. Default: data min/max.
        If min equals max, the range is widened by 0.5 on each side.
    density : bool, optional
        Return probability density (integrates to 1 over the bins) instead
        of counts.
    nan_policy : {'omit', 'raise'}, optional
        NaN handling; ``'propagate'`` is not meaningful here.

    Returns
    -------
    counts : numpy.ndarray, shape (n_bins,)
        Integer counts (float if ``density``). Bins are half-open ``[l, r)``
        except the last, which includes its right edge.
    edges : numpy.ndarray, shape (n_bins + 1,)

    Raises
    ------
    TypeError
        If ``a`` is not numeric.
    ValueError
        If ``bins`` is not positive/increasing, ``range`` is invalid, the
        automatic range is not finite, or NaN is present with ``'raise'``.

    Notes
    -----
    Values outside the edges are ignored. One ``searchsorted`` plus one
    ``bincount``: time O(n log b), space O(n).

    Examples
    --------
    >>> counts, edges = histogram([0, 1, 1, 2, 4], bins=2)
    >>> counts.tolist(), edges.tolist()
    ([3, 2], [0.0, 2.0, 4.0])
    """
    check_nan_policy(nan_policy, allowed=("omit", "raise"))
    arr = as_float_array(a, name="a").ravel()
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        if nan_policy == "raise":
            raise ValueError("a contains NaN")
        arr = arr[~nan_mask]

    if isinstance(bins, (int, np.integer)) and not isinstance(bins, bool):
        if bins < 1:
            raise ValueError(f"bins must be >= 1, got {bins}")
        if range is not None:
            lo, hi = float(range[0]), float(range[1])
            if not lo <= hi:
                raise ValueError(f"range must satisfy low <= high, got {range}")
        elif arr.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = float(arr.min()), float(arr.max())
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError("data range is not finite; pass range=(low, high) explicitly")
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, int(bins) + 1)
    else:
        edges = as_float_array(bins, name="bins", ndim=1)
        if edges.size < 2 or np.isnan(edges).any() or not (np.diff(edges) > 0).all():
            raise ValueError("bin edges must be a strictly increasing 1-D array of length >= 2")
    n_bins = edges.size - 1
    idx = np.searchsorted(edges, arr, side="right") - 1
    idx[arr == edges[-1]] = n_bins - 1
    inside = (idx >= 0) & (idx < n_bins)
    counts = np.bincount(idx[inside], minlength=n_bins)
    if density:
        total = counts.sum()
        widths = np.diff(edges)
        counts = counts / (total * widths) if total > 0 else np.zeros(n_bins)
    return counts, edges


def quantile(
    a,
    q,
    axis: Axis = None,
    interpolation: str = "linear",
    nan_policy: str = "propagate",
):
    """Quantiles (``q`` in [0, 1]) with NaN handling.

    Parameters
    ----------
    a : array_like of numbers
        Non-empty input.
    q : float or array_like of float, 1-D
        Quantile(s) in ``[0, 1]``.
    axis : int or None, optional
        Axis to reduce; ``None`` flattens.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, optional
        As in :func:`numcompute.rank.percentile`.
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        See module docstring.

    Returns
    -------
    float or numpy.ndarray
        Shape ``q.shape + reduced_shape``; scalar for scalar ``q`` and
        ``axis=None``.

    Raises
    ------
    TypeError
        If ``a`` or ``q`` is not numeric.
    ValueError
        If ``q`` is outside [0, 1] or any other argument is invalid.

    Notes
    -----
    Thin wrapper over :func:`numcompute.rank.percentile`. Time O(n log n),
    space O(n).

    Examples
    --------
    >>> quantile([1, 2, 3, 4, 5], [0.25, 0.5]).tolist()
    [2.0, 3.0]
    """
    q_arr = as_float_array(q, name="q")
    if np.isnan(q_arr).any() or (q_arr < 0).any() or (q_arr > 1).any():
        raise ValueError("q must lie in [0, 1]")
    return percentile(a, q_arr * 100.0, axis=axis, interpolation=interpolation, nan_policy=nan_policy)
