"""Evaluation metrics: classification, regression, ROC/AUC.

Labels may be integers, strings or booleans. Rows of a confusion matrix are
true classes, columns are predicted classes, both in sorted label order.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .utils import as_float_array

__all__ = [
    "accuracy",
    "confusion_matrix",
    "precision",
    "recall",
    "f1",
    "mse",
    "rmse",
    "mae",
    "r2_score",
    "roc_curve",
    "auc",
    "roc_auc_score",
]

_AVERAGES = ("binary", "macro", "micro", "weighted", None)


def _as_labels(y, name: str) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.dtype.kind == "f" and np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN")
    if arr.dtype.kind == "O":
        raise TypeError(f"{name} must hold numbers, strings or booleans, not objects")
    return arr


def _check_pair(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
    yt, yp = _as_labels(y_true, "y_true"), _as_labels(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {yt.shape} and {yp.shape}")
    if yt.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    return yt, yp


def _safe_div(num, den, zero_division: float) -> np.ndarray:
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), float(zero_division))


def accuracy(y_true, y_pred, normalize: bool = True):
    """Fraction (or count) of correct predictions.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n,)
        True and predicted labels.
    normalize : bool, optional
        Return a fraction (default) or the number of correct predictions.

    Returns
    -------
    float or int
        In ``[0, 1]`` if ``normalize`` else in ``[0, n]``.

    Raises
    ------
    TypeError
        If labels are objects.
    ValueError
        If the inputs are not 1-D, differ in shape, are empty, or contain NaN.

    Notes
    -----
    Time O(n), space O(n).

    Examples
    --------
    >>> accuracy([1, 0, 1, 1], [1, 0, 0, 1])
    0.75
    """
    yt, yp = _check_pair(y_true, y_pred)
    correct = int(np.count_nonzero(yt == yp))
    return correct / yt.size if normalize else correct


def confusion_matrix(y_true, y_pred, labels=None, normalize: Optional[str] = None) -> np.ndarray:
    """Confusion matrix ``C[i, j]`` = samples of true class i predicted as j.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n,)
        True and predicted labels.
    labels : array_like, optional
        Class order for rows/columns. Default: sorted union of both inputs.
        Samples whose label is not in ``labels`` are ignored.
    normalize : {None, 'true', 'pred', 'all'}, optional
        Divide by row sums, column sums, or the grand total (zero rows/columns
        stay 0).

    Returns
    -------
    numpy.ndarray, shape (n_labels, n_labels)
        Integer counts (float if normalised).

    Raises
    ------
    ValueError
        If inputs are invalid (see :func:`accuracy`), ``labels`` is not 1-D
        or contains duplicates, or ``normalize`` is invalid.

    Notes
    -----
    Built with one ``bincount`` on ``true_index * n_labels + pred_index``.
    Time O(n log L), space O(n + L^2).

    Examples
    --------
    >>> confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1]).tolist()
    [[1, 1], [0, 2]]
    """
    if normalize not in (None, "true", "pred", "all"):
        raise ValueError(f"normalize must be None, 'true', 'pred' or 'all', got {normalize!r}")
    yt, yp = _check_pair(y_true, y_pred)
    if labels is None:
        labs = np.unique(np.concatenate([yt, yp]))
    else:
        labs = np.asarray(labels)
        if labs.ndim != 1 or labs.size == 0:
            raise ValueError("labels must be a non-empty 1-D array")
        if np.unique(labs).size != labs.size:
            raise ValueError("labels must not contain duplicates")
    order = np.argsort(labs, kind="stable")
    sorted_labs = labs[order]

    def to_index(y):
        pos = np.minimum(np.searchsorted(sorted_labs, y), sorted_labs.size - 1)
        ok = sorted_labs[pos] == y
        return order[pos], ok

    it, ok_t = to_index(yt)
    ip, ok_p = to_index(yp)
    keep = ok_t & ok_p
    n_lab = labs.size
    cm = np.bincount(it[keep] * n_lab + ip[keep], minlength=n_lab * n_lab).reshape(n_lab, n_lab)
    if normalize is None:
        return cm
    cm = cm.astype(float)
    denom = {"true": cm.sum(axis=1, keepdims=True), "pred": cm.sum(axis=0, keepdims=True), "all": cm.sum()}[normalize]
    return np.where(denom > 0, cm / np.where(denom > 0, denom, 1.0), 0.0)


def _counts(y_true, y_pred, average, pos_label):
    """Return (labels, tp, fp, fn, support) for the chosen averaging."""
    if average not in _AVERAGES:
        raise ValueError(f"average must be one of {_AVERAGES}, got {average!r}")
    yt, yp = _check_pair(y_true, y_pred)
    labels = np.unique(np.concatenate([yt, yp]))
    if average == "binary":
        if labels.dtype.kind in "US" and not isinstance(pos_label, str):
            raise ValueError(
                f"labels are strings {labels.tolist()}; pass pos_label as one of them"
            )
        labels = np.unique(np.append(labels, pos_label))
        if labels.size > 2:
            raise ValueError(
                f"average='binary' needs at most 2 classes including pos_label={pos_label!r}, "
                f"found {labels.tolist()}; set pos_label or use average='macro'/'micro'/'weighted'"
            )
    cm = confusion_matrix(yt, yp, labels=labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1).astype(float)
    if average == "binary":
        j = int(np.searchsorted(labels, pos_label))
        return labels, tp[j:j + 1], fp[j:j + 1], fn[j:j + 1], support[j:j + 1]
    return labels, tp, fp, fn, support


def _average(values, average, tp, fp, fn, support, kind, zero_division):
    if average == "micro":
        t, p, n = tp.sum(), fp.sum(), fn.sum()
        if kind == "precision":
            return float(_safe_div(t, t + p, zero_division))
        if kind == "recall":
            return float(_safe_div(t, t + n, zero_division))
        return float(_safe_div(2 * t, 2 * t + p + n, zero_division))
    if average == "binary":
        return float(values[0])
    if average == "macro":
        return float(values.mean())
    if average == "weighted":
        total = support.sum()
        return float((values * support).sum() / total) if total > 0 else float(zero_division)
    return values


def precision(y_true, y_pred, average: Optional[str] = "binary", pos_label=1, zero_division: float = 0.0):
    """Precision = TP / (TP + FP).

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n,)
        True and predicted labels.
    average : {'binary', 'macro', 'micro', 'weighted', None}, optional
        ``binary``: score of ``pos_label`` only (needs at most two classes).
        ``macro``: unweighted mean over classes. ``micro``: global counts.
        ``weighted``: mean weighted by class support. ``None``: one score
        per class (sorted label order).
    pos_label : scalar, optional
        Positive class for ``average='binary'``. Default 1.
    zero_division : float, optional
        Value used when a denominator is 0. Default 0.

    Returns
    -------
    float or numpy.ndarray
        Array of shape ``(n_classes,)`` when ``average`` is None.

    Raises
    ------
    ValueError
        For invalid inputs, an unknown ``average``, or ``average='binary'``
        with more than two classes.

    Notes
    -----
    Time O(n log L), space O(n + L^2).

    Examples
    --------
    >>> precision([1, 0, 1, 1], [1, 1, 0, 1])
    0.6666666666666666
    """
    labels, tp, fp, fn, support = _counts(y_true, y_pred, average, pos_label)
    values = _safe_div(tp, tp + fp, zero_division)
    return _average(values, average, tp, fp, fn, support, "precision", zero_division)


def recall(y_true, y_pred, average: Optional[str] = "binary", pos_label=1, zero_division: float = 0.0):
    """Recall = TP / (TP + FN).

    Parameters
    ----------
    y_true, y_pred, average, pos_label, zero_division
        As in :func:`precision`.

    Returns
    -------
    float or numpy.ndarray
        As in :func:`precision`.

    Raises
    ------
    ValueError
        As in :func:`precision`.

    Notes
    -----
    Time O(n log L), space O(n + L^2).

    Examples
    --------
    >>> recall([1, 0, 1, 1], [1, 1, 0, 1])
    0.6666666666666666
    """
    labels, tp, fp, fn, support = _counts(y_true, y_pred, average, pos_label)
    values = _safe_div(tp, tp + fn, zero_division)
    return _average(values, average, tp, fp, fn, support, "recall", zero_division)


def f1(y_true, y_pred, average: Optional[str] = "binary", pos_label=1, zero_division: float = 0.0):
    """F1 score = 2 TP / (2 TP + FP + FN) (harmonic mean of precision and recall).

    Parameters
    ----------
    y_true, y_pred, average, pos_label, zero_division
        As in :func:`precision`.

    Returns
    -------
    float or numpy.ndarray
        As in :func:`precision`.

    Raises
    ------
    ValueError
        As in :func:`precision`.

    Notes
    -----
    Written without dividing by ``precision + recall``, so it is well defined
    when both are 0. Time O(n log L), space O(n + L^2).

    Examples
    --------
    >>> f1([1, 0, 1, 1], [1, 1, 0, 1])
    0.6666666666666666
    """
    labels, tp, fp, fn, support = _counts(y_true, y_pred, average, pos_label)
    values = _safe_div(2 * tp, 2 * tp + fp + fn, zero_division)
    return _average(values, average, tp, fp, fn, support, "f1", zero_division)


def _check_regression(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
    yt = as_float_array(y_true, name="y_true", allow_empty=False)
    yp = as_float_array(y_pred, name="y_pred", allow_empty=False)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {yt.shape} and {yp.shape}")
    if np.isnan(yt).any() or np.isnan(yp).any():
        raise ValueError("y_true / y_pred contain NaN")
    return yt, yp


def mse(y_true, y_pred) -> float:
    """Mean squared error.

    Parameters
    ----------
    y_true, y_pred : array_like of numbers
        Same shape (any number of dimensions), non-empty, no NaN.

    Returns
    -------
    float
        ``mean((y_true - y_pred) ** 2)`` over all elements.

    Raises
    ------
    TypeError
        If inputs are not numeric.
    ValueError
        If shapes differ, inputs are empty or contain NaN.

    Notes
    -----
    Time O(n), space O(n).

    Examples
    --------
    >>> mse([1, 2, 3], [1, 2, 5])
    1.3333333333333333
    """
    yt, yp = _check_regression(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error (square root of :func:`mse`).

    Parameters
    ----------
    y_true, y_pred : array_like of numbers
        As in :func:`mse`.

    Returns
    -------
    float

    Raises
    ------
    TypeError, ValueError
        As in :func:`mse`.

    Notes
    -----
    Time O(n), space O(n).
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error.

    Parameters
    ----------
    y_true, y_pred : array_like of numbers
        As in :func:`mse`.

    Returns
    -------
    float
        ``mean(abs(y_true - y_pred))``.

    Raises
    ------
    TypeError, ValueError
        As in :func:`mse`.

    Notes
    -----
    Time O(n), space O(n).
    """
    yt, yp = _check_regression(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination ``1 - SS_res / SS_tot``.

    Parameters
    ----------
    y_true, y_pred : array_like of numbers
        As in :func:`mse`.

    Returns
    -------
    float
        1 is perfect; can be negative. If ``y_true`` is constant the result
        is 1.0 for a perfect fit and 0.0 otherwise.

    Raises
    ------
    TypeError, ValueError
        As in :func:`mse`.

    Notes
    -----
    Time O(n), space O(n).
    """
    yt, yp = _check_regression(y_true, y_pred)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def roc_curve(y_true, y_score, pos_label=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC curve for binary classification.

    Parameters
    ----------
    y_true : array_like, shape (n,)
        Binary labels.
    y_score : array_like of float, shape (n,)
        Scores; higher means more likely positive. No NaN.
    pos_label : scalar, optional
        Label treated as positive. Default 1.

    Returns
    -------
    fpr, tpr : numpy.ndarray, shape (n_thresholds + 1,)
        False/true positive rates, starting at (0, 0) and ending at (1, 1).
    thresholds : numpy.ndarray, shape (n_thresholds + 1,)
        Decreasing; the first entry is ``inf`` (nothing predicted positive).

    Raises
    ------
    ValueError
        If shapes differ, inputs are empty/NaN, or only one class is present.

    Notes
    -----
    Tied scores share one threshold, so there is one point per distinct
    score. Time O(n log n), space O(n).

    Examples
    --------
    >>> fpr, tpr, thr = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    >>> fpr.tolist(), tpr.tolist()
    ([0.0, 0.0, 0.5, 0.5, 1.0], [0.0, 0.5, 0.5, 1.0, 1.0])
    """
    yt = _as_labels(y_true, "y_true")
    ys = as_float_array(y_score, name="y_score", ndim=1, allow_empty=False)
    if yt.shape != ys.shape:
        raise ValueError(f"y_true and y_score must have the same shape, got {yt.shape} and {ys.shape}")
    if np.isnan(ys).any():
        raise ValueError("y_score contains NaN")
    pos = yt == pos_label
    n_pos = int(pos.sum())
    n_neg = pos.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC needs both classes: y_true has only one class (check pos_label)")
    order = np.argsort(-ys, kind="stable")
    s, p = ys[order], pos[order]
    last_of_tie = np.r_[s[1:] != s[:-1], True]
    cut = np.nonzero(last_of_tie)[0]
    tps = np.cumsum(p)[cut]
    fps = (cut + 1) - tps
    fpr = np.r_[0.0, fps / n_neg]
    tpr = np.r_[0.0, tps / n_pos]
    thresholds = np.r_[np.inf, s[cut]]
    return fpr, tpr, thresholds


def auc(x, y) -> float:
    """Area under a curve by the trapezoidal rule.

    Parameters
    ----------
    x, y : array_like, shape (n,)
        Curve points, ``n >= 2``. ``x`` must be monotonic (increasing or
        decreasing).

    Returns
    -------
    float
        Area (positive for increasing ``x``).

    Raises
    ------
    ValueError
        If shapes differ, fewer than 2 points, or ``x`` is not monotonic.

    Notes
    -----
    Time O(n), space O(n).

    Examples
    --------
    >>> auc([0, 1], [0, 1])
    0.5
    """
    xs = as_float_array(x, name="x", ndim=1)
    ys = as_float_array(y, name="y", ndim=1)
    if xs.shape != ys.shape:
        raise ValueError(f"x and y must have the same shape, got {xs.shape} and {ys.shape}")
    if xs.size < 2:
        raise ValueError("at least 2 points are needed to compute an area")
    dx = np.diff(xs)
    if not (np.all(dx >= 0) or np.all(dx <= 0)):
        raise ValueError("x must be monotonic (increasing or decreasing)")
    direction = -1.0 if (dx < 0).any() else 1.0
    return float(direction * np.sum(dx * (ys[1:] + ys[:-1]) / 2.0))


def roc_auc_score(y_true, y_score, pos_label=1) -> float:
    """Area under the ROC curve.

    Parameters
    ----------
    y_true, y_score, pos_label
        As in :func:`roc_curve`.

    Returns
    -------
    float
        In ``[0, 1]``; 0.5 is chance level. Ties between a positive and a
        negative score count as half.

    Raises
    ------
    ValueError
        As in :func:`roc_curve`.

    Notes
    -----
    Time O(n log n), space O(n).

    Examples
    --------
    >>> roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    0.75
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label)
    return auc(fpr, tpr)
