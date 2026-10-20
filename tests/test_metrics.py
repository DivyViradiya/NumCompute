"""Unit tests for numcompute.metrics."""
import numpy as np
import pytest

from numcompute.metrics import (
    accuracy,
    auc,
    confusion_matrix,
    f1,
    mae,
    mse,
    precision,
    r2_score,
    recall,
    rmse,
    roc_auc_score,
    roc_curve,
)

Y_TRUE = [1, 0, 1, 1, 0, 1, 0, 0]
Y_PRED = [1, 0, 0, 1, 0, 1, 1, 0]
# TP=3, FP=1, FN=1, TN=3


# -------------------------------------------------------------------- accuracy
def test_accuracy_basic_and_count():
    assert accuracy(Y_TRUE, Y_PRED) == 0.75
    assert accuracy(Y_TRUE, Y_PRED, normalize=False) == 6


def test_accuracy_perfect_and_all_wrong():
    assert accuracy([1, 2, 3], [1, 2, 3]) == 1.0
    assert accuracy([1, 1], [0, 0]) == 0.0


def test_accuracy_string_and_bool_labels():
    assert accuracy(["a", "b", "a"], ["a", "b", "b"]) == pytest.approx(2 / 3)
    assert accuracy([True, False], [True, True]) == 0.5


def test_input_validation_for_labels():
    with pytest.raises(ValueError):
        accuracy([], [])
    with pytest.raises(ValueError):
        accuracy([1, 2], [1])
    with pytest.raises(ValueError):
        accuracy([[1, 2]], [[1, 2]])
    with pytest.raises(ValueError):
        accuracy([1.0, np.nan], [1.0, 2.0])


# ------------------------------------------------------------ confusion matrix
def test_confusion_matrix_binary():
    assert confusion_matrix(Y_TRUE, Y_PRED).tolist() == [[3, 1], [1, 3]]


def test_confusion_matrix_multiclass_known_example():
    cm = confusion_matrix([0, 1, 2, 2, 1, 0], [0, 2, 2, 2, 1, 1])
    assert cm.tolist() == [[1, 1, 0], [0, 1, 1], [0, 0, 2]]
    assert cm.sum() == 6


def test_confusion_matrix_label_order_and_missing_class():
    cm = confusion_matrix([0, 1], [0, 1], labels=[1, 0, 2])
    assert cm.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 0]]


def test_confusion_matrix_ignores_labels_not_listed():
    cm = confusion_matrix([0, 1, 2], [0, 1, 2], labels=[0, 1])
    assert cm.tolist() == [[1, 0], [0, 1]]


def test_confusion_matrix_string_labels():
    cm = confusion_matrix(["cat", "dog", "cat"], ["cat", "cat", "cat"])
    assert cm.tolist() == [[2, 0], [1, 0]]


def test_confusion_matrix_normalize():
    cm = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], normalize="true")
    assert np.allclose(cm, [[0.5, 0.5], [0.0, 1.0]])
    assert np.isclose(confusion_matrix(Y_TRUE, Y_PRED, normalize="all").sum(), 1.0)
    cm = confusion_matrix([0, 1], [0, 1], labels=[0, 1, 2], normalize="pred")
    assert np.isfinite(cm).all()  # empty column does not divide by zero


def test_confusion_matrix_invalid_arguments():
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0, 1], normalize="rows")
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0, 1], labels=[0, 0])


# -------------------------------------------------- precision / recall / f1
def test_binary_scores_known_values():
    assert precision(Y_TRUE, Y_PRED) == 0.75
    assert recall(Y_TRUE, Y_PRED) == 0.75
    assert f1(Y_TRUE, Y_PRED) == 0.75


def test_binary_scores_asymmetric_example():
    yt, yp = [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 1]
    assert precision(yt, yp) == 0.5
    assert recall(yt, yp) == pytest.approx(1 / 3)
    assert f1(yt, yp) == pytest.approx(0.4)


def test_f1_is_harmonic_mean_of_precision_and_recall():
    yt, yp = [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 1]
    p, r = precision(yt, yp), recall(yt, yp)
    assert f1(yt, yp) == pytest.approx(2 * p * r / (p + r))


def test_pos_label_zero_and_strings():
    assert precision(Y_TRUE, Y_PRED, pos_label=0) == 0.75
    assert recall(["a", "b", "b"], ["a", "a", "b"], pos_label="b") == 0.5


def test_zero_division_handling():
    assert precision([1, 1], [0, 0]) == 0.0
    assert precision([1, 1], [0, 0], zero_division=1.0) == 1.0
    assert recall([0, 0], [1, 1]) == 0.0
    assert f1([0, 0], [0, 0]) == 0.0
    assert f1([0, 0], [0, 0], zero_division=1.0) == 1.0


def test_multiclass_averages():
    yt = [0, 1, 2, 2, 1, 0]
    yp = [0, 2, 2, 2, 1, 1]
    p_each = precision(yt, yp, average=None)
    r_each = recall(yt, yp, average=None)
    assert np.allclose(p_each, [1.0, 0.5, 2 / 3])
    assert np.allclose(r_each, [0.5, 0.5, 1.0])
    assert precision(yt, yp, average="macro") == pytest.approx(p_each.mean())
    assert recall(yt, yp, average="micro") == pytest.approx(4 / 6)
    assert precision(yt, yp, average="micro") == accuracy(yt, yp)
    support = np.array([2, 2, 2])
    assert recall(yt, yp, average="weighted") == pytest.approx((r_each * support).sum() / 6)
    assert f1(yt, yp, average=None).shape == (3,)


def test_binary_average_rejects_multiclass_and_bad_option():
    with pytest.raises(ValueError):
        precision([0, 1, 2], [0, 1, 2])
    with pytest.raises(ValueError):
        precision([0, 1], [0, 1], average="samples")
    with pytest.raises(ValueError):
        precision(["a", "b"], ["a", "b"])  # pos_label=1 does not match string labels


def test_scores_with_single_class_present():
    assert precision([0, 0, 0], [0, 0, 0]) == 0.0  # no positives at all
    assert precision([0, 0, 0], [0, 0, 0], pos_label=0) == 1.0


# ------------------------------------------------------------------ regression
def test_mse_rmse_mae_known_values():
    assert mse([1, 2, 3], [1, 2, 5]) == pytest.approx(4 / 3)
    assert rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))
    assert mae([1, 2, 3], [1, 2, 5]) == pytest.approx(2 / 3)


def test_regression_perfect_and_multidimensional():
    A = np.random.default_rng(0).normal(size=(4, 3))
    assert mse(A, A) == 0.0 and mae(A, A) == 0.0
    assert mse(A, A + 1) == pytest.approx(1.0)


def test_regression_errors():
    with pytest.raises(ValueError):
        mse([1, 2], [1])
    with pytest.raises(ValueError):
        mse([], [])
    with pytest.raises(ValueError):
        mse([1, np.nan], [1, 2])
    with pytest.raises(TypeError):
        mse(["a"], ["b"])


def test_r2_score_cases():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, y) == 1.0
    assert r2_score(y, np.full(4, y.mean())) == pytest.approx(0.0)
    assert r2_score(y, y[::-1]) < 0
    assert r2_score([2, 2, 2], [2, 2, 2]) == 1.0 and r2_score([2, 2, 2], [1, 2, 3]) == 0.0


# ------------------------------------------------------------------ ROC / AUC
def test_roc_curve_known_example():
    fpr, tpr, thr = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert fpr.tolist() == [0.0, 0.0, 0.5, 0.5, 1.0]
    assert tpr.tolist() == [0.0, 0.5, 0.5, 1.0, 1.0]
    assert thr.tolist() == [np.inf, 0.8, 0.4, 0.35, 0.1]


def test_roc_auc_known_values():
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc_score([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_roc_auc_matches_pairwise_definition_with_ties():
    rng = np.random.default_rng(1)
    for _ in range(50):
        y = rng.integers(0, 2, size=20)
        if y.min() == y.max():
            continue
        s = rng.integers(0, 5, size=20).astype(float)  # many ties
        pos, neg = s[y == 1], s[y == 0]
        expected = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (pos.size * neg.size)
        assert roc_auc_score(y, s) == pytest.approx(expected)


def test_roc_curve_starts_at_origin_ends_at_one_and_is_monotone():
    rng = np.random.default_rng(2)
    y = np.r_[np.ones(10), np.zeros(10)]
    fpr, tpr, thr = roc_curve(y, rng.normal(size=20))
    assert (fpr[0], tpr[0]) == (0.0, 0.0) and (fpr[-1], tpr[-1]) == (1.0, 1.0)
    assert (np.diff(fpr) >= 0).all() and (np.diff(tpr) >= 0).all() and (np.diff(thr) < 0).all()


def test_roc_with_string_labels_and_pos_label():
    assert roc_auc_score(["n", "n", "p", "p"], [0.1, 0.4, 0.35, 0.8], pos_label="p") == 0.75


def test_roc_errors():
    with pytest.raises(ValueError):
        roc_curve([1, 1, 1], [0.1, 0.2, 0.3])  # one class only
    with pytest.raises(ValueError):
        roc_curve([0, 1], [0.1])
    with pytest.raises(ValueError):
        roc_curve([0, 1], [0.1, np.nan])
    with pytest.raises(ValueError):
        roc_curve([], [])


def test_auc_trapezoid_and_direction():
    assert auc([0, 1], [0, 1]) == 0.5
    assert auc([0, 0.5, 1], [0, 1, 1]) == 0.75
    assert auc([1, 0], [1, 0]) == 0.5  # decreasing x still gives a positive area


def test_auc_errors():
    with pytest.raises(ValueError):
        auc([0, 1, 0.5], [0, 1, 1])  # not monotonic
    with pytest.raises(ValueError):
        auc([0], [0])
    with pytest.raises(ValueError):
        auc([0, 1], [0, 1, 2])
