"""Unit tests for numcompute.preprocessing."""
import numpy as np
import pytest

from numcompute.preprocessing import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler
from numcompute.utils import NotFittedError


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    data = rng.normal(5, 3, size=(50, 4))
    data[3, 1] = np.nan
    data[:, 2] = 7.0  # constant column
    return data


# --------------------------------------------------------------- StandardScaler
def test_standard_scaler_matches_manual_zscore(X):
    mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    sd[sd == 0] = 1
    assert np.allclose(StandardScaler().fit_transform(X), (X - mu) / sd, equal_nan=True)


def test_standard_scaler_output_has_zero_mean_unit_std():
    Z = StandardScaler().fit_transform(np.random.default_rng(1).normal(3, 2, size=(200, 3)))
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-12) and np.allclose(Z.std(axis=0), 1)


def test_standard_scaler_constant_column_becomes_zeros_not_nan(X):
    Z = StandardScaler().fit_transform(X)
    assert (Z[:, 2] == 0).all()
    assert (StandardScaler().fit_transform(np.full((5, 1), 0.1)) == 0).all()


def test_standard_scaler_keeps_nan_and_ignores_it_when_fitting(X):
    Z = StandardScaler().fit_transform(X)
    assert np.isnan(Z[3, 1]) and np.isnan(Z).sum() == 1


def test_standard_scaler_flags_and_inverse(X):
    s = StandardScaler().fit(X)
    assert np.allclose(s.inverse_transform(s.transform(X)), X, equal_nan=True)
    assert np.allclose(StandardScaler(with_mean=False).fit_transform(X)[:, 0], X[:, 0] / np.nanstd(X[:, 0]))
    assert np.allclose(StandardScaler(with_std=False).fit_transform(X)[:, 0], X[:, 0] - X[:, 0].mean())


def test_standard_scaler_all_nan_column():
    out = StandardScaler().fit_transform([[1.0, np.nan], [3.0, np.nan]])
    assert out[:, 0].tolist() == [-1.0, 1.0] and np.isnan(out[:, 1]).all()


def test_standard_scaler_does_not_modify_input_and_handles_strides(X):
    before = X.copy()
    StandardScaler().fit_transform(X)
    assert np.array_equal(X, before, equal_nan=True)
    view = X[::2, ::2]
    assert np.allclose(StandardScaler().fit_transform(view), StandardScaler().fit_transform(np.ascontiguousarray(view)), equal_nan=True)


def test_standard_scaler_errors(X):
    with pytest.raises(NotFittedError):
        StandardScaler().transform(X)
    with pytest.raises(ValueError):
        StandardScaler().fit(X).transform(X[:, :2])
    with pytest.raises(ValueError):
        StandardScaler().fit([1.0, 2.0, 3.0])  # 1-D
    with pytest.raises(ValueError):
        StandardScaler().fit(np.empty((0, 3)))
    with pytest.raises(ValueError):
        StandardScaler().fit([[np.inf]])
    with pytest.raises(TypeError):
        StandardScaler().fit([["a"]])


# ----------------------------------------------------------------- MinMaxScaler
def test_minmax_default_range_and_custom_range(X):
    M = MinMaxScaler().fit_transform(X)
    assert np.nanmin(M[:, 0]) == 0 and np.nanmax(M[:, 0]) == 1
    M2 = MinMaxScaler((-1, 1)).fit_transform(X)
    assert np.nanmin(M2[:, 0]) == -1 and np.nanmax(M2[:, 0]) == 1


def test_minmax_constant_column_maps_to_low(X):
    assert (MinMaxScaler((-1, 1)).fit_transform(X)[:, 2] == -1).all()


def test_minmax_inverse_clip_and_nan(X):
    mm = MinMaxScaler((0, 10)).fit(X)
    assert np.allclose(mm.inverse_transform(mm.transform(X)), X, equal_nan=True)
    assert np.isnan(mm.transform(X)[3, 1])
    assert np.nanmax(MinMaxScaler(clip=True).fit(X).transform(X + 100)) == 1
    assert np.nanmax(MinMaxScaler().fit(X).transform(X + 100)) > 1  # unseen data may leave the range


def test_minmax_known_values():
    assert MinMaxScaler((-1, 1)).fit_transform([[0.0], [5.0], [10.0]]).ravel().tolist() == [-1.0, 0.0, 1.0]


def test_minmax_errors(X):
    with pytest.raises(ValueError):
        MinMaxScaler((1, 0)).fit(X)
    with pytest.raises(ValueError):
        MinMaxScaler((1, 1)).fit(X)
    with pytest.raises(ValueError):
        MinMaxScaler("bad").fit(X)
    with pytest.raises(NotFittedError):
        MinMaxScaler().transform(X)
    with pytest.raises(ValueError):
        MinMaxScaler().fit(X).transform(X[:, :3])


# ---------------------------------------------------------------------- Imputer
@pytest.mark.parametrize("strategy, fn", [("mean", np.nanmean), ("median", np.nanmedian)])
def test_imputer_mean_and_median(X, strategy, fn):
    out = Imputer(strategy).fit_transform(X)
    assert not np.isnan(out).any()
    assert np.isclose(out[3, 1], fn(X[:, 1]))


def test_imputer_constant_and_most_frequent():
    X = np.array([[1.0, 2], [1, np.nan], [3, 2], [np.nan, 5]])
    assert Imputer("constant", fill_value=-5).fit_transform(X)[1, 1] == -5
    out = Imputer("most_frequent").fit_transform(X)
    assert out[3, 0] == 1 and out[1, 1] == 2


def test_imputer_most_frequent_tie_picks_smallest():
    assert Imputer("most_frequent").fit_transform([[3.0], [1.0], [np.nan]])[2, 0] == 1.0


def test_imputer_sentinel_and_all_missing_column():
    assert Imputer(missing_values=-999).fit_transform([[1.0, -999], [3.0, 4.0]])[0, 1] == 4.0
    assert Imputer().fit_transform([[np.nan], [np.nan]])[0, 0] == 0.0
    assert Imputer(fill_value=9).fit_transform([[np.nan], [np.nan]])[0, 0] == 9.0


def test_imputer_uses_training_statistics_on_new_data():
    imp = Imputer("mean").fit([[1.0], [3.0]])
    assert imp.transform([[np.nan]])[0, 0] == 2.0


def test_imputer_leaves_complete_data_unchanged(X):
    clean = X[:, [0, 2]]
    assert np.array_equal(Imputer().fit_transform(clean), clean)


def test_imputer_errors(X):
    with pytest.raises(ValueError):
        Imputer("mode").fit(X)
    with pytest.raises(NotFittedError):
        Imputer().transform(X)
    with pytest.raises(ValueError):
        Imputer().fit(X).transform(X[:, :2])


# ------------------------------------------------------------------ OneHotEncoder
def test_one_hot_basic_and_category_order():
    enc = OneHotEncoder().fit([["b"], ["a"], ["c"], ["a"]])
    assert [c.tolist() for c in enc.categories_] == [["a", "b", "c"]]
    assert enc.transform([["c"], ["a"]]).tolist() == [[0, 0, 1], [1, 0, 0]]


def test_one_hot_multiple_columns_and_feature_names():
    enc = OneHotEncoder().fit([["a", "x"], ["b", "y"], ["a", "y"]])
    out = enc.transform([["a", "y"], ["b", "x"]])
    assert out.tolist() == [[1, 0, 0, 1], [0, 1, 1, 0]]
    assert enc.get_feature_names_out(["f", "g"]) == ["f_a", "f_b", "g_x", "g_y"]
    assert enc.n_output_features_ == 4


def test_one_hot_numeric_and_1d_input():
    assert OneHotEncoder().fit_transform([3, 1, 2, 1]).tolist() == [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]]


def test_one_hot_each_row_block_sums_to_one():
    X = np.array([["a", "x"], ["b", "y"], ["a", "y"]])
    out = OneHotEncoder().fit_transform(X)
    assert (out[:, :2].sum(axis=1) == 1).all() and (out[:, 2:].sum(axis=1) == 1).all()


def test_one_hot_unknown_category_error_and_ignore():
    enc = OneHotEncoder().fit([["a"], ["b"]])
    with pytest.raises(ValueError):
        enc.transform([["z"]])
    ignore = OneHotEncoder(handle_unknown="ignore").fit([["a"], ["b"]])
    assert ignore.transform([["z"], ["a"]]).tolist() == [[0, 0], [1, 0]]


def test_one_hot_inverse_transform():
    enc = OneHotEncoder().fit([["a", "x"], ["b", "y"]])
    X = np.array([["b", "x"], ["a", "y"]])
    assert (enc.inverse_transform(enc.transform(X)) == X).all()
    ignore = OneHotEncoder(handle_unknown="ignore").fit([["a"], ["b"]])
    back = ignore.inverse_transform(ignore.transform([["z"], ["a"]]))
    assert back[0, 0] is None and back[1, 0] == "a"


def test_one_hot_dtype_option():
    out = OneHotEncoder(dtype=int).fit_transform([["a"], ["b"]])
    assert out.dtype.kind == "i"


def test_one_hot_errors():
    with pytest.raises(ValueError):
        OneHotEncoder().fit([[np.nan], [1.0]])
    with pytest.raises(ValueError):
        OneHotEncoder().fit(np.empty((0, 1)))
    with pytest.raises(ValueError):
        OneHotEncoder(handle_unknown="skip").fit([["a"]])
    with pytest.raises(NotFittedError):
        OneHotEncoder().transform([["a"]])
    with pytest.raises(ValueError):
        OneHotEncoder().fit([["a", "b"]]).transform([["a"]])
    with pytest.raises(ValueError):
        OneHotEncoder().fit([["a"]]).get_feature_names_out(["x", "y"])
    with pytest.raises(ValueError):
        OneHotEncoder().fit([["a"], ["b"]]).inverse_transform(np.ones((1, 5)))
