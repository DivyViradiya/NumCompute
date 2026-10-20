"""Unit tests for numcompute.pipeline."""
import numpy as np
import pytest

from numcompute.pipeline import (
    BaseEstimator,
    ColumnSelector,
    Compose,
    FeatureUnion,
    FunctionTransformer,
    NearestCentroidClassifier,
    Pipeline,
    Transformer,
)
from numcompute.preprocessing import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler
from numcompute.utils import NotFittedError


@pytest.fixture
def X():
    data = np.random.default_rng(0).normal(5, 3, size=(50, 4))
    data[2, 1] = np.nan
    return data


# --------------------------------------------------------------- estimator API
def test_get_set_params_and_repr():
    s = StandardScaler(with_mean=False)
    assert s.get_params() == {"with_mean": False, "with_std": True}
    assert s.set_params(with_std=False) is s and s.with_std is False
    assert repr(s) == "StandardScaler(with_mean=False, with_std=False)"
    with pytest.raises(ValueError):
        s.set_params(foo=1)


def test_base_classes_must_be_overridden():
    with pytest.raises(NotImplementedError):
        Transformer().fit([[1.0]])
    with pytest.raises(NotImplementedError):
        Transformer().transform([[1.0]])


def test_fit_transform_equals_fit_then_transform(X):
    a = StandardScaler().fit_transform(X)
    b = StandardScaler().fit(X).transform(X)
    assert np.array_equal(a, b, equal_nan=True)


# -------------------------------------------------------------------- Pipeline
def test_pipeline_chains_steps_in_order(X):
    pipe = Pipeline([("impute", Imputer("mean")), ("scale", StandardScaler())])
    out = pipe.fit_transform(X)
    manual = StandardScaler().fit_transform(Imputer("mean").fit_transform(X))
    assert np.allclose(out, manual)
    assert not np.isnan(out).any() and np.allclose(pipe.transform(X), out)


def test_pipeline_fit_then_transform_new_data_uses_training_statistics(X):
    pipe = Pipeline([("scale", StandardScaler())]).fit(X[:25])
    ref = StandardScaler().fit(X[:25]).transform(X[25:])
    assert np.allclose(pipe.transform(X[25:]), ref, equal_nan=True)


def test_pipeline_access_helpers():
    scaler = StandardScaler()
    pipe = Pipeline([("a", Imputer()), ("b", scaler)])
    assert len(pipe) == 2 and pipe["b"] is scaler and pipe[0] is pipe.steps[0][1]
    assert list(pipe.named_steps) == ["a", "b"]
    with pytest.raises(KeyError):
        pipe["zzz"]


def test_pipeline_with_final_estimator(X):
    Xc = np.nan_to_num(X)
    y = (Xc[:, 0] > 5).astype(int)
    pipe = Pipeline([("scale", StandardScaler()), ("clf", NearestCentroidClassifier())]).fit(Xc, y)
    pred = pipe.predict(Xc)
    assert pred.shape == (50,) and (pred == y).mean() > 0.8
    assert np.array_equal(pipe.fit_predict(Xc, y), pred)
    with pytest.raises(TypeError):
        pipe.transform(Xc)  # estimator has no transform


def test_pipeline_inverse_transform():
    X = np.random.default_rng(1).normal(size=(10, 3))
    pipe = Pipeline([("mm", MinMaxScaler()), ("ss", StandardScaler())])
    assert np.allclose(pipe.inverse_transform(pipe.fit_transform(X)), X)
    with pytest.raises(TypeError):
        Pipeline([("imp", Imputer()), ("ss", StandardScaler())]).fit(X).inverse_transform(X)


def test_pipeline_validation_errors():
    with pytest.raises(ValueError):
        Pipeline([])
    with pytest.raises(ValueError):
        Pipeline([("a", StandardScaler()), ("a", StandardScaler())])
    with pytest.raises(TypeError):
        Pipeline([StandardScaler()])  # not a (name, step) pair
    with pytest.raises(TypeError):
        Pipeline([("clf", NearestCentroidClassifier()), ("s", StandardScaler())])  # estimator mid-pipeline
    with pytest.raises(TypeError):
        Pipeline([("s", StandardScaler()), ("bad", object())])
    with pytest.raises(TypeError):
        Pipeline([("s", StandardScaler()), ("clf", NearestCentroidClassifier())]).fit_transform(np.ones((3, 2)), [0, 1, 0])


def test_pipeline_unfitted_use_raises(X):
    with pytest.raises(NotFittedError):
        Pipeline([("s", StandardScaler())]).transform(X)


# ----------------------------------------------- FunctionTransformer / Compose
def test_function_transformer_and_inverse():
    ft = FunctionTransformer(np.log1p, np.expm1)
    X = np.array([[0.0, 1.0], [3.0, 7.0]])
    assert np.allclose(ft.inverse_transform(ft.fit_transform(X)), X)
    assert np.array_equal(FunctionTransformer().fit_transform(X), X)
    with pytest.raises(TypeError):
        FunctionTransformer(func=5).transform(X)


def test_compose_wraps_callables_and_names_steps():
    c = Compose([np.abs, StandardScaler()])
    assert [n for n, _ in c.steps] == ["step0", "step1"]
    X = np.random.default_rng(2).normal(size=(20, 2))
    assert np.allclose(c.fit_transform(X), StandardScaler().fit_transform(np.abs(X)))
    named = Compose([("abs", np.abs), ("sc", StandardScaler())])
    assert list(named.named_steps) == ["abs", "sc"]
    with pytest.raises(TypeError):
        Compose([42])


# ------------------------------------------------- ColumnSelector / FeatureUnion
def test_column_selector_variants():
    X = np.arange(12.0).reshape(3, 4)
    assert ColumnSelector([0, 2]).fit_transform(X).tolist() == X[:, [0, 2]].tolist()
    assert ColumnSelector(1).fit_transform(X).shape == (3, 1)
    assert ColumnSelector(slice(1, 3)).fit_transform(X).shape == (3, 2)
    assert ColumnSelector(np.array([True, False, False, True])).fit_transform(X).shape == (3, 2)
    assert ColumnSelector([0], dtype=int).fit_transform(X).dtype.kind == "i"


def test_column_selector_errors():
    X = np.ones((3, 4))
    with pytest.raises(NotFittedError):
        ColumnSelector([0]).transform(X)
    with pytest.raises(ValueError):
        ColumnSelector([9]).fit(X).transform(X)
    with pytest.raises(ValueError):
        ColumnSelector([0]).fit(np.ones(3))
    with pytest.raises(ValueError):
        ColumnSelector([0]).fit(X).transform(np.ones((3, 2)))


def test_feature_union_concatenates_blocks(X):
    fu = FeatureUnion([("scaled", StandardScaler()), ("minmax", MinMaxScaler())])
    out = fu.fit_transform(np.nan_to_num(X))
    assert out.shape == (50, 8)
    assert np.allclose(fu.transform(np.nan_to_num(X)), out)


def test_feature_union_numeric_and_categorical_columns_together():
    data = np.array([["1.0", "a", "10"], ["2.0", "b", "20"], ["3.0", "a", "30"], ["nan", "c", "40"]], dtype=object)
    union = FeatureUnion(
        [
            ("num", Pipeline([("sel", ColumnSelector([0, 2], dtype=float)), ("imp", Imputer()), ("sc", StandardScaler())])),
            ("cat", Pipeline([("sel", ColumnSelector([1])), ("oh", OneHotEncoder())])),
        ]
    )
    out = union.fit_transform(data)
    assert out.shape == (4, 5)  # 2 numeric + 3 categories
    assert not np.isnan(out).any() and np.allclose(union.transform(data), out)
    assert out[:, 2:].sum(axis=1).tolist() == [1, 1, 1, 1]


def test_feature_union_errors():
    with pytest.raises(ValueError):
        FeatureUnion([])
    with pytest.raises(ValueError):
        FeatureUnion([("a", StandardScaler()), ("a", MinMaxScaler())])
    with pytest.raises(TypeError):
        FeatureUnion([StandardScaler()])
    with pytest.raises(TypeError):
        FeatureUnion([("a", object())])
    with pytest.raises(NotFittedError):
        FeatureUnion([("a", StandardScaler())]).transform(np.ones((2, 2)))
    bad = FeatureUnion([("one_d", FunctionTransformer(lambda x: x[:, 0]))])
    with pytest.raises(ValueError):
        bad.fit_transform(np.ones((3, 2)))
    mismatch = FeatureUnion([("a", FunctionTransformer()), ("b", FunctionTransformer(lambda x: x[:2]))])
    with pytest.raises(ValueError):
        mismatch.fit_transform(np.ones((3, 2)))


# ------------------------------------------------------ NearestCentroidClassifier
def test_nearest_centroid_separable_clusters():
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(0, 0.3, size=(30, 2)), rng.normal(5, 0.3, size=(30, 2))])
    y = np.r_[np.zeros(30, int), np.ones(30, int)]
    clf = NearestCentroidClassifier().fit(X, y)
    assert (clf.predict(X) == y).all()
    assert clf.classes_.tolist() == [0, 1] and clf.centroids_.shape == (2, 2)
    assert np.allclose(clf.centroids_[1], X[30:].mean(axis=0))


def test_nearest_centroid_string_labels_and_metric():
    X = np.array([[0.0, 0], [0, 1], [10, 10], [10, 11]])
    clf = NearestCentroidClassifier(metric="manhattan").fit(X, ["lo", "lo", "hi", "hi"])
    assert clf.predict([[1.0, 0.0], [9.0, 9.0]]).tolist() == ["lo", "hi"]


def test_nearest_centroid_errors():
    X = np.ones((4, 2))
    with pytest.raises(NotFittedError):
        NearestCentroidClassifier().predict(X)
    with pytest.raises(ValueError):
        NearestCentroidClassifier().fit(X, [0, 1])
    with pytest.raises(ValueError):
        NearestCentroidClassifier().fit(np.array([[np.nan, 1.0]]), [0])
    with pytest.raises(ValueError):
        NearestCentroidClassifier().fit(np.empty((0, 2)), [])
    clf = NearestCentroidClassifier().fit(X, [0, 0, 1, 1])
    with pytest.raises(ValueError):
        clf.predict(np.ones((2, 3)))
    with pytest.raises(ValueError):
        clf.predict([[np.nan, 1.0]])


def test_baseestimator_is_importable():
    assert issubclass(Transformer, BaseEstimator)
