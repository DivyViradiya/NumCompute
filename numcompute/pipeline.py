"""Transformer / Estimator API, Pipeline, Compose, FeatureUnion.

Protocol
--------
* Transformers: ``fit(X, y=None) -> self`` and ``transform(X)``.
* Estimators:   ``fit(X, y) -> self`` and ``predict(X)``.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from .utils import check_2d, check_is_fitted, one_hot, pairwise_distances

__all__ = [
    "BaseEstimator",
    "Transformer",
    "Estimator",
    "FunctionTransformer",
    "ColumnSelector",
    "Pipeline",
    "Compose",
    "FeatureUnion",
    "NearestCentroidClassifier",
]


class BaseEstimator:
    """Base class giving ``get_params``, ``set_params`` and a readable ``repr``.

    Constructor arguments must be stored as attributes of the same name.
    """

    @classmethod
    def _param_names(cls) -> List[str]:
        sig = inspect.signature(cls.__init__)
        skip = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        return sorted(
            n for n, p in sig.parameters.items() if n != "self" and p.kind not in skip
        )

    def get_params(self) -> Dict[str, Any]:
        """Return constructor parameters as a dict.

        Returns
        -------
        dict
            ``{name: value}`` for every constructor argument.

        Notes
        -----
        Time O(p), space O(p) for p parameters.
        """
        return {n: getattr(self, n, None) for n in self._param_names()}

    def set_params(self, **params) -> "BaseEstimator":
        """Set constructor parameters.

        Parameters
        ----------
        **params
            Parameter names and new values.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If a name is not a constructor parameter.

        Notes
        -----
        Time O(p), space O(1).
        """
        valid = self._param_names()
        for name, value in params.items():
            if name not in valid:
                raise ValueError(
                    f"invalid parameter {name!r} for {type(self).__name__}; valid: {valid}"
                )
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"


class Transformer(BaseEstimator):
    """Base class for transformers (``fit`` + ``transform``)."""

    def fit(self, X, y=None):
        """Learn from ``X``. Subclasses must override. Returns ``self``."""
        raise NotImplementedError

    def transform(self, X):
        """Apply the learned transformation. Subclasses must override."""
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        """Fit to ``X`` and return the transformed ``X``.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.
        y : array_like, optional
            Ignored by most transformers; accepted for pipeline compatibility.

        Returns
        -------
        numpy.ndarray
            Transformed data.

        Notes
        -----
        Cost is that of ``fit`` plus ``transform``.
        """
        return self.fit(X, y).transform(X)


class Estimator(BaseEstimator):
    """Base class for models (``fit`` + ``predict``)."""

    def fit(self, X, y):
        """Train on ``(X, y)``. Subclasses must override. Returns ``self``."""
        raise NotImplementedError

    def predict(self, X):
        """Predict targets for ``X``. Subclasses must override."""
        raise NotImplementedError


class FunctionTransformer(Transformer):
    """Wrap a plain function as a stateless transformer.

    Parameters
    ----------
    func : callable, optional
        Applied to ``X`` in ``transform``. ``None`` means identity.
    inverse_func : callable, optional
        Applied in ``inverse_transform``. ``None`` means identity.

    Examples
    --------
    >>> FunctionTransformer(np.log1p).fit_transform([[0.0, 1.0]]).shape
    (1, 2)
    """

    def __init__(self, func: Callable = None, inverse_func: Callable = None):
        self.func = func
        self.inverse_func = inverse_func

    def fit(self, X, y=None):
        """No-op. Returns ``self``."""
        return self

    def transform(self, X):
        """Apply ``func`` to ``X``.

        Parameters
        ----------
        X : array_like
            Any shape.

        Returns
        -------
        numpy.ndarray
            ``func(X)``.

        Raises
        ------
        TypeError
            If ``func`` is not callable.
        """
        X = np.asarray(X)
        if self.func is None:
            return X
        if not callable(self.func):
            raise TypeError("func must be callable or None")
        return np.asarray(self.func(X))

    def inverse_transform(self, X):
        """Apply ``inverse_func`` to ``X`` (identity if None)."""
        X = np.asarray(X)
        return X if self.inverse_func is None else np.asarray(self.inverse_func(X))


class ColumnSelector(Transformer):
    """Select columns of a 2-D array (use with :class:`FeatureUnion`).

    Parameters
    ----------
    columns : int, slice, sequence of int, or boolean mask
        Columns to keep.
    dtype : type or str, optional
        Cast the selected block (e.g. ``float`` for numeric columns taken
        from a string array). ``None`` keeps the input dtype.

    Notes
    -----
    Output shape is ``(n_samples, n_selected)``. Time O(n * k), space O(n * k).
    """

    def __init__(self, columns, dtype=None):
        self.columns = columns
        self.dtype = dtype

    def fit(self, X, y=None):
        """Record the number of input features. Returns ``self``.

        Raises
        ------
        ValueError
            If ``X`` is not 2-D.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Return the selected columns as a 2-D array.

        Raises
        ------
        NotFittedError
            If ``fit`` was not called.
        ValueError
            If ``X`` is not 2-D, has a different feature count than at fit
            time, or ``columns`` is invalid.
        """
        check_is_fitted(self, "n_features_in_")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have shape (n, {self.n_features_in_}), got {X.shape}"
            )
        cols = self.columns
        if isinstance(cols, (int, np.integer)):
            cols = [int(cols)]
        try:
            out = X[:, cols]
        except IndexError as exc:
            raise ValueError(f"invalid columns {self.columns!r}: {exc}") from exc
        return out if self.dtype is None else out.astype(self.dtype)


def _is_transformer(obj) -> bool:
    return hasattr(obj, "fit") and hasattr(obj, "transform")


class Pipeline(BaseEstimator):
    """Chain transformers, optionally ending in an estimator.

    Parameters
    ----------
    steps : list of (str, object)
        ``(name, step)`` pairs. All steps but the last must implement
        ``fit`` and ``transform``; the last must implement ``fit`` (and
        ``transform`` or ``predict`` to be used that way). Names are unique.

    Raises
    ------
    TypeError
        If a step is not a ``(name, object)`` pair or lacks required methods.
    ValueError
        If ``steps`` is empty or names repeat.

    Examples
    --------
    >>> pipe = Pipeline([("impute", Imputer()), ("scale", StandardScaler())])  # doctest: +SKIP
    >>> Xt = pipe.fit_transform(X)  # doctest: +SKIP

    Notes
    -----
    Cost is the sum of the costs of the steps.
    """

    def __init__(self, steps: Sequence[Tuple[str, Any]]):
        self.steps = list(steps)
        self._validate()

    def _validate(self) -> None:
        if not self.steps:
            raise ValueError("steps must not be empty")
        names = []
        for item in self.steps:
            if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)):
                raise TypeError(f"each step must be a (name, object) tuple, got {item!r}")
            names.append(item[0])
        if len(set(names)) != len(names):
            raise ValueError(f"step names must be unique, got {names}")
        for name, step in self.steps[:-1]:
            if not _is_transformer(step):
                raise TypeError(
                    f"intermediate step {name!r} must implement fit and transform"
                )
        if not hasattr(self.steps[-1][1], "fit"):
            raise TypeError(f"final step {self.steps[-1][0]!r} must implement fit")

    @property
    def named_steps(self) -> Dict[str, Any]:
        """Dict mapping step name to step object."""
        return dict(self.steps)

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return self.named_steps[key]
            except KeyError:
                raise KeyError(f"no step named {key!r}; have {list(self.named_steps)}") from None
        return self.steps[key][1]

    def __len__(self) -> int:
        return len(self.steps)

    def _fit_intermediate(self, X, y):
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.fit_transform(Xt, y) if hasattr(step, "fit_transform") else step.fit(Xt, y).transform(Xt)
        return Xt

    def fit(self, X, y=None) -> "Pipeline":
        """Fit every step in order.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.
        y : array_like, optional
            Targets passed to every step's ``fit``.

        Returns
        -------
        Pipeline
            ``self``.
        """
        Xt = self._fit_intermediate(X, y)
        self.steps[-1][1].fit(Xt, y)
        return self

    def fit_transform(self, X, y=None):
        """Fit all steps and return the output of the last ``transform``.

        Raises
        ------
        TypeError
            If the final step has no ``transform``.
        """
        Xt = self._fit_intermediate(X, y)
        last = self.steps[-1][1]
        if hasattr(last, "fit_transform"):
            return last.fit_transform(Xt, y)
        if not hasattr(last, "transform"):
            raise TypeError("final step has no transform; use fit/predict instead")
        return last.fit(Xt, y).transform(Xt)

    def transform(self, X):
        """Apply ``transform`` of every step.

        Raises
        ------
        TypeError
            If the final step has no ``transform``.
        """
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        last = self.steps[-1][1]
        if not hasattr(last, "transform"):
            raise TypeError("final step has no transform; use predict instead")
        return last.transform(Xt)

    def predict(self, X):
        """Transform with all but the last step, then ``predict``.

        Raises
        ------
        TypeError
            If the final step has no ``predict``.
        """
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        last = self.steps[-1][1]
        if not hasattr(last, "predict"):
            raise TypeError("final step has no predict method")
        return last.predict(Xt)

    def fit_predict(self, X, y):
        """Fit the pipeline, then predict on ``X``."""
        return self.fit(X, y).predict(X)

    def inverse_transform(self, X):
        """Undo the transformations in reverse order.

        Raises
        ------
        TypeError
            If a step has no ``inverse_transform``.
        """
        Xt = X
        for name, step in reversed(self.steps):
            if not hasattr(step, "inverse_transform"):
                raise TypeError(f"step {name!r} has no inverse_transform")
            Xt = step.inverse_transform(Xt)
        return Xt


class Compose(Pipeline):
    """Sequential composition of transformers and plain functions.

    Parameters
    ----------
    steps : sequence
        Each item is a transformer, a callable (wrapped in
        :class:`FunctionTransformer`), or a ``(name, object)`` pair. Unnamed
        items are called ``step0``, ``step1``, ...

    Examples
    --------
    >>> Compose([np.log1p, StandardScaler()]).fit_transform([[1.0], [2.0], [4.0]]).shape  # doctest: +SKIP
    (3, 1)
    """

    def __init__(self, steps: Sequence[Any]):
        named: List[Tuple[str, Any]] = []
        for i, item in enumerate(steps):
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                name, obj = item
            else:
                name, obj = f"step{i}", item
            if not hasattr(obj, "fit"):
                if not callable(obj):
                    raise TypeError(f"step {name!r} must be a transformer or callable")
                obj = FunctionTransformer(obj)
            named.append((name, obj))
        super().__init__(named)


class FeatureUnion(Transformer):
    """Apply several transformers to the same input and concatenate columns.

    Parameters
    ----------
    transformer_list : list of (str, transformer)
        Each transformer sees the full ``X``; combine with
        :class:`ColumnSelector` to send different columns to different
        transformers.

    Raises
    ------
    TypeError
        If an item is not a ``(name, transformer)`` pair.
    ValueError
        If the list is empty, names repeat, or outputs are not 2-D with a
        common number of rows.

    Notes
    -----
    Output shape ``(n_samples, sum of output widths)``. Cost is the sum of
    the transformers' costs plus O(n * total_width) for concatenation.
    """

    def __init__(self, transformer_list: Sequence[Tuple[str, Any]]):
        self.transformer_list = list(transformer_list)
        if not self.transformer_list:
            raise ValueError("transformer_list must not be empty")
        names = []
        for item in self.transformer_list:
            if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)):
                raise TypeError(f"each item must be a (name, transformer) tuple, got {item!r}")
            if not _is_transformer(item[1]):
                raise TypeError(f"{item[0]!r} must implement fit and transform")
            names.append(item[0])
        if len(set(names)) != len(names):
            raise ValueError(f"names must be unique, got {names}")

    def fit(self, X, y=None) -> "FeatureUnion":
        """Fit every transformer on ``X``. Returns ``self``."""
        for _, t in self.transformer_list:
            t.fit(X, y)
        self.fitted_ = True
        return self

    @staticmethod
    def _stack(blocks: List[np.ndarray]) -> np.ndarray:
        for b in blocks:
            if b.ndim != 2:
                raise ValueError(f"every transformer must return a 2-D array, got shape {b.shape}")
        if len({b.shape[0] for b in blocks}) != 1:
            raise ValueError(f"transformers returned different row counts: {[b.shape[0] for b in blocks]}")
        return np.hstack(blocks)

    def transform(self, X):
        """Transform with every transformer and concatenate along axis 1.

        Raises
        ------
        NotFittedError
            If ``fit`` was not called.
        ValueError
            If outputs cannot be concatenated.
        """
        check_is_fitted(self, "fitted_")
        return self._stack([np.asarray(t.transform(X)) for _, t in self.transformer_list])

    def fit_transform(self, X, y=None):
        """Fit and transform in one pass."""
        blocks = []
        for _, t in self.transformer_list:
            if hasattr(t, "fit_transform"):
                blocks.append(np.asarray(t.fit_transform(X, y)))
            else:
                blocks.append(np.asarray(t.fit(X, y).transform(X)))
        self.fitted_ = True
        return self._stack(blocks)


class NearestCentroidClassifier(Estimator):
    """Minimal classifier used to demonstrate ``Pipeline.predict``.

    Each class is represented by the mean of its training rows; a sample gets
    the label of the closest centroid (ties go to the first class).

    Parameters
    ----------
    metric : str, optional
        Any metric accepted by :func:`numcompute.utils.pairwise_distances`.
        Default ``'euclidean'``.

    Attributes
    ----------
    classes_ : numpy.ndarray, shape (n_classes,)
        Sorted unique labels.
    centroids_ : numpy.ndarray, shape (n_classes, n_features)
        Mean of each class.

    Notes
    -----
    Fit: time O(n d + n c), space O(n c + c d). Predict: time O(m c d),
    space O(m c) for m samples, c classes, d features.
    """

    def __init__(self, metric: str = "euclidean"):
        self.metric = metric

    def fit(self, X, y) -> "NearestCentroidClassifier":
        """Compute class centroids.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data without NaN.
        y : array_like, shape (n_samples,)
            Class labels.

        Returns
        -------
        NearestCentroidClassifier
            ``self``.

        Raises
        ------
        ValueError
            If ``X`` is not 2-D, empty, contains NaN, or ``y`` has the wrong
            length or shape.
        """
        X = check_2d(X, "X", allow_empty=False)
        if np.isnan(X).any():
            raise ValueError("X contains NaN; impute it first")
        y = np.asarray(y)
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"y must have shape ({X.shape[0]},), got {y.shape}")
        self.classes_, inverse = np.unique(y, return_inverse=True)
        Y = one_hot(inverse.ravel(), len(self.classes_))
        self.centroids_ = (Y.T @ X) / Y.sum(axis=0)[:, None]
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Data without NaN.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Predicted labels.

        Raises
        ------
        NotFittedError
            If ``fit`` was not called.
        ValueError
            If ``X`` has the wrong shape or contains NaN.
        """
        check_is_fitted(self, "centroids_")
        X = check_2d(X, "X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fitted with "
                f"{self.n_features_in_}"
            )
        if np.isnan(X).any():
            raise ValueError("X contains NaN; impute it first")
        D = pairwise_distances(X, self.centroids_, metric=self.metric)
        return self.classes_[np.argmin(D, axis=1)]
