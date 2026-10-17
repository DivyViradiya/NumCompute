# NumCompute Conventions

Every module follows these rules. Tests should check them.

## Shapes and axes
- Feature matrices are `(n_samples, n_features)`; column-wise operations use `axis=0`.
- Functions that take an `axis` argument accept negative axes; an invalid axis raises `ValueError`.
- 1-D inputs to scalers/encoders are rejected with `ValueError` (reshape to 2-D first),
  except `OneHotEncoder`, which accepts 1-D and treats it as one column.

## NaN policy
A shared `nan_policy` argument is used wherever NaNs matter:
- `'propagate'` – a NaN in a slice makes that slice's result NaN.
- `'omit'`      – NaNs are ignored; an all-NaN slice gives NaN.
- `'raise'`     – `ValueError` if any NaN is present.
Exceptions (documented per function): `rank` (default `'omit'`: NaN elements get NaN rank),
`histogram` (`'omit'` or `'raise'` only), `topk` (NaN always ranks last, never preferred),
`quickselect` (NaN raises).

## Errors
- `TypeError` for wrong types (e.g. non-numeric data, non-integer `k`).
- `ValueError` for bad shapes, values, empty input where meaningless, bad option strings.
- `NotFittedError` (subclass of `ValueError`) when `transform`/`predict` precede `fit`.
- Messages say what was received and what was expected.

## Ties
- `topk`: ties at the boundary resolve to the lowest index; output is deterministic.
- `binary_search(a, x, side='left')`: returns `(insertion_index, found)` with `bisect_left`
  semantics by default; `side='right'` gives `bisect_right`.
- `rank`: `average`, `min`, `max`, `dense`, `ordinal` (ordinal = order of appearance).
- Sorts are stable.

## Estimator API
- `fit(X, y=None) -> self`, `transform(X)`, `fit_transform(X, y=None)`, `predict(X)`.
- Learned attributes end with an underscore (`mean_`, `categories_`).
- Transformers never modify their input in place.

## Docstrings (numpydoc)
Summary, Parameters (with shapes), Returns (with shapes), Raises, Notes with
**Complexity** (time/space), and Examples where useful.

## Style
PEP 8, type hints on public functions, `from __future__ import annotations`.
