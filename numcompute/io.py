"""CSV input: delimiters, missing values, dtype handling, chunked reading.

Built on :func:`numpy.genfromtxt` (no pandas). Float data represents missing
values as NaN; string data represents them as the literal ``"nan"`` unless a
fill value is requested.
"""
from __future__ import annotations

import itertools
import os
import warnings
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["MISSING_TOKENS", "read_header", "load_csv", "load_mixed_csv", "iter_csv_chunks"]

#: Strings treated as "missing" in addition to empty fields.
MISSING_TOKENS: Tuple[str, ...] = ("", "NA", "N/A", "NaN", "nan", "NULL", "null", "?")

_MISSING_POLICIES = ("nan", "fill", "skip")


def _check_path(path) -> str:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"path must be a str or PathLike, got {type(path).__name__}")
    p = os.fspath(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"CSV file not found: {p!r}")
    return p


def _check_common(delimiter, skip_header, missing) -> None:
    if not isinstance(delimiter, str) or delimiter == "":
        raise ValueError(f"delimiter must be a non-empty string, got {delimiter!r}")
    if isinstance(skip_header, bool) or not isinstance(skip_header, (int, np.integer)) or skip_header < 0:
        raise ValueError(f"skip_header must be a non-negative int, got {skip_header!r}")
    if missing not in _MISSING_POLICIES:
        raise ValueError(f"missing must be one of {_MISSING_POLICIES}, got {missing!r}")


def read_header(path, delimiter: str = ",", encoding: str = "utf-8") -> List[str]:
    """Read the first line of a CSV file as column names.

    Parameters
    ----------
    path : str or os.PathLike
        CSV file.
    delimiter : str, optional
        Field separator. Default ``','``.
    encoding : str, optional
        Text encoding. Default ``'utf-8'``.

    Returns
    -------
    list of str
        Column names (stripped of surrounding whitespace).

    Raises
    ------
    TypeError
        If ``path`` is not path-like.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty.

    Notes
    -----
    Time O(first line length), space O(first line length).
    """
    p = _check_path(path)
    with open(p, "r", encoding=encoding) as fh:
        line = fh.readline()
    if not line.strip():
        raise ValueError(f"{p!r} is empty; no header found")
    return [c.strip() for c in line.rstrip("\r\n").split(delimiter)]


def _read_strings(source, delimiter, skip_header, usecols, comments, encoding) -> np.ndarray:
    """Read a CSV source (path or list of lines) as a 2-D string array."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="genfromtxt: Empty input file")
            arr = np.genfromtxt(
            source,
                delimiter=delimiter,
                skip_header=skip_header,
                dtype=str,
                usecols=usecols,
                comments=comments,
                ndmin=2,
                encoding=encoding,
                filling_values="",
            )
    except ValueError as exc:
        raise ValueError(f"could not parse CSV (ragged rows or bad delimiter?): {exc}") from exc
    return np.char.strip(arr)


def _strings_to_float(arr: np.ndarray, tokens: Sequence[str]) -> np.ndarray:
    """Convert a string array to float; missing tokens become NaN."""
    missing = np.isin(arr, np.asarray(tokens, dtype=str))
    cleaned = np.where(missing, "nan", arr)
    try:
        return cleaned.astype(float)
    except ValueError as exc:
        raise ValueError(f"non-numeric value found in a numeric column: {exc}") from exc


def _apply_missing_float(data: np.ndarray, missing: str, fill_value) -> np.ndarray:
    if missing == "fill":
        return np.where(np.isnan(data), float(fill_value), data)
    if missing == "skip":
        return data[~np.isnan(data).any(axis=1)]
    return data


def _finish(arr_float: np.ndarray, dtype, missing: str, fill_value) -> np.ndarray:
    out = _apply_missing_float(arr_float, missing, fill_value)
    if dtype in (float, np.float64, "float", "float64", None):
        return out.astype(float, copy=False)
    kind = np.dtype(dtype).kind
    if kind in "iu":
        if np.isnan(out).any():
            raise ValueError(
                "integer dtype cannot hold missing values; use missing='fill' "
                "with an integer fill_value or missing='skip'"
            )
        return out.astype(dtype)
    if kind == "f":
        return out.astype(dtype)
    raise ValueError(f"unsupported numeric dtype {dtype!r}")


def load_csv(
    path,
    delimiter: str = ",",
    skip_header: int = 0,
    dtype=float,
    usecols: Optional[Sequence[int]] = None,
    missing: str = "nan",
    fill_value: Union[float, str] = 0.0,
    missing_tokens: Sequence[str] = MISSING_TOKENS,
    comments: str = "#",
    encoding: str = "utf-8",
) -> np.ndarray:
    """Load a CSV file into a 2-D NumPy array.

    Parameters
    ----------
    path : str or os.PathLike
        CSV file.
    delimiter : str, optional
        Field separator (``','``, ``'\\t'``, ``';'`` ...). Default ``','``.
    skip_header : int, optional
        Number of leading lines to skip (use 1 for a header row).
    dtype : type or str, optional
        ``float`` (default), an int type, or ``str`` for text. Integer dtypes
        reject missing values unless they are filled or skipped.
    usecols : sequence of int, optional
        Columns to read.
    missing : {'nan', 'fill', 'skip'}, optional
        How to treat missing entries (empty fields and ``missing_tokens``):

        * ``'nan'``  - keep as NaN (float) or the string ``'nan'`` (str).
        * ``'fill'`` - replace with ``fill_value``.
        * ``'skip'`` - drop every row that contains a missing value.
    fill_value : float or str, optional
        Replacement when ``missing='fill'``. Default 0.0.
    missing_tokens : sequence of str, optional
        Strings regarded as missing. Defaults to :data:`MISSING_TOKENS`.
    comments : str, optional
        Lines starting with this character are ignored.
    encoding : str, optional
        Text encoding.

    Returns
    -------
    numpy.ndarray
        2-D array of shape ``(n_rows, n_cols)``. A single row or column is
        still 2-D.

    Raises
    ------
    TypeError
        If ``path`` is not path-like.
    FileNotFoundError
        If the file does not exist.
    ValueError
        For bad options, ragged rows, non-numeric text in a numeric load,
        integer dtype with missing values, or when no data rows remain.

    Notes
    -----
    Whole file is read at once; for big files use :func:`iter_csv_chunks`.
    Time O(n_rows * n_cols), space O(n_rows * n_cols).

    Examples
    --------
    >>> X = load_csv("data.csv", skip_header=1, missing="fill", fill_value=0)  # doctest: +SKIP
    """
    p = _check_path(path)
    _check_common(delimiter, skip_header, missing)
    raw = _read_strings(p, delimiter, skip_header, usecols, comments, encoding)
    if raw.size == 0:
        raise ValueError(f"no data rows found in {p!r}")

    if dtype in (str, "str", np.str_) or np.dtype(dtype).kind in "US":
        is_missing = np.isin(raw, np.asarray(missing_tokens, dtype=str))
        if missing == "skip":
            return raw[~is_missing.any(axis=1)]
        repl = str(fill_value) if missing == "fill" else "nan"
        return np.where(is_missing, repl, raw)

    data = _strings_to_float(raw, missing_tokens)
    out = _finish(data, dtype, missing, fill_value)
    if out.size == 0 and missing == "skip":
        raise ValueError("every row contained a missing value; nothing left after skipping")
    return out


def load_mixed_csv(
    path,
    categorical_cols: Sequence[Union[int, str]],
    delimiter: str = ",",
    skip_header: int = 1,
    missing_tokens: Sequence[str] = MISSING_TOKENS,
    categorical_missing: str = "missing",
    comments: str = "#",
    encoding: str = "utf-8",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load a CSV that mixes numeric and categorical columns.

    Parameters
    ----------
    path : str or os.PathLike
        CSV file; the first line must be a header when columns are named.
    categorical_cols : sequence of int or str
        Columns holding categories, given by index or by header name.
    delimiter : str, optional
        Field separator.
    skip_header : int, optional
        Header lines to skip. Default 1.
    missing_tokens : sequence of str, optional
        Strings regarded as missing.
    categorical_missing : str, optional
        Label used for a missing categorical entry. Default ``'missing'``.
    comments : str, optional
        Comment marker.
    encoding : str, optional
        Text encoding.

    Returns
    -------
    X_num : numpy.ndarray, shape (n, n_numeric)
        Float data with NaN for missing values.
    X_cat : numpy.ndarray, shape (n, n_categorical)
        String data.
    num_names, cat_names : list of str
        Column names (or ``'col<i>'`` when the file has no header).

    Raises
    ------
    TypeError
        If ``path`` is not path-like.
    FileNotFoundError
        If the file does not exist.
    ValueError
        For unknown column names/indices, non-numeric text in a numeric
        column, ragged rows, or no data.

    Notes
    -----
    Time O(n * m), space O(n * m).
    """
    p = _check_path(path)
    _check_common(delimiter, skip_header, "nan")
    raw = _read_strings(p, delimiter, skip_header, None, comments, encoding)
    if raw.size == 0:
        raise ValueError(f"no data rows found in {p!r}")
    n_cols = raw.shape[1]
    names = (
        read_header(p, delimiter, encoding)
        if skip_header >= 1
        else [f"col{i}" for i in range(n_cols)]
    )
    if len(names) != n_cols:
        names = [f"col{i}" for i in range(n_cols)]

    cat_idx: List[int] = []
    for c in categorical_cols:
        if isinstance(c, str):
            if c not in names:
                raise ValueError(f"unknown column name {c!r}; available: {names}")
            cat_idx.append(names.index(c))
        else:
            if not -n_cols <= int(c) < n_cols:
                raise ValueError(f"column index {c} out of range for {n_cols} columns")
            cat_idx.append(int(c) % n_cols)
    cat_idx = sorted(set(cat_idx))
    num_idx = [i for i in range(n_cols) if i not in cat_idx]

    tokens = np.asarray(missing_tokens, dtype=str)
    X_cat = raw[:, cat_idx]
    X_cat = np.where(np.isin(X_cat, tokens), categorical_missing, X_cat)
    X_num = _strings_to_float(raw[:, num_idx], missing_tokens)
    return X_num, X_cat, [names[i] for i in num_idx], [names[i] for i in cat_idx]


def iter_csv_chunks(
    path,
    chunksize: int,
    delimiter: str = ",",
    skip_header: int = 0,
    usecols: Optional[Sequence[int]] = None,
    missing: str = "nan",
    fill_value: float = 0.0,
    missing_tokens: Sequence[str] = MISSING_TOKENS,
    comments: str = "#",
    encoding: str = "utf-8",
) -> Iterator[np.ndarray]:
    """Stream a numeric CSV in chunks of at most ``chunksize`` rows.

    Parameters
    ----------
    path : str or os.PathLike
        CSV file.
    chunksize : int
        Maximum rows per yielded array, ``>= 1``.
    delimiter, skip_header, usecols, comments, encoding
        As in :func:`load_csv`.
    missing : {'nan', 'fill', 'skip'}, optional
        As in :func:`load_csv` (applied per chunk).
    fill_value : float, optional
        Replacement when ``missing='fill'``.
    missing_tokens : sequence of str, optional
        Strings regarded as missing.

    Yields
    ------
    numpy.ndarray
        Float array of shape ``(<= chunksize, n_cols)``. Chunks that become
        empty after ``missing='skip'`` are not yielded.

    Raises
    ------
    TypeError
        If ``path`` is not path-like.
    FileNotFoundError
        If the file does not exist.
    ValueError
        For bad options or non-numeric text.

    Notes
    -----
    Only one chunk is held in memory: space O(chunksize * n_cols); total time
    O(n_rows * n_cols). Pair it with ``stats.RunningStats`` for one-pass
    statistics on files larger than memory. Blank and comment lines are
    skipped and do not count towards ``chunksize``.
    """
    if isinstance(chunksize, bool) or not isinstance(chunksize, (int, np.integer)) or chunksize < 1:
        raise ValueError(f"chunksize must be an int >= 1, got {chunksize!r}")
    p = _check_path(path)
    _check_common(delimiter, skip_header, missing)

    def good(line: str) -> bool:
        s = line.strip()
        return bool(s) and not s.startswith(comments)

    with open(p, "r", encoding=encoding) as fh:
        for _ in range(skip_header):
            next(fh, None)
        lines = (ln for ln in fh if good(ln))
        while True:
            block = list(itertools.islice(lines, chunksize))
            if not block:
                return
            raw = _read_strings(block, delimiter, 0, usecols, comments, None)
            data = _apply_missing_float(_strings_to_float(raw, missing_tokens), missing, fill_value)
            if data.shape[0]:
                yield data
