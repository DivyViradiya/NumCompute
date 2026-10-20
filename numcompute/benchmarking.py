"""Micro-benchmark harness and loop-vs-vectorised comparisons.

Timings use :func:`time.perf_counter`. Results are reproducible when the
random data are generated from a fixed seed (see :func:`compare`) and the
environment is recorded with :func:`environment_info`.
"""
from __future__ import annotations

import datetime
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["BenchmarkResult", "time_function", "compare", "format_table", "environment_info"]


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing summary in seconds **per call**.

    Attributes
    ----------
    name : str
        Label of the benchmarked function.
    repeats : int
        Number of timed repetitions.
    number : int
        Calls per repetition (the reported times are divided by this).
    best, mean, median, std : float
        Statistics over the repetitions (``std`` is 0 for a single repeat).
    """

    name: str
    repeats: int
    number: int
    best: float
    mean: float
    median: float
    std: float


def _check_count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    return int(value)


def time_function(
    func: Callable,
    *args: Any,
    repeats: int = 5,
    number: int = 1,
    warmup: int = 1,
    name: Optional[str] = None,
    **kwargs: Any,
) -> BenchmarkResult:
    """Time ``func(*args, **kwargs)``.

    Parameters
    ----------
    func : callable
        Function to benchmark.
    *args, **kwargs
        Passed to ``func`` on every call.
    repeats : int, optional
        Timed repetitions (>= 1). Default 5.
    number : int, optional
        Calls inside each repetition (>= 1). Default 1.
    warmup : int, optional
        Untimed calls beforehand (>= 0) to fill caches. Default 1.
    name : str, optional
        Label; defaults to ``func.__name__``.

    Returns
    -------
    BenchmarkResult
        Seconds per call. Use ``best`` for the least noisy comparison.

    Raises
    ------
    TypeError
        If ``func`` is not callable.
    ValueError
        If ``repeats``/``number`` is not a positive int or ``warmup`` < 0.

    Notes
    -----
    Time O((warmup + repeats * number) * cost(func)), space O(repeats).

    Examples
    --------
    >>> time_function(sum, range(100), repeats=2).repeats
    2
    """
    if not callable(func):
        raise TypeError("func must be callable")
    repeats, number = _check_count(repeats, "repeats"), _check_count(number, "number")
    if isinstance(warmup, bool) or not isinstance(warmup, (int, np.integer)) or warmup < 0:
        raise ValueError(f"warmup must be an int >= 0, got {warmup!r}")
    for _ in range(int(warmup)):
        func(*args, **kwargs)
    samples: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(number):
            func(*args, **kwargs)
        samples.append((time.perf_counter() - start) / number)
    return BenchmarkResult(
        name=name or getattr(func, "__name__", repr(func)),
        repeats=repeats,
        number=number,
        best=min(samples),
        mean=statistics.fmean(samples),
        median=statistics.median(samples),
        std=statistics.pstdev(samples) if len(samples) > 1 else 0.0,
    )


def _same(a: Any, b: Any, rtol: float, atol: float) -> bool:
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y, rtol, atol) for x, y in zip(a, b))
    try:
        xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return xa.shape == xb.shape and bool(np.allclose(xa, xb, rtol=rtol, atol=atol, equal_nan=True))
    except (TypeError, ValueError):
        return bool(a == b)


def compare(
    loop_func: Callable,
    vec_func: Callable,
    make_args: Callable,
    sizes: Sequence[int],
    repeats: int = 3,
    number: int = 1,
    seed: int = 0,
    check: bool = True,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> List[Dict[str, Any]]:
    """Benchmark a Python-loop implementation against a vectorised one.

    Parameters
    ----------
    loop_func, vec_func : callable
        Two implementations taking the same arguments.
    make_args : callable
        ``make_args(size, rng) -> tuple`` building the arguments for one
        problem size from a seeded ``numpy.random.Generator``.
    sizes : sequence of int
        Problem sizes to test.
    repeats, number : int, optional
        Passed to :func:`time_function`.
    seed : int, optional
        Seed for the generator, so every run uses identical data.
    check : bool, optional
        Verify both functions return the same result (within ``rtol``/``atol``).
    rtol, atol : float, optional
        Tolerances for the agreement check.

    Returns
    -------
    list of dict
        One row per size with keys ``size``, ``loop_s``, ``vec_s`` (best
        seconds per call), ``speedup`` (``loop_s / vec_s``) and ``match``
        (True/False, or None if ``check=False``).

    Raises
    ------
    ValueError
        If ``sizes`` is empty or an argument is invalid.

    Notes
    -----
    Disagreement between the two implementations is reported in ``match``
    rather than raised. Time O(sum over sizes of (cost(loop) + cost(vec)) * repeats).

    Examples
    --------
    >>> rows = compare(lambda a: sum(a), lambda a: a.sum(), lambda n, rng: (rng.random(n),), [10])
    >>> rows[0]["match"]
    True
    """
    if len(sizes) == 0:
        raise ValueError("sizes must not be empty")
    rows: List[Dict[str, Any]] = []
    for size in sizes:
        args = make_args(int(size), np.random.default_rng(seed))
        loop_t = time_function(loop_func, *args, repeats=repeats, number=number)
        vec_t = time_function(vec_func, *args, repeats=repeats, number=number)
        match = _same(loop_func(*args), vec_func(*args), rtol, atol) if check else None
        rows.append(
            {
                "size": int(size),
                "loop_s": loop_t.best,
                "vec_s": vec_t.best,
                "speedup": loop_t.best / vec_t.best if vec_t.best > 0 else float("inf"),
                "match": match,
            }
        )
    return rows


def _fmt_seconds(s: float) -> str:
    if s >= 1:
        return f"{s:.3f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.2f} ms"
    return f"{s * 1e6:.1f} us"


def format_table(rows: Sequence[Dict[str, Any]], title: Optional[str] = None) -> str:
    """Render :func:`compare` rows as a Markdown table.

    Parameters
    ----------
    rows : sequence of dict
        Output of :func:`compare`.
    title : str, optional
        Heading printed above the table.

    Returns
    -------
    str
        Markdown with columns Size, Loop, Vectorised, Speed-up, Match.

    Raises
    ------
    ValueError
        If ``rows`` is empty.

    Notes
    -----
    Time O(len(rows)), space O(len(rows)).
    """
    if len(rows) == 0:
        raise ValueError("rows must not be empty")
    lines = []
    if title:
        lines += [f"**{title}**", ""]
    lines += ["| Size | Loop | Vectorised | Speed-up | Match |", "|---:|---:|---:|---:|:---:|"]
    for r in rows:
        match = "-" if r["match"] is None else ("yes" if r["match"] else "NO")
        lines.append(
            f"| {r['size']:,} | {_fmt_seconds(r['loop_s'])} | {_fmt_seconds(r['vec_s'])} "
            f"| {r['speedup']:.1f}x | {match} |"
        )
    return "\n".join(lines)


def environment_info() -> Dict[str, str]:
    """Describe the machine and software, for reproducible benchmark notes.

    Returns
    -------
    dict of str
        Keys ``python``, ``numpy``, ``platform``, ``machine``, ``processor``,
        ``cpu_count`` and ``date`` (UTC, ISO format).

    Notes
    -----
    Time O(1), space O(1).
    """
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": str(os.cpu_count()),
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
