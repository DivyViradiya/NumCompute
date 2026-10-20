"""Unit tests for numcompute.benchmarking."""
import numpy as np
import pytest

from numcompute.benchmarking import (
    BenchmarkResult,
    compare,
    environment_info,
    format_table,
    time_function,
)


def test_time_function_returns_consistent_statistics():
    r = time_function(sum, range(1000), repeats=4, number=3)
    assert isinstance(r, BenchmarkResult)
    assert r.repeats == 4 and r.number == 3 and r.name == "sum"
    assert 0 < r.best <= r.median and r.best <= r.mean and r.std >= 0


def test_time_function_calls_function_expected_number_of_times():
    calls = []
    time_function(lambda: calls.append(1), repeats=3, number=2, warmup=2)
    assert len(calls) == 2 + 3 * 2


def test_time_function_passes_arguments_and_custom_name():
    seen = []
    time_function(lambda a, b=0: seen.append((a, b)), 5, b=7, repeats=1, warmup=0, name="demo")
    assert seen == [(5, 7)]
    assert time_function(len, [1], repeats=1, name="demo").name == "demo"


def test_time_function_single_repeat_has_zero_std():
    assert time_function(len, [1, 2], repeats=1).std == 0.0


def test_time_function_errors():
    with pytest.raises(TypeError):
        time_function(5)
    with pytest.raises(ValueError):
        time_function(len, [1], repeats=0)
    with pytest.raises(ValueError):
        time_function(len, [1], number=0)
    with pytest.raises(ValueError):
        time_function(len, [1], warmup=-1)
    with pytest.raises(ValueError):
        time_function(len, [1], repeats=True)


def slow_sum(a):
    total = 0.0
    for v in a.tolist():
        total += v
    return total


def test_compare_reports_speedup_and_match():
    rows = compare(slow_sum, np.sum, lambda n, rng: (rng.random(n),), [100, 20000], repeats=2)
    assert [r["size"] for r in rows] == [100, 20000]
    assert all(r["match"] for r in rows)
    assert rows[1]["speedup"] > 1 and rows[1]["loop_s"] > rows[1]["vec_s"]


def test_compare_is_reproducible_with_same_seed():
    seen = []
    compare(lambda a: seen.append(a.sum()), lambda a: None, lambda n, rng: (rng.random(n),), [10], repeats=1, check=False, seed=3)
    first = seen[0]
    seen.clear()
    compare(lambda a: seen.append(a.sum()), lambda a: None, lambda n, rng: (rng.random(n),), [10], repeats=1, check=False, seed=3)
    assert seen[0] == first


def test_compare_detects_mismatching_implementations():
    rows = compare(lambda a: a.sum(), lambda a: a.sum() + 1, lambda n, rng: (rng.random(n),), [10], repeats=1)
    assert rows[0]["match"] is False


def test_compare_check_disabled_and_tuple_results():
    rows = compare(lambda a: a.sum(), lambda a: -1, lambda n, rng: (rng.random(n),), [5], repeats=1, check=False)
    assert rows[0]["match"] is None
    rows = compare(lambda a: (a.min(), a.max()), lambda a: (a.min(), a.max()), lambda n, rng: (rng.random(n),), [5], repeats=1)
    assert rows[0]["match"] is True


def test_compare_empty_sizes():
    with pytest.raises(ValueError):
        compare(len, len, lambda n, rng: ([1],), [])


def test_format_table_is_markdown():
    rows = [{"size": 1000, "loop_s": 0.02, "vec_s": 0.0001, "speedup": 200.0, "match": True}]
    text = format_table(rows, title="demo")
    assert text.startswith("**demo**")
    assert "| Size | Loop | Vectorised | Speed-up | Match |" in text
    assert "1,000" in text and "20.00 ms" in text and "200.0x" in text and "yes" in text


def test_format_table_errors_and_unit_formatting():
    with pytest.raises(ValueError):
        format_table([])
    rows = [{"size": 1, "loop_s": 2.5, "vec_s": 5e-6, "speedup": 5e5, "match": False}]
    text = format_table(rows)
    assert "2.500 s" in text and "5.0 us" in text and "NO" in text


def test_environment_info_keys():
    info = environment_info()
    for key in ("python", "numpy", "platform", "machine", "processor", "cpu_count", "date"):
        assert key in info and isinstance(info[key], str)
    assert info["numpy"] == np.__version__
