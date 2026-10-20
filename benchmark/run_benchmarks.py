"""Loop-vs-vectorised benchmarks for NumCompute.

Run from the repository root::

    python benchmark/run_benchmarks.py            # full run (about a minute)
    python benchmark/run_benchmarks.py --quick    # small sizes, a few seconds

Results are printed and written to ``benchmark/results.md`` together with the
environment, so runs on different machines can be compared. Data come from a
fixed seed, so every run uses identical inputs.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from numcompute import optim, rank as rank_mod, sort_search, stats, utils  # noqa: E402
from numcompute.benchmarking import compare, environment_info, format_table  # noqa: E402
from numcompute.preprocessing import StandardScaler  # noqa: E402


# ----------------------------------------------------------- loop implementations
def standardize_loop(X):
    n, d = X.shape
    out = [[0.0] * d for _ in range(n)]
    for j in range(d):
        s = 0.0
        for i in range(n):
            s += X[i, j]
        mu = s / n
        ss = 0.0
        for i in range(n):
            ss += (X[i, j] - mu) ** 2
        sd = math.sqrt(ss / n) or 1.0
        for i in range(n):
            out[i][j] = (X[i, j] - mu) / sd
    return np.array(out)


def standardize_vec(X):
    return StandardScaler().fit_transform(X)


def softmax_loop(X):
    out = []
    for row in X.tolist():
        m = max(row)
        e = [math.exp(v - m) for v in row]
        t = sum(e)
        out.append([v / t for v in e])
    return np.array(out)


def pairwise_loop(A, B):
    out = [[0.0] * len(B) for _ in range(len(A))]
    for i, a in enumerate(A.tolist()):
        for j, b in enumerate(B.tolist()):
            out[i][j] = math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))
    return np.array(out)


def rank_loop(x):
    xs = x.tolist()
    out = []
    for v in xs:
        less = sum(1 for w in xs if w < v)
        equal = sum(1 for w in xs if w == v)
        out.append(less + (equal + 1) / 2)
    return np.array(out)


def topk_loop(x, k):
    xs = x.tolist()
    order = sorted(range(len(xs)), key=lambda i: (-xs[i], i))
    return np.array([xs[i] for i in order[:k]]), np.array(order[:k])


def histogram_loop(x, bins):
    lo, hi = min(x.tolist()), max(x.tolist())
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in x.tolist():
        counts[min(int((v - lo) / width), bins - 1)] += 1
    return np.array(counts)


def histogram_vec(x, bins):
    return stats.histogram(x, bins=bins)[0]


def welford_loop(x):
    n, mean, m2 = 0, 0.0, 0.0
    for v in x.tolist():
        n += 1
        delta = v - mean
        mean += delta / n
        m2 += delta * (v - mean)
    return mean, m2 / n


def welford_vec(x):
    rs = stats.RunningStats().update(x)
    return float(rs.mean[0]), float(rs.var()[0])


def grad_calls(f, x):
    return optim.grad(f, x)


def grad_batch(f, x):
    return optim.grad(f, x, vectorized=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="use small sizes")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    q = args.quick

    class QuadScalar:
        """Same function; scalar and batched call styles for ``grad``."""

        def __call__(self, v):
            v = np.asarray(v)
            return float(np.sum(v ** 2)) if v.ndim == 1 else np.sum(v ** 2, axis=1)

    quad = QuadScalar()

    cases = [
        ("Standardise columns (10 features)", standardize_loop, standardize_vec,
         lambda n, rng: (rng.normal(size=(n, 10)),), [200, 1000, 5000] if q else [1000, 5000, 20000]),
        ("Row softmax (20 classes)", softmax_loop, utils.softmax,
         lambda n, rng: (rng.normal(size=(n, 20)),), [100, 500, 2000] if q else [1000, 5000, 20000]),
        ("Pairwise Euclidean distances (8 features)", pairwise_loop, utils.pairwise_distances,
         lambda n, rng: (rng.normal(size=(n, 8)), rng.normal(size=(n, 8))), [20, 50, 100] if q else [100, 200, 400]),
        ("Average ranks with ties", rank_loop, rank_mod.rank,
         lambda n, rng: (rng.integers(0, n // 2, size=n).astype(float),), [100, 300, 600] if q else [500, 1000, 2000]),
        ("Top-10 of n values", topk_loop, lambda x, k: sort_search.topk(x, k),
         lambda n, rng: (rng.normal(size=n), 10), [1000, 10000, 50000] if q else [10000, 100000, 300000]),
        ("Histogram (20 bins)", histogram_loop, histogram_vec,
         lambda n, rng: (rng.normal(size=n), 20), [1000, 10000, 50000] if q else [10000, 100000, 500000]),
        ("Streaming mean/variance (Welford loop vs batch)", welford_loop, welford_vec,
         lambda n, rng: (rng.normal(size=n),), [1000, 10000, 50000] if q else [10000, 100000, 500000]),
        ("Finite-difference gradient (one call per point vs one batched call)", grad_calls, grad_batch,
         lambda n, rng: (quad, rng.normal(size=n)), [20, 100, 300] if q else [100, 500, 1500]),
    ]

    env = environment_info()
    out = ["# NumCompute benchmark results", "", "## Environment", ""]
    out += [f"- {k}: {v}" for k, v in env.items()]
    out += ["- seed: 0", f"- repeats: {args.repeats} (best time reported)", f"- mode: {'quick' if q else 'full'}", ""]
    for title, loop_fn, vec_fn, make, sizes in cases:
        print(f"running: {title} ...", flush=True)
        rows = compare(loop_fn, vec_fn, make, sizes, repeats=args.repeats)
        out += [f"## {title}", "", format_table(rows), ""]
    text = "\n".join(out)
    (ROOT / "benchmark" / "results.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print("\nSaved to benchmark/results.md")


if __name__ == "__main__":
    main()
