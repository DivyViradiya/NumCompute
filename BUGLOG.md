# NumCompute Bug Log

This log records the defects found while building and testing NumCompute, how each was diagnosed,
how it was fixed, and which test now protects against it. It also records the process problems
we ran into while working as a team, because they affected the code on `main`.

**Final state:** 281 unit tests, all passing (`python -m pytest tests -q`), all doctests passing,
`pyflakes` and `pycodestyle` clean.

## 1. How bugs were found

| Method | What it caught |
|---|---|
| Unit tests with edge cases (empty arrays, ties, NaN, extreme `k`, non-contiguous strides) | BUG-01, BUG-03 |
| Comparing results against reference implementations (NumPy, `bisect`, brute-force loops) | confidence in `rank`, `percentile`, `topk`, `binary_search`, histogram, metrics |
| Running tests with warnings treated as errors (`-W error`) | BUG-02 |
| Doctests (examples in docstrings are executed) | BUG-04 |
| Style checks (`pyflakes`, `pycodestyle`) | BUG-06 |
| Teammates running the suite on their own machines | PROC-01 |

Every defect follows the same loop: **a check fails -> reproduce it in isolation -> find the root
cause -> fix the code (not the symptom) -> keep a test that would fail if the bug came back.**

## 2. Summary

| ID | Module | Type | Short description | Status |
|---|---|---|---|---|
| BUG-01 | `sort_search.topk` | Code | Wrong output width for arrays with zero rows | Fixed |
| BUG-02 | `io` | Code | Empty file raised a NumPy warning instead of a clear error | Fixed |
| BUG-03 | `utils.one_hot` | Code | `one_hot([])` crashed with `IndexError` | Fixed |
| BUG-04 | `stats.histogram` | Documentation | Doctest example failed on floating-point output | Fixed |
| BUG-05 | `optim.grad` | Limitation | `relative=True` test failed because of floating-point rounding | Documented |
| BUG-06 | `io`, `optim` | Style | Indentation and over-long line flagged by `pycodestyle` | Fixed |
| TEST-01 | `tests/` | Test error | Several wrong expectations in newly written tests | Fixed |
| PROC-01 | repo / `main` | Process | Batch 4 tests failed: `io.py` and `utils.py` were missing from `main` | Fixed |
| PROC-02 | git | Process | `git pull` refused: untracked local `utils.py` would be overwritten | Resolved |
| PROC-03 | setup | Process | `pip install` blocked on macOS (externally-managed environment) | Resolved |

## 3. Defect details

### BUG-01 - `topk` returned the wrong width for empty leading dimensions
- **Module:** `numcompute/sort_search.py`
- **Symptom:** `topk(np.empty((0, 5)), 2)[0].shape` was `(0, 0)`; it should be `(0, 2)`.
- **Reproduce:** `topk(np.empty((0, 5)), 2)`
- **Root cause:** the shortcut branch for empty input always built arrays of width 0, ignoring the
  clamped `k`. (`k = 0` and `n = 0` really do give width 0, but zero *rows* with `n > 0` does not.)
- **Fix:** build the empty result with width `k` (already clamped to `n`).
- **Regression test:** `tests/test_sort_search.py::test_topk_noncontiguous_and_empty_inputs`
- **Commit / PR:** `11da362` (first commit of `sort_search.py`, merged in PR #1 `b31cc16`; the fix was already in the code when it was first committed)

### BUG-02 - empty CSV file produced a NumPy warning instead of a clear error
- **Module:** `numcompute/io.py`
- **Symptom:** loading an empty file printed `genfromtxt: Empty input file` as a warning and then
  failed later with an unclear message.
- **Root cause:** `numpy.genfromtxt` signals an empty file with a warning, not an exception.
- **Fix:** the reader catches that specific warning and returns an empty array, so every loader
  can raise `ValueError("no data rows found in ...")` (or yield no chunks when streaming).
- **Regression test:** `tests/test_io.py::test_empty_and_ragged_files`
- **Commit / PR:** `228d08b` (first commit of `io.py`; the fix was already in the code when it was first committed)

### BUG-03 - `one_hot([])` crashed
- **Module:** `numcompute/utils.py`
- **Symptom:** `IndexError: arrays used as indices must be of integer (or boolean) type`
- **Reproduce:** `one_hot([])`
- **Root cause:** an empty list converts to a *float* array, and NumPy refuses float arrays as indices
  even when they are empty.
- **Fix:** cast the validated labels to `intp` before indexing (one line).
- **Regression test:** `tests/test_utils.py::test_one_hot_basic_and_explicit_classes`
- **Commit / PR:** `128ef59` (one-line change to `utils.py`, merged in PR #2 `a34a3a4`). The first version of `utils.py`, in `228d08b` and `11da362`, still had the bug

### BUG-04 - doctest failed on floating-point output
- **Module:** `numcompute/stats.py` (`histogram` docstring)
- **Symptom:** the doctest printed `1.6666666666666665` where the docstring said `1.6666666666666667`.
- **Root cause:** the example used bin edges that are not exactly representable, and the expected
  text had been written by hand.
- **Fix:** changed the example to data with exact bin edges (`[0, 1, 1, 2, 4]`, 2 bins -> edges `[0, 2, 4]`).
- **Regression test:** doctest run (`python -m doctest`-style check over all modules).
- **Commit / PR:** `fcbe3ab` (first commit of `stats.py`; the docstring was corrected before it was committed)

### BUG-05 - `grad(..., relative=True)` test failed (floating-point limitation, not a logic error)
- **Module:** `numcompute/optim.py`
- **Symptom:** a test with `f(x) = x0^2 + x1^2` at `x = [1e6, 1]` returned `0` for the second
  component of the gradient.
- **Root cause:** `f` is about `1e12` there, so the spacing between neighbouring floating-point numbers
  is about `1e-4`. A perturbation of `1e-5` in `x1` changes `f` by less than that and is lost to rounding.
  The code was correct; the test asked for something floating point cannot deliver.
- **Resolution:** the limitation is now stated in the `grad` docstring (use `relative=True` or a
  larger `h`), and the test checks the relative-step option on a single-variable function.
- **Regression test:** `tests/test_optim.py::test_grad_per_coordinate_step_and_relative_step`
- **Commit / PR:** `728a478` (first commit of `optim.py` and `tests/test_optim.py`, merged in PR #4 `60ae707`; the docstring note and corrected test were already included)

### BUG-06 - style findings from `pycodestyle`
- **Modules:** `numcompute/io.py` (E122, continuation line indentation), `numcompute/optim.py` (E501, line 130 characters)
- **Root cause:** a patch to `io.py` left the argument list mis-indented; one error message in
  `optim.py` was written on a single line.
- **Fix:** re-indented the call; split the message across lines.
- **Check:** `python -m pycodestyle --max-line-length=120 numcompute benchmark` reports nothing.
- **Commit / PR:** `io.py` re-indented in `128ef59` (PR #2 `a34a3a4`); the `optim.py` line was split before its first commit `728a478`

### TEST-01 - wrong expectations in newly written tests
These were mistakes in the tests, not in the library. Each was diagnosed by reading the failing
assertion against the data. All were corrected before the test files were first committed
(`fcbe3ab` for `tests/test_stats.py`/`tests/test_metrics.py`, `728a478` for `tests/test_io.py` and `tests/test_optim.py`).

| Test | Wrong expectation | Correct behaviour |
|---|---|---|
| `test_io.py::test_missing_fill_and_skip`, `test_iter_csv_chunks_skip_drops_incomplete_rows` | Expected one surviving row when skipping missing values | Reading only columns 0 and 2, the row `40,,7` is complete, so two rows survive |
| `test_optim.py::test_grad_errors` | `np.log` at a non-positive point inside the test emitted a `RuntimeWarning` | Evaluate it inside `np.errstate`, so the test checks the library's `ValueError` instead |
| `test_optim.py` (relative step) | See BUG-05 | |
| `test_stats.py` / early drafts | `.max()` of an array containing NaN returns NaN | Use `np.nanmax` |
| `test_preprocessing.py::test_minmax_default_range_and_custom_range` | Strict `== 1` on float output failed due to 1-ULP drift (`1.0000000000000002`) | Use `np.isclose` instead of exact float equality |

## 4. Evidence that the tests can detect bugs (mutation checks)

To check the tests are strong, the code was deliberately broken in one place at a time; each time
the suite had to fail, and the original code was then restored and re-verified.

| Module | Deliberate change | Tests that failed |
|---|---|---|
| `rank` | shifted the average-rank formula by +0.5 | 11 |
| `sort_search.topk` | resolved boundary ties to the highest index | 15 |
| `sort_search.binary_search` | used `<` instead of `<=` for the right-hand side | 14 |
| `stats.RunningStats` | dropped a factor from the parallel-merge formula | 4 |
| `metrics.f1` | removed the factor 2 in the numerator | 3 |
| `metrics.roc_curve` | stopped merging tied scores | 1 |
| `optim.grad` | forgot the factor 2 in the central difference | 12 |
| `benchmarking.compare` | inverted the speed-up ratio | 1 |
| `utils.one_hot` | removed the BUG-03 fix | 1 |

## 5. Process incidents

### PROC-01 - batch 4 tests failed on `main` (reported by a teammate)
- **Symptom:** `ModuleNotFoundError: No module named 'numcompute.io'`, raised by
  `tests/test_stats.py::test_running_stats_works_with_chunked_csv`. All other tests passed
  (149 passed, 1 failed). Earlier, the batch 4 tests also failed on `utils` imports.
- **Root cause:** the modules were meant to be merged in dependency order (`utils`, `io`,
  `pipeline`, `preprocessing`, then `sort_search`/`rank`, then `stats`/`metrics`), but batches 1-2
  had not been pushed when batches 3-4 were. `stats.py` imports `utils` and `rank`; one test imports `io`.
- **Diagnosis:** `git ls-tree -r origin/main --name-only` listed the files actually on `main`
  and showed `io.py`, `pipeline.py`, `preprocessing.py` and the later modules missing.
- **Fix:** pushed the missing modules in a branch, merged it, then merged the batch 4 branch.
  Final check: `git checkout main && git pull && python -m pytest tests -q`.
- **Prevention:** later batches carry an instruction that they may only be pushed after the previous
  batch shows as merged on GitHub.
- **Commit / PR:** `11da362` (PR #1) added `utils.py` but not `io.py`, `pipeline.py` or `preprocessing.py`; `fcbe3ab` (stats and metrics) was committed on `feat/stats-metrics`; the missing modules reached `main` in `128ef59` (PR #2 `a34a3a4`), and stats/metrics were merged afterwards in PR #3 `b67f454`. The batch 1 and 2 commits `228d08b` and `5f5f390` sit on the branches `feat/utils-io` and `feat/pipeline-preprocessing`, which were never merged into `main`

### PROC-02 - `git pull` refused to overwrite an untracked file
- **Symptom:** `error: The following untracked working tree files would be overwritten by merge: numcompute/utils.py`
- **Root cause:** a local `utils.py` existed that git did not track, while `main` now contained one.
- **Fix:** moved the local copy to a backup folder, pulled, then replaced it with the newest version.
- **Lesson:** add files by name (`git add numcompute/utils.py ...`), not `git add .`, and keep branches
  in sync with `main` before copying files in.

### PROC-03 - `pip install` blocked on macOS
- **Symptom:** `error: externally-managed-environment`
- **Root cause:** Homebrew's Python refuses to install packages system-wide (PEP 668).
- **Fix:** created a virtual environment (`python3 -m venv ~/venv`, `source ~/venv/bin/activate`)
  and installed `numpy` and `pytest` there.

## 6. What we would do differently
- Merge strictly in dependency order and check `origin/main` before starting the next batch.
- Run the whole suite on `main` after every merge, on at least two operating systems (a teammate
  ran it on Windows; the author's checks ran on Linux and macOS).
- Add a continuous-integration job that runs `pytest` on every pull request, so an out-of-order
  merge fails visibly before it lands on `main`.

## 7. How to add a new entry

Copy this block, fill it in, and add a row to the summary table in section 2.

```markdown
### BUG-NN - one-line title
- **Module:** `numcompute/<file>.py`
- **Symptom:** what you saw (paste the error message).
- **Reproduce:** the smallest snippet that fails.
- **Root cause:** why it happened.
- **Fix:** what changed (not just "fixed").
- **Regression test:** `tests/<file>.py::<test_name>`
- **Commit / PR:** hash from `git log --oneline` or the pull request link
```

Find the commit that fixed something with `git log --oneline -- numcompute/<file>.py`.