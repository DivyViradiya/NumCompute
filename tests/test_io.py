"""Unit tests for numcompute.io."""
import numpy as np
import pytest

from numcompute.io import iter_csv_chunks, load_csv, load_mixed_csv, read_header


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "age,city,score\n"
        "25,Paris,3.5\n"
        ",London,\n"
        "40,,7\n"
        "# a comment line\n"
        "\n"
        "31,Rome,NA\n"
    )
    return path


def test_read_header(csv_file):
    assert read_header(csv_file) == ["age", "city", "score"]


def test_load_numeric_columns_keeps_missing_as_nan(csv_file):
    X = load_csv(csv_file, skip_header=1, usecols=[0, 2])
    assert X.shape == (4, 2) and X.dtype == float
    assert np.isnan(X[1]).all() and np.isnan(X[3, 1]) and X[0].tolist() == [25.0, 3.5]


def test_missing_fill_and_skip(csv_file):
    filled = load_csv(csv_file, skip_header=1, usecols=[0, 2], missing="fill", fill_value=-1)
    assert not np.isnan(filled).any() and filled[1].tolist() == [-1.0, -1.0]
    skipped = load_csv(csv_file, skip_header=1, usecols=[0, 2], missing="skip")
    # rows with a missing age or score ("," and "NA") are dropped; 25,3.5 and 40,7 remain
    assert skipped.tolist() == [[25.0, 3.5], [40.0, 7.0]]


def test_custom_delimiters(tmp_path):
    tsv, semi = tmp_path / "a.tsv", tmp_path / "b.csv"
    tsv.write_text("1\t2\n3\t4\n")
    semi.write_text("1;2\n3;4\n")
    assert load_csv(tsv, delimiter="\t").tolist() == [[1, 2], [3, 4]]
    assert load_csv(semi, delimiter=";").tolist() == [[1, 2], [3, 4]]


def test_single_row_and_single_column_stay_2d(tmp_path):
    row, col = tmp_path / "r.csv", tmp_path / "c.csv"
    row.write_text("1,2,3\n")
    col.write_text("1\n2\n3\n")
    assert load_csv(row).shape == (1, 3) and load_csv(col).shape == (3, 1)


def test_integer_dtype_and_its_missing_value_guard(tmp_path, csv_file):
    p = tmp_path / "ints.csv"
    p.write_text("1,2\n3,4\n")
    out = load_csv(p, dtype=int)
    assert out.dtype.kind == "i" and out.tolist() == [[1, 2], [3, 4]]
    with pytest.raises(ValueError):
        load_csv(csv_file, skip_header=1, usecols=[0, 2], dtype=int)
    filled = load_csv(csv_file, skip_header=1, usecols=[0, 2], dtype=int, missing="fill", fill_value=0)
    assert filled.dtype.kind == "i"


def test_string_dtype_handles_missing(csv_file):
    S = load_csv(csv_file, skip_header=1, dtype=str)
    assert S.shape == (4, 3) and S[0].tolist() == ["25", "Paris", "3.5"]
    assert S[1, 0] == "nan" and S[2, 1] == "nan"
    filled = load_csv(csv_file, skip_header=1, dtype=str, missing="fill", fill_value="?")
    assert filled[1].tolist() == ["?", "London", "?"]
    assert load_csv(csv_file, skip_header=1, dtype=str, missing="skip").shape == (1, 3)


def test_load_mixed_csv_splits_numeric_and_categorical(csv_file):
    num, cat, num_names, cat_names = load_mixed_csv(csv_file, categorical_cols=["city"])
    assert num.shape == (4, 2) and cat.shape == (4, 1)
    assert num_names == ["age", "score"] and cat_names == ["city"]
    assert cat.ravel().tolist() == ["Paris", "London", "missing", "Rome"]
    assert np.isnan(num[1]).all()
    num2, cat2, _, _ = load_mixed_csv(csv_file, categorical_cols=[1])
    assert np.array_equal(num, num2, equal_nan=True) and np.array_equal(cat, cat2)


def test_iter_csv_chunks_matches_full_load(csv_file):
    chunks = list(iter_csv_chunks(csv_file, chunksize=3, skip_header=1, usecols=[0, 2]))
    assert [c.shape[0] for c in chunks] == [3, 1]
    full = load_csv(csv_file, skip_header=1, usecols=[0, 2])
    assert np.array_equal(np.vstack(chunks), full, equal_nan=True)


def test_iter_csv_chunks_large_file_and_skip(tmp_path):
    data = np.arange(200.0).reshape(100, 2)
    p = tmp_path / "big.csv"
    np.savetxt(p, data, delimiter=",")
    sizes = [c.shape[0] for c in iter_csv_chunks(p, chunksize=30)]
    assert sizes == [30, 30, 30, 10]
    assert np.array_equal(np.vstack(list(iter_csv_chunks(p, chunksize=7))), data)


def test_iter_csv_chunks_skip_drops_incomplete_rows(csv_file):
    chunks = list(iter_csv_chunks(csv_file, chunksize=2, skip_header=1, usecols=[0, 2], missing="skip"))
    assert np.vstack(chunks).tolist() == [[25.0, 3.5], [40.0, 7.0]]


def test_errors(tmp_path, csv_file):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv")
    with pytest.raises(TypeError):
        load_csv(123)
    with pytest.raises(ValueError):
        load_csv(csv_file, missing="drop")
    with pytest.raises(ValueError):
        load_csv(csv_file, delimiter="")
    with pytest.raises(ValueError):
        load_csv(csv_file, skip_header=-1)
    with pytest.raises(ValueError):
        load_csv(csv_file, skip_header=1)  # text column in a numeric load
    with pytest.raises(ValueError):
        load_mixed_csv(csv_file, categorical_cols=["nope"])
    with pytest.raises(ValueError):
        load_mixed_csv(csv_file, categorical_cols=[9])
    with pytest.raises(ValueError):
        list(iter_csv_chunks(csv_file, chunksize=0))


def test_empty_and_ragged_files(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ValueError):
        load_csv(empty)
    with pytest.raises(ValueError):
        read_header(empty)
    assert list(iter_csv_chunks(empty, chunksize=5)) == []
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("1,2,3\n4,5\n")
    with pytest.raises(ValueError):
        load_csv(ragged)
