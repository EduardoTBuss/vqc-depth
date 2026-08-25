"""The three UCI datasets: identity, shape, class balance, missing values.

Why the ``network`` marker
--------------------------
These tests read ``data/cache/*.npz`` and, if it is missing, would have to
download from UCI.  This suite is the GATE of the Colab grid: an unreachable
UCI mirror must never be able to stop 120 runs from starting, so they are
skipped when the cache is absent (see the ``uci_available`` fixture in
``conftest.py``).  With the cache present -- the normal case -- they DO run.

Everything that does not need real data (error paths, cache round-trip,
identity guards) is in the second half of the file and always runs.

Numbers below were verified on 2026-08-25 against the UCI download.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from vqcdepth import config, data

# dataset -> (n, feature names, count of class 0, count of class 1,
#             {feature: n_nan})
EXPECTED = {
    "banknote": (
        1372, ["variance", "skewness", "curtosis", "entropy"], 762, 610,
        {"variance": 0, "skewness": 0, "curtosis": 0, "entropy": 0},
    ),
    "blood": (
        748, ["Recency", "Frequency", "Monetary", "Time"], 570, 178,
        {"Recency": 0, "Frequency": 0, "Monetary": 0, "Time": 0},
    ),
    "mammographic": (
        961, ["Age", "Shape", "Margin", "Density"], 516, 445,
        {"Age": 5, "Shape": 31, "Margin": 48, "Density": 76},
    ),
}


# ======================================================================
# 1. the real datasets
# ======================================================================
@pytest.mark.network
@pytest.mark.parametrize("name", list(EXPECTED))
def test_dataset_shape_dtypes_and_feature_names(uci_available, name):
    n, names, _, _, _ = EXPECTED[name]
    X, y, got_names = data.load_dataset(name)

    assert X.shape == (n, 4), f"{name}: shape {X.shape}, expected ({n}, 4)"
    assert X.shape[1] == config.N_QUBITS == 4, "4 native features, one per qubit"
    assert X.dtype == np.float64 and y.dtype == np.int64
    assert y.shape == (n,)
    assert got_names == names, f"{name}: features {got_names}, expected {names}"
    assert not np.isinf(X).any()


@pytest.mark.network
@pytest.mark.parametrize("name", list(EXPECTED))
def test_dataset_class_distribution_is_the_published_one(uci_available, name):
    n, _, c0, c1, _ = EXPECTED[name]
    _, y, _ = data.load_dataset(name)

    got0, got1 = int((y == 0).sum()), int((y == 1).sum())
    assert set(np.unique(y)) == {0, 1}, f"{name}: target is not binary"
    assert (got0, got1) == (c0, c1), (
        f"{name}: class counts {(got0, got1)} != published {(c0, c1)} "
        f"({100 * got0 / n:.2f}% / {100 * got1 / n:.2f}% vs "
        f"{100 * c0 / n:.2f}% / {100 * c1 / n:.2f}%). Wrong ucimlrepo id, a "
        f"re-indexed dataset, or a stale cache."
    )


@pytest.mark.network
@pytest.mark.parametrize("name", list(EXPECTED))
def test_missing_values_are_delivered_as_NaN_and_counted_exactly(uci_available, name):
    """The missing data must SURVIVE the load: imputing here would leak the
    global median into val/test.  Only ``splits.make_split`` may impute."""
    _, names, _, _, expected_nan = EXPECTED[name]
    X, _, got_names = data.load_dataset(name)

    got_nan = {got_names[j]: int(np.isnan(X[:, j]).sum()) for j in range(4)}
    assert got_nan == expected_nan, (
        f"{name}: NaN per feature {got_nan} != expected {expected_nan}"
    )
    assert sum(got_nan.values()) == sum(expected_nan.values())
    assert not np.isnan(X).all(axis=1).any(), (
        f"{name}: a row is 100% NaN -- column-wise median imputation cannot "
        f"recover it"
    )


@pytest.mark.network
def test_mammographic_is_the_only_dataset_with_missing_values(uci_available):
    totals = {n: int(np.isnan(data.load_dataset(n)[0]).sum()) for n in EXPECTED}
    assert totals == {"banknote": 0, "blood": 0, "mammographic": 160}, totals


@pytest.mark.network
def test_birads_is_excluded_from_mammographic(uci_available):
    """BI-RADS is the radiologist's own assessment; keeping it would leak the
    clinical judgement into the label."""
    _, _, names = data.load_dataset("mammographic")
    assert "BI-RADS" not in names
    assert len(names) == 4


@pytest.mark.network
@pytest.mark.parametrize("name", list(EXPECTED))
def test_loading_twice_returns_independent_arrays(uci_available, name):
    """A caller mutating X must not corrupt the next caller's copy."""
    X1, y1, _ = data.load_dataset(name)
    X1[0, 0] = 12345.0
    y1[0] = 1 - y1[0]
    X2, y2, _ = data.load_dataset(name)
    assert X2[0, 0] != 12345.0, f"{name}: load_dataset hands out shared state"


# ======================================================================
# 2. error paths and cache handling (no network, always run)
# ======================================================================
def test_unknown_dataset_is_rejected_by_name():
    with pytest.raises(ValueError):
        data.load_dataset("iris")


def _write_fake_cache(directory, name, X, y, names):
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / f"{name}.npz",
        X=X, y=y,
        feature_names=np.asarray(names, dtype="<U32"),
        uci_id=np.asarray(config.UCI_IDS[name], dtype=np.int64),
        uci_name=np.asarray("whatever", dtype="<U64"),
    )


def test_a_cache_with_the_wrong_number_of_rows_is_refused(tmp_path, monkeypatch):
    """The identity guard: a re-indexed or hand-edited cache must ABORT, not
    silently feed the experiment a different dataset."""
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    _write_fake_cache(tmp_path, "banknote",
                      np.zeros((10, 4)), np.array([0, 1] * 5, dtype=np.int64),
                      ["variance", "skewness", "curtosis", "entropy"])
    with pytest.raises(AssertionError, match="esperava n="):
        data.load_dataset("banknote")


def test_a_cache_with_the_wrong_feature_names_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    _write_fake_cache(tmp_path, "blood",
                      np.zeros((748, 4)), np.array([0, 1] * 374, dtype=np.int64),
                      ["a", "b", "c", "d"])
    with pytest.raises(AssertionError):
        data.load_dataset("blood")


def test_unexpected_missing_values_are_refused(tmp_path, monkeypatch):
    """banknote is declared complete; a NaN there means the file is not banknote."""
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    X = np.zeros((1372, 4))
    X[3, 2] = np.nan
    _write_fake_cache(tmp_path, "banknote", X,
                      np.array([0, 1] * 686, dtype=np.int64),
                      ["variance", "skewness", "curtosis", "entropy"])
    with pytest.raises(AssertionError, match="missing"):
        data.load_dataset("banknote")


def test_a_single_class_target_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    _write_fake_cache(tmp_path, "banknote",
                      np.zeros((1372, 4)), np.zeros(1372, dtype=np.int64),
                      ["variance", "skewness", "curtosis", "entropy"])
    with pytest.raises(AssertionError, match="binário|binario|nao binario"):
        data.load_dataset("banknote")


def test_a_failing_download_raises_a_readable_error_not_a_stack_trace(
        tmp_path, monkeypatch):
    """Offline + empty cache: the message must say what to do."""
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "empty")

    fake = types.ModuleType("ucimlrepo")

    def boom(*args, **kwargs):
        raise ConnectionError("no route to host")

    fake.fetch_ucirepo = boom
    monkeypatch.setitem(sys.modules, "ucimlrepo", fake)

    with pytest.raises(RuntimeError, match="cache"):
        data.load_dataset("banknote")


def test_cache_round_trip_preserves_values_and_dtypes(tmp_path, monkeypatch):
    """What is written by ``_save_cache`` is what ``load_dataset`` reads back."""
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1372, 4))
    y = (rng.random(1372) < 0.5).astype(np.int64)
    names = ["variance", "skewness", "curtosis", "entropy"]

    data._save_cache("banknote", X, y, names)
    assert (tmp_path / "banknote.npz").exists()
    assert not list(tmp_path.glob("*.tmp")), "atomic write left a temporary file"

    X2, y2, names2 = data.load_dataset("banknote")
    np.testing.assert_array_equal(X2, X)
    np.testing.assert_array_equal(y2, y)
    assert names2 == names and X2.dtype == np.float64 and y2.dtype == np.int64


def test_every_dataset_of_the_grid_has_a_declared_uci_id():
    assert set(config.UCI_IDS) == set(config.DATASETS)
    assert set(data._SPEC) == set(config.DATASETS)
    assert len(set(config.UCI_IDS.values())) == len(config.DATASETS), (
        "two datasets share the same ucimlrepo id"
    )
    for name, spec in data._SPEC.items():
        assert len(spec["features"]) == config.N_QUBITS, (
            f"{name}: {len(spec['features'])} features but {config.N_QUBITS} qubits"
        )
