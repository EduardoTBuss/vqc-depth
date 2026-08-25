"""Splits: determinism, pairing across models, and ABSENCE OF LEAKAGE.

This is the file the paper leans on hardest.

* The paired Wilcoxon is only legitimate if the 4 models of a given
  ``(dataset, seed)`` saw literally the same rows.  ``make_split`` does not even
  take a model argument, and these tests keep it that way -- including across
  PROCESSES with different ``PYTHONHASHSEED``, which is exactly where a
  ``hash()``-based implementation would silently start drifting.
* The scaler and the imputer must be fitted on the training partition ONLY.
  A reviewer's first question is "did you scale before splitting?"; the answer
  has to be a test, not a paragraph.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import helpers
from vqcdepth import config, splits

PI = config.FEATURE_RANGE[1]
CHILD = Path(__file__).resolve().parent / "_subprocess_split_child.py"


@pytest.fixture
def fake_dataset(monkeypatch):
    """Inject a synthetic raw dataset into ``splits`` -- no cache, no network.

    Returns ``(X, y, feature_names)`` so the test can recompute, by hand, what
    the split is supposed to produce.
    """
    raw = helpers.synthetic_raw(n=140, seed=0, n_nan=12)
    monkeypatch.setattr(splits, "load_dataset", lambda name: raw)
    return raw


# ======================================================================
# 1. determinism and pairing
# ======================================================================
def test_split_is_a_pure_function_of_dataset_and_seed(fake_dataset):
    a = splits.make_split("banknote", 3)
    b = splits.make_split("banknote", 3)
    assert a.hashes == b.hashes
    np.testing.assert_array_equal(a.idx_train, b.idx_train)
    np.testing.assert_array_equal(a.X_train, b.X_train)
    np.testing.assert_array_equal(a.X_test, b.X_test)


def test_all_four_models_of_a_seed_get_byte_identical_splits(fake_dataset):
    """THE test behind the paired Wilcoxon.

    ``make_split`` takes no model, so the guarantee is structural; this pins it
    through the object the runner actually consumes, ``RunConfig``.
    """
    for seed in (0, 5, 9):
        per_model = {}
        for model in config.MODELS:
            cfg = config.RunConfig(dataset="banknote", model=model, seed=seed)
            sp = splits.make_split(cfg.dataset, cfg.seed)
            per_model[model] = (sp.hashes,
                                sp.X_train.tobytes(),
                                sp.y_test.tobytes())
        ref = per_model[config.MODELS[0]]
        for model, got in per_model.items():
            assert got[0] == ref[0], (
                f"seed {seed}: model {model} got split hashes {got[0]} but "
                f"{config.MODELS[0]} got {ref[0]}. The Wilcoxon pairing of the "
                f"paper is INVALID until this is fixed."
            )
            assert got[1] == ref[1], f"seed {seed}: X_train differs for {model}"
            assert got[2] == ref[2], f"seed {seed}: y_test differs for {model}"


def test_different_seeds_produce_different_splits(fake_dataset):
    h0 = splits.make_split("banknote", 0).hashes
    h1 = splits.make_split("banknote", 1).hashes
    for key in ("train", "val", "test", "all"):
        assert h0[key] != h1[key], f"seed 0 and seed 1 share the {key} partition"


def test_partitions_are_disjoint_and_cover_the_dataset(fake_dataset):
    X, y, _ = fake_dataset
    sp = splits.make_split("banknote", 4)
    allidx = np.concatenate([sp.idx_train, sp.idx_val, sp.idx_test])
    np.testing.assert_array_equal(np.sort(allidx), np.arange(len(y)))
    assert len(set(allidx.tolist())) == len(y), "an index appears in two partitions"


def test_labels_follow_their_indices(fake_dataset):
    X, y, _ = fake_dataset
    sp = splits.make_split("banknote", 2)
    np.testing.assert_array_equal(sp.y_train, y[sp.idx_train])
    np.testing.assert_array_equal(sp.y_val, y[sp.idx_val])
    np.testing.assert_array_equal(sp.y_test, y[sp.idx_test])


def test_split_is_stratified_and_close_to_70_15_15(fake_dataset):
    X, y, _ = fake_dataset
    n = len(y)
    sp = splits.make_split("banknote", 6)
    sizes = np.array([sp.idx_train.size, sp.idx_val.size, sp.idx_test.size])
    np.testing.assert_allclose(sizes / n, [0.70, 0.15, 0.15], atol=0.02)

    base = float(np.mean(y))
    for part, yy in (("train", sp.y_train), ("val", sp.y_val), ("test", sp.y_test)):
        assert abs(float(np.mean(yy)) - base) < 0.10, (
            f"{part}: positive rate {np.mean(yy):.3f} vs base {base:.3f} -- "
            f"stratification is off"
        )


def test_indices_are_shuffled_not_sorted(fake_dataset):
    """Sorted indices would group the classes; a no-shuffle loop would then see
    single-class minibatches on banknote/blood, which are stored class-ordered."""
    sp = splits.make_split("banknote", 1)
    assert not np.array_equal(sp.idx_train, np.sort(sp.idx_train))


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError):
        splits.make_split("iris", 0)


# ======================================================================
# 2. index_hash contract
# ======================================================================
def test_index_hash_is_dtype_and_shape_invariant_but_order_sensitive():
    idx = np.array([5, 1, 9, 3], dtype=np.int32)
    assert splits.index_hash(idx) == splits.index_hash(idx.astype(np.int64))
    assert splits.index_hash(idx) == splits.index_hash(list(idx))
    assert splits.index_hash(idx) != splits.index_hash(np.sort(idx))
    with pytest.raises(ValueError):
        splits.index_hash(np.zeros((2, 2), dtype=np.int64))


def test_split_hashes_distinguish_swapped_partitions():
    a = np.array([0, 1, 2]); b = np.array([3, 4]); c = np.array([5, 6])
    assert splits.split_hashes(a, b, c)["all"] != splits.split_hashes(a, c, b)["all"]


# ======================================================================
# 3. cross-process stability
# ======================================================================
def _run_child(mode: str, dataset: str, seed: int, hashseed: str) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, str(CHILD), mode, dataset, str(seed)],
        capture_output=True, text=True, env=env, cwd=str(CHILD.parent.parent),
    )
    assert proc.returncode == 0, (
        f"child failed (PYTHONHASHSEED={hashseed}):\n{proc.stdout}\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_split_hashes_survive_a_change_of_PYTHONHASHSEED(fake_dataset):
    """Regression against seeding from the builtin ``hash()``.

    ``hash()`` of a str is salted per process, so an implementation based on it
    would give a DIFFERENT split every time the Colab runtime restarts -- and
    the 120 runs would no longer be paired.
    """
    a = _run_child("synthetic", "banknote", 0, "0")
    b = _run_child("synthetic", "banknote", 0, "12345")
    assert a["hashes"] == b["hashes"], (
        f"split changed with PYTHONHASHSEED: {a['hashes']} vs {b['hashes']}"
    )
    assert a["X_train_sha"] == b["X_train_sha"], (
        "indices matched but the preprocessed X_train did not -- the scaler or "
        "the imputer is picking up process-dependent state"
    )
    assert a["pythonhashseed"] != b["pythonhashseed"], "the children saw the same env"


@pytest.mark.network
def test_real_dataset_split_hashes_survive_a_change_of_PYTHONHASHSEED(uci_available):
    a = _run_child("real", "banknote", 0, "0")
    b = _run_child("real", "banknote", 0, "12345")
    parent = splits.make_split("banknote", 0)
    assert a["hashes"] == b["hashes"] == parent.hashes, (
        f"banknote/seed0 hashes differ between processes: child0={a['hashes']}, "
        f"child1={b['hashes']}, parent={parent.hashes}"
    )


# ======================================================================
# 4. NO LEAKAGE -- the reviewer-proof part
# ======================================================================
def test_train_partition_hits_exactly_0_and_pi_on_every_feature(fake_dataset):
    """The min-max fit happened on the TRAIN rows, so train spans [0, pi] exactly.

    If the scaler had been fitted on the full dataset, the training minimum
    would (almost surely) be > 0 and the maximum < pi.
    """
    sp = splits.make_split("banknote", 0)
    np.testing.assert_allclose(sp.X_train.min(axis=0), 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sp.X_train.max(axis=0), PI, rtol=0, atol=1e-12)


def test_validation_and_test_are_allowed_to_leave_the_encoding_range(fake_dataset):
    """Not clipping is a deliberate choice: clipping would pretend the training
    set had seen an extreme it never saw.  At least one held-out value must land
    outside [0, pi], otherwise the previous test would pass trivially."""
    outside = 0
    for seed in range(4):
        sp = splits.make_split("banknote", seed)
        for arr in (sp.X_val, sp.X_test):
            outside += int(((arr < 0.0) | (arr > PI)).sum())
    assert outside > 0, (
        "no held-out value fell outside [0, pi] -- either the data changed or "
        "someone started clipping the transform"
    )


def test_poisoning_validation_and_test_cannot_change_the_training_partition(
        monkeypatch):
    """THE anti-leakage test.

    Replace every held-out row with absurd values; the training partition and
    all four fingerprints must come out bit-identical.  Any statistic borrowed
    from val/test -- a global mean, a global median, min/max over everything --
    would move X_train and fail here.
    """
    X, y, names = helpers.synthetic_raw(n=140, seed=0, n_nan=12)
    monkeypatch.setattr(splits, "load_dataset", lambda name: (X, y, names))
    ref = splits.make_split("banknote", 0)

    poisoned = X.copy()
    poisoned[ref.idx_val] = 1.0e6
    poisoned[ref.idx_test] = -1.0e6
    monkeypatch.setattr(splits, "load_dataset", lambda name: (poisoned, y, names))
    got = splits.make_split("banknote", 0)

    assert got.hashes == ref.hashes, "poisoning changed the index assignment"
    np.testing.assert_array_equal(
        got.X_train, ref.X_train,
        err_msg=("LEAKAGE: X_train changed after corrupting only val/test rows. "
                 "The scaler and/or the imputer are seeing held-out data."),
    )
    np.testing.assert_array_equal(got.y_train, ref.y_train)
    # sanity: the poisoning mechanism is real -- held-out rows DID change
    assert not np.array_equal(got.X_test, ref.X_test)


def test_poisoning_the_training_partition_does_change_the_held_out_ones(monkeypatch):
    """Control for the test above: the transform really is fitted on train."""
    X, y, names = helpers.synthetic_raw(n=140, seed=0, n_nan=12)
    monkeypatch.setattr(splits, "load_dataset", lambda name: (X, y, names))
    ref = splits.make_split("banknote", 0)

    poisoned = X.copy()
    poisoned[ref.idx_train] *= 3.0
    monkeypatch.setattr(splits, "load_dataset", lambda name: (poisoned, y, names))
    got = splits.make_split("banknote", 0)

    assert not np.allclose(got.X_val, ref.X_val), (
        "changing the TRAINING rows left X_val untouched: the scaler is not "
        "being fitted on the training partition at all"
    )


def test_preprocessing_matches_an_independent_reference_implementation(fake_dataset):
    """Golden model: impute with the train median, min-max with the train bounds.

    Written from scratch in ``helpers.reference_preprocess`` (no sklearn), so a
    change of sklearn version cannot make both sides wrong the same way.
    """
    X, y, _ = fake_dataset
    sp = splits.make_split("banknote", 0)

    tr_raw = X[sp.idx_train]
    exp_tr, (exp_va, exp_te), med = helpers.reference_preprocess(
        tr_raw, [X[sp.idx_val], X[sp.idx_test]])

    np.testing.assert_allclose(sp.X_train, exp_tr, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sp.X_val, exp_va, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sp.X_test, exp_te, rtol=1e-12, atol=1e-12)

    # and the median really is the TRAIN median, not the global one
    global_med = np.nanmedian(X, axis=0)
    assert not np.allclose(med, global_med), (
        "train median and global median coincide on this fixture, so the test "
        "cannot tell them apart -- change the fixture"
    )


def test_imputed_cells_carry_the_train_median_and_not_the_global_median(fake_dataset):
    """Pin the imputed VALUE, cell by cell, in the scaled space."""
    X, y, _ = fake_dataset
    sp = splits.make_split("banknote", 0)

    tr_raw = X[sp.idx_train]
    med_train = np.nanmedian(tr_raw, axis=0)
    med_global = np.nanmedian(X, axis=0)
    lo, hi = np.where(np.isnan(tr_raw), med_train, tr_raw).min(axis=0), \
        np.where(np.isnan(tr_raw), med_train, tr_raw).max(axis=0)

    for part, idx, arr in (("train", sp.idx_train, sp.X_train),
                           ("val", sp.idx_val, sp.X_val),
                           ("test", sp.idx_test, sp.X_test)):
        rows, cols = np.where(np.isnan(X[idx]))
        assert len(rows) == sp.n_imputed[part], (
            f"{part}: n_imputed says {sp.n_imputed[part]} but there are "
            f"{len(rows)} NaN cells in the raw rows of that partition"
        )
        for r, c in zip(rows, cols):
            expected = (med_train[c] - lo[c]) / (hi[c] - lo[c]) * PI
            assert arr[r, c] == pytest.approx(expected, rel=0, abs=1e-12), (
                f"{part}[{r},{c}] was imputed with {arr[r, c]:.6f}, expected the "
                f"scaled TRAIN median {expected:.6f} (scaled GLOBAL median would "
                f"be {(med_global[c] - lo[c]) / (hi[c] - lo[c]) * PI:.6f})"
            )


def test_no_nan_or_inf_survives_the_split(fake_dataset):
    sp = splits.make_split("banknote", 0)
    for part, arr in (("train", sp.X_train), ("val", sp.X_val), ("test", sp.X_test)):
        assert np.isfinite(arr).all(), f"X_{part} still carries NaN/inf"
    assert sp.n_imputed["total"] == (sp.n_imputed["train"] + sp.n_imputed["val"]
                                     + sp.n_imputed["test"])


def test_split_carries_no_extra_rows_and_the_right_dtypes(fake_dataset):
    sp = splits.make_split("banknote", 0)
    assert sp.X_train.shape == (sp.idx_train.size, config.N_QUBITS)
    assert sp.X_train.dtype == np.float64 and sp.y_train.dtype == np.int64
    assert set(np.unique(sp.y_train)) <= {0, 1}


# ======================================================================
# 5. golden fingerprints of the REAL grid splits
# ======================================================================
@pytest.mark.network
def test_real_split_fingerprints_are_frozen(uci_available):
    """Golden values generated on 2026-08-25 with the current splits.py.

    They pin the ENTIRE split pipeline of the study (index derivation, shuffle
    order, largest-remainder allocation).  If they change, the 120 runs on disk
    are no longer comparable with new ones: regenerate ON PURPOSE and rerun the
    grid, do not just paste the new hashes.
    """
    golden = {
        ("banknote", 0): "622e04d38d8bf3ccd70af308b7952a9d855d8c176db56c79b6ffa317be18215c",
        ("blood", 0): "5f2744375fd654451f415b46823b63d43001f82ba74cdab15d4e8e72e1171d41",
        ("mammographic", 0): "767e49c6db6f16f9d96ec14fe5c9e72c23b92c63cf0a701d418a3a9383965b80",
    }
    for (ds, seed), expected in golden.items():
        got = splits.make_split(ds, seed).hashes["all"]
        assert got == expected, (
            f"{ds}/seed{seed}: split fingerprint changed from {expected} to {got}"
        )


@pytest.mark.network
def test_real_split_sizes_and_stratification(uci_available):
    expected_sizes = {
        "banknote": (961, 206, 205),
        "blood": (523, 113, 112),
        "mammographic": (672, 145, 144),
    }
    for ds in config.DATASETS:
        sp = splits.make_split(ds, 0)
        sizes = (sp.idx_train.size, sp.idx_val.size, sp.idx_test.size)
        assert sizes == expected_sizes[ds], f"{ds}: sizes {sizes}"
        assert sum(sizes) == {"banknote": 1372, "blood": 748,
                              "mammographic": 961}[ds]
