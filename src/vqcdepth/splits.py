"""Deterministic stratified 70/15/15 splits + leakage-free preprocessing.

For a given (dataset, seed) the four models receive literally the same indices,
across processes and sessions: the split is a function of (dataset, seed) only,
never of the model. That is what makes the paired Wilcoxon test legitimate.
Scaler and imputer are fit on the training split only.

Determinism choices:
1. Seeds come from SHA-256 of "vqc-depth|<dataset>|<seed>|<role>", never from
   Python's hash(), which is randomized per process via PYTHONHASHSEED.
2. Shuffling uses RandomState (Mersenne Twister), the only NumPy generator with
   a guaranteed identical stream across versions; default_rng/PCG64 has none.
3. Stratification is implemented here, not delegated to sklearn, so the split
   fingerprint does not depend on the sklearn version.
4. Indices come out shuffled: banknote and blood ship grouped by class, so a
   loop that ever iterated without shuffling would still see mixed minibatches.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from . import config
from .data import load_dataset


@dataclass(frozen=True)
class Split:
    """X_* are imputed and scaled to [0, pi]; idx_* index into the original
    order returned by data.load_dataset."""
    dataset: str
    seed: int
    idx_train: np.ndarray          # (n_train,) int64
    idx_val: np.ndarray            # (n_val,)   int64
    idx_test: np.ndarray           # (n_test,)  int64
    X_train: np.ndarray            # (n_train, 4) float64, in [0, pi]
    y_train: np.ndarray            # (n_train,)   int64 in {0, 1}
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    hashes: dict[str, str]         # {"train": sha256, "val": ..., "test": ..., "all": ...}
    n_imputed: dict[str, int]      # imputed cells per partition


def index_hash(idx: np.ndarray) -> str:
    """SHA-256 of the indices, stable across processes and platforms. The
    little-endian int64 cast is what makes it stable: without it the same split
    hashes differently on a big-endian machine or from an int32 array."""
    arr = np.ascontiguousarray(np.asarray(idx, dtype="<i8"))
    if arr.ndim != 1:
        raise ValueError(f"index_hash expects a 1-D vector, got shape {arr.shape}")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def split_hashes(idx_train, idx_val, idx_test) -> dict[str, str]:
    """Fingerprint of the three partitions. "all" hashes train+val+test in that
    order, so swapping val for test changes the fingerprint."""
    tr = np.asarray(idx_train, dtype="<i8")
    va = np.asarray(idx_val, dtype="<i8")
    te = np.asarray(idx_test, dtype="<i8")
    return {
        "train": index_hash(tr),
        "val": index_hash(va),
        "test": index_hash(te),
        "all": index_hash(np.concatenate([tr, va, te])),
    }


def _derive_seed(dataset: str, seed: int, role: str) -> int:
    key = f"vqc-depth|{dataset}|{int(seed)}|{role}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def _allocate(n: int, fracs: tuple[float, float, float]) -> tuple[int, int, int]:
    """Split n into 3 parts by largest remainder: plain flooring would always
    dump the leftover into the test partition."""
    raw = [n * f for f in fracs]
    base = [int(np.floor(r)) for r in raw]
    rest = n - sum(base)
    # Ties broken by the fixed (train, val, test) order: no RNG, no ambiguity.
    order = sorted(range(3), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[:rest]:
        base[i] += 1
    return base[0], base[1], base[2]


def _stratified_indices(
    y: np.ndarray, dataset: str, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """70/15/15 stratified per class, shuffled, function of (dataset, seed)."""
    fracs = (config.TRAIN_FRAC, config.VAL_FRAC, config.TEST_FRAC)
    assert abs(sum(fracs) - 1.0) < 1e-12, f"fractions do not sum to 1: {fracs}"

    parts: list[list[np.ndarray]] = [[], [], []]
    for cls in np.unique(y):
        # flatnonzero returns ascending order: a canonical starting point.
        idx_c = np.flatnonzero(y == cls).astype(np.int64)
        rng = np.random.RandomState(_derive_seed(dataset, seed, f"class{int(cls)}"))
        idx_c = idx_c[rng.permutation(idx_c.size)]
        n_tr, n_va, n_te = _allocate(idx_c.size, fracs)
        assert min(n_tr, n_va, n_te) > 0, (
            f"class {cls} has only {idx_c.size} samples: empty partition at 70/15/15"
        )
        parts[0].append(idx_c[:n_tr])
        parts[1].append(idx_c[n_tr:n_tr + n_va])
        parts[2].append(idx_c[n_tr + n_va:])

    out = []
    for k, role in enumerate(("train", "val", "test")):
        idx = np.concatenate(parts[k])
        rng = np.random.RandomState(_derive_seed(dataset, seed, f"shuffle-{role}"))
        out.append(np.ascontiguousarray(idx[rng.permutation(idx.size)], dtype=np.int64))

    idx_train, idx_val, idx_test = out
    total = idx_train.size + idx_val.size + idx_test.size
    assert total == y.size, f"the split lost samples: {total} != {y.size}"
    assert np.array_equal(
        np.sort(np.concatenate(out)), np.arange(y.size, dtype=np.int64)
    ), "partitions overlap or do not cover the dataset"
    return idx_train, idx_val, idx_test


def make_split(dataset: str, seed: int) -> Split:
    """Deterministic stratified 70/15/15 split, a function of (dataset, seed) only."""
    if dataset not in config.DATASETS:
        raise ValueError(f"unknown dataset: {dataset!r}; use one of {config.DATASETS}")
    seed = int(seed)

    X, y, feature_names = load_dataset(dataset)
    idx_train, idx_val, idx_test = _stratified_indices(y, dataset, seed)

    X_tr_raw, X_va_raw, X_te_raw = X[idx_train], X[idx_val], X[idx_test]

    # -- imputation: TRAINING median applied to all three partitions ------
    # nanmedian ignores the missing cells; a global median would leak val/test
    # statistics into training.
    n_imputed = {
        "train": int(np.isnan(X_tr_raw).sum()),
        "val": int(np.isnan(X_va_raw).sum()),
        "test": int(np.isnan(X_te_raw).sum()),
    }
    n_imputed["total"] = n_imputed["train"] + n_imputed["val"] + n_imputed["test"]

    if n_imputed["total"] > 0:
        with np.errstate(all="ignore"):
            median = np.nanmedian(X_tr_raw, axis=0)
        assert not np.isnan(median).any(), (
            f"{dataset}/seed{seed}: fully-NaN column in training, no median"
        )
        X_tr = np.where(np.isnan(X_tr_raw), median, X_tr_raw)
        X_va = np.where(np.isnan(X_va_raw), median, X_va_raw)
        X_te = np.where(np.isnan(X_te_raw), median, X_te_raw)
    else:
        X_tr, X_va, X_te = X_tr_raw.copy(), X_va_raw.copy(), X_te_raw.copy()

    # -- scaling to [0, pi], fit on training only -------------------------
    # MinMax rather than StandardScaler: under RY(x), |x| > pi aliases distinct
    # points onto the same state. Val/test may fall outside [0, pi] -- that is
    # legitimate extrapolation and is NOT clipped; clipping would pretend the
    # training split had seen an extreme it never saw.
    scaler = MinMaxScaler(feature_range=config.FEATURE_RANGE)
    X_tr = np.ascontiguousarray(scaler.fit_transform(X_tr), dtype=np.float64)
    X_va = np.ascontiguousarray(scaler.transform(X_va), dtype=np.float64)
    X_te = np.ascontiguousarray(scaler.transform(X_te), dtype=np.float64)

    for part, arr in (("train", X_tr), ("val", X_va), ("test", X_te)):
        assert np.isfinite(arr).all(), f"{dataset}/seed{seed}: NaN/inf in X_{part}"

    return Split(
        dataset=dataset,
        seed=seed,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        X_train=X_tr,
        y_train=np.ascontiguousarray(y[idx_train], dtype=np.int64),
        X_val=X_va,
        y_val=np.ascontiguousarray(y[idx_val], dtype=np.int64),
        X_test=X_te,
        y_test=np.ascontiguousarray(y[idx_test], dtype=np.int64),
        feature_names=list(feature_names),
        hashes=split_hashes(idx_train, idx_val, idx_test),
        n_imputed=n_imputed,
    )
