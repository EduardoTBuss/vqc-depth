"""Loading of the three UCI datasets -- 4 native features, no PCA.

The ucimlrepo id is an opaque number: a reindexed repository or a wrong digit
in config.UCI_IDS would make the experiment run "successfully" on the WRONG
dataset. Loading therefore checks name, instance count, feature names and
target alphabet against the expected contract and aborts on any divergence.

Missing values survive loading on purpose -- imputing here would leak whole
dataset statistics into the test split. They are filled in splits.make_split
with the median of the training split.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import numpy as np

from . import config

# src/vqcdepth/data.py -> parents[2] is the repository root.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CACHE_DIR: Final[Path] = Path(
    os.environ.get("VQCDEPTH_CACHE_DIR", _REPO_ROOT / "data" / "cache")
)

# Identity contract each download must satisfy; any deviation is an
# AssertionError instead of a silent run on the wrong data.
_SPEC: Final[dict[str, dict]] = {
    "banknote": {
        "uci_name": "Banknote Authentication",
        "n_instances": 1372,
        "features": ["variance", "skewness", "curtosis", "entropy"],
        "target": "class",
        "drop": [],
        "has_missing": False,
    },
    "blood": {
        "uci_name": "Blood Transfusion Service Center",
        "n_instances": 748,
        "features": ["Recency", "Frequency", "Monetary", "Time"],
        "target": "Donated_Blood",
        "drop": [],
        "has_missing": False,
    },
    "mammographic": {
        "uci_name": "Mammographic Mass",
        "n_instances": 961,
        # BI-RADS is dropped: it is the radiologist's subjective assessment,
        # not a measurement of the finding -- using it leaks clinical judgement
        # about the label into the features.
        "features": ["Age", "Shape", "Margin", "Density"],
        "target": "Severity",
        "drop": ["BI-RADS"],
        "has_missing": True,
    },
}


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.npz"


def _validate(name: str, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
    """Invariants every downstream consumer assumes without rechecking."""
    spec = _SPEC[name]
    assert X.ndim == 2 and X.shape[1] == 4, f"{name}: X must be (n, 4), got {X.shape}"
    assert X.dtype == np.float64, f"{name}: X must be float64, got {X.dtype}"
    assert y.shape == (X.shape[0],), f"{name}: y {y.shape} does not match X {X.shape}"
    assert y.dtype == np.int64, f"{name}: y must be int64, got {y.dtype}"
    assert X.shape[0] == spec["n_instances"], (
        f"{name}: expected n={spec['n_instances']}, got n={X.shape[0]}"
    )
    assert set(np.unique(y)) == {0, 1}, f"{name}: target is not binary: {np.unique(y)}"
    assert list(feature_names) == spec["features"], (
        f"{name}: features {feature_names} != expected {spec['features']}"
    )
    assert not np.isinf(X).any(), f"{name}: X contains inf"
    n_nan = int(np.isnan(X).sum())
    if not spec["has_missing"]:
        assert n_nan == 0, f"{name}: should have no missing values, found {n_nan} NaN"
    # A fully-NaN row cannot be recovered by per-column imputation.
    assert not np.isnan(X).all(axis=1).any(), f"{name}: a row has all 4 features NaN"


def _fetch_from_uci(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Download from UCI and check the dataset identity before accepting it."""
    from ucimlrepo import fetch_ucirepo  # late import: only on a cache miss

    spec = _SPEC[name]
    uci_id = config.UCI_IDS[name]
    try:
        repo = fetch_ucirepo(id=uci_id)
    except Exception as exc:  # network, TLS, unknown id...
        raise RuntimeError(
            f"could not download '{name}' (ucimlrepo id={uci_id}) and there is no "
            f"cache at {_cache_path(name)}. Go online once to populate the cache."
        ) from exc

    got_name = repo.metadata["name"]
    assert got_name == spec["uci_name"], (
        f"id {uci_id} returned '{got_name}', expected '{spec['uci_name']}' for "
        f"'{name}'. Fix config.UCI_IDS before running the grid."
    )

    df_X = repo.data.features
    df_y = repo.data.targets
    assert list(df_y.columns) == [spec["target"]], (
        f"{name}: target {list(df_y.columns)} != ['{spec['target']}']"
    )
    for col in spec["drop"]:
        assert col in df_X.columns, f"{name}: column to drop '{col}' does not exist"
    missing = [c for c in spec["features"] if c not in df_X.columns]
    assert not missing, f"{name}: features absent from the download: {missing}"

    X = df_X[spec["features"]].to_numpy(dtype=np.float64)
    y = df_y[spec["target"]].to_numpy()
    assert not np.isnan(y.astype(np.float64)).any(), f"{name}: target has missing values"
    y = y.astype(np.int64)

    feature_names = list(spec["features"])
    _validate(name, X, y, feature_names)
    return X, y, feature_names


def _save_cache(name: str, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(name)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            X=X,
            y=y,
            feature_names=np.asarray(feature_names, dtype="<U32"),
            uci_id=np.asarray(config.UCI_IDS[name], dtype=np.int64),
            uci_name=np.asarray(_SPEC[name]["uci_name"], dtype="<U64"),
        )
    os.replace(tmp, path)  # atomic swap: a partial cache is never read


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one dataset of the grid from the local cache, downloading once if
    needed. No imputation, scaling or shuffling happens here.

    Returns (X, y, feature_names) with X (n, 4) float64 -- possibly with NaN,
    the declared missing values imputed later at split time -- and y (n,) int64
    in {0, 1}.
    """
    if name not in _SPEC:
        raise ValueError(f"unknown dataset: {name!r}; use one of {config.DATASETS}")

    path = _cache_path(name)
    if path.exists():
        with np.load(path, allow_pickle=False) as npz:
            X = np.ascontiguousarray(npz["X"], dtype=np.float64)
            y = np.ascontiguousarray(npz["y"], dtype=np.int64)
            feature_names = [str(s) for s in npz["feature_names"]]
        _validate(name, X, y, feature_names)
        return X, y, feature_names

    X, y, feature_names = _fetch_from_uci(name)
    _save_cache(name, X, y, feature_names)
    return X, y, feature_names
