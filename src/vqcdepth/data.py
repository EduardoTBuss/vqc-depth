"""Carga dos 3 datasets UCI — 4 features NATIVAS, ZERO PCA.

Por que validar o id em tempo de carga
--------------------------------------
Um id do ucimlrepo é um número opaco: se o repositório reindexar ou se alguém
digitar errado em config.UCI_IDS, o experimento roda "com sucesso" sobre o
dataset ERRADO e ninguém percebe até a revisão do paper. Por isso a carga
compara nome, nº de instâncias, nº de features e alfabeto do alvo com o
esperado e ABORTA na divergência.

Por que o NaN sobrevive à carga
-------------------------------
Imputar aqui vazaria estatística do conjunto inteiro para dentro do teste.
O missing do mammographic é entregue como NaN e só é preenchido em
splits.make_split, com a mediana do TREINO.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import numpy as np

from . import config

# Raiz do repositório: src/vqcdepth/data.py -> parents[2].
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CACHE_DIR: Final[Path] = Path(
    os.environ.get("VQCDEPTH_CACHE_DIR", _REPO_ROOT / "data" / "cache")
)

# Contrato esperado de cada dataset. É a "prova de identidade" que o download
# tem de satisfazer; qualquer desvio vira AssertionError, não um run silencioso.
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
        # BI-RADS fica de fora: é avaliação subjetiva do radiologista, não
        # medida do achado — usá-la seria vazar o julgamento clínico no rótulo.
        "features": ["Age", "Shape", "Margin", "Density"],
        "target": "Severity",
        "drop": ["BI-RADS"],
        "has_missing": True,
    },
}


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.npz"


def _validate(name: str, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
    """Invariantes que todo consumidor rio abaixo assume sem checar de novo."""
    spec = _SPEC[name]
    assert X.ndim == 2 and X.shape[1] == 4, f"{name}: X deve ser (n, 4), veio {X.shape}"
    assert X.dtype == np.float64, f"{name}: X deve ser float64, veio {X.dtype}"
    assert y.shape == (X.shape[0],), f"{name}: y {y.shape} incompatível com X {X.shape}"
    assert y.dtype == np.int64, f"{name}: y deve ser int64, veio {y.dtype}"
    assert X.shape[0] == spec["n_instances"], (
        f"{name}: esperava n={spec['n_instances']}, veio n={X.shape[0]}"
    )
    assert set(np.unique(y)) == {0, 1}, f"{name}: alvo não binário: {np.unique(y)}"
    assert list(feature_names) == spec["features"], (
        f"{name}: features {feature_names} != esperado {spec['features']}"
    )
    assert not np.isinf(X).any(), f"{name}: X contém inf"
    n_nan = int(np.isnan(X).sum())
    if not spec["has_missing"]:
        assert n_nan == 0, f"{name}: não deveria ter missing, achou {n_nan} NaN"
    # Uma linha 100% NaN é irrecuperável pela imputação por coluna.
    assert not np.isnan(X).all(axis=1).any(), f"{name}: há linha com as 4 features NaN"


def _fetch_from_uci(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Baixa do UCI e confere a identidade do dataset antes de aceitar."""
    from ucimlrepo import fetch_ucirepo  # import tardio: só na falta de cache

    spec = _SPEC[name]
    uci_id = config.UCI_IDS[name]
    try:
        repo = fetch_ucirepo(id=uci_id)
    except Exception as exc:  # rede, TLS, id inexistente...
        raise RuntimeError(
            f"falha ao baixar '{name}' (ucimlrepo id={uci_id}) e não há cache em "
            f"{_cache_path(name)}. Conecte-se à internet uma vez para popular o cache."
        ) from exc

    got_name = repo.metadata["name"]
    assert got_name == spec["uci_name"], (
        f"id {uci_id} devolveu '{got_name}', esperava '{spec['uci_name']}' para "
        f"'{name}'. Corrija config.UCI_IDS antes de rodar o grid."
    )

    df_X = repo.data.features
    df_y = repo.data.targets
    assert list(df_y.columns) == [spec["target"]], (
        f"{name}: alvo {list(df_y.columns)} != ['{spec['target']}']"
    )
    for col in spec["drop"]:
        assert col in df_X.columns, f"{name}: coluna a descartar '{col}' não existe"
    missing = [c for c in spec["features"] if c not in df_X.columns]
    assert not missing, f"{name}: features ausentes no download: {missing}"

    X = df_X[spec["features"]].to_numpy(dtype=np.float64)
    y = df_y[spec["target"]].to_numpy()
    assert not np.isnan(y.astype(np.float64)).any(), f"{name}: alvo com missing"
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
    os.replace(tmp, path)  # troca atômica: cache parcial nunca é lido


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Carrega um dataset do grid.

    Usa o cache local (data/cache/<name>.npz) quando existe; senão baixa do UCI
    uma única vez. Nenhuma imputação, escala ou embaralhamento acontece aqui.

    Returns
    -------
    X : (n, 4) float64 — pode conter NaN (missing declarado, imputado no split)
    y : (n,) int64 em {0, 1}
    feature_names : list[str] de tamanho 4
    """
    if name not in _SPEC:
        raise ValueError(f"dataset desconhecido: {name!r}; use um de {config.DATASETS}")

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
