"""Splits estratificados 70/15/15 determinísticos + pré-processamento sem vazamento.

O PONTO QUE SUSTENTA O PAPER: para um mesmo (dataset, seed), os 4 modelos
recebem LITERALMENTE os mesmos índices — inclusive entre processos e sessões
diferentes. O split depende só de (dataset, seed), NUNCA do modelo.

Scaler e imputador são ajustados SOMENTE no treino.

Escolhas de determinismo (o "porquê")
-------------------------------------
1. A semente de cada sorteio vem de SHA-256 de "vqc-depth|<dataset>|<seed>|<papel>",
   nunca de hash() do Python (randomizado por processo via PYTHONHASHSEED).
2. O embaralhamento usa np.random.RandomState (Mersenne Twister), o único gerador
   do NumPy com garantia de fluxo idêntico entre versões e plataformas —
   default_rng/PCG64 nao da essa garantia por politica do projeto NumPy.
3. A estratificação é implementada aqui em vez de delegada ao sklearn: assim o
   fingerprint SHA-256 do split não depende da versão do sklearn.
4. Os índices saem EMBARALHADOS (classes intercaladas), não ordenados: se o laço
   de treino algum dia iterar sem shuffle, ainda assim verá minibatches mistos —
   banknote e blood vêm agrupados por classe no arquivo original.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from . import config
from .data import load_dataset


@dataclass(frozen=True)
class Split:
    """Um split pronto para consumo pelos modelos.

    Os arrays X_* já vêm imputados e escalados para [0, pi] (config.FEATURE_RANGE).
    Os idx_* são índices na ordem original do dataset carregado por data.load_dataset.
    """
    dataset: str
    seed: int
    idx_train: np.ndarray          # (n_train,) int64
    idx_val: np.ndarray            # (n_val,)   int64
    idx_test: np.ndarray           # (n_test,)  int64
    X_train: np.ndarray            # (n_train, 4) float64, em [0, pi]
    y_train: np.ndarray            # (n_train,)   int64 em {0, 1}
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    hashes: dict[str, str]         # {"train":sha256, "val":..., "test":..., "all":...}
    n_imputed: dict[str, int]      # imputações aplicadas por partição


def index_hash(idx: np.ndarray) -> str:
    """SHA-256 dos índices, estável entre processos e plataformas.

    O dtype é forçado a int64 little-endian ('<i8'): sem isso, o mesmo split
    hasharia diferente em máquina big-endian ou se o array viesse int32.
    """
    arr = np.ascontiguousarray(np.asarray(idx, dtype="<i8"))
    if arr.ndim != 1:
        raise ValueError(f"index_hash espera vetor 1-D, veio shape {arr.shape}")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def split_hashes(idx_train, idx_val, idx_test) -> dict[str, str]:
    """Fingerprint das três partições; a chave "all" identifica o split inteiro.

    "all" hasheia a concatenação train+val+test NESSA ordem, de modo que trocar
    val por test (mesmos índices, outra função) muda o fingerprint.
    """
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
    """Semente de 32 bits reprodutível a partir de (dataset, seed, papel)."""
    key = f"vqc-depth|{dataset}|{int(seed)}|{role}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def _allocate(n: int, fracs: tuple[float, float, float]) -> tuple[int, int, int]:
    """Reparte n em 3 partes pelo método do maior resto (determinístico).

    Piso simples jogaria a sobra sempre no teste; o maior resto mantém as
    proporções o mais perto possível de 70/15/15 dentro de cada classe.
    """
    raw = [n * f for f in fracs]
    base = [int(np.floor(r)) for r in raw]
    rest = n - sum(base)
    # Desempate fixo pela ordem (train, val, test) — sem RNG, sem ambiguidade.
    order = sorted(range(3), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[:rest]:
        base[i] += 1
    return base[0], base[1], base[2]


def _stratified_indices(
    y: np.ndarray, dataset: str, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """70/15/15 estratificado por classe, embaralhado, função de (dataset, seed)."""
    fracs = (config.TRAIN_FRAC, config.VAL_FRAC, config.TEST_FRAC)
    assert abs(sum(fracs) - 1.0) < 1e-12, f"frações não somam 1: {fracs}"

    parts: list[list[np.ndarray]] = [[], [], []]
    for cls in np.unique(y):
        # flatnonzero devolve ordem crescente: ponto de partida canônico.
        idx_c = np.flatnonzero(y == cls).astype(np.int64)
        rng = np.random.RandomState(_derive_seed(dataset, seed, f"class{int(cls)}"))
        idx_c = idx_c[rng.permutation(idx_c.size)]
        n_tr, n_va, n_te = _allocate(idx_c.size, fracs)
        assert min(n_tr, n_va, n_te) > 0, (
            f"classe {cls} tem só {idx_c.size} exemplos: partição vazia em 70/15/15"
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
    assert total == y.size, f"split perdeu amostras: {total} != {y.size}"
    assert np.array_equal(
        np.sort(np.concatenate(out)), np.arange(y.size, dtype=np.int64)
    ), "partições se sobrepõem ou não cobrem o dataset"
    return idx_train, idx_val, idx_test


def make_split(dataset: str, seed: int) -> Split:
    """Split 70/15/15 estratificado determinístico, função APENAS de (dataset, seed)."""
    if dataset not in config.DATASETS:
        raise ValueError(f"dataset desconhecido: {dataset!r}; use um de {config.DATASETS}")
    seed = int(seed)

    X, y, feature_names = load_dataset(dataset)
    idx_train, idx_val, idx_test = _stratified_indices(y, dataset, seed)

    X_tr_raw, X_va_raw, X_te_raw = X[idx_train], X[idx_val], X[idx_test]

    # ── Imputação: mediana do TREINO, aplicada às três partições ─────────
    # nanmedian ignora o missing; a mediana global vazaria val/test no treino.
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
            f"{dataset}/seed{seed}: coluna inteiramente NaN no treino, sem mediana"
        )
        X_tr = np.where(np.isnan(X_tr_raw), median, X_tr_raw)
        X_va = np.where(np.isnan(X_va_raw), median, X_va_raw)
        X_te = np.where(np.isnan(X_te_raw), median, X_te_raw)
    else:
        X_tr, X_va, X_te = X_tr_raw.copy(), X_va_raw.copy(), X_te_raw.copy()

    # ── Escala para [0, pi]: fit SÓ no treino ────────────────────────────
    # MinMax (não StandardScaler): em RY(x), |x| > pi alia estados distintos.
    # Val/test podem sair de [0, pi] — extrapolação legítima, NÃO se clipa;
    # clipar seria fingir que o treino viu um extremo que ele não viu.
    scaler = MinMaxScaler(feature_range=config.FEATURE_RANGE)
    X_tr = np.ascontiguousarray(scaler.fit_transform(X_tr), dtype=np.float64)
    X_va = np.ascontiguousarray(scaler.transform(X_va), dtype=np.float64)
    X_te = np.ascontiguousarray(scaler.transform(X_te), dtype=np.float64)

    for part, arr in (("train", X_tr), ("val", X_va), ("test", X_te)):
        assert np.isfinite(arr).all(), f"{dataset}/seed{seed}: NaN/inf em X_{part}"

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
