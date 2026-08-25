"""Persistência atômica: UM ARQUIVO POR RUN. Nunca um CSV compartilhado.

Por que assim
-------------
Drive + dois runtimes do Colab escrevendo no mesmo CSV = corrupção silenciosa.
Cada run escreve o SEU arquivo, de forma atômica (tmp + os.replace). Reabrir a
sessão = varrer os arquivos existentes e rodar só o que falta. Os CSVs
agregados são RECONSTRUÍDOS a partir desses arquivos (`--part rebuild`), e o
rebuild funciona com grid parcial.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

from . import config


# ── layout ────────────────────────────────────────────────────────────

def ensure_layout(outdir: str | os.PathLike) -> Path:
    """Cria a árvore de saída sob --outdir e devolve a raiz."""
    root = Path(outdir)
    for sub in config.SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def run_path(outdir: str | os.PathLike, run_id: str) -> Path:
    return Path(outdir) / "runs" / f"{run_id}.json"


def predictions_path(outdir: str | os.PathLike, run_id: str) -> Path:
    return Path(outdir) / "predictions" / f"{run_id}.parquet"


def history_path(outdir: str | os.PathLike, run_id: str) -> Path:
    return Path(outdir) / "runs" / f"{run_id}__history.csv"


def checkpoint_dir(outdir: str | os.PathLike, run_id: str) -> Path:
    return Path(outdir) / "checkpoints" / run_id


# ── escrita atômica ───────────────────────────────────────────────────

def atomic_write_bytes(path: str | os.PathLike, payload: bytes) -> Path:
    """Escreve tmp no MESMO diretório e faz os.replace (atômico no mesmo FS)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # não deixa lixo se algo explodiu no meio
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def atomic_write_text(path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path, obj: Any) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=2, default=_jsonable))


def _jsonable(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"não serializável: {type(o)}")


# ── leitura / varredura ───────────────────────────────────────────────

def run_exists(outdir, run_id: str) -> bool:
    """Um run só conta como feito se o JSON existe E está íntegro E completed.

    JSON truncado (sessão do Colab morta no meio da escrita) é tratado como
    ausente, para ser refeito — é justamente o caso que a escrita atômica
    deveria impedir, mas a leitura defensiva custa nada.
    """
    p = run_path(outdir, run_id)
    if not p.exists():
        return False
    try:
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    # JSON valido mas que nao e' um objeto ('[]', 'null', '3') passaria pelo
    # guarda acima e explodiria no .get - derrubando a varredura INTEIRA por
    # causa de um arquivo. Trata-se como ausente, para ser refeito.
    if not isinstance(rec, dict):
        return False
    return rec.get("status") == "completed"


def load_run(outdir, run_id: str) -> dict:
    with open(run_path(outdir, run_id), encoding="utf-8") as f:
        return json.load(f)


def iter_runs(outdir) -> Iterator[dict]:
    """Varre todos os runs gravados, ignorando os corrompidos (com aviso)."""
    d = Path(outdir) / "runs"
    if not d.exists():
        return
    for p in sorted(d.glob("*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [aviso] run ilegível, ignorado: {p.name} ({e})")
            continue
        if not isinstance(rec, dict):
            print(f"  [aviso] run não é objeto JSON, ignorado: {p.name}")
            continue
        yield rec


def write_env_json(outdir) -> dict:
    """Versões e hardware da rodada — é isto que vai para a seção de repro."""
    import importlib.metadata as md
    import platform
    import sys
    import time

    pkgs = {}
    for p in ("pennylane", "autograd", "torch", "numpy", "scipy",
              "pandas", "scikit-learn", "ucimlrepo", "pyarrow", "matplotlib"):
        try:
            pkgs[p] = md.version(p)
        except Exception:
            pkgs[p] = None

    env = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": pkgs,
        "device": "default.qubit (statevector analítico, sem shot noise)",
        "epoch_semantics": "1 época = varredura COMPLETA do treino em minibatches",
        "hyperparams": {
            "n_qubits": config.N_QUBITS,
            "vqc_layers": config.VQC_LAYERS,
            "mlp_hidden": list(config.MLP_HIDDEN),
            "n_epochs": config.N_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "seeds": list(config.SEEDS),
            "split": [config.TRAIN_FRAC, config.VAL_FRAC, config.TEST_FRAC],
            "feature_range": list(config.FEATURE_RANGE),
        },
    }
    atomic_write_json(Path(outdir) / "env.json", env)
    return env
