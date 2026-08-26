"""Orquestrador do grid: decide o que falta, executa, persiste, retoma.

Retomada e' POR RUN, nao por epoca. Runs sao curtos o bastante para que resume
intra-run seja over-engineering; os checkpoints por epoca existem como
ARTEFATO do experimento (curvas, analise de barren plateau), nao como
mecanismo de recuperacao.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, log_loss, precision_score, recall_score,
    roc_auc_score,
)

from . import config, storage
from .config import RunConfig


# -- metricas ---------------------------------------------------------

def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                           threshold: float = 0.5) -> dict:
    """Bateria completa. TN/FP/FN/TP saem como NUMEROS - a figura se recria."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-12, 1 - 1e-12)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    # AUC exige as duas classes presentes; com 15% de teste num dataset
    # desbalanceado isso pode falhar, e falhar em silencio seria pior.
    try:
        roc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc = float("nan")
    try:
        pr = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr = float("nan")

    return {
        "loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": roc,
        "pr_auc": pr,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


# -- selecao do que falta ---------------------------------------------

def build_grid(datasets: Sequence[str] | None = None,
               models: Sequence[str] | None = None,
               seeds: Sequence[int] | None = None,
               **run_kwargs) -> list[RunConfig]:
    ds = tuple(datasets) if datasets else config.DATASETS
    md = tuple(models) if models else config.MODELS
    sd = tuple(seeds) if seeds is not None else config.SEEDS
    for d in ds:
        if d not in config.DATASETS:
            raise ValueError(f"dataset desconhecido: {d} (use {config.DATASETS})")
    for m in md:
        if m not in config.MODELS:
            raise ValueError(f"modelo desconhecido: {m} (use {config.MODELS})")
    return [RunConfig(dataset=d, model=m, seed=int(s), **run_kwargs)
            for d in ds for s in sd for m in md]


def pending(grid: Iterable[RunConfig], outdir) -> list[RunConfig]:
    """O que ainda falta. E' isto que torna a sessao do Colab retomavel."""
    return [c for c in grid if not storage.run_exists(outdir, c.run_id)]


# -- execucao de UM run -----------------------------------------------

def execute_run(cfg: RunConfig, outdir) -> dict:
    """Treina, avalia no teste UMA vez, persiste tudo. Devolve o registro."""
    # imports tardios: manter o CLI leve e o --part rebuild sem PennyLane
    from .splits import make_split
    from .train import train_model

    t_wall = time.time()
    split = make_split(cfg.dataset, cfg.seed)
    ckpt_dir = storage.checkpoint_dir(outdir, cfg.run_id) if cfg.save_checkpoints else None
    out = train_model(cfg, split, checkpoint_dir=ckpt_dir)

    # O teste e' tocado AQUI e so aqui, com os pesos da melhor epoca de VALIDACAO.
    y_prob, inference_time = _predict_test(cfg, out, split)
    test = classification_metrics(split.y_test, y_prob)

    record = {
        "run_id": cfg.run_id,
        "dataset": cfg.dataset,
        "model": cfg.model,
        "seed": cfg.seed,
        "n_layers": cfg.n_layers,
        "n_qubits": config.N_QUBITS if cfg.is_quantum else None,
        "n_parameters": out.n_parameters,
        "train_samples": int(len(split.y_train)),
        "val_samples": int(len(split.y_val)),
        "test_samples": int(len(split.y_test)),
        "best_epoch": out.best_epoch,
        "final_epoch": len(out.history),
        "best_val_loss": out.best_val_loss,
        "best_val_f1": out.best_val_f1,
        "training_time_seconds": out.training_time_seconds,
        "inference_time_seconds": inference_time,
        "wall_time_seconds": time.time() - t_wall,
        "split_hash_train": split.hashes["train"],
        "split_hash_val": split.hashes["val"],
        "split_hash_test": split.hashes["test"],
        "split_hash_all": split.hashes["all"],
        "n_imputed": split.n_imputed,
        "n_epochs_configured": cfg.n_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "diff_method": cfg.diff_method if cfg.is_quantum else None,
        "shots": cfg.shots if cfg.is_quantum else None,
        "status": "completed",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    for k, v in test.items():
        record["test_" + k if k not in ("TN", "FP", "FN", "TP") else k] = v

    _persist(cfg, outdir, record, out, split, y_prob)
    return record


def _predict_test(cfg: RunConfig, out, split) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    if cfg.is_quantum:
        from .circuits import make_qnode, predict_proba
        # Inferência SEMPRE analítica, mesmo quando o TREINO usou shots: o que
        # se estuda com shots é o efeito do ruído na OTIMIZAÇÃO, e medir o teste
        # com ruído misturaria os dois efeitos numa métrica só.
        qnode = make_qnode(cfg.n_layers, diff_method="backprop", shots=None)
        y_prob = predict_proba(qnode, out.best_params, split.X_test)
    else:
        from .models import predict_proba_mlp
        y_prob = predict_proba_mlp(out.best_params, split.X_test)
    return np.asarray(y_prob, dtype=np.float64), time.perf_counter() - t0


def _persist(cfg, outdir, record, out, split, y_prob) -> None:
    """Historico completo + predicoes + registro. Registro por ULTIMO."""
    hist = pd.DataFrame([vars(e) if hasattr(e, "__dict__") else dict(e)
                         for e in out.history])
    hist.insert(0, "model", cfg.model)
    hist.insert(0, "seed", cfg.seed)
    hist.insert(0, "dataset", cfg.dataset)
    storage.atomic_write_text(storage.history_path(outdir, cfg.run_id),
                              hist.to_csv(index=False))

    # dataset/model/seed vao COMO COLUNAS, alem de estarem no nome do arquivo:
    # depender so do nome torna a leitura fragil a qualquer renomeacao.
    preds = pd.DataFrame({
        "dataset": cfg.dataset,
        "model": cfg.model,
        "seed": cfg.seed,
        "sample_id": np.asarray(split.idx_test, dtype=np.int64),
        "y_true": np.asarray(split.y_test, dtype=np.int64),
        "y_pred": (y_prob >= 0.5).astype(np.int64),
        "y_probability": y_prob,
    })
    _write_predictions(storage.predictions_path(outdir, cfg.run_id), preds)

    # o JSON do run e' o marcador de "feito": so depois de tudo o mais no disco
    storage.atomic_write_json(storage.run_path(outdir, cfg.run_id), record)


def _write_predictions(path: Path, df: pd.DataFrame) -> None:
    """Parquet se houver engine; senao CSV - nao vale perder o run por isso."""
    import io
    try:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        storage.atomic_write_bytes(path, buf.getvalue())
    except (ImportError, ValueError) as e:
        print(f"  [aviso] parquet indisponivel ({e}); gravando CSV")
        storage.atomic_write_text(path.with_suffix(".csv"), df.to_csv(index=False))


# -- laco do grid -----------------------------------------------------

def run_grid(grid: Sequence[RunConfig], outdir, stop_on_error: bool = False) -> dict:
    """Executa o que falta, um run por vez, persistindo a cada conclusao."""
    storage.ensure_layout(outdir)
    storage.write_env_json(outdir)

    todo = pending(grid, outdir)
    done = len(grid) - len(todo)
    print(f"\n  grid: {len(grid)} runs | ja no disco: {done} | a executar: {len(todo)}")

    ok, failed = 0, []
    for i, cfg in enumerate(todo, 1):
        t0 = time.time()
        print(f"  [{i}/{len(todo)}] {cfg.run_id} ...", flush=True)
        try:
            rec = execute_run(cfg, outdir)
            ok += 1
            print("      ok  test_f1={:.4f} acc={:.4f} best_ep={} ({:.1f}s)".format(
                rec["test_f1"], rec["test_accuracy"], rec["best_epoch"],
                time.time() - t0), flush=True)
        except KeyboardInterrupt:
            print("\n  interrompido pelo usuario; o que terminou esta no disco.")
            break
        except Exception as e:
            failed.append((cfg.run_id, repr(e)))
            print(f"      FALHOU: {e}")
            traceback.print_exc()
            if stop_on_error:
                raise
    print(f"\n  concluidos nesta sessao: {ok} | falhas: {len(failed)}")
    for rid, err in failed:
        print(f"    - {rid}: {err}")
    return {"executed": ok, "failed": failed, "pending_before": len(todo)}
