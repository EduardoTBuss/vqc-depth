"""Training loop -- structurally IDENTICAL for the VQCs and for the MLP.

The experiment has ONE independent variable: circuit depth.  Everything the
loop does (epoch definition, batch order, loss, optimizer, epoch budget, model
selection, logged quantities) is therefore shared between the quantum and the
classical branch; the only thing that differs is the object being
differentiated, isolated behind the two ``_Engine`` classes below.

Definition of EPOCH (differs from the qnn-benchmark repo!)
---------------------------------------------------------
One epoch = ONE FULL SWEEP over the training set in minibatches of
``cfg.batch_size``, with the training set reshuffled at the start of every
epoch.  Banknote has ~960 training samples -> ~60 optimizer steps per epoch ->
~6,000 steps per 100-epoch run.  In ``qnn-benchmark`` an "epoch" was a SINGLE
minibatch step; never extrapolate timings from there to here.

Test set
--------
``split.X_test`` / ``split.y_test`` are NEVER read by this module.  Model
selection ("best epoch") uses the VALIDATION set only: highest ``val_f1``,
ties broken by lowest ``val_loss``.  Test-set inference happens downstream,
once, using ``TrainOutput.best_params``.

Randomness
----------
Two independent, explicitly-seeded streams, both derived with ``hashlib``
(deterministic across processes -- the builtin ``hash()`` of a str is salted by
PYTHONHASHSEED and must NOT be used for this):

* batch stream  = f(dataset, seed)          -- NOT a function of the model.
  All four models therefore see the SAME shuffling of the SAME split for a
  given (dataset, seed).  This is what makes the comparison paired: the model
  is the independent variable, so data order is held fixed across models.
* init stream   = f(dataset, model, seed)   -- IS a function of the model,
  because the parameter tensors have different shapes (8/16/32 angles, 81
  torch weights).  Sharing one stream would make VQC-1's angles a prefix of
  VQC-2's, an arbitrary coupling with no scientific justification.

Cost of the per-epoch diagnostics
---------------------------------
``gradient_norm`` costs NOTHING extra: the loop reproduces what
``AdamOptimizer.step_and_cost`` does internally (``compute_grad`` then
``apply_grad``) and simply keeps the gradient that was already computed for the
update.  ``parameter_norm`` is a norm over the parameter tensor, no circuit
call.  The per-epoch metrics DO cost two extra forward passes (one over the
training set, one over the validation set), both fully broadcast (a single
device execution each) -- their measured cost is reported in the pilot and
also returned in ``TrainOutput.timing_breakdown``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
from pennylane import numpy as pnp
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from . import circuits, config, models
from .splits import Split

__all__ = [
    "EpochRecord",
    "TrainOutput",
    "stable_key",
    "batch_rng",
    "init_rng",
    "torch_generator",
    "make_engine",
    "train_model",
]

# --- numerical guards / optimizer constants --------------------------------
# Not experimental hyperparameters (those live in config.py), just numerics.
_BCE_EPS = 1e-7               # probability clipping, identical in both branches
_ADAM_BETAS = (0.9, 0.999)    # textbook Adam.  NOTE: qml.AdamOptimizer defaults
_ADAM_EPS = 1e-8              # to beta2=0.99 and torch to 0.999 -- pinned here
                              # so quantum and classical use the SAME optimizer.


# ======================================================================
# Records
# ======================================================================
@dataclass
class EpochRecord:
    """One row per epoch -- EVERY epoch is recorded, no subsampling.

    gradient_norm    : mean over the epoch's optimizer steps of ||grad||_2
                       (the gradients the optimizer already used; free).
    parameter_norm   : ||theta||_2 at the END of the epoch.
    elapsed_seconds  : wall time of THIS epoch (optimizer steps + metrics),
                       excluding checkpoint I/O.
    """
    epoch: int
    train_loss: float
    train_accuracy: float
    train_f1: float
    val_loss: float
    val_accuracy: float
    val_f1: float
    val_roc_auc: float
    gradient_norm: float
    parameter_norm: float
    elapsed_seconds: float


@dataclass
class TrainOutput:
    history: list[EpochRecord]
    best_epoch: int                 # 1-based; selected on VALIDATION only
    best_val_loss: float
    best_val_f1: float
    best_params: object             # weights of the best epoch (for test-time)
    final_params: object
    training_time_seconds: float
    n_parameters: int
    # extra, defaulted: pilot instrumentation (seconds spent per phase)
    timing_breakdown: dict = field(default_factory=dict)


# ======================================================================
# Deterministic seeding
# ======================================================================
def stable_key(*parts: object) -> int:
    """64-bit key from strings/ints, stable across processes and platforms.

    Uses sha256 -- NOT the builtin hash(), which is randomised per process for
    str/bytes (PYTHONHASHSEED) and silently breaks cross-process
    reproducibility.  That bug exists in qnn-benchmark's batch_rng; fixed here.
    """
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def batch_rng(dataset: str, seed: int) -> np.random.Generator:
    """Minibatch-shuffling stream.  Function of (dataset, seed) ONLY."""
    ss = np.random.SeedSequence(entropy=[int(seed), stable_key("batch", dataset)])
    return np.random.default_rng(ss)


def init_rng(dataset: str, model: str, seed: int) -> np.random.Generator:
    """Parameter-initialisation stream.  Function of (dataset, model, seed)."""
    ss = np.random.SeedSequence(entropy=[int(seed), stable_key("init", dataset, model)])
    return np.random.default_rng(ss)


def torch_generator(dataset: str, model: str, seed: int) -> torch.Generator:
    """torch.Generator for the MLP init -- same entropy as init_rng."""
    g = torch.Generator()
    g.manual_seed(int(init_rng(dataset, model, seed).integers(0, 2 ** 63 - 1)))
    return g


# ======================================================================
# Loss and metrics (identical functional in both branches)
# ======================================================================
def bce_numpy(proba: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(proba, dtype=np.float64), _BCE_EPS, 1.0 - _BCE_EPS)
    t = np.asarray(y, dtype=np.float64)
    return float(-np.mean(t * np.log(p) + (1.0 - t) * np.log(1.0 - p)))


def _metrics(y_true: np.ndarray, proba: np.ndarray, with_auc: bool) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    pred = (np.asarray(proba, dtype=np.float64) >= 0.5).astype(int)
    out = {
        "loss": bce_numpy(proba, y_true),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    if with_auc:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, proba))
        except ValueError:
            # only reachable if a partition is single-class (splits are
            # stratified, so this should never fire); NaN, not a fake 0.5.
            out["roc_auc"] = float("nan")
    return out


# ======================================================================
# Engines -- the ONLY place where quantum and classical differ
# ======================================================================
class _QuantumEngine:
    """PennyLane + autograd VQC, batched via parameter broadcasting."""

    kind = "quantum"

    def __init__(self, cfg):
        self.n_layers = cfg.n_layers
        self.circuit = circuits.make_qnode(self.n_layers)
        rng = init_rng(cfg.dataset, cfg.model, cfg.seed)
        self.params = circuits.init_weights(self.n_layers, rng)
        self.opt = qml.AdamOptimizer(
            stepsize=cfg.learning_rate,
            beta1=_ADAM_BETAS[0], beta2=_ADAM_BETAS[1], eps=_ADAM_EPS,
        )
        self.n_parameters = circuits.n_parameters(self.n_layers)

    def step(self, Xb: np.ndarray, yb: np.ndarray) -> tuple[float, float]:
        Xb = pnp.array(Xb, requires_grad=False)
        yt = pnp.array(np.asarray(yb, dtype=np.float64), requires_grad=False)

        def cost(w):
            p = (circuits.eval_batch(self.circuit, w, Xb) + 1.0) / 2.0
            p = pnp.clip(p, _BCE_EPS, 1.0 - _BCE_EPS)
            return -pnp.mean(yt * pnp.log(p) + (1.0 - yt) * pnp.log(1.0 - p))

        # Exactly what AdamOptimizer.step_and_cost does internally -- unrolled
        # only so the gradient can be read without a second circuit pass.
        grad, loss = self.opt.compute_grad(cost, (self.params,), {})
        self.params = self.opt.apply_grad(grad, (self.params,))[0]
        gnorm = float(np.linalg.norm(np.asarray(grad[0], dtype=np.float64).ravel()))
        return float(loss), gnorm

    def proba(self, X: np.ndarray) -> np.ndarray:
        return circuits.predict_proba(self.circuit, self.params, X)

    def parameter_norm(self) -> float:
        return float(np.linalg.norm(np.asarray(self.params, dtype=np.float64).ravel()))

    def snapshot(self) -> object:
        """Plain float64 array -- what circuits.predict_proba expects."""
        return np.array(np.asarray(self.params, dtype=np.float64), copy=True)

    @staticmethod
    def to_arrays(snap) -> dict[str, np.ndarray]:
        return {"weights": np.asarray(snap, dtype=np.float64)}


class _TorchEngine:
    """Classical MLP baseline.  Same loss, same optimizer, same schedule."""

    kind = "classical"

    def __init__(self, cfg):
        g = torch_generator(cfg.dataset, cfg.model, cfg.seed)
        self.model = models.build_mlp(g)
        self.dtype = next(self.model.parameters()).dtype
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=cfg.learning_rate,
            betas=_ADAM_BETAS, eps=_ADAM_EPS,
        )
        self.n_parameters = models.n_parameters(self.model)

    def step(self, Xb: np.ndarray, yb: np.ndarray) -> tuple[float, float]:
        xb = torch.as_tensor(np.asarray(Xb), dtype=self.dtype)
        yt = torch.as_tensor(np.asarray(yb, dtype=np.float64), dtype=self.dtype)
        self.opt.zero_grad(set_to_none=True)
        p = torch.clamp(self.model(xb), _BCE_EPS, 1.0 - _BCE_EPS)
        loss = -torch.mean(yt * torch.log(p) + (1.0 - yt) * torch.log(1.0 - p))
        loss.backward()
        # gradient already computed for the update -> free to read
        sq = sum(float(pp.grad.pow(2).sum()) for pp in self.model.parameters()
                 if pp.grad is not None)
        self.opt.step()
        return float(loss.detach()), float(np.sqrt(sq))

    def proba(self, X: np.ndarray) -> np.ndarray:
        return models.predict_proba(self.model, X)

    def parameter_norm(self) -> float:
        sq = sum(float(pp.detach().pow(2).sum()) for pp in self.model.parameters())
        return float(np.sqrt(sq))

    def snapshot(self) -> object:
        """A detached copy of the whole module (81 params -- cheap)."""
        return models.clone_mlp(self.model)

    @staticmethod
    def to_arrays(snap) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy()
                for k, v in snap.state_dict().items()}


def make_engine(cfg):
    return _QuantumEngine(cfg) if cfg.is_quantum else _TorchEngine(cfg)


# ======================================================================
# Checkpoints -- experiment ARTEFACTS, not a resume mechanism
# ======================================================================
def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _atomic_save_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _write_checkpoint(directory: Path, cfg, tag: str, engine, snap,
                      history: list[EpochRecord], epoch: int,
                      best_epoch: int, best_val_f1: float,
                      best_val_loss: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.run_id}__{tag}"
    _atomic_save_npz(directory / f"{stem}.npz", engine.to_arrays(snap))
    _atomic_save_json(directory / f"{stem}.json", {
        "run_id": cfg.run_id,
        "dataset": cfg.dataset,
        "model": cfg.model,
        "seed": cfg.seed,
        "n_layers": cfg.n_layers,
        "n_parameters": engine.n_parameters,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "history": [asdict(r) for r in history],
    })


# ======================================================================
# The loop
# ======================================================================
def train_model(cfg, split: Split, checkpoint_dir: str | os.PathLike | None = None,
                verbose: bool = False) -> TrainOutput:
    """Train one grid cell.  Same code path for vqc1 / vqc2 / vqc4 / mlp."""
    if cfg.dataset != getattr(split, "dataset", cfg.dataset):
        raise ValueError(f"split is for {split.dataset!r}, cfg for {cfg.dataset!r}")
    if cfg.seed != getattr(split, "seed", cfg.seed):
        raise ValueError(f"split seed {split.seed} != cfg seed {cfg.seed}")

    X_tr = np.asarray(split.X_train, dtype=np.float64)
    y_tr = np.asarray(split.y_train, dtype=np.float64)
    X_va = np.asarray(split.X_val, dtype=np.float64)
    y_va = np.asarray(split.y_val, dtype=np.float64)
    n_train = len(X_tr)
    if X_tr.shape[1] != config.N_QUBITS:
        raise ValueError(
            f"X_train has {X_tr.shape[1]} features but the circuit has "
            f"{config.N_QUBITS} qubits -- dimension/qubit mismatch"
        )

    engine = make_engine(cfg)
    rng = batch_rng(cfg.dataset, cfg.seed)
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

    history: list[EpochRecord] = []
    best_epoch, best_val_f1, best_val_loss = -1, -np.inf, np.inf
    best_snapshot = engine.snapshot()
    t_step = t_train_metrics = t_val_metrics = t_ckpt = 0.0

    t0 = time.perf_counter()
    for epoch in range(1, cfg.n_epochs + 1):
        t_epoch = time.perf_counter()

        # --- one FULL sweep of the training set, reshuffled ------------
        perm = rng.permutation(n_train)
        gnorms: list[float] = []
        ts = time.perf_counter()
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            _, gnorm = engine.step(X_tr[idx], y_tr[idx])
            gnorms.append(gnorm)
        t_step += time.perf_counter() - ts

        # --- metrics at the epoch's FINAL parameters -------------------
        ts = time.perf_counter()
        tr = _metrics(y_tr, engine.proba(X_tr), with_auc=False)
        t_train_metrics += time.perf_counter() - ts
        ts = time.perf_counter()
        va = _metrics(y_va, engine.proba(X_va), with_auc=True)
        t_val_metrics += time.perf_counter() - ts

        rec = EpochRecord(
            epoch=epoch,
            train_loss=tr["loss"], train_accuracy=tr["accuracy"], train_f1=tr["f1"],
            val_loss=va["loss"], val_accuracy=va["accuracy"], val_f1=va["f1"],
            val_roc_auc=va["roc_auc"],
            gradient_norm=float(np.mean(gnorms)),
            parameter_norm=engine.parameter_norm(),
            elapsed_seconds=time.perf_counter() - t_epoch,
        )
        history.append(rec)

        # --- model selection: VALIDATION only, test never touched ------
        improved = (rec.val_f1 > best_val_f1) or (
            rec.val_f1 == best_val_f1 and rec.val_loss < best_val_loss
        )
        if improved:
            best_epoch, best_val_f1, best_val_loss = epoch, rec.val_f1, rec.val_loss
            best_snapshot = engine.snapshot()

        if verbose:
            print(f"  ep {epoch:3d}/{cfg.n_epochs} "
                  f"loss={rec.train_loss:.4f} val_loss={rec.val_loss:.4f} "
                  f"val_f1={rec.val_f1:.4f} |g|={rec.gradient_norm:.4f} "
                  f"({rec.elapsed_seconds:.2f}s)", flush=True)

        # --- checkpoints (artefacts for the paper) ---------------------
        if ckpt_dir is not None and cfg.save_checkpoints:
            ts = time.perf_counter()
            if epoch % cfg.checkpoint_every == 0 or epoch == cfg.n_epochs:
                _write_checkpoint(ckpt_dir, cfg, f"epoch{epoch:03d}", engine,
                                  engine.snapshot(), history, epoch,
                                  best_epoch, best_val_f1, best_val_loss)
            if improved:
                _write_checkpoint(ckpt_dir, cfg, "best", engine, best_snapshot,
                                  history, epoch, best_epoch, best_val_f1,
                                  best_val_loss)
            t_ckpt += time.perf_counter() - ts

    total = time.perf_counter() - t0
    return TrainOutput(
        history=history,
        best_epoch=best_epoch,
        best_val_loss=float(best_val_loss),
        best_val_f1=float(best_val_f1),
        best_params=best_snapshot,
        final_params=engine.snapshot(),
        training_time_seconds=float(total),
        n_parameters=int(engine.n_parameters),
        timing_breakdown={
            "optimizer_steps": t_step,
            "train_metrics": t_train_metrics,
            "val_metrics": t_val_metrics,
            "checkpoints": t_ckpt,
            "total": total,
            "n_steps": int(np.ceil(n_train / cfg.batch_size)) * cfg.n_epochs,
        },
    )
