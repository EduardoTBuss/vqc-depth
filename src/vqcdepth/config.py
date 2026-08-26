"""Central configuration of the ERAMIA-RS 2026 VQC depth experiment.

Every experiment constant lives here; the single independent variable is
circuit depth (n_layers in 1, 2, 4). One epoch is a COMPLETE sweep over the
training set in minibatches of BATCH_SIZE (banknote: ~60 steps/epoch x 100
epochs = ~6000 optimizer steps per run); the `qnn-benchmark` repository calls a
single minibatch step an "epoch", so its runtimes do not transfer here.
"""

from __future__ import annotations

from dataclasses import dataclass

# -- experiment grid --------------------------------------------------
DATASETS: tuple[str, ...] = ("banknote", "blood", "mammographic")
MODELS: tuple[str, ...] = ("vqc1", "vqc2", "vqc4", "mlp")
SEEDS: tuple[int, ...] = tuple(range(10))

# -- quantum circuit (fixed; only N_LAYERS varies across models) -------
N_QUBITS: int = 4
VQC_LAYERS: dict[str, int] = {"vqc1": 1, "vqc2": 2, "vqc4": 4}
PARAMS_PER_LAYER: int = N_QUBITS * 2   # RY(theta) and RZ(theta) per qubit

# -- MLP baseline (not the object of study; no extensive tuning) -------
MLP_HIDDEN: tuple[int, ...] = (8, 4)   # 4 -> 8 -> 4 -> 1

# -- training (identical for VQC and MLP) ------------------------------
N_EPOCHS: int = 100
BATCH_SIZE: int = 16
LEARNING_RATE: float = 0.05
CHECKPOINT_EVERY: int = 5    # epochs; checkpoints are an artifact, not resume

# -- split -------------------------------------------------------------
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
TEST_FRAC: float = 0.15

# -- encoding: MinMaxScaler to [0, pi], fit on the training split only --
# StandardScaler is not an option: under RY(x), |x| > pi aliases two distinct
# points onto the same state.
FEATURE_RANGE: tuple[float, float] = (0.0, 3.141592653589793)

# -- ucimlrepo ids (validated at load time, see data.py) ---------------
UCI_IDS: dict[str, int] = {
    "banknote": 267,
    "blood": 176,
    "mammographic": 161,
}

# -- differentiation and sampling --------------------------------------
# Paper default: backprop on the analytic simulator. Neither is a variable of
# the study; both exist as a capability. backprop reproduces parameter-shift
# gradients to ~1e-16 at a fraction of the cost, and being able to run both is
# what backs that claim in the methods section. With shots != None backprop is
# no longer valid and parameter-shift is required.
DIFF_METHOD: str = "backprop"
SHOTS: int | None = None

# -- primary metric ----------------------------------------------------
PRIMARY_METRIC: str = "f1"

# -- output layout (root always comes from --outdir) -------------------
SUBDIRS: tuple[str, ...] = (
    "runs", "results", "predictions", "checkpoints", "figures", "tables", "logs",
)


@dataclass(frozen=True)
class RunConfig:
    """Configuration of a single cell of the grid."""
    dataset: str
    model: str
    seed: int
    n_epochs: int = N_EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    checkpoint_every: int = CHECKPOINT_EVERY
    save_checkpoints: bool = True
    diff_method: str = DIFF_METHOD
    shots: int | None = SHOTS

    @property
    def run_id(self) -> str:
        """Canonical identifier, e.g. banknote__vqc4__seed07. Non-default
        configurations get a suffix: without it a shots=1024 run would overwrite
        the analytic run with the same (dataset, model, seed) and the rebuilt
        CSV would silently mix two experiments in one row."""
        rid = f"{self.dataset}__{self.model}__seed{self.seed:02d}"
        if self.is_quantum:
            if self.shots is not None:
                rid += f"__shots{int(self.shots)}"
            if self.diff_method != DIFF_METHOD:
                rid += f"__{self.diff_method.replace('-', '')}"
        return rid

    def __post_init__(self):
        if self.n_epochs < 1:
            # With zero epochs there is no best epoch, and the record would be
            # written with -inf / inf, which is not valid JSON.
            raise ValueError(f"n_epochs must be >= 1, got {self.n_epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.shots is not None and self.diff_method == "backprop":
            raise ValueError(
                "backprop is invalid with finite shots (there is no statevector "
                "to differentiate). Use --diff-method parameter-shift."
            )

    @property
    def n_layers(self) -> int | None:
        return VQC_LAYERS.get(self.model)

    @property
    def is_quantum(self) -> bool:
        return self.model in VQC_LAYERS


def n_parameters(model: str) -> int:
    if model in VQC_LAYERS:
        return VQC_LAYERS[model] * PARAMS_PER_LAYER
    if model == "mlp":
        dims = (4,) + MLP_HIDDEN + (1,)
        return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
    raise ValueError(f"unknown model: {model}")
