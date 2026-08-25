"""Configuração central do experimento ERAMIA-RS 2026 (profundidade de VQC).

TODA constante do experimento vive aqui. Nenhum outro módulo define
hiperparâmetro. A variável independente do estudo é UMA SÓ: a profundidade
do circuito (n_layers em 1, 2, 4).

Nota crítica sobre ÉPOCA
------------------------
Aqui, uma ÉPOCA é uma varredura COMPLETA do conjunto de treino em minibatches
de tamanho BATCH_SIZE. Isso é diferente do repositório `qnn-benchmark`, onde
"época" era UM ÚNICO passo de minibatch. A diferença de custo é de ordens de
grandeza (Banknote: ~60 passos/época x 100 épocas = ~6.000 passos por run).
Nunca extrapole tempo do qnn-benchmark para cá.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# ── Grid do experimento ───────────────────────────────────────────────
DATASETS: tuple[str, ...] = ("banknote", "blood", "mammographic")
MODELS: tuple[str, ...] = ("vqc1", "vqc2", "vqc4", "mlp")
SEEDS: tuple[int, ...] = tuple(range(10))

# ── Circuito quântico (fixo; só N_LAYERS varia entre modelos) ─────────
N_QUBITS: int = 4
VQC_LAYERS: dict[str, int] = {"vqc1": 1, "vqc2": 2, "vqc4": 4}
# params por camada = N_QUBITS * 2  (RY(theta) e RZ(theta) por qubit)
PARAMS_PER_LAYER: int = N_QUBITS * 2

# ── MLP baseline (NÃO é objeto de estudo; sem tuning extensivo) ───────
MLP_HIDDEN: tuple[int, ...] = (8, 4)   # 4 -> 8 -> 4 -> 1

# ── Treino (idêntico para VQC e MLP) ──────────────────────────────────
N_EPOCHS: int = 100          # épocas completas (ver docstring do módulo)
BATCH_SIZE: int = 16
LEARNING_RATE: float = 0.05  # Adam, mesmo para VQC e MLP
CHECKPOINT_EVERY: int = 5    # épocas; é ARTEFATO do experimento, não resume

# ── Split ─────────────────────────────────────────────────────────────
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
TEST_FRAC: float = 0.15

# ── Encoding: MinMaxScaler para [0, pi], ajustado SÓ no treino ────────
# StandardScaler é proibido: com RY(x), |x| > pi alia (dois pontos
# distintos viram o mesmo estado).
FEATURE_RANGE: tuple[float, float] = (0.0, 3.141592653589793)

# ── ucimlrepo ids (VERIFICADOS EM TEMPO DE CARGA, ver data.py) ───────
UCI_IDS: dict[str, int] = {
    "banknote": 267,       # Banknote Authentication
    "blood": 176,          # Blood Transfusion Service Center
    "mammographic": 161,   # Mammographic Mass
}

# ── Métrica primária ──────────────────────────────────────────────────
PRIMARY_METRIC: str = "f1"

# ── Layout de saída (raiz sempre vem de --outdir) ────────────────────
SUBDIRS: tuple[str, ...] = (
    "runs", "results", "predictions", "checkpoints", "figures", "tables", "logs",
)


@dataclass(frozen=True)
class RunConfig:
    """Configuração de UMA execução do grid."""
    dataset: str
    model: str
    seed: int
    n_epochs: int = N_EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    checkpoint_every: int = CHECKPOINT_EVERY
    save_checkpoints: bool = True

    @property
    def run_id(self) -> str:
        """Identificador canônico: banknote__vqc4__seed07."""
        return f"{self.dataset}__{self.model}__seed{self.seed:02d}"

    @property
    def n_layers(self) -> int | None:
        return VQC_LAYERS.get(self.model)

    @property
    def is_quantum(self) -> bool:
        return self.model in VQC_LAYERS


def n_parameters(model: str) -> int:
    """Número de parâmetros treináveis de cada modelo."""
    if model in VQC_LAYERS:
        return VQC_LAYERS[model] * PARAMS_PER_LAYER
    if model == "mlp":
        dims = (4,) + MLP_HIDDEN + (1,)
        return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
    raise ValueError(f"modelo desconhecido: {model}")
