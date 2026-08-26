"""Classical MLP baseline (PyTorch), 4 -> 8 -> 4 -> 1, 81 parameters.

The baseline is not the object of study -- circuit depth is -- so the MLP is
deliberately left untuned (no scheduler, dropout, batch-norm or early stopping)
and gets exactly the same loss, optimizer, learning rate, batch size, epoch
budget and split as the VQCs.

Every weight is drawn from an explicit ``torch.Generator``; the global torch
RNG is never seeded nor consulted.  The initialisation formula is the one
``nn.Linear.reset_parameters`` uses (kaiming_uniform with a=sqrt(5), which
collapses to U(-1/sqrt(fan_in), +1/sqrt(fan_in)) for weight and bias alike),
reimplemented here only so it can be driven by that generator.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

from . import config

__all__ = ["MLP", "build_mlp", "clone_mlp", "n_parameters", "predict_proba",
           "predict_proba_mlp"]


class MLP(nn.Module):
    """4 -> 8 -> 4 -> 1, ReLU on the hidden layers, sigmoid on the output."""

    def __init__(self, n_features: int = config.N_QUBITS,
                 hidden: tuple[int, ...] = config.MLP_HIDDEN):
        super().__init__()
        dims = (n_features,) + tuple(hidden) + (1,)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns p(y=1) with shape (batch,)."""
        return torch.sigmoid(self.net(x)).reshape(-1)


def init_parameters(model: MLP, generator: torch.Generator) -> MLP:
    """Seed every Linear from an explicit generator (in module order)."""
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator (explicit stream)")
    with torch.no_grad():
        for module in model.net:
            if isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                bound = 1.0 / math.sqrt(fan_in)
                module.weight.uniform_(-bound, bound, generator=generator)
                module.bias.uniform_(-bound, bound, generator=generator)
    return model


def build_mlp(generator: torch.Generator, dtype: torch.dtype = torch.float64) -> MLP:
    """Build the baseline with explicitly seeded weights.

    float64 by default so both branches carry the same precision as the
    statevector simulator.
    """
    model = MLP().to(dtype=dtype)
    init_parameters(model, generator)
    return model


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict_proba(model: MLP, X) -> np.ndarray:
    """p(y=1) as a plain float64 numpy array (no grad)."""
    was_training = model.training
    model.eval()
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        xb = torch.as_tensor(np.atleast_2d(np.asarray(X)), dtype=dtype)
        p = model(xb).detach().cpu().numpy().astype(np.float64)
    if was_training:
        model.train()
    return p


def clone_mlp(model: MLP) -> MLP:
    """Detached deep copy, used to snapshot the best epoch."""
    clone = copy.deepcopy(model)
    for p in clone.parameters():
        p.requires_grad_(False)
    clone.eval()
    return clone


def predict_proba_mlp(params, X) -> np.ndarray:
    """p(y=1) from a snapshot: either an MLP module or a state_dict."""
    if isinstance(params, nn.Module):
        return predict_proba(params, X)
    if isinstance(params, dict):
        dtype = next(iter(params.values())).dtype
        model = MLP().to(dtype=dtype)
        model.load_state_dict(params)
        return predict_proba(model, X)
    raise TypeError(f"unsupported MLP snapshot type: {type(params)!r}")
