"""Variational quantum classifier used in the ERAMIA-RS 2026 depth study.

The ONLY thing that changes between vqc1 / vqc2 / vqc4 is ``n_layers``:
same encoding, same layer template, same observable, same weight layout.

Qubit ordering / conventions (read before touching anything)
------------------------------------------------------------
* PennyLane wires are ``0..N_QUBITS-1``; feature ``j`` of the sample is encoded
  on wire ``j`` (``RY(x_j)`` on wire ``j``).  4 features -> 4 qubits, 1:1.
* PennyLane's statevector ordering is BIG-ENDIAN in the wire index (wire 0 is
  the MOST significant bit of the computational-basis label).  This is the
  opposite of Qiskit's little-endian convention; if any number here is ever
  compared against a Qiskit table, wire order must be reversed there.
* The readout is ``<Z>`` on wire 0 -- i.e. the FIRST feature's qubit.  The chain
  of CNOTs is what lets information from wires 1..3 reach wire 0, so a circuit
  with 0 entangling blocks would be blind to features 1..3 by construction.
* p(y=1) = (1 + <Z_0>) / 2, so <Z_0> = +1 -> p = 1 and <Z_0> = -1 -> p = 0.

Circuit (n_layers = L)
----------------------
    RY(x_0) w0 ; RY(x_1) w1 ; RY(x_2) w2 ; RY(x_3) w3      <- encoding, ONCE
    repeat L times:
        for j in 0..3:  RZ(theta[l,j,0]) w_j ; RY(theta[l,j,1]) w_j
        CNOT(w3->w2) ; CNOT(w2->w1) ; CNOT(w1->w0)          <- chain TOWARD w0
    measure <Z_0>

There is NO data re-uploading: the encoding block appears exactly once, before
the first variational layer.  (``vqc_c`` in the qnn-benchmark repo DOES
re-upload; that is a different circuit and was not ported.)

Weight tensor shape: ``(n_layers, N_QUBITS, 2)`` -> 8 params per layer.
    index [..., 0] -> RZ angle,  index [..., 1] -> RY angle.

WHY THIS EXACT ORDER AND THIS EXACT CHAIN DIRECTION (measured, 2026-08-25)
--------------------------------------------------------------------------
Both choices exist to make EVERY angle count.  Pulling the readout back
(Heisenberg) through the final entangling block decides which angles are dead:

* With a CNOT **ring** (``j -> (j+1) % 4``) the pullback is
  ``U_ring^dag Z_0 U_ring = Z_1 Z_2 Z_3``: it acts as the IDENTITY on wire 0,
  so every last-layer rotation on wire 0 becomes invisible.
* With the CNOT **chain toward the readout** (3->2->1->0) the pullback is
  ``Z_0 Z_1 Z_2 Z_3``, which acts non-trivially on ALL four wires.
* The pullback is always DIAGONAL (a product of Z's), so whatever rotation sits
  LAST before the entangler must not commute with Z.  ``RZ`` does commute, so
  ``RZ`` must come FIRST and ``RY`` last inside the layer.

Measured effective angle counts (perturb each angle by 0.9 rad, count those
that move the output by more than 1e-12):

    ansatz                          L=1     L=2     L=4
    RY,RZ + ring   (first draft)    3/8    11/16   27/32
    RZ,RY + ring                    6/8    14/16   30/32
    RY only + chain 3->2->1->0      4/4     8/8    16/16
    RZ,RY + chain 3->2->1->0  <--   8/8    16/16   32/32   THIS ONE

The chosen ansatz is the only measured variant where nominal == effective at
every depth, so ``config.n_parameters`` (8 / 16 / 32) is honest and the
progression is exactly 1:2:4.  It also keeps two rotations per qubit, so the
state is NOT restricted to real amplitudes (an ``RY``-only ansatz would be).

``tests/test_ansatz.py`` pins these numbers: if the layer template or the chain
direction is ever changed, it fails on purpose and forces a conscious decision
instead of silently changing every number in the study.

Simulation
----------
``default.qubit``, analytic statevector (shots=None), float64, CPU.  4 qubits =
16 amplitudes, so cost is dominated by Python/autograd overhead, not by the
linear algebra.  ``diff_method="backprop"`` is pinned explicitly: on an ideal
simulator it returns the same exact gradient as parameter-shift (verified in
``_scratch_check_train.py``) at a fraction of the cost.  Switch to
"parameter-shift" only if the study ever moves to shot noise / hardware.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from . import config

__all__ = [
    "weights_shape",
    "n_parameters",
    "make_qnode",
    "init_weights",
    "eval_batch",
    "predict_proba",
]


def weights_shape(n_layers: int) -> tuple[int, int, int]:
    """Shape of the trainable tensor: (n_layers, n_qubits, 2)."""
    return (int(n_layers), config.N_QUBITS, 2)


def n_parameters(n_layers: int) -> int:
    """Number of trainable angles = n_layers * N_QUBITS * 2."""
    return int(np.prod(weights_shape(n_layers)))


def make_qnode(n_layers: int, device: object | None = None):
    """Build the QNode for a circuit with ``n_layers`` variational layers.

    The returned callable has signature ``circuit(inputs, weights)`` where
    ``inputs`` is either ``(N_QUBITS,)`` (one sample) or ``(batch, N_QUBITS)``
    (parameter broadcasting -- the whole minibatch in ONE device call) and
    ``weights`` has shape ``weights_shape(n_layers)``.
    """
    if int(n_layers) < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    n_layers = int(n_layers)
    dev = device if device is not None else qml.device(
        "default.qubit", wires=config.N_QUBITS, shots=None
    )

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(inputs, weights):
        # --- encoding: angle encoding, applied exactly once -------------
        # inputs[..., j] keeps the leading broadcast axis (if any) intact.
        for j in range(config.N_QUBITS):
            qml.RY(inputs[..., j], wires=j)
        # --- variational layers -----------------------------------------
        for layer in range(n_layers):
            for j in range(config.N_QUBITS):
                qml.RZ(weights[layer, j, 0], wires=j)
                qml.RY(weights[layer, j, 1], wires=j)
            # CNOT chain running TOWARD the readout wire (3->2->1->0).  The
            # direction is not cosmetic: see the note at the top of the module.
            for j in range(config.N_QUBITS - 1, 0, -1):
                qml.CNOT(wires=[j, j - 1])
        return qml.expval(qml.PauliZ(0))

    circuit.n_layers = n_layers
    return circuit


def init_weights(n_layers: int, rng: np.random.Generator) -> pnp.ndarray:
    """Uniform init in [-pi, pi] from an EXPLICIT Generator.

    Never touches the global NumPy RNG: reproducibility across processes must
    not depend on interpreter-wide state.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator (explicit stream)")
    w = rng.uniform(-np.pi, np.pi, size=weights_shape(n_layers))
    return pnp.array(w, requires_grad=True)


def eval_batch(circuit, weights, X):
    """Evaluate <Z_0> for the WHOLE batch in a single device call.

    Uses PennyLane parameter broadcasting: identical value AND identical
    gradient as ``[circuit(x, weights) for x in X]`` (checked to ~1e-12 in
    ``_scratch_check_train.py``), but one device execution instead of ``len(X)``.
    This is where the speedup of the whole grid comes from.

    Returns an array of shape ``(len(X),)`` -- also for a single-row X.
    """
    Xb = pnp.array(np.atleast_2d(X), requires_grad=False)
    out = circuit(Xb, weights)
    return pnp.reshape(out, (Xb.shape[0],))


def predict_proba(circuit, weights, X) -> np.ndarray:
    """p(y=1) = (1 + <Z_0>) / 2, as a plain float64 numpy array."""
    raw = np.asarray(eval_batch(circuit, weights, X), dtype=np.float64)
    return (raw + 1.0) / 2.0
