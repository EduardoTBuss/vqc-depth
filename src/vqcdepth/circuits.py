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
* The readout is ``<Z>`` on wire 0 -- i.e. the FIRST feature's qubit.  The ring
  of CNOTs is what lets information from wires 1..3 reach wire 0, so a circuit
  with 0 entangling blocks would be blind to features 1..3 by construction.
* p(y=1) = (1 + <Z_0>) / 2, so <Z_0> = +1 -> p = 1 and <Z_0> = -1 -> p = 0.

Circuit (n_layers = L)
----------------------
    RY(x_0) w0 ; RY(x_1) w1 ; RY(x_2) w2 ; RY(x_3) w3      <- encoding, ONCE
    repeat L times:
        for j in 0..3:  RY(theta[l,j,0]) w_j ; RZ(theta[l,j,1]) w_j
        for j in 0..3:  CNOT(w_j -> w_{(j+1) % 4})          <- ring
    measure <Z_0>

There is NO data re-uploading: the encoding block appears exactly once, before
the first variational layer.  (``vqc_c`` in the qnn-benchmark repo DOES
re-upload; that is a different circuit and was not ported.)

Weight tensor shape: ``(n_layers, N_QUBITS, 2)`` -> 8 params per layer.
    index [..., 0] -> RY angle,  index [..., 1] -> RZ angle.

MEASURED CAVEAT -- 5 angles of the LAST layer are INERT
-------------------------------------------------------
Pulling the readout back (Heisenberg) through the final ring of CNOTs gives
exactly ``U_ring^dag Z_0 U_ring = Z_1 Z_2 Z_3`` (verified against the 16x16
matrices).  Two consequences for the LAST layer only:

* the effective observable is diagonal, so it COMMUTES with the last layer's
  4 RZ rotations -> 4 dead angles;
* the effective observable acts as the IDENTITY on wire 0, so any final
  single-qubit rotation on wire 0 is invisible -> the last layer's RY on wire
  0 is a 5th dead angle.

Measured (``_scratch_check_train.py``): those 5 angles have gradient ~1e-17
and scrambling them changes the output by < 1e-16.  Effective parameter counts

    vqc1  3 of  8      vqc2  11 of 16      vqc4  27 of 32

against the nominal 8 / 16 / 32 in ``config.n_parameters``.  This is a
property of the ansatz AS SPECIFIED, not an implementation bug; it is left
untouched and reported, because changing it changes every number in the study
and that decision belongs to the researcher.  If the intent was "8 effective
angles per layer", the cheapest fixes are (a) apply RZ BEFORE RY inside the
layer -- which is what ``vqc_c`` does in qnn-benchmark -- and/or (b) put the
CNOT ring BEFORE the rotations in each layer, so the last rotations sit
between the ring and the measurement.

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
                qml.RY(weights[layer, j, 0], wires=j)
                qml.RZ(weights[layer, j, 1], wires=j)
            for j in range(config.N_QUBITS):
                qml.CNOT(wires=[j, (j + 1) % config.N_QUBITS])
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
