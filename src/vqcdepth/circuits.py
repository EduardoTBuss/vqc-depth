"""Variational quantum classifier for the circuit-depth study.

Only ``n_layers`` differs between vqc1 / vqc2 / vqc4: same angle encoding
(applied once, no data re-uploading), same layer template, same observable,
same weight layout ``(n_layers, N_QUBITS, 2)`` with ``[..., 0]`` the RZ angle
and ``[..., 1]`` the RY angle.  Feature ``j`` rides wire ``j``, readout is
``<Z_0>``, and ``p(y=1) = (1 + <Z_0>) / 2``.  Statevector labels are big-endian
in the wire index here, the opposite of Qiskit.

The layer layout -- RZ then RY, then a CNOT chain 3->2->1->0 -- keeps every
angle effective.  Pulling ``Z_0`` back through the entangling block always
yields a diagonal operator, so the rotation sitting last must not commute with
Z: an RZ there would be dead, hence RZ first and RY last.  The chain runs
toward the readout wire because a CNOT ring (``j -> (j+1) % 4``) pulls back to
``Z_1 Z_2 Z_3``, the identity on wire 0, which kills every last-layer rotation
on the only wire measured; the chain pulls back to ``Z_0 Z_1 Z_2 Z_3``.
Perturbing each angle and counting those that move the output gives
effective/nominal = 8/8, 16/16, 32/32 at L = 1, 2, 4, so the 1:2:4 parameter
progression is honest; two rotations per qubit also keep the state off the
real-amplitude subspace an RY-only ansatz is confined to.
``tests/test_ansatz.py`` pins these counts.

Simulation is analytic ``default.qubit`` in float64 (4 qubits, 16 amplitudes).
``diff_method="backprop"`` is pinned because on an ideal simulator it returns
the same gradient as parameter-shift at a fraction of the cost; parameter-shift
is only needed under shot noise or on hardware.
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
    return int(np.prod(weights_shape(n_layers)))


def make_qnode(n_layers: int, device: object | None = None,
               diff_method: str | None = None, shots: int | None = None):
    """QNode ``circuit(inputs, weights)``; ``inputs`` may be ``(N_QUBITS,)``
    or ``(batch, N_QUBITS)`` (broadcast: one device call per minibatch)."""
    if int(n_layers) < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    n_layers = int(n_layers)
    diff_method = config.DIFF_METHOD if diff_method is None else diff_method
    if shots is not None and diff_method == "backprop":
        raise ValueError(
            "backprop is invalid with finite shots: there is no statevector to "
            "differentiate. Use diff_method='parameter-shift'."
        )
    dev = device if device is not None else qml.device(
        "default.qubit", wires=config.N_QUBITS
    )

    @qml.qnode(dev, interface="autograd", diff_method=diff_method)
    def circuit(inputs, weights):
        # inputs[..., j] preserves the leading broadcast axis, if any.
        for j in range(config.N_QUBITS):
            qml.RY(inputs[..., j], wires=j)
        for layer in range(n_layers):
            for j in range(config.N_QUBITS):
                qml.RZ(weights[layer, j, 0], wires=j)
                qml.RY(weights[layer, j, 1], wires=j)
            # Chain toward the readout wire; direction is load-bearing, see the
            # module docstring.
            for j in range(config.N_QUBITS - 1, 0, -1):
                qml.CNOT(wires=[j, j - 1])
        return qml.expval(qml.PauliZ(0))

    # qml.set_shots is the supported route since PL 0.44; shots= on the device
    # is deprecated.  shots=None keeps the analytic path.
    if shots is not None:
        circuit = qml.set_shots(circuit, shots=int(shots))

    circuit.n_layers = n_layers
    # 'shots' and 'diff_method' are reserved on a QNode, hence the private names.
    circuit._vqcdepth_diff_method = diff_method
    circuit._vqcdepth_shots = shots
    return circuit


def init_weights(n_layers: int, rng: np.random.Generator) -> pnp.ndarray:
    """Uniform init in [-pi, pi] from an explicit Generator; the global NumPy
    RNG is never touched, so reproducibility does not depend on
    interpreter-wide state."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator (explicit stream)")
    w = rng.uniform(-np.pi, np.pi, size=weights_shape(n_layers))
    return pnp.array(w, requires_grad=True)


def eval_batch(circuit, weights, X):
    """Evaluate <Z_0> for the whole batch, shape ``(len(X),)``.

    Parameter broadcasting gives the same value and gradient as
    ``[circuit(x, weights) for x in X]`` (agreement ~1e-12) in one device
    execution instead of ``len(X)``.
    """
    Xb = pnp.array(np.atleast_2d(X), requires_grad=False)
    out = circuit(Xb, weights)
    return pnp.reshape(out, (Xb.shape[0],))


def predict_proba(circuit, weights, X) -> np.ndarray:
    """p(y=1) = (1 + <Z_0>) / 2, as a plain float64 numpy array."""
    raw = np.asarray(eval_batch(circuit, weights, X), dtype=np.float64)
    return (raw + 1.0) / 2.0
