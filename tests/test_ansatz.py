"""SENTINEL: every angle of the ansatz must actually do something.

READ THIS BEFORE "FIXING" A FAILURE HERE
========================================
These tests do NOT assert that the code is right or wrong in general.  They
FREEZE a measured property of the ansatz as it is specified today:

    RZ(theta) -> RY(theta) on every wire, then a CNOT chain 3->2->1->0,
    readout <Z_0>.

Both choices are deliberate, and both were picked by measurement on
2026-08-25 (see the header of ``circuits.py``):

* Pulling the readout back through the final entangling block gives a DIAGONAL
  operator (a product of Z's).  Whatever rotation sits LAST inside a layer must
  therefore NOT commute with Z -- so ``RZ`` goes first and ``RY`` goes last.
  With the opposite order (``RY`` then ``RZ``) the four ``RZ`` of the last
  layer are dead weight.
* The chain runs TOWARD the readout wire, so the pullback is
  ``Z_0 Z_1 Z_2 Z_3``, which acts non-trivially on all four wires.  A CNOT
  *ring* would give ``Z_1 Z_2 Z_3`` -- the identity on wire 0 -- killing every
  last-layer rotation on wire 0.  A chain pointing AWAY from wire 0 is worse
  still.

Measured effective (i.e. output-changing) angle counts:

    vqc1   8 of  8        vqc2  16 of 16        vqc4  32 of 32

i.e. nominal == effective at every depth, so ``config.n_parameters`` is honest
and the parameter progression is exactly 1:2:4.  That is what makes "depth" a
clean independent variable: no depth is handicapped by a degenerate final
layer.

For the record, the alternatives that were measured and rejected:

    ansatz                          L=1     L=2     L=4
    RY,RZ + ring   (first draft)    3/8    11/16   27/32
    RZ,RY + ring                    6/8    14/16   30/32
    RY only + chain 3->2->1->0      4/4     8/8    16/16
    RZ,RY + chain 3->2->1->0  <--   8/8    16/16   32/32   THIS ONE

IF THIS FILE FAILS, the ansatz changed.  Do not patch the expected numbers to
make it green: re-derive them, decide whether the change was intended, and
update BOTH this file and the header of ``circuits.py`` -- because every number
of the study moves with it.

Generated/verified on 2026-08-25 with pennylane 0.44.1, numpy 2.2.6.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from vqcdepth import config
from vqcdepth.circuits import init_weights, make_qnode, weights_shape

# (n_layers, nominal angles, effective angles)
DEPTHS = [(1, 8, 8), (2, 16, 16), (4, 32, 32)]

# A perturbation big enough that a live angle cannot hide behind float noise,
# small enough to stay away from any accidental period.
KICK = 0.9
DEAD_TOL = 1e-12

X = np.array([0.3, 1.1, 2.2, 0.7])


def _angle_sensitivities(n_layers: int) -> np.ndarray:
    """|output change| when each angle in turn is perturbed by KICK."""
    circuit = make_qnode(n_layers)
    w = np.asarray(init_weights(n_layers, np.random.default_rng(0)), dtype=float)
    base = float(circuit(X, w))
    out = np.empty(w.size)
    for i in range(w.size):
        flat = w.copy().ravel()
        flat[i] += KICK
        out[i] = abs(float(circuit(X, flat.reshape(w.shape))) - base)
    return out


@pytest.mark.parametrize("n_layers,nominal,effective", DEPTHS)
def test_every_angle_changes_the_output(n_layers, nominal, effective):
    """No dead weight: all `nominal` angles must move the output."""
    sens = _angle_sensitivities(n_layers)
    assert sens.size == nominal, (
        f"vqc{n_layers}: expected {nominal} angles, tensor has {sens.size}"
    )
    dead = np.flatnonzero(sens <= DEAD_TOL)
    assert dead.size == 0, (
        f"vqc{n_layers}: {dead.size} angle(s) do nothing (flat indices "
        f"{dead.tolist()}). The ansatz regressed to a degenerate final layer -- "
        f"see this file's docstring before touching anything."
    )
    assert sens.size == effective


@pytest.mark.parametrize("n_layers,nominal,_effective", DEPTHS)
def test_nominal_count_matches_config_and_is_honest(n_layers, nominal, _effective):
    """config.n_parameters may be quoted in the paper as-is."""
    model = {1: "vqc1", 2: "vqc2", 4: "vqc4"}[n_layers]
    assert config.n_parameters(model) == nominal
    assert int(np.prod(weights_shape(n_layers))) == nominal


@pytest.mark.parametrize("n_layers,nominal,_effective", DEPTHS)
def test_every_angle_has_a_nonzero_gradient(n_layers, nominal, _effective):
    """Sensitivity and gradient must agree: no angle is flat at the init point."""
    from pennylane import numpy as pnp

    circuit = make_qnode(n_layers)
    w = pnp.array(np.asarray(init_weights(n_layers, np.random.default_rng(0)),
                             dtype=float), requires_grad=True)
    g = np.asarray(qml.grad(lambda ww: circuit(X, ww))(w), dtype=float).ravel()
    assert g.size == nominal
    flat = np.flatnonzero(np.abs(g) <= 1e-14)
    assert flat.size == 0, (
        f"vqc{n_layers}: angle(s) {flat.tolist()} have a vanishing gradient; "
        f"they cannot be trained."
    )


def test_cnot_chain_maps_Z0_to_Z0Z1Z2Z3():
    """The algebraic fact behind all of the above, on the 16x16 matrices.

    The chain 3->2->1->0 pulls the readout back to Z_0 Z_1 Z_2 Z_3, which acts
    non-trivially on EVERY wire -- that is why no rotation is invisible.

    If this fails, the entanglement pattern itself changed (direction of the
    chain, order of the CNOTs, or the readout wire).
    """
    n = config.N_QUBITS
    ops = [qml.CNOT(wires=[j, j - 1]) for j in range(n - 1, 0, -1)]
    U = np.eye(2 ** n, dtype=complex)
    for op in ops:                                   # circuit order: left to right
        U = np.asarray(qml.matrix(op, wire_order=range(n))) @ U

    Z0 = np.asarray(qml.matrix(qml.PauliZ(0), wire_order=range(n)))
    Zall = np.asarray(qml.matrix(
        qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3),
        wire_order=range(n)))

    propagated = U.conj().T @ Z0 @ U
    np.testing.assert_allclose(propagated, Zall, rtol=0, atol=1e-12)


def test_a_cnot_ring_would_kill_wire_zero():
    """Documents WHY the ring was rejected: its pullback is identity on wire 0.

    Kept as an executable explanation, so the reasoning cannot silently rot.
    """
    n = config.N_QUBITS
    U = np.eye(2 ** n, dtype=complex)
    for j in range(n):
        U = np.asarray(qml.matrix(qml.CNOT(wires=[j, (j + 1) % n]),
                                  wire_order=range(n))) @ U

    Z0 = np.asarray(qml.matrix(qml.PauliZ(0), wire_order=range(n)))
    Z123 = np.asarray(qml.matrix(
        qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3), wire_order=range(n)))

    np.testing.assert_allclose(U.conj().T @ Z0 @ U, Z123, rtol=0, atol=1e-12)
