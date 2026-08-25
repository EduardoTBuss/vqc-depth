"""SENTINEL: how many angles of the ansatz actually do anything.

READ THIS BEFORE "FIXING" A FAILURE HERE
========================================
These tests do NOT assert that the code is right or wrong.  They FREEZE a
measured property of the ansatz as it is specified today:

    RY(theta) -> RZ(theta) on every wire, then a ring of CNOTs, readout <Z_0>.

Pulling the readout back through the FINAL ring gives, exactly,

    U_ring^dagger Z_0 U_ring = Z_1 Z_2 Z_3

so for the LAST layer only:
  * the effective observable is diagonal and therefore COMMUTES with the four
    RZ rotations of that layer  -> 4 angles do nothing;
  * the effective observable is the IDENTITY on wire 0, so the last layer's RY
    on wire 0 is invisible      -> a 5th angle does nothing.

Effective (i.e. output-changing) angle counts, measured:

    vqc1   3 of  8        vqc2  11 of 16        vqc4  27 of 32

This is a property of the ANSATZ, not a bug in the implementation, and the
paper must report the effective counts next to the nominal 8/16/32.

IF THIS FILE FAILS, the ansatz changed.  Do not patch the expected numbers to
make it green: re-derive them, decide whether the change was intended, and
update BOTH this file and the "MEASURED CAVEAT" section of ``circuits.py`` --
because every number of the study moves with it.

Generated/verified on 2026-08-25 with pennylane 0.44.1, numpy 2.2.6.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp

from vqcdepth import circuits, config

# (n_layers, nominal angles, EFFECTIVE angles)
EFFECTIVE = [(1, 8, 3), (2, 16, 11), (4, 32, 27)]

PERTURBATION = 0.9      # rad; large enough that any live angle moves the output
DEAD_TOL = 1e-12        # measured dead-angle response is < 1e-16


def _probe_inputs(n: int = 8, seed: int = 11) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.0, np.pi, (n, config.N_QUBITS))


def _response_per_angle(n_layers: int, seed: int) -> np.ndarray:
    """max_x |f(x; theta + 0.9 e_k) - f(x; theta)| for every angle k."""
    circuit = circuits.make_qnode(n_layers)
    X = _probe_inputs(seed=seed)
    w0 = np.asarray(circuits.init_weights(n_layers, np.random.default_rng(seed)),
                    dtype=np.float64)
    base = circuits.predict_proba(circuit, pnp.array(w0), X)

    resp = np.zeros(w0.size)
    for k, ix in enumerate(np.ndindex(*w0.shape)):
        w = w0.copy()
        w[ix] += PERTURBATION
        moved = circuits.predict_proba(circuit, pnp.array(w), X)
        resp[k] = float(np.max(np.abs(moved - base)))
    return resp.reshape(w0.shape)


@pytest.mark.parametrize("n_layers,nominal,effective", EFFECTIVE)
def test_number_of_effective_angles_is_the_documented_one(n_layers, nominal, effective):
    """Counts how many angles move the output when perturbed by 0.9 rad."""
    resp = _response_per_angle(n_layers, seed=n_layers)
    live = resp > DEAD_TOL
    n_live = int(live.sum())

    assert resp.size == nominal
    assert n_live == effective, (
        f"THE ANSATZ CHANGED. vqc{n_layers} now has {n_live} effective angles, "
        f"the recorded value is {effective} of {nominal}. Per-angle response:\n"
        f"{np.array2string(resp, precision=3)}\n"
        f"Re-derive the count, update this test AND the caveat in circuits.py, "
        f"and remember that every result in the paper moves with the ansatz."
    )


@pytest.mark.parametrize("n_layers,nominal,effective", EFFECTIVE)
def test_the_dead_angles_are_exactly_the_five_documented_ones(n_layers, nominal,
                                                              effective):
    """Not just HOW MANY are dead, but WHICH: last layer RZ (x4) + RY on wire 0."""
    resp = _response_per_angle(n_layers, seed=n_layers + 100)
    dead = resp <= DEAD_TOL

    expected_dead = np.zeros_like(dead)
    expected_dead[n_layers - 1, :, 1] = True     # the 4 RZ of the last layer
    expected_dead[n_layers - 1, 0, 0] = True     # the last layer's RY on wire 0

    assert np.array_equal(dead, expected_dead), (
        f"vqc{n_layers}: the set of inert angles moved.\n"
        f"dead now (layer, wire, RY=0/RZ=1):\n{np.argwhere(dead).tolist()}\n"
        f"expected:\n{np.argwhere(expected_dead).tolist()}"
    )
    assert int(dead.sum()) == 5
    assert nominal - int(dead.sum()) == effective


@pytest.mark.parametrize("n_layers,_nominal,_effective", EFFECTIVE)
def test_inert_angles_also_have_an_exactly_zero_gradient(n_layers, _nominal,
                                                         _effective):
    """A dead angle must not merely be flat locally: its gradient is 0 everywhere.

    Consequence for the study: the optimiser never updates those 5 angles, so
    their initial values persist to the end of training.
    """
    X = _probe_inputs(seed=n_layers + 7)
    circuit = circuits.make_qnode(n_layers)
    w = circuits.init_weights(n_layers, np.random.default_rng(n_layers + 7))

    def cost(ww):
        return pnp.mean(circuits.eval_batch(circuit, ww, X))

    g = np.asarray(qml.grad(cost)(w), dtype=np.float64)
    dead_grad = np.concatenate([g[n_layers - 1, :, 1], [g[n_layers - 1, 0, 0]]])
    assert np.max(np.abs(dead_grad)) < 1e-14, (
        f"vqc{n_layers}: an angle documented as inert has a non-zero gradient "
        f"{dead_grad}; the ansatz or the observable changed."
    )


def test_ring_of_cnots_maps_Z0_to_Z1Z2Z3():
    """The algebraic fact behind all of the above, checked on the 16x16 matrices.

    If this one fails, the entanglement pattern itself changed (direction of the
    ring, order of the CNOTs, or the readout wire).
    """
    n = config.N_QUBITS
    ops = [qml.CNOT(wires=[j, (j + 1) % n]) for j in range(n)]
    U = np.eye(2 ** n, dtype=complex)
    for op in ops:                                   # circuit order: left to right
        U = qml.matrix(op, wire_order=range(n)) @ U

    Z0 = qml.matrix(qml.PauliZ(0), wire_order=range(n))
    Z123 = qml.matrix(qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3),
                      wire_order=range(n))

    propagated = U.conj().T @ Z0 @ U
    np.testing.assert_allclose(propagated, Z123, rtol=0, atol=1e-12)
