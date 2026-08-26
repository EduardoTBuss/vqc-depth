"""Capability tests for --diff-method and --shots.

SCOPE NOTE, on purpose: neither of these is a variable of the study.  The grid
in the paper runs with backprop on an analytic simulator, and the independent
variable is ONE -- circuit depth.  These knobs exist so that

  * the methodological claim "backprop reproduces parameter-shift" is something
    a reviewer can re-run instead of taking on faith, and
  * the shot-noise extension is one flag away, outside this paper.

The tests below therefore guard the *plumbing*, not a scientific result.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp

from vqcdepth import circuits, config
from vqcdepth.config import RunConfig

X = np.array([0.3, 1.1, 2.2, 0.7])


def _bce(p, y):
    p = pnp.clip(p, 1e-7, 1 - 1e-7)
    return -pnp.mean(y * pnp.log(p) + (1 - y) * pnp.log(1 - p))


# ---------------------------------------------------------------- gradients

@pytest.mark.parametrize("n_layers", [1, 2, 4])
def test_backprop_and_parameter_shift_give_the_same_gradient(n_layers):
    """THE claim the paper makes. If this drifts, the justification is void.

    Measured on 2026-08-25: max abs difference ~1e-16, i.e. float64 noise.
    """
    rng = np.random.default_rng(20260825)
    Xb = rng.uniform(0.0, np.pi, (16, config.N_QUBITS))
    y = pnp.array(rng.integers(0, 2, 16).astype(float), requires_grad=False)
    w0 = rng.uniform(-np.pi, np.pi, circuits.weights_shape(n_layers))

    grads = {}
    for method in ("backprop", "parameter-shift"):
        circuit = circuits.make_qnode(n_layers, diff_method=method)
        w = pnp.array(w0, requires_grad=True)

        def cost(ww):
            return _bce((circuits.eval_batch(circuit, ww, Xb) + 1.0) / 2.0, y)

        grads[method] = np.asarray(qml.grad(cost)(w), dtype=np.float64)

    diff = np.abs(grads["backprop"] - grads["parameter-shift"]).max()
    assert diff < 1e-10, (
        f"L={n_layers}: backprop and parameter-shift disagree by {diff:.3e}. "
        f"The paper's justification for using backprop rests on this test."
    )


# -------------------------------------------------------------------- shots

def test_finite_shots_are_stochastic_and_analytic_is_not():
    """Sanity: shots actually sample. Two analytic calls must be identical."""
    w = circuits.init_weights(2, np.random.default_rng(0))

    exact = circuits.make_qnode(2)
    assert float(exact(X, w)) == float(exact(X, w))

    noisy = circuits.make_qnode(2, diff_method="parameter-shift", shots=256)
    draws = np.array([float(noisy(X, w)) for _ in range(12)])
    assert draws.std() > 0.0, "shots=256 produced a constant value; not sampling"


def test_more_shots_converge_toward_the_analytic_value():
    """Shot noise must shrink with the number of shots (~1/sqrt(N))."""
    w = circuits.init_weights(2, np.random.default_rng(1))
    exact = float(circuits.make_qnode(2)(X, w))

    def rmse(shots, reps=24):
        c = circuits.make_qnode(2, diff_method="parameter-shift", shots=shots)
        d = np.array([float(c(X, w)) for _ in range(reps)])
        return float(np.sqrt(np.mean((d - exact) ** 2)))

    coarse, fine = rmse(64), rmse(4096)
    assert fine < coarse, (
        f"error did not shrink with more shots: rmse(64)={coarse:.4f} "
        f"rmse(4096)={fine:.4f}"
    )


# ------------------------------------------------------- guards and run ids

def test_backprop_with_shots_is_rejected_early():
    """There is no statevector to differentiate -- fail loudly, not silently."""
    with pytest.raises(ValueError, match="backprop"):
        circuits.make_qnode(1, diff_method="backprop", shots=100)
    with pytest.raises(ValueError, match="backprop"):
        RunConfig("banknote", "vqc1", 0, shots=100)


def test_non_default_runs_get_a_distinct_run_id():
    """THE data-integrity guard.

    Without the suffix, a shots run would overwrite the analytic run of the
    same (dataset, model, seed) and the rebuilt CSV would silently mix two
    different experiments in one row.
    """
    base = RunConfig("banknote", "vqc4", 7)
    shifted = RunConfig("banknote", "vqc4", 7, diff_method="parameter-shift")
    noisy = RunConfig("banknote", "vqc4", 7,
                      diff_method="parameter-shift", shots=1024)

    assert base.run_id == "banknote__vqc4__seed07"
    ids = {base.run_id, shifted.run_id, noisy.run_id}
    assert len(ids) == 3, f"run_ids collide: {ids}"
    assert "shots1024" in noisy.run_id


def test_classical_baseline_ignores_quantum_knobs():
    """The MLP has no circuit; its id must not pick up quantum suffixes."""
    mlp = RunConfig("banknote", "mlp", 7,
                    diff_method="parameter-shift", shots=1024)
    assert mlp.run_id == "banknote__mlp__seed07"


def test_defaults_are_the_paper_configuration():
    """Guards against someone flipping the default and shifting the study."""
    assert config.DIFF_METHOD == "backprop"
    assert config.SHOTS is None
    cfg = RunConfig("banknote", "vqc1", 0)
    assert cfg.diff_method == "backprop" and cfg.shots is None
