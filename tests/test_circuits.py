"""Circuit contract: batched evaluation, gradients, parameter counts.

What this file protects
-----------------------
The whole grid runs on ``circuits.eval_batch`` (PennyLane parameter
broadcasting).  If broadcasting ever stopped agreeing with the naive
"one circuit call per sample" form -- in VALUE or in GRADIENT -- every number
in the paper would be wrong and nothing else would complain.  So the naive
loop is kept here, forever, as the reference model.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp

from vqcdepth import circuits, config, models, train

LAYERS = (1, 2, 4)
SEED = 20260825

# Batch/gradient agreement is measured at ~1e-16 (float64 round-off).  1e-10 is
# six orders of magnitude of slack: tight enough to catch a real divergence,
# loose enough never to flake on another BLAS/CPU.
TOL = 1e-10


def _loop_eval(circuit, weights, X):
    """Reference model: the obvious, slow, one-sample-at-a-time evaluation."""
    return pnp.stack([circuit(pnp.array(x, requires_grad=False), weights)
                      for x in X])


def _bce(p, y):
    p = pnp.clip(p, 1e-7, 1.0 - 1e-7)
    return -pnp.mean(y * pnp.log(p) + (1.0 - y) * pnp.log(1.0 - p))


@pytest.fixture(scope="module")
def batch():
    rng = np.random.default_rng(SEED)
    X = rng.uniform(0.0, np.pi, (config.BATCH_SIZE, config.N_QUBITS))
    y = pnp.array(rng.integers(0, 2, config.BATCH_SIZE).astype(float),
                  requires_grad=False)
    return X, y


# ======================================================================
# 1. vectorisation == loop, in value AND in gradient
# ======================================================================
@pytest.mark.parametrize("n_layers", LAYERS)
def test_eval_batch_value_equals_per_sample_loop(batch, n_layers):
    X, _ = batch
    circuit = circuits.make_qnode(n_layers)
    w = circuits.init_weights(n_layers, np.random.default_rng(SEED + n_layers))

    v_loop = np.asarray(_loop_eval(circuit, w, X), dtype=np.float64)
    v_batch = np.asarray(circuits.eval_batch(circuit, w, X), dtype=np.float64)

    err = float(np.max(np.abs(v_batch - v_loop)))
    assert err < TOL, (
        f"L={n_layers}: broadcasting disagrees with the per-sample loop in VALUE "
        f"by {err:.3e} (> {TOL:.0e}). Every expectation value of the grid comes "
        f"from the broadcast path -- do not run the grid until this is explained."
    )


@pytest.mark.parametrize("n_layers", LAYERS)
def test_eval_batch_gradient_equals_per_sample_loop(batch, n_layers):
    """The optimiser only ever sees the broadcast gradient; pin it to the loop."""
    X, y = batch
    circuit = circuits.make_qnode(n_layers)
    w = circuits.init_weights(n_layers, np.random.default_rng(SEED + n_layers))

    def cost_loop(ww):
        return _bce((_loop_eval(circuit, ww, X) + 1.0) / 2.0, y)

    def cost_batch(ww):
        return _bce((circuits.eval_batch(circuit, ww, X) + 1.0) / 2.0, y)

    d_loss = abs(float(cost_batch(w)) - float(cost_loop(w)))
    g_loop = np.asarray(qml.grad(cost_loop)(w), dtype=np.float64)
    g_batch = np.asarray(qml.grad(cost_batch)(w), dtype=np.float64)
    d_grad = float(np.max(np.abs(g_batch - g_loop)))

    assert d_loss < TOL, f"L={n_layers}: batched loss differs by {d_loss:.3e}"
    assert d_grad < TOL, (
        f"L={n_layers}: broadcasting disagrees with the per-sample loop in "
        f"GRADIENT by {d_grad:.3e} (> {TOL:.0e}); |g|max={np.max(np.abs(g_loop)):.3e}. "
        f"Training would silently follow the wrong direction."
    )


@pytest.mark.parametrize("n_layers", LAYERS)
def test_backprop_gradient_matches_central_finite_differences(n_layers):
    """Gradient check: analytic backprop vs (f(x+h) - f(x-h)) / 2h.

    Only the ACTIVE angles are scored: the inert ones (see ``test_ansatz.py``)
    have an exactly zero analytic gradient, so their relative error would be
    0/0 noise.
    """
    rng = np.random.default_rng(SEED)
    X = rng.uniform(0.0, np.pi, (8, config.N_QUBITS))
    y = pnp.array(rng.integers(0, 2, 8).astype(float), requires_grad=False)
    circuit = circuits.make_qnode(n_layers)
    w = circuits.init_weights(n_layers, np.random.default_rng(SEED + n_layers))

    def cost(ww):
        return _bce((circuits.eval_batch(circuit, ww, X) + 1.0) / 2.0, y)

    g_ana = np.asarray(qml.grad(cost)(w), dtype=np.float64)

    eps = 1e-6
    flat = np.asarray(w, dtype=np.float64)
    g_num = np.zeros_like(g_ana)
    for ix in np.ndindex(*flat.shape):
        wp, wm = flat.copy(), flat.copy()
        wp[ix] += eps
        wm[ix] -= eps
        g_num[ix] = (float(cost(pnp.array(wp, requires_grad=True)))
                     - float(cost(pnp.array(wm, requires_grad=True)))) / (2 * eps)

    # MIXED tolerance, not purely relative.  Central differences with eps=1e-6
    # carry an absolute error floor of ~1e-10 in float64, so dividing by a
    # genuinely small gradient blows the relative error up while the gradient
    # is perfectly correct.  Since the ansatz was fixed (2026-08-25) EVERY
    # angle is live, and some are legitimately small -- hence atol.
    # Measured worst case over L=1,2,4: 1.2e-10 absolute, ~100x inside atol.
    atol, rtol = 1e-8, 1e-6
    err = np.abs(g_ana - g_num)
    bound = atol + rtol * np.abs(g_ana)
    worst = int(np.argmax(err - bound))
    assert np.all(err <= bound), (
        f"L={n_layers}: analytic gradient disagrees with central finite "
        f"differences. Worst angle (flat index {worst}): analytic="
        f"{g_ana.ravel()[worst]:.6e} numeric={g_num.ravel()[worst]:.6e} "
        f"abs_err={err.ravel()[worst]:.3e}"
    )


@pytest.mark.parametrize("n_layers", LAYERS)
def test_backprop_gradient_matches_parameter_shift(n_layers):
    """backprop is pinned for speed; on an ideal simulator it must stay exact."""
    rng = np.random.default_rng(SEED + 1)
    X = rng.uniform(0.0, np.pi, (6, config.N_QUBITS))
    y = pnp.array(rng.integers(0, 2, 6).astype(float), requires_grad=False)

    c_bp = circuits.make_qnode(n_layers)
    dev = qml.device("default.qubit", wires=config.N_QUBITS, shots=None)
    c_ps = qml.QNode(c_bp.func, dev, interface="autograd",
                     diff_method="parameter-shift")
    w = circuits.init_weights(n_layers, np.random.default_rng(SEED + n_layers))

    def make_cost(c):
        return lambda ww: _bce((circuits.eval_batch(c, ww, X) + 1.0) / 2.0, y)

    g_bp = np.asarray(qml.grad(make_cost(c_bp))(w), dtype=np.float64)
    g_ps = np.asarray(qml.grad(make_cost(c_ps))(w), dtype=np.float64)
    err = float(np.max(np.abs(g_bp - g_ps)))
    assert err < 1e-9, (
        f"L={n_layers}: backprop and parameter-shift gradients differ by "
        f"{err:.3e}; on an analytic simulator they must be identical."
    )


# ======================================================================
# 2. analytic anchor
# ======================================================================
def test_zero_weights_one_layer_matches_closed_form():
    """With all angles zero and L=1, <Z_0> = prod_j cos(x_j), exactly.

    Derivation: the CNOT chain 3->2->1->0 maps Z_0 -> Z_0 Z_1 Z_2 Z_3
    (Heisenberg), the encoded state is a product state, and <Z_j> = cos(x_j)
    for RY(x_j)|0>.  ALL FOUR features appear -- including x_0.  (With a CNOT
    ring the pullback would be Z_1 Z_2 Z_3 and the readout qubit's own feature
    would be invisible at depth 1; that is one more reason the chain was
    chosen.)  Anchors wire order, rotation sense and chain direction at once.
    """
    circuit = circuits.make_qnode(1)
    w = pnp.array(np.zeros(circuits.weights_shape(1)), requires_grad=True)
    X = np.random.default_rng(4242).uniform(0.0, np.pi, (16, config.N_QUBITS))

    got = np.asarray(circuits.eval_batch(circuit, w, X), dtype=np.float64)
    expected = np.prod(np.cos(X), axis=1)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_predict_proba_is_affine_image_of_expectation_and_in_unit_interval():
    circuit = circuits.make_qnode(2)
    w = circuits.init_weights(2, np.random.default_rng(7))
    X = np.random.default_rng(8).uniform(0.0, np.pi, (32, config.N_QUBITS))

    z = np.asarray(circuits.eval_batch(circuit, w, X), dtype=np.float64)
    p = circuits.predict_proba(circuit, w, X)

    np.testing.assert_allclose(p, (z + 1.0) / 2.0, rtol=0, atol=1e-15)
    assert p.dtype == np.float64
    assert np.all((p >= 0.0) & (p <= 1.0)), f"p out of [0,1]: {p.min()}, {p.max()}"


# ======================================================================
# 3. edge cases of the batching helper
# ======================================================================
def test_eval_batch_accepts_a_single_sample_and_still_returns_1d():
    circuit = circuits.make_qnode(1)
    w = circuits.init_weights(1, np.random.default_rng(3))
    x = np.full(config.N_QUBITS, 0.7)

    out = circuits.eval_batch(circuit, w, x)
    assert np.shape(out) == (1,), f"single sample gave shape {np.shape(out)}"
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float64),
        np.asarray(circuits.eval_batch(circuit, w, x[None, :]), dtype=np.float64),
        rtol=0, atol=0,
    )


def test_eval_batch_row_order_is_preserved():
    """A silent row permutation would pair every prediction with the wrong label."""
    circuit = circuits.make_qnode(2)
    w = circuits.init_weights(2, np.random.default_rng(5))
    X = np.random.default_rng(6).uniform(0.0, np.pi, (12, config.N_QUBITS))
    perm = np.array([7, 0, 11, 3, 5, 1, 9, 2, 10, 4, 8, 6])

    ref = np.asarray(circuits.eval_batch(circuit, w, X), dtype=np.float64)
    got = np.asarray(circuits.eval_batch(circuit, w, X[perm]), dtype=np.float64)
    np.testing.assert_allclose(got, ref[perm], rtol=0, atol=1e-14)


def test_make_qnode_rejects_depth_below_one():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            circuits.make_qnode(bad)


def test_init_weights_refuses_implicit_global_randomness():
    with pytest.raises(TypeError):
        circuits.init_weights(1, 12345)                       # a seed, not a stream
    with pytest.raises(TypeError):
        circuits.init_weights(1, np.random.RandomState(0))    # legacy API


def test_init_weights_is_a_pure_function_of_the_generator():
    a = np.asarray(circuits.init_weights(2, np.random.default_rng(99)), dtype=float)
    b = np.asarray(circuits.init_weights(2, np.random.default_rng(99)), dtype=float)
    np.testing.assert_array_equal(a, b)
    assert a.shape == circuits.weights_shape(2)
    assert np.all(np.abs(a) <= np.pi)


def test_init_weights_never_consumes_the_global_numpy_stream():
    np.random.seed(0)
    before = np.random.random()
    np.random.seed(0)
    circuits.init_weights(4, np.random.default_rng(1))
    after = np.random.random()
    assert before == after, "init_weights consumed the global legacy RNG"


# ======================================================================
# 4. parameter counts (nominal)
# ======================================================================
@pytest.mark.parametrize("model,n_layers,expected", [("vqc1", 1, 8),
                                                     ("vqc2", 2, 16),
                                                     ("vqc4", 4, 32)])
def test_vqc_parameter_count_agrees_with_config_and_with_the_tensor(
        model, n_layers, expected):
    assert config.VQC_LAYERS[model] == n_layers
    assert config.n_parameters(model) == expected, "config.n_parameters drifted"
    assert circuits.n_parameters(n_layers) == expected
    assert circuits.weights_shape(n_layers) == (n_layers, config.N_QUBITS, 2)

    w = circuits.init_weights(n_layers, np.random.default_rng(0))
    assert np.asarray(w).size == expected, (
        f"{model}: weight tensor holds {np.asarray(w).size} angles but "
        f"config.n_parameters says {expected}"
    )


def test_mlp_parameter_count_agrees_with_config_and_with_the_module():
    expected = 81   # 4*8+8 + 8*4+4 + 4*1+1
    assert config.n_parameters("mlp") == expected
    mlp = models.build_mlp(train.torch_generator("banknote", "mlp", 0))
    assert models.n_parameters(mlp) == expected
    assert sum(p.numel() for p in mlp.parameters()) == expected


def test_config_rejects_an_unknown_model_instead_of_guessing():
    with pytest.raises(ValueError):
        config.n_parameters("vqc8")
