"""Training loop: test-set isolation, model selection, seeding, GOLDEN run.

Two things here can invalidate the whole study on their own:

1. the test partition leaking into training or into model selection;
2. a silent change of ansatz / optimizer / RNG stream that shifts every number
   without anybody noticing.

(1) is checked by re-running the identical training with a DELIBERATELY
corrupted test partition and demanding bit-identical results.  (2) is checked
by the golden mini-run at the bottom of the file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import helpers
from vqcdepth import circuits, config, models, train

GOLDEN_SPLIT = dict(dataset="banknote", seed=7, n_train=96, n_val=48, n_test=48)


@pytest.fixture(scope="module")
def split():
    return helpers.synthetic_split(**GOLDEN_SPLIT)


def _cfg(model: str, n_epochs: int = 5, **kw):
    return config.RunConfig(dataset=GOLDEN_SPLIT["dataset"], model=model,
                            seed=GOLDEN_SPLIT["seed"], n_epochs=n_epochs,
                            save_checkpoints=False, **kw)


def _history_tuples(out):
    """Everything the loop recorded, as plain comparable numbers."""
    return [(r.epoch, r.train_loss, r.train_accuracy, r.train_f1,
             r.val_loss, r.val_accuracy, r.val_f1, r.val_roc_auc,
             r.gradient_norm, r.parameter_norm) for r in out.history]


# ======================================================================
# 1. THE TEST SET IS NEVER TOUCHED
# ======================================================================
@pytest.mark.parametrize("model", ["vqc1", "mlp"])
def test_training_is_bit_identical_when_the_test_partition_is_garbage(split, model):
    """``train_model`` receives the whole Split, including X_test/y_test.

    Train once normally, then again on a Split whose test partition was
    replaced by uniform noise with flipped labels.  Loss curves, best epoch and
    best validation F1 must be IDENTICAL to the last bit.  Any peek at the test
    set -- for early stopping, for normalisation, even for a printout that
    consumed a random number -- breaks this.
    """
    ref = train.train_model(_cfg(model), split)
    poisoned = train.train_model(_cfg(model), helpers.poison_test_partition(split))

    assert _history_tuples(poisoned) == _history_tuples(ref), (
        f"{model}: the training history changed when ONLY the test partition "
        f"was corrupted -- the test set is influencing training or model "
        f"selection. This invalidates every test-set number in the paper."
    )
    assert poisoned.best_epoch == ref.best_epoch
    assert poisoned.best_val_f1 == ref.best_val_f1
    assert poisoned.best_val_loss == ref.best_val_loss


def test_the_epoch_records_contain_no_test_metric(split):
    out = train.train_model(_cfg("vqc1", n_epochs=2), split)
    fields = set(vars(out.history[0]))
    assert not any("test" in f for f in fields), (
        f"an EpochRecord field mentions the test set: {sorted(fields)}"
    )


# ======================================================================
# 2. model selection uses validation only, and is self-consistent
# ======================================================================
@pytest.mark.parametrize("model", ["vqc4", "mlp"])
def test_best_epoch_is_the_validation_argmax_with_the_documented_tie_break(
        split, model):
    """Reference implementation of the selection rule: highest val_f1, ties
    broken by lowest val_loss, first epoch wins."""
    out = train.train_model(_cfg(model), split)

    best_ep, best_f1, best_loss = -1, -math.inf, math.inf
    for r in out.history:
        if r.val_f1 > best_f1 or (r.val_f1 == best_f1 and r.val_loss < best_loss):
            best_ep, best_f1, best_loss = r.epoch, r.val_f1, r.val_loss

    assert out.best_epoch == best_ep, (
        f"{model}: best_epoch={out.best_epoch}, the rule says {best_ep}"
    )
    assert out.best_val_f1 == best_f1 == max(r.val_f1 for r in out.history)
    assert out.best_val_loss == best_loss


@pytest.mark.parametrize("model", ["vqc2", "mlp"])
def test_best_params_reproduce_the_recorded_best_validation_f1(split, model):
    """The snapshot must be the parameters OF the best epoch, not of the last."""
    from sklearn.metrics import f1_score

    out = train.train_model(_cfg(model), split)
    if model == "mlp":
        proba = models.predict_proba_mlp(out.best_params, split.X_val)
    else:
        proba = circuits.predict_proba(circuits.make_qnode(config.VQC_LAYERS[model]),
                                       out.best_params, split.X_val)
    f1 = f1_score(split.y_val, (proba >= 0.5).astype(int), zero_division=0)
    assert f1 == pytest.approx(out.best_val_f1, rel=0, abs=1e-12), (
        f"{model}: best_params scores {f1:.6f} on validation but the run "
        f"recorded {out.best_val_f1:.6f} -- the wrong epoch was snapshotted"
    )


def test_every_epoch_is_recorded_and_the_diagnostics_are_finite(split):
    out = train.train_model(_cfg("vqc2", n_epochs=4), split)
    assert [r.epoch for r in out.history] == [1, 2, 3, 4]
    assert out.n_parameters == config.n_parameters("vqc2") == 16
    for r in out.history:
        assert math.isfinite(r.train_loss) and math.isfinite(r.val_loss)
        assert 0.0 <= r.train_accuracy <= 1.0 and 0.0 <= r.val_accuracy <= 1.0
        assert math.isfinite(r.gradient_norm) and r.gradient_norm >= 0.0
        assert math.isfinite(r.parameter_norm) and r.parameter_norm > 0.0
        assert r.elapsed_seconds >= 0.0
    tb = out.timing_breakdown
    assert tb["n_steps"] == math.ceil(96 / config.BATCH_SIZE) * 4


@pytest.mark.parametrize("model", list(config.MODELS))
def test_training_actually_learns_on_a_separable_toy_problem(split, model):
    """Overfit-a-tiny-batch smoke test: if the loss does not go down, the sign
    of the gradient or the optimizer wiring is wrong."""
    out = train.train_model(_cfg(model), split)
    assert out.history[-1].train_loss < out.history[0].train_loss, (
        f"{model}: training loss did not decrease "
        f"({out.history[0].train_loss:.4f} -> {out.history[-1].train_loss:.4f})"
    )


# ======================================================================
# 3. seeding: same batches for every model, different initialisations
# ======================================================================
def test_batch_order_is_shared_by_all_four_models_of_a_seed():
    """Data order is held FIXED across models: the model is the only variable."""
    orders = {m: train.batch_rng("banknote", 3).permutation(50).tolist()
              for m in config.MODELS}
    ref = orders[config.MODELS[0]]
    for m, o in orders.items():
        assert o == ref, f"{m} would see a different minibatch order"


def test_batch_order_changes_with_dataset_and_with_seed():
    a = train.batch_rng("banknote", 0).permutation(50).tolist()
    assert a != train.batch_rng("banknote", 1).permutation(50).tolist()
    assert a != train.batch_rng("blood", 0).permutation(50).tolist()


def test_initialisation_streams_differ_between_models():
    """Sharing the stream would make vqc1's angles a prefix of vqc2's."""
    draws = {m: train.init_rng("banknote", m, 0).random(8).tolist()
             for m in config.MODELS}
    assert len({tuple(v) for v in draws.values()}) == len(config.MODELS), (
        f"two models share an initialisation stream: {draws}"
    )


def test_stable_key_is_deterministic_and_not_the_builtin_hash():
    a = train.stable_key("init", "banknote", "vqc4")
    assert a == train.stable_key("init", "banknote", "vqc4")
    assert a != train.stable_key("init", "banknote", "vqc1")
    assert a != hash(("init", "banknote", "vqc4"))
    assert 0 <= a < 2 ** 64
    # the separator must make the parts unambiguous
    assert train.stable_key("ab", "c") != train.stable_key("a", "bc")


def test_torch_generator_is_reproducible_and_matches_the_init_stream():
    g1 = train.torch_generator("banknote", "mlp", 4)
    g2 = train.torch_generator("banknote", "mlp", 4)
    a = models.build_mlp(g1)
    b = models.build_mlp(g2)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)
    c = models.build_mlp(train.torch_generator("banknote", "mlp", 5))
    assert not torch.equal(next(a.parameters()), next(c.parameters()))


def test_mlp_weights_do_not_depend_on_the_global_torch_rng():
    """Reproducibility must come from the explicit Generator only.

    NOTE (measured 2026-08-25): building the module DOES consume the global
    torch stream, because ``nn.Linear.__init__`` calls ``reset_parameters``
    before ``init_parameters`` overwrites everything.  That is a side effect,
    not a reproducibility hazard -- the weights that survive come exclusively
    from our generator, which is what this test pins.  It would only start to
    matter if some other component of the project ever relied on the global
    torch RNG.
    """
    torch.manual_seed(0)
    a = models.build_mlp(train.torch_generator("banknote", "mlp", 0))
    torch.manual_seed(999)
    torch.rand(17)                      # desynchronise the global stream
    b = models.build_mlp(train.torch_generator("banknote", "mlp", 0))
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)


# ======================================================================
# 4. loss, guards and error paths
# ======================================================================
def test_bce_matches_the_closed_form_and_is_clipped_at_the_extremes():
    p = np.array([0.9, 0.2, 0.5])
    y = np.array([1, 0, 1])
    expected = -np.mean([np.log(0.9), np.log(0.8), np.log(0.5)])
    assert train.bce_numpy(p, y) == pytest.approx(expected, rel=1e-12)

    # a confidently wrong prediction must give a large but FINITE loss
    hard = train.bce_numpy(np.array([0.0, 1.0]), np.array([1, 0]))
    assert math.isfinite(hard) and hard > 10.0


def test_train_model_refuses_a_split_that_belongs_to_another_cell(split):
    with pytest.raises(ValueError, match="split"):
        train.train_model(_cfg("vqc1", n_epochs=1),
                          helpers.synthetic_split(dataset="blood", seed=7,
                                                  n_train=32, n_val=16, n_test=16))
    with pytest.raises(ValueError, match="seed"):
        train.train_model(_cfg("vqc1", n_epochs=1),
                          helpers.synthetic_split(dataset="banknote", seed=1,
                                                  n_train=32, n_val=16, n_test=16))


def test_train_model_refuses_a_feature_qubit_mismatch():
    import dataclasses

    sp = helpers.synthetic_split("banknote", 7, 32, 16, 16)
    bad = dataclasses.replace(sp, X_train=np.zeros((32, 3)))
    with pytest.raises(ValueError, match="qubit"):
        train.train_model(_cfg("vqc1", n_epochs=1), bad)


# ======================================================================
# 5. checkpoints are artefacts, written atomically
# ======================================================================
def test_checkpoints_land_on_disk_complete_and_without_temporary_files(
        split, tmp_path):
    cfg = config.RunConfig(dataset="banknote", model="vqc1", seed=7,
                           n_epochs=4, checkpoint_every=2)
    out = train.train_model(cfg, split, checkpoint_dir=tmp_path)

    names = sorted(p.name for p in tmp_path.iterdir())
    assert not any(n.endswith(".tmp") for n in names), f"temp file left: {names}"
    assert f"{cfg.run_id}__epoch002.npz" in names
    assert f"{cfg.run_id}__epoch004.npz" in names
    assert f"{cfg.run_id}__best.npz" in names

    w = np.load(tmp_path / f"{cfg.run_id}__best.npz")["weights"]
    assert w.shape == circuits.weights_shape(1)

    meta = json.loads((tmp_path / f"{cfg.run_id}__best.json").read_text("utf-8"))
    assert meta["run_id"] == cfg.run_id
    assert meta["best_epoch"] == out.best_epoch
    assert meta["best_val_f1"] == pytest.approx(out.best_val_f1, rel=0, abs=0)
    assert meta["n_parameters"] == 8


def test_no_checkpoint_directory_means_no_disk_writes(split, tmp_path):
    train.train_model(_cfg("vqc1", n_epochs=2), split, checkpoint_dir=None)
    assert list(tmp_path.iterdir()) == []


# ======================================================================
# 6. GOLDEN MINI-RUN
# ======================================================================
# Generated on 2026-08-25 by RUNNING the code below on:
#   pennylane 0.44.1, torch 2.7.1+cpu, numpy 2.2.6, scikit-learn 1.7.0, Windows.
# Setup: helpers.synthetic_split("banknote", seed=7, 96/48/48), 5 full epochs,
# batch 16, Adam lr 0.05, betas (0.9, 0.999), eps 1e-8.
#
# THESE NUMBERS ARE A TRIPWIRE, NOT A REQUIREMENT.  They will (and must) change
# if the ansatz, the optimizer, the seeding scheme or the epoch definition
# changes.  When that happens ON PURPOSE: regenerate them by running this test
# and pasting the reported values, note the date, and remember that any grid
# already on disk was produced by the OLD code and cannot be mixed with the new.
GOLDEN = {
    #        best_epoch, best_val_f1,         best_val_loss,       final_train_loss
    "vqc1": (3, 0.9,                0.5276245304990551, 0.521374767196214),
    "vqc2": (5, 0.7272727272727273, 0.5109093703150295, 0.37340822396420953),
    "vqc4": (5, 0.9433962264150944, 0.39346800293961465, 0.30969368453798296),
    "mlp":  (5, 0.5641025641025641, 0.6110279803912959, 0.4843255658937544),
}

# Tolerance: the arithmetic is deterministic on a given machine, but BLAS and
# CPU differences can move the last couple of bits.  1e-9 relative is ~7 orders
# of magnitude tighter than any real change of ansatz/optimizer would produce.
GOLDEN_RTOL = 1e-9


@pytest.mark.parametrize("model", list(GOLDEN))
def test_golden_mini_run_is_unchanged(split, model):
    best_epoch, best_f1, best_loss, final_train_loss = GOLDEN[model]
    out = train.train_model(_cfg(model, n_epochs=5), split)

    got = (out.best_epoch, out.best_val_f1, out.best_val_loss,
           out.history[-1].train_loss)
    msg = (
        f"GOLDEN MINI-RUN CHANGED for {model}.\n"
        f"  expected: best_epoch={best_epoch} best_val_f1={best_f1!r} "
        f"best_val_loss={best_loss!r} final_train_loss={final_train_loss!r}\n"
        f"  got:      best_epoch={got[0]} best_val_f1={got[1]!r} "
        f"best_val_loss={got[2]!r} final_train_loss={got[3]!r}\n"
        f"Something about the ansatz, the optimizer, the RNG streams or the "
        f"epoch definition moved. If it was intentional, regenerate GOLDEN in "
        f"tests/test_train.py with the values above and record today's date."
    )
    assert out.best_epoch == best_epoch, msg
    assert out.best_val_f1 == pytest.approx(best_f1, rel=GOLDEN_RTOL), msg
    assert out.best_val_loss == pytest.approx(best_loss, rel=GOLDEN_RTOL), msg
    assert out.history[-1].train_loss == pytest.approx(
        final_train_loss, rel=GOLDEN_RTOL), msg


def test_repeating_the_golden_run_in_the_same_process_is_bit_identical(split):
    """No hidden global state between two runs of the same cell."""
    a = train.train_model(_cfg("vqc2"), split)
    b = train.train_model(_cfg("vqc2"), split)
    assert _history_tuples(a) == _history_tuples(b)
