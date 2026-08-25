"""Resume: an interrupted grid must converge to the SAME result as one that
never stopped.

The Colab runtime dies.  It dies mid-grid, it dies mid-run, and it dies while
writing to Drive.  The whole storage design (one file per run, atomic writes,
"completed" as the only marker of done) exists for that.  These tests kill a
grid on purpose -- by deleting files, by corrupting them, and by actually
killing the process -- and demand that the reconstructed CSVs come out
identical to an uninterrupted execution.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import helpers
from vqcdepth import config, runner, splits, storage, tables

REPO_ROOT = Path(__file__).resolve().parents[1]

# Columns that legitimately differ between two executions of the same cell.
VOLATILE = ["timestamp", "training_time_seconds", "inference_time_seconds",
            "wall_time_seconds"]


@pytest.fixture
def tiny_dataset(monkeypatch):
    """A 140-row stand-in for the UCI data: the real make_split, tiny inputs."""
    raw = helpers.synthetic_raw(n=140, seed=0, n_nan=8)
    monkeypatch.setattr(splits, "load_dataset", lambda name: raw)
    return raw


def _grid():
    return runner.build_grid(datasets=["banknote"], models=["vqc1", "vqc2", "mlp"],
                             seeds=[0, 1], n_epochs=2, save_checkpoints=False)


def _stable(outdir) -> pd.DataFrame:
    """The reconstructed raw_results, minus wall-clock noise."""
    raw = tables.collect_raw(outdir)
    return (raw.drop(columns=[c for c in VOLATILE if c in raw.columns])
               .sort_values("run_id").reset_index(drop=True))


def _predictions(outdir) -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted((Path(outdir) / "predictions").glob("*.parquet")):
        out[p.stem] = pd.read_parquet(p)
    return out


# ======================================================================
# 1. pending() is the resume mechanism
# ======================================================================
def test_pending_lists_everything_before_the_first_run(outdir, tiny_dataset):
    grid = _grid()
    assert len(runner.pending(grid, outdir)) == len(grid) == 6


def test_pending_never_reruns_a_completed_cell(outdir, tiny_dataset, capsys):
    grid = _grid()
    runner.run_grid(grid, outdir)
    capsys.readouterr()

    assert runner.pending(grid, outdir) == [], "a completed cell is still pending"
    summary = runner.run_grid(grid, outdir)
    assert summary["executed"] == 0 and summary["pending_before"] == 0, (
        "the second pass re-executed work that was already on disk; a resumed "
        "Colab session would redo the whole grid"
    )


def test_pending_reclaims_a_run_whose_json_was_lost_or_corrupted(outdir, tiny_dataset,
                                                                 capsys):
    grid = _grid()
    runner.run_grid(grid, outdir)
    capsys.readouterr()

    storage.run_path(outdir, "banknote__vqc1__seed00").unlink()
    storage.run_path(outdir, "banknote__mlp__seed01").write_text(
        '{"run_id": "banknote__mlp__seed01", "stat', encoding="utf-8")
    storage.atomic_write_json(
        storage.run_path(outdir, "banknote__vqc2__seed00"),
        {**json.loads(storage.run_path(outdir, "banknote__vqc2__seed00")
                      .read_text(encoding="utf-8")), "status": "failed"})

    todo = {c.run_id for c in runner.pending(grid, outdir)}
    assert todo == {"banknote__vqc1__seed00", "banknote__mlp__seed01",
                    "banknote__vqc2__seed00"}, todo


# ======================================================================
# 2. IDEMPOTENCE: interrupted grid == uninterrupted grid
# ======================================================================
def test_an_interrupted_grid_rebuilds_to_the_same_numbers(tmp_path, tiny_dataset,
                                                          capsys):
    """Run everything; destroy part of it the way a dead runtime would; resume.

    The reconstructed raw_results and the per-run predictions must come back
    IDENTICAL (bit for bit, except wall-clock fields).  If they do not, results
    depend on WHEN the runtime died -- and the grid is not reproducible.
    """
    grid = _grid()

    reference = storage.ensure_layout(tmp_path / "reference")
    runner.run_grid(grid, reference)

    resumed = storage.ensure_layout(tmp_path / "resumed")
    runner.run_grid(grid, resumed)

    # --- simulate the crash: two runs vanish, one is a half-written file ---
    for rid in ("banknote__vqc1__seed00", "banknote__mlp__seed01"):
        storage.run_path(resumed, rid).unlink()
        storage.history_path(resumed, rid).unlink()
        storage.predictions_path(resumed, rid).unlink()
    storage.run_path(resumed, "banknote__vqc2__seed01").write_text(
        '{"run_id": "banknote__vqc2__seed01", "status": "compl', encoding="utf-8")

    summary = runner.run_grid(grid, resumed)
    capsys.readouterr()
    assert summary["executed"] == 3, (
        f"expected exactly the 3 damaged cells to be redone, got {summary}"
    )
    assert summary["failed"] == []

    ref_df, got_df = _stable(reference), _stable(resumed)
    assert len(ref_df) == len(got_df) == 6
    pd.testing.assert_frame_equal(
        ref_df, got_df, check_exact=True,
        obj="raw_results after a crash vs. an uninterrupted run",
    )

    ref_pred, got_pred = _predictions(reference), _predictions(resumed)
    assert set(ref_pred) == set(got_pred)
    for rid in ref_pred:
        pd.testing.assert_frame_equal(ref_pred[rid], got_pred[rid],
                                      check_exact=True, obj=f"predictions[{rid}]")


def test_running_the_grid_twice_changes_nothing_on_disk(outdir, tiny_dataset,
                                                        capsys):
    grid = _grid()
    runner.run_grid(grid, outdir)
    first = _stable(outdir)
    mtimes = {p.name: p.stat().st_mtime_ns
              for p in (Path(outdir) / "runs").glob("*.json")}

    time.sleep(0.01)
    runner.run_grid(grid, outdir)
    capsys.readouterr()

    pd.testing.assert_frame_equal(first, _stable(outdir), check_exact=True)
    after = {p.name: p.stat().st_mtime_ns
             for p in (Path(outdir) / "runs").glob("*.json")}
    assert after == mtimes, "a completed run file was rewritten on the second pass"


def test_a_failing_cell_does_not_stop_the_others(outdir, tiny_dataset, monkeypatch,
                                                 capsys):
    """One exploding run must leave the other five completed and resumable."""
    real_train = runner.execute_run

    def flaky(cfg, out):
        if cfg.run_id == "banknote__vqc2__seed00":
            raise RuntimeError("simulated CUDA/Drive hiccup")
        return real_train(cfg, out)

    monkeypatch.setattr(runner, "execute_run", flaky)
    grid = _grid()
    summary = runner.run_grid(grid, outdir)
    capsys.readouterr()

    assert summary["executed"] == 5
    assert [rid for rid, _ in summary["failed"]] == ["banknote__vqc2__seed00"]
    assert [c.run_id for c in runner.pending(grid, outdir)] == \
        ["banknote__vqc2__seed00"]


# ======================================================================
# 3. what a completed run must contain
# ======================================================================
def test_a_completed_record_carries_the_split_fingerprints_and_the_test_metrics(
        outdir, tiny_dataset, capsys):
    runner.run_grid(_grid(), outdir)
    capsys.readouterr()
    rec = storage.load_run(outdir, "banknote__vqc2__seed00")

    sp = splits.make_split("banknote", 0)
    assert rec["split_hash_all"] == sp.hashes["all"]
    assert rec["split_hash_train"] == sp.hashes["train"]
    assert rec["n_parameters"] == config.n_parameters("vqc2") == 16
    assert rec["test_samples"] == len(sp.y_test)
    assert rec["TN"] + rec["FP"] + rec["FN"] + rec["TP"] == len(sp.y_test)
    assert 0.0 <= rec["test_f1"] <= 1.0
    assert rec["status"] == "completed"


def test_all_models_of_a_seed_recorded_the_same_split_hashes(outdir, tiny_dataset,
                                                             capsys):
    """End-to-end version of the pairing guarantee: read it back from the files
    the paper is actually built from."""
    runner.run_grid(_grid(), outdir)
    capsys.readouterr()
    raw = tables.collect_raw(outdir)

    for seed, sub in raw.groupby("seed"):
        assert sub["split_hash_all"].nunique() == 1, (
            f"seed {seed}: the models saw different splits "
            f"({dict(zip(sub['model'], sub['split_hash_all']))}). The paired "
            f"Wilcoxon of the paper would be invalid."
        )
    assert raw.groupby("model")["split_hash_all"].nunique().eq(2).all(), (
        "each model should have seen exactly the 2 distinct seeds of this grid"
    )


def test_predictions_are_indexed_by_the_original_sample_id(outdir, tiny_dataset,
                                                           capsys):
    runner.run_grid(_grid(), outdir)
    capsys.readouterr()
    df = pd.read_parquet(storage.predictions_path(outdir,
                                                  "banknote__vqc1__seed00"))
    sp = splits.make_split("banknote", 0)
    np.testing.assert_array_equal(df["sample_id"].to_numpy(), sp.idx_test)
    np.testing.assert_array_equal(df["y_true"].to_numpy(), sp.y_test)
    assert set(df["y_pred"]) <= {0, 1}
    assert ((df["y_probability"] >= 0.0) & (df["y_probability"] <= 1.0)).all()
    np.testing.assert_array_equal(df["y_pred"].to_numpy(),
                                  (df["y_probability"] >= 0.5).astype(np.int64))


def test_classification_metrics_agree_with_hand_computed_values():
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.8, 0.9, 0.6, 0.2, 0.3])
    m = runner.classification_metrics(y, p)
    # predicted: [0, 1, 1, 1, 0, 0] -> TP=2 FP=1 FN=1 TN=2
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["balanced_accuracy"] == pytest.approx(2 / 3)


def test_classification_metrics_report_nan_auc_for_a_single_class_partition():
    m = runner.classification_metrics(np.zeros(5, dtype=int), np.full(5, 0.3))
    assert np.isnan(m["roc_auc"]), "AUC over one class must be NaN, not a fake 0.5"
    assert m["TP"] == 0 and m["TN"] == 5


def test_build_grid_rejects_unknown_cells():
    with pytest.raises(ValueError):
        runner.build_grid(datasets=["iris"])
    with pytest.raises(ValueError):
        runner.build_grid(models=["cnn"])


def test_build_grid_covers_the_full_experiment_by_default():
    grid = runner.build_grid()
    assert len(grid) == 120 == len(config.DATASETS) * len(config.MODELS) * \
        len(config.SEEDS)
    assert len({c.run_id for c in grid}) == 120, "duplicated run_id in the grid"


# ======================================================================
# 4. through the CLI, with a REAL process kill
# ======================================================================
def _cli(args, timeout=180):
    return subprocess.run([sys.executable, "-m", "vqcdepth", *args],
                          capture_output=True, text=True, timeout=timeout,
                          cwd=str(REPO_ROOT),
                          env={**dict(__import__("os").environ),
                               "PYTHONPATH": str(REPO_ROOT / "src")})


def test_cli_status_reports_the_pending_grid_without_running_anything(tmp_path):
    out = tmp_path / "out"
    proc = _cli(["--part", "status", "--outdir", str(out)])
    assert proc.returncode == 0, proc.stderr
    assert "grid selecionado: 120" in proc.stdout
    assert "faltando:         120" in proc.stdout
    assert list((out / "runs").iterdir()) == [], "status executed a run"


def test_cli_rejects_an_unknown_dataset(tmp_path):
    proc = _cli(["--part", "status", "--outdir", str(tmp_path), "--datasets", "iris"])
    assert proc.returncode != 0 and "invalid choice" in proc.stderr


@pytest.mark.slow
@pytest.mark.network
def test_killing_the_cli_mid_grid_leaves_only_intact_runs_and_resumes_cleanly(
        tmp_path, uci_available):
    """The real thing: SIGKILL the process while it is writing to ``runs/``.

    Uses the smallest real dataset with 1 epoch per run.  Two guarantees:
      * every JSON left behind parses and says "completed" -- atomic writes
        mean a killed process never leaves a half-run marked as done;
      * finishing the grid afterwards reproduces, exactly, the numbers of an
        uninterrupted execution.
    """
    common = ["--part", "train", "--outdir", None, "--datasets", "blood",
              "--models", "vqc1", "--seeds", "0", "1", "2", "3",
              "--epochs", "1", "--no-checkpoint"]

    reference = tmp_path / "reference"
    common[3] = str(reference)
    proc = _cli(list(common))
    assert proc.returncode == 0, proc.stderr
    assert len(list((reference / "runs").glob("*.json"))) == 4, proc.stdout

    killed = tmp_path / "killed"
    common[3] = str(killed)
    child = subprocess.Popen(
        [sys.executable, "-m", "vqcdepth", *common],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=str(REPO_ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")})
    try:
        child.wait(timeout=4.0)          # long enough for a run or two to land
    except subprocess.TimeoutExpired:
        pass
    child.kill()
    child.communicate()

    survivors = sorted((killed / "runs").glob("*.json"))
    for p in survivors:
        rec = json.loads(p.read_text(encoding="utf-8"))   # must parse
        assert rec["status"] == "completed", (
            f"{p.name} survived the kill in a non-completed state: the write "
            f"was not atomic"
        )

    finish = _cli(list(common))
    assert finish.returncode == 0, finish.stderr
    assert len(list((killed / "runs").glob("*.json"))) == 4

    pd.testing.assert_frame_equal(
        _stable(reference), _stable(killed), check_exact=True,
        obj="raw_results after a real process kill vs. an uninterrupted run",
    )
