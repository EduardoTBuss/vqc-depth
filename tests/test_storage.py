"""Persistence: atomic writes and defensive reads.

The grid runs on Colab, writing to Google Drive, with a runtime that can be
killed at any moment.  A run counts as done only when its JSON exists, parses
and says ``completed``; anything else must look absent so the run is redone,
and must never take down the scan of the other 119 runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from vqcdepth import config, storage


def _record(run_id: str, status: str = "completed") -> dict:
    return {"run_id": run_id, "dataset": "banknote", "model": "vqc1",
            "seed": 0, "status": status, "test_f1": 0.5}


def _write_run(outdir: Path, run_id: str, status: str = "completed") -> Path:
    return storage.atomic_write_json(storage.run_path(outdir, run_id),
                                     _record(run_id, status))


# --- 1. Layout ---
def test_ensure_layout_creates_every_subdirectory_and_is_idempotent(tmp_path):
    root = storage.ensure_layout(tmp_path / "out")
    for sub in config.SUBDIRS:
        assert (root / sub).is_dir(), f"missing subdirectory {sub}"
    before = sorted(p.name for p in root.iterdir())
    assert storage.ensure_layout(root) == root
    assert sorted(p.name for p in root.iterdir()) == before


def test_path_helpers_agree_with_the_run_id_convention(tmp_path):
    cfg = config.RunConfig(dataset="mammographic", model="vqc4", seed=7)
    assert cfg.run_id == "mammographic__vqc4__seed07"
    assert storage.run_path(tmp_path, cfg.run_id).name == f"{cfg.run_id}.json"
    assert storage.predictions_path(tmp_path, cfg.run_id).suffix == ".parquet"
    assert storage.history_path(tmp_path, cfg.run_id).name.endswith("__history.csv")
    assert storage.checkpoint_dir(tmp_path, cfg.run_id).name == cfg.run_id


# --- 2. Atomic writes ---
def test_atomic_write_leaves_no_temporary_file_behind(tmp_path):
    p = storage.atomic_write_text(tmp_path / "deep" / "a.txt", "hello")
    assert p.read_text(encoding="utf-8") == "hello"
    leftovers = [q.name for q in (tmp_path / "deep").iterdir() if q.name != "a.txt"]
    assert leftovers == [], f"temporary files left behind: {leftovers}"


def test_atomic_write_replaces_the_previous_content_in_one_shot(tmp_path):
    path = tmp_path / "a.json"
    storage.atomic_write_json(path, {"v": 1})
    storage.atomic_write_json(path, {"v": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"v": 2}
    assert len(list(tmp_path.iterdir())) == 1


def test_a_failed_write_neither_destroys_the_old_file_nor_leaves_garbage(tmp_path):
    path = tmp_path / "a.txt"
    storage.atomic_write_text(path, "good")
    with pytest.raises(TypeError):
        storage.atomic_write_bytes(path, "not bytes")     # type: ignore[arg-type]
    assert path.read_text(encoding="utf-8") == "good", "the old file was clobbered"
    assert [q.name for q in tmp_path.iterdir()] == ["a.txt"]


def test_json_writer_understands_numpy_scalars_and_arrays(tmp_path):
    payload = {"i": np.int64(3), "f": np.float64(1.5), "a": np.arange(3),
               "nan": float("nan")}
    path = storage.atomic_write_json(tmp_path / "x.json", payload)
    got = json.loads(path.read_text(encoding="utf-8"))
    assert got["i"] == 3 and got["f"] == 1.5 and got["a"] == [0, 1, 2]
    assert got["nan"] != got["nan"], "NaN must survive as NaN, not become null"


def test_json_writer_refuses_something_it_cannot_serialise(tmp_path):
    with pytest.raises(TypeError):
        storage.atomic_write_json(tmp_path / "x.json", {"o": object()})
    assert list(tmp_path.iterdir()) == []


def test_unicode_survives_the_round_trip(tmp_path):
    text = "acentuação — ✓ ünïcödé 中文"
    path = storage.atomic_write_text(tmp_path / "u.txt", text)
    assert path.read_text(encoding="utf-8") == text
    rec = {"note": text}
    p2 = storage.atomic_write_json(tmp_path / "u.json", rec)
    assert json.loads(p2.read_text(encoding="utf-8")) == rec


# --- 3. run_exists: only a complete, parseable, completed run counts ---
def test_run_exists_is_false_when_there_is_no_file(outdir):
    assert storage.run_exists(outdir, "banknote__vqc1__seed00") is False


def test_run_exists_is_true_only_for_status_completed(outdir):
    _write_run(outdir, "a", status="completed")
    _write_run(outdir, "b", status="failed")
    _write_run(outdir, "c", status="running")
    assert storage.run_exists(outdir, "a") is True
    assert storage.run_exists(outdir, "b") is False, "a failed run must be redone"
    assert storage.run_exists(outdir, "c") is False, "a run in flight is not done"


@pytest.mark.parametrize("payload", [
    '{"run_id": "trunc", "status": "compl',      # killed mid-write
    "",                                          # zero bytes
    "\x00\x00\x00\x00",                          # Drive returned garbage
    '{"run_id": "trunc"}',                       # parses, but has no status
])
def test_a_damaged_run_json_counts_as_absent(outdir, payload):
    path = storage.run_path(outdir, "trunc")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    assert storage.run_exists(outdir, "trunc") is False, (
        "a damaged run file was accepted as completed; the grid would keep a "
        "corrupted result instead of redoing the run"
    )


# Regression guard: a JSON file in runs/ that parses but is not an object was
# once accepted as a completed run.
@pytest.mark.parametrize("payload", ["[]", "null", '"x"', "3"])
def test_a_run_json_that_parses_to_a_non_dict_counts_as_absent(outdir, payload):
    path = storage.run_path(outdir, "weird")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    assert storage.run_exists(outdir, "weird") is False


def test_load_run_round_trips_the_record(outdir):
    _write_run(outdir, "banknote__vqc1__seed00")
    rec = storage.load_run(outdir, "banknote__vqc1__seed00")
    assert rec["run_id"] == "banknote__vqc1__seed00" and rec["status"] == "completed"


# --- 4. iter_runs never dies on a bad file ---
def test_iter_runs_skips_damaged_files_and_yields_the_healthy_ones(outdir, capsys):
    for rid in ("a", "b", "c"):
        _write_run(outdir, rid)
    (storage.run_path(outdir, "zz_broken")).write_text('{"nope', encoding="utf-8")

    got = sorted(r["run_id"] for r in storage.iter_runs(outdir))
    assert got == ["a", "b", "c"], (
        f"one corrupted file changed the scan of the healthy ones: {got}"
    )
    assert "warning" in capsys.readouterr().out, "the skip must be reported, not silent"


def test_iter_runs_on_a_missing_directory_is_empty_not_an_error(tmp_path):
    assert list(storage.iter_runs(tmp_path / "does-not-exist")) == []


def test_iter_runs_ignores_the_history_csv_files(outdir):
    _write_run(outdir, "a")
    storage.atomic_write_text(storage.history_path(outdir, "a"), "epoch\n1\n")
    assert [r["run_id"] for r in storage.iter_runs(outdir)] == ["a"]


# --- 5. Environment manifest ---
def test_env_json_records_versions_and_the_frozen_hyperparameters(outdir):
    env = storage.write_env_json(outdir)
    path = Path(outdir) / "env.json"
    assert path.exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(env))

    assert env["packages"]["pennylane"], "pennylane version not recorded"
    hp = env["hyperparams"]
    assert hp["n_qubits"] == config.N_QUBITS
    assert hp["vqc_layers"] == config.VQC_LAYERS
    assert hp["n_epochs"] == config.N_EPOCHS
    assert hp["seeds"] == list(config.SEEDS)
    assert hp["split"] == [config.TRAIN_FRAC, config.VAL_FRAC, config.TEST_FRAC]


def test_nothing_here_writes_inside_the_repository(outdir):
    """Guard for the guard: every path used above lives under tmp_path."""
    repo_root = Path(__file__).resolve().parents[1]
    assert repo_root not in Path(outdir).resolve().parents
    assert os.path.commonpath([str(repo_root), str(Path(outdir).resolve())]) \
        != str(repo_root)
