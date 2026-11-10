"""Atomic persistence: ONE FILE PER RUN, never a shared CSV.

Google Drive plus two Colab runtimes writing the same CSV means silent
corruption. Each run writes its own file atomically (tmp + os.replace), so
reopening a session is just a matter of scanning what exists and running what
is missing. The aggregated CSVs are rebuilt from those files (`--part rebuild`)
and the rebuild works on a partial grid.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

from . import config


# -- layout -----------------------------------------------------------

def ensure_layout(outdir: str | os.PathLike) -> Path:
    """Create the output tree under --outdir and return its root."""
    root = Path(outdir)
    for sub in config.SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def run_path(outdir: str | os.PathLike, run_id: str) -> Path:
    return Path(outdir) / "runs" / f"{run_id}.json"


def predictions_path(outdir: str | os.PathLike, run_id: str) -> Path:
    return Path(outdir) / "predictions" / f"{run_id}.parquet"


def history_path(outdir: str | os.PathLike, run_id: str) -> Path:
    return Path(outdir) / "runs" / f"{run_id}__history.csv"


def checkpoint_dir(outdir: str | os.PathLike, run_id: str) -> Path:
    return Path(outdir) / "checkpoints" / run_id


# -- atomic writes ----------------------------------------------------

def atomic_write_bytes(path: str | os.PathLike, payload: bytes) -> Path:
    """Write to a tmp file in the SAME directory, then os.replace it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)      # never leave a stray tmp file behind
        except OSError:
            pass
        raise
    return path


def atomic_write_text(path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path, obj: Any) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=2, default=_jsonable))


def _jsonable(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


# -- reading / scanning -----------------------------------------------

def run_exists(outdir, run_id: str) -> bool:
    """A run only counts as done if its JSON exists, parses and is completed.

    A truncated JSON (Colab session killed mid-write) is treated as absent so
    that it gets redone. Valid JSON that is not an object ('[]', 'null', '3')
    would pass json.load and then blow up on .get, taking down the whole scan
    because of a single file.
    """
    p = run_path(outdir, run_id)
    if not p.exists():
        return False
    try:
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(rec, dict):
        return False
    return rec.get("status") == "completed"


def load_run(outdir, run_id: str) -> dict:
    with open(run_path(outdir, run_id), encoding="utf-8") as f:
        return json.load(f)


def iter_runs(outdir) -> Iterator[dict]:
    """Scan every stored run, skipping the damaged ones with a warning."""
    d = Path(outdir) / "runs"
    if not d.exists():
        return
    for p in sorted(d.glob("*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [warning] unreadable run, skipped: {p.name} ({e})")
            continue
        if not isinstance(rec, dict):
            print(f"  [warning] run is not a JSON object, skipped: {p.name}")
            continue
        yield rec


def write_env_json(outdir, grid=None) -> dict:
    """Versions, hardware and actual selected grid for reproducibility.

    CLI overrides must be recorded instead of silently falling back to module
    defaults. The grid remains optional for callers that only want the frozen
    paper defaults.
    """
    import importlib.metadata as md
    import platform
    import sys
    import time

    pkgs = {}
    for p in ("pennylane", "autograd", "torch", "numpy", "scipy",
              "pandas", "scikit-learn", "ucimlrepo", "pyarrow", "matplotlib"):
        try:
            pkgs[p] = md.version(p)
        except Exception:
            pkgs[p] = None

    selected = tuple(grid or ())

    def selected_value(field: str, default):
        values = sorted({getattr(run, field) for run in selected})
        if not values:
            return default
        return values[0] if len(values) == 1 else values

    env = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": pkgs,
        "device": "default.qubit (analytic statevector, no shot noise)",
        "epoch_semantics": "1 epoch = COMPLETE sweep over the training set in minibatches",
        "hyperparams": {
            "n_qubits": config.N_QUBITS,
            "vqc_layers": config.VQC_LAYERS,
            "mlp_hidden": list(config.MLP_HIDDEN),
            "n_epochs": selected_value("n_epochs", config.N_EPOCHS),
            "batch_size": selected_value("batch_size", config.BATCH_SIZE),
            "learning_rate": selected_value("learning_rate", config.LEARNING_RATE),
            "seeds": sorted({run.seed for run in selected}) if selected else list(config.SEEDS),
            "split": [config.TRAIN_FRAC, config.VAL_FRAC, config.TEST_FRAC],
            "feature_range": list(config.FEATURE_RANGE),
        },
        "selected_grid": {
            "run_count": len(selected),
            "datasets": sorted({run.dataset for run in selected}),
            "models": sorted({run.model for run in selected}),
        } if selected else None,
    }
    atomic_write_json(Path(outdir) / "env.json", env)
    return env
