"""Child process used by the cross-process determinism tests.

Prints ONLY a JSON object with the split fingerprints, so that a parent running
several children under different ``PYTHONHASHSEED`` values can compare them
byte for byte.  Not collected by pytest (the file name does not start with
``test_``).

    python tests/_subprocess_split_child.py synthetic <dataset> <seed>
    python tests/_subprocess_split_child.py real      <dataset> <seed>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                       # tests/  -> helpers
sys.path.insert(0, str(_HERE.parent / "src"))        # src/    -> vqcdepth

import numpy as np  # noqa: E402

from vqcdepth import splits  # noqa: E402


def main() -> int:
    mode, dataset, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])

    if mode == "synthetic":
        import helpers

        raw = helpers.synthetic_raw(n=140, seed=0, n_nan=12)
        splits.load_dataset = lambda name: raw          # noqa: ARG005

    sp = splits.make_split(dataset, seed)
    payload = {
        "hashes": sp.hashes,
        "n_imputed": sp.n_imputed,
        # a float fingerprint too: the hashes only cover the INDICES, and a
        # preprocessing difference between processes would slip past them.
        "X_train_sha": __import__("hashlib").sha256(
            np.ascontiguousarray(sp.X_train, dtype="<f8").tobytes()).hexdigest(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
