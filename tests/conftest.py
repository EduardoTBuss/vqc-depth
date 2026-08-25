"""Suite-wide fixtures.

The repository root ``conftest.py`` already puts ``src/`` on ``sys.path``; this
one only adds ``tests/`` so that the test modules can ``import helpers``, plus
the fixtures that decide whether the UCI-backed tests can run offline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from vqcdepth import config as vqc_config  # noqa: E402
from vqcdepth import data  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: needs the UCI download (auto-skipped when data/cache is empty)",
    )
    config.addinivalue_line("markers", "slow: takes more than a couple of seconds")


def cache_is_complete() -> bool:
    """True when every dataset of the grid is already in ``data/cache``."""
    return all((data.CACHE_DIR / f"{n}.npz").exists() for n in vqc_config.DATASETS)


@pytest.fixture(scope="session")
def uci_available() -> bool:
    """Skip UCI-backed tests unless the cache exists or downloads are allowed.

    Rationale: this suite is the GATE of the Colab grid.  A flaky UCI mirror
    must never be able to stop 120 runs from starting.  Set
    ``VQCDEPTH_ALLOW_DOWNLOAD=1`` to let the tests populate the cache instead.
    """
    if cache_is_complete():
        return True
    if os.environ.get("VQCDEPTH_ALLOW_DOWNLOAD") == "1":
        return True
    pytest.skip(
        "data/cache incompleto e VQCDEPTH_ALLOW_DOWNLOAD != 1: "
        "testes que dependem do UCI foram pulados"
    )


@pytest.fixture
def outdir(tmp_path: Path) -> Path:
    """An isolated output root -- no test may ever write inside the repo."""
    from vqcdepth import storage

    return storage.ensure_layout(tmp_path / "out")
