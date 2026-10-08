"""Shared test fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture()
def tmp_figure_folder(tmp_path) -> Path:
    folder = tmp_path / "figures"
    folder.mkdir()
    return folder


@pytest.fixture(autouse=True)
def _cwd_repo_root(monkeypatch):
    """All tests run with cwd = repo root so relative data/ paths resolve."""
    monkeypatch.chdir(REPO_ROOT)
    yield
