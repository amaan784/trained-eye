"""Keep the developer's own settings out of the tests: every TE_* override and every key
is removed from the environment, and the project folder (where .env, the ledger and the
cache live) is a fresh temporary folder for each test."""

import os

import pytest

from backend import config
from backend.config import SECRET_NAMES


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)  # never the real .env or ledger
    for name in list(os.environ):
        if name.startswith("TE_") or name in SECRET_NAMES or name == "WANDB_ENTITY":
            monkeypatch.delenv(name)
