"""Keep the developer's own shell settings out of the tests: every TE_* override and
every key is removed from the environment before each test."""

import os

import pytest

from backend.config import SECRET_NAMES


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith("TE_") or name in SECRET_NAMES or name == "WANDB_ENTITY":
            monkeypatch.delenv(name)
