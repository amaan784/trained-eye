import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend.config import (
    DEFAULT_MODELS,
    STUDENT_BASE_REVISION,
    ConfigError,
    MissingSecret,
    describe,
    load_settings,
    read_env_file,
)

ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "sk-test-do-not-print-0123456789"


def test_defaults():
    s = load_settings(env={}, env_file=None)
    assert s.base_url == "https://api.meta.ai/v1"
    assert s.models == DEFAULT_MODELS
    assert s.skill_cap_usd == 10.0 and s.global_cap_usd == 120.0
    assert s.cache_dir == Path("data/cache")
    assert s.ledger_path == Path("data/ledger.jsonl")
    assert s.price("muse-image-1.0").per_image == 0.01
    assert len(STUDENT_BASE_REVISION) == 40


def test_env_overrides():
    s = load_settings(
        env={
            "TE_MODEL_LABELER": "muse-spark-1.4",
            "TE_DATA_DIR": "/tmp/te",
            "TE_GLOBAL_CAP_USD": "50",
            "TE_BASE_URL": "https://example.test/v1/",
        },
        env_file=None,
    )
    assert s.model("labeler") == "muse-spark-1.4"
    assert s.model("confirm") == DEFAULT_MODELS["confirm"]
    assert s.data_dir == Path("/tmp/te")
    assert s.global_cap_usd == 50.0
    assert s.base_url == "https://example.test/v1"


@pytest.mark.parametrize("job, model", [
    ("LABELER", "muse-spark-1.3-contributor"),
    ("STUDENT_BASE", "meta-models/Muse-Glimmer-30B-Contributor"),
])
def test_contributor_models_are_rejected(job, model):
    with pytest.raises(ConfigError, match="contributor"):
        load_settings(env={f"TE_MODEL_{job}": model}, env_file=None)


@pytest.mark.parametrize("name", ["TE_SKILL_CAP_USD", "TE_GLOBAL_CAP_USD", "TE_MAX_RETRIES"])
def test_bad_numbers_are_config_errors(name):
    with pytest.raises(ConfigError, match=name):
        load_settings(env={name: "five"}, env_file=None)


def test_mangled_secret_is_a_config_error_without_the_value():
    s = load_settings(env={"MODEL_API_KEY": "\u201csk-pasted-with-smart-quotes\u201d"},
                      env_file=None)
    with pytest.raises(ConfigError) as err:
        s.secret("MODEL_API_KEY")
    assert "sk-pasted" not in str(err.value)


def test_unknown_job_and_price():
    s = load_settings(env={}, env_file=None)
    with pytest.raises(ConfigError):
        s.model("nope")
    with pytest.raises(ConfigError):
        s.price("nope")


def test_env_file_parsing_and_precedence(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n\nexport MODEL_API_KEY='from-file'\nHF_TOKEN=\"hf-file\"\nJUNK\n"
        "TE_SKILL_CAP_USD=10  # per skill\nWANDB_API_KEY='a # b'\n"
    )
    assert read_env_file(env_file) == {"MODEL_API_KEY": "from-file", "HF_TOKEN": "hf-file",
                                       "TE_SKILL_CAP_USD": "10", "WANDB_API_KEY": "a # b"}
    s = load_settings(env={"MODEL_API_KEY": "from-env"}, env_file=env_file)
    assert s.secret("MODEL_API_KEY") == "from-env"
    assert s.secret("HF_TOKEN") == "hf-file"
    assert read_env_file(tmp_path / "missing.env") == {}


def test_secrets_never_shown():
    s = load_settings(env={"MODEL_API_KEY": FAKE_KEY}, env_file=None)
    assert s.secret("MODEL_API_KEY") == FAKE_KEY
    with pytest.raises(MissingSecret, match="WANDB_API_KEY"):
        s.secret("WANDB_API_KEY")
    assert FAKE_KEY not in repr(s)
    text = describe(s)
    assert FAKE_KEY not in text
    assert "MODEL_API_KEY  set" in text and "WANDB_API_KEY  missing" in text


def test_empty_values_mean_unset():
    s = load_settings(env={"TE_DATA_DIR": "", "TE_BASE_URL": "", "TE_SKILL_CAP_USD": ""},
                      env_file=None)
    assert s.data_dir == Path("data") and s.base_url == "https://api.meta.ai/v1"
    assert s.skill_cap_usd == 10.0


def test_module_entry_point(tmp_path):
    clean = {k: v for k, v in os.environ.items()
             if not k.startswith("TE_") and k not in ("MODEL_API_KEY", "HF_TOKEN", "WANDB_API_KEY")}
    env = {**clean, "PYTHONPATH": str(ROOT), "MODEL_API_KEY": FAKE_KEY}
    out = subprocess.run(
        [sys.executable, "-m", "backend.config"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    ).stdout
    assert out.startswith("Trained Eye settings")
    assert FAKE_KEY not in out
