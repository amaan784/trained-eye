"""Settings for Trained Eye: model IDs, prices, spend caps and paths.

Secrets come from the process environment or the project's .env file. Their values are
never printed or logged; only whether each one is set. The .env file and the data folder
(ledger, cache, skills) belong to the checkout, so `te` finds the same ones from any folder.

Run `python -m backend.config` to see the resolved settings.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

SECRET_NAMES = ("MODEL_API_KEY", "HF_TOKEN", "WANDB_API_KEY")

# The checkout this code runs from (an editable install or `python -m backend.cli`). A copy
# installed elsewhere has no pyproject.toml next to backend/ and uses the current folder.
_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = _ROOT if (_ROOT / "pyproject.toml").is_file() else Path(".")

# One model per job. Override any of them with TE_MODEL_<JOB>, e.g. TE_MODEL_LABELER.
DEFAULT_MODELS = {
    "transcribe": "muse-voice-transcribe-1.0",
    "compiler": "muse-wonder-1.0",
    "sam": "sam-3.1",
    "image": "muse-image-1.0",
    "labeler": "muse-spark-1.3",
    "confirm": "muse-spark-1.3",
    "student_base": "meta-models/Muse-Glimmer-30B",
}

# The revision Meta's cookbook Axolotl recipe pins (muse-glimmer/post-training/sft/axolotl).
STUDENT_BASE_REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"


@dataclass(frozen=True)
class Price:
    """USD prices. Token prices are per million tokens."""

    input_per_m: float = 0.0
    output_per_m: float = 0.0
    per_image: float = 0.0
    per_audio_hour: float = 0.0


# From SPEC.md (dev.meta.ai/docs/pricing-rate-limits as read on Oct 28-29).
PRICES = {
    "muse-spark-1.3": Price(input_per_m=1.25, output_per_m=4.25),
    "muse-wonder-1.0": Price(input_per_m=4.00, output_per_m=20.00),
    "sam-3.1": Price(per_image=2.50 / 1000),
    "muse-image-1.0": Price(per_image=0.01),
    "muse-voice-transcribe-1.0": Price(per_audio_hour=0.18),
}


class ConfigError(RuntimeError):
    pass


class MissingSecret(ConfigError):
    def __init__(self, name: str):
        super().__init__(f"{name} is not set. Add it to .env (see .env.example).")
        self.name = name


def check_model_allowed(model: str) -> None:
    """Meta's '-contributor' model variants must never be used, in any spelling."""
    if "-contributor" in model.lower():
        raise ConfigError(f"'-contributor' model variants are not allowed ({model})")


@dataclass(frozen=True)
class Settings:
    base_url: str = "https://api.meta.ai/v1"
    data_dir: Path = Path("data")
    models: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_MODELS))
    skill_cap_usd: float = 10.0
    global_cap_usd: float = 120.0
    timeout_s: float = 120.0
    max_retries: int = 5
    # False = SPEC.md's fallback: train on real frames and the user's own labels only.
    train_on_model_outputs: bool = True
    # The trained student's OpenAI-compatible endpoint for watch mode (llama-server on the
    # Mac or vLLM on Modal) and the model name it serves.
    student_url: str | None = None
    student_model: str = "trained-eye"
    secrets: Mapping[str, str] = field(default_factory=dict, repr=False)

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def ledger_path(self) -> Path:
        return self.data_dir / "ledger.jsonl"

    def model(self, job: str) -> str:
        try:
            return self.models[job]
        except KeyError:
            raise ConfigError(f"no model configured for job {job!r}") from None

    def price(self, model: str) -> Price:
        check_model_allowed(model)
        try:
            return PRICES[model]
        except KeyError:
            raise ConfigError(f"no price for model {model!r}; add it to PRICES") from None

    def secret(self, name: str) -> str:
        value = self.secrets.get(name, "")
        if not value:
            raise MissingSecret(name)
        if not (value.isascii() and value.isprintable()) or " " in value:
            raise ConfigError(f"{name} has spaces or non-ASCII characters (smart quotes?); "
                              "re-type it in .env")
        return value

    def has_secret(self, name: str) -> bool:
        return bool(self.secrets.get(name))


# A quoted value, optionally followed by a comment: KEY="value"  # note
_QUOTED = re.compile(r"""(['"])(.*?)\1\s*(?:#.*)?""")


def read_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines. Comments, blank lines and `export ` prefixes are allowed."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ")
        key, _, value = line.partition("=")
        value = value.strip()
        quoted = _QUOTED.fullmatch(value)
        value = quoted.group(2) if quoted else re.split(r"\s+#", value, maxsplit=1)[0]
        values[key.strip()] = value
    return values


def load_settings(
    env: Mapping[str, str] | None = None, env_file: Path | str | None = ".env"
) -> Settings:
    """Build Settings from the environment, falling back to the .env file.

    Real environment variables win over .env entries. A relative env_file or TE_DATA_DIR
    is taken inside the project, not the current folder.
    """
    merged: dict[str, str] = {}
    if env_file is not None:
        merged.update(read_env_file(PROJECT_ROOT / env_file))
    merged.update(os.environ if env is None else env)

    models = dict(DEFAULT_MODELS)
    for job in models:
        override = merged.get(f"TE_MODEL_{job.upper()}")
        if override:
            models[job] = override
    for model in models.values():
        check_model_allowed(model)

    def number(name: str, default, kind=float):
        raw = merged.get(name)
        if raw is None or raw == "":
            return default
        try:
            value = kind(raw)
        except ValueError:
            raise ConfigError(f"{name} must be a number, got {raw!r}") from None
        if not math.isfinite(value) or value < 0:  # nan would switch a spend cap off
            raise ConfigError(f"{name} must be a finite number >= 0, got {raw!r}")
        return value

    def switch(name: str, default: bool) -> bool:
        raw = (merged.get(name) or "").strip().lower()
        if not raw:
            return default
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
        # Fail closed: a misspelt "off" must not leave training on model outputs switched on.
        raise ConfigError(f"{name} must be 1 or 0, got {merged[name]!r}")

    raw_dir = merged.get("TE_DATA_DIR") or str(Settings.data_dir)
    try:
        data_dir = Path(raw_dir).expanduser()  # here, as .env files are not read by a shell
    except RuntimeError:
        raise ConfigError(f"TE_DATA_DIR {raw_dir!r}: cannot expand ~ (no such user?)") from None

    return Settings(
        base_url=(merged.get("TE_BASE_URL") or Settings.base_url).rstrip("/"),
        data_dir=PROJECT_ROOT / data_dir,
        models=models,
        skill_cap_usd=number("TE_SKILL_CAP_USD", Settings.skill_cap_usd),
        global_cap_usd=number("TE_GLOBAL_CAP_USD", Settings.global_cap_usd),
        max_retries=number("TE_MAX_RETRIES", Settings.max_retries, int),
        train_on_model_outputs=switch("TE_TRAIN_ON_MODEL_OUTPUTS",
                                      Settings.train_on_model_outputs),
        student_url=(merged.get("TE_STUDENT_URL") or "").strip().rstrip("/") or None,
        student_model=(merged.get("TE_STUDENT_MODEL") or "").strip() or Settings.student_model,
        secrets={name: merged[name] for name in SECRET_NAMES if merged.get(name)},
    )


def shown_path(path: Path) -> str:
    """A path relative to the current folder when it is inside it, else in full."""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(Path(path).resolve())


def describe(settings: Settings) -> str:
    lines = [
        "Trained Eye settings",
        f"  base_url: {settings.base_url}",
        f"  data_dir: {shown_path(settings.data_dir)}",
        (
            f"  spend caps: ${settings.skill_cap_usd:.2f} per skill, "
            f"${settings.global_cap_usd:.2f} global"
        ),
        f"  train on model outputs: {'yes' if settings.train_on_model_outputs else 'no'}",
        (f"  student: {settings.student_url or 'not set (TE_STUDENT_URL)'} "
         f"({settings.student_model})"),
        "  models:",
    ]
    width = max(len(job) for job in settings.models)
    lines += [f"    {job:<{width}}  {model}" for job, model in settings.models.items()]
    lines.append(f"    {'':<{width}}  (student_base revision {STUDENT_BASE_REVISION[:12]})")
    lines.append("  secrets:")
    width = max(len(name) for name in SECRET_NAMES)
    for name in SECRET_NAMES:
        state = "set" if settings.has_secret(name) else "missing"
        lines.append(f"    {name:<{width}}  {state}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe(load_settings()))
