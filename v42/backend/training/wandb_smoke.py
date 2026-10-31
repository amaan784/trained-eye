"""Smoke test: log one Weights & Biases run from a Modal container.

Run from the repo root on the Mac (needs `pip install -e ".[train]"` and `modal token set`):

    modal run backend/training/wandb_smoke.py

It reads WANDB_API_KEY from the Modal Secret named "wandb" (create it in the Modal
dashboard under Secrets). The key never passes through this code or the terminal.
CPU only, about a minute of container time.
"""

from __future__ import annotations

import math
import subprocess

import modal

WANDB_PROJECT = "trained-eye"
MODAL_SECRET = "wandb"

app = modal.App("trained-eye-wandb-smoke")
image = modal.Image.debian_slim(python_version="3.12").pip_install("wandb>=0.19,<1")


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@app.function(image=image, secrets=[modal.Secret.from_name(MODAL_SECRET)], timeout=600)
def smoke(commit: str, entity: str | None = None) -> str:
    import wandb

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=entity,
        name="m0-smoke",
        job_type="smoke",
        tags=["m0", "smoke", f"commit:{commit}"],
        config={"experiment": "m0", "git_commit": commit, "source": "modal"},
    )
    for step in range(20):
        run.log({"smoke/loss": math.exp(-step / 5), "smoke/step": step}, step=step)
    run.summary["smoke/ok"] = True
    url = run.url
    run.finish()
    return url


@app.local_entrypoint()
def main(entity: str = "") -> None:
    url = smoke.remote(git_commit(), entity or None)
    print(f"W&B smoke run logged: {url}")
