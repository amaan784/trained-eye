"""QLoRA training of the student on Modal, with Meta's Axolotl recipe.

Run from the repo root on the Mac (needs `pip install -e ".[train]"` and `modal token
set`), normally through `te train`, which bundles the data first and asks before
spending:

    modal run backend/training/modal_train.py --bundle data/train/exp1-mixed --run exp1-mixed --smoke

The container follows the recipe's setup (meta-oss-cookbook b30ca2c,
muse-glimmer/post-training/sft/axolotl/README.md): Python 3.12, CUDA 13.0, torch 2.12.1,
torchvision 0.27.1, Axolotl 0.19.0, flash-attn 2.8.3 and Cut Cross Entropy at the pinned
commit, built for the H100. Its first build compiles flash-attn and takes a while; Modal
caches it after that.

Data and results live on the Modal volume "trained-eye": the bundle under /vol/bundles,
each run under /vol/runs/<run> (the adapter, the config, the log), and the Hugging Face
cache under /vol/hf so the 30B base is downloaded once. WANDB_API_KEY comes from the
Modal Secret "wandb" and HF_TOKEN from "huggingface"; neither passes through this code.

`--smoke` runs the recipe's two-step check (max 2 steps, a checkpoint at step 2). Its
seconds per step price the real run before it starts.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import modal

APP = "trained-eye-train"
VOLUME = "trained-eye"
MOUNT = "/vol"
GPU = "H100"
WANDB_SECRET = "wandb"
HF_SECRET = "huggingface"
TIMEOUT_S = 4 * 3600
CUDA_IMAGE = "nvidia/cuda:13.0.0-devel-ubuntu24.04"  # the recipe needs the full toolkit
CCE = ("cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/"
       "ml-cross-entropy.git@4dfa52212d0552805aec5a6a2176c0abbea277d2")
BUILD_ENV = {"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "9.0",
             "FLASH_ATTN_CUDA_ARCHS": "90", "MAX_JOBS": "8", "NVCC_THREADS": "2",
             "UV_TORCH_BACKEND": "cu130"}

# The trainer's summary line, e.g. {'train_runtime': 41.2, 'train_samples_per_second': 0.05, ...}
SUMMARY = re.compile(r"'train_(runtime|samples_per_second)':\s*([0-9.]+)")


def timings(log: str) -> dict:
    """train_runtime (s) and train_samples_per_second from the Axolotl log, if present."""
    return {f"train_{k}": float(v) for k, v in SUMMARY.findall(log)}


app = modal.App(APP)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .env(BUILD_ENV)
    .run_commands(
        'uv pip install --system "torch==2.12.1" "torchvision==0.27.1"',
        'uv pip install --system "axolotl==0.19.0"',
        'uv pip install --system --no-build-isolation "flash-attn==2.8.3"',
        f'uv pip install --system --no-build-isolation "{CCE}"',
        "uv pip check --system",
    )
    .env({"HF_HOME": f"{MOUNT}/hf"})  # the base model is downloaded to the volume once
)


@app.function(image=image, gpu=GPU, volumes={MOUNT: volume}, timeout=TIMEOUT_S,
              secrets=[modal.Secret.from_name(WANDB_SECRET), modal.Secret.from_name(HF_SECRET)])
def train(run: str, config: dict, tags: list[str]) -> dict:
    """Train one run with `config` (from backend.training.axolotl) on the volume."""
    import os

    run_dir = Path(MOUNT) / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    config_path.write_text(json.dumps(config, indent=1))  # JSON is valid YAML
    env = {**os.environ, "WANDB_TAGS": ",".join(tags)}
    start = time.monotonic()
    with (run_dir / "train.log").open("w") as log:
        done = subprocess.run(["axolotl", "train", str(config_path)], env=env, stdout=log,
                              stderr=subprocess.STDOUT, check=False)
    seconds = time.monotonic() - start
    volume.commit()
    text = (run_dir / "train.log").read_text(errors="replace")
    return {"run": run, "ok": done.returncode == 0, "seconds": round(seconds, 1),
            "steps": config.get("max_steps"), "output_dir": config["output_dir"],
            **timings(text), "log_tail": text.splitlines()[-30:]}


def upload(bundle: Path, run: str) -> str:
    """Copy the local bundle to /vol/bundles/<run> and return that path."""
    remote = f"/bundles/{run}"
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(bundle), remote)
    return f"{MOUNT}{remote}"


@app.local_entrypoint()
def main(bundle: str, run: str, smoke: bool = False, tags: str = "", result: str = "",
         max_steps: int = 0) -> None:
    from backend.training.axolotl import qlora_config

    manifest = json.loads((Path(bundle) / "manifest.json").read_text())
    if manifest["image_root"] != f"{MOUNT}/bundles/{run}":
        raise SystemExit(f"the bundle was made for {manifest['image_root']}, not run {run}")
    remote = upload(Path(bundle), run)
    val = f"{remote}/val.jsonl" if manifest.get("val_examples") else None
    config = qlora_config(run, f"{remote}/train.jsonl", f"{MOUNT}/runs/{run}/out",
                          val_jsonl=val, smoke=smoke)
    if max_steps and not smoke:  # fixed steps for Experiments 1 to 3 (backend/eval/plan.py)
        config["max_steps"] = max_steps
    out = train.remote(run, config, [t for t in tags.split(",") if t])
    summary = {k: v for k, v in out.items() if k != "log_tail"}
    print(json.dumps(summary))
    if result:  # for te train, which prices the next run from a smoke's timings
        Path(result).write_text(json.dumps({**summary, "smoke": smoke}, indent=1) + "\n")
    if not out["ok"]:
        print("\n".join(out["log_tail"]))
        raise SystemExit(1)
