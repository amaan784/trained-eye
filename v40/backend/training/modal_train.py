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

train() also writes its result to /vol/runs/<run>/result.json, so a run launched with
`modal run --detach` (te train --detach) can be collected later (te train --collect) even
if this Mac lost the connection. W&B gets the tags te train passes, the bundle summary in
the run's config (wandb-config.yaml next to the config, through WANDB_CONFIG_PATHS),
`--wandb-entity` for a team, and with `--wandb-artifacts` (te train asks first) the bundle
as a dataset artifact and the adapter at the end.

`--preflight` runs no training: the CPU function preflight() loads the pinned processor
(no weights) and renders a few bundle examples (backend/training/preflight.py).
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

APP = "trained-eye-train"
VOLUME = "trained-eye"
MOUNT = "/vol"
GPU = "H100"
WANDB_SECRET = "wandb"
HF_SECRET = "huggingface"
TIMEOUT_S = 4 * 3600
PREFLIGHT_TIMEOUT_S = 30 * 60  # our choice: the processor and a few images, no weights
PREFLIGHT_EXAMPLES = 4  # our choice: records rendered, each in both prompt layouts
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
# The same image (its transformers) plus preflight.py, which imports nothing from backend.
preflight_image = image.add_local_file(Path(__file__).with_name("preflight.py"),
                                       "/root/te_preflight.py")


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_bundle(run: str, config: dict, tags: list[str], summary: dict) -> str:
    """Log /vol/bundles/<run> as a W&B dataset artifact named <run>-<data hash>; only with
    te train --wandb-artifacts, after the author said yes. Returns the artifact's name."""
    import wandb

    info = summary["trained_eye"]["value"]
    name = f"{run}-{info['data_hash'][:12]}"
    wb = wandb.init(project=config["wandb_project"], entity=config.get("wandb_entity"),
                    name=f"{run}-bundle", job_type="dataset", tags=tags,
                    config={"trained_eye": info})
    artifact = wandb.Artifact(name, type="dataset", metadata=info)
    artifact.add_dir(f"{MOUNT}/bundles/{run}")
    wb.log_artifact(artifact)
    wb.finish()
    return name


@app.function(image=image, gpu=GPU, volumes={MOUNT: volume}, timeout=TIMEOUT_S,
              secrets=[modal.Secret.from_name(WANDB_SECRET), modal.Secret.from_name(HF_SECRET)])
def train(run: str, config: dict, tags: list[str], smoke: bool = False,
          summary: dict | None = None, artifacts: bool = False) -> dict:
    """Train one run with `config` (from backend.training.axolotl) on the volume; the
    result, without the log tail, also goes to /vol/runs/<run>/result.json."""
    import os

    run_dir = Path(MOUNT) / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    result_path.unlink(missing_ok=True)  # never collect an earlier run's result
    started = now_iso()
    config_path = run_dir / "config.yaml"
    config_path.write_text(json.dumps(config, indent=1))  # JSON is valid YAML
    env = {**os.environ, "WANDB_TAGS": ",".join(tags)}
    if summary:  # W&B adds these files' keys to the run's config
        summary_path = run_dir / "wandb-config.yaml"
        summary_path.write_text(json.dumps(summary, indent=1))
        env["WANDB_CONFIG_PATHS"] = str(summary_path)
    volume.commit()
    artifact = log_bundle(run, config, tags, summary) if artifacts and summary else None
    start = time.monotonic()
    with (run_dir / "train.log").open("w") as log:
        done = subprocess.run(["axolotl", "train", str(config_path)], env=env, stdout=log,
                              stderr=subprocess.STDOUT, check=False)
    seconds = time.monotonic() - start
    text = (run_dir / "train.log").read_text(errors="replace")
    result = {"run": run, "ok": done.returncode == 0, "seconds": round(seconds, 1),
              "steps": config.get("max_steps"), "output_dir": config["output_dir"],
              "smoke": smoke, "started": started, "finished": now_iso(),
              **({"wandb_artifact": artifact} if artifact else {}), **timings(text)}
    result_path.write_text(json.dumps(result, indent=1) + "\n")
    volume.commit()
    return {**result, "log_tail": text.splitlines()[-30:]}


@app.function(image=preflight_image, volumes={MOUNT: volume}, timeout=PREFLIGHT_TIMEOUT_S,
              secrets=[modal.Secret.from_name(HF_SECRET)])
def preflight(base: str, revision: str, examples: list[dict]) -> dict:
    """Render `examples` (preflight.examples, image paths on the volume) with the pinned
    processor on CPU, as the cookbook's build_processor_length_counter: no model weights."""
    import sys

    sys.path.insert(0, "/root")  # where preflight_image puts te_preflight.py
    import te_preflight
    from transformers import AutoProcessor, GenerationConfig

    processor = AutoProcessor.from_pretrained(base, revision=revision)
    try:  # a small JSON file: the stop tokens generation uses
        eos = GenerationConfig.from_pretrained(base, revision=revision).eos_token_id
    except (OSError, ValueError):
        eos = None
    return te_preflight.render_all(processor, examples, generation_eos=eos)


def upload(bundle: Path, run: str) -> str:
    """Copy the local bundle to /vol/bundles/<run> and return that path."""
    remote = f"/bundles/{run}"
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(bundle), remote)
    return f"{MOUNT}{remote}"


@app.local_entrypoint()
def main(bundle: str, run: str, smoke: bool = False, tags: str = "", result: str = "",
         max_steps: int = 0, wandb_entity: str = "", wandb_artifacts: bool = False,
         detached: bool = False, preflight: bool = False) -> None:
    from backend.training.axolotl import qlora_config, wandb_config

    manifest = json.loads((Path(bundle) / "manifest.json").read_text())
    if manifest["image_root"] != f"{MOUNT}/bundles/{run}":
        raise SystemExit(f"the bundle was made for {manifest['image_root']}, not run {run}")
    if preflight:  # the flag; the Modal function of that name runs in run_preflight
        return run_preflight(Path(bundle), run, manifest, result)
    remote = upload(Path(bundle), run)
    val = f"{remote}/val.jsonl" if manifest.get("val_examples") else None
    config = qlora_config(run, f"{remote}/train.jsonl", f"{MOUNT}/runs/{run}/out",
                          val_jsonl=val, smoke=smoke, entity=wandb_entity or None,
                          log_model=wandb_artifacts)
    if max_steps and not smoke:  # fixed steps for Experiments 1 to 3 (backend/eval/plan.py)
        config["max_steps"] = max_steps
    tag_list = [t for t in tags.split(",") if t]
    start = time.monotonic()
    try:
        out = train.remote(run, config, tag_list, smoke=smoke,
                           summary=wandb_config(manifest, run, tag_list),
                           artifacts=wandb_artifacts)
    except BaseException:
        if detached:  # the app goes on without this client: te train --collect
            print(f"the run goes on in the detached app; te train --collect {run} fetches "
                  "its result when it is done")
        elif result:  # the GPU time spent so far, for te train's Modal ledger
            Path(result).write_text(json.dumps({
                "run": run, "ok": False, "smoke": smoke, "seconds_upper_bound": True,
                "seconds": round(time.monotonic() - start, 1)}, indent=1) + "\n")
        raise
    summary = {k: v for k, v in out.items() if k != "log_tail"}
    print(json.dumps(summary))
    if result:  # for te train, which prices the next run from a smoke's timings
        Path(result).write_text(json.dumps({**summary, "smoke": smoke}, indent=1) + "\n")
    if not out["ok"]:
        print("\n".join(out["log_tail"]))
        raise SystemExit(1)


def run_preflight(bundle: Path, run: str, manifest: dict, result: str) -> None:
    """te train --preflight: render a few examples on Modal, judge them here."""
    from backend import config
    from backend.dataengine.dataset import SYSTEM_PROMPT
    from backend.training import preflight as checks
    from backend.training.axolotl import RECIPE

    records = [json.loads(line) for split in ("train", "val")
               if (bundle / f"{split}.jsonl").is_file()
               for line in (bundle / f"{split}.jsonl").read_text().splitlines() if line]
    items = checks.examples(records, PREFLIGHT_EXAMPLES, SYSTEM_PROMPT)
    if not items:
        raise SystemExit("the bundle has no examples to render")
    base = config.DEFAULT_MODELS["student_base"]
    config.check_model_allowed(base)
    upload(bundle, run)  # the images, where the records point
    rendered = preflight.remote(base, config.STUDENT_BASE_REVISION, items)
    report = checks.summarize(rendered, sequence_len=RECIPE["sequence_len"],
                              system_prompt=SYSTEM_PROMPT,
                              bundle_system_in_user=bool(manifest.get("system_in_user")))
    print(json.dumps({k: report[k] for k in ("ok", "longest", "advice")}))
    if result:
        Path(result).write_text(json.dumps({**report, "run": run}, indent=1) + "\n")
