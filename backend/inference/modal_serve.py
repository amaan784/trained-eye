"""Serve the student with vLLM's OpenAI-compatible server on Modal: a run's merged model,
the base model, or the base with several runs' adapters. From the repo root (te
serve-student runs these and asks first):

    TE_SERVE_RUN=exp1-mixed modal deploy backend/inference/modal_serve.py
    TE_SERVE_BASE=1 modal deploy backend/inference/modal_serve.py
    TE_SERVE_LORAS=exp1-real,exp1-mixed modal deploy backend/inference/modal_serve.py

The first serves /vol/runs/<run>/merged (backend/training/export.py writes it) as the app
trained-eye-student, model name trained-eye. The other two deploy the app trained-eye-base,
so a base and a fine-tuned endpoint can be up together (Experiment 4, te regress): the base
at the pinned revision, model name muse-glimmer, served from its snapshot folder in the
Hugging Face cache on the volume (/vol/hf, filled by modal_train.py's download_base), the
cookbook's pre-downloaded-weights form, with the Hub offline. Served by its Hub id instead,
vLLM's structured-output tokenizer did not follow --tokenizer-revision and failed to start
(Nov 4: "Couldn't instantiate the backend tokenizer"). TE_SERVE_LORAS also loads each run's QLoRA adapter
(/vol/runs/<run>/out, from modal_train.py) with --enable-lora, each run a model name of its
own (its run name), so Experiments 1 to 3 need no merge and no deploy per run.

Modal prints the URL. Watch mode then uses TE_STUDENT_URL=<that URL>/v1 and the model name
(TE_STUDENT_MODEL), and every request needs the key in the Modal Secret "student"
(STUDENT_API_KEY) as a Bearer token. The URL stays the same when the app is deployed again;
`modal app stop trained-eye-student` (or trained-eye-base) stops it.

The command follows the cookbook (meta-oss-cookbook b30ca2c,
muse-glimmer/inference-server/vllm.md): the vllm/vllm-openai:muse-glimmer image, as no
released vLLM wheel has the model; the recipe's single-card arguments; no --chat-template,
as the checkpoint ships chat_template.jinja (export.py copies the base's if the save wrote
none); parser names with underscores. vllm.md budgets 72 GB of VRAM for bf16: one 80 GB GPU.

Tried on Nov 4: the base model on this image on Modal answers an image request with
logprobs and top_logprobs (te serve-student --base). The adapters with --enable-lora do not
start: the engine fails while it profiles, with an AssertionError in vLLM's LoRA kernel
(lora_shrink_op.py, token_lora_mapping.size(0) == M), so a run is served merged
(te export, then te serve-student --run). LoRA serving is not documented for Muse Glimmer
(vllm.md has no --enable-lora). vLLM serves the base in bf16 while the adapters
were trained on the NF4 base: the same pairing as export.py's default bf16 reference (Meta
evaluated its QLoRA adapters on an NF4 base only).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import modal

APP = "trained-eye-student"
VOLUME = "trained-eye"
MOUNT = "/vol"
VLLM_IMAGE = "vllm/vllm-openai:muse-glimmer"  # published (vllm.md); a tag, not pinned to a digest
GPU_TYPE = "H100"  # 80 GB
GPU_COUNT = 1  # also the tensor parallel size
SCALEDOWN_S = 5 * 60  # our choice: idle time billed against a cold start that reloads 60 GB
STARTUP_TIMEOUT_S = 30 * 60  # our choice, not measured: weights from the volume, then warmup
MAX_INPUTS = 16  # our choice: requests one container takes at once (vLLM batches them)
PORT = 8000  # vLLM's default port
API_SECRET = "student"
API_KEY_ENV = "STUDENT_API_KEY"
SERVED_MODEL_NAME = "trained-eye"  # Settings.student_model (TE_STUDENT_MODEL)
BASE_APP = "trained-eye-base"  # its own app, so it can be up next to APP
BASE_MODEL_NAME = "muse-glimmer"  # vllm.md's --served-model-name label
BASE_MODEL = "meta-models/Muse-Glimmer-30B"  # config.DEFAULT_MODELS["student_base"] (tests)
BASE_REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"  # config.STUDENT_BASE_REVISION
HF_SECRET = "huggingface"  # HF_TOKEN, for the base download
MAX_LORA_RANK = 32  # the recipe's lora_r (backend/training/axolotl.py); vLLM's default is 16
MAX_LORAS = 1  # our choice: adapters in one batch (GPU memory); te watch scores one run
PARSER = "muse_glimmer"  # underscores: vLLM matches the name literally (vllm.md)
# vLLM refuses a request whose top_logprobs is above --max-logprobs. The scorer asks for 10
# (student.py). 20 is vLLM's usual default and the OpenAI API's top_logprobs limit; not
# checked against this image, so it is set explicitly.
MAX_LOGPROBS = 20
# From vllm.md's tighter-memory command. Training used sequence_len 4096 and the scorer asks
# for at most 512 new tokens, so a frame request should fit (not measured).
MAX_MODEL_LEN = 8300
RUN_ENV = "TE_SERVE_RUN"  # read at deploy time and baked into the image, as the two below
BASE_ENV = "TE_SERVE_BASE"  # "1": the base model (app BASE_APP)
LORAS_ENV = "TE_SERVE_LORAS"  # run names, comma-separated: the base with their adapters
RUN_NAME = re.compile(r"[a-z0-9][a-z0-9-]{1,39}")  # as te train checks it


def base_dir() -> str:
    """The pinned base's snapshot folder in the volume's Hugging Face cache."""
    return (f"{MOUNT}/hf/hub/models--{BASE_MODEL.replace('/', '--')}/snapshots/"
            f"{BASE_REVISION}")


def model_dir(run: str) -> str:
    """Where export.py saved the run's merged model."""
    if not RUN_NAME.fullmatch(run):
        raise ValueError(f"deploy with {RUN_ENV}=<run name>, e.g. exp1-mixed (got {run!r})")
    return f"{MOUNT}/runs/{run}/merged"


def lora_modules(runs: str) -> list[str]:
    """vLLM's --lora-modules values for TE_SERVE_LORAS: <run>=/vol/runs/<run>/out."""
    names = [name.strip() for name in runs.split(",") if name.strip()]
    for name in names:
        if not RUN_NAME.fullmatch(name):
            raise ValueError(f"{LORAS_ENV}: bad run name {name!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"{LORAS_ENV}: a run is named twice")
    return [f"{name}={MOUNT}/runs/{name}/out" for name in names]


def serve_command(model: str, name: str = SERVED_MODEL_NAME, revision: str | None = None,
                  loras: list[str] | tuple = ()) -> list[str]:
    """The vLLM command. It never holds the key: vLLM logs its arguments at startup, so the
    key goes in the VLLM_API_KEY environment variable instead (read when --api-key is not
    given). A Hub model gets its pinned revision, for the weights and the tokenizer."""
    command = ["vllm", "serve", model, "--served-model-name", name,
               # the recipe's single-card arguments (vllm.md); tool calling is unused here
               "--tensor-parallel-size", str(GPU_COUNT), "--enable-auto-tool-choice",
               "--tool-call-parser", PARSER, "--reasoning-parser", PARSER,
               "--generation-config", "auto",
               "--max-model-len", str(MAX_MODEL_LEN), "--max-logprobs", str(MAX_LOGPROBS),
               # all interfaces, as Modal's web_server routes to the port inside the container
               "--host", "0.0.0.0", "--port", str(PORT)]
    if revision:
        command += ["--revision", revision, "--tokenizer-revision", revision]
    if loras:  # each adapter is a model name; unverified for Muse Glimmer (module docstring)
        command += ["--enable-lora", "--max-lora-rank", str(MAX_LORA_RANK),
                    "--max-loras", str(MAX_LORAS), "--max-cpu-loras", str(len(loras)),
                    "--lora-modules", *loras]
    return command


def wants_base(env) -> bool:
    """Whether this deploy serves the base (TE_SERVE_BASE or TE_SERVE_LORAS set)."""
    return env.get(BASE_ENV, "") == "1" or bool(env.get(LORAS_ENV, "").strip())


# The deploy's environment is baked into the image, so the container picks the same app.
BASE = wants_base(os.environ)
app = modal.App(BASE_APP if BASE else APP)
volume = modal.Volume.from_name(VOLUME)  # no create: training or export.py filled it
# The image's ENTRYPOINT is `vllm serve` (vllm.md); cleared so Modal can start the container.
# It has python3 (3.12) but no `python`, and Modal reads the version from `python`: the first
# deploy (Nov 4) failed with "unable to determine the version of Python installed".
image = (modal.Image.from_registry(
             VLLM_IMAGE, setup_dockerfile_commands=["RUN ln -sf /usr/bin/python3 /usr/bin/python"])
         .entrypoint([])
         .env({name: os.environ.get(name, "") for name in (RUN_ENV, BASE_ENV, LORAS_ENV)})
         .env({"HF_HOME": f"{MOUNT}/hf"}))  # the base's cache, as modal_train.py's
SECRETS = [modal.Secret.from_name(API_SECRET)] + (
    [modal.Secret.from_name(HF_SECRET)] if BASE else [])


@app.function(image=image, gpu=f"{GPU_TYPE}:{GPU_COUNT}", volumes={MOUNT: volume},
              secrets=SECRETS, scaledown_window=SCALEDOWN_S, timeout=STARTUP_TIMEOUT_S)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT_S)
def serve() -> None:
    """Start vLLM in the background; Modal sends requests to PORT once it listens."""
    if BASE:
        loras = lora_modules(os.environ.get(LORAS_ENV, ""))
        for module in loras:
            adapter = module.split("=", 1)[1]
            if not Path(adapter, "adapter_config.json").is_file():
                raise RuntimeError(f"no adapter in {adapter}: train that run first")
        model = base_dir()
        if not Path(model, "config.json").is_file():
            raise RuntimeError(f"no base model in {model}: run modal run "
                               "backend/training/modal_train.py --fetch-base first")
        command = serve_command(model, BASE_MODEL_NAME, loras=loras)
    else:
        model = model_dir(os.environ.get(RUN_ENV, ""))
        if not Path(model, "config.json").is_file():
            raise RuntimeError(f"no merged model in {model}: run backend/training/export.py "
                               "first")
        command = serve_command(model)
    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        raise RuntimeError(f"the Modal Secret {API_SECRET!r} has no {API_KEY_ENV}")
    # a list, no shell; the key only in the child's environment, never on its command line;
    # every model is a folder on the volume, so nothing is looked up on the Hub
    subprocess.Popen(command, env={**os.environ, "VLLM_API_KEY": api_key,
                                   "HF_HUB_OFFLINE": "1"})
