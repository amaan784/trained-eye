"""Serve the merged student with vLLM's OpenAI-compatible server on Modal.

After backend/training/export.py has written /vol/runs/<run>/merged, deploy from the repo root:

    TE_SERVE_RUN=exp1-mixed modal deploy backend/inference/modal_serve.py

Modal prints the URL. Watch mode then uses TE_STUDENT_URL=<that URL>/v1 and the model name
trained-eye, and every request needs the key in the Modal Secret "student" (STUDENT_API_KEY)
as a Bearer token. The URL stays the same when another run is deployed;
`modal app stop trained-eye-student` stops it.

The command follows the cookbook (meta-oss-cookbook b30ca2c,
muse-glimmer/inference-server/vllm.md): the vllm/vllm-openai:muse-glimmer image, as no
released vLLM wheel has the model; the recipe's single-card arguments; no --chat-template,
as the checkpoint ships chat_template.jinja (export.py copies the base's if the save wrote
none); parser names with underscores. vllm.md budgets 72 GB of VRAM for bf16: one 80 GB GPU.

Unverified: vllm.md calls this path "unverified end to end" and has no attested image
request; this image on Modal, and a merged fine-tune in place of Meta's checkpoint, have not
been tried. Send one image request (te watch on a short clip) before relying on it.
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
PARSER = "muse_glimmer"  # underscores: vLLM matches the name literally (vllm.md)
# vLLM refuses a request whose top_logprobs is above --max-logprobs. The scorer asks for 10
# (student.py). 20 is vLLM's usual default and the OpenAI API's top_logprobs limit; not
# checked against this image, so it is set explicitly.
MAX_LOGPROBS = 20
# From vllm.md's tighter-memory command. Training used sequence_len 4096 and the scorer asks
# for at most 512 new tokens, so a frame request should fit (not measured).
MAX_MODEL_LEN = 8300
RUN_ENV = "TE_SERVE_RUN"  # read at deploy time and baked into the image
RUN_NAME = re.compile(r"[a-z0-9][a-z0-9-]{1,39}")  # as te train checks it


def model_dir(run: str) -> str:
    """Where export.py saved the run's merged model."""
    if not RUN_NAME.fullmatch(run):
        raise ValueError(f"deploy with {RUN_ENV}=<run name>, e.g. exp1-mixed (got {run!r})")
    return f"{MOUNT}/runs/{run}/merged"


def serve_command(model: str) -> list[str]:
    """The vLLM command. It never holds the key: vLLM logs its arguments at startup, so the
    key goes in the VLLM_API_KEY environment variable instead (read when --api-key is not
    given)."""
    return ["vllm", "serve", model, "--served-model-name", SERVED_MODEL_NAME,
            # the recipe's single-card arguments (vllm.md); tool calling is unused here
            "--tensor-parallel-size", str(GPU_COUNT), "--enable-auto-tool-choice",
            "--tool-call-parser", PARSER, "--reasoning-parser", PARSER,
            "--generation-config", "auto",
            "--max-model-len", str(MAX_MODEL_LEN), "--max-logprobs", str(MAX_LOGPROBS),
            # all interfaces, as Modal's web_server routes to the port inside the container
            "--host", "0.0.0.0", "--port", str(PORT)]


app = modal.App(APP)
volume = modal.Volume.from_name(VOLUME)  # no create: export.py must have filled it
# The image's ENTRYPOINT is `vllm serve` (vllm.md); cleared so Modal can start the container.
image = (modal.Image.from_registry(VLLM_IMAGE).entrypoint([])
         .env({RUN_ENV: os.environ.get(RUN_ENV, "")}))


@app.function(image=image, gpu=f"{GPU_TYPE}:{GPU_COUNT}", volumes={MOUNT: volume},
              secrets=[modal.Secret.from_name(API_SECRET)], scaledown_window=SCALEDOWN_S,
              timeout=STARTUP_TIMEOUT_S)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT_S)
def serve() -> None:
    """Start vLLM in the background; Modal sends requests to PORT once it listens."""
    model = model_dir(os.environ.get(RUN_ENV, ""))
    if not Path(model, "config.json").is_file():
        raise RuntimeError(f"no merged model in {model}: run backend/training/export.py first")
    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        raise RuntimeError(f"the Modal Secret {API_SECRET!r} has no {API_KEY_ENV}")
    # a list, no shell; the key only in the child's environment, never on its command line
    subprocess.Popen(serve_command(model), env={**os.environ, "VLLM_API_KEY": api_key})
