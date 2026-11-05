"""The public playground on Modal: create_app(public=True) behind a Modal web endpoint.

`te playground --deploy` asks first (it publishes the site), copies what the public copy
shows (backend/playground/publish.py: never a generated image, an original keyframe or an
unblurred video) to the private Volume below, then runs, from the repo root:

    modal deploy backend/playground/modal_app.py

Modal prints the URL; `modal app stop trained-eye-playground` takes the site down.

The Volume (private to the Modal workspace) holds data/ and eval/ under /vol. The site's
own state lives there too: data/playground/budget.json, the ledger and the call cache,
committed to the Volume every COMMIT_S seconds while they change. One container at most:
the budget file belongs to one process. MODEL_API_KEY comes from the Modal Secret "meta"
(create it in the Modal dashboard under Secrets; the key never passes through this code).

The caps are read when deploying and baked into the image: per visitor and per day (UTC),
and the all-time cap of the ledger's skill "playground" (TE_SKILL_CAP_USD; this Volume's
ledger holds only playground calls, so TE_GLOBAL_CAP_USD is set to the same).

Unverified on Modal: the visitor's address. behind_proxy() takes the last
X-Forwarded-For entry, the visitor's if Modal's proxy appends it. After deploying, check
from two networks that /api/status shows separate visitor budgets before relying on the
per-visitor cap; the daily and all-time caps hold either way. Also unverified: that
/root (where backend/ is copied) is on the container's import path (PYTHONPATH says so).
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import modal

APP = "trained-eye-playground"
VOLUME = "trained-eye-playground"  # not the training Volume: no weights or bundles here
MOUNT = "/vol"
DATA_DIR = f"{MOUNT}/data"
EVAL_DIR = f"{MOUNT}/eval"
API_SECRET = "meta"  # holds MODEL_API_KEY
PYTHON = "3.12"
# pyproject.toml's dependencies (tests/test_playground_modal.py keeps them the same)
PACKAGES = ("httpx>=0.27", "numpy>=1.26", "pillow>=10", "pydantic>=2.6", "starlette>=0.37",
            "uvicorn[standard]>=0.30")
MAX_CONTAINERS = 1  # the budget file belongs to one process
MAX_INPUTS = 64  # our choice: requests one container takes at once (it is async)
SCALEDOWN_S = 5 * 60  # our choice: a run takes a minute or two, the page polls meanwhile
COMMIT_S = 30  # our choice: how often the budget, ledger and cache are saved to the Volume
VISITOR_CAP_ENV = "TE_PLAYGROUND_VISITOR_CAP"
DAILY_CAP_ENV = "TE_PLAYGROUND_DAILY_CAP"
CAP_ENVS = (VISITOR_CAP_ENV, DAILY_CAP_ENV, "TE_SKILL_CAP_USD", "TE_GLOBAL_CAP_USD")
# The checkout (parent thrice, not parents[2], which fails in the container's /root).
ROOT = Path(__file__).resolve().parent.parent.parent

app = modal.App(APP)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = (modal.Image.debian_slim(python_version=PYTHON).pip_install(*PACKAGES)
         .env({"TE_DATA_DIR": DATA_DIR, "PYTHONPATH": "/root",
               **{name: os.environ.get(name, "") for name in CAP_ENVS}})
         .add_local_dir(ROOT / "backend", remote_path="/root/backend",
                        ignore=["**/__pycache__/**", "**/*.pyc"])
         .add_local_file(ROOT / "docs" / "how-it-works.html",
                         remote_path="/root/docs/how-it-works.html"))


def state_files() -> tuple[str, ...]:
    """What a live run changes on the Volume."""
    return (f"{DATA_DIR}/playground/budget.json", f"{DATA_DIR}/ledger.jsonl",
            f"{DATA_DIR}/cache")


def stamp() -> tuple:
    out = []
    for path in state_files():
        try:
            st = os.stat(path)
            out.append((st.st_mtime_ns, st.st_size))
        except OSError:
            out.append(None)
    return tuple(out)


def keep_committing() -> None:
    """Commit the Volume every COMMIT_S seconds while the state files change, so a new
    container starts from the same budget and ledger."""
    def loop() -> None:
        last = stamp()
        while True:
            time.sleep(COMMIT_S)
            now = stamp()
            if now != last:
                volume.commit()
                last = now

    threading.Thread(target=loop, name="volume-commit", daemon=True).start()


def cap(name: str) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        raise RuntimeError(f"{name} is not set: deploy with te playground --deploy")
    return float(raw)


@app.function(image=image, volumes={MOUNT: volume}, secrets=[modal.Secret.from_name(API_SECRET)],
              max_containers=MAX_CONTAINERS, scaledown_window=SCALEDOWN_S)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.asgi_app()
def web():
    """The public copy: no generated images, only face-blurred keyframes and clips."""
    from backend.config import load_settings
    from backend.meta.client import MetaClient
    from backend.playground.app import behind_proxy, create_app

    keep_committing()
    site = create_app(load_settings().data_dir, Path(EVAL_DIR), lambda: MetaClient(load_settings()),
                      visitor_cap_usd=cap(VISITOR_CAP_ENV), daily_cap_usd=cap(DAILY_CAP_ENV),
                      public=True)
    return behind_proxy(site)
