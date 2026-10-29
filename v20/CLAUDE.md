# Trained Eye

Teach your glasses to notice anything. Build spec: SPEC.md (kept outside the repo).
PROGRESS.md is the source of truth between sessions: read it first.

## Setup (MacBook)

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"          # add ,train for Modal and W&B
    cp .env.example .env             # then fill in the keys

## Commands

    pytest                           # tests; never call paid APIs
    ruff check .
    te spend                         # Meta API spend from data/ledger.jsonl
    te serve                         # backend, WebSocket at ws://<mac>:8765/ws
    te ws-selftest                   # synthetic frames through the WebSocket, in process
    te api-smoke                     # one tiny paid Muse Spark call (needs MODEL_API_KEY)
    modal run backend/training/wandb_smoke.py   # one W&B run from Modal

## Conventions

- Model IDs and prices live in backend/config.py. Never use `-contributor` models.
- Every Meta call goes through backend/meta/client.py (cache + spend ledger + retries).
- Secrets: .env locally, Modal Secrets ("wandb", "huggingface") in Modal jobs. Never print them.
- Local data (footage, cache, ledger, generated images, weights) lives in data/ and is gitignored.
- Packages under backend/ are namespace packages (no __init__.py files).
- Ask before any job over $15, before publishing anything, and before anything physical.
- Policy: no skills about specific people; never hide the capture LED; keep Muse Image
  provenance; blur faces in anything published.
- Numbers in docs come only from eval scripts.

## Pointers

- docs/glasses-check.md: how to verify the glasses with Meta's DAT sample app.
- Meta cookbook: github.com/meta-models/meta-oss-cookbook (muse-glimmer/post-training).
- DAT iOS SDK: github.com/facebook/meta-wearables-dat-ios (samples/CameraAccess).
