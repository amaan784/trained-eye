# Trained Eye

Teach your glasses to notice anything.

Meta glasses answer when you ask them something, but they can't keep an eye on
anything for you. With Trained Eye you look at something for about ten seconds and
say what to watch for ("let me know when the washer says done"). Meta's models label
that footage and generate training variations, a small open model (Muse Glimmer) is
fine-tuned on them, and from then on the glasses speak up when it happens.

Built for the Meta Global AI Developer Hackathon (October 29 to November 4, 2026).
All code here was written during the event.

## Build stage v7 (M0): Key check commands

Two commands for the first live check with a real key: te models (free) lists the
models the key can use and names any configured model that is missing, and te
api-smoke makes one tiny Muse Spark call through the cache and the ledger (under
$0.01; a second run is a free cache hit). Configuration and API errors now end with a
one-line message instead of a traceback. Adds .env.example.

Files added or changed in this stage:

- `.env.example` (new): template for the keys (copy to .env)
- `backend/cli.py` (changed): the te command line
- `tests/test_cli.py` (new): tests
- `README.md` (changed): this file: what the stage adds and how to try it

## Setup

Run everything from inside this folder.

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    cp .env.example .env    # then fill in the keys

## Try it

These commands were run inside this folder on 2026-10-29 (Python 3.13, Linux, no keys
set and no network to Meta); the output under each is what they printed.

    $ python -m pytest
    ................................................................         [100%]
    64 passed in 0.47s

    $ ruff check .
    All checks passed!

    $ python -m backend.cli models
    te: MODEL_API_KEY is not set. Add it to .env (see .env.example).
    (exit code 2)

With a real MODEL_API_KEY in .env, te models lists the model IDs the key can use
(free), and te api-smoke makes one call to muse-spark-1.3 costing at most about
$0.0024. Neither was run here: the build machine had no key and no network access to
Meta.

After `pip install -e .`, `te <command>` is the same as `python -m backend.cli <command>`.

## What is in the project so far

- `backend/config.py`: settings: model IDs for every job, prices, spend caps and data
  paths; secrets come from the environment or a local .env file and are never printed.
- `backend/meta/cache.py`: content-addressed cache for Meta API responses: the key is
  a SHA-256 of model, endpoint and normalized request, with media replaced by its own
  hash.
- `backend/meta/ledger.py`: spend ledger: estimated cost checked against the per-skill
  ($10) and global ($120) caps before each call, actual cost recorded after.
- `backend/cli.py`: the te command line.
- `backend/meta/client.py`: MetaClient, the only way the code calls Meta's API: cache
  first, then cap check, retries with backoff for 429, 5xx and timeouts, and spend
  recorded from the reported usage.
- `tests/`: 64 tests, all offline; none calls a paid API.

## Next

Stage v8 adds the WebSocket backend that receives camera frames from the phone.

## Privacy and policy

- No skill may identify or track a specific person.
- Watching never hides or suppresses the glasses' capture LED.
- Frames are not stored unless the user saves a clip.
- Muse Image outputs keep their watermark and provenance metadata. Generated images
  are never committed; only the prompt manifest and a script to regenerate them.
- Faces are blurred in anything published.
