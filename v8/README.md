# Trained Eye

Teach your glasses to notice anything.

Meta glasses answer when you ask them something, but they can't keep an eye on
anything for you. With Trained Eye you look at something for about ten seconds and
say what to watch for ("let me know when the washer says done"). Meta's models label
that footage and generate training variations, a small open model (Muse Glimmer) is
fine-tuned on them, and from then on the glasses speak up when it happens.

Built for the Meta Global AI Developer Hackathon (October 29 to November 4, 2026).
All code here was written during the event.

## Build stage v8 (M0): WebSocket frame server

The backend the iPhone app will stream to: a Starlette app with /health and a
WebSocket at /ws (protocol 1: a hello, then one JPEG frame per binary message, each
acknowledged with its size). Bad frames get an error and the session carries on.
Frames are decoded only to check them, never written to disk, and only their metadata
is logged. Adds te serve and te ws-selftest.

Files added or changed in this stage:

- `backend/cli.py` (changed): the te command line
- `backend/server.py` (new): the backend the iPhone app talks to: JPEG frames arrive
  over a WebSocket at /ws and each one is acknowledged; frames stay in memory
- `pyproject.toml` (changed): requirements and tool settings
- `tests/test_server.py` (new): tests
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
    ........................................................................ [ 86%]
    ...........                                                              [100%]
    83 passed in 1.06s

    $ ruff check .
    All checks passed!

    $ python -m backend.cli ws-selftest
    connected: session s1, protocol 1
    frame 1: ack 640x480, 5427 bytes
    frame 2: ack 640x480, 5426 bytes
    frame 3: ack 640x480, 5426 bytes
    sent 3 frames, got 3 acks

    $ python -m backend.cli serve
    Trained Eye backend: ws://0.0.0.0:8765/ws  (Ctrl+C to stop)

    $ curl -s http://127.0.0.1:8765/health    # in a second terminal
    {"ok":true,"protocol":1}

te serve keeps running until you press Ctrl+C; the curl line was run in a second
terminal while it was up. The WebSocket at /ws needs the websockets package, which pip
install brings in through uvicorn[standard]. The build machine did not have it, so
there the WebSocket was exercised in process by te ws-selftest and the tests.

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
- `backend/server.py`: the backend the iPhone app talks to: JPEG frames arrive over a
  WebSocket at /ws and each one is acknowledged; frames stay in memory.
- `tests/`: 83 tests, all offline; none calls a paid API.

## Next

Stage v9 adds a Modal app that logs one Weights & Biases run.

## Privacy and policy

- No skill may identify or track a specific person.
- Watching never hides or suppresses the glasses' capture LED.
- Frames are not stored unless the user saves a clip.
- Muse Image outputs keep their watermark and provenance metadata. Generated images
  are never committed; only the prompt manifest and a script to regenerate them.
- Faces are blurred in anything published.
