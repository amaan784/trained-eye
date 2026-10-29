# Trained Eye

Teach your glasses to notice anything.

Meta glasses answer when you ask them something, but they can't keep an eye on
anything for you. With Trained Eye you look at something for about ten seconds and
say what to watch for ("let me know when the washer says done"). Meta's models label
that footage and generate training variations, a small open model (Muse Glimmer) is
fine-tuned on them, and from then on the glasses speak up when it happens.

Built for the Meta Global AI Developer Hackathon (October 29 to November 4, 2026).
All code here was written during the event.

## Build stage v5 (M0): Meta API client

MetaClient, the single place that calls the Meta Model API. Every call looks in the
cache first (a hit makes no network call and costs nothing), checks the caps, sends
the request and retries 429s, 5xx errors and timeouts with backoff (honouring
Retry-After), but never a 4xx. Spend is counted conservatively: a retried attempt that
may have reached the server is charged at its estimate, and so is a reply without
usable usage. This stage has structured output (Chat Completions with a strict
json_schema, as in Meta's cookbook) and model listing. The key only goes to the Meta
base URL and is removed from any error text.

Files added or changed in this stage:

- `backend/meta/client.py` (new): MetaClient, the only way the code calls Meta's API:
  cache first, then cap check, retries with backoff for 429, 5xx and timeouts, and spend
  recorded from the reported usage
- `pyproject.toml` (changed): requirements and tool settings
- `tests/test_client.py` (new): tests
- `README.md` (changed): this file: what the stage adds and how to try it

## Setup

Run everything from inside this folder.

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"

## Try it

These commands were run inside this folder on 2026-10-29 (Python 3.13, Linux, no keys
set and no network to Meta); the output under each is what they printed.

    $ python -m pytest
    ..................................................                       [100%]
    50 passed in 0.43s

    $ ruff check .
    All checks passed!

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
- `tests/`: 50 tests, all offline; none calls a paid API.

## Next

Stage v6 adds the media calls: Voice Transcribe, SAM 3.1 and Muse Image.

## Privacy and policy

- No skill may identify or track a specific person.
- Watching never hides or suppresses the glasses' capture LED.
- Frames are not stored unless the user saves a clip.
- Muse Image outputs keep their watermark and provenance metadata. Generated images
  are never committed; only the prompt manifest and a script to regenerate them.
- Faces are blurred in anything published.
