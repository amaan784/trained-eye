# Trained Eye

Teach your glasses to notice anything.

Meta glasses answer when you ask them something, but they can't keep an eye on
anything for you. With Trained Eye you look at something for about ten seconds and
say what to watch for ("let me know when the washer says done"). Meta's models label
that footage and generate training variations, a small open model (Muse Glimmer) is
fine-tuned on them, and from then on the glasses speak up when it happens.

Built for the Meta Global AI Developer Hackathon (October 29 to November 4, 2026).
All code here was written during the event.

## Build stage v4 (M0): te command: spend

The te command line with its first subcommand, te spend, which prints the ledger
report. Also adds a test fixture that removes every TE_* variable and every key from
the environment before each test, so a developer's own shell settings can't change
test results.

Files added or changed in this stage:

- `backend/cli.py` (new): the te command line
- `pyproject.toml` (changed): requirements and tool settings
- `tests/conftest.py` (new): tests
- `tests/test_ledger.py` (changed): tests
- `README.md` (changed): this file: what the stage adds and how to try it

## Setup

Run everything from inside this folder.

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"

## Try it

These commands were run inside this folder on 2026-10-29 (Python 3.13, Linux, no keys
set and no network to Meta); the output under each is what they printed.

    $ python -m pytest
    ..........................                                               [100%]
    26 passed in 0.19s

    $ ruff check .
    All checks passed!

    $ python -m backend.cli spend
    Meta API spend (ledger: data/ledger.jsonl)
      calls: 0  errors: 0
      actual: $0.0000  (estimated before calls: $0.0000)
      global cap: $120.00  remaining: $120.0000
      per-skill cap: $10.00
      by model:
        (no calls yet)
      by skill:
        (no calls yet)

te spend reads data/ledger.jsonl; no paid call has been made yet, so it shows zeros.

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
- `tests/`: 26 tests, all offline; none calls a paid API.

## Next

Stage v5 adds the Meta API client: structured output, retries with backoff and
conservative spend accounting.

## Privacy and policy

- No skill may identify or track a specific person.
- Watching never hides or suppresses the glasses' capture LED.
- Frames are not stored unless the user saves a clip.
- Muse Image outputs keep their watermark and provenance metadata. Generated images
  are never committed; only the prompt manifest and a script to regenerate them.
- Faces are blurred in anything published.
