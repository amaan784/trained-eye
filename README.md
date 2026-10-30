# Trained Eye

Teach your glasses to notice anything.

Meta glasses answer when you ask them something, but they can't keep an eye on
anything for you. With Trained Eye you look at something for about ten seconds and
say what to watch for ("let me know when the washer says done"). Meta's models label
that footage and generate training variations, a small open model (Muse Glimmer) is
fine-tuned on them, and from then on the glasses speak up when it happens.

Built for the Meta Global AI Developer Hackathon (October 29 to November 4, 2026).
All code here was written during the event.

## Build stage v3 (M0): Spend ledger

The spend ledger: an append-only JSONL file with one entry per paid call (model,
endpoint, skill, estimated and actual cost, usage). Before a call, its estimate is
checked against the per-skill and global caps; after it, the actual cost is recorded.
Includes the cost helpers for token-priced and unit-priced models and the spend
report.

Files added or changed in this stage:

- `backend/meta/ledger.py` (new): spend ledger: estimated cost checked against the
  per-skill ($10) and global ($120) caps before each call, actual cost recorded after
- `tests/test_ledger.py` (new): tests
- `README.md` (changed): this file: what the stage adds and how to try it

## Setup

Run everything from inside this folder.

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"

## Try it

These commands were run inside this folder on 2026-10-29 (Python 3.13, Linux, no keys
set and no network to Meta); the output under each is what they printed.

    $ python -m pytest
    .........................                                                [100%]
    25 passed in 0.23s

    $ ruff check .
    All checks passed!

    $ python -c "from backend.meta.ledger import Ledger, format_report; print(format_report(Ledger('data/ledger.jsonl', 10, 120)))"
    Meta API spend (ledger: data/ledger.jsonl)
      calls: 0  errors: 0
      actual: $0.0000  (estimated before calls: $0.0000)
      global cap: $120.00  remaining: $120.0000
      per-skill cap: $10.00
      by model:
        (no calls yet)
      by skill:
        (no calls yet)

## What is in the project so far

- `backend/config.py`: settings: model IDs for every job, prices, spend caps and data
  paths; secrets come from the environment or a local .env file and are never printed.
- `backend/meta/cache.py`: content-addressed cache for Meta API responses: the key is
  a SHA-256 of model, endpoint and normalized request, with media replaced by its own
  hash.
- `backend/meta/ledger.py`: spend ledger: estimated cost checked against the per-skill
  ($10) and global ($120) caps before each call, actual cost recorded after.
- `tests/`: 25 tests, all offline; none calls a paid API.

## Next

Stage v4 adds the te command line with a spend report, and test isolation from your
shell settings.

## Privacy and policy

- No skill may identify or track a specific person.
- Watching never hides or suppresses the glasses' capture LED.
- Frames are not stored unless the user saves a clip.
- Muse Image outputs keep their watermark and provenance metadata. Generated images
  are never committed; only the prompt manifest and a script to regenerate them.
- Faces are blurred in anything published.
