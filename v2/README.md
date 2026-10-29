# Trained Eye

Teach your glasses to notice anything.

Meta glasses answer when you ask them something, but they can't keep an eye on
anything for you. With Trained Eye you look at something for about ten seconds and
say what to watch for ("let me know when the washer says done"). Meta's models label
that footage and generate training variations, a small open model (Muse Glimmer) is
fine-tuned on them, and from then on the glasses speak up when it happens.

Built for the Meta Global AI Developer Hackathon (October 29 to November 4, 2026).
All code here was written during the event.

## Build stage v2 (M0): Response cache

A content-addressed cache for Meta API calls. The key is a SHA-256 over the model, the
endpoint and the normalized request: keys are sorted, empty values dropped, and every
image or audio clip (raw bytes or a base64 data URL) is replaced by its own SHA-256,
so the same frame always gives the same key. A schema's property order is kept,
because strict structured output follows it. Generated images are stored byte for
byte, so Muse Image watermarks and provenance metadata survive. Also adds the Trained
Eye entries to .gitignore: data/, outputs/ and copies of .env.

Files added or changed in this stage:

- `.gitignore` (new): keeps secrets, local data and outputs out of git
- `backend/meta/cache.py` (new): content-addressed cache for Meta API responses: the
  key is a SHA-256 of model, endpoint and normalized request, with media replaced by its
  own hash
- `tests/test_cache.py` (new): tests
- `README.md` (changed): this file: what the stage adds and how to try it

## Setup

Run everything from inside this folder.

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"

## Try it

These commands were run inside this folder on 2026-10-29 (Python 3.13, Linux, no keys
set and no network to Meta); the output under each is what they printed.

    $ python -m pytest
    .....................                                                    [100%]
    21 passed in 0.18s

    $ ruff check .
    All checks passed!

    $ python -c "from backend.meta.cache import cache_key; print(cache_key('sam-3.1', '/responses', {'image': b'frame bytes'}))"
    ffca9d9969bbcdc5a85fd69435c9a8fd1569576c0ee296a79607ad0f2a1d62b3

## What is in the project so far

- `backend/config.py`: settings: model IDs for every job, prices, spend caps and data
  paths; secrets come from the environment or a local .env file and are never printed.
- `backend/meta/cache.py`: content-addressed cache for Meta API responses: the key is
  a SHA-256 of model, endpoint and normalized request, with media replaced by its own
  hash.
- `tests/`: 21 tests, all offline; none calls a paid API.

## Next

Stage v3 adds the spend ledger, with per-skill and global caps checked before every
paid call.

## Privacy and policy

- No skill may identify or track a specific person.
- Watching never hides or suppresses the glasses' capture LED.
- Frames are not stored unless the user saves a clip.
- Muse Image outputs keep their watermark and provenance metadata. Generated images
  are never committed; only the prompt manifest and a script to regenerate them.
- Faces are blurred in anything published.
