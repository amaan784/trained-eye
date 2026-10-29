# Trained Eye

Teach your glasses to notice anything.

Meta glasses answer when you ask them something, but they can't keep an eye on
anything for you. With Trained Eye you look at something for about ten seconds and
say what to watch for ("let me know when the washer says done"). Meta's models label
that footage and generate training variations, a small open model (Muse Glimmer) is
fine-tuned on them, and from then on the glasses speak up when it happens.

Built for the Meta Global AI Developer Hackathon (October 29 to November 4, 2026).
All code here was written during the event.

## Build stage v12 (M1): Skill spec and policy check

The skill spec as a versioned pydantic model (target with its SAM prompt, the positive
condition, near-miss negatives, the alert, cooldown, cloud check, refusal), the
plain-English read-back the user confirms, and a local policy check. The check
refuses, before any API call and at no cost, requests that would identify or track a
person (by name, pronoun or role), concern bodies, health or medication, monitor
someone covertly, watch a sensitive place, or coach sports. Words that also name
objects only count inside phrases, so walk signs, kitchen scales and meat thermometers
pass. te policy runs it on any sentence.

Files added or changed in this stage:

- `backend/cli.py` (changed): the te command line
- `backend/skills/spec.py` (new): the skill spec (version 1), the local policy check
  that refuses skills about people, health, covert monitoring or sports, and the
  plain-English read-back
- `pyproject.toml` (changed): requirements and tool settings
- `tests/test_skills.py` (new): tests
- `README.md` (changed): this file: what the stage adds and how to try it

## Setup

Run everything from inside this folder.

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev,train]"
    cp .env.example .env    # then fill in the keys

## Try it

These commands were run inside this folder on 2026-10-29 (Python 3.13, Linux, no keys
set and no network to Meta); the output under each is what they printed.

    $ python -m pytest
    ........................................................................ [ 47%]
    ........................................................................ [ 95%]
    .......                                                                  [100%]
    151 passed in 1.41s

    $ ruff check .
    All checks passed!

    $ python -m backend.cli policy "Tell me when Sam gets here"
    refused: I can't watch for or recognize specific people. Skills can cover objects, states, text, displays, signs and vehicles.
    (exit code 1)

    $ python -m backend.cli policy "let me know when the washer says done"
    ok: no local policy objection (the skill compiler checks again)

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
- `backend/training/wandb_smoke.py`: a Modal app that logs one Weights & Biases run,
  with the key coming only from a Modal Secret.
- `backend/skills/spec.py`: the skill spec (version 1), the local policy check that
  refuses skills about people, health, covert monitoring or sports, and the
  plain-English read-back.
- `docs/glasses-check.md`: how to check, with Meta's own sample app, that the glasses
  stream camera frames, deliver the mic and play audio.
- `PROGRESS.md`: milestone checklist, decisions, docs-vs-spec differences, open issues
  and spend.
- `CLAUDE.md`: short setup, commands and conventions.
- `tests/`: 151 tests, all offline; none calls a paid API.

## Next

Stage v13 adds the skill compiler, which turns a request and 3 keyframes into a spec
with Muse Wonder.

## Privacy and policy

- No skill may identify or track a specific person.
- Watching never hides or suppresses the glasses' capture LED.
- Frames are not stored unless the user saves a clip.
- Muse Image outputs keep their watermark and provenance metadata. Generated images
  are never committed; only the prompt manifest and a script to regenerate them.
- Faces are blurred in anything published.
