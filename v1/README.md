# Trained Eye

Teach your glasses to notice anything.

Meta glasses answer when you ask them something, but they can't keep an eye on
anything for you. With Trained Eye you look at something for about ten seconds and
say what to watch for ("let me know when the washer says done"). Meta's models label
that footage and generate training variations, a small open model (Muse Glimmer) is
fine-tuned on them, and from then on the glasses speak up when it happens.

Built for the Meta Global AI Developer Hackathon (October 29 to November 4, 2026).
All code here was written during the event.

## Build stage v1 (M0): Settings

The project skeleton and the settings module every later layer reads: one model ID per
job (never a '-contributor' variant), Meta's prices, the spend caps, the data folder,
and the three keys, read from the environment or a local .env file. Keys are never
printed, only whether each one is set. pyproject.toml holds the requirements and the
pytest and ruff settings, and the CI workflow runs ruff and pytest on every push.

Files added or changed in this stage:

- `.github/workflows/ci.yml` (new): CI: ruff and pytest on every push
- `backend/config.py` (new): settings: model IDs for every job, prices, spend caps and
  data paths; secrets come from the environment or a local .env file and are never
  printed
- `pyproject.toml` (new): requirements and tool settings
- `tests/test_config.py` (new): tests
- `README.md` (new): this file: what the stage adds and how to try it

## Setup

Run everything from inside this folder.

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"

## Try it

These commands were run inside this folder on 2026-10-29 (Python 3.13, Linux, no keys
set and no network to Meta); the output under each is what they printed.

    $ python -m pytest
    .............                                                            [100%]
    13 passed in 0.17s

    $ ruff check .
    All checks passed!

    $ python -m backend.config
    Trained Eye settings
      base_url: https://api.meta.ai/v1
      data_dir: data
      spend caps: $10.00 per skill, $120.00 global
      models:
        transcribe    muse-voice-transcribe-1.0
        compiler      muse-wonder-1.0
        sam           sam-3.1
        image         muse-image-1.0
        labeler       muse-spark-1.3
        confirm       muse-spark-1.3
        student_base  meta-models/Muse-Glimmer-30B
                      (student_base revision a4e59da52a7b)
      secrets:
        MODEL_API_KEY  missing
        HF_TOKEN       missing
        WANDB_API_KEY  missing

## What is in the project so far

- `backend/config.py`: settings: model IDs for every job, prices, spend caps and data
  paths; secrets come from the environment or a local .env file and are never printed.
- `tests/`: 13 tests, all offline; none calls a paid API.

## Next

Stage v2 adds the response cache: identical Meta API requests are answered from disk
for free.

## Privacy and policy

- No skill may identify or track a specific person.
- Watching never hides or suppresses the glasses' capture LED.
- Frames are not stored unless the user saves a clip.
- Muse Image outputs keep their watermark and provenance metadata. Generated images
  are never committed; only the prompt manifest and a script to regenerate them.
- Faces are blurred in anything published.
