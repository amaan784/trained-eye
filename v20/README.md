# Trained Eye

Teach your glasses to notice anything.

Meta glasses answer when you ask them something, but they can't keep an eye on
anything for you. With Trained Eye you look at something for about ten seconds and
say what to watch for ("let me know when the washer says done"). Meta's models label
that footage and generate training variations, a small open model (Muse Glimmer) is
fine-tuned on them, and from then on the glasses speak up when it happens.

Built for the Meta Global AI Developer Hackathon (October 29 to November 4, 2026).
All code here was written during the event.

## Build stage v20 (M1): te teach

The te teach command: it measures the clip and the audio, prints a cost estimate per
step, asks before spending, reads the spec back and asks again, then runs the teach
flow and prints the spend report. --estimate-only stops after the estimate, --labels
takes your own keyframe labels, and --yes skips both questions. A skill name that is
not snake_case is rejected first.

Files added or changed in this stage:

- `backend/cli.py` (changed): the te command line
- `tests/test_cli.py` (changed): tests
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
    ........................................................................ [ 38%]
    ........................................................................ [ 77%]
    .........................................                                [100%]
    185 passed in 4.55s

    $ ruff check .
    All checks passed!

    $ ffmpeg -v error -y -f lavfi -i testsrc=duration=10:size=1280x720:rate=30 -pix_fmt yuv420p /tmp/te-demo.mp4

    $ python -m backend.cli teach --skill washer_done --clip /tmp/te-demo.mp4 --request "let me know when the washer says done" --estimate-only
    estimate for washer_done (before cache hits): transcribe $0.00, compile $0.10, sam $0.03, images $3.00, labels $2.05, total $5.18

The estimate is an upper bound: each label is priced at its full output budget and
each image at 1,500 input tokens, since the real image token count is not known yet.
Video clips are decoded with ffmpeg (on the Mac: brew install ffmpeg). The demo clip
is ffmpeg's built-in test pattern, written to /tmp, outside the folder.

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
- `backend/skills/compiler.py`: the skill compiler: the request and 3 keyframes go to
  Muse Wonder (structured output), which returns a spec or a refusal.
- `backend/dataengine/keyframes.py`: the keyframe picker: 20 to 30 sharp, distinct
  frames from the 10 second teaching clip.
- `backend/dataengine/dataset.py`: the Axolotl dataset writer, in Meta's recipe format
  (image path part, then text; 512x512 padded PNGs).
- `backend/dataengine/roi.py`: the region of interest: SAM 3.1 on each keyframe; the
  raw output is kept and parsing is still to come.
- `backend/dataengine/synth.py`: the synthetic set: a deterministic plan of Muse Image
  edits of the keyframes (positives, hard negatives, nuisance variation), saved
  untouched with a manifest.
- `backend/dataengine/labels.py`: Muse Spark labels for every image, the filter that
  keeps generated images only when Spark agrees, and agreement stats.
- `backend/teach.py`: the teach flow end to end, cached so a rerun costs nothing.
- `docs/glasses-check.md`: how to check, with Meta's own sample app, that the glasses
  stream camera frames, deliver the mic and play audio.
- `PROGRESS.md`: milestone checklist, decisions, docs-vs-spec differences, open issues
  and spend.
- `CLAUDE.md`: short setup, commands and conventions.
- `tests/`: 185 tests, all offline; none calls a paid API.

## Next

Stage v21 adds the recording protocol for real clips and the updated project notes.

## Privacy and policy

- No skill may identify or track a specific person.
- Watching never hides or suppresses the glasses' capture LED.
- Frames are not stored unless the user saves a clip.
- Muse Image outputs keep their watermark and provenance metadata. Generated images
  are never committed; only the prompt manifest and a script to regenerate them.
- Faces are blurred in anything published.
