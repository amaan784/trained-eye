# Trained Eye

Teach your glasses to notice anything.

Meta glasses answer when you ask them something, but they can't keep an eye on
anything for you. With Trained Eye you look at something for about ten seconds and
say what to watch for ("let me know when the washer says done"). Meta's models label
that footage and generate training variations, a small open model (Muse Glimmer) is
fine-tuned on them, and from then on the glasses speak up when it happens.

Built for the Meta Global AI Developer Hackathon (October 29 to November 4, 2026).
All code here was written during the event.

## Build stage v36 (M1 review): Fix: teach flow and te teach

te teach now checks the clip, the audio file, the request (its length and the local
policy) and the labels file before it shows the estimate, and the key before it asks;
a mistake is one 'te:' line, never a traceback after you said yes. The estimate adds a
worst case for labels that need the larger budget. A dataset that would hold only yes
or only no stops the run, and the dataset line shows both counts. With training on
model outputs off, your labels must include both answers. A new spec removes the old
dataset, the clip may not be the skill's own keyframes folder, only keyframe files are
cleaned up, contradicting labels are refused, and labels are saved when a run stops.
After a stop for too many failures, te teach says a rerun pays for them again. te
serve prints the address for the phone and keeps the server's own log lines.

Files added or changed in this stage:

- `backend/cli.py` (changed): the te command line
- `backend/teach.py` (changed): the teach flow end to end, cached so a rerun costs
  nothing
- `tests/test_cli.py` (changed): tests
- `tests/test_teach.py` (changed): tests
- `README.md` (changed): this file: what the stage adds and how to try it

## Setup

Run everything from inside this folder.

    python3.12 -m venv .venv && source .venv/bin/activate   # Mac: brew install python@3.12
    pip install -e ".[dev,train]"
    cp .env.example .env    # then fill in the keys

## Try it

These commands were run inside this folder on 2026-10-31 (Python 3.13, Linux, no keys
set and no network to Meta); the output under each is what they printed.

    $ python -m pytest
    ........................................................................ [ 18%]
    ........................................................................ [ 36%]
    ........................................................................ [ 54%]
    ........................................................................ [ 72%]
    ........................................................................ [ 90%]
    ........................................                                 [100%]
    400 passed in 7.47s

    $ ruff check .
    All checks passed!

    $ ffmpeg -v error -y -f lavfi -i testsrc=duration=10:size=1280x720:rate=30 -pix_fmt yuv420p /tmp/te-demo.mp4

    $ python -m backend.cli teach --skill washer_done --clip /tmp/te-demo.mp4 --request "let me know when the washer says done" --estimate-only
    estimate for washer_done (before cache hits): transcribe $0.00, compile $0.10, sam $0.03, images $3.00, labels $2.05, total $5.18
      at most $11.32 if every label has to be asked again with a larger budget
    train on model outputs: yes
    te: note: $10.00 is left under the spend caps for washer_done. If many labels need the larger budget the run could pass that, and the cap would stop it partway (a rerun with a higher cap continues).

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
- `docs/recording-protocol.md`: what real clips to record for the three skills, and
  how to keep training and test sessions apart.
- `PROGRESS.md`: milestone checklist, decisions, docs-vs-spec differences, open issues
  and spend.
- `CLAUDE.md`: short setup, commands and conventions.
- `HANDOVER.md`: how to pick the project up in a new session: state, the author's
  rules, open items with exact commands, and the plan for M2 to M5 with the facts from
  the docs.
- `SETUP.md`: accounts and keys, setting up the Mac, setting up a cloud session, and
  git on the stage branch.
- `tests/`: 400 tests, all offline; none calls a paid API.

## Next

Stage v37 adds CI and setup: ffmpeg in CI, Python 3.12 and ffmpeg in the Mac setup,
and the W&B team note.

## Privacy and policy

- No skill may identify or track a specific person.
- Watching never hides or suppresses the glasses' capture LED.
- Frames are not stored unless the user saves a clip.
- Muse Image outputs keep their watermark and provenance metadata. Generated images
  are never committed; only the prompt manifest and a script to regenerate them.
- Faces are blurred in anything published.
