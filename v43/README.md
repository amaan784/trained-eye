# Trained Eye

Teach your glasses to notice anything.

Meta glasses answer when you ask them something, but they can't keep an eye on
anything for you. With Trained Eye you look at something for about ten seconds and
say what to watch for ("let me know when the washer says done"). Meta's models label
that footage and generate training variations, a small open model (Muse Glimmer) is
fine-tuned on them, and from then on the glasses speak up when it happens.

Built for the Meta Global AI Developer Hackathon (October 29 to November 4, 2026).
All code here was written during the event.

## Build stage v43 (M2): Teach sessions from the phone

backend/teach_session.py runs one teach started from the phone:
backend.teach.run_teach in a background thread, with its questions answered later over
HTTP. The spend question comes first, here, because run_teach's first paid step, the
transcription, comes before its only question, the read-back. A no to the read-back
that carries a correction, at most 300 characters, compiles the spec again with it
from the cached transcript, and each correction replaces the one before, as with te
teach --correct. A question nobody answers within 600 seconds counts as no. status()
gives the state (waiting for the spend, running, waiting for the read-back, then done,
stopped or failed), the open question, the progress lines, what was spent and, at the
end, a summary of the dataset and the labels. stop(), for when the server shuts down,
refuses the next paid call, cuts a retry's wait short, answers an open question no and
waits for the call in flight, so the ledger records it. Failed calls and spend caps
end the run with what to do next. Its HTTP routes come in v44.

Files added or changed in this stage:

- `backend/teach_session.py` (new): a phone teach session: the spend question before
  any paid call, the read-back, progress lines and timeouts
- `tests/test_teach_session.py` (new): tests
- `README.md` (changed): this file: what the stage adds and how to try it

## Setup

Run everything from inside this folder.

    python3.12 -m venv .venv && source .venv/bin/activate   # Mac: brew install python@3.12
    pip install -e ".[dev,train]"
    cp .env.example .env    # then fill in the keys

## Try it

These commands were run inside this folder on 2026-11-03 (Python 3.13, Linux, no keys
set and no network to Meta); the output under each is what they printed.

    $ python -m pytest
    ........................................................................ [ 10%]
    ........................................................................ [ 20%]
    ........................................................................ [ 31%]
    ........................................................................ [ 41%]
    ........................................................................ [ 52%]
    ........................................................................ [ 62%]
    ........................................................................ [ 73%]
    ........................................................................ [ 83%]
    ........................................................................ [ 94%]
    .........................................                                [100%]
    689 passed in 27.06s

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
- `backend/cli_train.py`: te train: the smoke, the priced run, detach and collect, the
  preflight.
- `backend/meta/client.py`: MetaClient, the only way the code calls Meta's API: cache
  first, then cap check, retries with backoff for 429, 5xx and timeouts, and spend
  recorded from the reported usage.
- `backend/server.py`: the backend the iPhone app talks to: JPEG frames arrive over a
  WebSocket at /ws and each one is acknowledged; frames stay in memory.
- `backend/teach_session.py`: a phone teach session: the spend question before any
  paid call, the read-back, progress lines and timeouts.
- `backend/training/wandb_smoke.py`: a Modal app that logs one Weights & Biases run,
  with the key coming only from a Modal Secret.
- `backend/training/axolotl.py`: the Axolotl config for a training run: Meta's QLoRA
  recipe with the run's paths, W&B names and gradient accumulation for one H100.
- `backend/training/prepare.py`: bundles taught skills into one training set for Modal
  (real, synthetic or mixed, with a validation split and volume paths).
- `backend/training/modal_train.py`: the Modal app that trains on one H100 with the
  recipe's pins, the trained-eye volume and the wandb and huggingface Secrets.
- `backend/skills/spec.py`: the skill spec (version 1), the local policy check that
  refuses skills about people, health, covert monitoring or sports, and the
  plain-English read-back.
- `backend/skills/compiler.py`: the skill compiler: the request and 3 keyframes go to
  Muse Wonder (structured output), which returns a spec or a refusal.
- `backend/dataengine/keyframes.py`: the keyframe picker: 20 to 30 sharp, distinct
  frames from the 10 second teaching clip.
- `backend/dataengine/dataset.py`: the Axolotl dataset writer, in Meta's recipe format
  (image path part, then text; 512x512 padded PNGs).
- `backend/dataengine/sam_text.py`: our reader of SAM 3.1's output text, written from
  the grammar in Meta's SAM 3 protocol: object ids, boxes and confidences.
- `backend/dataengine/roi.py`: the region of interest: SAM 3.1 on each keyframe, its
  boxes scaled to the keyframe, and a crop around each box for the dataset.
- `backend/dataengine/synth.py`: the synthetic set: a deterministic plan of Muse Image
  edits of the keyframes (positives, hard negatives, nuisance variation), saved
  untouched with a manifest.
- `backend/dataengine/labels.py`: Muse Spark labels for every image, the filter that
  keeps generated images only when Spark agrees, and agreement stats.
- `backend/teach.py`: the teach flow end to end, cached so a rerun costs nothing.
- `backend/watch/trigger.py`: when a skill fires: 3 of the last 4 frames over the
  threshold, then the cooldown; the threshold calibration on labelled clips.
- `backend/inference/student.py`: scores a frame with the trained model on an
  OpenAI-compatible endpoint, read at its yes or no answer token.
- `backend/watch/watcher.py`: watch mode: scores, the Muse Spark check before an
  alert, alert messages and the frame log.
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
- `tests/`: 689 tests, all offline; none calls a paid API.

## Next

Stage v44 adds the phone's teach API: the capture uploaded, checked and priced, then
the spend and read-back questions answered over HTTP, each request with the server
token.

## Privacy and policy

- No skill may identify or track a specific person.
- Watching never hides or suppresses the glasses' capture LED.
- Frames are not stored unless the user saves a clip.
- Muse Image outputs keep their watermark and provenance metadata. Generated images
  are never committed; only the prompt manifest and a script to regenerate them.
- Faces are blurred in anything published.
