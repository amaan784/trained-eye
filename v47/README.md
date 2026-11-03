# Trained Eye

Teach your glasses to notice anything.

Meta glasses answer when you ask them something, but they can't keep an eye on
anything for you. With Trained Eye you look at something for about ten seconds and
say what to watch for ("let me know when the washer says done"). Meta's models label
that footage and generate training variations, a small open model (Muse Glimmer) is
fine-tuned on them, and from then on the glasses speak up when it happens.

Built for the Meta Global AI Developer Hackathon (October 29 to November 4, 2026).
All code here was written during the event.

## Build stage v47 (M2): Merging and checking the adapter

backend/training/export.py merges a run's QLoRA adapter into the bf16 base on Modal,
on one H100: the base at the pinned revision, the adapter from runs/<run>/out on the
volume, merged with peft's merge_and_unload() and saved as safetensors with the
processor in runs/<run>/merged, the folder the serving app will serve. It then reloads
that folder and compares its yes or no answers with base plus adapter's on up to 50
examples of the run's bundle, the validation split if there is one. By default the
reference is the bf16 base with the adapter; --reference nf4 compares with the NF4
base that training saw. Loading and prompting copy the cookbook's evaluator, the
base's small files the save did not write, such as chat_template.jinja, are copied
over, and tensor names are compared with the base's. backend/training/export_check.py
is the plain Python part, run on the Mac: the examples in the evaluator's fixed order,
the answer read after "to=user", and the agreement, where an unreadable answer never
agrees. The result records the remote call's seconds for the Modal ledger, also when
it fails.

Files added or changed in this stage:

- `backend/training/export.py` (new): the Modal app that merges a trained adapter and
  checks the merged model's answers
- `backend/training/export_check.py` (new):
- `tests/test_export.py` (new): tests
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
    ........................................................................ [  9%]
    ........................................................................ [ 19%]
    ........................................................................ [ 29%]
    ........................................................................ [ 38%]
    ........................................................................ [ 48%]
    ........................................................................ [ 58%]
    ........................................................................ [ 67%]
    ........................................................................ [ 77%]
    ........................................................................ [ 87%]
    ........................................................................ [ 96%]
    .......................                                                  [100%]
    743 passed in 40.27s

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
- `backend/cli_watch.py`: te watch and te calibrate.
- `backend/meta/client.py`: MetaClient, the only way the code calls Meta's API: cache
  first, then cap check, retries with backoff for 429, 5xx and timeouts, and spend
  recorded from the reported usage.
- `backend/server.py`: the backend the iPhone app talks to: JPEG frames arrive over a
  WebSocket at /ws and each one is acknowledged; frames stay in memory.
- `backend/teach_session.py`: a phone teach session: the spend question before any
  paid call, the read-back, progress lines and timeouts.
- `backend/teach_upload.py`: the phone's teach upload: a multipart parser, the input
  checks, the saved capture and the estimate.
- `backend/teach_api.py`: the teach routes for the phone: /teach/start, /teach/status
  and /teach/answer, behind SERVER_TOKEN.
- `backend/training/wandb_smoke.py`: a Modal app that logs one Weights & Biases run,
  with the key coming only from a Modal Secret.
- `backend/training/axolotl.py`: the Axolotl config for a training run: Meta's QLoRA
  recipe with the run's paths, W&B names and gradient accumulation for one H100.
- `backend/training/prepare.py`: bundles taught skills into one training set for Modal
  (real, synthetic or mixed, with a validation split and volume paths).
- `backend/training/modal_train.py`: the Modal app that trains on one H100 with the
  recipe's pins, the trained-eye volume and the wandb and huggingface Secrets.
- `backend/training/export.py`: the Modal app that merges a trained adapter and checks
  the merged model's answers.
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
- `tests/`: 743 tests, all offline; none calls a paid API.

## Next

Stage v48 adds te export and te serve-student: the merge run from the Mac, and the
trained model, the base or the base with adapters served by vLLM on Modal.

## Privacy and policy

- No skill may identify or track a specific person.
- Watching never hides or suppresses the glasses' capture LED.
- Frames are not stored unless the user saves a clip.
- Muse Image outputs keep their watermark and provenance metadata. Generated images
  are never committed; only the prompt manifest and a script to regenerate them.
- Faces are blurred in anything published.
