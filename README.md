# Trained Eye

Teach your glasses to notice anything.

Meta glasses answer when you ask them something, but they can't keep an eye on
anything for you. With Trained Eye you look at something for about ten seconds and
say what to watch for ("tell me if I leave the burner on"). Meta's models label
that footage and generate training variations, a small open model (Muse Glimmer) is
fine-tuned on them, and from then on the glasses speak up when it happens.

Built for the Meta Global AI Developer Hackathon (October 29 to November 4, 2026).
All code here was written during the event.

## Build stage v93 (M4): Notes brought up to date

This is the last stage, and it changes no code, so by the author's choice it is
outside the 200 to 400 line rule. HANDOVER.md and PROGRESS.md give the stage numbers
after the re-cut (M2 is v31 to v63, M3 v64 to v77, M4 v78 to v92), the line rule from
v31, the test count, the rule that stages go to main only when the author asks, a
session log entry for Nov 3 and the spend through M4, and they no longer list as still
to build what is built: the SAM box reader, the phone's teach route, te manifests and
te regenerate, the release license and the student key kept off the vLLM command line.
SETUP.md lists every Modal Secret the jobs attach (huggingface before the first te
train, student, meta) and the two values you make up (STUDENT_API_KEY, SERVER_TOKEN),
and says how a stage goes to main. CLAUDE.md names the four Modal Secrets and adds te
replay. ios/README.md and docs/glasses-check.md point at the right step of the glasses
check.

Files added or changed in this stage:

- `CLAUDE.md` (changed): short setup, commands and conventions
- `HANDOVER.md` (changed): how to pick the project up in a new session: state, the
  author's rules, open items with exact commands, and the plan for M2 to M5 with the
  facts from the docs
- `PROGRESS.md` (changed): milestone checklist, decisions, docs-vs-spec differences,
  open issues and spend
- `SETUP.md` (changed): accounts and keys, setting up the Mac, setting up a cloud
  session, and git on the stage branch
- `docs/glasses-check.md` (changed): how to check, with Meta's own sample app, that
  the glasses stream camera frames, deliver the mic and play audio
- `ios/README.md` (changed): how to build the iPhone app in Xcode (not compiled yet)
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
    ........................................................................ [  7%]
    ........................................................................ [ 14%]
    ........................................................................ [ 21%]
    ........................................................................ [ 28%]
    ..................................................s..................... [ 35%]
    ........................................................................ [ 42%]
    ........................................................................ [ 49%]
    ........................................................................ [ 56%]
    ........................................................................ [ 63%]
    ........................................................................ [ 71%]
    ........................................................................ [ 78%]
    ........................................................................ [ 85%]
    ........................................................................ [ 92%]
    ........................................................................ [ 99%]
    .....                                                                    [100%]
    1012 passed, 1 skipped in 61.98s (0:01:01)

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
- `backend/cli_serving.py`: te export and te serve-student.
- `backend/cli_eval.py`: te eval and te label.
- `backend/cli_experiments.py`: te experiments, te unfiltered, te clips, te
  spark-errors and te regress.
- `backend/cli_replay.py`: te replay: the recorded test clips through watch mode.
- `backend/cli_release.py`: the M4 release commands.
- `backend/cli_playground.py`: te playground and te playground-clip.
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
- `backend/inference/modal_serve.py`: the Modal app that serves the merged model with
  vLLM behind STUDENT_API_KEY.
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
  threshold for the skill's hold (the fridge 30 s, the tap 20 s, the burner 10 s), then
  the cooldown; the threshold calibration on labelled clips.
- `backend/inference/student.py`: scores a frame with the trained model on an
  OpenAI-compatible endpoint, read at its yes or no answer token.
- `backend/watch/watcher.py`: watch mode: scores, the Muse Spark check before an
  alert, alert messages and the frame log.
- `backend/labeltool/clips.py`: the labelling tool for clips: where the condition is
  true, saved as time ranges.
- `backend/labeltool/images.py`: the labelling tool for generated images, for
  Experiment 2.
- `backend/eval/metrics.py`: evaluation metrics from watch frame logs and labels.
- `backend/eval/report.py`: the evaluation runner: the plan, results.json, results.md
  and the SVG charts.
- `backend/eval/plan.py`: the experiment plan: every training run with fixed steps and
  its price.
- `backend/eval/plan_file.py`: eval/plan.json: the runs, clips and logs make eval
  reads.
- `backend/regenerate.py`: the prompt manifests and regenerating images from them.
- `backend/release/model_card.py`: the Hugging Face model card.
- `backend/release/package.py`: the release folder (base license, blurred keyframes),
  the upload and the collection.
- `backend/playground/app.py`: the web playground.
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
- `ios/README.md`: how to build the iPhone app in Xcode (it builds for the simulator).
- `CONTRIBUTIONS.md`: credits, licenses and the copied Meta sample files.
- `tests/`: 1039 tests, all offline; none calls a paid API.

## Next

This is the last stage: the folder is the finished project, the same as the repository
root.

## Privacy and policy

- No skill may identify or track a specific person.
- Watching never hides or suppresses the glasses' capture LED.
- Frames are not stored unless the user saves a clip.
- Muse Image outputs keep their watermark and provenance metadata. Generated images
  are never committed; only the prompt manifest and a script to regenerate them.
- Faces are blurred in anything published.
