# Trained Eye

Teach your glasses to notice anything.

Meta glasses answer when you ask them something, but they can't keep an eye on
anything for you. With Trained Eye you look at something for about ten seconds and
say what to watch for ("let me know when the washer says done"). Meta's models label
that footage and generate training variations, a small open model (Muse Glimmer) is
fine-tuned on them, and from then on the glasses speak up when it happens.

Built for the Meta Global AI Developer Hackathon (October 29 to November 4, 2026).
All code here was written during the event.

## Build stage v75 (M3): The experiment commands and eval/plan.json

backend/cli_experiments.py adds five commands, registered by backend/cli.py. te
experiments --skills prints every training run with its price and what it still needs;
--write-plan also writes eval/plan.json, never over an existing one unless --force. te
unfiltered writes Experiment 2's arm without the Spark filter, te clips turns a
skill's labelled session clips into dataset-clips/ and says what it left out, and te
spark-errors saves Spark's error rate against the hand labels; all three are free. te
regress asks a base and a fine-tuned endpoint the questions with the student's
sampling and saves eval/regression.json and regression.md. backend/eval/plan_file.py
builds the plan te eval reads: one evaluation per run and skill, plus Experiment 4's
three judges, each with its test clips (or a placeholder named as the protocol says,
pending until recorded), their frame logs and labels, the run whose replays they read,
the prices and the skill's spoken alert. Each run the student replays also gets its
session 2 clips under calibration. The held-back te clips test comes back here.

Files added or changed in this stage:

- `backend/cli.py` (changed): the te command line
- `backend/cli_experiments.py` (new): te experiments, te unfiltered, te clips, te
  spark-errors and te regress
- `backend/eval/plan_file.py` (new): eval/plan.json: the runs, clips and logs make
  eval reads
- `tests/test_cli_experiments.py` (new): tests
- `tests/test_plan_file.py` (new): tests
- `tests/test_realclips.py` (changed): tests
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
    ........................................................................ [  8%]
    ........................................................................ [ 16%]
    ........................................................................ [ 24%]
    ........................................................................ [ 32%]
    ........................................................................ [ 40%]
    ........................................................................ [ 48%]
    ........................................................................ [ 56%]
    ........................................................................ [ 65%]
    ........................................................................ [ 73%]
    ........................................................................ [ 81%]
    ........................................................................ [ 89%]
    ........................................................................ [ 97%]
    ......................                                                   [100%]
    886 passed in 54.18s

    $ ruff check .
    All checks passed!

    $ python -m backend.cli experiments --help
    usage: te experiments [-h] --skills SKILLS [--write-plan] [--force]
                          [--endpoint-hours ENDPOINT_HOURS]

    options:
      -h, --help            show this help message and exit
      --skills SKILLS       comma-separated, e.g. washer_done
      --write-plan          also write eval/plan.json
      --force               replace an existing eval/plan.json
      --endpoint-hours ENDPOINT_HOURS
                            hours of serving to price per evaluated model
                            (default: backend/eval/plan.py's
                            DEFAULT_ENDPOINT_HOURS)

    $ python -m backend.cli clips --help
    usage: te clips [-h] --skill SKILL [--clips CLIPS]

    options:
      -h, --help     show this help message and exit
      --skill SKILL
      --clips CLIPS  the clip folder (default data/clips/<skill>)

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
  threshold, then the cooldown; the threshold calibration on labelled clips.
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
- `ios/README.md`: how to build the iPhone app in Xcode (not compiled yet).
- `tests/`: 886 tests, all offline; none calls a paid API.

## Next

Stage v76 adds the replay engine that fills the plan's frame logs (warm-up, validation
clips, then test clips) and the W&B report built from results.md.

## Privacy and policy

- No skill may identify or track a specific person.
- Watching never hides or suppresses the glasses' capture LED.
- Frames are not stored unless the user saves a clip.
- Muse Image outputs keep their watermark and provenance metadata. Generated images
  are never committed; only the prompt manifest and a script to regenerate them.
- Faces are blurred in anything published.
