# Trained Eye

Teach your glasses to notice anything.

Meta glasses answer when you ask them something, but they can't keep an eye on
anything for you. With Trained Eye you look at something for about ten seconds and
say what to watch for ("let me know when the washer says done"). Meta's models label
that footage and generate training variations, a small open model (Muse Glimmer) is
fine-tuned on them, and from then on the glasses speak up when it happens.

Built for the Meta Global AI Developer Hackathon (October 29 to November 4, 2026).
All code here was written during the event.

## Build stage v42 (M2): te train on Modal

backend/cli_train.py adds te train, which bundles taught skills and trains the student
with the Modal training app. --smoke runs the recipe's two-step check on one H100 and
saves its timing. A real run is priced from that timing, the smoke's setup time plus
the examples at its measured samples per second at config.py's H100 price, and asks
past $15 even with --yes, but that path reads backend/eval/plan.py, so it works from
v74, where its tests come. A job that would pass the Modal credit asks even with
--yes, and so does --wandb-artifacts, after saying what becomes visible on W&B. Each
run is tagged with its experiment, skills, mix, data hash and commit, its status file
in data/train says running and then done or failed, and its GPU time goes into the
Modal ledger, also when it fails or is interrupted. --detach leaves the run going in
Modal and --collect fetches its result from the volume later; --preflight runs the
processor check on a CPU. The training app's test now checks the volume name te train
uses.

Files added or changed in this stage:

- `backend/cli.py` (changed): the te command line
- `backend/cli_train.py` (new): te train: the smoke, the priced run, detach and
  collect, the preflight
- `tests/test_cli_train.py` (new): tests
- `tests/test_modal_train.py` (changed): tests
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
    ........................................................................ [ 21%]
    ........................................................................ [ 31%]
    ........................................................................ [ 42%]
    ........................................................................ [ 53%]
    ........................................................................ [ 63%]
    ........................................................................ [ 74%]
    ........................................................................ [ 84%]
    ........................................................................ [ 95%]
    ...............................                                          [100%]
    679 passed in 31.54s

    $ ruff check .
    All checks passed!

    $ python -m backend.cli train --help
    usage: te train [-h] [--skills SKILLS] [--run RUN]
                    [--variant {mixed,real,synthetic}] [--limit LIMIT]
                    [--max-steps MAX_STEPS]
                    [--dataset {dataset,dataset-unfiltered}] [--smoke]
                    [--preflight] [--detach] [--collect RUN]
                    [--wandb-entity WANDB_ENTITY] [--wandb-artifacts]
                    [--estimate-only] [--yes]

    options:
      -h, --help            show this help message and exit
      --skills SKILLS       comma-separated, e.g. washer_done (needed but with
                            --collect)
      --run RUN             run name, e.g. exp1-mixed (needed but with --collect)
      --variant {mixed,real,synthetic}
      --limit LIMIT         at most this many examples per skill
      --max-steps MAX_STEPS
                            fixed optimizer steps for Experiments 1 to 3
                            (backend/eval/plan.py)
      --dataset {dataset,dataset-unfiltered}
                            the skill's dataset folder (Experiment 2: dataset-
                            unfiltered)
      --smoke               the recipe's two-step check
      --preflight           instead of training: render a few examples with the
                            pinned processor on Modal (CPU): lengths, the system
                            turn, the yes/no target
      --detach              modal run --detach: the run goes on if this Mac sleeps
                            or disconnects (then te train --collect)
      --collect RUN         fetch a detached run's result from the Modal volume
                            and print it as te train does
      --wandb-entity WANDB_ENTITY
                            the W&B team for the run (default: your own account)
      --wandb-artifacts     also upload the bundle and the adapter as W&B
                            Artifacts (asks first, even with --yes)
      --estimate-only       bundle and price only
      --yes                 no question (not over $15)

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
- `tests/`: 679 tests, all offline; none calls a paid API.

## Next

Stage v43 adds the teach session behind teaching from the phone: run_teach in a
background thread, the spend question first, read-back corrections, timeouts and a
clean stop.

## Privacy and policy

- No skill may identify or track a specific person.
- Watching never hides or suppresses the glasses' capture LED.
- Frames are not stored unless the user saves a clip.
- Muse Image outputs keep their watermark and provenance metadata. Generated images
  are never committed; only the prompt manifest and a script to regenerate them.
- Faces are blurred in anything published.
