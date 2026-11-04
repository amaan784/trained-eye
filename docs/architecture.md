# Architecture

Trained Eye has three parts: the teach flow, which turns about ten seconds of
first-person video and one spoken or typed sentence into a training set; training and
serving, which fine-tune Muse Glimmer on that set on Modal and serve it; and watch mode,
which scores the glasses' frames and speaks up when the condition has lasted long enough.
[how-it-works.html](how-it-works.html) walks through the whole flow with diagrams.

## Teach

1. Keyframes: 20 to 30 sharp, distinct frames from the teaching clip
   (`backend/dataengine/keyframes.py`).
2. The request: Muse Voice Transcribe when it is spoken, or the typed text.
3. The skill: Muse Wonder turns the request and three keyframes into a skill spec (what
   to watch, what counts, near-misses, the spoken alert), or refuses it under the policy
   in [privacy-and-policy.md](privacy-and-policy.md). The read-back is shown before any
   further spend (`backend/skills/`).
4. The region of interest: SAM 3.1 finds the target in each keyframe; the box around all
   its confident matches is the region, and each keyframe is also cropped to it
   (`backend/dataengine/roi.py`, `sam_text.py`).
5. Variations: Muse Image edits the keyframes into positives and hard negatives under
   other light, angles and distances (`backend/dataengine/synth.py`).
6. Labels: Muse Spark labels every image; a generated image is kept only when Spark
   agrees with what it was made to show (`backend/dataengine/labels.py`).
7. The dataset, in Meta's Axolotl recipe format (`backend/dataengine/dataset.py`).

Image edits and labels run several at a time, paced under the API's rate limits, and
every call is cached and recorded in a spend ledger (`backend/meta/`).

## Train and serve

Meta's QLoRA recipe for Muse Glimmer 30B on one H100 on Modal: a CPU job downloads the
pinned base to a volume first, a two-step smoke prices the run, then the run trains
(`backend/training/`). The adapter is merged into the base and checked against base plus
adapter (`backend/training/export.py`), and the merged model is served with Meta's vLLM
image behind a key (`backend/inference/modal_serve.py`).

## Watch

The phone streams the glasses' frames to the backend over a WebSocket
(`backend/server.py`). The student scores a frame a second at its yes or no answer token
(`backend/inference/student.py`); a skill fires when 3 of its last 4 frames are over its
threshold for the skill's hold, then waits for its cooldown (`backend/watch/`).

## Modules

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
- `ios/README.md`: how to build the iPhone app in Xcode (it builds for the simulator with XcodeGen and Xcode).
- `CONTRIBUTIONS.md`: credits, licenses and the copied Meta sample files.
- `tests/`: the test suite, all offline; none calls a paid API.
