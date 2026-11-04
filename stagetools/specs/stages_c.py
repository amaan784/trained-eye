"""Per-stage metadata for the consolidated stages from v31 (200 to 400 lines of code each).

Each stage adds whole files in their final form, with their tests; see
stagetools/CONSOLIDATED_PLAN.md for how the old stages v31 to v105 map to these.
"""

import importlib.util
import os
from pathlib import Path

_spec = importlib.util.spec_from_file_location("stages_late", Path(__file__).with_name("stages_late.py"))
_late = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_late)

INTRO, POLICY, PYTEST, RUFF = _late.INTRO, _late.POLICY, _late.PYTEST, _late.RUFF
VENV = _late.VENV
DATE = "2026-11-03"

_REPLACE = {
    "backend/dataengine/roi.py": "the region of interest: SAM 3.1 on each keyframe, its boxes "
    "scaled to the keyframe, and a crop around each box for the dataset",
    "backend/eval/plan.py": "the experiment plan: every training run with fixed steps and its "
    "price",
    "backend/release/model_card.py": "the Hugging Face model card",
}
_BEFORE = {
    "backend/dataengine/roi.py": [
        ("backend/dataengine/sam_text.py", "our reader of SAM 3.1's output text, written from "
         "the grammar in Meta's SAM 3 protocol: object ids, boxes and confidences"),
    ],
}
_AFTER = {
    "backend/cli.py": [
        ("backend/cli_train.py", "te train: the smoke, the priced run, detach and collect, the "
         "preflight"),
        ("backend/cli_watch.py", "te watch and te calibrate"),
        ("backend/cli_serving.py", "te export and te serve-student"),
        ("backend/cli_eval.py", "te eval and te label"),
        ("backend/cli_experiments.py", "te experiments, te unfiltered, te clips, "
         "te spark-errors and te regress"),
        ("backend/cli_replay.py", "te replay: the recorded test clips through watch mode"),
        ("backend/cli_release.py", "the M4 release commands"),
        ("backend/cli_playground.py", "te playground and te playground-clip"),
    ],
    "backend/eval/plan.py": [
        ("backend/eval/plan_file.py", "eval/plan.json: the runs, clips and logs make eval "
         "reads"),
    ],
    "backend/release/model_card.py": [
        ("backend/release/package.py", "the release folder (base license, blurred keyframes), "
         "the upload and the collection"),
    ],
}
_LISTED = {p for items in _AFTER.values() for p, _ in items}
COMPONENTS = []
for _path, _text in _late.COMPONENTS:
    if _path in _LISTED:
        continue
    COMPONENTS += _BEFORE.get(_path, [])
    COMPONENTS.append((_path, _REPLACE.get(_path, _text)))
    COMPONENTS += _AFTER.get(_path, [])


def stage(next, layer, adds, summary, paragraph, commands=None):
    return dict(next=next, layer=layer, adds=adds, commands=commands or [PYTEST, RUFF],
                date=DATE, summary=summary, paragraph=paragraph)


STAGES = {
    31: stage(
        next="the settings and the spend ledger the later stages need: the student endpoint, "
             "the two new secrets, and spend reserved across threads and processes.",
        layer="Settings and the spend ledger",
        adds="The settings every later stage reads arrive here. STUDENT_API_KEY (the trained "
             "model's endpoint) and SERVER_TOKEN (teaching from the phone) are secrets like "
             "the other keys: read from .env, never printed. The student endpoint is set with "
             "TE_STUDENT_URL, TE_STUDENT_MODEL and TE_STUDENT_MAX_TOKENS, TE_SYSTEM_IN_USER "
             "folds the system prompt into new datasets, and TE_DATA_DIR may start with ~. "
             "The Modal GPU price, the $15 rule and the $150 Modal credit live in config.py "
             "too. The spend ledger now reserves each call's estimate until the call is "
             "recorded, under a lock and an flock beside the ledger, so calls in flight in "
             "any thread or process count against the caps together; a reservation left by a "
             "thread or process that ended is dropped. Entries record the images and text "
             "tokens of token-priced calls, so image_tokens() can tell what an image costs "
             "from the ledger's own usage. CI installs ffmpeg for the video tests.",
        commands=[PYTEST, RUFF, "python -m backend.config", "python -m backend.cli spend"],
        summary="Add M2 settings and reserve spend across processes",
        paragraph="Adds the settings the later stages read: STUDENT_API_KEY and SERVER_TOKEN "
                  "as secrets that are never printed, the student endpoint "
                  "(TE_STUDENT_URL, TE_STUDENT_MODEL, TE_STUDENT_MAX_TOKENS), "
                  "TE_SYSTEM_IN_USER, ~ in TE_DATA_DIR, the Modal GPU price, the $15 rule "
                  "and the $150 Modal credit. The spend ledger reserves each estimate until "
                  "the call is recorded, under a lock and an flock, so calls in flight in "
                  "any thread or process count against the caps together, and it measures "
                  "what an image costs from the reported usage. CI installs ffmpeg. Tests "
                  "cover each.",
    ),
    32: stage(
        next="the policy check's new rules (hiding the glasses' capture light, politics and "
             "war) and checks on the compiled skill's own wording.",
        layer="Policy and the skill compiler",
        adds="The local policy check gains two rules. A request to hide, cover or turn off "
             "the glasses' capture light is refused (another device's recording light is "
             "fine), and so is a skill about politics or war, matched as phrases after "
             "look-alikes such as a glue gun, toy soldiers, a Swiss army knife or a laundry "
             "campaign are taken out. The people, covert and sports rules match fewer "
             "innocent words: \"his\" does not match \"hiss\", and \"my lift\" counts as a "
             "sport only next to a sport word. The compiler checks the model's wording as "
             "well as the request: the description, negatives, trigger and alert all go "
             "through the policy, the part of a correction that says what it is not (\"not "
             "the bathroom light\") is left out of its check, a compiled cooldown is kept "
             "between a minute and an hour, and a request is at most 500 characters. The "
             "read-back gets helpers that make a condition read after \"tell you when\" and "
             "drop conditions that differ only in case.",
        commands=[PYTEST, RUFF,
                  'python -m backend.cli policy "let me know when the washer says done"',
                  'python -m backend.cli policy "help me hide the capture light on my glasses"',
                  'python -m backend.cli policy "tell me when a tank drives past"',
                  'python -m backend.cli policy "tell me when the glue gun is hot"'],
        summary="Extend the local policy and check compiled wording",
        paragraph="Refuses skills that hide the glasses' capture light and skills about "
                  "politics or war (phrases, with look-alikes such as a glue gun or toy "
                  "soldiers taken out first), and narrows the people, covert and sports "
                  "rules. The compiler runs the policy on the model's description, "
                  "negatives, trigger and alert too, leaves a correction's \"not the ...\" "
                  "part out of its check, clamps the cooldown to a minute to an hour and "
                  "caps a request at 500 characters. Adds read-back helpers. Tests cover "
                  "each rule.",
    ),
    33: stage(
        next="the Meta client's hardening: interrupted and failed calls charged, the caps "
             "checked again before every retry, and SAM 3.1's streamed reply read by the "
             "protocol's rules.",
        layer="Meta calls: client, labels and image edits",
        adds="A call stopped with Ctrl+C, or one whose reply fails after the request went "
             "through, is charged as billed. An empty transcript is billed but not cached, a "
             "cached reply that today's checks reject is fetched again, a generated image "
             "whose download fails twice becomes an output error the data engine can skip, "
             "and a reply without usable images is charged for the images asked for. The "
             "caps are checked again before every retry. Failed calls carry what they cost "
             "and the finish reason, structured calls can ask the cache alone, and each "
             "entry records the images and text tokens sent. SAM 3.1 calls ask for "
             "confidences and read the streamed reply by the lane rules of Meta's SAM 3 "
             "protocol: a refusal or a failed or incomplete response is an error, and an "
             "empty answer is valid. Labels can come from the cache alone, and the synthetic "
             "set stops with TooManyFailures when too many edits fail.",
        summary="Harden Meta calls, SAM streams and image edits",
        paragraph="Charges interrupted and failed-after-send calls, re-checks the caps "
                  "before every retry, refetches stale cache entries, skips failed image "
                  "downloads as output errors, and records each call's images, text tokens, "
                  "cost and finish reason. SAM 3.1 replies are read by the protocol's lane "
                  "rules with confidences asked for. Labels gain a cache-only lookup and the "
                  "synthetic set stops on too many failures. Tests cover each case.",
    ),
    34: stage(
        next="our reader of SAM 3.1's text format, with boxes and crops for every keyframe.",
        layer="Keyframes, SAM boxes and the dataset",
        adds="backend/dataengine/sam_text.py reads SAM 3.1's output text. It is our own "
             "implementation of the grammar in Meta's SAM 3 protocol (protocol/sam3.md in "
             "meta-models/meta-sam): frame markers, object ids, box and mask tokens, "
             "inclusive boxes made half-open, confidences from 0 to 1, and the protocol's "
             "rules for unknown fields, malformed segments and prose lines. Mask payloads are "
             "checked but not decoded, since the region of interest needs only the box. "
             "roi.py uses it: boxes are scaled to the keyframe, problems are kept with each "
             "frame, and each box gives a crop with a 25% margin, at least 64 pixels a side. "
             "Frames from a folder are turned by their EXIF orientation and HEIC frames are "
             "refused with a message, and each dataset entry records where its image came "
             "from.",
        summary="Parse SAM 3.1 boxes and crop keyframes",
        paragraph="Adds backend/dataengine/sam_text.py, our reader of SAM 3.1's output text "
                  "written from the grammar in Meta's SAM 3 protocol, and uses it in the "
                  "region of interest: boxes scaled to the keyframe, problems kept per frame, "
                  "and a crop around each box. Frame folders follow EXIF orientation and "
                  "refuse HEIC, and dataset entries record their source. Tests cover the "
                  "grammar, the crops and the keyframes.",
    ),
    35: stage(
        next="the watch trigger (3 of the last 4 frames over the threshold, then the "
             "cooldown) and the student scorer on an OpenAI-compatible endpoint.",
        layer="Watch trigger and the student scorer",
        adds="The first two parts of watch mode. backend/watch/trigger.py decides when a "
             "skill fires: at least 3 of its last 4 frames at or over the threshold, frames "
             "more than 10 seconds apart not counted together, then the skill's cooldown; "
             "calibrate() picks the lowest threshold that gives at most one false alarm an "
             "hour on labelled validation clips. backend/inference/student.py scores a frame "
             "with the trained model on an OpenAI-compatible endpoint (vLLM on Modal or "
             "llama-server on the Mac), sent exactly as the training examples were built. "
             "The probability is read at the answer token, the last yes or no, from its top "
             "logprobs; a reply without one, or one cut off at the reply budget, is an "
             "error, never a guess. Sampling follows Meta's serving notes.",
        summary="Add the watch trigger and the student scorer",
        paragraph="Adds backend/watch/trigger.py (3 of the last 4 frames over the threshold, "
                  "then the cooldown, and the threshold calibration on labelled clips) and "
                  "backend/inference/student.py (a frame scored on an OpenAI-compatible "
                  "endpoint, read at the yes or no answer token, with no guess on a cut-off "
                  "reply). Tests cover both, with no network.",
    ),
}

STAGES.update({
    36: stage(
        next="the training run records (running, done or failed, and which run is served) "
             "and the Modal spend ledger with its check against the $150 credit.",
        layer="Training run records and the Modal spend ledger",
        adds="Two record keepers that the training and serving commands use later; neither "
             "calls Modal. backend/training/runs.py keeps a status file per training run in "
             "data/train (running, done or failed, its skills, start and finish). A run whose "
             "te train process has gone from this Mac reads as failed, unless it was "
             "launched detached, and serving.json says which run the student endpoint "
             "serves. GET /skills reads these for the app's Skills screen. "
             "backend/training/gpu_ledger.py keeps Modal GPU time in "
             "data/modal_ledger.jsonl, one line per training, export or serving job, priced "
             "at config.MODAL_GPU_USD_PER_HOUR, so te spend can print the Modal total and te "
             "train and te export can ask before the runs pass the $150 Modal credit. "
             "Serving time is an upper bound, and a row with a key is written only once.",
        summary="Add training run records and the Modal spend ledger",
        paragraph="Adds backend/training/runs.py (each run's status, a vanished te train read "
                  "as failed unless detached, and the served run, for GET /skills) and "
                  "backend/training/gpu_ledger.py (Modal GPU time per training, export and "
                  "serving job, priced from config, and the check against the $150 "
                  "credit). Neither calls Modal. Tests cover both.",
    ),
    37: stage(
        next="watch mode itself: the student's scores, the trigger, the Muse Spark check "
             "before an alert, cooldowns kept across sessions and the frame log.",
        layer="The watcher",
        adds="backend/watch/watcher.py is watch mode for a set of taught skills: frames in, "
             "alerts out. For every frame and skill the student scores the frame, the "
             "skill's trigger decides whether it fires, and a skill that asks for it has "
             "Muse Spark check the frame first. If Spark says no, the skill is snoozed for a "
             "minute; if the check itself fails, the alert goes out marked unconfirmed, as "
             "missing the washer is worse than one unchecked alert. Thresholds come from te "
             "watch --threshold, then the trained run's calibration, then the skill's, else "
             "0.5, and each frame log row says which. One CooldownBook holds each skill's "
             "quiet time for every session and is saved to data/watch/cooldowns.json, so a "
             "reconnect, a second phone or a restart does not announce a condition again "
             "inside its cooldown. Every scored frame is logged as one JSON line; frames "
             "themselves are never written.",
        summary="Add the watcher: scores, Spark check, alerts",
        paragraph="Adds backend/watch/watcher.py: each frame scored by the student, the "
                  "trigger, the Muse Spark check before an alert (a no snoozes the skill, a "
                  "failed check sends the alert marked unconfirmed), thresholds per run or "
                  "skill with their source logged, cooldowns kept across sessions and "
                  "restarts, and a JSON line per scored frame. Tests cover each.",
    ),
    38: stage(
        next="the training bundle (real, synthetic or mixed, with a fixed validation split) "
             "and Meta's QLoRA recipe as the Axolotl config.",
        layer="Training bundle and the recipe's config",
        adds="backend/training/prepare.py bundles taught skills into one training set for "
             "Modal: one adapter learns every skill, since the condition is in the prompt. "
             "The bundle holds train.jsonl, val.jsonl, the images and a manifest, with image "
             "paths where the folder will be on the Modal volume. The real, synthetic and "
             "mixed variants are Experiment 1's arms; the validation split is a fixed 10% "
             "once there are enough examples, a limit keeps a fixed subset per skill, and "
             "the same inputs always give the same bundle. Skills written with a different "
             "TE_SYSTEM_IN_USER are not mixed. backend/training/axolotl.py writes the run's "
             "Axolotl config from Meta's QLoRA recipe for Muse Glimmer (meta-oss-cookbook "
             "b30ca2c): every training value is the recipe's own, and only the paths, the "
             "W&B names and the gradient accumulation for one H100 change. The tests for "
             "real session clips come with the clips stage.",
        summary="Add the training bundle and the recipe's config",
        paragraph="Adds backend/training/prepare.py (one deterministic bundle of taught "
                  "skills for Modal, in Experiment 1's real, synthetic and mixed variants, "
                  "with a fixed validation split and volume paths) and "
                  "backend/training/axolotl.py (Meta's QLoRA recipe as the run's Axolotl "
                  "config, with only paths, W&B names and gradient accumulation changed). "
                  "Tests cover both.",
    ),
    39: stage(
        next="the processor preflight: bundle examples rendered with the pinned Glimmer "
             "processor before the paid smoke.",
        layer="The processor preflight",
        adds="backend/training/preflight.py checks, before the paid smoke, how training will "
             "see a few bundle examples. On Modal (a CPU job, no weights) the pinned Muse "
             "Glimmer processor renders each example in both layouts, with the system "
             "prompt as its own turn and folded into the user text, counting tokens the way "
             "the cookbook does. On this Mac summarize() judges the result: every example "
             "within the recipe's sequence length, the system prompt rendered once, the yes "
             "and no token IDs, the stop tokens, and which TE_SYSTEM_IN_USER to use. The "
             "file imports nothing from backend, so the training app can copy it into the "
             "container; the test of that Modal call comes with the training app.",
        summary="Add the processor preflight",
        paragraph="Adds backend/training/preflight.py: bundle examples rendered in both "
                  "layouts with the pinned Glimmer processor (on Modal, CPU only), and a "
                  "local verdict on sequence length, the system prompt, the yes and no "
                  "tokens, the stop tokens and TE_SYSTEM_IN_USER. Tests cover it with a "
                  "stand-in processor.",
    ),
    40: stage(
        next="the Modal training app: Meta's QLoRA recipe on one H100 with the recipe's pins, "
             "the two-step smoke, detached runs and the processor preflight on a CPU.",
        layer="The Modal training app",
        adds="backend/training/modal_train.py trains the student on Modal with Meta's Axolotl "
             "recipe. The container follows the recipe's setup (meta-oss-cookbook b30ca2c): "
             "Python 3.12, CUDA 13.0, torch 2.12.1, Axolotl 0.19.0, flash-attn 2.8.3 and Cut "
             "Cross Entropy at the pinned commit, built for the H100. Bundles, runs and the "
             "Hugging Face cache live on the Modal volume trained-eye, so the 30B base is "
             "downloaded once; WANDB_API_KEY and HF_TOKEN come only from the Modal Secrets "
             "wandb and huggingface. --smoke runs the recipe's two-step check, whose seconds "
             "per step price the real run. Every run writes result.json on the volume, so a "
             "detached run can be collected later. W&B gets the run's tags, the bundle "
             "summary and, only when asked, the bundle and adapter as artifacts. "
             "--preflight runs no training: a CPU function loads the pinned processor and "
             "renders examples with preflight.py. Nothing here runs in the tests, which "
             "read the file's syntax tree; the te train check comes with te train.",
        summary="Add the Modal training app with the recipe's pins",
        paragraph="Adds backend/training/modal_train.py: QLoRA training on one H100 with "
                  "Meta's Axolotl recipe and its pinned versions, the trained-eye volume "
                  "and the wandb and huggingface Secrets, the two-step smoke, result.json "
                  "for detached runs, W&B tags and optional artifacts, and the processor "
                  "preflight as a CPU function. Tests read the file without running "
                  "Modal, and the preflight's Modal test comes back.",
    ),
})


STAGES.update({
    41: stage(
        next="te teach's checks before any money is mentioned, the worst-case price with the "
             "$15 rule, and the keyframes cropped to SAM's box.",
        layer='Teach checks, worst case and crops',
        adds='te teach now checks everything it needs before any money is mentioned: the '
             'skill name, a --request of at most 500 characters that the policy allows, the '
             "audio and clip files, a clip outside the skill's own folder with usable "
             "frames, and --labels with no repeated key, naming only this clip's keyframes. "
             'With training on model outputs off, your labels must say both yes and no, or '
             'the keyframes are saved for labelling and nothing is spent. The estimate '
             'prices images at the tokens the ledger measured and shows the worst case, '
             'every label asked again with the larger budget; past $15 it asks even with '
             '--yes, and a missing MODEL_API_KEY stops before the question. te spend adds '
             'the Modal total. In backend/teach.py, each keyframe SAM found a box in adds a '
             'crop labelled like it, a new spec removes the old dataset, the labels so far '
             'are saved when labelling stops, and a dataset with only one answer stops '
             "instead of being written. The crop test comes in v73, with Experiment 2's "
             'unfiltered set.',
        summary='Check te teach inputs first and add SAM crops',
        paragraph="Checks every te teach input before the estimate, with the fallback's own "
                  'labels needing both answers, and prices images at the measured tokens with a '
                  "worst case that asks past $15 even with --yes. Adds a crop to SAM's box for "
                  'each keyframe that has one, labelled like the keyframe, and stops on a '
                  'one-answer dataset. te spend prints the Modal total. Tests cover the input '
                  'checks, the questions, the crops and the dataset rules.',
        commands=[PYTEST, RUFF, 'python -m backend.cli teach --help'],
    ),
    42: stage(
        next='te train: the bundle, the two-step smoke on Modal, detached runs collected '
             "later, the processor preflight, and the real run's pricing from the smoke, "
             "which needs v74's experiment plan.",
        layer='te train on Modal',
        adds='backend/cli_train.py adds te train, which bundles taught skills and trains the '
             "student with the Modal training app. --smoke runs the recipe's two-step check "
             'on one H100 and saves its timing. A real run is priced from that timing, the '
             "smoke's setup time plus the examples at its measured samples per second at "
             "config.py's H100 price, and asks past $15 even with --yes, but that path reads "
             'backend/eval/plan.py, so it works from v74, where its tests come. A job that '
             'would pass the Modal credit asks even with --yes, and so does '
             '--wandb-artifacts, after saying what becomes visible on W&B. Each run is '
             'tagged with its experiment, skills, mix, data hash and commit, its status file '
             'in data/train says running and then done or failed, and its GPU time goes into '
             'the Modal ledger, also when it fails or is interrupted. --detach leaves the '
             'run going in Modal and --collect fetches its result from the volume later; '
             "--preflight runs the processor check on a CPU. The training app's test now "
             'checks the volume name te train uses.',
        summary='Add te train: the smoke, detach, collect and preflight',
        paragraph='Adds backend/cli_train.py, registered by cli.py: te train bundles taught '
                  'skills, runs the two-step smoke on Modal and saves its timing, and asks past '
                  'the Modal credit or before W&B artifacts even with --yes. Runs are tagged, '
                  'recorded as running, done or failed, and their GPU time goes into the Modal '
                  'ledger; --detach and --collect let a run outlive the terminal, and --preflight '
                  "runs the processor check. The real run's pricing from the smoke, with its $15 "
                  'question, reads backend/eval/plan.py and so works from v74. Tests cover the '
                  'smoke, input errors, run states, tags, artifacts, detach and collect, and the '
                  "preflight, with Modal faked; the priced run's tests come in v74.",
        commands=[PYTEST, RUFF, 'python -m backend.cli train --help'],
    ),
    43: stage(
        next='the teach session behind teaching from the phone: run_teach in a background '
             'thread, the spend question first, read-back corrections, timeouts and a clean '
             'stop.',
        layer='Teach sessions from the phone',
        adds='backend/teach_session.py runs one teach started from the phone: '
             'backend.teach.run_teach in a background thread, with its questions answered '
             "later over HTTP. The spend question comes first, here, because run_teach's "
             'first paid step, the transcription, comes before its only question, the '
             'read-back. A no to the read-back that carries a correction, at most 300 '
             'characters, compiles the spec again with it from the cached transcript, and '
             'each correction replaces the one before, as with te teach --correct. A '
             'question nobody answers within 600 seconds counts as no. status() gives the '
             'state (waiting for the spend, running, waiting for the read-back, then done, '
             'stopped or failed), the open question, the progress lines, what was spent and, '
             'at the end, a summary of the dataset and the labels. stop(), for when the '
             "server shuts down, refuses the next paid call, cuts a retry's wait short, "
             'answers an open question no and waits for the call in flight, so the ledger '
             'records it. Failed calls and spend caps end the run with what to do next. Its '
             'HTTP routes come in v44.',
        summary='Add the teach session behind phone teaching',
        paragraph='Adds backend/teach_session.py: a teach run in a background thread with the '
                  'spend question asked first and the read-back answered later, a correction '
                  'compiling the spec again from the cached transcript, an unanswered question '
                  'counted as no after 600 seconds, and stop() refusing the next paid call and '
                  'waiting for the one in flight. Tests cover the timeouts, the answers, a '
                  'refused request, failures and their wording, and the stop.',
        commands=[PYTEST, RUFF],
    ),
    44: stage(
        next="the phone's teach API: the capture uploaded, checked and priced, then the "
             'spend and read-back questions answered over HTTP, each request with the server '
             'token.',
        layer="The phone's teach API",
        adds='backend/teach_api.py has the routes the phone teaches through: POST '
             '/teach/start with the skill, the spoken request or its text, and the JPEG '
             'frames of the 10 second capture as multipart form data, GET /teach/status, and '
             'POST /teach/answer for the two questions. Every route needs SERVER_TOKEN from '
             '.env as a Bearer token: without it the answer is 401, and with no token set on '
             'the Mac, 503. One session runs at a time, and a body over 200 MB or an answer '
             'over 64 KB is refused. backend/teach_upload.py parses the multipart body '
             "itself (Starlette's parser needs python-multipart) and checks the upload the "
             "way te teach checks its inputs: audio or text but not both, the request's "
             'length and the policy, 1 to 600 camera frames. It saves the capture under '
             'data/teach/<skill>/ so te teach can repeat the run on the Mac, and a run that '
             'needs your own labels or could cost over $15 is refused with the te teach '
             'command to run instead. Nothing is spent before the spend answer; the server '
             'mounts the routes in v45.',
        summary="Add the teach API and the phone's upload checks",
        paragraph='Adds backend/teach_api.py (POST /teach/start, GET /teach/status and POST '
                  '/teach/answer, each needing SERVER_TOKEN, one session at a time, with size '
                  'limits) and backend/teach_upload.py (the multipart upload parsed without '
                  'python-multipart, checked as te teach checks its inputs, saved under '
                  'data/teach/ and priced, with runs that need your own labels or could cost over '
                  '$15 sent back to the Mac). Tests cover a full run, a no to the spend, a '
                  "correction, the token, bad inputs and odd answers, through Starlette's "
                  'TestClient and a fake Meta API.',
        commands=[PYTEST, RUFF],
    ),
    45: stage(
        next="the server's watch mode (scores in each ack, alerts as messages, one cooldown "
             'book for every session), GET /skills, the teach routes, and a te serve that '
             'prints the address the phone uses.',
        layer='Watch mode and teaching on the server',
        adds='In backend/server.py, a hello whose "watch" lists skill names starts watch '
             'mode for that session: the student scores every frame, each ack carries the '
             'scores, and an alert goes to the phone as a message of its own, to be spoken '
             'through the glasses; a watch request that cannot be met gets an error. The '
             'default watcher reads the student endpoint from the settings and takes '
             'thresholds and the prompt layout from the served run in serving.json; every '
             'session shares one cooldown book, saved in data/watch/cooldowns.json, so a '
             'reconnect or a second phone does not announce a condition again too soon. GET '
             '/skills lists the taught skills for the Skills screen, with their threshold, '
             'training run and whether the served run has them. The teach routes are '
             'mounted, and on shutdown a teach run stops before its next paid call. te serve '
             "prints this Mac's address on the local network and the /health check, says "
             "whether teaching from the phone is on, and keeps the server's log lines.",
        summary='Wire watch mode, GET /skills and teaching into the server',
        paragraph='Wires watch mode into the WebSocket: a hello listing skills gets a watcher, '
                  'each ack carries the scores and alerts go to the phone, with one cooldown book '
                  "for every session. Adds GET /skills for the app's Skills screen, mounts the "
                  'teach routes and stops a teach run cleanly on shutdown. te serve prints the '
                  'local network address and the health check, says whether teaching from the '
                  "phone is on, and keeps the server's log lines. Tests cover watch mode, "
                  'cooldowns across sessions, GET /skills, the teach routes, shutdown and te '
                  'serve.',
        commands=[PYTEST, RUFF, 'python -m backend.cli serve --help'],
    ),
    46: stage(
        next='te watch, which replays a clip through watch mode with the trained model, and '
             "te calibrate, which sets a skill's or a run's threshold from labelled frame "
             'logs.',
        layer='Replaying clips and calibrating thresholds',
        adds='backend/cli_watch.py adds te watch and te calibrate. te watch replays a clip, '
             'a video or a folder of frames, through watch mode: the student on '
             'TE_STUDENT_URL scores each frame for the named skills, each alert is printed '
             'with its time, probability and the Muse Spark check (skipped with --no-cloud), '
             "and every scored frame goes to a frame log. --run uses that run's threshold "
             "before the skill's and names the run in the log, --threshold overrides both, "
             'and the student is asked with the prompt laid out as its training data was '
             "written. te calibrate reads frame logs, each with a labels file of the clip's "
             "true ranges, picks the threshold with the trigger's calibration and writes it "
             'for the skill or, with --run, for that run alone. It says when no threshold '
             'keeps false alarms under one an hour. pyproject.toml names backend and tests '
             'as first-party for ruff, so imports sort the same in a stage folder that lacks '
             "some backend packages. --scorer spark, Muse Spark on every frame, needs v74's "
             'experiment plan, as do its tests.',
        summary='Add te watch and te calibrate',
        paragraph='Adds backend/cli_watch.py: te watch replays a clip through watch mode, scored '
                  "by the student at its run's or skill's threshold, prints the alerts and writes "
                  "a frame log, and te calibrate sets a skill's threshold, or one run's, from "
                  'frame logs and labelled true ranges. pyproject.toml names backend and tests as '
                  "first-party for ruff's import sorting. Tests cover the replay, the endpoint "
                  'check, calibration per skill and per run, the prompt layout and the served-run '
                  'note.',
        commands=[PYTEST, RUFF, 'python -m backend.cli watch --help', 'python -m backend.cli calibrate --help'],
    ),
    47: stage(
        next="the export: a run's adapter merged into the bf16 base on Modal, then the "
             'merged model compared with base plus adapter on up to 50 bundle examples.',
        layer='Merging and checking the adapter',
        adds="backend/training/export.py merges a run's QLoRA adapter into the bf16 base on "
             'Modal, on one H100: the base at the pinned revision, the adapter from '
             "runs/<run>/out on the volume, merged with peft's merge_and_unload() and saved "
             'as safetensors with the processor in runs/<run>/merged, the folder the serving '
             'app will serve. It then reloads that folder and compares its yes or no answers '
             "with base plus adapter's on up to 50 examples of the run's bundle, the "
             'validation split if there is one. By default the reference is the bf16 base '
             'with the adapter; --reference nf4 compares with the NF4 base that training '
             "saw. Loading and prompting copy the cookbook's evaluator, the base's small "
             'files the save did not write, such as chat_template.jinja, are copied over, '
             "and tensor names are compared with the base's. "
             'backend/training/export_check.py is the plain Python part, run on the Mac: the '
             'examples in the evaluator\'s fixed order, the answer read after "to=user", and '
             'the agreement, where an unreadable answer never agrees. The result records the '
             "remote call's seconds for the Modal ledger, also when it fails.",
        summary='Add the adapter merge and its check on Modal',
        paragraph="Adds backend/training/export.py, which merges a run's adapter into the bf16 "
                  "base on one H100, saves it to runs/<run>/merged with the base's small files "
                  'and compares its answers with base plus adapter on up to 50 bundle examples, '
                  'and backend/training/export_check.py, the plain helpers that pick the '
                  'examples, read the yes or no after "to=user" and count the agreement. Tests '
                  "read export.py's syntax tree without Modal and run the helpers.",
        commands=[PYTEST, RUFF],
    ),
    48: stage(
        next='te export and te serve-student: the merge run from the Mac, and the trained '
             'model, the base or the base with adapters served by vLLM on Modal.',
        layer='Export and the student endpoint',
        adds="backend/inference/modal_serve.py serves the student with vLLM's "
             "OpenAI-compatible server on Modal as the cookbook's vllm.md describes: its "
             'vllm/vllm-openai:muse-glimmer image and single-card arguments on one H100. '
             "TE_SERVE_RUN serves a run's merged model as the app trained-eye-student, model "
             'name trained-eye; TE_SERVE_BASE serves the base at its pinned revision as the '
             'app trained-eye-base, model name muse-glimmer, and TE_SERVE_LORAS adds each '
             "run's adapter as a model of its own, so the base and a fine-tuned endpoint can "
             'be up together. Every request needs STUDENT_API_KEY from the Modal Secret '
             'student, which reaches vLLM through its environment, never its command line. '
             'backend/cli_serving.py adds te export, which asks first, runs the merge (at '
             'most 3 hours of GPU time), records that time also when interrupted and reports '
             'the check, and te serve-student, which asks before a deploy, stops the '
             'endpoint with --stop, and records the served run for GET /skills and the '
             'deploy in the Modal ledger. The test that te train and te export ask when the '
             'Modal credit would run out comes in v74, as its te train half needs the '
             'experiment plan.',
        summary='Add te export, te serve-student and the vLLM app',
        paragraph="Adds backend/inference/modal_serve.py (vLLM on one H100 with the cookbook's "
                  "image and arguments, serving a run's merged model, the base, or the base with "
                  "runs' adapters, with the key kept off the command line) and "
                  'backend/cli_serving.py (te export and te serve-student, which ask first, '
                  'record GPU time in the Modal ledger and the served run for GET /skills). Tests '
                  "cover both commands with Modal faked and read modal_serve.py's settings and "
                  'plain functions without importing Modal.',
        commands=[PYTEST, RUFF, 'python -m backend.cli export --help', 'python -m backend.cli serve-student --help'],
    ),
    49: stage(
        next="the iPhone app's first files: its settings, with the server token in the "
             "keychain, and the backend's WebSocket, /health and /skills messages as Swift "
             'types.',
        layer='iOS: settings and the protocol',
        adds='The iPhone app starts here. ios/TrainedEye/Models/Settings.swift keeps what '
             'the app remembers between launches in UserDefaults: the backend address '
             '(http://192.168.1.20:8765 until one is typed), the skills switched on for '
             'watching and which camera to use (by default the glasses, else the iPhone). '
             'The server token, which the teach routes need, goes to the keychain, readable '
             'only on this phone after the first unlock. The address may be http, https, ws '
             'or wss; the settings turn it into the HTTP base and the WebSocket address at '
             '/ws, and say whether it is on the local network (a private IPv4 address or a '
             '.local name) or taken to be a tunnel. ios/TrainedEye/Models/Protocol.swift has '
             "the WebSocket protocol's messages (the hello, the welcome, acks with scores, "
             'alerts and errors), the /health and /skills replies, and the status line the '
             'Skills screen shows for each skill. The training and served fields are read '
             'leniently, so an older backend still decodes.',
        summary="Add the iOS app's settings and protocol types",
        paragraph='Adds Settings.swift (the backend address, the watched skills and the camera '
                  'choice in UserDefaults, the server token in the keychain, and the HTTP and '
                  'WebSocket addresses made from what was typed) and Protocol.swift (the '
                  "WebSocket messages, the /health and /skills replies, and each skill's status "
                  'line). Not compiled here, as there is no Swift toolchain; the build steps are '
                  'in ios/README.md from v63.',
        commands=[PYTEST, RUFF],
    ),
    50: stage(
        next="the app's two clients: plain HTTP calls to te serve with the server token, and "
             'the watch WebSocket with one frame in flight, reconnects and a keepalive.',
        layer='iOS: backend and WebSocket clients',
        adds='ios/TrainedEye/Network/BackendClient.swift makes the plain HTTP calls to te '
             'serve: GET /health and GET /skills with a 10 second timeout, and the request '
             'helpers the teach calls use later. Every request carries the server token as a '
             'Bearer header when one is set, and a status outside 200 to 299 becomes an '
             "error with the backend's own message. ios/TrainedEye/Network/WatchSocket.swift "
             'speaks the watch protocol: a hello naming the camera, the watched skills and a '
             'pace of one frame a second, then each JPEG frame as a binary message, and it '
             "reads the welcome, the acks with each skill's score, the alerts and the "
             'errors. One frame is in flight at a time, so a frame that would have to wait '
             'is dropped rather than queued. A dropped connection is opened again after 1, '
             '2, 4, 8, then 15 seconds, each time as a new server session. After the welcome '
             'a ping goes every 20 seconds; no pong within 10 seconds, or no ack for 90 '
             'seconds, drops the socket and reconnects.',
        summary='Add the iOS backend and WebSocket clients',
        paragraph='Adds BackendClient.swift (GET /health and GET /skills with the server token, '
                  "and the backend's error message on a failed status) and WatchSocket.swift (the "
                  'hello, JPEG frames with one in flight, acks, alerts and errors, reconnects '
                  'with a growing pause, a keepalive ping and a timeout for a stuck backend). Not '
                  'compiled here, as there is no Swift toolchain; the build steps are in '
                  'ios/README.md from v63.',
        commands=[PYTEST, RUFF],
    ),
    51: stage(
        next="the HEVC frame decoder and keyframe check from Meta's CameraAccess sample, "
             "copied unchanged with Meta's LICENSE, and the teach API's messages as Swift "
             'types.',
        layer="iOS: Meta's HEVC decoder and teach messages",
        adds="Two files under ios/TrainedEye/Meta/ are Meta's own sample code from the DAT "
             "iOS SDK 1.0.1, copied unchanged with Meta's header kept: "
             'VideoFrameDecoder.swift from the CameraAccess sample, and HEVCKeyframe.swift, '
             'the isHEVCKeyframe() check from its VideoCaptureHandler.swift, moved into a '
             "file of its own. The SDK's LICENSE (the Meta Wearables Developer Terms, not an "
             "open-source license) sits next to them. The decoder turns the glasses' HEVC "
             'frames into images, decoding in software so the session survives the app going '
             'to the background; while it waits for a keyframe, or when a frame fails, it '
             'hands back the last good image, and it builds a new session after 3 failures '
             "in a row. ios/TrainedEye/Models/TeachModels.swift holds the teach API's "
             'messages: the status that all three teach routes return (the state, the open '
             'question, progress lines, the estimate, the read-back, the summary and the '
             'spend), read leniently so an unknown field never hides the rest, and the '
             'answer the app sends.',
        summary="Add Meta's HEVC decoder and the teach messages",
        paragraph='Adds VideoFrameDecoder.swift and HEVCKeyframe.swift, copied unchanged from '
                  "Meta's CameraAccess sample (DAT 1.0.1) with Meta's header, and the SDK's "
                  'LICENSE beside them under ios/TrainedEye/Meta/. Adds TeachModels.swift: the '
                  'teach status, read leniently, and the answer. Not compiled here, as there is '
                  'no Swift toolchain; the build steps are in ios/README.md from v63.',
        commands=[PYTEST, RUFF],
    ),
    52: stage(
        next='the glasses connection (registration with the Meta AI app, link, battery, '
             "firmware) and DAT's mock glasses for Debug builds, with a drawn washer clip as "
             "the simulator's feed.",
        layer='iOS: glasses connection and the mock device',
        adds='ios/TrainedEye/Glasses/GlassesConnection.swift registers the app with the Meta '
             "AI app, which calls back through the app's URL scheme, and follows the paired "
             "glasses as Meta's CameraAccess and BirdSpotter samples do: it listens to each "
             "device's link, compatibility and state, and reports the first connected pair "
             '(else the first paired one) with its name, link and battery, which stays '
             'unknown while DAT reports 0. It says when the glasses need a firmware update '
             'and can open that update, or the update of the DAT app on the glasses. '
             "ios/TrainedEye/Glasses/MockGlasses.swift, in Debug builds only, drives DAT's "
             'Mock Device Kit: it pairs the "Meta Glasses" model, powers it on, unfolds and '
             'dons it, and sets its camera feed before any stream starts, to '
             "Resources/mock_feed.mp4 when the app has it, else the iPhone's back camera. "
             'Fold, unfold and a touchpad tap let the simulator check that watching comes '
             'back after a fold and is left alone while paused. ios/make_mock_feed.py makes '
             'that clip with Pillow and ffmpeg: a drawn washer display counting down to '
             '"End", HEVC tagged hvc1, 360x640, 10 fps, 16 seconds.',
        summary='Add the glasses connection and the mock device',
        paragraph="Adds GlassesConnection.swift (registration, the chosen pair's name, link and "
                  'battery, and the firmware and glasses app updates) and MockGlasses.swift '
                  "(DAT's mock glasses in Debug builds, with the feed set before any stream). "
                  'Adds ios/make_mock_feed.py and the synthetic washer clip it makes, '
                  'Resources/mock_feed.mp4. ruff checks the script. Not compiled here, as there '
                  'is no Swift toolchain; the build steps are in ios/README.md from v63.',
        commands=[PYTEST, RUFF],
    ),
    53: stage(
        next='the glasses camera: a DAT session that comes back by itself after a fold or a '
             'link drop, the camera permission asked first, and a change of stream size.',
        layer='iOS: the glasses camera',
        adds="ios/TrainedEye/Glasses/GlassesCamera.swift runs the glasses' camera through "
             "Meta's DAT SDK, with the lifecycle of the CameraAccess sample: a session from "
             'the auto device selector, the camera permission, then the camera and its '
             'stream. Meta AI is opened for the camera permission only after the user '
             'confirms, and after a refusal no stream starts. Unlike CameraAccess, the '
             'session is not ended when the app goes to the background. While the camera is '
             'wanted it comes back by itself: after a fold or a link drop, a new session '
             'starts when the device selector reports the glasses again, never while a '
             'session is paused, and at most 3 times without a frame in between. A change of '
             'stream size stops the camera and adds it again at the new size once its stream '
             'has stopped, keeping the session. The class also holds the status and '
             'interruption lines that the Connect and Watch screens show. Nothing here '
             'touches the capture LED. The stream half, which this file needs, comes in the '
             'next stage.',
        summary='Add the iOS glasses camera session',
        paragraph='Adds GlassesCamera.swift: the DAT session and the camera permission, a new '
                  'session after a fold or a link drop (never while paused, at most 3 times '
                  'without a frame), a change of stream size that keeps the session, and the '
                  'status lines. Its stream half follows in v54. Not compiled here, as there is '
                  'no Swift toolchain; the build steps are in ios/README.md from v63.',
        commands=[PYTEST, RUFF],
    ),
    54: stage(
        next='the glasses stream (HEVC at 7 fps, a stopped stream added again, repeated '
             'images skipped) and the throttle that turns frames into one JPEG a second.',
        layer='iOS: the glasses stream and frame throttle',
        adds='ios/TrainedEye/Glasses/GlassesCamera+Stream.swift is the stream half of the '
             'glasses camera. Once the session has started and the camera permission is '
             'granted, it adds a camera streaming HEVC (hvc1) at 7 fps, at low resolution '
             "(360x640) or medium (504x896), and decodes every frame on the SDK's thread "
             "with Meta's decoder. An image the decoder hands back unchanged went out "
             'already, so it is skipped and counted rather than sent as a new frame. A '
             'stream that stops on its own while the session is up is added again, up to 3 '
             "times 750 ms apart, as BirdSpotter's bringCameraBack does; frames flowing "
             'again reset the counts. ios/TrainedEye/Camera/FrameThrottle.swift turns frames '
             'into JPEGs (quality 0.7) at the pace it is set to, one a second unless told '
             'otherwise; the app model of v58 sets three a second for a teach capture. Only '
             'due frames are encoded, and the schedule moves on one interval at a time, so a '
             '7 fps stream thinned to three a second gives three a second.',
        summary='Add the glasses stream and the frame throttle',
        paragraph='Adds GlassesCamera+Stream.swift (the stream sizes, the hvc1 camera at 7 fps, '
                  "frames decoded on the SDK's thread with repeated images skipped, and a stream "
                  'that stopped on its own added again up to 3 times) and FrameThrottle.swift '
                  '(JPEG frames at a set pace, one a second by default, encoding only frames that '
                  'are due). Not compiled here, as there is no Swift toolchain; the build steps '
                  'are in ios/README.md from v63.',
        commands=[PYTEST, RUFF],
    ),
    55: stage(
        next='the iPhone camera for when the glasses are not there, and spoken alerts that '
             "play only over Bluetooth, never from the phone's speaker.",
        layer='iOS: phone camera and spoken alerts',
        adds='ios/TrainedEye/Camera/PhoneCamera.swift is the fallback when the glasses are '
             "unavailable, built like the phone camera source in Meta's BirdSpotter sample: "
             'one capture session on its own queue with the back camera at 640x480, frames '
             'turned upright by a quarter turn, and only the frames the throttle wants '
             'converted. It asks for camera access the first time. iOS stops the camera in '
             'the background, so this source watches only with the app open. '
             'ios/TrainedEye/Audio/Speaker.swift speaks alerts on the glasses. DAT has no '
             'speaker API, so, as BirdSpotter does, a line is spoken only when the audio '
             'goes to a Bluetooth output and is dropped otherwise, never played from the '
             "iPhone's speaker. Lines queue rather than cut each other off, and the speaker "
             'reports whether a line was said. While the microphone is in use for a teach '
             'request, nothing is spoken and a line being said is cut short.',
        summary='Add the iPhone camera and spoken alerts',
        paragraph='Adds PhoneCamera.swift (the back camera at 640x480, turned upright, converting '
                  'only frames that are due) and Speaker.swift (alerts spoken only over '
                  'Bluetooth, queued, and silenced while the request is being recorded). Not '
                  'compiled here, as there is no Swift toolchain; the build steps are in '
                  'ios/README.md from v63.',
        commands=[PYTEST, RUFF],
    ),
    56: stage(
        next='the request recorder: the spoken teach request as 16 kHz WAV in memory, from '
             "the glasses' microphone when it is the input, else the iPhone's.",
        layer='iOS: the spoken request recorder',
        adds='ios/TrainedEye/Audio/RequestRecorder.swift records the spoken teach request '
             '("tell me when the washer is done") as 16 kHz mono 16-bit WAV, since the '
             "backend reads a WAV file's length for its cost estimate. The audio session is "
             "set up the way CameraAccess records the glasses' microphone over Bluetooth "
             "HFP, with the glasses' input preferred; when it is not the input, the iPhone's "
             "microphone records instead. An audio engine tap in the input's own format is "
             'converted to 16 kHz as BirdSpotter does, and the tap is set up again when a '
             'device joins or leaves the route, when an interruption such as a call ends, '
             "and when the engine's configuration changes, keeping what was said before. The "
             'recorder notes which microphone recorded ("glasses", "iPhone" or "glasses, '
             'then iPhone") for the Teach screen. A recording under half a second counts as '
             'a tap and gives nothing, anything past a minute is dropped, and the WAV is '
             'built in memory with a 44-byte header: nothing goes to disk.',
        summary='Add the iOS request recorder',
        paragraph='Adds RequestRecorder.swift: the request as 16 kHz mono WAV in memory, the '
                  "glasses' HFP microphone preferred over the iPhone's, the tap set up again "
                  'after route changes, interruptions and engine changes, and which microphone '
                  'recorded. Not compiled here, as there is no Swift toolchain; the build steps '
                  'are in ios/README.md from v63.',
        commands=[PYTEST, RUFF],
    ),
    57: stage(
        next='the teach client and the teach flow on the phone: hold to talk, a 10 second '
             "capture, the upload, then the Mac's spend and read-back questions.",
        layer='iOS: the teach client and flow',
        adds='ios/TrainedEye/Network/TeachClient.swift calls the teach routes: POST '
             '/teach/start with the skill, the request (a WAV recording or typed text) and '
             'the captured JPEGs as a multipart form built in memory, with a 120 second '
             'timeout since the server checks every frame first; GET /teach/status for one '
             'skill, or for the latest session; and POST /teach/answer. '
             "ios/TrainedEye/Teach/TeachFlow.swift runs the Teach screen's steps: a skill "
             'name the backend accepts (snake_case, 2 to 40 characters), hold to talk or '
             'type the request, then a 3 second countdown and a 10 second capture that keeps '
             "three frames a second in memory. After the upload it sends the user's answers "
             "to the server's two questions (spend the estimate, and is the read-back right, "
             'with an optional correction) and polls the status once a second until the run '
             'ends. Frames are dropped once the Mac has them and kept for another try when '
             'the upload fails. A session still running on the Mac can be picked up again at '
             'launch, when the server token is set. The flow uses the app model of the next '
             'stage.',
        summary='Add the iOS teach client and teach flow',
        paragraph='Adds TeachClient.swift (the three teach routes, with the request and frames as '
                  'multipart form data) and TeachFlow.swift (name, request, countdown and '
                  'capture, upload, the spend and read-back answers, polling, and a running '
                  'session picked up at launch). TeachFlow needs AppModel from v58. Not compiled '
                  'here, as there is no Swift toolchain; the build steps are in ios/README.md '
                  'from v63.',
        commands=[PYTEST, RUFF],
    ),
    58: stage(
        next='the app model that ties the parts together: which camera runs, where its '
             'frames go, what an alert does, and what went out while the screen was locked.',
        layer='iOS: the app model',
        adds='ios/TrainedEye/App/AppModel.swift ties the parts together. It starts the '
             'camera the setting asks for (by default the glasses when they are available, '
             "else the iPhone, with DAT's mock glasses standing in for the glasses in Debug "
             'builds) and sends its frames to the WebSocket while watching, or to the teach '
             'flow while capturing; a teach capture is refused while watching. Watching uses '
             "one frame a second and the glasses' low resolution; the teach capture asks for "
             'medium resolution and three frames a second, and waits up to 20 seconds for '
             'the first frame, not counting the permission step. An alert is spoken on the '
             'glasses and shown. The last 10 frames are held in memory for Save clip, and '
             'nothing is written unless the user taps it. Each stay in the background while '
             'watching is counted (frames from the camera, sent, acked and repeats skipped) '
             'for the screen-locked check on the glasses. Changing the watched skills says '
             'hello again, and the skills list comes from GET /skills. Save clip uses '
             'ClipStore, which comes in v63.',
        summary='Add the iOS app model',
        paragraph='Adds AppModel.swift: the camera choice, frames to the socket or the teach '
                  'flow, alerts spoken and shown, the last 10 frames for Save clip, the '
                  'resolution and pace for watching and teaching, the background counts and the '
                  'skills list. It uses ClipStore from v63. Not compiled here, as there is no '
                  'Swift toolchain; the build steps are in ios/README.md from v63.',
        commands=[PYTEST, RUFF],
    ),
    59: stage(
        next="the app's look: a dark, minimal theme with one accent colour in Inter 4.1, and "
             'the cards, buttons and rows every screen uses.',
        layer='iOS: theme, components and the Inter fonts',
        adds="ios/TrainedEye/UI/Theme.swift sets the app's look in one place: a near-black "
             'ground, one raised surface, two greys for text and one accent colour for live '
             'states, the main action and the camera-on indicator, with flat shapes and no '
             'gradients. Text is set in Inter 4.1, whose four static faces (Regular, Medium, '
             'SemiBold and Bold) come in ios/TrainedEye/Fonts/ with their SIL Open Font '
             'License as LICENSE.txt; when a face is missing from the target, the system '
             'font in the same text style and weight is used. ThemeModifier applies the '
             "tokens once at the root, in a dark colour scheme, as in Meta's BirdSpotter "
             'sample; the app entry of v60 puts it there. ios/TrainedEye/UI/Components.swift '
             'has the building blocks every screen uses: a scrolling screen with its title, '
             'a flat card with an optional heading, a filled and an outlined button style, a '
             'status dot, a label and value row, a notice line for help or errors, and the '
             'preview of the newest frame.',
        summary='Add the iOS theme, components and Inter fonts',
        paragraph='Adds Theme.swift (the colours, spacing and text styles, Inter with the system '
                  'font as the fallback, and a modifier that applies them at the root) and '
                  'Components.swift (screen, card, two button styles, status dot, row, notice and '
                  "frame preview), with Inter 4.1's four static faces and their license in "
                  'ios/TrainedEye/Fonts/. Not compiled here, as there is no Swift toolchain; the '
                  'build steps are in ios/README.md from v63.',
        commands=[PYTEST, RUFF],
    ),
    60: stage(
        next='the app entry that configures the DAT SDK, the five tabs under a camera-on '
             "bar, and the Info.plist and entitlements, with the SDK's analytics and crash "
             'reporting off.',
        layer='iOS: app entry, tabs and project files',
        adds="ios/TrainedEye/App/TrainedEyeApp.swift is the app entry: it configures Meta's "
             'DAT SDK once at launch, before the app model reads it, applies the theme at '
             "the root, and hands the Meta AI app's callbacks (registration and permission) "
             'to the SDK, as the CameraAccess sample does. ios/TrainedEye/UI/RootView.swift '
             'lays out the five tabs, Connect, Teach, Skills, Watch and Clips, with a bar '
             'above every one while any camera is on: it says which camera, or that the '
             'glasses are folded or out of reach, and opens the Watch screen when tapped. It '
             'also asks before the Meta AI app opens for the camera permission. '
             "ios/TrainedEye-Info.plist has the trainedeye:// callback scheme, the SDK's "
             'settings with its analytics and crash reporting turned off, the background '
             'modes (processing, Bluetooth, audio and the external accessory link), the '
             'Inter fonts, the camera, microphone, Bluetooth and local network usage '
             'strings, plain HTTP on the local network, and file sharing for saved clips. '
             "ios/TrainedEye.entitlements has the keychain group both of Meta's samples "
             'declare. The screens come in the next three stages.',
        summary='Add the iOS app entry, tabs and project files',
        paragraph='Adds TrainedEyeApp.swift (the SDK configured at launch, the theme applied at '
                  "the root, and the Meta AI app's callbacks forwarded to the SDK), "
                  'RootView.swift (the five tabs, the camera-on bar and the permission question), '
                  'TrainedEye-Info.plist and TrainedEye.entitlements. RootView needs the screens '
                  'of v61 to v63. Not compiled here, as there is no Swift toolchain; the build '
                  'steps are in ios/README.md from v63.',
        commands=[PYTEST, RUFF],
    ),
    61: stage(
        next='the Connect screen (the backend address and token, the glasses, the camera '
             "choice, the Debug mock) and the Teach screen with its capture and the Mac's "
             'questions.',
        layer='iOS: Connect and Teach screens',
        adds='ios/TrainedEye/UI/ConnectView.swift is where the app finds the Mac and the '
             'glasses. The backend card takes the address te serve prints and the server '
             'token, says what kind of address it is, and checks the connection with GET '
             '/health, then fetches the skills. The glasses card shows the Meta AI '
             "registration, the paired glasses, their battery and the camera's state, with "
             'buttons to connect or disconnect and to update the glasses or the app on them '
             'when needed. A segmented picker chooses the camera, and Debug builds add a '
             'card for the mock glasses with fold, unfold and a touchpad tap. '
             'ios/TrainedEye/UI/TeachView.swift is the Teach screen: the name, hold to talk '
             '(at most 30 seconds) or type the request, with a note that skills never cover '
             'a person, then the 10 second capture with its preview, frame size and '
             'countdown, and Send to the Mac. During the run on the Mac it shows the '
             'estimate to agree to, the read-back to confirm or correct, the last 8 progress '
             'lines, and the result and spend.',
        summary='Add the iOS Connect and Teach screens',
        paragraph='Adds ConnectView.swift (the backend address, token and health check, the '
                  "glasses' registration and updates, the camera picker and the Debug mock card) "
                  'and TeachView.swift (the name, request and capture cards, then the spend '
                  'question, the read-back, the progress and the result). Not compiled here, as '
                  'there is no Swift toolchain; the build steps are in ios/README.md from v63.',
        commands=[PYTEST, RUFF],
    ),
    62: stage(
        next='the Skills screen (each taught skill with its watch switch and status) and the '
             'Watch screen (alerts, scores, the connection, the glasses and the background '
             'counts).',
        layer='iOS: Skills and Watch screens',
        adds='ios/TrainedEye/UI/SkillsView.swift lists the taught skills from GET /skills, '
             'fetched when the screen opens, on a pull or with Refresh. Each has a switch '
             'for watch mode, its alert text, its status (training on Modal, trained, ready '
             "with its threshold, else how far the teach flow got), the newest run's time "
             'and its quiet time after an alert in minutes. Changing a switch while watching '
             'says hello again. ios/TrainedEye/UI/WatchView.swift starts and stops watching '
             'and shows which camera is on and the newest frame; the last alert, with its '
             'probability, whether Muse Spark agreed and whether it was spoken on the '
             "glasses; each watched skill's latest score next to its threshold; the "
             'connection with frames sent and acked; the last stay in the background with '
             "what the camera gave, what went out and what came back acked; and the glasses' "
             'camera state, frame size, recoveries and battery. The Save the last 10 seconds '
             'button is on only while watching, and the screen says frames are stored only '
             'that way.',
        summary='Add the iOS Skills and Watch screens',
        paragraph='Adds SkillsView.swift (the skills with their switches, status and quiet time) '
                  'and WatchView.swift (start and stop, the last alert and its cloud check, '
                  'scores against thresholds, the connection, the background counts, the glasses '
                  'card and Save clip). Not compiled here, as there is no Swift toolchain; the '
                  'build steps are in ios/README.md from v63.',
        commands=[PYTEST, RUFF],
    ),
    63: stage(
        next='the clips the user saves, as frame folders te watch can replay, and '
             'ios/README.md: how to build the app in Xcode and what to check on the glasses.',
        layer='iOS: saved clips and the build guide',
        adds='ios/TrainedEye/Clips/ClipStore.swift keeps the clips the user saved from the '
             'Watch screen, and nothing else: one folder per clip under Documents/Clips, '
             'named by the time it was saved, with the frames as f000.jpg, f001.jpg and so '
             'on and a clip.json (when, which camera, the watched skills, the last alert). A '
             'folder of frames is what te watch --clip and te teach --clip read, so a clip '
             'copied to the Mac can be replayed there. Nothing is written without the tap, '
             'and a clip goes only when the user deletes it. '
             'ios/TrainedEye/UI/ClipsView.swift lists them newest first, opens one to its '
             "frames and deletes it after a confirmation. With these the app's files are all "
             'there. ios/README.md says how to build it in Xcode (the project, DAT 1.0.1 as '
             'an exact version, the build settings, the Info.plist and capabilities), how to '
             'point it at te serve, the mock device, the privacy rules, the files table and '
             'the checks still to do on the glasses. The code has still not been compiled.',
        summary='Add the iOS clips and the Xcode build guide',
        paragraph='Adds ClipStore.swift (saved clips as frame folders with a clip.json under '
                  'Documents/Clips, written only on Save clip) and ClipsView.swift (the list, the '
                  'frames and delete). Adds ios/README.md: the Xcode project, the DAT package, '
                  'the build settings, the Info.plist, capabilities, the mock device, the files '
                  'table and the checks still to do on the glasses. Nothing here is compiled, as '
                  'there is no Swift toolchain; the README says how to build the app in Xcode.',
        commands=[PYTEST, RUFF],
    ),
    64: stage(
        next='the local page for hand-labelling generated training images, which shows each '
             "image with the question Muse Spark was asked and never Spark's answer.",
        layer='Hand labels for generated images',
        adds='backend/labeltool/images.py is the first of two local labelling tools, for '
             "Experiment 2's question of how often Muse Spark is wrong. sample_images() "
             "picks generated images at random from a skill's labels.jsonl, the same ones in "
             'the same order for the same seed, and never a real keyframe. The page shows '
             "one image at a time with the question Spark was asked, taken from the skill's "
             "spec.json, but never Spark's answer or the file name, since generated files "
             'are named for the answer they were made to show. The keys y, n and u answer '
             'yes, no or unsure, and Backspace goes back. Each answer is saved at once to a '
             'JSON file written in one step, the page resumes at the first image without an '
             'answer, and labels in the file for other images are kept. The page is '
             'static/images.html; static/style.css is the look both labelling tools share, '
             'with Inter bundled in static/fonts/ under the SIL Open Font License. te label, '
             'which starts the page, comes with te eval.',
        summary='Add the hand-labelling page for generated images',
        paragraph="Adds backend/labeltool/images.py: a seeded sample of a skill's generated "
                  'images (never real keyframes), a local Starlette page that shows each one with '
                  "Spark's question but not its answer, and every yes, no or unsure saved at once "
                  'so the page can resume. Adds the shared style.css and the bundled Inter font '
                  'with its license. Tests cover the sampling, what the page shows, saving and '
                  'resuming, and bad image lists.',
        commands=[PYTEST, RUFF],
    ),
    65: stage(
        next="the clip labelling page: marking where a skill's condition is true in a "
             'recorded clip, saved as the time ranges te calibrate reads.',
        layer='Hand labels for recorded clips',
        adds='backend/labeltool/clips.py is the second labelling tool: it marks where a '
             "skill's condition is true in a recorded clip. Its page, static/clips.html, "
             'plays the clip with a bar under it: [ marks where the condition becomes true '
             'and ] where it stops, Space plays or pauses, the arrow keys step 1 s (0.1 s '
             'with Shift), and Ctrl or Cmd with S saves. The ranges can be fixed in a list '
             'before saving. clean_ranges() sorts them, merges the ones that overlap or '
             "touch, and refuses any range that is not inside the clip's length. The file is "
             'one line, {"clip", "duration_s", "true"}, the format te calibrate --labels '
             'reads; an existing file is loaded to resume, and one that holds labels for a '
             'different clip is refused. The video is served with Range requests answered as '
             '206, which Safari needs. Test labels come from a person only.',
        summary='Add the clip labelling page',
        paragraph='Adds backend/labeltool/clips.py and static/clips.html: a local page that plays '
                  'a recorded clip, marks with [ and ] where the condition is true, and saves the '
                  'sorted, merged ranges in the format te calibrate reads. Ranges outside the '
                  'clip are refused, an existing file resumes, and labels for another clip are '
                  'refused. Tests cover the page, range requests for the video, saving, bad '
                  'ranges and resuming.',
        commands=[PYTEST, RUFF],
    ),
    66: stage(
        next='the evaluation metrics from te watch frame logs and human labels (frame and '
             'event scores, false alarms per hour, latency, dollars per hour) and the shared '
             'SVG chart style.',
        layer='Evaluation metrics and the chart style',
        adds="backend/eval/metrics.py scores watch mode from te watch's frame logs and the "
             'human labels of each clip. Per skill, over one or more clips, a frame says yes '
             'when it scored at or over the threshold the watcher used, and it is true when '
             'its time falls inside a labelled range: that gives frame precision, recall, F1 '
             'and accuracy. Each labelled range is one event, caught within the window when '
             'its first alert comes at most 5 s after it starts, or caught at all when any '
             'alert falls inside it. It also counts false alarms per hour, time to alert, '
             'the median and 90th percentile latency, student errors, and dollars per hour '
             "in three parts: the student's GPU, the Muse Spark checks and any per-frame "
             'scorer. backend/eval/svg.py holds what every chart shares: one accent colour '
             'and a grey, Inter with the system sans as fallback, light colours with dark '
             'ones under prefers-color-scheme, dot markers with a hover title, and all text '
             'escaped.',
        summary='Add the evaluation metrics and the shared SVG style',
        paragraph='Adds backend/eval/metrics.py: frame precision, recall, F1 and accuracy, event '
                  'recall within 5 s and at all, false alarms per hour, time to alert, latency '
                  'and dollars per hour, from frame logs and labelled true ranges. Adds '
                  'backend/eval/svg.py, the palette, style, escaping, axes and marks every chart '
                  'uses, light and dark. Tests cover the metrics on one clip and several, a log '
                  'written by the watcher, the costs, bad inputs, and the SVG pieces.',
        commands=[PYTEST, RUFF],
    ),
    67: stage(
        next='the evaluation charts drawn as SVG by hand: grouped bars, F1 against the '
             'number of synthetic images, F1 against dollars per hour, and the live timeline '
             'of one clip.',
        layer='Evaluation charts and the live timeline',
        adds='backend/eval/charts.py draws the experiment charts as SVG with the pieces in '
             'svg.py. grouped_bars() shows one group per variant and one or two series from '
             '0 to 1, the first in the accent colour and the second in grey, with each value '
             'written on its bar and n/a where one is missing (Experiments 1 and 2). '
             'line_chart() plots F1 against the number of synthetic images on a log scale, '
             'with a note such as the generation cost under each point (Experiment 3), and '
             'scatter() places each judge by its F1 and its dollars per hour, the points '
             'marked highlight (the fine-tuned one) in the accent colour and points without '
             'numbers listed under the chart (Experiment 4). backend/eval/timeline.py draws '
             "one skill's frames from a frame log: the yes-probability as a line with gaps "
             'where the student failed, the threshold dashed, the labelled true ranges '
             'shaded, and each alert as a dot with the words the glasses spoke, its label on '
             'one of two rows so two close alerts do not overlap.',
        summary='Add the evaluation charts and the live timeline',
        paragraph='Adds backend/eval/charts.py (grouped bars, a log-scale line of F1 against '
                  'synthetic images, and a scatter of F1 against dollars per hour) and '
                  "backend/eval/timeline.py (one skill's probability, threshold, labelled ranges "
                  'and spoken alerts over a clip). Both write plain SVG with the shared style. '
                  'Tests cover the light and dark style, labels on every value, missing values, '
                  'escaping, gaps for student errors and ranges outside the clip.',
        commands=[PYTEST, RUFF],
    ),
    68: stage(
        next='the regression check of the base model against the fine-tuned one on general '
             'visual questions, and Muse Spark as a per-frame scorer for Experiment 4.',
        layer='Regression check and Spark as scorer',
        adds='backend/eval/regression.py asks whether fine-tuning on yes or no frames costs '
             'general visual skill. It reads a JSON-lines file of questions written by hand, '
             'each with an optional image, an answer and a match rule (exact, contains or '
             'choice), asks the base and the fine-tuned model each one through an '
             'OpenAI-compatible endpoint, at temperature 0 by default, and grades the '
             'replies, ignoring case, spaces and end punctuation. A failed request counts as '
             'wrong and is listed, and the report gives accuracy with and without images, '
             'the questions whose result changed and a markdown table. '
             'backend/eval/spark_scorer.py makes Muse Spark a per-frame judge with the '
             "student's shape, so the watcher uses it unchanged: yes is 1.0, no 0.0 and "
             'unsure 0.5, and each request is cached and billed to the skill. A latency log '
             'keeps the seconds of the frames actually sent, so a cached frame reports the '
             'latency measured when it was sent, and one cached by another caller is marked '
             'as having none.',
        summary='Add the regression check and the Spark scorer',
        paragraph='Adds backend/eval/regression.py: hand-written general questions asked of the '
                  'base and the fine-tuned model through OpenAI-compatible endpoints, graded by '
                  'exact, contains or choice rules, with failed requests counted as wrong and '
                  "listed. Adds backend/eval/spark_scorer.py: Muse Spark's yes, no or unsure as a "
                  'probability for every frame, cached and billed to the skill, with a latency '
                  'log of the frames actually sent. Tests cover grading, the comparison, failed '
                  "requests, bad question rows, and the scorer's billing, cache, latency and "
                  'errors.',
        commands=[PYTEST, RUFF],
    ),
    69: stage(
        next='the results write-up: the SVG charts and results.md, one section per '
             'experiment with its tables of accuracy, speed and cost.',
        layer='The results write-up',
        adds='backend/eval/writeup.py turns the scored evaluations into the charts and '
             'results.md that run_eval, in v71, writes. write_charts() saves the SVG charts: '
             'grouped bars for Experiments 1 and 2, a line chart per skill for Experiment 3, '
             'the cost scatter for Experiment 4 and a live timeline for each clip that asks '
             'for one. render() returns the text of results.md: one section per experiment '
             'with its charts, a table of accuracy (precision, recall, F1, events caught, '
             'false alarms per hour, time to alert, and the threshold with where it came '
             'from) and one of speed and cost. An experiment with no evaluations says so, '
             'and evaluations waiting for clips are listed as pending. Experiment 2 also '
             "gets Muse Spark's error rate against the hand labels and the filter's "
             'agreement, Experiment 4 its judges pooled over the skills, and after the '
             'experiments come the regression check, or a note that it has not run, and '
             'links to the failure galleries, each when run_eval passes it in. Every number '
             'comes from the computation passed in; n/a means there was nothing to compute '
             'it from.',
        summary='Add the results write-up: charts and results.md',
        paragraph='Adds backend/eval/writeup.py, which writes the SVG charts and renders '
                  'results.md from the scored evaluations: a section per experiment with its '
                  'charts, an accuracy table and a speed and cost table, pending runs listed, '
                  "Spark's error rate under Experiment 2, the pooled judges under Experiment 4, "
                  'and the regression check and galleries when present. Tests cover the tables, '
                  'pending experiments, the charts, the threshold column and the extra sections.',
        commands=[PYTEST, RUFF],
    ),
    70: stage(
        next="the failure galleries of Experiment 1: each run's worst false alarms and "
             'misses as small frames of the recorded clips, on a page that stays on this '
             'Mac.',
        layer='Failure galleries for Experiment 1',
        adds="backend/eval/gallery.py writes Experiment 1's failure galleries next to "
             'results.md. For each scored Experiment 1 evaluation it picks the worst false '
             'alarms (frames labelled false that scored over the threshold, highest p first) '
             'and the worst misses (frames inside a true range that stayed under it, lowest '
             'p first), at least 5 s apart so one long mistake does not fill the row. Each '
             'frame is read again from its clip at the logged time, at the frame rate taken '
             "from the log's frame step, and saved as a JPEG thumbnail at most 240 pixels a "
             'side; each clip is read once, however many evaluations replayed it. Each run '
             'gets one HTML page with the time, p and threshold under every frame. Without '
             'the clip, or without ffmpeg for a video, the page lists the frames without '
             'pictures. The frames may show faces, so galleries/ holds its own .gitignore, '
             'nothing goes to W&B, and the page says to run te blur before publishing any '
             'frame. Its tests come in v71 with the evaluation runner, as they run galleries '
             'through run_eval.',
        summary='Add the failure galleries for Experiment 1',
        paragraph='Adds backend/eval/gallery.py: the worst false alarms and misses of each scored '
                  'Experiment 1 evaluation, spread at least 5 s apart, read again from the '
                  'recorded clips as thumbnails, on one local HTML page per run with a .gitignore '
                  'beside it. Its own tests arrive in v71 with the evaluation runner, since they '
                  'run the galleries through run_eval.',
        commands=[PYTEST, RUFF],
    ),
    71: stage(
        next='the evaluation report: a JSON plan of evaluations scored into results.json, '
             'results.md, the charts and the galleries, with W&B logging only when asked.',
        layer='The evaluation plan and report',
        adds='backend/eval/report.py is what te eval and make eval will run. The plan is a '
             'JSON file of evaluations, each with a unique name, its experiment (1 to 4), '
             "skill and variant, one or more pairs of a te watch frame log and the clip's "
             "human labels, the student's GPU price (none for the Mac), the price of a Spark "
             'check or of a per-frame scorer, and for Experiment 3 the synthetic images and '
             'what they cost. An evaluation whose log or labels do not exist yet is pending, '
             'so the plan can come before the clips. run_eval() scores the rest, records the '
             "thresholds each evaluation's logs were scored at and where they came from, "
             "pools Experiment 4's judges over the skills, reads Spark's error rate, label "
             'agreement and the regression result when given, and writes results.json, '
             'results.md, the charts and the galleries. With log_wandb, and wandb installed, '
             'it logs every number, the table and the charts to one W&B run in the '
             'trained-eye project, and the key is never shown. The gallery tests held back '
             'from v70 come back here.',
        summary='Add the evaluation plan and report',
        paragraph="Adds backend/eval/report.py: the plan's format, checked by pydantic, with "
                  'evaluations pending until their logs and labels exist, and run_eval, which '
                  'scores the rest into results.json, results.md, the charts and the galleries, '
                  "pools Experiment 4's judges, takes in Spark's error rate and the regression "
                  'check, and logs to W&B only when asked. The gallery tests come with it. Tests '
                  'cover the output, bad plans, W&B logging and its team, the pooled judges, and '
                  'the galleries with and without the clip.',
        commands=[PYTEST, RUFF],
    ),
    72: stage(
        next='te eval and te label on the command line, make eval, and the README results '
             'block written from results.json only.',
        layer='te eval, te label and make eval',
        adds='backend/cli_eval.py adds two commands, registered by backend/cli.py. te eval '
             '--plan <plan> --out <folder> runs report.run_eval with the data folder and '
             'eval/regression.json, says how many evaluations are scored and how many wait '
             'for clips, and points to the failure galleries with a reminder that they stay '
             'local; --wandb logs the run, and --wandb-entity names the team and needs '
             '--wandb. te label starts a labelling tool on 127.0.0.1, port 8766 by default, '
             'and opens the browser: --clip for a recorded clip, its labels saved beside it '
             "by default under the clip's name with .labels.json in place of its extension, "
             'or --skill with --sample (200 by default) and --seed for generated images. The '
             "Makefile's make eval runs te eval on eval/plan.json into eval/. "
             'backend/eval/readme.py writes, with te eval --readme, a results block between '
             'two marker comments in a README: headline tables, chart links, the regression '
             "line and Spark's error rate, all copied from results.json, whose SHA-256 it "
             'names as the source. It refuses when nothing is scored or the markers are '
             'broken, leaving the file as it was, and never links the galleries.',
        summary='Add te eval, te label, make eval and the README block',
        paragraph='Adds backend/cli_eval.py with te eval (the plan scored into results.md, the '
                  'charts and the galleries, with optional W&B) and te label (the clip or image '
                  'labelling tool on localhost), registered by backend/cli.py, and a Makefile '
                  'whose make eval runs te eval on eval/plan.json. Adds backend/eval/readme.py, '
                  "which writes te eval --readme's results block from results.json only and names "
                  "its SHA-256. Tests cover both commands, the block's contents, the markers and "
                  'the refusals.',
        commands=[PYTEST, RUFF, 'python -m backend.cli eval --help', 'python -m backend.cli label --help'],
    ),
    73: stage(
        next="labelled session clips as real training and validation frames, Muse Spark's "
             "label error rate against hand labels, and Experiment 2's arm without the Spark "
             'filter.',
        layer="Session clips and Spark's error rate",
        adds='backend/dataengine/realclips.py turns the labelled session clips of the '
             'recording protocol into real examples for Experiment 1: session 1 clips train, '
             'session 2 clips only validate, and session 3 test clips are never read. Frames '
             'are sampled at 1 fps; a frame inside a labelled range answers yes and one '
             'outside no, and frames within 1.5 s of a change are left out, as the moment of '
             "change is unclear. The examples go to dataset-clips/ with the teach dataset's "
             'prompt layout, and clips without usable labels are listed as skipped. '
             "backend/eval/spark_errors.py compares Muse Spark's labels with the hand labels "
             "from the labelling page: Spark's error rate with a 95% Wilson interval, a "
             'confusion table, and the error rates of the generated-for labels and of the '
             'ones the filter keeps. write_unfiltered_dataset() writes dataset-unfiltered/, '
             'every generated image with the label it was made for. '
             'docs/recording-protocol.md gains notes on recording the teach look. The bundle '
             'tests for session clips and the keyframe crop test come back here; the te '
             'clips test arrives in v75 with that command.',
        summary="Add session clips, Spark's error rate, unfiltered arm",
        paragraph='Adds backend/dataengine/realclips.py (labelled session 1 and 2 clips sampled '
                  'at 1 fps into dataset-clips/, frames near a change left out, session 2 for '
                  "validation only) and backend/eval/spark_errors.py (Spark's error rate against "
                  "hand labels with a Wilson interval, and the unfiltered arm's dataset). Adds "
                  'notes on recording the teach look to docs/recording-protocol.md. Tests cover '
                  'the frame answers, the clip search, the splits, unreadable videos, the error '
                  'rates and the unfiltered arm, and the held-back bundle and crop tests come '
                  'back.',
        commands=[PYTEST, RUFF],
    ),
    74: stage(
        next='the experiment plan: every training run of Experiments 1 to 3 at the same '
             'fixed steps, priced from the smoke with exports, serving and the Modal credit.',
        layer='The experiment plan and its price',
        adds='backend/eval/plan.py lists every te train run of Experiments 1 to 3 with its '
             'arguments: exp1-real, exp1-synthetic and exp1-mixed, exp2-filtered and '
             'exp2-unfiltered, and exp3-25 to exp3-300 for 25, 50, 100, 200 and 300 '
             'generated images per skill. Every run trains the same fixed optimizer steps, '
             "one epoch over 300 images per skill, so Experiment 3's curve measures images "
             'rather than steps. Runs that would train the same bundle are marked as repeats '
             'of the first, which the printed plan says can be skipped. plan_cost() prices '
             "the runs from the smoke's timing as te train does, adds the final run, an "
             'export per model that needs one, a guessed number of serving hours per '
             'evaluated model and the Modal spend so far, and compares the total with the '
             '$150 credit; runs already finished on this Mac count in the spend instead. '
             "spark_usd_per_check() gives a Spark check's price, measured or estimated. The "
             "held-back tests of te train's pricing and fixed steps, te watch --scorer "
             "spark, the credit check before te train and te export, and the plan's Spark "
             'price come back here.',
        summary='Add the experiment plan with fixed steps and prices',
        paragraph='Adds backend/eval/plan.py: every training run of Experiments 1 to 3 with its '
                  'te train arguments and the same fixed optimizer steps, runs with the same '
                  "bundle marked, and the price of the whole plan from the smoke's timing with "
                  'the final run, exports, serving and the spend so far, checked against the $150 '
                  "Modal credit. The held-back tests of te train's pricing, te watch --scorer "
                  'spark and the credit check come back with it. Tests cover the runs, their '
                  'counts, the cost, the credit check and the progress read from finished runs.',
        commands=[PYTEST, RUFF],
    ),
    75: stage(
        next='te experiments, te unfiltered, te clips, te spark-errors and te regress, and '
             'eval/plan.json written from the experiment plan with its test and validation '
             'clips.',
        layer='The experiment commands and eval/plan.json',
        adds='backend/cli_experiments.py adds five commands, registered by backend/cli.py. '
             'te experiments --skills prints every training run with its price and what it '
             'still needs; --write-plan also writes eval/plan.json, never over an existing '
             "one unless --force. te unfiltered writes Experiment 2's arm without the Spark "
             "filter, te clips turns a skill's labelled session clips into dataset-clips/ "
             "and says what it left out, and te spark-errors saves Spark's error rate "
             'against the hand labels; all three are free. te regress asks a base and a '
             "fine-tuned endpoint the questions with the student's sampling and saves "
             'eval/regression.json and regression.md. backend/eval/plan_file.py builds the '
             "plan te eval reads: one evaluation per run and skill, plus Experiment 4's "
             'three judges, each with its test clips (or a placeholder named as the protocol '
             'says, pending until recorded), their frame logs and labels, the run whose '
             "replays they read, the prices and the skill's spoken alert. Each run the "
             'student replays also gets its session 2 clips under calibration. The held-back '
             'te clips test comes back here.',
        summary='Add the experiment commands and eval/plan.json',
        paragraph='Adds backend/cli_experiments.py with te experiments, te unfiltered, te clips, '
                  'te spark-errors and te regress, registered by backend/cli.py, and '
                  'backend/eval/plan_file.py, which writes eval/plan.json from the experiment '
                  'plan: one evaluation per run, judge and skill with its clips, logs, labels and '
                  'prices, and the validation clips of each run the student replays, for '
                  'calibration. The held-back te clips test comes back. Tests cover each command, '
                  'the plan skeleton and the shared replays.',
        commands=[PYTEST, RUFF, 'python -m backend.cli experiments --help', 'python -m backend.cli clips --help'],
    ),
    76: stage(
        next="the replay engine that fills the plan's frame logs (warm-up, validation clips, "
             'then test clips) and the W&B report built from results.md.',
        layer='Replay engine and the W&B report',
        adds='backend/eval/replay.py is the engine of te replay, which comes in v77. '
             "plan_jobs() takes a run's clips from eval/plan.json, its validation clips "
             'first and then the test clips of every evaluation whose run it is, each log '
             'once, and plan_scorer() says whether the student or Muse Spark scores them. '
             'warm_up() sends one frame every 15 s until a cold endpoint answers or the '
             "deadline passes (900 s is te replay's default), and gives up at once on a "
             'client error such as a wrong key. replay_clip() replays one clip through a '
             'fresh Watcher into <log>.partial and renames it to the log only when at least '
             'one frame scored, so an interrupted replay stays pending. '
             'backend/eval/wandb_report.py plans the W&B report "Trained Eye: experiments": '
             "one section per experiment with its text from results.md, the training runs' "
             "loss panels found by their exp tags, and the eval run's table and charts. It "
             'is saved only as a draft, only after a yes, and the key is never shown. '
             "replay.py's tests arrive in v77 with te replay.",
        summary='Add the replay engine and the W&B report',
        paragraph="Adds backend/eval/replay.py: a run's validation and test clips taken from the "
                  'plan, the endpoint warm-up, and one clip replayed through the watcher into a '
                  'log that appears only when frames scored. Adds backend/eval/wandb_report.py, '
                  "which plans the W&B report from results.md and the runs' tags and saves it as "
                  "a draft only after a yes. Tests cover the report's sections and panels, the "
                  "eval run check and saving; replay.py's tests come with te replay.",
        commands=[PYTEST, RUFF],
    ),
    77: stage(
        next="te replay: a run's pending clips of eval/plan.json replayed into the plan's "
             'logs, its threshold calibrated on its own validation clips before the test '
             'clips.',
        layer="te replay: filling the plan's logs",
        adds='backend/cli_replay.py adds te replay, registered by backend/cli.py. te replay '
             '--run <run> reads eval/plan.json, refuses a run the plan does not name (or one '
             "that reads another run's replays, naming that run), and replays only the clips "
             'whose logs do not exist yet. With the student it needs TE_STUDENT_URL, warns '
             'when te serve-student last deployed another run, and waits for the endpoint to '
             "warm up. The validation clips are replayed first and the run's threshold is "
             'calibrated on them as te calibrate --run does, and again when their labels '
             'change later; a skill whose calibration fails or is missing keeps its test '
             'clips pending unless --threshold is given, so no test log is scored at the 0.5 '
             'default. With the spark scorer the frames are priced and te replay asks before '
             'sending any, even with --yes when the price is over $15, and every threshold '
             'is 1.0, so only a yes counts. At the end it says how many clips are ready for '
             'make eval and why the rest are still pending. tests/test_replay.py also covers '
             'replay.py from v76.',
        summary="Add te replay for the plan's pending clips",
        paragraph='Adds backend/cli_replay.py with te replay, registered by backend/cli.py: a '
                  "run's pending clips from eval/plan.json, the endpoint warm-up, the validation "
                  'clips replayed and calibrated first, the test clips scored at that threshold, '
                  "and Spark's frames priced before any is sent. Tests cover the warm-up, a log "
                  'written only when frames scored, jobs and scorers from the plan, calibration '
                  'before the test clips, refusals, the served-run warning, the Spark price and '
                  'recalibration.',
        commands=[PYTEST, RUFF, 'python -m backend.cli replay --help'],
    ),
    78: stage(
        next="face blurring for anything published: OpenCV's Haar cascades plus manual "
             'boxes, every output rebuilt without its metadata, and a blur record for each '
             'video.',
        layer='Face blurring',
        adds='backend/release/faces.py blurs faces in anything published. blur_boxes blurs '
             'given boxes with Pillow alone, first shrinking each box to 8 pixels across its '
             'shorter side, so the detail is gone and not only softened. detect_faces uses '
             'the Haar cascades OpenCV ships (frontal, and profile facing either way) and '
             'imports OpenCV only when called, naming opencv-python-headless in its error. '
             'The cascades miss small, dark, tilted or covered faces, so manual boxes, per '
             "image file or per stretch of a video, are added to the detector's, and a box "
             'that misses the picture or falls on no frame is refused before anything is '
             'written. Every output is rebuilt from the pixels alone: EXIF, XMP and ICC are '
             'left out and an EXIF rotation is applied first. It is for real footage only, '
             'as Muse Image outputs keep their provenance. blur_folder writes sources.json '
             "(each original's SHA-256 to its blurred file), and blur_video runs ffmpeg, "
             'drops the audio and the container metadata, and writes a blur record with the '
             'hashes of the source and the output.',
        summary='Add face blurring for published images and video',
        paragraph='Adds backend/release/faces.py: boxes blurred with Pillow so the detail is '
                  "gone, faces found with OpenCV's Haar cascades (imported only when needed), and "
                  'manual boxes for faces the detector misses, refused when they miss the '
                  'picture. Every output is rebuilt from the pixels, so EXIF and other metadata '
                  "are left out. Folders get a sources.json of the originals' hashes, and videos "
                  'go through ffmpeg without audio and with a blur record. Tests cover the blur, '
                  'the metadata, folders, videos and manual boxes; the detection test skips '
                  'without OpenCV.',
        commands=[PYTEST, RUFF],
    ),
    79: stage(
        next='the Hugging Face model card, written only from the recipe, the bundle manifest '
             "and the eval results, under the base model's apache-2.0 license.",
        layer='The Hugging Face model card',
        adds='backend/release/model_card.py writes the model card: README.md with its front '
             'matter, then the training, data, evaluation, uses, limitations, privacy and '
             "license sections. Every number comes from a file: the recipe's values from "
             'axolotl.RECIPE, the examples per skill, split, source and answer from the '
             "bundle's manifest.json, and results only from the scored evaluations in te "
             "eval's results.json, copied as they are with that file's SHA-256; without "
             'results the card says it is not evaluated yet. A run with a fixed step count '
             'says so. The card is written for the LoRA adapter or the merged build and says '
             'whether the bundle has generated images: they are never included, only the '
             'prompt manifests, and it names the skills that ship face-blurred keyframes for '
             "te regenerate. The license is the base model's, apache-2.0 from the Muse "
             'Glimmer card at the pinned revision, with a statement that the weights are a '
             "modification and a link to Meta's Usage Policy. The release folder and the "
             'upload come in the next stage.',
        summary='Add the Hugging Face model card',
        paragraph="Adds backend/release/model_card.py: the card's front matter and sections, with "
                  "the recipe's values, the bundle's counts and the eval results copied from "
                  'their files, the adapter or merged wording, the paragraph on generated images '
                  "and the base model's apache-2.0 license with Meta's Usage Policy. Tests check "
                  'that the card says only what its files say, and its wording for fixed steps '
                  'and a bundle of real frames only.',
        commands=[PYTEST, RUFF],
    ),
    80: stage(
        next='the release folder (the card, the weights, the prompt manifests and the base '
             "model's LICENSE), the guarded Hugging Face upload and the collection of "
             'release repos.',
        layer='The release folder and its upload',
        adds='backend/release/package.py assembles and publishes the Hugging Face release. '
             'prepare_release writes the folder and uploads nothing: the model card, the '
             "run's top-level weight and config files (no pickles, checkpoints or trainer "
             'README), the prompt manifests of the skills with generated images, the base '
             "model's license as LICENSE (backend/release/BASE_LICENSE.txt, the Apache-2.0 "
             "text) and release.json with every file's SHA-256. Given a blur_folder output "
             "per skill, it also ships that skill's face-blurred keyframes: only those its "
             "sources.json lists whose originals the manifest's edits start from, and a "
             'generated image among them or a missing keyframe stops it. upload() refuses '
             "without the author's yes, without LICENSE or weights, with any other image or "
             'video in the folder, and when an existing repo has another visibility than the '
             'one asked for; huggingface_hub is imported only then, and the token never '
             'shows in an error. make_collection groups model and dataset repos into a '
             "collection, again only with the author's yes and only when every repo has the "
             'visibility asked for. The test that te regenerate finds the shipped keyframes '
             'comes in stage v81.',
        summary='Add the release folder, upload and collection',
        paragraph="Adds backend/release/package.py and the base model's license text: "
                  'prepare_release assembles the card, the weights, the prompt manifests, LICENSE '
                  'and release.json, with face-blurred keyframes only when blur_folder made them '
                  "from frames the manifests need. upload() refuses without the author's yes, "
                  'with generated or unlisted images, without weights or LICENSE, or when the '
                  "repo's visibility is not the one asked for, and make_collection() checks the "
                  "yes and every repo's visibility; neither shows the token in an error. Tests "
                  'cover the folder, the refusals and the collection against a fake Hugging Face '
                  'Hub; the keyframe test that needs te regenerate comes in v81.',
        commands=[PYTEST, RUFF],
    ),
    81: stage(
        next='regeneration from the prompt manifests (copied out of data/ without images, '
             "then the edits run again) and the download of a run's weights from the Modal "
             'volume.',
        layer='Regeneration and fetching the weights',
        adds='Generated images are never published, so backend/regenerate.py ships what '
             "rebuilds them. export_manifests copies each skill's synthetic/manifest.jsonl "
             'into a committed folder with only the fields that remake an image and check '
             'its provenance (no image bytes, file names or costs), sorted so reruns give '
             'the same file. regenerate runs the same edits again through MetaClient, so '
             'every call is cached, capped and in the ledger, and free where the images were '
             'made. It finds the keyframes by SHA-256, or face-blurred copies through the '
             'sources.json blur_folder writes, hands the plan and the estimate to a confirm '
             'callback before any call, so a no spends nothing, and counts how many images '
             "match the manifest's hashes, never failing on a difference. "
             "backend/release/weights.py downloads a run's adapter or merged build from the "
             'Modal volume trained-eye: only the top-level weight and config files, each '
             'checked against the size the volume lists, into a hidden folder first, so the '
             'result is always complete. The release test of the blurred keyframes, held '
             'back in v80, comes back here.',
        summary='Add regeneration from manifests and weight fetching',
        paragraph='Adds backend/regenerate.py (the prompt manifests copied out of data/ without '
                  'images, and their edits run again through the cached, capped client, with the '
                  'estimate first and a count of matching hashes) and backend/release/weights.py '
                  "(a run's top-level weight files from the Modal volume, size-checked, into a "
                  'complete folder). The release test of the blurred keyframes comes back. Tests '
                  'cover both against a fake Meta API and a fake volume.',
        commands=[PYTEST, RUFF],
    ),
    82: stage(
        next='the human-labelled session 3 test clips as a Hugging Face dataset folder: only '
             'blurred videos with a matching blur record, under the license the author '
             'chooses.',
        layer='The test clips as a dataset',
        adds='backend/release/test_set.py prepares the held-out test clips for a dataset '
             'repo. Only session 3 clips go in (session3_test_<name>, recorded apart from '
             'the training clips), and only .mp4 videos blur_video wrote, whose blur '
             "record's output SHA-256 is the video's. Each clip needs the author's own "
             'labels from te label, found next to the blurred video or beside the original '
             "under a labels root, <root>/<skill>/. They are checked against the video's "
             'length from ffprobe, with 1.0 second of slack, and, when the original is '
             'there, its hash against the blur record. prepare_test_set writes '
             '<skill>/<clip>.mp4 with its labels, metadata.jsonl, a dataset card that states '
             'the labels are human only and lists the privacy steps, and release.json with '
             "each clip's blur record. There is no default license: the author picks one. "
             'check_folder, which package.upload runs for a test set, refuses any image, an '
             'unlisted or unblurred video, and metadata that does not match release.json.',
        summary='Add the labelled test clips as a dataset folder',
        paragraph='Adds backend/release/test_set.py: blurred session 3 clips with a matching blur '
                  "record and the author's labels, checked against the video's length, written as "
                  'a dataset folder with metadata.jsonl, a dataset card and release.json, under a '
                  'license the author chooses. check_folder stops an upload with any image, an '
                  'unlisted or unblurred video, or metadata that does not match. Tests cover what '
                  'goes in, the license and label checks, and the private dataset upload, with '
                  'ffmpeg and the Hub faked.',
        commands=[PYTEST, RUFF],
    ),
    83: stage(
        next='the M4 release commands: te manifests, te regenerate, te blur, te '
             'fetch-weights, te release, te release-test-set, te release-collection and te '
             'wandb-report, each asking before it spends or publishes unless given --yes.',
        layer='The release commands',
        adds='backend/cli_release.py holds the M4 release commands, and backend/cli.py '
             'registers them. te manifests copies the prompt manifests into manifests/ for '
             'committing. te regenerate prints the plan and the estimate, stops there with '
             '--estimate-only, and asks before spending; over $15 it asks even with --yes. '
             'te blur takes an image folder or a video, with --boxes for faces the detector '
             'missed, and ends with a reminder to look at every frame. te fetch-weights '
             "downloads a run's adapter or merged build into data/weights/<run>/<kind>, "
             'where te release looks by default. te release assembles the folder, with '
             '--keyframes <skill>=<folder> for blurred keyframes, and uploads only with '
             '--upload, a repo and a yes, private unless --public. te release-test-set '
             'builds the dataset folder and uploads only to a private dataset repo. te '
             'release-collection and te wandb-report (a draft report that only the author '
             'can see) ask before making anything.',
        summary='Add the M4 release commands',
        paragraph='Adds backend/cli_release.py with te manifests, te regenerate, te blur, te '
                  'fetch-weights, te release, te release-test-set, te release-collection and te '
                  'wandb-report, and registers them in backend/cli.py. Every command that spends '
                  'or publishes asks first unless given --yes, and te regenerate asks even with '
                  '--yes over $15. Tests run every command with the paid and remote calls faked, '
                  'checking its output and errors and the questions of te regenerate, te '
                  'release-test-set, te release-collection and te wandb-report.',
        commands=[PYTEST, RUFF, 'python -m backend.cli blur --help', 'python -m backend.cli release --help'],
    ),
    84: stage(
        next="the playground's read-only data: recorded clips with their frame logs, the "
             "evaluation's charts, and the media files a page may link to, with a public "
             'mode.',
        layer='Playground: clips, results and media',
        adds='The web playground starts with backend/playground/data.py, which reads local '
             'results when a page asks and never calls Meta or writes a file. A clip is a '
             'folder data/playground/clips/<name>/ with the frame log te watch writes, the '
             "video and optional labels; clip_timeline turns it into each skill's threshold, "
             '[t, p] points and alerts, each alert with the spoken text from the log or else '
             "from the skill's spec. results() lists the evaluation's SVG charts and "
             'results.md. media_path is the only way to a file: clip videos, keyframes, '
             'generated variations, the provided keyframes and the charts, each a plain name '
             'inside its folder. The public copy serves no generated image and no original '
             "keyframe, keyframes only from a te blur folder's sources.json, and a video "
             'only when its blur record matches it. .gitignore now leaves out eval/clips/ '
             'and eval/galleries/, as both hold real footage.',
        summary="Add the playground's clip, result and media reader",
        paragraph='Adds backend/playground/data.py: clips and their frame logs as timelines, the '
                  "evaluation's charts and results.md, and media_path, which reaches only the "
                  'files the pages link to and, in the public copy, only blurred keyframes and '
                  'videos with a matching blur record. .gitignore leaves out eval/clips/ and '
                  'eval/galleries/. Tests cover the clip list, the timeline, the results and what '
                  'each copy serves.',
        commands=[PYTEST, RUFF],
    ),
    85: stage(
        next='how each taught skill was built, for the playground: the read-back, the '
             "keyframes, SAM's box on a teaching frame, the labelled training wall and the "
             'keyframes a visitor may pick.',
        layer='Playground: how each skill was built',
        adds='backend/playground/skills.py reads each taught skill from data/skills/<skill>/ '
             'as the teach flow left it, skipping refused skills, and links images where '
             'they are without copying them. skill_detail gives the read-back, the keyframes '
             "with Spark's and the human labels, SAM's box on the first teaching frame that "
             "has one (read again from SAM's stored text, at no cost, for a skill taught "
             'before the parser existed), and the training wall: up to 48 generated '
             'variations spread over all of them, each with what it was made to show and '
             'what Spark said. examples() picks, per skill, the first 6 positives and 6 '
             'near-misses Spark labelled, for teach it yourself, and provided_keyframes '
             'offers the keyframes a visitor may pick: a curated set in '
             "data/playground/keyframes/, else 3 of each skill's own. The public copy shows "
             'no generated image (an empty wall with a note, and no examples) and only '
             "face-blurred keyframes, whose sources.json gives each one its original's "
             "SHA-256, so SAM's box is still found.",
        summary="Add the playground's skill pages and training wall",
        paragraph="Adds backend/playground/skills.py: each taught skill's read-back, keyframes, "
                  'SAM box on a teaching frame (read again from the stored text when needed) and '
                  'training wall, the precomputed teach-it-yourself examples, and the keyframes a '
                  'visitor may pick. The public copy shows no generated image and only '
                  'face-blurred keyframes. Tests cover each part and the public copy.',
        commands=[PYTEST, RUFF],
    ),
    86: stage(
        next="the playground's spend caps per visitor and per day, and the metered ledger "
             "that holds each live call to its run's reservation.",
        layer='Playground: spend caps and metered runs',
        adds="Two pieces keep the playground's live Meta calls within limits. "
             'backend/playground/budget.py caps spend per visitor and per day (UTC). A '
             'visitor counts by their visitor id and by their address (an IPv6 address by '
             'its /64), the address kept only as a salted hash; each counts against the cap '
             "on its own, so a new id does not reset it. A run's estimate is reserved first "
             'and may grow while the caps allow; when the run ends, the recorded spend '
             'replaces it, and the totals of the last 7 days are saved in a JSON file meant '
             'for a single process. backend/playground/runs.py prices a run before cache '
             'hits (one compile, then an image edit and a label for each of 6 positives and '
             "6 near-misses), wraps the client's ledger so each call must fit in the run's "
             'reservation together with its other calls in flight, and holds the Job the '
             'page polls. A call that would pass a cap is never sent.',
        summary="Add the playground's spend caps and metered runs",
        paragraph='Adds backend/playground/budget.py (caps per visitor and per day, a visitor '
                  'known by an id and a salted hash of their address, estimates reserved and then '
                  "settled with the recorded spend) and backend/playground/runs.py (a run's "
                  "estimate, a ledger wrapper that holds each call to the run's reservation, and "
                  'the Job the page polls). Tests cover the caps, the IPv6 /64, the saved totals, '
                  'calls in flight and the estimate.',
        commands=[PYTEST, RUFF],
    ),
    87: stage(
        next="teach it yourself: a visitor's condition checked by the local policy for free, "
             'then compiled, with 6 edits of a provided keyframe that show it and 6 '
             'near-misses, labelled live within the caps.',
        layer='Playground: teach it yourself',
        adds='backend/playground/api.py runs teach it yourself. Teacher.start takes a '
             'condition of at most 300 characters and a provided keyframe, turns "the kettle '
             'light turns off" into "Tell me when the kettle light turns off" and runs the '
             'local policy check, so a refusal costs nothing. Without a Meta key, or with a '
             'cap reached, it answers with a message and the precomputed examples. Otherwise '
             "it reserves the run's estimate, allows one run at a time per visitor and 4 at "
             'once in all, and works in a thread the page polls: Muse Wonder compiles the '
             'skill, Muse Image makes 6 edits that show the condition and 6 near-misses, and '
             'Muse Spark labels each one. Every call goes through MetaClient under the skill '
             'name playground, and a cap stops the run before its next call. Each image is '
             "returned as a data URL of Muse Image's own bytes, so the only copy on disk is "
             "the client's call cache. backend/playground/publish.py lists what the public "
             'copy may hold for a deploy; it is tested through te playground --deploy in '
             'stage v92.',
        summary='Add teach it yourself and the public file list',
        paragraph="Adds backend/playground/api.py: a live run from a visitor's condition and a "
                  'provided keyframe, with the free policy check first, the estimate reserved, '
                  'one run per visitor and 4 in all, the compile, the edits and the labels in a '
                  'polled thread, and a stop before the next call once a cap is reached. Adds '
                  'backend/playground/publish.py, the files the public copy may hold. Tests run '
                  'teach it yourself against the fake Meta API: 12 labelled variations with no '
                  'image written outside the call cache, a free refusal, bad input and the caps.',
        commands=[PYTEST, RUFF],
    ),
    88: stage(
        next='the playground page, with its clips, skills, teach it yourself, results and '
             'privacy sections, the stylesheet with dark and light themes, and the bundled '
             'Inter font.',
        layer='Playground: the page and its style',
        adds="The playground's page arrives in backend/playground/static/. index.html has a "
             'section for each part: the recorded clips, with the video, the skill picker '
             'and the timeline; how each skill was built, with the keyframes, what SAM looks '
             'at and the training wall; teach it yourself, a form with a condition of at '
             'most 300 characters and a keyframe picker; the results; and the privacy notes '
             '(faces blurred, frames not stored, Muse Image provenance kept, no skills about '
             'specific people, nothing loaded from other sites, the address kept only as a '
             'salted hash). The clips, skills, teach and results sections each say whether '
             "they are precomputed or live. style.css follows the system's dark or light "
             "setting and styles the timeline, the wall's tiles and the yes and no tags. The "
             'Inter 4.1 variable font is bundled in fonts/ with its SIL Open Font License, '
             'so the page fetches nothing from a CDN. The scripts come in the next two '
             "stages, and the app's tests in v91 check the page.",
        summary='Add the playground page, stylesheet and font',
        paragraph='Adds backend/playground/static/index.html (the clips, skills, teach it '
                  'yourself, results and privacy sections), style.css (dark and light themes, the '
                  'timeline and the wall) and the Inter variable font with its license. Nothing '
                  "here is tested on its own: the app's tests in stage v91 load the page and "
                  'these files and check that nothing comes from another site.',
        commands=[PYTEST, RUFF],
    ),
    89: stage(
        next="the playground's two drawings: the live timeline of a clip's scores against "
             'the threshold with its spoken alerts, and the training wall of labelled '
             'variations.',
        layer='Playground: the timeline and the wall',
        adds='Two scripts for the page, in backend/playground/static/. timeline.js draws one '
             "skill's timeline for a clip as SVG: the student's yes-probability for each "
             'scored frame as a line, broken where the student gave no score, the threshold '
             'dashed, the true ranges shaded, and each alert marked with its spoken text, on '
             'two rows to keep the labels apart. A playhead follows the video, hovering '
             "shows a frame's score, a click seeks, a sentence below reads the score against "
             'the threshold, and a banner shows what was spoken for 5 seconds after each '
             'alert. wall.js draws the training wall: generated variations flip in one after '
             "another, each tagged with Muse Spark's label; the ones Spark disagreed with "
             "fade and are marked rejected, and each image's alt text says what it was made "
             'to show. Wall.add adds tiles one at a time for a live run, and Wall.summary '
             "counts them in a sentence. The page's script uses both in the next stage.",
        summary="Add the playground's timeline and training wall",
        paragraph='Adds backend/playground/static/timeline.js (scores, threshold, true ranges and '
                  'spoken alerts over a clip, with a playhead that follows the video, hover, and '
                  'a click to seek) and wall.js (labelled variations flipping in, rejected ones '
                  'faded, and a summary sentence). There are no tests here: both run in the '
                  "browser, and the app's tests in stage v91 check that the page loads them.",
        commands=[PYTEST, RUFF],
    ),
    90: stage(
        next='the page script that ties the playground together: the clip and skill pickers, '
             "SAM's outline, a live teach run followed as it goes, the cost and the caps, "
             'and the results.',
        layer='Playground: the page script',
        adds='backend/playground/static/app.js runs the page, and every URL it asks for is '
             'relative, so the site also works under a path prefix. It lists the clips and '
             "opens one with its video and a button per skill that draws that skill's "
             'timeline, redrawn when the window is resized. For a skill it shows the '
             "read-back, the keyframe strip, SAM 3.1's box drawn over the teaching frame "
             "with how many keyframes had one, and the training wall with Spark's agreement, "
             "or the public copy's note in its place. Teach it yourself loads the status "
             '(the estimate, the caps and what the visitor and the day have left), offers '
             'the keyframes, posts the condition and then polls the run every 1200 ms, '
             'adding each variation to the wall as it arrives and ending with what the run '
             'cost. A refusal says it cost nothing; with no key or the budget used up, the '
             'page shows the precomputed examples instead. The results section shows the '
             'charts and results.md.',
        summary="Add the playground's page script",
        paragraph='Adds backend/playground/static/app.js: the clip and skill pickers, the '
                  "timeline and the wall wired to the API, SAM's box drawn on a teaching frame, "
                  'teach it yourself with its status, cost and polling, the precomputed examples '
                  'when live teaching is off, and the results. There are no tests here: it runs '
                  "in the browser, and the app's tests in stage v91 serve it.",
        commands=[PYTEST, RUFF],
    ),
    91: stage(
        next='the playground app: one page and a JSON API over the local results, a visitor '
             'cookie for the budget, live runs refused from other sites, and a public copy '
             'for the open web.',
        layer='The playground app',
        adds='backend/playground/app.py serves the page and its JSON API with Starlette: the '
             'status, clips, skills, examples, keyframes, teach and results routes, and '
             '/media/ for only the files those answers link to. A visitor gets a random id '
             'in an httponly cookie for 30 days, and the budget counts it together with '
             'their address; the default caps are $0.50 per visitor a day and $10 a day for '
             "everyone. A live run is refused when the browser's Sec-Fetch-Site header says "
             'another site sent it, and its body must be JSON of at most 4096 bytes, so '
             "another site cannot spend for a visitor. The page's Content-Security-Policy "
             'lets it load nothing from other sites, and the SVG charts get one that runs no '
             'code. create_app(public=True) is the copy for the open web: no generated '
             'image, keyframes only from te blur folders, and videos only with a matching '
             "blur record. behind_proxy takes the visitor's address from the last "
             'X-Forwarded-For entry, for a host such as Modal.',
        summary='Add the playground app and its JSON API',
        paragraph='Adds backend/playground/app.py: the page, the static files and a JSON API over '
                  'the local results, a visitor cookie for the budget, live runs refused from '
                  'other sites and limited to a small JSON body, security headers on the page and '
                  'the charts, the public copy, and behind_proxy for a hosted copy. Tests cover '
                  'the page and its files, the data APIs, a live run, both caps, bad bodies, '
                  "requests from other sites, the public copy and the proxy's address.",
        commands=[PYTEST, RUFF],
    ),
    92: stage(
        next='te playground and te playground-clip, the public playground on Modal, and the '
             'notes for the finished code, with the credits, licenses and copied Meta sample '
             'files.',
        layer='Playground commands, Modal app and notes',
        adds='This is the last stage. backend/cli_playground.py adds te playground, which '
             'serves the playground on 127.0.0.1 port 8100 by default (--public for the '
             'public copy) and refuses --forwarded-allow-ips "*", which would let a visitor '
             'reset their cap, and te playground-clip, which blurs a test clip, or copies '
             'one te blur already blurred, and checks that its frame log and labels fit the '
             "video's length before it adds the clip's folder. te playground --deploy lists "
             'what the public copy may hold, asks first (even with --yes when the total cap '
             'is over $15), clears the published folders on a private Modal Volume, copies '
             'the files and deploys backend/playground/modal_app.py: one container, the '
             'Modal Secret meta for the key, the caps baked into the image, and the budget '
             'and ledger committed to the Volume every 30 seconds while they change. '
             'CLAUDE.md, HANDOVER.md and PROGRESS.md take their final form, and '
             "CONTRIBUTIONS.md gives the credits and licenses: Meta's cookbook recipe, the "
             'copied DAT sample files, the SAM 3 protocol, Inter, Axolotl and the Meta '
             'models.',
        summary='Add the playground commands, Modal app and final notes',
        paragraph='Adds backend/cli_playground.py (te playground locally or as the public copy, '
                  'te playground --deploy to a private Modal Volume after a question, and te '
                  'playground-clip for blurred test clips) and backend/playground/modal_app.py '
                  '(the public copy on Modal in one container, its state committed to the '
                  'Volume), and registers both commands in backend/cli.py. Brings CLAUDE.md, '
                  'HANDOVER.md and PROGRESS.md to their final form and adds CONTRIBUTIONS.md with '
                  'the credits, licenses and copied Meta sample files. Tests cover the commands, '
                  "the deploy's question and file list, the clip checks, and the Modal app's "
                  'settings read from its syntax tree.',
        commands=[PYTEST, RUFF, 'python -m backend.cli playground --help', 'python -m backend.cli playground-clip --help'],
    ),
    93: stage(
        next='the notes brought up to date after the re-cut, with no code changed: the stage '
             'numbers and rules, the test count, how stages go to main, and the Modal '
             'Secrets the jobs need.',
        layer='Notes brought up to date',
        adds='This is the last stage, and it changes no code, so by the author\'s choice it is '
             'outside the 200 to 400 line rule. HANDOVER.md and PROGRESS.md give the stage '
             'numbers after the re-cut (M2 is v31 to v63, M3 v64 to v77, M4 v78 to v92), the '
             'line rule from v31, the test count, the rule that stages go to main only when '
             'the author asks, a session log entry for Nov 3 and the spend through M4, and '
             'they no longer list as still to build what is built: the SAM box reader, the '
             'phone\'s teach route, te manifests and te regenerate, the release license and '
             'the student key kept off the vLLM command line. SETUP.md lists every Modal '
             'Secret the jobs attach (huggingface before the first te train, student, meta) '
             'and the two values you make up (STUDENT_API_KEY, SERVER_TOKEN), and says how a '
             'stage goes to main. CLAUDE.md names the four Modal Secrets and adds te replay. '
             'ios/README.md and docs/glasses-check.md point at the right step of the glasses '
             'check.',
        summary='Bring the notes up to date after the re-cut',
        paragraph='Changes no code. HANDOVER.md and PROGRESS.md take the stage numbers and the '
                  '200 to 400 line rule of the re-cut, the current test count, the rule that '
                  'stages go to main only when the author asks, a session log entry for Nov 3 '
                  'and the spend through M4, and stop listing as still to build what is built. '
                  'SETUP.md and CLAUDE.md name every Modal Secret the jobs use, SETUP.md says '
                  'how a stage goes to main and CLAUDE.md adds te replay. ios/README.md and '
                  'docs/glasses-check.md fix their pointers into the glasses check.',
        commands=[PYTEST, RUFF],
    ),
})


ALL = dict(STAGES)
# TE_RENDER_STAGES=35-39 renders only those (stages already on main keep their README).
if os.environ.get("TE_RENDER_STAGES"):
    _a, _b = map(int, os.environ["TE_RENDER_STAGES"].split("-"))
    STAGES = {k: v for k, v in ALL.items() if _a <= k <= _b}
_last = max(STAGES)
NEXT_M0_DONE = (f"Stage v{_last + 1} adds {ALL[_last + 1]['next']}" if _last + 1 in ALL else
                "This is the last stage: the folder is the finished project, the same as the "
                "repository root.")
