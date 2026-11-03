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


ALL = dict(STAGES)
# TE_RENDER_STAGES=35-39 renders only those (stages already on main keep their README).
if os.environ.get("TE_RENDER_STAGES"):
    _a, _b = map(int, os.environ["TE_RENDER_STAGES"].split("-"))
    STAGES = {k: v for k, v in ALL.items() if _a <= k <= _b}
_last = max(STAGES)
NEXT_M0_DONE = (f"Stage v{_last + 1} adds {ALL[_last + 1]['next']}" if _last + 1 in ALL else
                "This is the last stage: the folder is the finished project, the same as the "
                "repository root.")
