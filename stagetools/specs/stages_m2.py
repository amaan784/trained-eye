"""Per-stage metadata for stages v41 to v48: the parts of M2 that need neither the Mac
nor Modal."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("stages_v39", Path(__file__).with_name("stages_v39.py"))
_v39 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v39)

INTRO, POLICY, PYTEST, RUFF = _v39.INTRO, _v39.POLICY, _v39.PYTEST, _v39.RUFF
COMPONENTS = _v39.COMPONENTS
VENV = _v39.VENV
DATE = _v39.DATE

STAGES = {
    41: dict(
        next="the watch trigger, 3 of the last 4 frames over the threshold, then "
             "the cooldown, and the threshold calibration.",
        layer="Watch trigger and calibration",
        adds="backend/watch/trigger.py decides when a skill fires: at least 3 of its last 4 "
             "frames (about one a second) at or over its threshold, then quiet for the "
             "skill's cooldown; a pause in the stream starts the window again, and a "
             "snooze serves the cloud check. calibrate() picks the lowest threshold with at "
             "most one false alarm per hour of labelled clips and reports the events caught "
             "and the median delay; labelled_frames() reads a watch frame log.",
        commands=[PYTEST, RUFF],
        date=DATE,
        summary="Add the watch trigger and threshold calibration",
        paragraph="Adds the 3-of-4 trigger with cooldown, gap reset and snooze, the alert "
                  "simulation and evaluation over labelled clips, the calibration that picks "
                  "the lowest threshold with at most one false alarm per hour, and the "
                  "reader for watch frame logs, with tests.",
    ),
    42: dict(
        next="the student scorer: the trained model behind an OpenAI-compatible endpoint, "
             "read at its answer token.",
        layer="Student scorer",
        adds="backend/inference/student.py scores one frame for one skill on an "
             "OpenAI-compatible chat endpoint (vLLM on Modal or llama.cpp's llama-server on "
             "the Mac). The frame is sent exactly as the training examples were built: the "
             "system prompt, the padded 512x512 image and the condition text. Glimmer "
             "reasons first, so the probability is read at the last generated yes or no, "
             "from its top logprobs, renormalised over the two; a reply without one is an "
             "error, never a guess.",
        commands=[PYTEST, RUFF],
        date=DATE,
        summary="Add the student scorer for an OpenAI-style endpoint",
        paragraph="Adds OpenAIStudent, which sends a frame as the dataset builds it and reads "
                  "the yes/no probability from the answer token's logprobs, with tests for "
                  "the request, the reasoning before the answer and endpoint failures.",
    ),
    43: dict(
        next="the watcher: frames in, alerts out, with the Muse Spark check and a frame log.",
        layer="Watcher",
        adds="backend/watch/watcher.py runs watch mode for a set of taught skills: the "
             "student scores each frame, the skill's trigger decides, and Muse Spark checks "
             "the frame before an alert goes out if the skill asks for it. A Spark no "
             "snoozes the skill for a minute; a failed check still alerts, marked "
             "unconfirmed. Every scored frame is logged as one JSON line (time, "
             "probability, decision, the cloud answer); frames are never written. "
             "load_skill() reads a taught skill and its calibrated threshold.",
        commands=[PYTEST, RUFF],
        date=DATE,
        summary="Add the watcher with the Spark check and frame log",
        paragraph="Adds Watcher, the alert message, the frame log and load_skill, with tests "
                  "for a confirmed alert, a cloud no, a failed check, skills without the "
                  "check, student errors and two skills at once.",
    ),
    44: dict(
        next="watch mode on the WebSocket: the phone names the skills, and the server sends "
             "scores and alerts.",
        layer="Watch mode on the server",
        adds="A hello with \"watch\": [skill names] now starts watch mode for the session: "
             "the welcome lists the skills, each ack carries their scores, and an alert "
             "message follows when one fires. Without a watch request the protocol is "
             "unchanged. The server builds the watcher from the settings: the student "
             "endpoint (TE_STUDENT_URL and TE_STUDENT_MODEL, new settings shown by python -m "
             "backend.config), Muse Spark for the check and a frame log under data/watch/.",
        commands=[PYTEST, RUFF, "python -m backend.config", "python -m backend.cli ws-selftest"],
        date=DATE,
        summary="Add watch mode to the WebSocket server",
        paragraph="Lets a session ask to watch taught skills, scores each frame in a worker "
                  "thread, sends scores in the acks and alert messages, and reports a watch "
                  "request that cannot be met without ending the session. Adds the "
                  "student endpoint settings. Tests cover the protocol, the errors and the "
                  "default watcher.",
    ),
    45: dict(
        next="the training data: taught skills bundled for Modal, and the Axolotl config from "
             "Meta's recipe.",
        layer="Training bundle and config",
        adds="backend/training/prepare.py bundles taught skills into one training set (one "
             "adapter learns every skill, since the condition is in the prompt): real, "
             "synthetic or mixed examples, an optional limit per skill for the image-count "
             "curve, a deterministic validation split, and image paths pointing where the "
             "bundle will be on the Modal volume. backend/training/axolotl.py writes the "
             "Axolotl config: Meta's QLoRA recipe value for value, with this run's paths, "
             "W&B names and gradient accumulation 32 to keep the global batch of 32 on one "
             "H100, and the recipe's two-step smoke.",
        commands=[PYTEST, RUFF],
        date=DATE,
        summary="Add the training bundle and the Axolotl config",
        paragraph="Adds the bundle writer with variants, limits, validation split and volume "
                  "paths, and the config builder that copies Meta's QLoRA recipe and changes "
                  "only paths, names and gradient accumulation. Tests compare the config "
                  "with the recipe and cover every bundle option.",
    ),
    46: dict(
        next="the Modal training app: one H100, the recipe's pins, the volume and both "
             "Modal Secrets.",
        layer="Modal training app",
        adds="backend/training/modal_train.py trains one run on Modal: an image built as the "
             "recipe's setup says (Python 3.12, CUDA 13.0, torch 2.12.1, torchvision 0.27.1, "
             "Axolotl 0.19.0, flash-attn 2.8.3, Cut Cross Entropy at its pinned commit), one "
             "H100, the trained-eye volume for bundles, runs and the Hugging Face cache, and "
             "the wandb and huggingface Modal Secrets. It uploads the bundle, writes the "
             "config, runs axolotl train, and returns the time and the trainer's samples per "
             "second. Modal is not installed here, so the tests read the script.",
        commands=[PYTEST, RUFF],
        date=DATE,
        summary="Add the Modal training app with the recipe pins",
        paragraph="Adds the Modal app, image, volume and training function, the bundle "
                  "upload and the local entrypoint, with static tests for the pins, the GPU, "
                  "the secrets and the bundle check.",
    ),
    47: dict(
        next="the commands: te train on Modal, te watch to replay a clip, te calibrate for "
             "the threshold.",
        layer="te train, te watch, te calibrate",
        adds="te train bundles the named skills and runs the two-step smoke on Modal (--smoke), "
             "then saves its timing; a real run is priced from that timing at the H100 price "
             "in the settings (not checked yet) and asks first, always when it is over $15. "
             "te watch replays a video or frame folder through watch mode at one frame a "
             "second and prints the alerts. te calibrate takes frame logs from te watch and "
             "the time ranges where the condition is true, and saves the skill's threshold, "
             "which watch mode then uses. CLAUDE.md lists the three commands.",
        commands=[PYTEST, RUFF, "python -m backend.cli train --help"],
        date=DATE,
        summary="Add te train, te watch and te calibrate",
        paragraph="Adds the three commands, the H100 price setting and their CLAUDE.md lines. "
                  "Tests cover the smoke, "
                  "a priced run, the over-$15 question, input errors, a failed run, a replay "
                  "and a calibration that watch mode then uses.",
    ),
    48: dict(
        next="the notes for these stages: the review round, what M2 has so far, and the "
             "steps that need the Mac or Modal.",
        layer="Docs and notes",
        adds="HANDOVER.md and PROGRESS.md record the third review round and the parts of M2 "
             "built so far, list the next steps that need Modal or the Mac (the smoke, the "
             "real run, export, serving, the iOS app, thresholds from real footage), record "
             "the decisions (scoring at the answer token, the alert rules, the config as "
             "JSON), and correct the keyframe count to 20 to 24 with its cost figures. "
             ".env.example gains the student endpoint settings, and the recording protocol "
             "asks for movement during the teaching look and for both states when training "
             "on model outputs is off.",
        commands=[PYTEST, RUFF, "python -m backend.cli spend"],
        date=DATE,
        summary="Update the notes for the review and M2 so far",
        paragraph="Records the review fixes and the offline half of M2 in HANDOVER.md and "
                  "PROGRESS.md, with the next steps on Modal and the Mac and corrected cost "
                  "figures, adds the student settings to .env.example, and extends the "
                  "teaching-clip section of docs/recording-protocol.md.",
    ),
}

NEXT_M0_DONE = ("The last small fixes from the review of v31 to v38, in te teach: a blank "
                "request, negative counts, the failure message and a label message.")
