"""Per-stage metadata for stages v39 and v40: a review of v31 to v38."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("stages_v31", Path(__file__).with_name("stages_v31.py"))
_v31 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v31)

INTRO, POLICY, PYTEST, RUFF = _v31.INTRO, _v31.POLICY, _v31.PYTEST, _v31.RUFF
NEW_COMPONENTS = {
    "backend/training/wandb_smoke.py": [
        ("backend/training/axolotl.py", "the Axolotl config for a training run: Meta's QLoRA "
         "recipe with the run's paths, W&B names and gradient accumulation for one H100"),
        ("backend/training/prepare.py", "bundles taught skills into one training set for "
         "Modal (real, synthetic or mixed, with a validation split and volume paths)"),
        ("backend/training/modal_train.py", "the Modal app that trains on one H100 with the "
         "recipe's pins, the trained-eye volume and the wandb and huggingface Secrets"),
    ],
    "backend/teach.py": [
        ("backend/watch/trigger.py", "when a skill fires: 3 of the last 4 frames over the "
         "threshold, then the cooldown; the threshold calibration on labelled clips"),
        ("backend/inference/student.py", "scores a frame with the trained model on an "
         "OpenAI-compatible endpoint, read at its yes or no answer token"),
        ("backend/watch/watcher.py", "watch mode: scores, the Muse Spark check before an "
         "alert, alert messages and the frame log"),
    ],
}
COMPONENTS = []
for _path, _text in _v31.COMPONENTS:
    COMPONENTS.append((_path, _text))
    COMPONENTS += NEW_COMPONENTS.get(_path, [])
VENV = _v31.VENV
DATE = "2026-10-31"

STAGES = {
    39: dict(
        next="the client and settings fixes from a review of v31 to v38: costs that are not "
             "numbers, and an unknown home folder.",
        layer="Fix: client and settings, third round",
        adds="A reply whose reported cost works out to NaN, infinity or below zero (Voice "
             "Transcribe's duration is taken from the reply) is now charged at the estimate "
             "and marked unpriced; before, a NaN entry switched both spend caps off for "
             "every later call. TE_DATA_DIR with ~ and a user that does not exist is a "
             "settings error with the setting's name instead of a traceback.",
        commands=[PYTEST, RUFF, "python -m backend.cli spend"],
        date=DATE,
        summary="Fix cost checks and the TE_DATA_DIR home error",
        paragraph="Falls back to the estimate when a computed cost is not finite or is "
                  "negative, so a malformed usage or duration can no longer switch the spend "
                  "caps off, and turns an unexpandable ~user in TE_DATA_DIR into a "
                  "ConfigError. Tests cover NaN, -9e9 and 1e999 durations and the unknown "
                  "user.",
    ),
    40: dict(
        next="the policy and compiler fixes from that review: two-name arrivals, titles, the "
             "glasses' light, and fewer false refusals.",
        layer="Fix: policy check and compiler, third round",
        adds="The review of the second round found named people getting through again and "
             "new false refusals. Refused again: two names ('when Sam and Jake arrive'), "
             "full names, a title with a brand name ('Mr. Dyson'), 'Sam's at the door', "
             "'the guys' and irregular plurals, and more ways to hide or switch off the "
             "glasses' capture light. Allowed again: brand readings ('Ecobee gets to 70 "
             "degrees', 'Brita's filter light comes on'), 'my lift' and 'my pitch deck', "
             "'a discreet alert', Snoopy, counting planks or Lego sets, a camcorder's "
             "recording light, the green men on a walk signal. A correction's 'not' now "
             "covers at most five words and stops at a new clause, so 'not the light, tell "
             "me when Sam gets here' is refused; the model's trigger is checked with its "
             "'when' put back; a bus shelter in a description is not a sensitive place; and "
             "a read-back no longer keeps a capital 'The' mid-sentence.",
        commands=[PYTEST, RUFF,
                  'python -m backend.cli policy "Tell me when Sam and Jake get home"',
                  'python -m backend.cli policy "Tell me when Ecobee gets to 70 degrees"'],
        date=DATE,
        summary="Fix named-person gaps and false refusals in policy",
        paragraph="Reworks the named-arrival patterns (two names, titles, contracted 's, "
                  "readings with units), narrows the capture-light, covert, sports and count "
                  "rules to what they mean, adds object phrases, bounds the negation in "
                  "corrections, checks the model's trigger with its 'when', and fixes the "
                  "read-back's capitals. Tests cover every case the review reported.",
    ),
}

NEXT_M0_DONE = ("Milestone M2, the parts that need neither the Mac nor Modal, starting with the "
                "watch trigger: 3 of the last 4 frames over the threshold, then the cooldown.")
