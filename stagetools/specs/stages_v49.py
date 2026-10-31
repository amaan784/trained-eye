"""Per-stage metadata for stage v49: the last small fixes from the review of v31 to v38."""

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
    49: dict(
        next="the last small fixes from the review of v31 to v38, in te teach.",
        layer="Fix: te teach messages",
        adds="te teach now refuses a --request of only spaces and negative --positives or "
             "--negatives before it shows the estimate (before, the first was only noticed "
             "after the spend question, and the second printed a negative estimate). After "
             "a stop for too many failures it prints the reason first, then the hint, and "
             "exits with code 2. A --labels file that names a keyframe twice, the second "
             "time with an unusable value, now says the value is the problem. The keyframe "
             "warning says what it checks: fewer than 20.",
        commands=[PYTEST, RUFF],
        date=DATE,
        summary="Fix te teach input checks and stop messages",
        paragraph="Checks the normalised request text, adds a non-negative count type for "
                  "--positives and --negatives, prints the failure reason before the hint, "
                  "and corrects two messages. Tests cover each.",
    ),
}

NEXT_M0_DONE = ("The rest of M2 on Modal and the Mac: the two-step smoke and the first "
                "training run (asking first), export and serving, the iOS app, and thresholds "
                "from labelled validation footage.")
