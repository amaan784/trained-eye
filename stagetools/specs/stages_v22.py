"""Per-stage metadata for stage v22: the handover and setup guide (after M1)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("stages_m1", Path(__file__).with_name("stages_m1.py"))
_m1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m1)

INTRO, POLICY, PYTEST, RUFF = _m1.INTRO, _m1.POLICY, _m1.PYTEST, _m1.RUFF

COMPONENTS = list(_m1.COMPONENTS) + [
    ("HANDOVER.md", "how to pick the project up in a new session: state, the author's rules, "
     "open items with exact commands, and the plan for M2 to M5 with the facts from the docs"),
    ("SETUP.md", "accounts and keys, setting up the Mac, setting up a cloud session, and git "
     "on the stage branch"),
]

STAGES = {
    22: dict(
        next="the handover and the setup guide for the next session.",
        layer="Handover and setup guide",
        adds="HANDOVER.md, for whoever picks the project up next in a local or cloud "
             "session: where everything is, what is done and what is not (with the exact "
             "commands for each open item), the author's rules for work, policy, version "
             "stages and git, the facts from Meta's docs that the next milestones depend "
             "on, and the plan for M2 to M5. SETUP.md covers accounts and keys, the Mac, a "
             "cloud session (network, keys, the workarounds that were needed) and git on "
             "the stage branch. CLAUDE.md and PROGRESS.md now point to both. No code "
             "changes.",
        commands=[PYTEST, RUFF, "python -m backend.cli spend"],
        summary="Add handover and setup guide",
        paragraph="Adds HANDOVER.md (state, open items with commands, rules, facts from the "
                  "docs, and the plan for M2 to M5) and SETUP.md (accounts, Mac, cloud "
                  "session, git), and points CLAUDE.md and PROGRESS.md at them. No code "
                  "changes.",
    ),
}

NEXT_M0_DONE = _m1.NEXT_M0_DONE
