"""Peel spec for stages v31 to v38: the second half of the review after M1 (F30 to F62)
and the faults found in the v23 to v30 fixes, on v30.

Every file changes in one stage only, so each step restores files from v30.
"""

BASE, FIRST, LAST = 30, 31, 38

PEEL = {
    38: {"restore": ["HANDOVER.md", "PROGRESS.md", "docs/recording-protocol.md"]},
    37: {"restore": [".github/workflows/ci.yml", ".env.example", "CLAUDE.md"]},
    36: {"restore": ["backend/teach.py", "tests/test_teach.py", "backend/cli.py",
                     "tests/test_cli.py"]},
    35: {"restore": ["backend/dataengine/labels.py", "tests/test_labels.py",
                     "backend/dataengine/roi.py", "tests/test_roi.py"]},
    34: {"restore": ["backend/dataengine/keyframes.py", "tests/test_keyframes.py",
                     "backend/dataengine/synth.py", "tests/test_synth.py"]},
    33: {"restore": ["backend/skills/spec.py", "backend/skills/compiler.py",
                     "tests/test_skills.py", "tests/test_compiler.py"]},
    32: {"restore": ["backend/config.py", "tests/test_config.py", "backend/meta/ledger.py",
                     "tests/test_ledger.py"]},
    31: {"restore": ["backend/meta/client.py", "tests/test_client.py"]},
}
