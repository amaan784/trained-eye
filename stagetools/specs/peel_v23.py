"""Peel spec for stages v23 to v30: fixes from the review after M1 (F1 to F29), on v22.

Every file changes in one stage only, except backend/meta/ledger.py (v23 shows paths
relative to the current folder, v24 documents the retry cap check), so most steps just
restore the file from v22.
"""

BASE, FIRST, LAST = 22, 23, 30

LEDGER_DOC_NEW = '''        """Raise BudgetExceeded if this call could push spend past a cap.

        The client calls this before each call, and again before retrying an attempt that
        may have been billed, with the estimate grown by those attempts.
        """
'''
LEDGER_DOC_OLD = '''        """Raise BudgetExceeded if this call could push spend past a cap."""
'''

PEEL = {
    30: {"restore": ["docs/glasses-check.md", "HANDOVER.md", "PROGRESS.md", "CLAUDE.md"]},
    29: {"restore": ["backend/cli.py", "tests/test_cli.py", ".env.example"]},
    28: {"restore": ["backend/teach.py", "tests/test_teach.py"]},
    27: {"restore": ["backend/dataengine/synth.py", "backend/dataengine/labels.py",
                     "backend/dataengine/roi.py", "tests/test_labels.py"]},
    26: {"restore": ["backend/dataengine/keyframes.py", "tests/test_keyframes.py"]},
    25: {"restore": ["backend/skills/spec.py", "backend/skills/compiler.py",
                     "tests/test_skills.py", "tests/test_compiler.py"]},
    24: {"restore": ["backend/meta/client.py", "tests/test_client.py"],
         "edit": {"backend/meta/ledger.py": [(LEDGER_DOC_NEW, LEDGER_DOC_OLD)]}},
    23: {"restore": ["backend/config.py", "backend/meta/ledger.py", "tests/conftest.py",
                     "tests/test_config.py"]},
}
