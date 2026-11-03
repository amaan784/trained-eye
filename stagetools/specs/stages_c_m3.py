"""Stages v64 to v77 (M3) of stages_c.py, for versions_md.py."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("stages_c", Path(__file__).with_name("stages_c.py"))
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)
STAGES = {k: v for k, v in _c.ALL.items() if 64 <= k <= 77}
