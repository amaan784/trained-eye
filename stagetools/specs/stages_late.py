"""Shared text for stages v50 to v105 (cut under the 150 to 300 line rule)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("stages_v39", Path(__file__).with_name("stages_v39.py"))
_v39 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v39)

INTRO, POLICY, PYTEST, RUFF = _v39.INTRO, _v39.POLICY, _v39.PYTEST, _v39.RUFF
VENV = _v39.VENV
DATE = "2026-11-02"

_AFTER = {
    "backend/server.py": [
        ("backend/teach_session.py", "a phone teach session: the spend question before any "
         "paid call, the read-back, progress lines and timeouts"),
        ("backend/teach_upload.py", "the phone's teach upload: a multipart parser, the input "
         "checks, the saved capture and the estimate"),
        ("backend/teach_api.py", "the teach routes for the phone: /teach/start, "
         "/teach/status and /teach/answer, behind SERVER_TOKEN"),
    ],
    "backend/training/modal_train.py": [
        ("backend/training/export.py", "the Modal app that merges a trained adapter and "
         "checks the merged model's answers"),
        ("backend/inference/modal_serve.py", "the Modal app that serves the merged model with "
         "vLLM behind STUDENT_API_KEY"),
    ],
    "backend/watch/watcher.py": [
        ("backend/labeltool/clips.py", "the labelling tool for clips: where the condition is "
         "true, saved as time ranges"),
        ("backend/labeltool/images.py", "the labelling tool for generated images, for "
         "Experiment 2"),
        ("backend/eval/metrics.py", "evaluation metrics from watch frame logs and labels"),
        ("backend/eval/report.py", "the evaluation runner: the plan, results.json, results.md "
         "and the SVG charts"),
        ("backend/eval/plan.py", "the experiment plan: every training run with fixed steps "
         "and its price"),
        ("backend/regenerate.py", "the prompt manifests and regenerating images from them"),
        ("backend/release/model_card.py", "the Hugging Face model card and release folder"),
        ("backend/playground/app.py", "the web playground"),
    ],
}
COMPONENTS = []
for _path, _text in _v39.COMPONENTS:
    COMPONENTS.append((_path, _text))
    COMPONENTS += _AFTER.get(_path, [])
COMPONENTS += [
    ("ios/README.md", "how to build the iPhone app in Xcode (not compiled yet)"),
    ("CONTRIBUTIONS.md", "credits, licenses and the copied Meta sample files"),
]


def stage(next, layer, adds, summary, paragraph, commands=None):
    return dict(next=next, layer=layer, adds=adds, commands=commands or [PYTEST, RUFF],
                date=DATE, summary=summary, paragraph=paragraph)
