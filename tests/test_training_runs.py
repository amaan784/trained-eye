"""backend/training/runs.py: the status files te train and te serve-student write for GET
/skills. Files only; nothing runs on Modal."""

import json
import os
import re

import pytest

from backend.training import runs


def test_run_names_and_times():
    assert runs.check_run("exp1-mixed") == "exp1-mixed"
    for bad in ("Exp 1", "../x", "x", "", None):
        with pytest.raises(ValueError, match="run name"):
            runs.check_run(bad)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", runs.now_iso())


def test_a_status_is_written_atomically_and_read_back(tmp_path):
    status = runs.write_status(tmp_path, "exp1-mixed", "running", ["pot_boils", "burner_left_on"],
                               started="2026-11-03T10:00:00Z")
    path = tmp_path / "train/exp1-mixed.status.json"
    assert json.loads(path.read_text()) == status and status["pid"] == os.getpid()
    assert status["finished"] is None and status["smoke"] is False
    assert not list(path.parent.glob("*.tmp"))
    assert runs.read_status(tmp_path, "exp1-mixed") == {**status, "run": "exp1-mixed"}
    with pytest.raises(ValueError, match="state"):
        runs.write_status(tmp_path, "exp1-mixed", "paused", [], started="x")
    path.write_text(json.dumps({**status, "state": "paused"}))
    assert runs.read_status(tmp_path, "exp1-mixed") is None and runs.all_status(tmp_path) == []


def test_a_process_that_is_gone_is_not_alive():
    assert runs._alive(os.getpid())
    assert not runs._alive(2**22 + 12345)  # above Linux's and macOS's highest pid
    assert runs._alive(None) and runs._alive(True)  # cannot tell: assume it runs


def test_serving_lists_the_run_s_skills_from_its_status_or_bundle(tmp_path):
    assert runs.serving(tmp_path) is None
    bundle = tmp_path / "train/exp1-mixed"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text('{"skills": {"burner_left_on": "sha"}}')
    served = runs.write_serving(tmp_path, "exp1-mixed")
    assert served["skills"] == ["burner_left_on"] and served["run"] == "exp1-mixed"
    assert runs.serving(tmp_path) == served
    runs.write_status(tmp_path, "exp1-mixed", "done", ["pot_boils", "burner_left_on"],
                      started="2026-11-03T10:00:00Z")
    path = runs.serving_path(tmp_path)
    path.write_text(json.dumps({"run": "exp1-mixed", "deployed": "x"}))  # as the contract
    assert runs.serving(tmp_path)["skills"] == ["burner_left_on", "pot_boils"]
    path.write_text(json.dumps({"run": "../etc", "deployed": "x"}))
    assert runs.serving(tmp_path) is None
    assert not runs.serving_known(tmp_path / "elsewhere")
    runs.clear_serving(tmp_path)
    runs.clear_serving(tmp_path)  # twice is fine
    assert runs.serving(tmp_path) is None and runs.serving_known(tmp_path)  # stopped, known
