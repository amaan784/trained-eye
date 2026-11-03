"""CLI tests for te eval and te label (backend/cli_eval.py). Nothing is sent or billed,
and the labelling tool's server is not started."""

import pytest

from backend import cli


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_API_KEY", "sk-test-cli-key")
    return tmp_path


def test_label_starts_the_clip_or_image_tool_on_localhost(env, monkeypatch, capsys):
    import uvicorn

    ran = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: ran.append(kw))
    clip = env / "session3.mp4"
    clip.write_bytes(b"\x00" * 64)
    assert cli.main(["label", "--clip", str(clip), "--no-browser"]) == 0
    assert ran == [{"host": "127.0.0.1", "port": 8766, "log_level": "warning"}]
    assert "session3.labels.json" in capsys.readouterr().out
    assert cli.main(["label", "--skill", "nope", "--no-browser"]) == 2
    assert cli.main(["label", "--skill", "Bad-Name", "--no-browser"]) == 2


def test_eval_reports_a_missing_plan_and_scores_an_empty_one(env, capsys):
    assert cli.main(["eval", "--plan", str(env / "nope.json"), "--out", str(env / "eval")]) == 2
    (env / "plan.json").write_text('{"evaluations": []}')
    assert cli.main(["eval", "--plan", str(env / "plan.json"), "--out", str(env / "eval")]) == 0
    assert "0 of 0 evaluations scored" in capsys.readouterr().out
    assert (env / "eval/results.md").is_file()


def test_eval_wandb_entity_needs_wandb(env, capsys):
    plan = env / "plan.json"
    plan.write_text('{"evaluations": []}')
    args = ["eval", "--plan", str(plan), "--out", str(env / "out"), "--wandb-entity", "team"]
    assert cli.main(args) == 2 and "--wandb-entity needs --wandb" in capsys.readouterr().err
