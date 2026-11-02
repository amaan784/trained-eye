"""CLI tests for M3's experiment commands (backend/cli_experiments.py) and te watch with
Muse Spark as the scorer. Fakes only: nothing is sent or billed."""

import json

import httpx
import pytest

from backend import cli, config
from tests.test_cli import answer, taught, use_transport, write_clip
from tests.test_teach import FakeMeta


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_API_KEY", "sk-test-cli-key")
    return tmp_path


def test_experiments_prints_the_runs_and_writes_the_plan_once(env, capsys):
    taught(env)
    assert cli.main(["experiments", "--skills", "washer_done"]) == 0
    out = capsys.readouterr().out
    assert "optimizer steps per run, fixed" in out and "exp1-mixed" in out
    assert "no smoke timing yet" in out
    assert cli.main(["experiments", "--skills", "washer_done", "--write-plan"]) == 0
    plan = json.loads((config.PROJECT_ROOT / "eval" / "plan.json").read_text())
    assert plan["evaluations"]
    assert cli.main(["experiments", "--skills", "washer_done", "--write-plan"]) == 2
    assert "--force replaces it" in capsys.readouterr().err
    assert cli.main(["experiments", "--skills", "Bad-Name"]) == 2


def test_unfiltered_and_spark_errors_report_clean_errors(env, capsys):
    assert cli.main(["unfiltered", "--skill", "washer_done"]) == 2
    assert cli.main(["spark-errors", "--skill", "washer_done"]) == 2
    err = capsys.readouterr().err
    assert err.count("te: ") == 2


def test_spark_errors_writes_its_result(env, capsys):
    folder = env / "data/skills/washer_done"
    folder.mkdir(parents=True)
    rows = [{"file": f"synthetic/img{i}.png", "source": "synthetic", "label": "yes",
             "spark_label": "yes" if i < 8 else "no", "label_error": False} for i in range(10)]
    (folder / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (folder / "human_labels.json").write_text(json.dumps(
        {"labels": {f"img{i}.png": "yes" for i in range(10)}}))
    assert cli.main(["spark-errors", "--skill", "washer_done"]) == 0
    out = capsys.readouterr().out
    assert "Spark wrong: 2 of 10 (20.0%" in out
    assert json.loads((folder / "spark_errors.json").read_text())["spark"]["errors"] == 2


def test_regress_asks_both_models_with_the_student_sampling(env, monkeypatch, capsys, tmp_path):
    from backend.eval import regression

    sent = []

    def fake(url, model, **kw):
        sent.append((url, model, kw.get("temperature"), kw.get("extra_body")))
        return real(url, model, **kw, transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "blue"},
                                                             "finish_reason": "stop"}]})))

    real = regression.Endpoint
    monkeypatch.setattr(regression, "Endpoint", fake)
    questions = tmp_path / "q.jsonl"
    questions.write_text(json.dumps({"id": "q1", "question": "What colour is the sky?",
                                     "answer": "blue", "match": "contains"}) + "\n")
    code = cli.main(["regress", "--questions", str(questions), "--base-url", "http://b/v1",
                     "--base-model", "base", "--ft-url", "http://f/v1", "--precision", "bf16"])
    assert code == 0 and "1/1 (100%)" in capsys.readouterr().out
    assert [s[2] for s in sent] == [None, None]  # never greedy, as Meta's notes say
    assert all(s[3]["chat_template_kwargs"]["reasoning_strength"] == "low" for s in sent)
    assert (config.PROJECT_ROOT / "eval/regression.md").is_file()


def test_watch_with_spark_as_the_scorer_prices_and_asks_first(env, monkeypatch, capsys):
    taught(env)
    fake = FakeMeta(answer=lambda label, real: "yes")
    use_transport(monkeypatch, fake)
    clip = write_clip(env / "clip", n=4)
    prompts = answer(monkeypatch, "n")
    assert cli.main(["watch", "--skills", "washer_done", "--clip", str(clip),
                     "--scorer", "spark"]) == 1
    assert prompts and fake.calls == {}  # nothing paid before the yes
    log = env / "replay.jsonl"
    assert cli.main(["watch", "--skills", "washer_done", "--clip", str(clip), "--scorer",
                     "spark", "--yes", "--no-cloud", "--log", str(log)]) == 0
    out = capsys.readouterr().out
    assert "Muse Spark on 4 frames x 1 skills" in out and '"frames": 4' in out
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["p"] for r in rows] == [1.0] * 4
