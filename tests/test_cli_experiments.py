"""CLI tests for M3's experiment commands (backend/cli_experiments.py). Fakes only:
nothing is sent or billed."""

import json

import httpx
import pytest

from backend import cli, config
from tests.test_cli import taught


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


def test_unfiltered_passes_te_system_in_user_on(env, monkeypatch, capsys):
    from backend.eval import spark_errors

    seen = []
    monkeypatch.setattr(spark_errors, "write_unfiltered_dataset",
                        lambda skill_dir, **kw: seen.append(kw) or {"examples": 1, "counts": {}})
    monkeypatch.setenv("TE_SYSTEM_IN_USER", "1")
    assert cli.main(["unfiltered", "--skill", "washer_done"]) == 0
    assert seen == [{"train_on_model_outputs": True, "system_in_user": True}]


def test_experiments_prices_the_credit_with_the_ledger_final_run_exports_and_serving(
        env, capsys):
    from backend.training import gpu_ledger

    taught(env)
    data = env / "data"
    (data / "train").mkdir(parents=True)
    (data / "train/smoke-timing.json").write_text(json.dumps(
        {"seconds": 400.0, "train_runtime": 100.0, "train_samples_per_second": 0.5}))
    gpu_ledger.record(data, "export", "exp1-real", 1800.0)  # half an hour: $1.98
    gpu_ledger.record(data, "train", "exp1-real", 3600.0)  # $3.95
    (data / "train/exp1-real.result.json").write_text(json.dumps({"ok": True}))
    (data / "train/exp1-real.export.json").write_text(json.dumps({"merged_dir": "/vol/m"}))
    assert cli.main(["experiments", "--skills", "washer_done", "--endpoint-hours", "2"]) == 0
    out = capsys.readouterr().out
    assert "Modal spend so far: $5.93 (te spend)" in out
    assert "exp1-real           done" in out and "final" in out
    assert "at 0.5 h (the longest export timed so far)" in out
    assert "7 models at 2 h each" in out  # 5 bundles, the final run and the base endpoint
    assert "trained already, in the spend: exp1-real" in out
    with pytest.raises(SystemExit):
        cli.main(["experiments", "--skills", "washer_done", "--endpoint-hours", "0"])
    assert cli.main(["experiments", "--skills", "washer_done", "--write-plan"]) == 0
    assert ("te replay --plan eval/plan.json --run <run> fills its logs, for runs exp1-real, "
            "exp1-synthetic, exp1-mixed, exp2-unfiltered, exp3-25, exp4-base, exp4-spark") \
        in capsys.readouterr().out
