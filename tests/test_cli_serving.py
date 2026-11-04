"""CLI tests for te export and te serve-student (backend/cli_serving.py): the questions,
the deploys and stops, and the GPU time in the Modal ledger, also for te train. Modal is
faked: nothing runs or is billed."""

import json
import subprocess
from pathlib import Path

import pytest

from backend import cli, cli_serving
from tests.test_cli import TRAIN, answer, fake_modal, taught
from tests.test_cli_train import SMOKE_RESULT


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_API_KEY", "sk-test-cli-key")
    return tmp_path


def test_export_needs_a_bundle_asks_and_reports_the_check(env, monkeypatch, capsys):
    import ast

    taught(env)
    assert cli.main(["export", "--run", "exp1-mixed"]) == 2  # no bundle yet
    (env / "data/train/exp1-mixed").mkdir(parents=True)
    (env / "data/train/exp1-mixed/manifest.json").write_text("{}")
    runs = fake_modal(monkeypatch, {"merged_dir": "/vol/runs/exp1-mixed/merged", "examples": 50,
                                    "agreement": 1.0, "all_agree": True, "bin_files": 0,
                                    "tensor_names_match": True, "disagreements": []})
    prompts = answer(monkeypatch, "n")
    assert cli.main(["export", "--run", "exp1-mixed"]) == 1 and runs == [] and prompts
    assert cli.main(["export", "--run", "exp1-mixed", "--yes"]) == 0
    [command] = runs
    assert command[:3] == ["modal", "run", "backend/training/export.py"]
    assert "same answer on 100% of 50 examples" in capsys.readouterr().out
    tree = ast.parse((cli.PROJECT_ROOT / "backend/training/export.py").read_text())
    timeout = next(eval(compile(ast.Expression(n.value), "export.py", "eval"))  # 3 * 3600
                   for n in tree.body if isinstance(n, ast.Assign)
                   and getattr(n.targets[0], "id", "") == "TIMEOUT_S")
    assert timeout == cli_serving.EXPORT_MAX_HOURS * 3600  # the worst case printed is the limit


def test_serve_student_deploys_with_the_run_and_stops(env, monkeypatch, capsys):
    calls = []

    def run(command, cwd=None, check=False, env=None, **kw):
        calls.append((command, (env or {}).get("TE_SERVE_RUN")))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    assert cli.main(["serve-student"]) == 2
    assert cli.main(["serve-student", "--run", "exp1-mixed", "--yes"]) == 0
    assert cli.main(["serve-student", "--stop"]) == 0
    assert calls == [(["modal", "deploy", "backend/inference/modal_serve.py"], "exp1-mixed"),
                     (["modal", "app", "stop", "trained-eye-student"], None)]
    assert "TE_STUDENT_URL" in capsys.readouterr().out


def test_serve_student_records_the_served_run_and_stop_clears_it(env, monkeypatch, capsys):
    from backend.training import runs

    codes = []
    monkeypatch.setattr(cli.subprocess, "run",
                        lambda command, **kw: subprocess.CompletedProcess(command, codes.pop(0)))
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    runs.write_status(env / "data", "exp1-mixed", "done", ["burner_left_on"],
                      started="2026-11-03T10:00:00Z", finished="2026-11-03T11:00:00Z")
    codes[:] = [1]
    assert cli.main(["serve-student", "--run", "exp1-mixed", "--yes"]) == 1
    assert not (env / "data/train/serving.json").exists()  # the deploy failed
    codes[:] = [0]
    assert cli.main(["serve-student", "--run", "exp1-mixed", "--yes"]) == 0
    serving = json.loads((env / "data/train/serving.json").read_text())
    assert serving["run"] == "exp1-mixed" and serving["skills"] == ["burner_left_on"]
    assert serving["deployed"].endswith("Z")
    assert "run exp1-mixed serves burner_left_on" in capsys.readouterr().out
    codes[:] = [1]
    assert cli.main(["serve-student", "--stop"]) == 1  # not stopped: still served
    assert (env / "data/train/serving.json").exists()
    codes[:] = [0]
    assert cli.main(["serve-student", "--stop"]) == 0
    from backend.training import runs

    assert runs.serving(env / "data") is None and runs.serving_known(env / "data")  # stopped


def test_train_and_export_ask_when_the_modal_credit_would_run_out(env, monkeypatch, capsys):
    from backend.training import gpu_ledger

    taught(env)
    (env / "data/train").mkdir(parents=True)
    (env / "data/train/smoke-timing.json").write_text(json.dumps(SMOKE_RESULT))
    gpu_ledger.record(env / "data", "train", "earlier", 149.5 / 3.95 * 3600)
    runs = fake_modal(monkeypatch, {"seconds": 990.0, "ok": True})
    prompts = answer(monkeypatch, "n")
    assert cli.main([*TRAIN, "--yes"]) == 1 and runs == [] and len(prompts) == 1
    assert "plus this $1.01 would pass the $150 of Modal credit" in capsys.readouterr().out
    prompts = answer(monkeypatch, "n")
    (env / "data/train/exp1-mixed/manifest.json").write_text("{}")
    assert cli.main(["export", "--run", "exp1-mixed", "--yes"]) == 1 and runs == []
    assert "plus this $11.85 would pass" in capsys.readouterr().out and len(prompts) == 1


def test_export_records_the_timed_remote_call_also_when_it_fails(env, monkeypatch, capsys):
    from backend.training import gpu_ledger

    (env / "data/train/exp1-mixed").mkdir(parents=True)
    (env / "data/train/exp1-mixed/manifest.json").write_text("{}")
    fake_modal(monkeypatch, {"merged_dir": "/vol/runs/exp1-mixed/merged", "examples": 50,
                             "agreement": 1.0, "all_agree": True, "bin_files": 0,
                             "tensor_names_match": True, "disagreements": [], "seconds": 1800.0})
    assert cli.main(["export", "--run", "exp1-mixed", "--yes"]) == 0
    [row] = gpu_ledger.read(env / "data")
    assert (row["kind"], row["name"], row["seconds"], row["usd"]) == (
        "export", "exp1-mixed", 1800.0, 1.975)
    fake_modal(monkeypatch, {"run": "exp1-mixed", "ok": False, "seconds": 60.0}, code=1)
    assert cli.main(["export", "--run", "exp1-mixed", "--yes"]) == 1
    out, err = capsys.readouterr()
    assert "the last export took 30 min, about $1.98" in out and "the export failed" in err
    assert [r["seconds"] for r in gpu_ledger.read(env / "data")] == [1800.0, 60.0]


def test_serve_student_base_and_adapters_get_their_own_app_and_the_time_is_recorded(
        env, monkeypatch, capsys):
    from backend.training import gpu_ledger

    calls = []

    def run(command, cwd=None, check=False, env=None, **kw):
        calls.append((command, {k: (env or {}).get(k) for k in cli_serving.SERVE_ENVS}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    assert cli.main(["serve-student", "--base", "--yes"]) == 0
    assert calls[-1] == (["modal", "deploy", "backend/inference/modal_serve.py"],
                         {"TE_SERVE_RUN": "", "TE_SERVE_BASE": "1", "TE_SERVE_LORAS": ""})
    out = capsys.readouterr().out
    assert "deploys trained-eye-base serving the base model meta-models/Muse-Glimmer-30B at " \
           "revision a4e59da52a7b as muse-glimmer" in out
    assert "TE_STUDENT_MODEL=muse-glimmer" in out and "--base-model muse-glimmer" in out
    assert "te serve-student --stop --base stops it now" in out
    assert not (env / "data/train/serving.json").exists()  # the base serves no taught skill
    monkeypatch.setenv("TE_SERVE_BASE", "1")  # a stray shell setting never reaches a deploy
    assert cli.main(["serve-student", "--run", "exp1-mixed", "--yes"]) == 0
    assert calls[-1][1] == {"TE_SERVE_RUN": "exp1-mixed", "TE_SERVE_BASE": "",
                            "TE_SERVE_LORAS": ""}
    (env / "data/train/exp1-mixed.result.json").write_text('{"ok": true, "smoke": false}')
    assert cli.main(["serve-student", "--runs", "exp1-real,exp1-mixed", "--yes"]) == 0
    assert calls[-1][1]["TE_SERVE_LORAS"] == "exp1-real,exp1-mixed"
    out = capsys.readouterr().out
    assert "LoRA serving is not documented for Muse Glimmer" in out
    assert "no finished full run of exp1-real" in out and "of exp1-mixed" not in out
    assert "TE_STUDENT_MODEL=exp1-real with te watch --run exp1-real" in out
    for bad in (["--runs", "exp1,exp1"], ["--runs", "Bad Run"], ["--runs", ","],
                ["--run", "exp1-mixed", "--base"]):
        assert cli.main(["serve-student", *bad, "--yes"]) == 2, bad
    assert cli.main(["serve-student", "--stop", "--base"]) == 0
    assert calls[-1][0] == ["modal", "app", "stop", "trained-eye-base"]
    assert "trained-eye-base was deployed for" in capsys.readouterr().out
    rows = gpu_ledger.read(env / "data")
    assert [(r["kind"], r["name"]) for r in rows] == [
        ("deploy", "trained-eye-base"), ("deploy", "trained-eye-student"),
        ("deploy", "trained-eye-base"), ("serve", "trained-eye-base")]
    assert rows[-1]["upper_bound"] and set(gpu_ledger.open_deploys(env / "data")) == {
        "trained-eye-student"}


def test_an_interrupted_train_or_export_still_records_its_gpu_time(env, monkeypatch):
    from backend.training import gpu_ledger

    taught(env)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)

    def interrupted(command, **kw):
        if command[0] != "modal":
            return subprocess.CompletedProcess(command, 0, stdout="abc1234\n", stderr="")
        if "backend/training/modal_train.py" in command:  # as modal_train.main writes on ^C
            path = command[command.index("--result") + 1]
            Path(path).write_text(json.dumps({"run": "exp1-mixed", "ok": False, "smoke": True,
                                              "seconds_upper_bound": True, "seconds": 1800.0}))
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.subprocess, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        cli.main([*TRAIN, "--smoke", "--yes"])
    [row] = gpu_ledger.read(env / "data")
    assert (row["kind"], row["seconds"]) == ("train", 1800.0)
    (env / "data/train/exp1-mixed/manifest.json").write_text("{}")  # a bundle to export
    with pytest.raises(KeyboardInterrupt):
        cli.main(["export", "--run", "exp1-mixed", "--yes"])
    rows = gpu_ledger.read(env / "data")
    assert [r["kind"] for r in rows] == ["train", "export"] and rows[1]["upper_bound"]
