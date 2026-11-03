"""CLI tests for te train (backend/cli_train.py): the smoke and the price from its timing,
the questions, the run's status, the W&B tags and artifacts, --detach and --collect, and the
preflight. Modal is faked: nothing runs or is billed."""

import json
import subprocess
from pathlib import Path

import pytest

from backend import cli
from tests.test_cli import TRAIN, answer, fake_modal, taught


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_API_KEY", "sk-test-cli-key")
    return tmp_path


def test_train_smoke_asks_runs_modal_and_saves_the_timing(env, monkeypatch, capsys):
    taught(env)
    runs = fake_modal(monkeypatch, {"seconds": 900.0, "train_runtime": 60.0,
                                    "train_samples_per_second": 0.5, "ok": True})
    prompts = answer(monkeypatch, "y")
    assert cli.main([*TRAIN, "--smoke"]) == 0
    assert prompts == ["Run the smoke on Modal? [y/N] "]
    [command] = runs
    assert command[:3] == ["modal", "run", "backend/training/modal_train.py"]
    assert command[-1] == "--smoke" and "--bundle" in command
    tags = command[command.index("--tags") + 1].split(",")
    assert {"exp:exp1", "skills:washer_done", "mix:mixed", "seed:42",
            "commit:abc1234"} <= set(tags) and any(t.startswith("examples:") for t in tags)
    timing = json.loads((env / "data/train/smoke-timing.json").read_text())
    assert timing["train_samples_per_second"] == 0.5
    manifest = json.loads((env / "data/train/exp1-mixed/manifest.json").read_text())
    assert manifest["image_root"] == "/vol/bundles/exp1-mixed"
    out = capsys.readouterr().out
    assert "bundle exp1-mixed:" in out and "smoke timing saved" in out


@pytest.mark.parametrize("args, message", [
    (["--skills", "washer_done", "--run", "Exp 1"], "run name"),
    (["--skills", "nope", "--run", "exp1"], "no dataset yet"),
    (["--skills", ",", "--run", "exp1"], "at least one"),
])
def test_train_input_errors(env, monkeypatch, capsys, args, message):
    taught(env)
    fake_modal(monkeypatch)
    assert cli.main(["train", *args]) == 2 and message in capsys.readouterr().err


def test_a_failed_training_run_says_where_the_log_is(env, monkeypatch, capsys):
    taught(env)
    fake_modal(monkeypatch, None, code=1)
    assert cli.main([*TRAIN, "--smoke", "--yes"]) == 1
    assert "runs/exp1-mixed/train.log" in capsys.readouterr().err


def status_of(env, run="exp1-mixed"):
    return json.loads((env / f"data/train/{run}.status.json").read_text())


def test_train_marks_the_run_running_then_done_or_failed(env, monkeypatch):
    taught(env)
    assert cli.main([*TRAIN, "--smoke", "--estimate-only"]) == 0
    assert not (env / "data/train/exp1-mixed.status.json").exists()  # nothing ran
    during = []
    fake_modal(monkeypatch, {"seconds": 900.0, "ok": True})
    modal = cli.subprocess.run

    def watching(command, **kw):
        if command[0] == "modal":
            during.append(status_of(env))
        return modal(command, **kw)

    monkeypatch.setattr(cli.subprocess, "run", watching)
    assert cli.main([*TRAIN, "--smoke", "--yes"]) == 0
    [running] = during
    assert (running["state"], running["skills"], running["finished"], running["smoke"]) == (
        "running", ["washer_done"], None, True)
    done = status_of(env)
    assert done["state"] == "done" and done["started"] == running["started"]
    assert done["finished"] >= done["started"]

    fake_modal(monkeypatch, None, code=1)
    assert cli.main([*TRAIN, "--smoke", "--yes"]) == 1
    assert status_of(env)["state"] == "failed"

    def interrupted(command, **kw):
        if command[0] == "modal":
            raise KeyboardInterrupt
        return subprocess.CompletedProcess(command, 0, stdout="abc1234\n", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        cli.main([*TRAIN, "--smoke", "--yes"])
    assert status_of(env)["state"] == "failed" and status_of(env)["finished"]


SMOKE_RESULT = {"seconds": 900.0, "train_runtime": 60.0, "train_samples_per_second": 0.5,
                "ok": True}


def test_train_tags_the_data_and_base_passes_the_wandb_team_and_records_the_spend(
        env, monkeypatch):
    from backend.config import STUDENT_BASE_REVISION
    from backend.training import gpu_ledger
    from backend.training.axolotl import data_hash

    taught(env)
    runs = fake_modal(monkeypatch, SMOKE_RESULT)
    assert cli.main([*TRAIN, "--smoke", "--yes", "--limit", "20", "--wandb-entity", "team"]) == 0
    [command] = runs
    tags = command[command.index("--tags") + 1].split(",")
    manifest = json.loads((env / "data/train/exp1-mixed/manifest.json").read_text())
    assert {f"data:{data_hash(manifest)[:12]}", f"base:{STUDENT_BASE_REVISION[:12]}",
            "dataset:dataset", "images:20"} <= set(tags)
    assert command[command.index("--wandb-entity") + 1] == "team" and command[-1] == "--smoke"
    assert not {"--wandb-artifacts", "--detach", "--detached"} & set(command)
    [row] = gpu_ledger.read(env / "data")
    assert (row["kind"], row["name"], row["seconds"], row["usd"]) == (
        "train", "exp1-mixed", 900.0, 0.9875)


def test_wandb_artifacts_ask_even_with_yes_and_say_what_becomes_visible(env, monkeypatch,
                                                                        capsys):
    from backend.training.axolotl import data_hash

    taught(env)
    runs = fake_modal(monkeypatch, SMOKE_RESULT)
    prompts = answer(monkeypatch, "n")
    assert cli.main([*TRAIN, "--smoke", "--yes", "--wandb-artifacts"]) == 1
    assert runs == [] and prompts == ["Upload them to W&B? (asked even with --yes) [y/N] "]
    out, err = capsys.readouterr()
    manifest = json.loads((env / "data/train/exp1-mixed/manifest.json").read_text())
    assert f"dataset artifact exp1-mixed-{data_hash(manifest)[:12]} (40 images" in out
    assert "30 are Muse Image outputs" in out and "Faces in these images are not blurred" in out
    assert "nothing uploaded or trained" in err
    prompts = answer(monkeypatch, "y")
    assert cli.main([*TRAIN, "--smoke", "--yes", "--wandb-artifacts"]) == 0
    assert len(prompts) == 1  # --yes skipped the cost question, never this one
    [command] = runs
    assert "--wandb-artifacts" in command
    copied = env / "data/train/exp1-mixed/provenance/washer_done-teach.manifest.json"
    assert json.loads(copied.read_text())["examples_index"]  # the provenance goes along


def test_train_detach_outlives_the_client_and_collect_fetches_the_result(env, monkeypatch,
                                                                       capsys):
    from backend.training import gpu_ledger
    from backend.training import runs as run_files

    taught(env)
    calls = []

    def lost(command, cwd=None, check=False, **kw):
        if command[0] != "modal":
            return subprocess.CompletedProcess(command, 0, stdout="abc1234\n", stderr="")
        calls.append(command)
        return subprocess.CompletedProcess(command, 1)  # the connection dropped

    monkeypatch.setattr(cli.subprocess, "run", lost)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    assert cli.main([*TRAIN, "--smoke", "--yes", "--detach"]) == 1
    [command] = calls
    assert command[:4] == ["modal", "run", "--detach", "backend/training/modal_train.py"]
    assert "--detached" in command
    status = status_of(env)
    assert (status["state"], status["detached"], status["pid"]) == ("running", True, None)
    assert run_files.read_status(env / "data", "exp1-mixed")["state"] == "running"
    assert "te train --collect exp1-mixed" in capsys.readouterr().err

    remote = {"run": "exp1-mixed", "smoke": True, "started": status["started"],
              "finished": status["started"], **SMOKE_RESULT}
    gets = []

    def volume_get(command, cwd=None, check=False, **kw):
        gets.append(command)
        Path(command[-1], "result.json").write_text(json.dumps(remote))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", volume_get)
    assert cli.main(["train", "--collect", "exp1-mixed"]) == 0
    assert gets[0][:5] == ["modal", "volume", "get", "trained-eye", "/runs/exp1-mixed/result.json"]
    assert json.loads((env / "data/train/exp1-mixed.result.json").read_text()) == remote
    timing = json.loads((env / "data/train/smoke-timing.json").read_text())
    assert timing["train_samples_per_second"] == 0.5
    done = status_of(env)
    assert (done["state"], done["finished"], done["skills"]) == (
        "done", remote["finished"], ["washer_done"])
    out = capsys.readouterr().out
    assert "collected:" in out and "smoke timing saved" in out and "done: 900 s" in out
    assert not list((env / "data/train").glob(".collect-*"))
    assert cli.main(["train", "--collect", "exp1-mixed"]) == 0  # again: counted once
    assert [r["kind"] for r in gpu_ledger.read(env / "data")] == ["train"]


def test_collect_says_when_there_is_no_result_or_only_an_earlier_one(env, monkeypatch, capsys):
    from backend.training import runs as run_files

    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(cli.subprocess, "run",
                        lambda command, **kw: subprocess.CompletedProcess(command, 1))
    assert cli.main(["train", "--collect", "exp1-mixed"]) == 1
    assert "no result for exp1-mixed on the Modal volume yet" in capsys.readouterr().err
    assert cli.main(["train", "--collect", "Bad Run"]) == 2
    assert cli.main(["train"]) == 2 and "--skills and --run" in capsys.readouterr().err
    run_files.write_status(env / "data", "exp1-mixed", "running", ["washer_done"],
                           started="2026-11-03T15:00:00Z", detached=True)

    def stale(command, **kw):
        Path(command[-1], "result.json").write_text(json.dumps(
            {"run": "exp1-mixed", "ok": True, "seconds": 5.0, "finished": "2026-11-03T14:00:00Z"}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", stale)
    assert cli.main(["train", "--collect", "exp1-mixed"]) == 1
    assert "from an earlier run of exp1-mixed" in capsys.readouterr().err
    assert not (env / "data/train/exp1-mixed.result.json").exists()
    assert status_of(env)["state"] == "running"


def test_train_preflight_asks_runs_the_cpu_check_and_prints_the_verdict(env, monkeypatch,
                                                                       capsys):
    from backend.dataengine.dataset import SYSTEM_PROMPT
    from backend.training import gpu_ledger, preflight

    taught(env)
    render = {"answer": "yes", "tokens": 300, "text": f"<|start|>system<|message|>"
              f"{SYSTEM_PROMPT}<|eot|>...", "target_text": "yes<|eot|>",
              "target_ids": [5, 200008], "target_tokens": ["yes", "<|eot|>"]}
    rendered = {"processor": "MuseProcessor", "generation_eos": [200001, 200008],
                "renders": [{"id": "a", "layout": layout, **render}
                            for layout in preflight.LAYOUTS],
                "ids": {"yes": [5], "no": [6], "special": dict(preflight.SPECIAL),
                        "eos_token_id": 200001}}
    report = preflight.summarize(rendered, sequence_len=4096, system_prompt=SYSTEM_PROMPT,
                                 bundle_system_in_user=False)
    runs = fake_modal(monkeypatch, report)
    prompts = answer(monkeypatch, "n")
    assert cli.main([*TRAIN, "--preflight"]) == 1 and runs == [] and len(prompts) == 1
    assert cli.main([*TRAIN, "--preflight", "--estimate-only"]) == 0 and runs == []
    answer(monkeypatch, "y")
    assert cli.main([*TRAIN, "--preflight"]) == 0
    [command] = runs
    assert command[:3] == ["modal", "run", "backend/training/modal_train.py"]
    assert "--preflight" in command and "--smoke" not in command
    out = capsys.readouterr().out
    assert "verdict: TE_SYSTEM_IN_USER=0 (the default) works" in out
    assert "preflight: passed" in out and "exp1-mixed.preflight.json" in out
    assert not (env / "data/train/exp1-mixed.status.json").exists()  # nothing trained
    assert gpu_ledger.read(env / "data") == []  # a CPU job: not in the GPU ledger
