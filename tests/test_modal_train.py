"""Static checks on the Modal training app. Modal is not installed in CI, so the script
is parsed, not run; the config it sends is tested in test_training_prep.py."""

import ast
import copy
import json
import re
import sys
import time
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend import cli_train

SCRIPT = Path(__file__).resolve().parents[1] / "backend" / "training" / "modal_train.py"
SECRET_NAMES = ("WANDB_API_KEY", "HF_TOKEN", "MODEL_API_KEY")


def tree():
    return ast.parse(SCRIPT.read_text())


def constants():
    return {t.id: n.value.value for n in tree().body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}


def test_the_image_follows_the_recipe_pins():
    text = SCRIPT.read_text()
    for pin in ('"torch==2.12.1"', '"torchvision==0.27.1"', '"axolotl==0.19.0"',
                '"flash-attn==2.8.3"', "4dfa52212d0552805aec5a6a2176c0abbea277d2",
                'add_python="3.12"', "13.0"):
        assert pin in text, pin
    assert "deepspeed" not in text  # the recipe says not to install that extra


def test_train_runs_on_one_h100_with_both_modal_secrets_and_the_volume():
    c = constants()
    assert (c["GPU"], c["WANDB_SECRET"], c["HF_SECRET"], c["VOLUME"]) == (
        "H100", "wandb", "huggingface", "trained-eye")
    [decorator] = [d for n in tree().body if isinstance(n, ast.FunctionDef) and n.name == "train"
                   for d in n.decorator_list if isinstance(d, ast.Call)]
    keywords = {kw.arg: ast.unparse(kw.value) for kw in decorator.keywords}
    assert keywords["gpu"] == "GPU" and keywords["volumes"] == "{MOUNT: volume}"
    assert "modal.Secret.from_name(WANDB_SECRET)" in keywords["secrets"]
    assert "modal.Secret.from_name(HF_SECRET)" in keywords["secrets"]


def test_the_script_never_reads_prints_or_hardcodes_keys():
    for node in ast.walk(tree()):
        text = ast.unparse(node) if isinstance(node, (ast.Subscript, ast.Call)) else ""
        if "environ" in text or "getenv" in text:
            assert not any(name in text for name in SECRET_NAMES), text
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            assert name not in ("wandb.login", "huggingface_hub.login"), text
            if name == "print":
                assert "environ" not in text and "env" not in text.split("(", 1)[1], text


def test_the_entrypoint_checks_the_bundle_was_made_for_this_run():
    [main] = [n for n in tree().body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    body = ast.unparse(main)
    assert "manifest['image_root'] != f'{MOUNT}/bundles/{run}'" in body
    assert "qlora_config(" in body and "smoke=smoke" in body


# -- the plain parts, run without modal: train(), log_bundle(), main() ------------------

def function(name, **namespace):
    """`name` from the script without its Modal decorators, run in `namespace` (over the
    script's constants and its plain helpers timings and now_iso)."""
    helpers = [n for n in tree().body if isinstance(n, ast.Assign)
               and not any(w in ast.unparse(n) for w in ("modal", "image", "__file__"))]
    helpers += [n for n in tree().body if isinstance(n, ast.FunctionDef)
                and n.name in ("timings", "now_iso")]
    node = copy.deepcopy(next(n for n in tree().body
                              if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    ns = {"json": json, "re": re, "time": time, "Path": Path, "UTC": UTC,
          "datetime": datetime}
    code = compile(ast.Module(body=[*helpers, node], type_ignores=[]), str(SCRIPT), "exec")
    exec(code, ns)  # noqa: S102 - our own script's function, without modal
    ns.update(namespace)
    return ns[name]


SUMMARY = {"trained_eye": {"value": {"data_hash": "0123456789abcdef" * 4, "run": "exp1-mixed"}}}


def test_train_writes_its_result_to_the_volume_and_points_wandb_at_the_summary(tmp_path):
    commits, seen = [], {}
    old = tmp_path / "runs/exp1-mixed/result.json"
    old.parent.mkdir(parents=True)
    old.write_text('{"run": "exp1-mixed", "from": "an earlier run"}')

    def run(command, env, stdout, stderr, check):
        assert not old.exists()  # removed before training: --collect never sees it
        seen.update(command=command, env=env)
        stdout.write("{'train_runtime': 41.2, 'train_samples_per_second': 0.05}\n")
        return types.SimpleNamespace(returncode=0)

    train = function("train", MOUNT=str(tmp_path),
                     volume=types.SimpleNamespace(commit=lambda: commits.append(1)),
                     subprocess=types.SimpleNamespace(run=run, STDOUT=-2),
                     log_bundle=lambda *a: pytest.fail("no artifacts were asked for"))
    config = {"max_steps": 2, "output_dir": "/vol/runs/exp1-mixed/out",
              "wandb_project": "trained-eye"}
    out = train("exp1-mixed", config, ["exp:exp1", "data:0123456789ab"], smoke=True,
                summary=SUMMARY)
    saved = json.loads(old.read_text())
    assert saved == {k: v for k, v in out.items() if k != "log_tail"} and "from" not in saved
    assert (saved["ok"], saved["smoke"], saved["steps"], saved["train_samples_per_second"]) == (
        True, True, 2, 0.05)
    assert saved["finished"] >= saved["started"] and saved["started"].endswith("Z")
    assert seen["command"][:2] == ["axolotl", "train"]
    assert seen["env"]["WANDB_TAGS"] == "exp:exp1,data:0123456789ab"
    assert json.loads(Path(seen["env"]["WANDB_CONFIG_PATHS"]).read_text()) == SUMMARY
    assert len(commits) == 2 and "wandb_artifact" not in saved


def test_artifacts_log_the_bundle_named_after_the_run_and_its_data_hash(tmp_path, monkeypatch):
    calls = {}

    class Artifact:
        def __init__(self, name, type, metadata):
            calls.update(name=name, type=type, metadata=metadata)

        def add_dir(self, path):
            calls["dir"] = path

    wb = types.SimpleNamespace(log_artifact=lambda a: calls.update(logged=True),
                               finish=lambda: calls.update(finished=True))
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(
        init=lambda **kw: calls.update(init=kw) or wb, Artifact=Artifact))
    log_bundle = function("log_bundle")
    name = log_bundle("exp1-mixed", {"wandb_project": "trained-eye", "wandb_entity": "team"},
                      ["exp:exp1"], SUMMARY)
    assert name == calls["name"] == "exp1-mixed-0123456789ab" and calls["type"] == "dataset"
    assert calls["dir"] == "/vol/bundles/exp1-mixed" and calls["logged"] and calls["finished"]
    assert (calls["init"]["entity"], calls["init"]["job_type"]) == ("team", "dataset")

    # train() calls it only when asked, and records the artifact's name
    def run(command, env, stdout, stderr, check):
        return types.SimpleNamespace(returncode=0)

    train = function("train", MOUNT=str(tmp_path), log_bundle=lambda *a: "exp1-mixed-0123",
                     volume=types.SimpleNamespace(commit=lambda: None),
                     subprocess=types.SimpleNamespace(run=run, STDOUT=-2))
    out = train("exp1-mixed", {"output_dir": "/o"}, [], summary=SUMMARY, artifacts=True)
    assert out["wandb_artifact"] == "exp1-mixed-0123"


def bundle_for(tmp_path, run="exp1-mixed"):
    folder = tmp_path / "bundle"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({
        "image_root": f"/vol/bundles/{run}", "variant": "mixed", "limit": None,
        "dataset_dir": "dataset", "skills": {"burner_left_on": "ab" * 32}, "counts": {},
        "system_in_user": False, "train_examples": 40, "train_sha256": "cd" * 32}))
    return folder


def entrypoint(remote):
    return function("main", train=types.SimpleNamespace(remote=remote),
                    upload=lambda bundle, run: f"/vol/bundles/{run}")


def test_the_entrypoint_passes_entity_artifacts_and_the_summary(tmp_path, capsys):
    seen = {}

    def remote(run, config, tags, **kw):
        seen.update(config=config, tags=tags, **kw)
        return {"run": run, "ok": True, "seconds": 9.0, "finished": "2026-11-03T15:00:00Z",
                "log_tail": ["..."]}

    result = tmp_path / "result.json"
    entrypoint(remote)(str(bundle_for(tmp_path)), "exp1-mixed", tags="exp:exp1,commit:abc1234",
                       result=str(result), wandb_entity="team", wandb_artifacts=True)
    assert (seen["config"]["wandb_entity"], seen["config"]["wandb_log_model"]) == ("team", "end")
    assert seen["artifacts"] is True and seen["smoke"] is False
    info = seen["summary"]["trained_eye"]["value"]
    assert info["commit"] == "abc1234" and info["bundle"]["train_sha256"] == "cd" * 32
    saved = json.loads(result.read_text())
    assert saved["ok"] and "log_tail" not in saved and saved["smoke"] is False

    seen.clear()
    entrypoint(remote)(str(bundle_for(tmp_path / "b")), "exp1-mixed", result=str(result))
    assert "wandb_entity" not in seen["config"] and seen["config"]["wandb_log_model"] is None
    assert seen["artifacts"] is False


def test_a_lost_client_leaves_a_detached_run_to_collect(tmp_path, capsys):
    def lost(*a, **kw):
        raise ConnectionError("the connection to Modal was lost")

    result = tmp_path / "result.json"
    main = entrypoint(lost)
    with pytest.raises(ConnectionError):
        main(str(bundle_for(tmp_path)), "exp1-mixed", result=str(result), detached=True)
    assert not result.exists()  # the run may still be going: no failure is written
    assert "te train --collect exp1-mixed" in capsys.readouterr().out
    with pytest.raises(ConnectionError):  # attached: the time so far, as an upper bound
        main(str(bundle_for(tmp_path / "b")), "exp1-mixed", result=str(result), smoke=True)
    saved = json.loads(result.read_text())
    assert (saved["ok"], saved["smoke"], saved["seconds_upper_bound"]) == (False, True, True)
    assert saved["seconds"] >= 0


def test_the_preflight_is_a_cpu_function_with_the_pinned_processor():
    [call] = [d for n in tree().body if isinstance(n, ast.FunctionDef) and n.name == "preflight"
              for d in n.decorator_list if isinstance(d, ast.Call)]
    keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
    assert "gpu" not in keywords and keywords["image"] == "preflight_image"
    assert keywords["volumes"] == "{MOUNT: volume}"  # the images and the HF cache
    assert keywords["secrets"] == "[modal.Secret.from_name(HF_SECRET)]"
    funcs = {n.name: ast.unparse(n) for n in tree().body if isinstance(n, ast.FunctionDef)}
    assert "AutoProcessor.from_pretrained(base, revision=revision)" in funcs["preflight"]
    assert "te_preflight.render_all(processor, examples" in funcs["preflight"]
    assert ("image.add_local_file(Path(__file__).with_name('preflight.py'), "
            "'/root/te_preflight.py')") in ast.unparse(tree())
    for piece in ("config.STUDENT_BASE_REVISION", "RECIPE['sequence_len']",
                  "config.check_model_allowed(base)", "upload(bundle, run)"):
        assert piece in funcs["run_preflight"], piece
    assert "return run_preflight(" in funcs["main"]
    assert cli_train.TRAIN_VOLUME == constants()["VOLUME"]


# -- the W&B parts of the config (backend/training/axolotl.py) -------------------------

def manifest(run="exp1-mixed", **extra):
    return {"variant": "mixed", "limit": None, "image_root": f"/vol/bundles/{run}",
            "dataset_dir": "dataset", "skills": {"burner_left_on": "ab" * 32},
            "system_in_user": False, "counts": {"train/burner_left_on/real/yes": 5},
            "train_examples": 5, "train_sha256": f"{run}-paths", **extra}


def test_wandb_entity_and_model_uploads_are_opt_in():
    from backend.training.axolotl import qlora_config

    plain = qlora_config("exp1-mixed", "/vol/t.jsonl", "/vol/o")
    assert "wandb_entity" not in plain and plain["wandb_log_model"] is None
    asked = qlora_config("exp1-mixed", "/vol/t.jsonl", "/vol/o", entity="team", log_model=True)
    assert (asked["wandb_entity"], asked["wandb_log_model"]) == ("team", "end")


def test_the_data_hash_is_the_same_under_any_run_name_and_tags_say_the_arm():
    from backend.config import STUDENT_BASE_REVISION
    from backend.training.axolotl import data_hash, wandb_config, wandb_tags

    assert data_hash(manifest("exp1-mixed")) == data_hash(manifest("exp2-filtered"))
    assert data_hash(manifest()) != data_hash(manifest(dataset_dir="dataset-unfiltered"))
    assert data_hash(manifest()) != data_hash(manifest(limit=25))
    tags = wandb_tags(manifest(dataset_dir="dataset-unfiltered", limit=25))
    assert tags == [f"data:{data_hash(manifest(dataset_dir='dataset-unfiltered', limit=25))[:12]}",
                    f"base:{STUDENT_BASE_REVISION[:12]}", "dataset:dataset-unfiltered",
                    "images:25"]
    assert wandb_tags(manifest())[-1] == "images:all" and all(len(t) <= 64 for t in tags)
    config = wandb_config(manifest(), "exp1-mixed", ["exp:exp1", "commit:abc1234"])
    info = config["trained_eye"]["value"]  # W&B's config-defaults format
    assert (info["run"], info["commit"], info["base_revision"]) == (
        "exp1-mixed", "abc1234", STUDENT_BASE_REVISION)
    assert info["data_hash"] == data_hash(manifest()) and "image_root" not in info["bundle"]
    assert info["bundle"]["train_sha256"] == "exp1-mixed-paths"
    assert json.loads(json.dumps(config)) == config  # written as JSON, which YAML reads
