"""Static checks on the Modal vLLM app. Modal and vLLM are not installed in CI, so the script
is parsed, not imported: its constants and plain functions (model_dir, lora_modules,
serve_command, wants_base) are run on their own, without modal."""

import ast
import inspect
import re
from pathlib import Path

import pytest

from backend import cli_serving
from backend.config import DEFAULT_MODELS, STUDENT_BASE_REVISION, Settings
from backend.inference.student import OpenAIStudent
from backend.training.axolotl import RECIPE

SCRIPT = Path(__file__).resolve().parents[1] / "backend" / "inference" / "modal_serve.py"
TREE = ast.parse(SCRIPT.read_text())
FUNCS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}
PLAIN = ("base_dir", "model_dir", "lora_modules", "serve_command", "wants_base")


def constants():
    return {n.targets[0].id: n.value.value for n in TREE.body if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Constant)}


def plain():
    """The module's constants and PLAIN functions, run without modal or os.environ."""
    body = [n for n in TREE.body if isinstance(n, ast.Assign)
            and "modal" not in ast.unparse(n) and "os.environ" not in ast.unparse(n)]
    body += [FUNCS[name] for name in PLAIN]
    namespace = {"re": re, "Path": Path}
    code = compile(ast.Module(body=body, type_ignores=[]), str(SCRIPT), "exec")
    exec(code, namespace)  # noqa: S102 - our own script's plain functions, without modal
    return namespace


def command(*args, **kwargs):
    return plain()["serve_command"](*(args or ("/vol/runs/exp1-mixed/merged",)), **kwargs)


def flags(cmd=None):
    cmd = cmd or command()
    return {a: (b if not b.startswith("--") else None) for a, b in zip(cmd, cmd[1:] + ["--"])
            if a.startswith("--")}


def test_the_cookbook_image_and_single_card_flags_on_one_80_gb_gpu():
    c, f = constants(), flags()
    assert (c["VLLM_IMAGE"], c["GPU_TYPE"], c["GPU_COUNT"]) == (
        "vllm/vllm-openai:muse-glimmer", "H100", 1)
    assert command()[:3] == ["vllm", "serve", "/vol/runs/exp1-mixed/merged"]
    assert f["--tensor-parallel-size"] == "1" and "--enable-auto-tool-choice" in f
    assert f["--tool-call-parser"] == f["--reasoning-parser"] == "muse_glimmer"
    assert f["--generation-config"] == "auto" and "--chat-template" not in f
    assert (f["--host"], f["--port"]) == ("0.0.0.0", str(c["PORT"]))
    assert "--revision" not in f and "--enable-lora" not in f  # a merged folder: neither
    assert ".entrypoint([])" in SCRIPT.read_text()
    # the image has python3 only; Modal needs `python` to read the version (first deploy)
    assert 'setup_dockerfile_commands=["RUN ln -sf /usr/bin/python3 /usr/bin/python"]' in (
        SCRIPT.read_text())


def test_it_serves_the_merged_folder_as_trained_eye_with_enough_logprobs():
    f = flags()
    assert f["--served-model-name"] == "trained-eye" == Settings().student_model
    asked = inspect.signature(OpenAIStudent).parameters["top_logprobs"].default
    assert int(f["--max-logprobs"]) >= asked == 10
    assert plain()["model_dir"]("exp1-mixed") == "/vol/runs/exp1-mixed/merged"
    with pytest.raises(ValueError, match="TE_SERVE_RUN"):
        plain()["model_dir"]("../x")
    assert constants()["MOUNT"] == "/vol" and constants()["VOLUME"] == "trained-eye"
    assert "model = model_dir(os.environ.get(RUN_ENV, ''))" in ast.unparse(FUNCS["serve"])


def test_the_base_is_the_pinned_hub_model_under_its_own_app_and_model_name():
    c, ns = constants(), plain()
    assert (c["BASE_MODEL"], c["BASE_REVISION"]) == (DEFAULT_MODELS["student_base"],
                                                     STUDENT_BASE_REVISION)
    assert c["BASE_APP"] != c["APP"]  # both endpoints can be up at once
    assert (cli_serving.STUDENT_APP, cli_serving.BASE_APP, cli_serving.BASE_MODEL_NAME) == (
        c["APP"], c["BASE_APP"], c["BASE_MODEL_NAME"])
    assert cli_serving.SERVE_ENVS == (c["RUN_ENV"], c["BASE_ENV"], c["LORAS_ENV"])
    # the pinned snapshot folder that download_base fills, not the Hub id: by id, vLLM's
    # structured-output tokenizer did not follow --tokenizer-revision (first deploy, Nov 4)
    snapshot = ns["base_dir"]()
    assert snapshot == ("/vol/hf/hub/models--meta-models--Muse-Glimmer-30B/snapshots/"
                        + STUDENT_BASE_REVISION)
    cmd = ns["serve_command"](snapshot, c["BASE_MODEL_NAME"])
    f = flags(cmd)
    assert cmd[:3] == ["vllm", "serve", snapshot]
    assert f["--served-model-name"] == "muse-glimmer" and "--revision" not in f
    assert ns["wants_base"]({"TE_SERVE_BASE": "1"}) and ns["wants_base"](
        {"TE_SERVE_LORAS": "exp1-mixed"})
    assert not ns["wants_base"]({"TE_SERVE_RUN": "exp1-mixed", "TE_SERVE_BASE": ""})
    text = SCRIPT.read_text()
    assert "app = modal.App(BASE_APP if BASE else APP)" in text
    assert '.env({"HF_HOME": f"{MOUNT}/hf"})' in text  # the cache training filled
    serve = ast.unparse(FUNCS["serve"])
    assert "serve_command(model, BASE_MODEL_NAME, loras=loras)" in serve
    assert "model = base_dir()" in serve and "'HF_HUB_OFFLINE': '1'" in serve


def test_lora_runs_are_model_names_on_one_endpoint():
    ns = plain()
    modules = ns["lora_modules"]("exp1-real, exp1-mixed,")
    assert modules == ["exp1-real=/vol/runs/exp1-real/out", "exp1-mixed=/vol/runs/exp1-mixed/out"]
    assert ns["lora_modules"]("") == []
    for bad, match in (("exp1-real,exp1-real", "twice"), ("Exp 1", "bad run name")):
        with pytest.raises(ValueError, match=match):
            ns["lora_modules"](bad)
    cmd = ns["serve_command"]("meta-models/Muse-Glimmer-30B", "muse-glimmer", "abc", modules)
    f = flags(cmd)
    assert "--enable-lora" in f and int(f["--max-lora-rank"]) >= RECIPE["lora_r"] == 32
    assert int(f["--max-cpu-loras"]) == 2 and int(f["--max-loras"]) >= 1
    assert cmd[cmd.index("--lora-modules") + 1:] == modules  # every module after the flag
    serve = ast.unparse(FUNCS["serve"])
    assert "Path(adapter, 'adapter_config.json').is_file()" in serve  # refuses a missing one
    assert "LoRA serving is not documented for Muse Glimmer" in ast.get_docstring(TREE)


def test_modal_settings_are_named_constants():
    decorators = [ast.unparse(d) for d in FUNCS["serve"].decorator_list]
    assert "modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT_S)" in decorators
    assert "modal.concurrent(max_inputs=MAX_INPUTS)" in decorators
    [call] = [d for d in FUNCS["serve"].decorator_list if ast.unparse(d.func) == "app.function"]
    keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
    assert keywords["gpu"] == "f'{GPU_TYPE}:{GPU_COUNT}'"
    assert keywords["scaledown_window"] == "SCALEDOWN_S"
    assert keywords["volumes"] == "{MOUNT: volume}"


def test_the_api_key_comes_from_the_student_secret_and_is_never_printed():
    c, text = constants(), SCRIPT.read_text()
    assert (c["API_SECRET"], c["API_KEY_ENV"], c["HF_SECRET"]) == (
        "student", "STUDENT_API_KEY", "huggingface")
    assert "--api-key" not in command()  # vLLM logs its arguments: the key goes in the env
    [call] = [d for d in FUNCS["serve"].decorator_list if ast.unparse(d.func) == "app.function"]
    assert {kw.arg: ast.unparse(kw.value) for kw in call.keywords}["secrets"] == "SECRETS"
    assert ("SECRETS = [modal.Secret.from_name(API_SECRET)] + "
            "([modal.Secret.from_name(HF_SECRET)] if BASE else [])") in ast.unparse(TREE)
    names = [n for n in ast.walk(TREE) if isinstance(n, ast.Constant) and n.value == c[
        "API_KEY_ENV"]]
    assert len(names) == 1 and "environ[" not in text  # only API_KEY_ENV, read with .get
    for node in ast.walk(TREE):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            assert name != "print" and not name.startswith(("logging.", "logger.")), name
            assert "shell" not in {kw.arg for kw in node.keywords}
    assert ("subprocess.Popen(command, env={**os.environ, 'VLLM_API_KEY': api_key, "
            "'HF_HUB_OFFLINE': '1'})" in ast.unparse(FUNCS["serve"]))
