"""Static checks on the Modal vLLM app. Modal and vLLM are not installed in CI, so the script
is parsed, not run; the vLLM command is read from serve_command's returned list."""

import ast
import inspect
from pathlib import Path

from backend.config import Settings
from backend.inference.student import OpenAIStudent

SCRIPT = Path(__file__).resolve().parents[1] / "backend" / "inference" / "modal_serve.py"
TREE = ast.parse(SCRIPT.read_text())
FUNCS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}


def constants():
    return {n.targets[0].id: n.value.value for n in TREE.body if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Constant)}


def command(model="/vol/runs/exp1-mixed/merged"):
    """serve_command's list, with the constants (and str() of them) filled in."""
    values = {**constants(), "model": model}
    [ret] = [n for n in ast.walk(FUNCS["serve_command"]) if isinstance(n, ast.Return)]
    out = []
    for elt in ret.value.elts:
        if isinstance(elt, ast.Call):
            assert ast.unparse(elt.func) == "str"
            elt = elt.args[0]
        out.append(str(elt.value) if isinstance(elt, ast.Constant) else str(values[elt.id]))
    return out


def flags():
    cmd = command()
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
    assert ".entrypoint([])" in SCRIPT.read_text()


def test_it_serves_the_merged_folder_as_trained_eye_with_enough_logprobs():
    f = flags()
    assert f["--served-model-name"] == "trained-eye" == Settings().student_model
    asked = inspect.signature(OpenAIStudent).parameters["top_logprobs"].default
    assert int(f["--max-logprobs"]) >= asked == 10
    assert "return f'{MOUNT}/runs/{run}/merged'" in ast.unparse(FUNCS["model_dir"])
    assert constants()["MOUNT"] == "/vol" and constants()["VOLUME"] == "trained-eye"
    assert "model = model_dir(os.environ.get(RUN_ENV, ''))" in ast.unparse(FUNCS["serve"])


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
    assert (c["API_SECRET"], c["API_KEY_ENV"]) == ("student", "STUDENT_API_KEY")
    assert "--api-key" not in command()  # vLLM logs its arguments: the key goes in the env
    [call] = [d for d in FUNCS["serve"].decorator_list if ast.unparse(d.func) == "app.function"]
    assert {kw.arg: ast.unparse(kw.value) for kw in call.keywords}["secrets"] == (
        "[modal.Secret.from_name(API_SECRET)]")
    names = [n for n in ast.walk(TREE) if isinstance(n, ast.Constant) and n.value == c[
        "API_KEY_ENV"]]
    assert len(names) == 1 and "environ[" not in text  # only API_KEY_ENV, read with .get
    for node in ast.walk(TREE):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            assert name != "print" and not name.startswith(("logging.", "logger.")), name
            assert "shell" not in {kw.arg for kw in node.keywords}
    assert ("subprocess.Popen(serve_command(model), env={**os.environ, 'VLLM_API_KEY': api_key})"
            in ast.unparse(FUNCS["serve"]))
