"""Static checks on the Modal smoke script. Modal and W&B are not installed in CI,
so the script is parsed, not run."""

import ast
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "backend" / "training" / "wandb_smoke.py"
SECRET_NAMES = ("WANDB_API_KEY", "HF_TOKEN", "MODEL_API_KEY")


def tree():
    return ast.parse(SCRIPT.read_text())


def test_function_gets_the_wandb_modal_secret():
    module = tree()
    consts = {t.id: n.value.value for n in module.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    assert consts["MODAL_SECRET"] == "wandb" and consts["WANDB_PROJECT"] == "trained-eye"
    decorators = [d for n in module.body if isinstance(n, ast.FunctionDef)
                  for d in n.decorator_list if isinstance(d, ast.Call)
                  and ast.unparse(d.func) == "app.function"]
    assert decorators, "smoke() must be a Modal function"
    secrets = [kw.value for kw in decorators[0].keywords if kw.arg == "secrets"]
    assert secrets and "modal.Secret.from_name(MODAL_SECRET)" in ast.unparse(secrets[0])


def test_script_never_reads_prints_or_hardcodes_keys():
    for node in ast.walk(tree()):
        text = ast.unparse(node) if isinstance(node, (ast.Subscript, ast.Call)) else ""
        if "environ" in text or "getenv" in text:
            assert not any(name in text for name in SECRET_NAMES), text
        if isinstance(node, ast.Call):
            assert ast.unparse(node.func) != "wandb.login", "the Modal secret provides the key"
            assert not any(kw.arg in ("key", "api_key") for kw in node.keywords), text
