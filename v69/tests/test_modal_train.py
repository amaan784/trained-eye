"""Static checks on the Modal training app. Modal is not installed in CI, so the script
is parsed, not run; the config it sends is tested in test_training_prep.py."""

import ast
from pathlib import Path

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
