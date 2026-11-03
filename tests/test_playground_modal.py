"""Static checks on the playground's Modal app. Modal is not installed in CI, so the script
is parsed, not run (as test_modal_serve.py does)."""

import ast
import tomllib
from pathlib import Path

from backend import cli_playground
from backend.config import Settings

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "backend" / "playground" / "modal_app.py"
TEXT = SCRIPT.read_text()
TREE = ast.parse(TEXT)
FUNCS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}


def constants():
    """Module-level constants and tuples of them, in order (a name in a tuple is looked up)."""
    out = {}
    for n in TREE.body:
        if not (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)):
            continue
        if isinstance(n.value, ast.Constant):
            out[n.targets[0].id] = n.value.value
        elif isinstance(n.value, ast.Tuple):
            out[n.targets[0].id] = tuple(out[e.id] if isinstance(e, ast.Name) else e.value
                                         for e in n.value.elts)
    return out


def decorators(name="web"):
    return [ast.unparse(d) for d in FUNCS[name].decorator_list]


def function_keywords():
    [call] = [d for d in FUNCS["web"].decorator_list
              if isinstance(d, ast.Call) and ast.unparse(d.func) == "app.function"]
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords}


def test_one_container_with_the_private_volume_and_the_meta_secret():
    c, kw = constants(), function_keywords()
    assert c["MAX_CONTAINERS"] == 1 and kw["max_containers"] == "MAX_CONTAINERS"
    assert kw["volumes"] == "{MOUNT: volume}" and kw["image"] == "image"
    assert kw["secrets"] == "[modal.Secret.from_name(API_SECRET)]" and c["API_SECRET"] == "meta"
    assert "modal.asgi_app()" in decorators()
    assert "modal.concurrent(max_inputs=MAX_INPUTS)" in decorators()
    assert "modal.Volume.from_name(VOLUME, create_if_missing=True)" in TEXT
    assert c["VOLUME"] != "trained-eye"  # not the training Volume with weights and bundles
    source = ast.unparse(TREE)
    assert "DATA_DIR = f'{MOUNT}/data'" in source and "EVAL_DIR = f'{MOUNT}/eval'" in source


def test_it_serves_the_public_copy_behind_the_proxy():
    body = ast.unparse(FUNCS["web"])
    assert "public=True" in body and "return behind_proxy(site)" in body
    assert "visitor_cap_usd=cap(VISITOR_CAP_ENV)" in body
    assert "daily_cap_usd=cap(DAILY_CAP_ENV)" in body
    assert "Path(EVAL_DIR)" in body and "load_settings().data_dir" in body
    assert "keep_committing()" in body


def test_the_image_has_the_projects_packages_and_backend_only():
    c = constants()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert list(c["PACKAGES"]) == project["dependencies"]
    assert "pip_install(*PACKAGES)" in TEXT and "python_version=PYTHON" in TEXT
    assert c["PYTHON"] == "3.12"
    calls = [ast.unparse(n) for n in ast.walk(TREE) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr.startswith("add_local")]
    assert len(calls) == 1 and "ROOT / 'backend'" in calls[0]  # never .env or data/
    assert "'TE_DATA_DIR': DATA_DIR" in ast.unparse(TREE)
    assert "parents[" not in ast.unparse(TREE)  # /root/modal_app.py has two parents


def test_the_budget_and_the_ledger_are_committed_to_the_volume():
    files = ast.unparse(FUNCS["state_files"])
    assert "playground/budget.json" in files and "ledger.jsonl" in files
    assert Settings().ledger_path.name == "ledger.jsonl"
    assert "volume.commit()" in ast.unparse(FUNCS["keep_committing"])
    assert constants()["COMMIT_S"] > 0


def test_the_cli_names_the_same_app_volume_secret_and_caps():
    c = constants()
    assert cli_playground.PLAYGROUND_APP == c["APP"]
    assert cli_playground.PLAYGROUND_VOLUME == c["VOLUME"]
    assert cli_playground.PLAYGROUND_SECRET == c["API_SECRET"]
    assert set(cli_playground.CAP_ENVS) == {c["VISITOR_CAP_ENV"], c["DAILY_CAP_ENV"]}
    assert set(c["CAP_ENVS"]) >= {*cli_playground.CAP_ENVS, "TE_SKILL_CAP_USD",
                                  "TE_GLOBAL_CAP_USD"}


def test_no_key_is_read_or_printed_here():
    assert "print(" not in TEXT and "MODEL_API_KEY" not in ast.unparse(FUNCS["web"])
    docstring = TREE.body[0].value
    names = [n.value for n in ast.walk(TREE) if isinstance(n, ast.Constant)
             and isinstance(n.value, str) and n is not docstring]
    assert not any("MODEL_API_KEY" in name for name in names)  # the Secret sets it
