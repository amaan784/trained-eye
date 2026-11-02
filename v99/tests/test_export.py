"""The export: static checks on the Modal app (Modal, torch and transformers are not installed
in CI, so export.py is parsed, not run) and real tests of export_check, its plain helpers."""

import ast
import json
from pathlib import Path

import pytest

from backend.config import DEFAULT_MODELS
from backend.training.export_check import LIMIT, compare, pick, yes_no

SCRIPT = Path(__file__).resolve().parents[1] / "backend" / "training" / "export.py"
TREE = ast.parse(SCRIPT.read_text())
FUNCS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}
SECRET_NAMES = ("WANDB_API_KEY", "HF_TOKEN", "MODEL_API_KEY", "STUDENT_API_KEY")


def constants():
    return {n.targets[0].id: n.value.value for n in TREE.body if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Constant)}


def test_one_h100_the_hf_secret_the_volume_and_the_training_pins():
    c, text = constants(), SCRIPT.read_text()
    assert (c["GPU"], c["HF_SECRET"], c["VOLUME"], c["MOUNT"]) == (
        "H100", "huggingface", "trained-eye", "/vol")
    [call] = [d for d in FUNCS["export"].decorator_list if isinstance(d, ast.Call)]
    keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
    assert keywords["gpu"] == "GPU" and keywords["volumes"] == "{MOUNT: volume}"
    assert keywords["secrets"] == "[modal.Secret.from_name(HF_SECRET)]"
    for pin in ('"torch==2.12.1"', '"torchvision==0.27.1"', '"axolotl==0.19.0"',
                'add_python="3.12"', '"HF_HOME": f"{MOUNT}/hf"'):
        assert pin in text, pin
    assert "deepspeed" not in text


def test_the_base_comes_from_config_and_the_merge_lands_in_merged():
    main, body = ast.unparse(FUNCS["main"]), ast.unparse(FUNCS["export"])
    assert DEFAULT_MODELS["student_base"] == "meta-models/Muse-Glimmer-30B"
    assert "export.remote(run, base, config.STUDENT_BASE_REVISION, examples" in main
    assert "config.DEFAULT_MODELS['student_base']" in main
    assert "config.check_model_allowed(base)" in main
    assert "manifest['image_root'] != f'{MOUNT}/bundles/{run}'" in main
    for part in ("run_dir = Path(MOUNT) / 'runs' / run",
                 "adapter, merged_dir = (run_dir / 'out', run_dir / 'merged')",
                 "is_trainable=False", "model.merge_and_unload()",
                 "merged.save_pretrained(str(merged_dir))",
                 "processor.save_pretrained(str(merged_dir))", "volume.commit()",
                 "load_model(transformers, torch, str(merged_dir), None)"):
        assert part in body, part


def test_loading_and_prompting_follow_the_cookbook_evaluator():
    [classes] = [n.value for n in TREE.body if isinstance(n, ast.Assign)
                 and n.targets[0].id == "MODEL_CLASSES"]
    assert ast.literal_eval(classes) == (
        "AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM")
    load, gen = ast.unparse(FUNCS["load_model"]), ast.unparse(FUNCS["generate"])
    assert "'torch_dtype': torch.bfloat16" in load and "bnb_4bit_quant_type='nf4'" in load
    assert "bnb_4bit_quant_storage=torch.bfloat16" in load
    assert ("apply_chat_template(messages, tokenize=False, add_generation_prompt=True, "
            "reasoning_strength=strength, current_date=today)") in gen
    assert "do_sample=False" in gen and "'<|eot|>'" in gen and "skip_special_tokens=True" in gen


def test_the_script_never_reads_prints_or_hardcodes_keys():
    for node in ast.walk(TREE):
        text = ast.unparse(node) if isinstance(node, (ast.Subscript, ast.Call)) else ""
        assert not any(name in text for name in SECRET_NAMES), text
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "print":
            assert text == "print(json.dumps(summary))", text  # no raw generations, no env


def test_the_answer_is_read_after_to_user():
    assert yes_no("to=self The drum has stopped, so yes. to=user Yes.") == "yes"
    assert yes_no("to=user No, it is still spinning. to=self more") == "no"
    assert yes_no("to=self thinking yes, then the token cap") is None  # reasoning only
    assert yes_no("to=user I cannot tell.") is None


def test_examples_are_picked_in_a_fixed_order_whatever_the_file_order():
    lines = [json.dumps({"id": f"e{i}"}) for i in range(80)] + [""]
    picked = [e["id"] for e in pick(lines)]
    assert LIMIT == 50 and len(picked) == 50 and len(set(picked)) == 50
    assert picked == [e["id"] for e in pick(lines[::-1])] != [f"e{i}" for i in range(50)]
    assert len(pick(lines, limit=3)) == 3 and len(pick(lines[:2])) == 2


def test_compare_counts_agreement_and_an_unreadable_answer_never_agrees():
    out = compare(["a", "b", "c"], ["yes", "no", "no"], ["yes", None, "no"], ["yes", None, "yes"])
    assert out["agreement"] == 1 / 3 and out["disagreements"] == ["b", "c"]
    assert out["unreadable"] == {"reference": 1, "merged": 1} and not out["all_agree"]
    assert out["label_match"] == {"reference": 2 / 3, "merged": 1 / 3}
    assert compare(["a"], ["no"], ["no"], ["no"])["all_agree"]
    with pytest.raises(ValueError):
        compare([], [], [], [])
