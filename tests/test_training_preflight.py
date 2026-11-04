"""The processor preflight (backend/training/preflight.py) with a fake processor: transformers
is not installed in CI. The fake renders a Muse-like template: <|start|>role<|message|>
content<|eot|>, an image as image_tokens <|image|> tokens, and the generation prompt
<|start|>assistant<|message|>."""

import ast
import copy
import json
import re
import types
from pathlib import Path

import pytest

from backend.dataengine.dataset import SYSTEM_PROMPT, Example, build_record
from backend.training import preflight

SCRIPT = Path(__file__).resolve().parents[1] / "backend" / "training"
SPECIAL = {"<|begin_of_text|>": 200000, "<|end_of_text|>": 200001, "<|start|>": 200005,
           "<|message|>": 200006, "<|eom|>": 200007, "<|eot|>": 200008, "<|image|>": 200090}


class FakeTokenizer:
    def __init__(self, split_yes=False, special=None):
        self.vocab, self.split_yes = dict(special or SPECIAL), split_yes
        self.eos_token_id = 200001

    def pieces(self, text):
        out = re.findall(r"<\|[a-z_]+\|>|\w+|[^\w\s]|\s+", text)
        return [p for word in out for p in (["y", "es"] if self.split_yes and word == "yes"
                                             else [word])]

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(p, len(self.vocab) + 10) for p in self.pieces(text)]

    def get_vocab(self):
        return dict(self.vocab)

    def convert_ids_to_tokens(self, ids):
        inverse = {v: k for k, v in self.vocab.items()}
        return [inverse[i] for i in ids]


class FakeProcessor:
    def __init__(self, system="keep", image_tokens=10, **tokenizer):
        self.tokenizer, self.system, self.image_tokens = FakeTokenizer(**tokenizer), system, \
            image_tokens

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False,
                            return_dict=False, return_tensors=None):
        parts = ["<|begin_of_text|>"]
        for m in messages:
            if m["role"] == "system" and self.system == "fail":
                raise ValueError("this template has no system role")
            if m["role"] == "system" and self.system == "drop":
                continue
            content = m["content"] if isinstance(m["content"], str) else "".join(
                "<|image|>" * self.image_tokens if p["type"] == "image" else p["text"]
                for p in m["content"])
            parts.append(f"<|start|>{m['role']}<|message|>{content}<|eot|>")
        if add_generation_prompt:
            parts.append("<|start|>assistant<|message|>")
        text = "".join(parts)
        if not tokenize:
            return text
        assert return_dict and return_tensors is None  # the cookbook's call
        return {"input_ids": [self.tokenizer.encode(text)]}  # batched, as the processor


def record(i, answer, folded=False):
    example = Example(f"burner_left_on-real-{i:03d}", b"", answer, "real", "burner_left_on",
                      "the burner is lit with no pan on it", ["a pan being stirred on it"])
    return build_record(example, Path(f"/vol/bundles/exp1/images/{i}.png"), folded)


RECORDS = [record(i, "yes" if i % 2 else "no") for i in range(6)]


def report(processor, records=RECORDS, folded=False, **kw):
    items = preflight.examples(records, 4, SYSTEM_PROMPT)
    rendered = preflight.render_all(processor, items, generation_eos=kw.pop("eos", [200001,
                                                                                    200008]))
    return preflight.summarize(rendered, sequence_len=kw.pop("sequence_len", 4096),
                               system_prompt=SYSTEM_PROMPT, bundle_system_in_user=folded)


def test_both_layouts_from_either_way_the_dataset_was_written():
    for folded in (False, True):
        layouts = preflight.both_layouts(record(1, "yes", folded), SYSTEM_PROMPT)
        turn, user = layouts["system_turn"], layouts["system_in_user"]
        assert turn[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert turn[1]["content"][1]["text"].startswith("Condition: the burner is lit with no pan on it.")
        assert user[0]["role"] == "user" and user[-1] == {"role": "assistant", "content": "yes"}
        assert user[0]["content"][1]["text"] == f"{SYSTEM_PROMPT}\n\n" + turn[1]["content"][1][
            "text"]
        assert user[0]["content"][0]["type"] == "image"  # the image part stays first
    original = record(2, "no")
    before = copy.deepcopy(original)
    preflight.both_layouts(original, SYSTEM_PROMPT)
    assert original == before  # the record itself is left alone


def test_pick_takes_yes_and_no_in_turn_by_id():
    picked = preflight.pick(list(reversed(RECORDS)), 3)
    assert [r["id"][-3:] for r in picked] == ["000", "001", "002"]
    assert [r["messages"][-1]["content"] for r in picked] == ["no", "yes", "no"]
    assert len(preflight.examples(RECORDS, 2, SYSTEM_PROMPT)) == 4  # each in both layouts


def test_a_template_that_renders_the_system_turn_passes():
    r = report(FakeProcessor())
    assert r["ok"] and r["use_system_in_user"] is False and r["system_turn"] == "renders once"
    assert r["advice"].startswith("TE_SYSTEM_IN_USER=0 (the default) works")
    assert r["layout_checked"] == "system_turn" and r["over_length"] == [] and r["notes"] == []
    assert r["longest"]["tokens"] > 10  # the image counts, as the cookbook's counter counts it
    assert r["rendered"]["system_turn"].startswith(f"<|begin_of_text|><|start|>system<|message|>"
                                                   f"{SYSTEM_PROMPT}<|eot|>")
    yes = next(t for t in r["targets"] if t["answer"] == "yes")
    assert yes["text"] == "yes<|eot|>" and yes["tokens"] == ["yes", "<|eot|>"]
    assert yes["answer_in_target"] and yes["ends_with_stop"]
    assert r["ids"]["special"] == {"<|end_of_text|>": 200001, "<|eot|>": 200008,
                                   "<|eom|>": 200007}
    text = preflight.format_report(r)
    assert "verdict: TE_SYSTEM_IN_USER=0" in text and text.endswith("preflight: passed")
    assert "target for 'no'" in text and "--- rendered (system_turn) ---" in text
    json.dumps(r)  # saved as JSON next to the bundle


@pytest.mark.parametrize("system, state", [("drop", "missing from the rendered text"),
                                           ("fail", "fails: ValueError")])
def test_a_bad_system_turn_says_to_set_te_system_in_user(system, state):
    r = report(FakeProcessor(system=system))
    assert r["system_turn"].startswith(state) and r["use_system_in_user"] is True
    assert r["advice"].startswith("set TE_SYSTEM_IN_USER=1")
    assert not r["ok"] and r["layout_checked"] == "system_in_user"
    assert any("set TE_SYSTEM_IN_USER=1 in .env, run te teach again" in n for n in r["notes"])
    folded = report(FakeProcessor(system=system), [record(i, "yes" if i % 2 else "no", True)
                                                    for i in range(4)], folded=True)
    assert folded["ok"] and folded["notes"] == []  # a bundle written folded already passes
    assert "look at the lines above" in preflight.format_report(r)


def test_over_length_examples_and_odd_tokens_are_reported():
    r = report(FakeProcessor(image_tokens=5000))
    assert not r["ok"] and len(r["over_length"]) == 4
    assert "tokens)" in r["over_length"][0] and r["longest"]["tokens"] > 4096
    split = report(FakeProcessor(split_yes=True), eos=[200001, 200007])
    notes = " ".join(split["notes"])
    assert "'yes' is 2 tokens, not one" in notes
    assert "does not hold the 'yes' token on its own" not in notes  # the pair is in the target
    assert "eos_token_id is [200001, 200007]: vllm.md wants 200001 and 200008" in notes
    odd = report(FakeProcessor(special={**SPECIAL, "<|eot|>": 9}))
    assert any("special token IDs differ" in n for n in odd["notes"])


def test_the_preflight_file_imports_nothing_from_backend():
    tree = ast.parse((SCRIPT / "preflight.py").read_text())  # copied alone into the container
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module == "__future__", node.module
        elif isinstance(node, ast.Import):
            assert all(alias.name in ("copy",) for alias in node.names)


def test_run_preflight_renders_on_modal_and_saves_the_verdict(tmp_path):
    from backend.config import STUDENT_BASE_REVISION

    tree = ast.parse((SCRIPT / "modal_train.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "run_preflight")
    seen = {}

    def remote(base, revision, items):
        seen.update(base=base, revision=revision, n=len(items))
        return preflight.render_all(FakeProcessor(), items, generation_eos=[200001, 200008])

    ns = {"json": json, "Path": Path, "PREFLIGHT_EXAMPLES": 4,
          "upload": lambda bundle, run: seen.update(uploaded=run),
          "preflight": types.SimpleNamespace(remote=remote)}
    code = compile(ast.Module(body=[node], type_ignores=[]), "modal_train.py", "exec")
    exec(code, ns)  # noqa: S102 - our own script's function, without modal
    bundle = tmp_path / "exp1-mixed"
    bundle.mkdir()
    (bundle / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    result = tmp_path / "exp1-mixed.preflight.json"
    ns["run_preflight"](bundle, "exp1-mixed", {"system_in_user": False}, str(result))
    assert (seen["base"], seen["revision"], seen["n"], seen["uploaded"]) == (
        "meta-models/Muse-Glimmer-30B", STUDENT_BASE_REVISION, 8, "exp1-mixed")
    saved = json.loads(result.read_text())
    assert saved["ok"] and saved["run"] == "exp1-mixed" and saved["sequence_len"] == 4096
