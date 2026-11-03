"""The model card: its front matter, sections and license text, from its files."""

import json

import pytest

from backend.config import STUDENT_BASE_REVISION
from backend.release.model_card import USAGE_POLICY, model_card
from backend.training.axolotl import RECIPE

COUNTS = {"train/washer_done/real/no": 18, "train/washer_done/real/yes": 3,
          "train/washer_done/synthetic/yes": 120, "train/washer_done/synthetic/no": 131,
          "val/washer_done/synthetic/yes": 9, "train/bus_m4/real/yes": 4}


def inputs(tmp_path):
    bundle, manifests, weights = tmp_path / "bundle.json", tmp_path / "manifests", tmp_path / "out"
    bundle.write_text(json.dumps({"variant": "mixed", "train_examples": 276, "val_examples": 9,
                                  "skills": {"washer_done": "", "bus_m4": ""}, "counts": COUNTS}))
    manifests.mkdir()
    (manifests / "washer_done.jsonl").write_text('{"id": "x"}\n')
    (weights / "checkpoint-2").mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors", "README.md", "args.bin"):
        (weights / name).write_text(name)
    return bundle, manifests, weights


def test_the_card_says_only_what_its_files_say(tmp_path):
    bundle, _, _ = inputs(tmp_path)
    card = model_card(tmp_path / "README.md", run="final", bundle_manifest=bundle).read_text()
    front = card.split("---\n")[1]
    assert "base_model: meta-models/Muse-Glimmer-30B" in front and "library_name: peft" in front
    assert "license: apache-2.0" in front and STUDENT_BASE_REVISION in card
    assert "as the base model's" in card and "a modification of that model" in card
    assert USAGE_POLICY in card
    assert "Not evaluated yet" in card and f"| `lora_r` | `{RECIPE['lora_r']}` |" in card
    assert "| washer_done | train | 3 | 18 | 120 | 131 |" in card
    assert "| washer_done | val | 0 | 0 | 9 | 0 |" in card and "| bus_m4 | train | 4 |" in card
    assert all(w in card.lower() for w in ("covert", "health", "coaching", "faces are blurred"))
    assert not set(card) & {chr(0x2013), chr(0x2014)}  # no en or em dashes
    (tmp_path / "results.json").write_text(json.dumps({"evaluations": [
        {"name": "exp4-tuned", "experiment": 4, "skill": "washer_done", "variant": "tuned|a",
         "status": "scored", "metrics": {"f1": 0.5, "hours": None},
         "cost_per_hour": {"total_usd": 1.25}},
        {"name": "exp1-real", "status": "pending", "clips": [{"log": "/Users/x/log.jsonl"}]}]}))
    card = model_card(tmp_path / "README.md", run="final", bundle_manifest=bundle, kind="merged",
                      results=tmp_path / "results.json", license_id="apache-2.0").read_text()
    assert "| exp4-tuned | 4 | washer_done | tuned\\|a | n/a | n/a | n/a | 0.5 |" in card
    assert "| 1.25 |" in card and "scored: 1, pending: 1" in card
    assert "exp1-real" not in card and "/Users" not in card
    assert "license: apache-2.0" in card and "relation: finetune" in card
    assert "Not evaluated" not in card
    for bad in ({"kind": "gguf"}, {"license_id": "MIT\nbase_model: other"}):
        with pytest.raises(ValueError):
            model_card(tmp_path / "README.md", run="final", bundle_manifest=bundle, **bad)



def test_the_card_states_fixed_steps_and_a_real_only_bundle(tmp_path):
    bundle = tmp_path / "real.json"
    bundle.write_text(json.dumps({"variant": "real", "train_examples": 22, "val_examples": 0,
                                  "skills": {"washer_done": ""},
                                  "counts": {"train/washer_done/real/no": 17,
                                             "train/washer_done/real/yes": 5}}))
    card = model_card(tmp_path / "README.md", run="exp1-real", bundle_manifest=bundle,
                      max_steps=10).read_text()
    assert "| `max_steps` | `10` |" in card and "fixed 10 optimizer steps" in card
    assert "no generated images in this bundle" in card and "Synthetic:" not in card
    assert "Muse Image edits" not in card
    (tmp_path / "m").mkdir()
    mixed, _, _ = inputs(tmp_path / "m")
    assert "Most training images are Muse Image edits" in model_card(
        tmp_path / "m.md", run="final", bundle_manifest=mixed).read_text()
