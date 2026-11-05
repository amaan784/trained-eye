"""The model card: its front matter, sections and license text, from its files."""

import json

import pytest

from backend.config import STUDENT_BASE_REVISION
from backend.release.model_card import USAGE_POLICY, model_card
from backend.training.axolotl import RECIPE

COUNTS = {"train/burner_left_on/real/no": 18, "train/burner_left_on/real/yes": 3,
          "train/burner_left_on/synthetic/yes": 120, "train/burner_left_on/synthetic/no": 131,
          "val/burner_left_on/synthetic/yes": 9, "train/tap_left_running/real/yes": 4}


def inputs(tmp_path):
    bundle, manifests, weights = tmp_path / "bundle.json", tmp_path / "manifests", tmp_path / "out"
    bundle.write_text(json.dumps({"variant": "mixed", "train_examples": 276, "val_examples": 9,
                                  "skills": {"burner_left_on": "", "tap_left_running": ""}, "counts": COUNTS}))
    manifests.mkdir()
    (manifests / "burner_left_on.jsonl").write_text('{"id": "x"}\n')
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
    assert "| burner_left_on | train | 3 | 18 | 120 | 131 |" in card
    assert "| burner_left_on | val | 0 | 0 | 9 | 0 |" in card and "| tap_left_running | train | 4 |" in card
    assert all(w in card.lower() for w in ("covert", "health", "coaching", "faces are blurred"))
    assert not set(card) & {chr(0x2013), chr(0x2014)}  # no en or em dashes
    (tmp_path / "results.json").write_text(json.dumps({"evaluations": [
        {"name": "exp4-tuned", "experiment": 4, "skill": "burner_left_on", "variant": "tuned|a",
         "status": "scored", "metrics": {"f1": 0.5, "hours": None},
         "cost_per_hour": {"total_usd": 1.25}},
        {"name": "exp1-real", "status": "pending", "clips": [{"log": "/Users/x/log.jsonl"}]}]}))
    card = model_card(tmp_path / "README.md", run="final", bundle_manifest=bundle, kind="merged",
                      results=tmp_path / "results.json", license_id="apache-2.0").read_text()
    assert "| exp4-tuned | 4 | burner_left_on | tuned\\|a | n/a | n/a | n/a | 0.5 |" in card
    assert "| 1.25 |" in card and "scored: 1, pending: 1" in card
    assert "exp1-real" not in card and "/Users" not in card
    assert "license: apache-2.0" in card and "relation: finetune" in card
    assert "Not evaluated" not in card
    for bad in ({"kind": "gguf"}, {"license_id": "MIT\nbase_model: other"}):
        with pytest.raises(ValueError):
            model_card(tmp_path / "README.md", run="final", bundle_manifest=bundle, **bad)


def test_the_card_shows_this_run_and_its_judges_names_the_footage_and_its_models(tmp_path):
    bundle, _, _ = inputs(tmp_path)

    def ev(name, run, experiment, variant):
        return {"name": name, "run": run, "experiment": experiment, "skill": "burner_left_on",
                "variant": variant, "status": "scored", "metrics": {"f1": 0.9},
                "cost_per_hour": {"total_usd": 3.95}}

    (tmp_path / "results.json").write_text(json.dumps({
        "footage": "other kitchens (EPIC-KITCHENS-100, head-mounted camera)",
        "threshold_objective": "frames",
        "evaluations": [ev("epic-exp4-finetuned", "final", 4, "finetuned"),
                        ev("epic-exp4-base", "exp4-base", 4, "base"),
                        ev("epic-exp4-spark", "exp4-spark", 4, "spark"),
                        ev("epic-exp1-epic", "exp1-epic", 1, "real, EPIC")]}))
    card = model_card(tmp_path / "README.md", run="final", bundle_manifest=bundle,
                      results=tmp_path / "results.json", spoken=False).read_text()
    assert "epic-exp4-finetuned" in card and "epic-exp4-base" in card and "epic-exp4-spark" in card
    assert "epic-exp1-epic" not in card  # another trained model: not this card's
    assert "Test footage: other kitchens (EPIC-KITCHENS-100, head-mounted camera)." in card
    assert "CC BY-NC 4.0" in card and "never trained on it" in card
    assert "calibrated for frame F1" in card
    assert "transcribe" not in card.split("## Training data")[0]  # typed requests only

def test_the_card_states_fixed_steps_and_a_real_only_bundle(tmp_path):
    bundle = tmp_path / "real.json"
    bundle.write_text(json.dumps({"variant": "real", "train_examples": 22, "val_examples": 0,
                                  "skills": {"burner_left_on": ""},
                                  "counts": {"train/burner_left_on/real/no": 17,
                                             "train/burner_left_on/real/yes": 5}}))
    card = model_card(tmp_path / "README.md", run="exp1-real", bundle_manifest=bundle,
                      max_steps=10).read_text()
    assert "| `max_steps` | `10` |" in card and "fixed 10 optimizer steps" in card
    assert "no generated images in this bundle" in card and "Generated:" not in card
    assert "Muse Image edits" not in card
    (tmp_path / "m").mkdir()
    mixed, _, _ = inputs(tmp_path / "m")
    assert "Most training images are Muse Image edits" in model_card(
        tmp_path / "m.md", run="final", bundle_manifest=mixed).read_text()


def test_the_card_says_what_the_repo_holds_and_how_to_prompt_it(tmp_path):
    from backend.dataengine.dataset import SYSTEM_PROMPT

    bundle, _, _ = inputs(tmp_path)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"0" * 2_000_000)
    (tmp_path / "manifests" / "burner_left_on.jsonl").write_text('{"id": 1}\n{"id": 2}\n')
    specs = {"burner_left_on": {"request": "Tell me when a burner is on",
                                "positive": "a burner is lit.", "negatives": ["a pan on it."]}}
    card = model_card(tmp_path / "README.md", run="final", bundle_manifest=bundle,
                      specs=specs, repo="someone/trained-eye").read_text()
    files = card.split("## What is in this repository")[1].split("## How to use it")[0]
    assert "| `adapter_model.safetensors` |" in files and "2 MB." in files
    assert "(2 for burner_left_on)" in files and "| `LICENSE` |" in files
    assert "`config.json`" not in files  # only what the folder holds
    assert "It is not a model on its own" in card and "trained on two skills" in card
    assert SYSTEM_PROMPT in card and "512 x 512" in card
    assert "| burner_left_on | Tell me when a burner is on | a burner is lit | a pan on it |" in card
    assert 'PeftModel.from_pretrained(model, "someone/trained-eye")' in card
    assert "## Citation" in card and "@software{sheikh2026trainedeye" in card
    assert not set(card) & {chr(0x2013), chr(0x2014)}
