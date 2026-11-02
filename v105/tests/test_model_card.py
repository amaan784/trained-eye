"""The model card, the release folder and the upload guard; huggingface_hub is a fake."""

import json
import sys
import types

import pytest

from backend.config import STUDENT_BASE_REVISION, MissingSecret
from backend.release.model_card import LICENSE_TODO, model_card, prepare_release, upload
from backend.training.axolotl import RECIPE

COUNTS = {"train/washer_done/real/no": 18, "train/washer_done/real/yes": 3,
          "train/washer_done/synthetic/yes": 120, "train/washer_done/synthetic/no": 131,
          "val/washer_done/synthetic/yes": 9, "train/bus_m4/real/yes": 4}
REPO = "someone/trained-eye"


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
    assert "license" not in front and LICENSE_TODO in card and STUDENT_BASE_REVISION in card
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
    assert "Not evaluated" not in card and LICENSE_TODO not in card
    for bad in ({"kind": "gguf"}, {"license_id": "MIT\nbase_model: other"}):
        with pytest.raises(ValueError):
            model_card(tmp_path / "README.md", run="final", bundle_manifest=bundle, **bad)


def test_the_release_folder_holds_the_card_weights_and_manifests(tmp_path):
    bundle, manifests, weights = inputs(tmp_path)
    release = prepare_release(tmp_path / "release", run="final", bundle_manifest=bundle,
                              manifests_dir=manifests, weights_dir=weights)
    assert {f["path"] for f in release["files"]} == {
        "README.md", "adapter_config.json", "adapter_model.safetensors", "manifests/washer_done.jsonl"}
    assert release["skipped"] == ["README.md", "args.bin", "checkpoint-2"]
    with pytest.raises(ValueError, match="not a release folder"):
        prepare_release(manifests, run="final", bundle_manifest=bundle, manifests_dir=manifests)
    (manifests / "washer_done.jsonl").unlink()
    with pytest.raises(ValueError, match="te manifests"):
        prepare_release(tmp_path / "release", run="final", bundle_manifest=bundle,
                        manifests_dir=manifests)


def fake_hub(calls: list, existing_private=None):
    """A hub whose create_repo, like huggingface_hub's with exist_ok, leaves an existing
    repo's visibility alone."""
    def api(token):
        state = {}

        def create_repo(repo, **kw):
            calls.append((repo, kw))
            state["private"] = kw["private"] if existing_private is None else existing_private

        def upload_folder(**kw):
            calls.append(("upload", kw["repo_id"]))
            if token != "hf_good":
                raise OSError(f"401 for token {token}")
        return types.SimpleNamespace(
            upload_folder=upload_folder, create_repo=create_repo,
            repo_info=lambda repo, **kw: types.SimpleNamespace(private=state["private"]))
    return types.SimpleNamespace(HfApi=api)


def test_upload_needs_the_authors_yes_and_never_shows_the_token(tmp_path, monkeypatch):
    bundle, manifests, weights = inputs(tmp_path)
    folder = tmp_path / "release"
    prepare_release(folder, run="final", bundle_manifest=bundle, manifests_dir=manifests)
    with pytest.raises(PermissionError):
        upload(folder, REPO)
    with pytest.raises(ValueError, match="TODO"):
        upload(folder, REPO, confirmed=True)
    prepare_release(folder, run="final", bundle_manifest=bundle, manifests_dir=manifests,
                    license_id="apache-2.0")
    with pytest.raises(ValueError, match="no adapter weights"):  # a card alone is not a release
        upload(folder, REPO, confirmed=True)
    prepare_release(folder, run="final", bundle_manifest=bundle, manifests_dir=manifests,
                    license_id="apache-2.0", weights_dir=weights)
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub(calls))
    with pytest.raises(MissingSecret):
        upload(folder, REPO, confirmed=True)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    with pytest.raises(RuntimeError) as err:
        upload(folder, REPO, confirmed=True)
    assert "hf_secret_value" not in str(err.value) and "[redacted]" in str(err.value)
    assert calls == [(REPO, {"repo_type": "model", "private": True, "exist_ok": True}),
                     ("upload", REPO)]
    monkeypatch.setenv("HF_TOKEN", "hf_good")
    assert upload(folder, REPO, confirmed=True) == f"https://huggingface.co/{REPO}"
    (folder / "frame.png").write_bytes(b"")
    with pytest.raises(ValueError, match="never published"):
        upload(folder, REPO, confirmed=True)
    (folder / "frame.png").unlink()
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(RuntimeError, match="ask the author"):
        upload(folder, REPO, confirmed=True)


def test_upload_never_publishes_into_an_existing_public_repo(tmp_path, monkeypatch):
    bundle, manifests, weights = inputs(tmp_path)
    folder = tmp_path / "release"
    prepare_release(folder, run="final", bundle_manifest=bundle, manifests_dir=manifests,
                    license_id="apache-2.0", weights_dir=weights)
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub(calls, existing_private=False))
    monkeypatch.setenv("HF_TOKEN", "hf_good")
    with pytest.raises(ValueError, match="already exists and is public"):
        upload(folder, REPO, confirmed=True)
    assert ("upload", REPO) not in calls


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
