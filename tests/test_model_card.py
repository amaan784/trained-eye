"""The model card, the release folder and the upload guard; huggingface_hub is a fake."""

import json
import sys
import types

import pytest

from backend.config import STUDENT_BASE_REVISION, MissingSecret
from backend.meta.cache import sha256_hex
from backend.release.faces import blur_folder
from backend.release.model_card import (
    USAGE_POLICY,
    make_collection,
    model_card,
    prepare_release,
    upload,
)
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


def test_the_release_folder_holds_the_card_weights_and_manifests(tmp_path):
    bundle, manifests, weights = inputs(tmp_path)
    release = prepare_release(tmp_path / "release", run="final", bundle_manifest=bundle,
                              manifests_dir=manifests, weights_dir=weights)
    assert {f["path"] for f in release["files"]} == {
        "README.md", "LICENSE", "adapter_config.json", "adapter_model.safetensors",
        "manifests/washer_done.jsonl"}
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
    assert (folder / "LICENSE").read_text().lstrip().startswith("Apache License")
    (folder / "LICENSE").unlink()
    with pytest.raises(ValueError, match="no LICENSE"):  # the base model's license must ship
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


def keyframe_inputs(tmp_path):
    """A mixed bundle whose washer_done manifest starts from two keyframes, the te blur
    folder of those keyframes, and a generated image."""
    from tests.test_faces import noise

    bundle, manifests, weights = inputs(tmp_path)
    originals = tmp_path / "keyframes"
    originals.mkdir()
    (originals / "kf_00.png").write_bytes(noise(size=(40, 30)))
    (originals / "kf_01.jpg").write_bytes(noise("JPEG", size=(40, 30)))
    generated = noise(size=(41, 30))
    rows = [{"id": f"pos-{i}", "keyframe_sha256": sha256_hex(p.read_bytes()),
             "image_sha256": sha256_hex(generated) if i == 0 else "e" * 64}
            for i, p in enumerate(sorted(originals.iterdir()))]
    (manifests / "washer_done.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    blurred = tmp_path / "blurred"
    blur_folder(originals, blurred, detector=lambda data: [(0, 0, 8, 8)])  # a face in each
    return bundle, manifests, weights, blurred, generated


def test_the_release_ships_the_blurred_keyframes_te_regenerate_needs(tmp_path, monkeypatch):
    from backend.regenerate import find_keyframes

    bundle, manifests, weights, blurred, generated = keyframe_inputs(tmp_path)
    (blurred / "notes.png").write_bytes(b"not listed in sources.json")
    folder = tmp_path / "release"
    release = prepare_release(folder, run="final", bundle_manifest=bundle,
                              manifests_dir=manifests, weights_dir=weights,
                              keyframes={"washer_done": blurred})
    shipped = folder / "keyframes" / "washer_done"
    assert sorted(p.name for p in shipped.iterdir()) == ["kf_00.png", "kf_01.jpg",
                                                         "sources.json"]
    assert release["keyframes"] == {"washer_done": 2}
    assert "keyframes/washer_done/kf_00.png" in {f["path"] for f in release["files"]}
    needed = {json.loads(line)["keyframe_sha256"]
              for line in (folder / "manifests/washer_done.jsonl").read_text().splitlines()}
    found, mapped = find_keyframes(shipped, needed)  # what the card's te regenerate reads
    assert set(found) == needed and mapped == 2
    card = (folder / "README.md").read_text()
    assert ("te regenerate --skill <skill> --manifest manifests/<skill>.jsonl --keyframes "
            "keyframes/<skill>") in card and "blurred pixels" in card
    assert "(for washer_done)" in card and "which are not published" not in card
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub(calls))
    monkeypatch.setenv("HF_TOKEN", "hf_good")
    assert upload(folder, REPO, confirmed=True) == f"https://huggingface.co/{REPO}"
    (shipped / "kf_02.png").write_bytes(generated)  # a generated image slipped in, listed
    sources = json.loads((shipped / "sources.json").read_text())
    (shipped / "sources.json").write_text(json.dumps({**sources, "f" * 64: "kf_02.png"}))
    with pytest.raises(ValueError, match="never published"):
        upload(folder, REPO, confirmed=True)
    (shipped / "sources.json").write_text(json.dumps(sources))  # listed no more
    with pytest.raises(ValueError, match="kf_02.png"):
        upload(folder, REPO, confirmed=True)
    (shipped / "kf_02.png").unlink()
    (folder / "keyframes" / "bus_m4").mkdir()  # no manifest, so no keyframes may ship
    (folder / "keyframes" / "bus_m4" / "kf_00.png").write_bytes(b"")
    with pytest.raises(ValueError, match="bus_m4/kf_00.png"):
        upload(folder, REPO, confirmed=True)
    assert [c for c in calls if c[0] == "upload"] == [("upload", REPO)]


def test_keyframes_are_refused_unless_te_blur_made_all_of_them_from_real_frames(tmp_path):
    bundle, manifests, weights, blurred, generated = keyframe_inputs(tmp_path)
    folder = tmp_path / "release"

    def release(**keyframes):
        return prepare_release(folder, run="final", bundle_manifest=bundle,
                               manifests_dir=manifests, weights_dir=weights,
                               keyframes=keyframes)

    with pytest.raises(ValueError, match="no generated images"):
        release(bus_m4=blurred)
    with pytest.raises(ValueError, match="no sources.json"):
        release(washer_done=tmp_path / "keyframes")  # the originals, not a te blur folder
    synthetic = tmp_path / "synthetic"
    synthetic.mkdir()
    (synthetic / "pos.png").write_bytes(generated)
    blur_folder(synthetic, tmp_path / "blurred_synthetic", detector=lambda data: [])
    with pytest.raises(ValueError, match="generated image"):
        release(washer_done=tmp_path / "blurred_synthetic")
    (blurred / "kf_01.jpg").unlink()
    with pytest.raises(ValueError, match="misses 1 of the 2 keyframes"):
        release(washer_done=blurred)
    assert not folder.exists()  # refused before anything was written
    release()
    card = (folder / "README.md").read_text()
    assert "For washer_done, `te regenerate` needs the original keyframes" in card


def fake_collections(calls: list, repos: dict, existing_private=None):
    """A hub with repo_info for `repos` ({(type, id): private}) and collections."""
    def api(token):
        def repo_info(repo, repo_type):
            calls.append(("info", repo_type, repo))
            if (repo_type, repo) not in repos:
                raise OSError(f"404 {repo} with token {token}")
            return types.SimpleNamespace(private=repos[repo_type, repo])

        def create_collection(title, **kw):
            calls.append(("create", title, kw))
            private = kw["private"] if existing_private is None else existing_private
            return types.SimpleNamespace(slug="someone/trained-eye-0123", private=private)

        def add_collection_item(slug, **kw):
            calls.append(("add", slug, kw))
        return types.SimpleNamespace(repo_info=repo_info, create_collection=create_collection,
                                     add_collection_item=add_collection_item)
    return types.SimpleNamespace(HfApi=api)


def test_a_collection_groups_existing_repos_of_the_asked_visibility(tmp_path, monkeypatch):
    repos = {("model", REPO): True, ("dataset", "someone/trained-eye-test"): True,
             ("model", "someone/public-adapter"): False}
    wanted = [REPO, "datasets/someone/trained-eye-test"]
    with pytest.raises(PermissionError):
        make_collection("Trained Eye", wanted)
    for bad in (["someone"], ["datasets/x"], [REPO, REPO], []):
        with pytest.raises(ValueError):
            make_collection("Trained Eye", bad, confirmed=True)
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_collections(calls, repos))
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    url = make_collection("Trained Eye", wanted, confirmed=True)
    assert url == "https://huggingface.co/collections/someone/trained-eye-0123"
    assert calls == [
        ("info", "model", REPO), ("info", "dataset", "someone/trained-eye-test"),
        ("create", "Trained Eye", {"namespace": None, "description": None, "private": True,
                                   "exists_ok": True}),
        ("add", "someone/trained-eye-0123", {"item_id": REPO, "item_type": "model",
                                             "exists_ok": True}),
        ("add", "someone/trained-eye-0123", {"item_id": "someone/trained-eye-test",
                                             "item_type": "dataset", "exists_ok": True})]
    for repos_asked, match in (([REPO, "someone/missing"], "missing or not reachable"),
                               ([REPO, "someone/public-adapter"], "is public, not private")):
        calls.clear()
        with pytest.raises(ValueError, match=match) as err:
            make_collection("Trained Eye", repos_asked, confirmed=True)
        assert "hf_secret_value" not in str(err.value)
        assert not [c for c in calls if c[0] in ("create", "add")]  # nothing was made
    with pytest.raises(ValueError, match="is private, not public"):
        make_collection("Trained Eye", [REPO], confirmed=True, private=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        fake_collections(calls, repos, existing_private=False))
    calls.clear()
    with pytest.raises(ValueError, match="already exists and is public"):
        make_collection("Trained Eye", [REPO], confirmed=True)
    assert not [c for c in calls if c[0] == "add"]
