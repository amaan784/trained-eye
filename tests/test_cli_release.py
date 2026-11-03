"""CLI tests for M4's release commands (backend/cli_release.py). Fakes only: nothing is
sent, billed or uploaded."""

import io
import json

import pytest
from PIL import Image

from backend import cli, config
from backend.release import faces
from tests.test_cli import answer, use_transport


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_API_KEY", "sk-test-cli-key")
    return tmp_path


def manifest_rows(skill="washer_done"):
    return [{"id": f"{skill}-pos-{i:03d}", "skill": skill, "label": "yes", "kind": "positive",
             "condition": "the display shows End", "nuisance": "in the same lighting",
             "keyframe_sha256": "a" * 64, "prompt": f"Edit this photo {i}.", "file": f"{i}.png",
             "image_sha256": "b" * 64, "model": "muse-image-1.0", "cache_key": "c" * 64,
             "cost_usd": 0.01, "cached": False} for i in range(3)]


def test_manifests_copies_prompts_out_of_data_and_never_images(env, capsys):
    assert cli.main(["manifests"]) == 1  # nothing taught yet
    synthetic = env / "data/skills/washer_done/synthetic"
    synthetic.mkdir(parents=True)
    (synthetic / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in manifest_rows()))
    assert cli.main(["manifests"]) == 0
    out = config.PROJECT_ROOT / "manifests/washer_done.jsonl"
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 3 and all("file" not in r and "cost_usd" not in r for r in rows)
    assert "3 prompts" in capsys.readouterr().out


def test_regenerate_prints_the_estimate_and_spends_nothing_without_a_yes(env, monkeypatch,
                                                                         capsys):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    keyframes = env / "data/skills/washer_done/keyframes"
    keyframes.mkdir(parents=True)
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (10, 20, 30)).save(buf, format="JPEG")
    (keyframes / "kf_00_00000000.jpg").write_bytes(buf.getvalue())
    import hashlib

    sha = hashlib.sha256(buf.getvalue()).hexdigest()
    manifest = env / "washer_done.jsonl"
    manifest.write_text("".join(json.dumps({**r, "keyframe_sha256": sha}) + "\n"
                                for r in manifest_rows()))
    assert cli.main(["regenerate", "--skill", "washer_done", "--manifest", str(manifest),
                     "--estimate-only"]) == 0
    assert "3 images to regenerate" in capsys.readouterr().out
    answer(monkeypatch, "n")
    assert cli.main(["regenerate", "--skill", "washer_done", "--manifest", str(manifest)]) == 1
    capsys.readouterr()
    monkeypatch.setattr(config, "ASK_ABOVE_USD", 0.001)  # these 3 images now count as over $15
    prompts = answer(monkeypatch, "n")
    assert cli.main(["regenerate", "--skill", "washer_done", "--manifest", str(manifest),
                     "--yes"]) == 1 and len(prompts) == 1
    assert "needs your yes even with --yes" in capsys.readouterr().out
    assert calls == []
    assert cli.main(["regenerate", "--skill", "Bad-Name"]) == 2


def test_blur_and_release_report_clean_errors(env, capsys):
    assert cli.main(["blur", "--src", str(env / "nope"), "--dst", str(env / "out")]) == 2
    assert cli.main(["release", "--run", "final", "--out", str(env / "release")]) == 2
    assert capsys.readouterr().err.count("te: ") == 2


def test_blur_adds_manual_boxes_and_records_a_video(env, monkeypatch, capsys):
    from tests.test_faces import fake_ffmpeg, noise

    monkeypatch.setattr(faces, "detect_faces", lambda data: [])
    src = env / "keyframes"
    src.mkdir()
    (src / "kf_00.png").write_bytes(noise())
    boxes = env / "boxes.json"
    boxes.write_text(json.dumps({"kf_00.png": [[0, 0, 10, 10]]}))
    assert cli.main(["blur", "--src", str(src), "--dst", str(env / "out"), "--boxes",
                     str(boxes)]) == 0
    assert "0 faces found and 1 manual boxes blurred" in capsys.readouterr().out
    boxes.write_text(json.dumps({"kf_99.png": [[0, 0, 10, 10]]}))  # a typo
    assert cli.main(["blur", "--src", str(src), "--dst", str(env / "out2"), "--boxes",
                     str(boxes)]) == 2
    assert "not images" in capsys.readouterr().err and not (env / "out2").exists()
    fake_ffmpeg(monkeypatch)
    (env / "clip.mp4").write_bytes(b"video")
    boxes.write_text(json.dumps([{"start_s": 0.0, "end_s": 0.5, "box": [0, 0, 8, 8]}]))
    args = ["blur", "--src", str(env / "clip.mp4"), "--dst", str(env / "blurred.mp4"),
            "--fps", "10", "--boxes", str(boxes)]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "1 manual boxes on 6 frames" in out and "blurred.mp4.blur.json" in out
    record = json.loads((env / "blurred.mp4.blur.json").read_text())
    assert record["manual_boxes"] == [{"start_s": 0.0, "end_s": 0.5, "box": [0, 0, 8, 8]}]
    boxes.write_text(json.dumps({"kf_00.png": []}))  # a folder's boxes for a video
    assert cli.main(args) == 2 and "boxes for a video" in capsys.readouterr().err


def release_inputs(env, run="final"):
    """A training bundle manifest for `run` with generated images, and its prompt manifest."""
    bundle = env / "data" / "train" / run
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({
        "variant": "mixed", "train_examples": 3, "val_examples": 0,
        "skills": {"washer_done": ""}, "counts": {"train/washer_done/real/yes": 1,
                                                  "train/washer_done/synthetic/no": 2}}))
    (env / "manifests").mkdir()
    (env / "manifests" / "washer_done.jsonl").write_text('{"id": "x"}\n')


def test_fetch_weights_feeds_te_release_by_default(env, monkeypatch, capsys):
    import sys
    import types

    from tests.test_release_weights import ADAPTER, FakeVolume

    monkeypatch.setattr(config, "PROJECT_ROOT", env)
    release_inputs(env)
    assert cli.main(["release", "--run", "final", "--out", str(env / "rel")]) == 0
    assert "no weights at data/weights/final/adapter" in capsys.readouterr().out
    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(Volume=types.SimpleNamespace(
        from_name=lambda name: FakeVolume(ADAPTER))))
    assert cli.main(["fetch-weights", "--run", "final"]) == 0
    out = capsys.readouterr().out
    assert "3 files" in out and "data/weights/final/adapter" in out and "checkpoint-10" in out
    assert cli.main(["fetch-weights", "--run", "final"]) == 2  # never over an earlier fetch
    assert "already has files" in capsys.readouterr().err
    assert cli.main(["release", "--run", "final", "--out", str(env / "rel")]) == 0
    release = json.loads((env / "rel" / "release.json").read_text())
    assert {"adapter_config.json", "adapter_model.safetensors"} <= {
        f["path"] for f in release["files"]}
    assert cli.main(["release", "--run", "final", "--out", str(env / "rel"), "--keyframes",
                     "washer_done"]) == 2
    assert "<skill>=<folder>" in capsys.readouterr().err


def test_release_test_set_asks_before_a_private_dataset_upload(env, monkeypatch, capsys):
    import sys

    from tests.test_release_package import fake_hub
    from tests.test_test_set import footage

    _, blurred = footage(env, monkeypatch)  # the labels are in data/clips, the default
    monkeypatch.setattr("backend.release.test_set.video_duration_s", lambda path: 120.0)
    base = ["release-test-set", "--clips", f"washer_done={blurred}", "--out", str(env / "ts")]
    assert cli.main(base) == 2 and "no default" in capsys.readouterr().err
    assert cli.main([*base, "--license", "cc-by-4.0", "--upload"]) == 2
    assert "--upload needs --repo" in capsys.readouterr().err
    prompts = answer(monkeypatch, "n")
    upload = [*base, "--license", "cc-by-4.0", "--upload", "--repo", "someone/te-test"]
    assert cli.main(upload) == 1
    out = capsys.readouterr().out
    assert "left out washer_done/session1_kitchen.mp4" in out and "nothing was uploaded" in out
    assert "datasets/someone/te-test (private)" in prompts[0]
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub(calls))
    monkeypatch.setenv("HF_TOKEN", "hf_good")
    answer(monkeypatch, "y")
    assert cli.main(upload) == 0
    assert "huggingface.co/datasets/someone/te-test" in capsys.readouterr().out
    assert calls[0][1]["repo_type"] == "dataset" and calls[0][1]["private"] is True


def test_release_collection_asks_first(env, monkeypatch, capsys):
    import sys

    from tests.test_release_package import fake_collections

    calls = []
    repos = {("model", "someone/te"): True, ("dataset", "someone/te-test"): True}
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_collections(calls, repos))
    monkeypatch.setenv("HF_TOKEN", "hf_good")
    args = ["release-collection", "--title", "Trained Eye", "--repos", "someone/te",
            "datasets/someone/te-test"]
    prompts = answer(monkeypatch, "n")
    assert cli.main(args) == 1 and calls == [] and "private" in prompts[0]
    assert "someone/te-test (dataset)" in capsys.readouterr().out
    assert cli.main([*args, "--yes"]) == 0
    assert "collections/someone/trained-eye-0123" in capsys.readouterr().out
    assert cli.main([*args, "--public", "--yes"]) == 2  # the repos are private
    assert "is private, not public" in capsys.readouterr().err
    assert cli.main(["release-collection", "--title", "x", "--repos", "nope"]) == 2


def test_wandb_report_asks_before_saving_a_draft(env, monkeypatch, capsys):
    import sys

    from backend.eval import wandb_report
    from backend.eval.writeup import render
    from tests.test_wandb_report import fake_reports, results

    monkeypatch.setattr(config, "PROJECT_ROOT", env)
    assert cli.main(["wandb-report"]) == 2 and "make eval" in capsys.readouterr().err
    (env / "eval").mkdir()
    (env / "eval/results.json").write_text(json.dumps(results(wandb="not logged")))
    (env / "eval/results.md").write_text(render(results()))
    assert cli.main(["wandb-report"]) == 2 and "te eval --wandb" in capsys.readouterr().err
    (env / "eval/results.json").write_text(json.dumps(results()))
    saved = []
    monkeypatch.setitem(sys.modules, wandb_report.MODULE, fake_reports(saved))
    answer(monkeypatch, "n")
    assert cli.main(["wandb-report"]) == 1 and saved == []
    assert "someone/trained-eye: 4 sections" in capsys.readouterr().out
    assert cli.main(["wandb-report", "--yes", "--entity", "team"]) == 0
    assert saved[0][0]["entity"] == "team" and saved[0][1] is True
    assert "draft report: https://wandb.ai/" in capsys.readouterr().out
