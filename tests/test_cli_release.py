"""CLI tests for M4's release commands (backend/cli_release.py). Fakes only: nothing is
sent, billed or uploaded."""

import io
import json
import shutil
import subprocess
from pathlib import Path

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


def test_playground_runs_on_localhost(env, monkeypatch, capsys):
    import uvicorn

    ran = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: ran.append(kw))
    assert cli.main(["playground"]) == 0
    assert ran[0]["host"] == "127.0.0.1" and ran[0]["port"] == 8100
    assert "live teaching on" in capsys.readouterr().out
    for allowed in ("*", "10.0.0.1, *"):
        assert cli.main(["playground", "--forwarded-allow-ips", allowed]) == 2
        assert "reset their spend cap" in capsys.readouterr().err
    assert len(ran) == 1


def test_playground_public_serves_the_public_copy(env, monkeypatch, capsys):
    import uvicorn
    from starlette.testclient import TestClient

    apps = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: apps.append(app))
    assert cli.main(["playground"]) == cli.main(["playground", "--public"]) == 0
    assert "public copy" in capsys.readouterr().out
    local, public = (TestClient(a).get("/api/status").json()["public"] for a in apps)
    assert (local, public) == (False, True)


def public_fixture(env):
    """A data folder with things the public copy may show and things it may not."""
    from tests.test_playground_data import make_clip
    from tests.test_teach import png

    data = env / "data"
    skill = data / "skills/washer_done"
    for sub in ("keyframes", "synthetic", "dataset", "cache"):
        (skill / sub).mkdir(parents=True)
    (skill / "spec.json").write_text("{}")
    (skill / "roi.json").write_text("[]")
    (skill / "keyframes/kf_00.png").write_bytes(png((200, 0, 0)))
    (skill / "synthetic/pos.png").write_bytes(png((0, 200, 0)))
    (skill / "synthetic/manifest.jsonl").write_text("{}\n")
    faces.blur_folder(skill / "keyframes", skill / "keyframes_blurred", detector=lambda b: [])
    (data / "cache").mkdir()
    (data / "cache/call.json").write_text("{}")
    (data / "ledger.jsonl").write_text("{}\n")
    make_clip(data, blurred=True)
    make_clip(data, "raw_clip")
    (env / "eval").mkdir()
    (env / "eval/f1.svg").write_text("<svg/>")
    return data


def test_playground_deploy_asks_first_and_copies_only_public_files(env, monkeypatch, capsys):
    from backend import cli_release

    public_fixture(env)
    monkeypatch.setattr(config, "PROJECT_ROOT", env)
    monkeypatch.setattr(cli, "modal_ready", lambda: True)
    commands, staged = [], set()

    def fake_run(command, **kw):
        commands.append((command, kw.get("env", {})))
        if command[1:3] == ["volume", "put"]:
            local = Path(command[-2])
            staged.update(f"{command[-1]}/{p.relative_to(local).as_posix()}"
                          for p in local.rglob("*") if p.is_file())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli_release.subprocess, "run", fake_run)
    answer(monkeypatch, "n")
    assert cli.main(["playground", "--deploy"]) == 1 and commands == []
    out = capsys.readouterr().out
    assert "nothing was copied" in out and "$10.00 in all" in out
    assert "no blurred keyframes in data/playground/keyframes" in out
    assert "clip raw_clip: no matching blur record" in out
    prompts = answer(monkeypatch, "y")
    assert cli.main(["playground", "--deploy", "--visitor-cap", "0.25"]) == 0
    assert "public URL" in prompts[0]
    assert [c[1:3] for c, _ in commands] == [["volume", "create"], ["volume", "put"],
                                             ["volume", "put"], ["deploy",
                                                                 "backend/playground/modal_app.py"]]
    assert staged == {
        "/data/skills/washer_done/spec.json", "/data/skills/washer_done/roi.json",
        "/data/skills/washer_done/synthetic/manifest.jsonl",
        "/data/skills/washer_done/keyframes_blurred/sources.json",
        "/data/skills/washer_done/keyframes_blurred/kf_00.png",
        "/data/playground/clips/washer_1/log.jsonl", "/data/playground/clips/washer_1/clip.mp4",
        "/data/playground/clips/washer_1/clip.mp4.blur.json",
        "/data/playground/clips/raw_clip/log.jsonl", "/eval/f1.svg"}
    deploy_env = commands[-1][1]
    assert (deploy_env["TE_PLAYGROUND_VISITOR_CAP"], deploy_env["TE_PLAYGROUND_DAILY_CAP"],
            deploy_env["TE_SKILL_CAP_USD"], deploy_env["TE_GLOBAL_CAP_USD"]) == (
        "0.25", "10.0", "10.0", "10.0")


def test_playground_deploy_over_15_dollars_asks_even_with_yes(env, monkeypatch, capsys):
    from backend import cli_release

    monkeypatch.setattr(cli_release.subprocess, "run", lambda *a, **kw: pytest.fail("ran"))
    prompts = answer(monkeypatch, "n")
    assert cli.main(["playground", "--deploy", "--total-cap", "40", "--yes"]) == 1
    assert len(prompts) == 1 and "even with --yes" in capsys.readouterr().out
    assert cli.main(["playground", "--deploy", "--daily-cap", "0", "--yes"]) == 2


def make_video(path, seconds=2):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc=size=64x48:rate=10:duration={seconds}", "-pix_fmt", "yuv420p",
                    str(path)], check=True)
    return path


def frame_log(path, last_t=1.9):
    rows = [{"t": round(i * 0.5, 3), "skill": "washer_done", "threshold": 0.6, "p": 0.1 * i}
            for i in range(int(last_t / 0.5) + 1)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_playground_clip_blurs_and_assembles_the_clip_folder(env, monkeypatch, capsys):
    from backend.playground import data

    seen = []
    monkeypatch.setattr(faces, "detect_faces", lambda b: seen.append(1) or [(0, 0, 8, 8)])
    clip, log = make_video(env / "session3_test_1.mp4"), frame_log(env / "replay.jsonl")
    labels = env / "session3_test_1.labels.json"
    labels.write_text(json.dumps({"clip": clip.name, "duration_s": 2.0, "true": [[0.5, 1.0]]}))
    args = ["playground-clip", "--clip", str(clip), "--log", str(log), "--labels", str(labels),
            "--name", "washer_1", "--fps", "10"]
    assert cli.main(args) == 0
    folder = env / "data/playground/clips/washer_1"
    assert sorted(p.name for p in folder.iterdir()) == ["clip.mp4", "clip.mp4.blur.json",
                                                        "labels.json", "log.jsonl"]
    record = json.loads((folder / "clip.mp4.blur.json").read_text())
    assert record["frames"] == len(seen) == 20 and record["faces"] == 20
    assert record["source_sha256"] == data.file_sha256(clip)
    assert data.blurred_video(folder) == "clip.mp4"
    assert data.clip_timeline(env / "data", "washer_1", public=True)["true_ranges"] == [[0.5, 1.0]]
    assert "20 faces blurred in 20 frames" in capsys.readouterr().out
    assert cli.main(args) == 2 and "already exists" in capsys.readouterr().err


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_playground_clip_refuses_bad_names_logs_labels_and_unrecorded_blurs(env, monkeypatch,
                                                                           capsys):
    from backend.playground import data

    monkeypatch.setattr(faces, "detect_faces", lambda b: [])
    clip, log = make_video(env / "test.mp4"), frame_log(env / "replay.jsonl")

    def run(*extra, name="kettle", log=log):
        return cli.main(["playground-clip", "--clip", str(clip), "--log", str(log),
                         f"--name={name}", "--fps", "10", *extra])

    for bad in ("../x", "a b", "-x", "", "x" * 81, ".hidden"):
        assert run(name=bad) == 2
    assert run(log=frame_log(env / "long.jsonl", last_t=30.0)) == 2
    assert "is it this clip's log?" in capsys.readouterr().err
    (env / "empty.jsonl").write_text('{"note": "no frames"}\n')
    assert run(log=env / "empty.jsonl") == 2
    late = env / "late.labels.json"
    late.write_text(json.dumps({"true": [[1.0, 9.0]]}))
    assert run("--labels", str(late)) == 2 and "are they this clip's?" in capsys.readouterr().err
    (env / "bad.labels.json").write_text(json.dumps({"true": [[3, 1]]}))
    assert run("--labels", str(env / "bad.labels.json")) == 2
    assert run("--blurred") == 2 and "no blur record" in capsys.readouterr().err
    (env / "test.mp4.blur.json").write_text(json.dumps({"output_sha256": "0" * 64}))
    assert run("--blurred") == 2  # a record for another file
    (env / "test.mp4.blur.json").write_text(json.dumps({"output_sha256": data.file_sha256(clip),
                                                         "frames": 20, "faces": 0}))
    assert run("--blurred") == 0 and "copied as blurred" in capsys.readouterr().out
    folder = env / "data/playground/clips/kettle"
    assert (folder / "clip.mp4").read_bytes() == clip.read_bytes()
    assert data.blurred_video(folder) == "clip.mp4" and not (folder / "labels.json").exists()
    monkeypatch.setattr(faces, "detect_faces", lambda b: (_ for _ in ()).throw(
        RuntimeError("face detection needs OpenCV")))
    assert run(name="no_opencv") == 2 and "needs OpenCV" in capsys.readouterr().err
    root = env / "data/playground/clips"
    assert sorted(p.name for p in root.iterdir()) == ["kettle"]  # no half-made folder left


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

    from tests.test_model_card import fake_hub
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

    from tests.test_model_card import fake_collections

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
