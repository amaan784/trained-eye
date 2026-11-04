"""CLI tests for M4's playground commands (backend/cli_playground.py): te playground, its
--deploy and te playground-clip. Fakes only: nothing is sent, billed, uploaded or deployed."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from backend import cli, config
from backend.release import faces
from tests.test_cli import answer


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_API_KEY", "sk-test-cli-key")
    return tmp_path


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
    skill = data / "skills/burner_left_on"
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
    from backend import cli_playground

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

    monkeypatch.setattr(cli_playground.subprocess, "run", fake_run)
    answer(monkeypatch, "n")
    assert cli.main(["playground", "--deploy"]) == 1 and commands == []
    out = capsys.readouterr().out
    assert "nothing was copied" in out and "$10.00 in all" in out
    assert "no blurred keyframes in data/playground/keyframes" in out
    assert "clip raw_clip: no matching blur record" in out
    prompts = answer(monkeypatch, "y")
    assert cli.main(["playground", "--deploy", "--visitor-cap", "0.25"]) == 0
    assert "public URL" in prompts[0]
    assert [c[1:3] for c, _ in commands] == [
        ["volume", "create"], *[["volume", "rm"]] * 4, ["volume", "put"], ["volume", "put"],
        ["deploy", "backend/playground/modal_app.py"]]
    removed = [c[-1] for c, _ in commands if c[1:3] == ["volume", "rm"]]
    assert removed == list(cli_playground.PUBLISHED_TREES)  # not the budget, ledger or cache
    assert "/data/playground/budget.json" not in removed and "/data" not in removed
    assert staged == {
        "/data/skills/burner_left_on/spec.json", "/data/skills/burner_left_on/roi.json",
        "/data/skills/burner_left_on/synthetic/manifest.jsonl",
        "/data/skills/burner_left_on/keyframes_blurred/sources.json",
        "/data/skills/burner_left_on/keyframes_blurred/kf_00.png",
        "/data/playground/clips/burner_1/log.jsonl", "/data/playground/clips/burner_1/clip.mp4",
        "/data/playground/clips/burner_1/clip.mp4.blur.json",
        "/data/playground/clips/raw_clip/log.jsonl", "/eval/f1.svg"}
    deploy_env = commands[-1][1]
    assert (deploy_env["TE_PLAYGROUND_VISITOR_CAP"], deploy_env["TE_PLAYGROUND_DAILY_CAP"],
            deploy_env["TE_SKILL_CAP_USD"], deploy_env["TE_GLOBAL_CAP_USD"]) == (
        "0.25", "10.0", "10.0", "10.0")


def test_playground_deploy_over_15_dollars_asks_even_with_yes(env, monkeypatch, capsys):
    from backend import cli_playground

    monkeypatch.setattr(cli_playground.subprocess, "run", lambda *a, **kw: pytest.fail("ran"))
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
    rows = [{"t": round(i * 0.5, 3), "skill": "burner_left_on", "threshold": 0.6, "p": 0.1 * i}
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
            "--name", "burner_1", "--fps", "10"]
    assert cli.main(args) == 0
    folder = env / "data/playground/clips/burner_1"
    assert sorted(p.name for p in folder.iterdir()) == ["clip.mp4", "clip.mp4.blur.json",
                                                        "labels.json", "log.jsonl"]
    record = json.loads((folder / "clip.mp4.blur.json").read_text())
    assert record["frames"] == len(seen) == 20 and record["faces"] == 20
    assert record["source_sha256"] == data.file_sha256(clip)
    assert data.blurred_video(folder) == "clip.mp4"
    assert data.clip_timeline(env / "data", "burner_1", public=True)["true_ranges"] == [[0.5, 1.0]]
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
