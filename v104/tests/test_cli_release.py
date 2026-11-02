"""CLI tests for M4's release commands (backend/cli_release.py). Fakes only: nothing is
sent, billed or uploaded."""

import io
import json

import pytest
from PIL import Image

from backend import cli, config
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
