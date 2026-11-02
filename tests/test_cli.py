"""CLI tests for the commands that talk to the Meta API. The client is swapped for one
backed by httpx.MockTransport, so nothing is sent or billed."""

import httpx
import pytest

from backend import cli
from backend.meta.client import MetaClient


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_API_KEY", "sk-test-cli-key")
    return tmp_path


def use_transport(monkeypatch, handler):
    calls = []

    def recording(request):
        calls.append(request)
        return handler(request)

    monkeypatch.setattr(
        cli, "make_client",
        lambda settings: MetaClient(settings, transport=httpx.MockTransport(recording)),
    )
    return calls


def test_api_smoke_then_cache_hit(env, monkeypatch, capsys):
    reply = {"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 40, "completion_tokens": 30}}
    calls = use_transport(monkeypatch, lambda r: httpx.Response(200, json=reply))
    assert cli.main(["api-smoke"]) == 0
    out = capsys.readouterr().out
    assert "one structured call to muse-spark-1.3" in out
    assert 'result: {"ok": true}' in out and "cost: $0.000178" in out
    assert cli.main(["api-smoke"]) == 0
    assert "cache hit" in capsys.readouterr().out
    assert len(calls) == 1
    assert cli.main(["spend"]) == 0
    assert "smoke" in capsys.readouterr().out


def test_models_reports_missing(env, monkeypatch, capsys):
    offered = {"data": [{"id": m} for m in
                        ("muse-spark-1.3", "muse-wonder-1.0", "sam-3.1", "muse-image-1.0")]}
    use_transport(monkeypatch, lambda r: httpx.Response(200, json=offered))
    assert cli.main(["models"]) == 1
    out = capsys.readouterr().out
    assert "configured models missing for this key: muse-voice-transcribe-1.0" in out


def test_missing_key_is_a_clean_error(env, monkeypatch, capsys):
    monkeypatch.delenv("MODEL_API_KEY")
    use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    assert cli.main(["api-smoke"]) == 2
    err = capsys.readouterr().err
    assert "MODEL_API_KEY is not set" in err


def test_api_error_is_a_clean_error(env, monkeypatch, capsys):
    use_transport(monkeypatch, lambda r: httpx.Response(401, json={"error": "invalid key"}))
    assert cli.main(["models"]) == 2
    err = capsys.readouterr().err
    assert "401" in err and "sk-test-cli-key" not in err


def write_clip(folder, n=12):
    from backend.cli import synthetic_jpeg

    folder.mkdir()
    for i in range(n):
        (folder / f"f{i:03d}.jpg").write_bytes(synthetic_jpeg(i, size=(320, 240)))
    return folder


def test_teach_estimate_only_makes_no_call(env, monkeypatch, capsys):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    clip = write_clip(env / "clip")
    code = cli.main(["teach", "--skill", "washer_done", "--clip", str(clip), "--request",
                     "let me know when the washer says done", "--estimate-only"])
    assert code == 0 and calls == []
    out = capsys.readouterr().out
    assert out.startswith("estimate for washer_done") and "images $3.00" in out


def test_teach_refused_request_stops_for_free(env, monkeypatch, capsys):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    clip = write_clip(env / "clip")
    code = cli.main(["teach", "--skill", "sam_arrives", "--clip", str(clip), "--request",
                     "Tell me when Sam gets here", "--yes"])
    assert code == 1 and calls == []
    out = capsys.readouterr().out
    assert "I can't learn that one." in out and "stopped: refused" in out


def test_teach_rejects_a_bad_skill_name_before_anything(env, monkeypatch, capsys):
    calls = use_transport(monkeypatch, lambda r: pytest.fail("no request should be sent"))
    clip = write_clip(env / "clip")
    code = cli.main(["teach", "--skill", "washer-done", "--clip", str(clip), "--request",
                     "let me know when the washer says done", "--yes"])
    assert code == 2 and calls == []
    assert "snake_case" in capsys.readouterr().err
    assert not (env / "data" / "skills").exists()
