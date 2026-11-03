"""The teach API from the phone's side (backend/teach_api.py), through Starlette's
TestClient and the fake Meta API of test_teach, so nothing is sent or billed. The run is
in a background thread: the tests poll /teach/status until a state, with a deadline."""

import io
import json
import time
from pathlib import Path

import httpx
import pytest
from PIL import Image
from starlette.applications import Starlette
from starlette.testclient import TestClient

from backend.config import load_settings
from backend.meta.client import MetaClient
from backend.teach_api import teach_routes
from tests.test_teach import FakeMeta, clip, wav

REQUEST = "let me know when the washer says done"
TOKEN = "t0ken-for-tests"


def ended(status):
    return status["state"] in ("done", "stopped", "failed")


@pytest.fixture
def fake():
    return FakeMeta()


@pytest.fixture
def settings(tmp_path):
    return load_settings(env={"TE_DATA_DIR": str(tmp_path / "data"), "MODEL_API_KEY": "sk-test",
                              "SERVER_TOKEN": TOKEN}, env_file=None)


def meta_for(fake):
    return lambda s: MetaClient(s, transport=httpx.MockTransport(fake), sleep=lambda _: None)


@pytest.fixture
def api(fake, settings):
    app = Starlette(routes=teach_routes(lambda: settings, meta_for(fake)))
    with TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}) as tc:
        yield tc


@pytest.fixture
def frames(tmp_path):
    return [p.read_bytes() for p in sorted(clip(tmp_path / "clip").iterdir())]


@pytest.fixture
def audio(tmp_path):
    return wav(tmp_path / "request.wav").read_bytes()


def start(api, frames, skill="washer_done", audio=None, **fields):
    files = [("frames", (f"{i}.jpg", data, "image/jpeg")) for i, data in enumerate(frames)]
    if audio:
        files.append(("audio", ("request.wav", audio, "audio/wav")))
    fields = {"skill": skill, "positives": 6, "negatives": 6, **fields}
    return api.post("/teach/start", data={k: str(v) for k, v in fields.items()}, files=files)


def answer(api, step, yes, **extra):
    return api.post("/teach/answer", json={"skill": "washer_done", "step": step, "yes": yes,
                                           **extra})


def wait(api, until, skill="washer_done", timeout=30.0):
    deadline = time.monotonic() + timeout
    while True:
        status = api.get("/teach/status", params={"skill": skill}).json()
        if until(status):
            return status
        assert time.monotonic() < deadline, status
        time.sleep(0.01)


def test_a_full_run_from_the_phone(api, fake, settings, frames, audio):
    started = start(api, frames, audio=audio)
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["state"] == "waiting_spend" and body["question"] == "spend"
    assert 0 < body["estimate"]["total"] < body["estimate"]["at_most"] < 15
    assert fake.calls == {}  # nothing is paid before the spend is agreed
    capture = Path(body["capture"]).resolve()
    assert capture.parent == (settings.data_dir / "teach" / "washer_done").resolve()
    assert sorted(p.name for p in capture.iterdir()) == [
        *(f"f{i:03d}.jpg" for i in range(len(frames))), "request.wav"]
    assert (capture / "f000.jpg").read_bytes() == frames[0]

    assert answer(api, "spend", True).json()["state"] == "running"
    asked = wait(api, lambda s: s["state"] == "waiting_read_back")
    assert asked["question"] == "read_back" and asked["read_back"].startswith("I'll watch")
    assert fake.calls["/audio/transcriptions"] == 1 and "/images/edits" not in fake.calls
    assert answer(api, "read_back", True).status_code == 200
    done = wait(api, ended)
    assert done["state"] == "done", done
    summary = done["summary"]
    assert summary["yes"] and summary["no"] and summary["generated"] == fake.edits == 12
    assert (settings.data_dir / "skills/washer_done/dataset/washer_done-teach.jsonl").is_file()
    assert any(line.startswith("dataset:") for line in done["lines"])
    assert done["spent_usd"] == pytest.approx(MetaClient(settings).ledger.spent(), abs=1e-4)
    assert done["spent_usd"] > 0 and done["stopped"] is None and done["error"] is None


def test_no_to_the_spend_question_spends_nothing(api, fake, settings, frames, audio):
    assert start(api, frames, audio=audio).status_code == 200
    assert answer(api, "spend", False).status_code == 200
    stopped = wait(api, ended)
    assert stopped["state"] == "stopped" and "nothing was spent" in stopped["stopped"]
    assert fake.calls == {} and MetaClient(settings).ledger.entries() == []


def test_a_correction_at_the_read_back_compiles_again(api, fake, settings, frames, audio):
    start(api, frames, audio=audio)
    answer(api, "spend", True)
    wait(api, lambda s: s["state"] == "waiting_read_back")
    assert answer(api, "read_back", False, correction="the light  on the lid").status_code == 200
    again = wait(api, lambda s: s["state"] == "waiting_read_back"
                 and 'correction: "the light on the lid"' in s["lines"])
    assert again["read_back"].startswith("I'll watch")
    assert fake.calls["/chat/completions"] == 2 and fake.calls["/audio/transcriptions"] == 1
    answer(api, "read_back", False)
    stopped = wait(api, ended)
    assert stopped["state"] == "stopped" and stopped["stopped"] == "not confirmed"
    assert "/images/edits" not in fake.calls
    request = json.loads((settings.data_dir / "skills/washer_done/request.json").read_text())
    assert request["correction"] == "the light on the lid"


def test_one_session_at_a_time_and_answers_must_fit(api, frames):
    assert start(api, frames, request=REQUEST).status_code == 200
    second = start(api, frames, skill="pot_boils", request="tell me when the pot boils over")
    assert second.status_code == 409 and "washer_done" in second.json()["error"]
    wrong = answer(api, "read_back", True)
    assert wrong.status_code == 409 and "waiting_spend" in wrong.json()["error"]
    assert answer(api, "spend", True, correction="the lid").status_code == 400
    assert api.post("/teach/answer", content=b"yes").status_code == 400
    assert api.post("/teach/answer", json={"skill": "pot_boils", "step": "spend",
                                           "yes": True}).status_code == 404
    answer(api, "spend", False)
    wait(api, ended)
    assert start(api, frames, request=REQUEST).status_code == 200  # free again
    assert api.get("/teach/status").json()["question"] == "spend"  # the latest session
    answer(api, "spend", False)
    wait(api, ended)


def test_a_run_over_the_ask_limit_is_refused(api, fake, frames):
    refused = start(api, frames, request=REQUEST, positives=600, negatives=600)
    assert refused.status_code == 422
    body = refused.json()
    assert body["estimate"]["at_most"] > 15
    assert "te teach --skill washer_done --clip " in body["error"]
    assert "--positives 600 --negatives 600" in body["error"]
    assert api.get("/teach/status", params={"skill": "washer_done"}).status_code == 404
    assert fake.calls == {}


@pytest.mark.parametrize("case, status, message", [
    ("bad name", 400, "must be snake_case"),
    ("no request", 400, "send the spoken request as audio"),
    ("no frames", 400, "JPEG frames"),
    ("not multipart", 400, "multipart/form-data"),
    ("policy", 422, "I can't learn that one."),
])
def test_bad_inputs_are_refused_before_anything_is_saved(api, fake, settings, frames, audio,
                                                         case, status, message):
    reply = {
        "bad name": lambda: start(api, frames, skill="Washer Done", request=REQUEST),
        "no request": lambda: start(api, frames),
        "no frames": lambda: start(api, [], audio=audio),
        "not multipart": lambda: api.post("/teach/start", json={"skill": "washer_done"}),
        "policy": lambda: start(api, frames, request="tell me when Sam gets here"),
    }[case]()
    assert reply.status_code == status and message in reply.json()["error"], reply.text
    assert not (settings.data_dir / "teach").exists() and fake.calls == {}


def test_the_teach_routes_need_the_server_token(fake, settings, frames):
    app = Starlette(routes=teach_routes(lambda: settings, meta_for(fake)))
    with TestClient(app) as bare:
        for reply in (bare.get("/teach/status"), start(bare, frames, request=REQUEST),
                      bare.post("/teach/answer", json={"skill": "washer_done"}),
                      bare.get("/teach/status", headers={"Authorization": "Bearer nope"})):
            assert reply.status_code == 401 and "SERVER_TOKEN" in reply.json()["error"]
    off = load_settings(env={"TE_DATA_DIR": str(settings.data_dir), "MODEL_API_KEY": "sk-test"},
                        env_file=None)
    app = Starlette(routes=teach_routes(lambda: off, meta_for(fake)))
    with TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}) as tc:
        reply = start(tc, frames, request=REQUEST)
        assert reply.status_code == 503 and "set SERVER_TOKEN in .env" in reply.json()["error"]
    spaced = load_settings(env={"TE_DATA_DIR": str(settings.data_dir), "MODEL_API_KEY": "sk-test",
                                "SERVER_TOKEN": "has a space"}, env_file=None)
    app = Starlette(routes=teach_routes(lambda: spaced, meta_for(fake)))
    with TestClient(app, headers={"Authorization": "Bearer has a space"}) as tc:
        reply = tc.get("/teach/status")  # a 503 that says what to fix, not a 500
        assert reply.status_code == 503 and "SERVER_TOKEN has spaces" in reply.json()["error"]
    assert fake.calls == {} and not (settings.data_dir / "teach").exists()


def test_odd_answers_are_refused(api, frames):
    assert start(api, frames, request=REQUEST).status_code == 200
    for step in (["spend"], {"a": 1}):
        reply = api.post("/teach/answer", json={"skill": "washer_done", "step": step, "yes": True})
        assert reply.status_code == 400 and "step must be" in reply.json()["error"]
    assert api.post("/teach/answer", content=b"[" * 50_000).status_code == 400
    big = api.post("/teach/answer", content=b" " * (65 * 1024))
    assert big.status_code == 413 and big.json()["error"] == "the answer is over 64 KB"
    assert api.get("/teach/status").json()["question"] == "spend"  # still open
    answer(api, "spend", False)
    wait(api, ended)


def jpeg(width, height):
    out = io.BytesIO()
    Image.new("L", (width, height), 128).save(out, "JPEG")
    return out.getvalue()


def test_frames_that_are_not_camera_frames_and_endless_audio_are_refused(api, fake, settings,
                                                                         frames):
    tall = start(api, [*frames[:2], jpeg(8, 4000)], request=REQUEST)
    assert tall.status_code == 400
    assert tall.json()["error"] == "frame 2: 8x4000 is not a camera frame"
    files = [("frames", (f"{i}.jpg", data, "image/jpeg")) for i, data in enumerate(frames)]
    for value in ("inf", "1e400", "nan"):
        endless = api.post("/teach/start", data={"skill": "washer_done", "audio_s": value},
                           files=[*files, ("audio", ("request.m4a", b"m4a", "audio/mp4"))])
        assert endless.status_code == 400 and "audio_s must be a finite" in endless.text
    assert not (settings.data_dir / "teach").exists() and fake.calls == {}
