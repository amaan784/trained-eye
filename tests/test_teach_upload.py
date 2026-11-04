"""The checks on the phone's teach upload (backend/teach_upload.py), on bodies that httpx
encodes the way the phone's multipart would be. No Meta call is made here: a session is
only built, never started."""

import httpx
import pytest

from backend import teach_upload
from backend.config import ConfigError, load_settings
from backend.meta.client import MetaClient
from backend.teach_upload import Problem, new_session, parse_multipart, save_capture
from tests.test_teach import clip, wav

REQUEST = "let me know when the washer says done"


def encode(fields, files):
    request = httpx.Request("POST", "http://te/teach/start", data=fields, files=files)
    return request.read(), request.headers["content-type"]


@pytest.fixture
def frames(tmp_path):
    return [("frames", (p.name, p.read_bytes(), "image/jpeg"))
            for p in sorted(clip(tmp_path / "clip").iterdir())]


def upload(tmp_path, files, env=None, **fields):
    settings = load_settings(env={"TE_DATA_DIR": str(tmp_path / "data"), "MODEL_API_KEY": "k",
                                  **(env or {})}, env_file=None)
    form = parse_multipart(*encode({"skill": "burner_left_on", **fields}, files))
    return new_session(form, settings, MetaClient, 5.0), settings


def test_the_multipart_body_is_read_as_sent():
    tricky = b"\xff\xd8\r\n--not-the-boundary\r\n\r\nstill the frame\r\n"
    form = parse_multipart(*encode({"skill": "burner_left_on"}, [
        ("frames", ("a.jpg", tricky, "image/jpeg")), ("frames", ("b.jpg", b"", "image/jpeg")),
        ("audio", ("request.m4a", b"\0\1\2", "audio/mp4"))]))
    assert form == {"skill": [(None, b"burner_left_on")],
                    "frames": [("a.jpg", tricky), ("b.jpg", b"")],
                    "audio": [("request.m4a", b"\0\1\2")]}
    body, content_type = encode({"skill": "burner_left_on"}, [("audio", ("r.wav", b"x"))])
    with pytest.raises(ValueError, match="cut off"):
        parse_multipart(body[:-10], content_type)
    with pytest.raises(ValueError, match="multipart/form-data"):
        parse_multipart(b"skill=burner_left_on", "application/x-www-form-urlencoded")


def test_an_upload_becomes_a_session_waiting_for_the_spend(tmp_path, frames):
    session, settings = upload(tmp_path, [*frames, ("audio", ("Request.M4A", b"\0aac"))],
                               audio_s="4.5", positives="7", negatives="0")
    assert session.state == "waiting_spend" and not session.finished
    opts = session.opts
    assert (opts.positives, opts.negatives, opts.audio_s, opts.request) == (7, 0, 4.5, None)
    assert opts.audio.name == "request.m4a" and opts.audio.read_bytes() == b"\0aac"
    assert opts.clip.parent == settings.data_dir / "teach" / "burner_left_on"
    assert min(p.name for p in opts.clip.glob("*.jpg")) == "f000.jpg"
    assert session.status()["estimate"]["images"] == pytest.approx(7 * 0.01)
    assert session.status()["warning"] is None


def test_two_captures_in_the_same_second_get_their_own_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(teach_upload.time, "strftime", lambda *a: "20261102T120000Z")
    first = save_capture(tmp_path, [b"a"], None, "")
    second = save_capture(tmp_path, [b"b"], b"RIFF", "wav")
    assert (first.name, second.name) == ("20261102T120000Z", "20261102T120000Z-2")
    assert sorted(p.name for p in second.iterdir()) == ["f000.jpg", "request.wav"]


@pytest.mark.parametrize("case, error, message", [
    ("not a jpeg", ValueError, "frame 0: frame is not a JPEG"),
    ("unknown field", ValueError, "unknown fields: labels"),
    ("both", ValueError, "(one of them)"),
    ("audio type", ValueError, "audio must be a file ending in one of aac, flac"),
    ("bad count", ValueError, "positives must be a number, not 'many'"),
    ("negative count", ValueError, "negatives must be a finite number from 0 to 100000"),
    ("long request", ValueError, "over the 500 allowed"),
    ("policy", Problem, "I can't learn that one."),
    ("no key", ConfigError, "MODEL_API_KEY is not set"),
])
def test_bad_uploads_are_refused_before_anything_is_saved(tmp_path, frames, case, error,
                                                          message):
    audio = ("audio", ("request.txt", b"words"))
    kwargs = {"not a jpeg": ([("frames", ("f.jpg", b"GIF89a"))], {"request": REQUEST}),
              "unknown field": (frames, {"request": REQUEST, "labels": "{}"}),
              "both": ([*frames, audio], {"request": REQUEST}),
              "audio type": ([*frames, audio], {}),
              "bad count": (frames, {"request": REQUEST, "positives": "many"}),
              "negative count": (frames, {"request": REQUEST, "negatives": "-1"}),
              "long request": (frames, {"request": "the washer " * 60}),
              "policy": (frames, {"request": "tell me when Sam gets here"}),
              "no key": (frames, {"request": REQUEST, "env": {"MODEL_API_KEY": ""}})}[case]
    files, fields = kwargs
    with pytest.raises(error, match=message.replace("(", r"\(").replace(")", r"\)")) as err:
        upload(tmp_path, files, **fields)
    assert not (tmp_path / "data" / "teach").exists()
    if case == "policy":
        assert err.value.status == 422 and err.value.extra == {"refused": True}


def test_own_labels_needed_saves_the_keyframes_and_says_how(tmp_path, frames):
    audio = ("audio", ("request.wav", wav(tmp_path / "r.wav").read_bytes()))
    with pytest.raises(Problem, match="needs your own labels") as err:
        upload(tmp_path, [*frames, audio], env={"TE_TRAIN_ON_MODEL_OUTPUTS": "0"})
    assert err.value.status == 422
    assert "te teach --skill burner_left_on --clip " in str(err.value)
    assert "--audio " in str(err.value) and str(err.value).endswith("--labels <file>")
    assert any((tmp_path / "data/skills/burner_left_on/keyframes").iterdir())
