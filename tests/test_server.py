import io

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.cli import main, synthetic_jpeg
from backend.server import create_app, inspect_jpeg

HELLO = {"type": "hello", "client": "test", "source": "mock", "fps": 1}


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"ok": True, "protocol": 1}


def test_frames_are_acked(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json(HELLO)
        assert ws.receive_json() == {"type": "welcome", "session": "s1", "protocol": 1}
        for i in range(3):
            frame = synthetic_jpeg(i, size=(320, 240))
            ws.send_bytes(frame)
            assert ws.receive_json() == {
                "type": "ack", "seq": i + 1, "bytes": len(frame), "width": 320, "height": 240,
            }
        ws.send_json({"type": "bye"})


def test_bad_frames_get_errors_and_session_continues(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json(HELLO)
        ws.receive_json()
        ws.send_bytes(b"not an image")
        assert ws.receive_json() == {"type": "error", "seq": 1, "error": "frame is not a JPEG"}
        ws.send_bytes(b"\xff\xd8 truncated")
        assert ws.receive_json()["seq"] == 2
        ws.send_text("{}")
        assert ws.receive_json() == {"type": "error", "error": "unknown message"}
        ws.send_bytes(synthetic_jpeg(0))
        assert ws.receive_json()["type"] == "ack"


@pytest.mark.parametrize(
    "first", ['{"type": "hello", "source": "webcam"}', "[]", "not json", '{"type": "bye"}']
)
def test_hello_required(client, first):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(first)
        assert ws.receive_json()["type"] == "error"
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_sessions_are_numbered(client):
    for expected in ("s1", "s2"):
        with client.websocket_connect("/ws") as ws:
            ws.send_json(HELLO)
            assert ws.receive_json()["session"] == expected


def test_inspect_jpeg_limits():
    with pytest.raises(ValueError, match="limit"):
        inspect_jpeg(b"\xff\xd8" + b"\0" * (8 * 1024 * 1024))
    assert inspect_jpeg(synthetic_jpeg(1, size=(64, 32))) == (64, 32)


def huge_header_jpeg(width: int, height: int) -> bytes:
    """A tiny real JPEG whose SOF0 header claims a different size."""
    data = bytearray(synthetic_jpeg(0, size=(16, 16)))
    sof = data.index(b"\xff\xc0")
    data[sof + 5:sof + 9] = height.to_bytes(2, "big") + width.to_bytes(2, "big")
    return bytes(data)


@pytest.mark.parametrize("frame", [
    huge_header_jpeg(60000, 60000),  # decompression bomb
    huge_header_jpeg(5000, 5000),  # over the frame pixel limit
    synthetic_jpeg(3, size=(640, 480))[:700],  # truncated
])
def test_bad_frames_do_not_end_the_session(client, frame):
    with client.websocket_connect("/ws") as ws:
        ws.send_json(HELLO)
        ws.receive_json()
        ws.send_bytes(frame)
        assert ws.receive_json()["type"] == "error"
        ws.send_bytes(synthetic_jpeg(0))
        assert ws.receive_json()["type"] == "ack"


def test_mpo_frames_are_accepted():
    from PIL import Image

    buf = io.BytesIO()
    first, second = Image.new("RGB", (64, 48), "gray"), Image.new("RGB", (64, 48), "white")
    first.save(buf, format="MPO", save_all=True, append_images=[second])
    assert inspect_jpeg(buf.getvalue()) == (64, 48)


@pytest.mark.parametrize("text", ["[]", "null", "1", '"bye"'])
def test_non_object_text_messages(client, text):
    with client.websocket_connect("/ws") as ws:
        ws.send_json(HELLO)
        ws.receive_json()
        ws.send_text(text)
        assert ws.receive_json() == {"type": "error", "error": "unknown message"}


def test_frames_never_touch_disk_or_logs(client, tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    frame = synthetic_jpeg(5)
    with caplog.at_level("INFO", logger="trained_eye.server"), \
            client.websocket_connect("/ws") as ws:
        ws.send_json(HELLO)
        ws.receive_json()
        for _ in range(3):
            ws.send_bytes(frame)
            ws.receive_json()
        ws.send_json({"type": "bye"})
    assert list(tmp_path.rglob("*")) == []
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "frame 3: 640x480" in logged
    assert frame[:16].hex() not in logged and "base64" not in logged
    assert all(len(r.getMessage()) < 200 for r in caplog.records)


def test_ws_selftest_command(capsys):
    assert main(["ws-selftest", "--frames", "2"]) == 0
    out = capsys.readouterr().out
    assert "connected: session s1, protocol 1" in out
    assert "sent 2 frames, got 2 acks" in out


class FakeWatcher:
    """Scores 0.9 for every frame and alerts on the third, like a 3-of-4 trigger."""

    def __init__(self, names):
        self.skills = dict.fromkeys(names)
        self.frames = 0

    def on_frame(self, data):
        from backend.watch.watcher import Alert

        self.frames += 1
        alerts = [Alert(name, "Laundry's done.", float(self.frames), 0.9, True)
                  for name in self.skills] if self.frames == 3 else []
        return dict.fromkeys(self.skills, 0.9), alerts


def test_watch_mode_scores_frames_and_sends_alerts():
    made = []

    def factory(names, session):
        made.append((names, session))
        return FakeWatcher(names)

    with TestClient(create_app(factory)) as c, c.websocket_connect("/ws") as ws:
        ws.send_json({**HELLO, "watch": ["washer_done"]})
        assert ws.receive_json() == {"type": "welcome", "session": "s1", "protocol": 1,
                                     "watching": ["washer_done"]}
        for i in range(3):
            ack = (ws.send_bytes(synthetic_jpeg(i, size=(320, 240))), ws.receive_json())[1]
            assert ack["type"] == "ack" and ack["scores"] == {"washer_done": 0.9}
        assert ws.receive_json() == {"type": "alert", "skill": "washer_done",
                                     "text": "Laundry's done.", "t": 3.0, "p": 0.9,
                                     "confirmed": True}
        ws.send_json({"type": "bye"})
    assert made == [(["washer_done"], "s1")]


@pytest.mark.parametrize("watch, factory, message", [
    (["washer_done"], None, "not set up"),
    ("washer_done", lambda n, s: FakeWatcher(n), "list of skill names"),
    (["nope"], lambda n, s: (_ for _ in ()).throw(ValueError("no taught skill named 'nope'")),
     "no taught skill"),
])
def test_a_watch_request_that_cannot_be_met_is_an_error_and_frames_are_still_acked(
        watch, factory, message):
    with TestClient(create_app(factory)) as c, c.websocket_connect("/ws") as ws:
        ws.send_json({**HELLO, "watch": watch})
        error = ws.receive_json()
        assert error["type"] == "error" and message in error["error"]
        assert "watching" not in ws.receive_json()
        ws.send_bytes(synthetic_jpeg(0))
        assert "scores" not in ws.receive_json()


def test_the_default_watcher_needs_a_student_endpoint_and_taught_skills(tmp_path, monkeypatch):
    from backend.server import default_watcher
    from backend.skills.spec import SkillSpec, Target

    monkeypatch.setenv("TE_DATA_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="TE_STUDENT_URL"):
        default_watcher(["washer_done"], "s1")
    monkeypatch.setenv("TE_STUDENT_URL", "http://127.0.0.1:8080/v1")
    with pytest.raises(ValueError, match="no taught skill"):
        default_watcher(["washer_done"], "s1")
    skill = tmp_path / "skills/washer_done"
    skill.mkdir(parents=True)
    skill.joinpath("spec.json").write_text(SkillSpec(
        name="washer_done", request="tell me when the washer says End",
        target=Target(sam_prompt="washer display", description="the display"),
        positive="the display shows End", negatives=["display off"],
        alert="Laundry's done.").model_dump_json())
    watcher = default_watcher(["washer_done"], "s7")
    assert list(watcher.skills) == ["washer_done"]
    assert watcher.log.path.parent == tmp_path / "watch" and watcher.log.path.name.endswith(
        "-s7.jsonl")


def test_skills_lists_taught_skills_for_the_app(tmp_path):
    from backend.skills.spec import SkillSpec, Target

    def taught(name, refused=False, threshold=None):
        folder = tmp_path / "skills" / name
        (folder / "dataset").mkdir(parents=True)
        spec = (SkillSpec(name=name, request="tell me when Sam gets here", refused=True,
                          refusal_reason="I can't watch for or recognize specific people.")
                if refused else SkillSpec(
                    name=name, request="tell me when the washer says End",
                    target=Target(sam_prompt="washer display", description="the display"),
                    positive="the display shows End", negatives=["display off"],
                    alert="Laundry's done."))
        (folder / "spec.json").write_text(spec.model_dump_json())
        if threshold is not None:
            (folder / "threshold.json").write_text(f'{{"threshold": {threshold}}}')

    taught("washer_done", threshold=0.82)
    taught("pot_boils")
    taught("sam_arrives", refused=True)
    (tmp_path / "skills/washer_done/dataset/washer_done-teach.jsonl").write_text("{}\n")
    (tmp_path / "skills/broken").mkdir()
    (tmp_path / "skills/broken/spec.json").write_text("not json")
    (tmp_path / "skills/binary").mkdir()
    (tmp_path / "skills/binary/spec.json").write_bytes(b"\xff\xfe")  # not UTF-8
    taught("hot_pan", threshold="NaN")  # JSON reads NaN; it cannot be sent back
    with TestClient(create_app(data_dir=lambda: tmp_path)) as c:
        listed = c.get("/skills").json()["skills"]
    assert listed == [
        {"name": "hot_pan", "alert": "Laundry's done.", "cooldown_s": 600, "dataset": False,
         "calibrated": False, "threshold": None},
        {"name": "pot_boils", "alert": "Laundry's done.", "cooldown_s": 600, "dataset": False,
         "calibrated": False, "threshold": None},
        {"name": "washer_done", "alert": "Laundry's done.", "cooldown_s": 600, "dataset": True,
         "calibrated": True, "threshold": 0.82},
    ]


def test_the_served_app_has_the_phone_teach_routes():
    from backend.server import app

    paths = {route.path for route in app.routes}
    assert {"/health", "/skills", "/ws", "/teach/start", "/teach/status",
            "/teach/answer"} <= paths


def test_shutdown_hooks_run_when_the_server_stops():
    stopped = []
    with TestClient(create_app(on_shutdown=[lambda: stopped.append("teach")])) as client:
        assert client.get("/health").status_code == 200 and stopped == []
    assert stopped == ["teach"]
