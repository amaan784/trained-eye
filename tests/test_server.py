import io
import json
import time

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
        alerts = [Alert(name, "The burner is still on.", float(self.frames), 0.9, True)
                  for name in self.skills] if self.frames == 3 else []
        return dict.fromkeys(self.skills, 0.9), alerts


def test_watch_mode_scores_frames_and_sends_alerts():
    made = []

    def factory(names, session, cooldowns):
        made.append((names, session))
        return FakeWatcher(names)

    with TestClient(create_app(factory)) as c, c.websocket_connect("/ws") as ws:
        ws.send_json({**HELLO, "watch": ["burner_left_on"]})
        assert ws.receive_json() == {"type": "welcome", "session": "s1", "protocol": 1,
                                     "watching": ["burner_left_on"]}
        for i in range(3):
            ack = (ws.send_bytes(synthetic_jpeg(i, size=(320, 240))), ws.receive_json())[1]
            assert ack["type"] == "ack" and ack["scores"] == {"burner_left_on": 0.9}
        assert ws.receive_json() == {"type": "alert", "skill": "burner_left_on",
                                     "text": "The burner is still on.", "t": 3.0, "p": 0.9,
                                     "confirmed": True}
        ws.send_json({"type": "bye"})
    assert made == [(["burner_left_on"], "s1")]


@pytest.mark.parametrize("watch, factory, message", [
    (["burner_left_on"], None, "not set up"),
    ("burner_left_on", lambda n, s, c: FakeWatcher(n), "list of skill names"),
    (["nope"], lambda n, s, c: (_ for _ in ()).throw(ValueError("no taught skill named 'nope'")),
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
        default_watcher(["burner_left_on"], "s1")
    monkeypatch.setenv("TE_STUDENT_URL", "http://127.0.0.1:8080/v1")
    with pytest.raises(ValueError, match="no taught skill"):
        default_watcher(["burner_left_on"], "s1")
    skill = tmp_path / "skills/burner_left_on"
    skill.mkdir(parents=True)
    skill.joinpath("spec.json").write_text(SkillSpec(
        name="burner_left_on", request="tell me when the burner is left on",
        target=Target(sam_prompt="stove burner", description="the display"),
        positive="the burner is lit with no pan on it", negatives=["a pan being stirred on it"],
        alert="The burner is still on.").model_dump_json())
    watcher = default_watcher(["burner_left_on"], "s7")
    assert list(watcher.skills) == ["burner_left_on"]
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
                    name=name, request="tell me when the burner is left on",
                    target=Target(sam_prompt="stove burner", description="the display"),
                    positive="the burner is lit with no pan on it", negatives=["a pan being stirred on it"],
                    alert="The burner is still on."))
        (folder / "spec.json").write_text(spec.model_dump_json())
        if threshold is not None:
            (folder / "threshold.json").write_text(f'{{"threshold": {threshold}}}')

    taught("burner_left_on", threshold=0.82)
    taught("pot_boils")
    taught("sam_arrives", refused=True)
    (tmp_path / "skills/burner_left_on/dataset/burner_left_on-teach.jsonl").write_text("{}\n")
    (tmp_path / "skills/broken").mkdir()
    (tmp_path / "skills/broken/spec.json").write_text("not json")
    (tmp_path / "skills/binary").mkdir()
    (tmp_path / "skills/binary/spec.json").write_bytes(b"\xff\xfe")  # not UTF-8
    taught("hot_pan", threshold="NaN")  # JSON reads NaN; it cannot be sent back
    with TestClient(create_app(data_dir=lambda: tmp_path)) as c:
        listed = c.get("/skills").json()["skills"]
    untrained = {"training": None, "served": None}  # no deploy recorded: unknown
    assert listed == [
        {"name": "burner_left_on", "alert": "The burner is still on.", "cooldown_s": 600, "hold_s": 0,
         "dataset": True, "calibrated": True, "threshold": 0.82, **untrained},
        {"name": "hot_pan", "alert": "The burner is still on.", "cooldown_s": 600, "hold_s": 0,
         "dataset": False, "calibrated": False, "threshold": None, **untrained},
        {"name": "pot_boils", "alert": "The burner is still on.", "cooldown_s": 600, "hold_s": 0,
         "dataset": False, "calibrated": False, "threshold": None, **untrained},
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



def watch_skill(name, threshold=0.5):
    from backend.skills.spec import SkillSpec, Target
    from backend.watch.trigger import Trigger
    from backend.watch.watcher import WatchedSkill

    spec = SkillSpec(name=name, request=f"tell me when {name}", positive="it is so",
                     target=Target(sam_prompt="thing", description="the thing"),
                     negatives=["it is not"], alert=f"{name}!", confirm_with_cloud=False)
    return WatchedSkill(spec, Trigger(threshold, 600.0), True, "skill")


class Always:
    """A student that sees the condition in every frame."""

    def score(self, image, spec):
        from backend.inference.student import Score

        return Score(0.9, "yes", 0.01)


def real_watchers(names, session, cooldowns):
    from backend.watch.watcher import Watcher

    return Watcher([watch_skill(n) for n in names], Always(), cooldowns=cooldowns)


def alerts_in_a_session(client, names, frames=4):
    alerts = []
    with client.websocket_connect("/ws") as ws:
        ws.send_json({**HELLO, "watch": names})
        assert ws.receive_json()["watching"] == sorted(names)
        for i in range(frames):
            ws.send_bytes(synthetic_jpeg(i, size=(64, 48)))
        ws.send_text("{}")  # its error reply comes after every ack and alert
        while (message := ws.receive_json())["type"] != "error":
            if message["type"] == "alert":
                alerts.append(message["skill"])
        ws.send_json({"type": "bye"})
    return alerts


def test_a_reconnect_or_a_skills_toggle_does_not_alert_again_inside_the_cooldown(tmp_path):
    app = create_app(real_watchers, data_dir=lambda: tmp_path)
    with TestClient(app) as c:
        assert alerts_in_a_session(c, ["burner_left_on"]) == ["burner_left_on"]
        assert alerts_in_a_session(c, ["burner_left_on"]) == []  # the phone reconnected
        # a toggle opens a new session with the new list: only the new skill alerts
        assert alerts_in_a_session(c, ["burner_left_on", "pot_boils"]) == ["pot_boils"]
    saved = json.loads((tmp_path / "watch/cooldowns.json").read_text())["quiet_until"]
    assert set(saved) == {"burner_left_on", "pot_boils"}
    assert all(until > time.time() + 590 for until in saved.values())
    with TestClient(create_app(real_watchers, data_dir=lambda: tmp_path)) as c:
        assert alerts_in_a_session(c, ["burner_left_on"]) == []  # te serve restarted


def test_each_app_shares_one_book_with_every_session():
    from backend.watch.watcher import CooldownBook

    books = []
    book = CooldownBook()

    def factory(names, session, cooldowns):
        books.append(cooldowns)
        return FakeWatcher(names)

    with TestClient(create_app(factory, cooldowns=book)) as c:
        for _ in range(2):
            with c.websocket_connect("/ws") as ws:
                ws.send_json({**HELLO, "watch": ["burner_left_on"]})
                ws.receive_json()
    assert books == [book, book]


def taught_skill(data_dir, name, threshold=None):
    from backend.skills.spec import SkillSpec, Target

    folder = data_dir / "skills" / name
    (folder / "dataset").mkdir(parents=True)
    folder.joinpath("spec.json").write_text(SkillSpec(
        name=name, request="tell me when the burner is left on",
        target=Target(sam_prompt="stove burner", description="the display"),
        positive="the burner is lit with no pan on it", negatives=["a pan being stirred on it"],
        alert="The burner is still on.").model_dump_json())
    if threshold is not None:
        folder.joinpath("threshold.json").write_text(json.dumps({"threshold": threshold}))
    return folder


def test_skills_report_training_runs_and_what_is_served(tmp_path, monkeypatch):
    from backend.training import runs

    for name in ("burner_left_on", "pot_boils", "hot_pan", "door_open"):
        taught_skill(tmp_path, name, threshold=0.6 if name == "burner_left_on" else None)
    runs.write_status(tmp_path, "exp1-mixed", "done", ["burner_left_on", "pot_boils"],
                      started="2026-11-01T10:00:00Z", finished="2026-11-01T11:00:00Z")
    runs.write_status(tmp_path, "exp1-real", "failed", ["burner_left_on"],
                      started="2026-11-02T10:00:00Z", finished="2026-11-02T10:05:00Z")
    runs.write_status(tmp_path, "exp3-n100", "running", ["pot_boils"],
                      started="2026-11-03T09:00:00Z")
    runs.write_status(tmp_path, "smoke-check", "done", ["hot_pan"], smoke=True,
                      started="2026-11-03T08:00:00Z", finished="2026-11-03T08:20:00Z")
    ours = json.loads(runs.status_path(tmp_path, "exp1-real").read_text())["pid"]
    runs.write_status(tmp_path, "exp2-unfiltered", "running", ["door_open"],
                      started="2026-11-03T07:00:00Z")
    stale = runs.status_path(tmp_path, "exp2-unfiltered")  # its te train was killed
    stale.write_text(json.dumps({**json.loads(stale.read_text()), "pid": 2**22 + 12345}))
    monkeypatch.setattr(runs, "_alive", lambda pid: pid == ours)
    (tmp_path / "train/exp1-mixed").mkdir()
    (tmp_path / "train/exp1-mixed/manifest.json").write_text('{"skills": {}}')
    (tmp_path / "train/broken.status.json").write_text("not json")

    def listed():
        with TestClient(create_app(data_dir=lambda: tmp_path)) as c:
            return {s["name"]: s for s in c.get("/skills").json()["skills"]}

    skills = listed()
    assert skills["burner_left_on"]["training"] == {
        "run": "exp1-real", "state": "failed", "started": "2026-11-02T10:00:00Z",
        "finished": "2026-11-02T10:05:00Z"}  # the newest run that lists it
    assert skills["pot_boils"]["training"] == {
        "run": "exp3-n100", "state": "running", "started": "2026-11-03T09:00:00Z",
        "finished": None}
    assert skills["hot_pan"]["training"] is None  # a finished smoke leaves no adapter
    assert skills["door_open"]["training"]["state"] == "failed"  # running, but te train died
    assert all(s["served"] is None for s in skills.values())  # nothing recorded: unknown

    runs.write_serving(tmp_path, "exp1-mixed")
    run_file = tmp_path / "watch/exp1-mixed/burner_left_on/threshold.json"
    run_file.parent.mkdir(parents=True)
    run_file.write_text('{"threshold": 0.75}')
    skills = listed()
    assert {n for n, s in skills.items() if s["served"]} == {"burner_left_on", "pot_boils"}
    assert (skills["burner_left_on"]["calibrated"], skills["burner_left_on"]["threshold"]) == (
        True, 0.75)  # the served run's own threshold, as watch mode uses it
    runs.clear_serving(tmp_path)  # te serve-student --stop
    skills = listed()
    assert skills["burner_left_on"]["threshold"] == 0.6
    assert not any(s["served"] for s in skills.values()) and skills["burner_left_on"]["served"] is False


def test_the_default_watcher_follows_the_served_run_and_the_settings(tmp_path, monkeypatch):
    from backend.server import default_watcher
    from backend.training import runs
    from backend.watch.watcher import CooldownBook

    monkeypatch.setenv("TE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TE_STUDENT_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("TE_STUDENT_MAX_TOKENS", "1024")
    folder = taught_skill(tmp_path, "burner_left_on", threshold=0.6)
    (folder / "dataset/burner_left_on-teach.manifest.json").write_text(
        '{"system_in_user": true}')
    book = CooldownBook()
    watcher = default_watcher(["burner_left_on"], "s1", book)
    assert watcher.cooldowns is book and watcher.scorer.max_tokens == 1024
    assert watcher.scorer.system_in_user is True  # as the dataset was written
    assert watcher.skills["burner_left_on"].threshold_source == "skill"
    assert watcher.log.fields == {}

    runs.write_status(tmp_path, "exp1-mixed", "done", ["burner_left_on"], started=runs.now_iso())
    (tmp_path / "train/exp1-mixed").mkdir()
    (tmp_path / "train/exp1-mixed/manifest.json").write_text(
        '{"skills": {"burner_left_on": "x"}, "system_in_user": false}')
    runs.write_serving(tmp_path, "exp1-mixed")
    run_file = tmp_path / "watch/exp1-mixed/burner_left_on/threshold.json"
    run_file.parent.mkdir(parents=True)
    run_file.write_text('{"threshold": 0.8}')
    watcher = default_watcher(["burner_left_on"], "s2")
    skill = watcher.skills["burner_left_on"]
    assert (skill.trigger.threshold, skill.threshold_source) == (0.8, "run")
    assert watcher.scorer.system_in_user is False  # what the served run trained on
    assert watcher.log.fields == {"run": "exp1-mixed"}
