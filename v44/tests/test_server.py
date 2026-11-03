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
