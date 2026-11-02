import json

import pytest
from starlette.testclient import TestClient

from backend.labeltool.clips import create_clip_app
from backend.watch.trigger import labelled_frames

VIDEO = bytes(range(256)) * 4  # 1024 bytes; the server never decodes the video
BAD = [{"duration_s": 60, "true": t} for t in (
    [[10, 10]], [[20, 10]], [[50, 61]], [[-1, 5]], [[1, "2"]], [[1, 2, 3]], [[1, True]], "1-2")]
BAD += [{"duration_s": 0, "true": []}, {"true": [[1, 2]]}, [[1, 2]], b"[[1,"]
BAD += [b'{"duration_s": 60, "true": [[1, 1' + b"0" * 400 + b"]]}",  # too large for a float
        b'{"duration_s": 60, "true": [[1, NaN]]}', b'{"duration_s": Infinity, "true": []}']


@pytest.fixture
def clip(tmp_path):
    (tmp_path / "session3_test_a.mp4").write_bytes(VIDEO)
    return tmp_path / "session3_test_a.mp4"


def test_page_empty_labels_and_video_range_requests(clip, tmp_path):
    client = TestClient(create_clip_app(clip, tmp_path / "out.json"))
    page = client.get("/")
    assert page.status_code == 200 and "<video" in page.text and "static/style.css" in page.text
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/labels").json() == {"clip": "session3_test_a.mp4", "duration_s": None,
                                            "true": [], "out": "out.json"}
    full = client.get("/video")
    assert full.status_code == 200 and full.content == VIDEO
    assert full.headers["accept-ranges"] == "bytes" and full.headers["content-type"] == "video/mp4"
    part = client.get("/video", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and part.content == VIDEO[100:200]
    assert part.headers["content-range"] == "bytes 100-199/1024"
    assert client.get("/video", headers={"Range": "bytes=0-1"}).content == VIDEO[:2]  # Safari
    assert client.get("/video", headers={"Range": "bytes=5000-"}).status_code == 416


def test_save_sorts_merges_and_feeds_calibrate(clip, tmp_path):
    out = tmp_path / "labels" / "a.json"
    ranges = [[30, 40], [5, 10], [8, 12.5], [40, 45], [50.25, 60]]
    reply = TestClient(create_clip_app(clip, out)).post(
        "/labels", json={"duration_s": 60.0, "true": ranges})
    assert reply.status_code == 200 and reply.json()["out"] == "a.json"
    assert out.read_text() == ('{"clip": "session3_test_a.mp4", "duration_s": 60.0, '
                               '"true": [[5.0, 12.5], [30.0, 45.0], [50.25, 60.0]]}\n')
    rows = [{"skill": "s", "t": t, "p": 0.5} for t in (4, 6, 20, 45, 59)]
    truth = [f.truth for f in labelled_frames(rows, "s", json.loads(out.read_text())["true"])]
    assert truth == [False, True, False, True, True]


@pytest.mark.parametrize("body", BAD)
def test_save_rejects_bad_ranges(clip, tmp_path, body):
    out = tmp_path / "out.json"
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    reply = TestClient(create_clip_app(clip, out)).post("/labels", content=raw)
    assert reply.status_code == 400 and reply.json()["error"]
    assert not out.exists()


def test_resumes_and_refuses_another_clips_labels(clip, tmp_path):
    out = tmp_path / "out.json"
    out.write_text(json.dumps({"clip": clip.name, "duration_s": 60.0, "true": [[1.0, 2.0]]}))
    client = TestClient(create_clip_app(clip, out))
    assert client.get("/labels").json()["true"] == [[1.0, 2.0]]
    client.post("/labels", json={"duration_s": 60.0, "true": [[1.0, 2.0], [3.0, 4.0]]})
    resumed = TestClient(create_clip_app(clip, out)).get("/labels").json()
    assert resumed["true"] == [[1.0, 2.0], [3.0, 4.0]] and resumed["duration_s"] == 60.0
    out.write_text(json.dumps({"clip": "other.mp4", "duration_s": 9.0, "true": []}))
    with pytest.raises(ValueError, match="other.mp4"):
        create_clip_app(clip, out)
    with pytest.raises(FileNotFoundError):
        create_clip_app(tmp_path / "missing.mp4", tmp_path / "x.json")
