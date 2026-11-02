"""The playground's local reads: clips with frame logs, the eval charts, media paths."""

import json

from backend.playground import data

ROWS = [  # as backend/watch/watcher.py logs them
    {"t": 0.0, "skill": "washer_done", "threshold": 0.6, "p": 0.1, "over": False, "fires": False},
    {"t": 1.0, "skill": "washer_done", "threshold": 0.6, "p": None, "error": "timeout"},
    {"t": 2.0, "skill": "washer_done", "threshold": 0.6, "p": 0.8, "over": True, "fires": False},
    {"t": 3.0, "skill": "washer_done", "threshold": 0.6, "p": 0.9, "over": True, "fires": True,
     "alert": True, "cloud": "yes"},
    {"t": 4.0, "skill": "washer_done", "threshold": 0.6, "p": 0.95, "fires": True,
     "alert": "Custom words."},
]


def make_clip(eval_dir, name="washer_1", rows=ROWS, video=True, labels=None):
    folder = eval_dir / "clips" / name
    folder.mkdir(parents=True)
    lines = [json.dumps(r) for r in rows] + ['{"t": 5.0, "skill": "washer_d']  # a torn line
    (folder / "log.jsonl").write_text("\n".join(lines) + "\n")
    if video:
        (folder / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42 fake")
    if labels is not None:
        (folder / "labels.json").write_text(json.dumps(labels))
    return folder


def write_spec(data_dir, skill="washer_done", alert="Your laundry's done."):
    folder = data_dir / "skills" / skill
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "spec.json").write_text(json.dumps({"name": skill, "alert": alert}))


def test_list_clips(tmp_path):
    make_clip(tmp_path, labels={"true": [[2.5, 4.0]]})
    make_clip(tmp_path, "no_video", rows=ROWS[:1], video=False)
    (tmp_path / "clips" / "no_log").mkdir()
    clips = data.list_clips(tmp_path)
    assert [c["name"] for c in clips] == ["no_video", "washer_1"]
    assert clips[1] == {"name": "washer_1", "video": True, "skills": ["washer_done"],
                        "frames": 5, "alerts": 2, "labelled": True}
    assert clips[0]["video"] is False and clips[0]["alerts"] == 0
    assert data.list_clips(tmp_path / "missing") == []


def test_clip_timeline(tmp_path):
    eval_dir, data_dir = tmp_path / "eval", tmp_path / "data"
    make_clip(eval_dir, labels={"true": [[2.5, 4.0], [3, 1], "bad", [0, "x"]]})
    write_spec(data_dir)
    clip = data.clip_timeline(eval_dir, data_dir, "washer_1")
    assert clip["video"] == "media/clips/washer_1/clip.mp4"
    assert clip["duration"] == 4.0
    assert clip["true_ranges"] == [[2.5, 4.0]]  # reversed and malformed ranges dropped
    skill = clip["skills"]["washer_done"]
    assert skill["threshold"] == 0.6
    assert skill["points"] == [[0.0, 0.1], [1.0, None], [2.0, 0.8], [3.0, 0.9], [4.0, 0.95]]
    assert skill["alerts"] == [{"t": 3.0, "p": 0.9, "text": "Your laundry's done."},
                               {"t": 4.0, "p": 0.95, "text": "Custom words."}]


def test_clip_timeline_without_spec_or_labels(tmp_path):
    make_clip(tmp_path, video=False)
    clip = data.clip_timeline(tmp_path, tmp_path / "no_data", "washer_1")
    assert clip["video"] is None and clip["true_ranges"] == []
    assert clip["skills"]["washer_done"]["alerts"][0]["text"] == "Alert"
    for name in ("missing", "../washer_1", "washer_1/..", ""):
        assert data.clip_timeline(tmp_path, tmp_path, name) is None


def test_results(tmp_path):
    assert data.results(tmp_path) == {"charts": [], "results_md": None}
    (tmp_path / "cost_vs_accuracy.svg").write_text("<svg/>")
    (tmp_path / "exp1-mix.svg").write_text("<svg/>")
    (tmp_path / "notes.txt").write_text("not a chart")
    (tmp_path / "results.md").write_text("# Results\n\nF1 0.9\n")
    out = data.results(tmp_path)
    assert out["charts"] == [
        {"name": "cost vs accuracy", "url": "media/results/cost_vs_accuracy.svg"},
        {"name": "exp1 mix", "url": "media/results/exp1-mix.svg"}]
    assert out["results_md"].startswith("# Results")


def test_media_path_serves_only_linked_files(tmp_path):
    data_dir, eval_dir = tmp_path / "data", tmp_path / "eval"
    make_clip(eval_dir)
    (eval_dir / "chart.svg").write_text("<svg/>")
    kf = data_dir / "skills" / "washer_done" / "keyframes"
    kf.mkdir(parents=True)
    (kf / "kf_00_1a2b3c4d.jpg").write_bytes(b"\xff\xd8\xff")
    (kf.parent / "spec.json").write_text("{}")
    (data_dir / "playground" / "keyframes").mkdir(parents=True)
    (data_dir / "playground" / "keyframes" / "door.png").write_bytes(b"\x89PNG")
    (tmp_path / ".env").write_text("MODEL_API_KEY=secret")

    def found(path):
        hit = data.media_path(data_dir, eval_dir, path)
        return hit and (hit[0].name, hit[1])

    assert found("clips/washer_1/clip.mp4") == ("clip.mp4", "video/mp4")
    assert found("skills/washer_done/keyframes/kf_00_1a2b3c4d.jpg") == ("kf_00_1a2b3c4d.jpg",
                                                                       "image/jpeg")
    assert found("playground/keyframes/door.png") == ("door.png", "image/png")
    assert found("results/chart.svg") == ("chart.svg", "image/svg+xml")
    for path in ("clips/washer_1/log.jsonl", "skills/washer_done/spec.json",
                 "skills/washer_done/keyframes/../spec.json", "results/../.env",
                 "skills/../../.env", "clips/../../.env/clip.mp4", "results/missing.svg",
                 "skills/washer_done/dataset/x.png", "", "results"):
        assert not found(path), path
