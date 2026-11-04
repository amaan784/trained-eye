"""The playground's local reads: clips with frame logs, the eval charts, media paths, and
what the public copy refuses to serve."""

import hashlib
import json
from pathlib import Path

from backend.playground import data
from backend.release import faces

ROWS = [  # as backend/watch/watcher.py logs them
    {"t": 0.0, "skill": "burner_left_on", "threshold": 0.6, "p": 0.1, "over": False, "fires": False},
    {"t": 1.0, "skill": "burner_left_on", "threshold": 0.6, "p": None, "error": "timeout"},
    {"t": 2.0, "skill": "burner_left_on", "threshold": 0.6, "p": 0.8, "over": True, "fires": False},
    {"t": 3.0, "skill": "burner_left_on", "threshold": 0.6, "p": 0.9, "over": True, "fires": True,
     "alert": True, "cloud": "yes"},
    {"t": 4.0, "skill": "burner_left_on", "threshold": 0.6, "p": 0.95, "fires": True,
     "alert": "Custom words."},
]


def make_clip(data_dir, name="burner_1", rows=ROWS, video=True, labels=None, blurred=False):
    """data/playground/clips/<name>/ as te playground-clip leaves it (fake video bytes)."""
    folder = data.clips_root(data_dir) / name
    folder.mkdir(parents=True)
    lines = [json.dumps(r) for r in rows] + ['{"t": 5.0, "skill": "washer_d']  # a torn line
    (folder / "log.jsonl").write_text("\n".join(lines) + "\n")
    if video:
        (folder / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42 fake")
    if labels is not None:
        (folder / "labels.json").write_text(json.dumps(labels))
    if blurred:
        record = {"output_sha256": hashlib.sha256((folder / "clip.mp4").read_bytes()).hexdigest()}
        (folder / "clip.mp4.blur.json").write_text(json.dumps(record))
    return folder


def write_spec(data_dir, skill="burner_left_on", alert="The burner is still on."):
    folder = data_dir / "skills" / skill
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "spec.json").write_text(json.dumps({"name": skill, "alert": alert}))


def test_list_clips(tmp_path):
    make_clip(tmp_path, labels={"true": [[2.5, 4.0]]})
    make_clip(tmp_path, "no_video", rows=ROWS[:1], video=False)
    (data.clips_root(tmp_path) / "no_log").mkdir()
    clips = data.list_clips(tmp_path)
    assert [c["name"] for c in clips] == ["burner_1", "no_video"]
    assert clips[0] == {"name": "burner_1", "video": True, "skills": ["burner_left_on"],
                        "frames": 5, "alerts": 2, "labelled": True}
    assert clips[1]["video"] is False and clips[1]["alerts"] == 0
    assert data.list_clips(tmp_path / "missing") == []


def test_clips_live_under_data_which_git_ignores():
    assert data.clips_root(Path("data")) == Path("data/playground/clips")
    root = Path(__file__).resolve().parents[1]
    for line in ("data/", "eval/clips/"):  # real footage is never committed, old place too
        assert line in (root / ".gitignore").read_text().splitlines()


def test_clip_timeline(tmp_path):
    data_dir = tmp_path / "data"
    make_clip(data_dir, labels={"true": [[2.5, 4.0], [3, 1], "bad", [0, "x"]]})
    write_spec(data_dir)
    clip = data.clip_timeline(data_dir, "burner_1")
    assert clip["video"] == "media/clips/burner_1/clip.mp4"
    assert clip["duration"] == 4.0
    assert clip["true_ranges"] == [[2.5, 4.0]]  # reversed and malformed ranges dropped
    skill = clip["skills"]["burner_left_on"]
    assert skill["threshold"] == 0.6
    assert skill["points"] == [[0.0, 0.1], [1.0, None], [2.0, 0.8], [3.0, 0.9], [4.0, 0.95]]
    assert skill["alerts"] == [{"t": 3.0, "p": 0.9, "text": "The burner is still on."},
                               {"t": 4.0, "p": 0.95, "text": "Custom words."}]


def test_clip_timeline_without_spec_or_labels(tmp_path):
    make_clip(tmp_path, video=False)
    clip = data.clip_timeline(tmp_path, "burner_1")
    assert clip["video"] is None and clip["true_ranges"] == []
    assert clip["skills"]["burner_left_on"]["alerts"][0]["text"] == "Alert"
    for name in ("missing", "../burner_1", "burner_1/..", ""):
        assert data.clip_timeline(tmp_path, name) is None


def test_results(tmp_path):
    assert data.results(tmp_path) == {"charts": [], "results_md": None}
    (tmp_path / "cost_vs_accuracy.svg").write_text("<svg/>")
    (tmp_path / "exp1-mix.svg").write_text("<svg/>")
    (tmp_path / "notes.txt").write_text("not a chart")
    (tmp_path / "results.md").write_text("# Results\n\nF1 0.9\n")
    out = data.results(tmp_path)
    assert out["charts"] == [
        {"name": "cost vs accuracy", "url": "media/results/cost_vs_accuracy.svg",
         "caption": None},
        {"name": "exp1 mix", "url": "media/results/exp1-mix.svg",
         "caption": "Frame F1 and events caught on time for each mix of training data."}]
    assert out["results_md"].startswith("# Results")
    # Experiment 4's sentence is te eval's summary, numbers from results.json
    (tmp_path / "exp4_cost.svg").write_text("<svg/>")
    judges = [{"variant": v, "metrics": {"f1": f1, "latency_median_s": lat}, "usd_per_hour": usd}
              for v, f1, lat, usd in (("finetuned", 0.88, 0.98, 3.95), ("base", 0.89, 2.07, 3.95),
                                      ("spark", 0.86, 11.16, 16.53))]
    (tmp_path / "results.json").write_text(json.dumps(
        {"footage": "other kitchens (EPIC)", "experiment4": judges}))
    exp4 = data.results(tmp_path)["charts"][-1]
    assert exp4["caption"].startswith("On other kitchens, the fine-tuned model") and "0.98 s" in \
        exp4["caption"]


def test_media_path_serves_only_linked_files(tmp_path):
    data_dir, eval_dir = tmp_path / "data", tmp_path / "eval"
    make_clip(data_dir)
    eval_dir.mkdir()
    (eval_dir / "chart.svg").write_text("<svg/>")
    kf = data_dir / "skills" / "burner_left_on" / "keyframes"
    kf.mkdir(parents=True)
    (kf / "kf_00_1a2b3c4d.jpg").write_bytes(b"\xff\xd8\xff")
    (kf.parent / "spec.json").write_text("{}")
    (data_dir / "playground" / "keyframes").mkdir(parents=True)
    (data_dir / "playground" / "keyframes" / "door.png").write_bytes(b"\x89PNG")
    (tmp_path / ".env").write_text("MODEL_API_KEY=secret")

    def found(path):
        hit = data.media_path(data_dir, eval_dir, path)
        return hit and (hit[0].name, hit[1])

    assert found("clips/burner_1/clip.mp4") == ("clip.mp4", "video/mp4")
    assert found("skills/burner_left_on/keyframes/kf_00_1a2b3c4d.jpg") == ("kf_00_1a2b3c4d.jpg",
                                                                       "image/jpeg")
    assert found("playground/keyframes/door.png") == ("door.png", "image/png")
    assert found("results/chart.svg") == ("chart.svg", "image/svg+xml")
    for path in ("clips/burner_1/log.jsonl", "skills/burner_left_on/spec.json",
                 "skills/burner_left_on/keyframes/../spec.json", "results/../.env",
                 "skills/../../.env", "clips/../../.env/clip.mp4", "results/missing.svg",
                 "skills/burner_left_on/dataset/x.png", "", "results"):
        assert not found(path), path


def blur_folder(src, dst, faces_found=0):
    """te blur's image folder output, with a fake detector."""
    return faces.blur_folder(src, dst, detector=lambda b: [(0, 0, 4, 4)] * faces_found)


def test_public_copy_serves_only_blurred_files(tmp_path):
    from tests.test_teach import png

    data_dir, eval_dir = tmp_path / "data", tmp_path / "eval"
    make_clip(data_dir, blurred=True)
    make_clip(data_dir, "raw_clip")  # no blur record
    swapped = make_clip(data_dir, "swapped", blurred=True)
    (swapped / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42 another video")
    skill = data_dir / "skills" / "burner_left_on"
    (skill / "keyframes").mkdir(parents=True)
    (skill / "synthetic").mkdir()
    (skill / "keyframes" / "kf_00.png").write_bytes(png((200, 10, 10)))
    (skill / "synthetic" / "pos.png").write_bytes(png((10, 200, 10)))
    blur_folder(skill / "keyframes", skill / "keyframes_blurred")
    (skill / "keyframes_blurred" / "dropped_in.png").write_bytes(png((1, 2, 3)))  # not listed
    curated = data_dir / "playground" / "keyframes"
    curated.mkdir(parents=True)
    (curated / "hand_picked.png").write_bytes(png((9, 9, 9)))  # no sources.json yet

    def found(path, public=True):
        hit = data.media_path(data_dir, eval_dir, path, public=public)
        return hit and hit[0].name

    assert found("clips/burner_1/clip.mp4") == "clip.mp4"
    assert found("skills/burner_left_on/keyframes_blurred/kf_00.png") == "kf_00.png"
    for path in ("clips/raw_clip/clip.mp4", "clips/swapped/clip.mp4",
                 "skills/burner_left_on/keyframes/kf_00.png", "skills/burner_left_on/synthetic/pos.png",
                 "skills/burner_left_on/keyframes_blurred/dropped_in.png",
                 "skills/burner_left_on/keyframes_blurred/sources.json",
                 "playground/keyframes/hand_picked.png"):
        assert not found(path), path
    assert found("clips/raw_clip/clip.mp4", public=False) == "clip.mp4"  # the local copy
    assert found("skills/burner_left_on/synthetic/pos.png", public=False) == "pos.png"
    assert not found("skills/burner_left_on/keyframes_blurred/kf_00.png", public=False)
    videos = {c["name"]: c["video"] for c in data.list_clips(data_dir, public=True)}
    assert videos == {"raw_clip": False, "swapped": False, "burner_1": True}
    assert data.clip_timeline(data_dir, "raw_clip", public=True)["video"] is None
    assert data.clip_timeline(data_dir, "burner_1", public=True)["video"].endswith("clip.mp4")

    (tmp_path / "picked").mkdir()
    (tmp_path / "picked" / "door.png").write_bytes(png((50, 60, 70)))
    blur_folder(tmp_path / "picked", curated)
    assert found("playground/keyframes/door.png") == "door.png"
    assert not found("playground/keyframes/hand_picked.png")
    assert data.SOURCES == faces.SOURCES_FILE
