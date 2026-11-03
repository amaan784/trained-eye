"""Labelled session clips as real examples (backend/dataengine/realclips.py) and te clips.
Frame folders stand in for the recorded videos: frame i of a folder is at i seconds."""

import json
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from backend import cli
from backend.dataengine.realclips import (
    find_clips,
    frame_answers,
    labels_path,
    write_clip_dataset,
)
from backend.skills.spec import SkillSpec, Target

SPEC = SkillSpec(name="washer_done", request="tell me when the washer says End",
                 target=Target(sam_prompt="washer display", description="the washer display"),
                 positive="the display shows End", negatives=["display off"],
                 alert="Laundry's done.")


def skill_folder(root, spec=SPEC):
    folder = root / spec.name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "spec.json").write_text(spec.model_dump_json())
    return folder


def frame_clip(folder, seconds, labels=None, seed=0):
    """A frame folder of `seconds` frames (one a second) and, when given, its labels."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for i in range(seconds):
        blocks = rng.integers(0, 255, size=(6, 8), dtype=np.uint8)
        img = Image.fromarray(blocks).resize((160, 120), Image.NEAREST).convert("RGB")
        img.save(folder / f"f{i:03d}.jpg", format="JPEG", quality=90)
    if labels is not None:
        labels_path(folder).write_text(json.dumps(labels))
    return folder


def session_clips(clips):
    """Session 1: 30 s, true from 10 to 20. Session 2: 12 s, true from the start to 5 s, in
    a hand-written file without clip or duration. Plus clips that are never read."""
    frame_clip(clips / "session1_kitchen", 30,
               {"clip": "session1_kitchen", "duration_s": 30.0, "true": [[10, 20]]})
    frame_clip(clips / "session2_evening", 12, {"true": [[0, 5]]}, seed=1)
    frame_clip(clips / "session1_unlabelled", 5, seed=2)
    frame_clip(clips / "session3_test_a", 5, {"true": [[1, 2]]}, seed=3)
    return clips


def test_answers_inside_and_outside_the_ranges_with_a_gap_at_each_change():
    times = [float(t) for t in range(30)]
    answers = frame_answers(times, [[10.0, 20.0]], 30.0)
    assert [t for t, a in zip(times, answers, strict=True) if a is None] == [9, 10, 11, 19, 20,
                                                                             21]
    assert [t for t, a in zip(times, answers, strict=True) if a == "yes"] == list(range(12, 19))
    assert answers[:9] == ["no"] * 9 and answers[22:] == ["no"] * 8
    # true from the start or to the end: no change there, so nothing is left out
    assert frame_answers([0.0, 1.0, 4.0], [[0.0, 5.0]], 5.0) == ["yes", "yes", "yes"]
    assert frame_answers([0.0, 5.0], [], 6.0) == ["no", "no"]
    assert frame_answers([12.5], [[10.0, 11.0]], 20.0, guard_s=1.5) == [None]  # 1.5: inclusive


def test_only_session_1_and_2_clips_are_found(tmp_path):
    clips = session_clips(tmp_path / "clips")
    (clips / "session1_kitchen.mp4").write_bytes(b"")
    (clips / "session1").mkdir()  # no underscore: not a session clip
    (clips / "notes.txt").write_text("")
    found = [(c.path.name, c.session, c.split) for c in find_clips(clips)]
    assert found == [("session1_kitchen", "session1", "train"),
                     ("session1_kitchen.mp4", "session1", "train"),
                     ("session1_unlabelled", "session1", "train"),
                     ("session2_evening", "session2", "val")]
    assert find_clips(tmp_path / "none") == []
    assert labels_path(clips / "session1_kitchen.mp4") == clips / "session1_kitchen.labels.json"


def test_labelled_clips_become_real_examples_split_by_session(tmp_path):
    skill = skill_folder(tmp_path / "skills")
    manifest = write_clip_dataset(skill, session_clips(tmp_path / "clips"))
    out = skill / "dataset-clips"
    assert (out / "washer_done-clips.jsonl").is_file()
    assert json.loads((out / "washer_done-clips.manifest.json").read_text()) == manifest
    # session 1: 30 frames less 6 near a change; session 2: 12 frames less 3 around 5 s
    assert manifest["examples"] == 24 + 9 and manifest["splits"] == {"train": 24, "val": 9}
    assert manifest["counts"] == {"washer_done/real/no": 17 + 5, "washer_done/real/yes": 7 + 4}
    by_clip = {c["clip"]: c for c in manifest["clips"]}
    assert by_clip["session1_kitchen"] == {
        "clip": "session1_kitchen", "session": "session1", "split": "train",
        "labels": "session1_kitchen.labels.json", "frames": 30, "examples": 24, "yes": 7,
        "no": 17, "near_a_change": 6}
    assert set(by_clip) == {"session1_kitchen", "session2_evening"}  # never session 3
    [skipped] = manifest["skipped"]
    assert skipped["clip"] == "session1_unlabelled" and "te label --clip" in skipped["reason"]
    index = {e["id"]: e for e in manifest["examples_index"]}
    entry = index["washer_done-real-session1_kitchen-00015.0s"]
    assert {k: entry[k] for k in ("kind", "clip", "t_s", "session", "split", "source",
                                  "answer")} == {
        "kind": "clip", "clip": "session1_kitchen", "t_s": 15.0, "session": "session1",
        "split": "train", "source": "real", "answer": "yes"}
    assert {e["split"] for e in index.values() if e["session"] == "session2"} == {"val"}
    assert "washer_done-real-session1_kitchen-00010.0s" not in index  # at the change
    record = json.loads((out / "washer_done-clips.jsonl").read_text().splitlines()[0])
    assert record["messages"][1]["content"][1]["text"].startswith(
        "Condition: the display shows End. Not: display off.")
    again = write_clip_dataset(skill, tmp_path / "clips")
    assert again["jsonl_sha256"] == manifest["jsonl_sha256"]  # deterministic


def test_clips_need_a_spec_usable_labels_and_the_teach_layout(tmp_path):
    clips = tmp_path / "clips"
    with pytest.raises(ValueError, match="no spec yet: run te teach"):
        write_clip_dataset(tmp_path / "skills/washer_done", clips)
    skill = skill_folder(tmp_path / "skills")
    with pytest.raises(ValueError, match="no session1_. or session2_. clips"):
        write_clip_dataset(skill, clips)
    frame_clip(clips / "session1_a", 4, {"clip": "other.mp4", "duration_s": 4, "true": []})
    frame_clip(clips / "session1_b", 4, {"duration_s": 3, "true": [[1, 9]]}, seed=1)
    frame_clip(clips / "session2_c", 4, {"duration_s": 4, "true": [[0, 2]]}, seed=2)
    labels_path(clips / "session2_d").write_text("not json")
    frame_clip(clips / "session2_d", 4, seed=3)
    with pytest.raises(ValueError) as err:
        write_clip_dataset(skill, clips, guard_s=3)
    message = str(err.value)
    assert "session1_a: its labels are for other.mp4" in message
    assert "session1_b: range [1, 9] must have 0 <= start < end <= 3" in message
    assert "session2_c: every frame is near a change" in message and "session2_d: " in message
    assert not (skill / "dataset-clips").exists()

    from tests.test_training_prep import teach_dataset

    teach_dataset(tmp_path / "skills", "washer_done", system_in_user=True)
    with pytest.raises(ValueError, match="TE_SYSTEM_IN_USER=1, but it is 0 now"):
        write_clip_dataset(skill, clips, system_in_user=False)
    manifest = write_clip_dataset(skill, clips)  # the teach dataset's layout
    assert manifest["system_in_user"] is True and manifest["splits"] == {"train": 0, "val": 1}


def test_te_clips_writes_the_dataset_and_says_what_it_left_out(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TE_DATA_DIR", str(tmp_path / "data"))
    assert cli.main(["clips", "--skill", "washer_done"]) == 2
    assert "no spec yet" in capsys.readouterr().err
    skill_folder(tmp_path / "data/skills")
    session_clips(tmp_path / "data/clips/washer_done")
    assert cli.main(["clips", "--skill", "washer_done"]) == 0
    out = capsys.readouterr().out
    assert "  session1_kitchen (train): 24 of 30 frames, 7 yes and 17 no; 6 near a change" in out
    assert "  left out session1_unlabelled: no session1_unlabelled.labels.json" in out
    assert ("washer_done: dataset-clips with 33 examples (24 from session 1 to train, 9 from "
            "session 2 to validate) -> data/skills/washer_done/dataset-clips") in out
    elsewhere = frame_clip(tmp_path / "drive/session1_a", 3, {"true": []})
    assert cli.main(["clips", "--skill", "washer_done", "--clips", str(elsewhere.parent)]) == 0
    assert "no session 2 frames, so the clips hold nothing out" in capsys.readouterr().out
    assert cli.main(["clips", "--skill", "Bad-Name"]) == 2


def test_a_video_ffmpeg_cannot_read_is_left_out(tmp_path, monkeypatch):
    from backend.dataengine import realclips

    real_load = realclips.load_frames

    def load(path, fps):
        if path.suffix == ".mp4":
            raise subprocess.CalledProcessError(1, ["ffmpeg"])
        return real_load(path, fps)

    monkeypatch.setattr(realclips, "load_frames", load)
    clips = session_clips(tmp_path / "clips")
    (clips / "session2_broken.mp4").write_bytes(b"not a video")
    labels_path(clips / "session2_broken.mp4").write_text('{"true": []}')
    manifest = write_clip_dataset(skill_folder(tmp_path / "skills"), clips)
    assert {"clip": "session2_broken.mp4", "reason": "ffmpeg cannot read it"} in manifest[
        "skipped"]
    assert manifest["splits"] == {"train": 24, "val": 9}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_a_video_is_sampled_once_a_second(tmp_path):
    frames = frame_clip(tmp_path / "src", 60)  # 6 s at 10 fps
    clips = tmp_path / "clips"
    clips.mkdir()
    video = clips / "session1_washer.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", "10", "-i", str(frames / "f%03d.jpg"),
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    labels_path(video).write_text(json.dumps(
        {"clip": "session1_washer.mp4", "duration_s": 6.0, "true": [[3.0, 6.0]]}))
    manifest = write_clip_dataset(skill_folder(tmp_path / "skills"), clips)
    times = {e["t_s"]: e["answer"] for e in manifest["examples_index"]}
    assert times == {0.0: "no", 1.0: "no", 5.0: "yes"}  # 2 to 4 s: near the change at 3 s
