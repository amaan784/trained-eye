"""The human-labelled test set (backend/release/test_set.py) and its dataset upload. ffmpeg
and huggingface_hub are fakes: nothing is encoded for real or uploaded."""

import json
import sys

import pytest

from backend.release import faces
from backend.release.package import upload
from backend.release.test_set import check_folder, prepare_test_set
from tests.test_faces import fake_ffmpeg
from tests.test_release_package import fake_hub

REPO = "someone/trained-eye-test"


def footage(tmp_path, monkeypatch, labels=None):
    """data/clips/washer_done/ with originals and labels, and blurred/washer_done/ as te
    blur leaves it: a session 3 clip, a session 1 clip, an unblurred copy and a .mov."""
    fake_ffmpeg(monkeypatch)
    clips, blurred = tmp_path / "data/clips/washer_done", tmp_path / "blurred/washer_done"
    clips.mkdir(parents=True)
    for name in ("session3_test_1.mp4", "session1_kitchen.mp4"):
        (clips / name).write_bytes(f"original {name}".encode())
        faces.blur_video(clips / name, blurred / name, fps=10, detector=lambda data: [])
    (clips / "session3_test_1.labels.json").write_text(json.dumps(
        labels or {"clip": "session3_test_1.mp4", "duration_s": 120.0, "true": [[30, 90]]}))
    (blurred / "session3_test_2.mp4").write_bytes(b"never blurred")
    (blurred / "session3_test_3.mov").write_bytes(b"a mov")
    return tmp_path / "data/clips", tmp_path / "blurred/washer_done"


def prepare(tmp_path, labels_root, blurred, **kw):
    return prepare_test_set(tmp_path / "testset", {"washer_done": blurred},
                            **{"license_id": "cc-by-4.0", "labels_root": labels_root,
                               "duration": lambda path: 120.4, **kw})


def test_only_blurred_session3_clips_go_in_with_their_labels(tmp_path, monkeypatch):
    labels_root, blurred = footage(tmp_path, monkeypatch)
    release = prepare(tmp_path, labels_root, blurred)
    out = tmp_path / "testset"
    assert [c["path"] for c in release["clips"]] == ["washer_done/session3_test_1.mp4"]
    assert dict(release["skipped"]) == {
        "washer_done/session1_kitchen.mp4": "not a session 3 test clip (from "
                                            "session1_kitchen.mp4)",
        "washer_done/session3_test_2.mp4": "no blur record from te blur that matches it",
        "washer_done/session3_test_3.mov": "not .mp4 (te blur --dst <name>.mp4)"}
    assert sorted(f["path"] for f in release["files"]) == [
        "README.md", "metadata.jsonl", "washer_done/session3_test_1.labels.json",
        "washer_done/session3_test_1.mp4"]
    assert (out / "washer_done/session3_test_1.mp4").read_bytes() == (
        blurred / "session3_test_1.mp4").read_bytes()
    [row] = [json.loads(line) for line in (out / "metadata.jsonl").read_text().splitlines()]
    assert row == {"file_name": "washer_done/session3_test_1.mp4", "skill": "washer_done",
                   "session": 3, "duration_s": 120.4, "true_ranges": [[30.0, 90.0]],
                   "sha256": faces.file_sha256(blurred / "session3_test_1.mp4"),
                   "labels": "washer_done/session3_test_1.labels.json"}
    labels = json.loads((out / "washer_done/session3_test_1.labels.json").read_text())
    assert labels == {"clip": "session3_test_1.mp4", "duration_s": 120.4, "true": [[30.0, 90.0]]}
    [clip] = release["clips"]
    assert clip["blur"]["output_sha256"] == row["sha256"] and clip["blur"]["audio"] == "removed"
    assert clip["blur"]["source_sha256"] == faces.file_sha256(
        labels_root / "washer_done/session3_test_1.mp4")
    card = (out / "README.md").read_text()
    assert card.startswith("---\nlicense: cc-by-4.0\n") and "chosen by the author" in card
    assert "| washer_done | 1 | 2.0 | 1.0 |" in card  # 120.4 s, 60 s true
    assert "Human labels only" in card and "No model labelled" in card
    assert "audio track" in card and "Faces were blurred" in card
    assert not set(card) & {chr(0x2013), chr(0x2014)}
    assert str(tmp_path) not in (out / "release.json").read_text()  # no folders of the author
    check_folder(out, json.loads((out / "release.json").read_text()))


def test_the_license_is_the_authors_and_the_labels_must_be_this_clips(tmp_path, monkeypatch):
    labels_root, blurred = footage(tmp_path, monkeypatch)
    for license_id in ("", None, "CC BY"):
        with pytest.raises(ValueError, match="no default"):
            prepare(tmp_path, labels_root, blurred, license_id=license_id)
    with pytest.raises(ValueError, match="is 50 s long"):  # labels for a 120 s clip
        prepare(tmp_path, labels_root, blurred, duration=lambda path: 50.0)
    with pytest.raises(ValueError, match="needs ffprobe"):
        prepare(tmp_path, labels_root, blurred, duration=lambda path: None)
    labels = labels_root / "washer_done/session3_test_1.labels.json"
    labels.write_text(json.dumps({"clip": "other.mp4", "duration_s": 120.0, "true": []}))
    with pytest.raises(ValueError, match="not te label's file"):
        prepare(tmp_path, labels_root, blurred)
    labels.unlink()
    with pytest.raises(ValueError, match="no labels"):
        prepare(tmp_path, labels_root, blurred)
    labels.write_text(json.dumps({"clip": "session3_test_1.mp4", "true": [[30, 90]]}))
    (labels_root / "washer_done/session3_test_1.mp4").write_bytes(b"another take")
    with pytest.raises(ValueError, match="blur it again"):  # the record is of another original
        prepare(tmp_path, labels_root, blurred)
    with pytest.raises(ValueError, match="no blurred session 3 test clip"):
        prepare_test_set(tmp_path / "empty", {"washer_done": labels_root / "washer_done"},
                         license_id="cc-by-4.0", duration=lambda path: 120.0)
    assert not (tmp_path / "testset").exists()  # refused before anything was written


def test_the_upload_goes_to_a_private_dataset_repo_and_refuses_anything_unlisted(
        tmp_path, monkeypatch):
    labels_root, blurred = footage(tmp_path, monkeypatch)
    prepare(tmp_path, labels_root, blurred)
    out = tmp_path / "testset"
    with pytest.raises(PermissionError):
        upload(out, REPO)
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub(calls))
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    with pytest.raises(RuntimeError) as err:
        upload(out, REPO, confirmed=True)
    assert "hf_secret_value" not in str(err.value) and "[redacted]" in str(err.value)
    assert calls[0] == (REPO, {"repo_type": "dataset", "private": True, "exist_ok": True})
    monkeypatch.setenv("HF_TOKEN", "hf_good")
    assert upload(out, REPO, confirmed=True) == f"https://huggingface.co/datasets/{REPO}"
    clip = out / "washer_done/session3_test_1.mp4"
    for name, data, match in (("washer_done/frame.jpg", b"", "an image"),
                              ("washer_done/session3_test_9.mp4", b"raw", "never published"),
                              ("raw.mov", b"raw", "never published")):
        (out / name).write_bytes(data)
        with pytest.raises(ValueError, match=match):
            upload(out, REPO, confirmed=True)
        (out / name).unlink()
    good = clip.read_bytes()
    clip.write_bytes(good + b"edited after blurring")
    with pytest.raises(ValueError, match="never published"):
        upload(out, REPO, confirmed=True)
    clip.write_bytes(good)
    card = (out / "README.md").read_text()
    (out / "README.md").write_text(card.replace("license: cc-by-4.0\n", ""))
    with pytest.raises(ValueError, match="license"):
        upload(out, REPO, confirmed=True)
    (out / "README.md").write_text(card)
    (out / "metadata.jsonl").write_text("")
    with pytest.raises(ValueError, match="metadata.jsonl"):
        upload(out, REPO, confirmed=True)
    assert [c for c in calls if c[0] == "upload"] == [("upload", REPO), ("upload", REPO)]
