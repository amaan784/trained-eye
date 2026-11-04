"""Face blurring for anything published. Blurring is pure Pillow and always tested; the
OpenCV detector is tested only where opencv-python-headless is installed."""

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.meta.cache import sha256_hex
from backend.release import faces
from backend.release.faces import (
    SOURCES_FILE,
    blur_boxes,
    blur_folder,
    blur_image,
    blur_video,
    detect_faces,
    image_boxes,
    video_boxes,
)


def noise(fmt="PNG", size=(64, 48), exif=None) -> bytes:
    pixels = np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format=fmt, **({"exif": exif} if exif else {}))
    return buf.getvalue()


def pixels(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).astype(int)


def test_blur_changes_pixels_inside_boxes_only():
    before = noise()
    after = pixels(blur_boxes(before, [(10, 5, 40, 35)]))
    before = pixels(before)
    inside = np.zeros(before.shape[:2], dtype=bool)
    inside[5:35, 10:40] = True
    changed = (after != before).any(axis=2)
    assert changed[inside].mean() > 0.9
    assert not changed[~inside].any()


def test_boxes_are_clipped_and_empty_ones_ignored():
    before = noise()
    assert (pixels(blur_boxes(before, [(70, 0, 90, 10), (5, 5, 5, 20)])) == pixels(before)).all()
    changed = (pixels(blur_boxes(before, [(-10, -10, 8, 8)])) != pixels(before)).any(axis=2)
    assert changed[:8, :8].any() and not changed[8:, :].any() and not changed[:, 8:].any()
    with pytest.raises(ValueError):
        blur_boxes(before, [], strength=0)


def test_exif_is_left_out_and_its_rotation_kept():
    exif = Image.Exif()
    exif[0x010F] = "TestCam"  # Make
    exif[0x0112] = 6  # Orientation: turn 90 degrees to display
    out = blur_boxes(noise("JPEG", exif=exif.tobytes()), [])
    img = Image.open(io.BytesIO(out))
    assert img.format == "JPEG" and img.size == (48, 64)  # the turn is in the pixels now
    assert not img.getexif() and "exif" not in img.info


def test_blur_folder_writes_blurred_copies_and_the_source_hashes(tmp_path):
    src, dst = tmp_path / "keyframes", tmp_path / "published"
    src.mkdir()
    originals = {"kf_00.png": noise(), "kf_01.jpg": noise("JPEG")}
    for name, data in originals.items():
        (src / name).write_bytes(data)
    (src / "notes.txt").write_text("not an image")
    (src / "._kf_00.png").write_bytes(b"macOS leftovers")
    report = blur_folder(src, dst, detector=lambda data: [(0, 0, 20, 20)])
    assert report == {"images": 2, "faces": 2, "per_file": {"kf_00.png": 1, "kf_01.jpg": 1},
                      "manual_boxes": 0}
    assert sorted(p.name for p in dst.iterdir()) == ["kf_00.png", "kf_01.jpg", SOURCES_FILE]
    sources = json.loads((dst / SOURCES_FILE).read_text())
    assert sources == {sha256_hex(data): name for name, data in originals.items()}
    assert (dst / "kf_00.png").read_bytes() != originals["kf_00.png"]
    with pytest.raises(ValueError, match="another folder"):
        blur_folder(src, src, detector=lambda data: [])


def test_detection_without_opencv_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)  # import cv2 fails
    with pytest.raises(RuntimeError, match='pip install "opencv-python-headless<5"'):
        detect_faces(noise())
    with pytest.raises(RuntimeError, match="opencv-python-headless"):
        blur_image(noise())


def test_an_opencv_without_the_haar_detector_says_which_to_install(monkeypatch):
    # OpenCV 5.0's headless build (Nov 4) has no CascadeClassifier: te blur failed with an
    # AttributeError inside detect_faces.
    import types

    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(__version__="5.0.0"))
    with pytest.raises(RuntimeError, match=r'OpenCV 5\.0\.0 has no Haar cascade.*"opencv-'
                                           r'python-headless<5"'):
        detect_faces(noise())


def test_detection_finds_nothing_in_a_plain_scene():
    pytest.importorskip("cv2")
    blank = io.BytesIO()
    Image.new("RGB", (160, 120), (90, 120, 150)).save(blank, format="PNG")
    assert detect_faces(blank.getvalue()) == []
    out, found = blur_image(blank.getvalue())
    assert found == 0 and (pixels(out) == pixels(blank.getvalue())).all()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_blur_video_blurs_every_frame(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=size=64x48:rate=10:duration=1", "-pix_fmt", "yuv420p", str(clip)],
                   check=True)
    seen = []
    report = blur_video(clip, tmp_path / "out" / "clip.mp4", fps=10,
                        detector=lambda data: seen.append(1) or [(0, 0, 16, 16)])
    assert (report["frames"], report["faces"], report["frames_with_faces"]) == (10, 10, 10)
    assert len(seen) == 10 and (tmp_path / "out" / "clip.mp4").stat().st_size > 0
    record = json.loads((tmp_path / "out" / "clip.mp4.blur.json").read_text())
    assert record == report and record["output_sha256"] == faces.file_sha256(
        tmp_path / "out" / "clip.mp4")


def test_manual_boxes_are_added_to_the_detectors_in_a_folder(tmp_path):
    src = tmp_path / "keyframes"
    src.mkdir()
    (src / "kf_00.png").write_bytes(noise())
    (src / "kf_01.png").write_bytes(noise())
    boxes = image_boxes({"kf_01.png": [[30, 20, 50, 40]]})
    report = blur_folder(src, tmp_path / "out", detector=lambda data: [], extra=boxes)
    assert report["faces"] == 0 and report["manual_boxes"] == 1
    assert (pixels((tmp_path / "out/kf_00.png").read_bytes()) == pixels(noise())).all()
    changed = (pixels((tmp_path / "out/kf_01.png").read_bytes()) != pixels(noise())).any(axis=2)
    assert changed[20:40, 30:50].mean() > 0.9 and not changed[:20].any()
    with pytest.raises(ValueError, match="not images"):  # a typo must not leave a face
        blur_folder(src, tmp_path / "out", detector=lambda data: [],
                    extra={"kf_1.png": [(0, 0, 4, 4)]})


@pytest.mark.parametrize("raw", [
    [], {"a.png": [[0, 0, 0, 10]]}, {"a.png": [[0, 0, 10]]}, {"a.png": [[0, 0, 10, True]]},
    {"a.png": [0, 0, 10, 10]}, {"a.png": [[0, 0, float("nan"), 10]]}])
def test_bad_image_boxes_are_refused(raw):
    with pytest.raises((TypeError, ValueError)):
        image_boxes(raw)


@pytest.mark.parametrize("raw", [
    {}, [{"start_s": 0, "end_s": 1}], [{"start_s": 2, "end_s": 1, "box": [0, 0, 4, 4]}],
    [{"start_s": -1, "end_s": 1, "box": [0, 0, 4, 4]}],
    [{"start": 0, "end_s": 1, "box": [0, 0, 4, 4]}],
    [{"start_s": 0, "end_s": 1, "box": [0, 0, 4, 4], "note": "x"}]])
def test_bad_video_boxes_are_refused(raw):
    with pytest.raises((TypeError, ValueError)):
        video_boxes(raw)


def fake_ffmpeg(monkeypatch, frames=10, calls=None):
    """ffmpeg without ffmpeg: decoding writes `frames` PNGs, encoding joins them."""
    def run(*args):
        (calls if calls is not None else []).append(args[0])
        if args[0] == "-i":
            out = Path(args[-1])
            for i in range(frames):
                (out.parent / f"f{i + 1:06d}.png").write_bytes(noise(size=(32, 24)))
        else:
            folder = Path(args[args.index("-i") + 1]).parent
            Path(args[-1]).write_bytes(b"".join(p.read_bytes()
                                                for p in sorted(folder.glob("f*.png"))))
    monkeypatch.setattr(faces, "_ffmpeg", run)
    monkeypatch.setattr(faces.shutil, "which", lambda name: f"/usr/bin/{name}")


def test_blur_video_writes_its_blur_record_with_the_manual_boxes(tmp_path, monkeypatch):
    fake_ffmpeg(monkeypatch)
    src, dst = tmp_path / "session3_test_1.mp4", tmp_path / "out" / "session3_test_1.mp4"
    src.write_bytes(b"original video bytes")
    spans = video_boxes([{"start_s": 0.2, "end_s": 0.4, "box": [0, 0, 8, 8]}])
    detected = iter([[(1, 1, 5, 5)]] + [[]] * 9)
    record = blur_video(src, dst, fps=10, detector=lambda data: next(detected), extra=spans)
    assert record["source"] == src.name and record["source_sha256"] == faces.file_sha256(src)
    assert record["output_sha256"] == faces.file_sha256(dst) and record["fps"] == 10
    assert record["faces_per_frame"] == [1] + [0] * 9 and record["faces"] == 1
    assert record["manual_boxes"] == spans and record["frames_with_manual_boxes"] == 3
    assert (record["audio"], record["metadata"]) == ("removed", "removed")
    assert json.loads(Path(f"{dst}.blur.json").read_text()) == record
    assert str(tmp_path) not in json.dumps(record)  # names only, no folders of the author


def test_a_manual_box_on_no_frame_is_refused_before_anything_is_written(tmp_path,
                                                                        monkeypatch):
    calls = []
    fake_ffmpeg(monkeypatch, frames=10, calls=calls)
    src, dst = tmp_path / "clip.mp4", tmp_path / "out.mp4"
    src.write_bytes(b"original")
    blur_video(src, dst, fps=10, detector=lambda data: [])
    before = Path(f"{dst}.blur.json").read_text()
    late = [{"start_s": 5.0, "end_s": 6.0, "box": [0, 0, 8, 8]}]  # the clip is 1 s long
    with pytest.raises(ValueError, match="fall on no frame"):
        blur_video(src, dst, fps=10, detector=lambda data: [], extra=late)
    assert calls == ["-i", "-framerate", "-i"]  # decoded, never encoded
    assert Path(f"{dst}.blur.json").read_text() == before  # the earlier output keeps its record


def test_a_manual_box_off_the_picture_is_refused_before_anything_is_written(tmp_path,
                                                                           monkeypatch):
    src = tmp_path / "keyframes"
    src.mkdir()
    (src / "kf_00.png").write_bytes(noise())  # 64x48
    with pytest.raises(ValueError, match="outside the 64x48 picture"):
        blur_folder(src, tmp_path / "out", detector=lambda data: [],
                    extra=image_boxes({"kf_00.png": [[100, 10, 140, 30]]}))
    assert not (tmp_path / "out").exists()
    fake_ffmpeg(monkeypatch, frames=10)  # 32x24 frames
    clip, dst = tmp_path / "clip.mp4", tmp_path / "out.mp4"
    clip.write_bytes(b"original")
    off = [{"start_s": 0.0, "end_s": 0.5, "box": [40, 0, 60, 20]}]
    with pytest.raises(ValueError, match="outside the 32x24 frame"):
        blur_video(clip, dst, fps=10, detector=lambda data: [], extra=off)
    assert not dst.exists() and not Path(f"{dst}.blur.json").exists()
