"""Face blurring for anything published. Blurring is pure Pillow and always tested; the
OpenCV detector is tested only where opencv-python-headless is installed."""

import io
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from backend.meta.cache import sha256_hex
from backend.release.faces import (
    SOURCES_FILE,
    blur_boxes,
    blur_folder,
    blur_image,
    blur_video,
    detect_faces,
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
    assert report == {"images": 2, "faces": 2, "per_file": {"kf_00.png": 1, "kf_01.jpg": 1}}
    assert sorted(p.name for p in dst.iterdir()) == ["kf_00.png", "kf_01.jpg", SOURCES_FILE]
    sources = json.loads((dst / SOURCES_FILE).read_text())
    assert sources == {sha256_hex(data): name for name, data in originals.items()}
    assert (dst / "kf_00.png").read_bytes() != originals["kf_00.png"]
    with pytest.raises(ValueError, match="another folder"):
        blur_folder(src, src, detector=lambda data: [])


def test_detection_without_opencv_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)  # import cv2 fails
    with pytest.raises(RuntimeError, match="pip install opencv-python-headless"):
        detect_faces(noise())
    with pytest.raises(RuntimeError, match="opencv-python-headless"):
        blur_image(noise())


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
    assert report == {"frames": 10, "faces": 10, "frames_with_faces": 10} and len(seen) == 10
    assert (tmp_path / "out" / "clip.mp4").stat().st_size > 0
