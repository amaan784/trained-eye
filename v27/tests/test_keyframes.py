import io
import os
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image, ImageFilter

from backend.dataengine.keyframes import (
    dhash,
    hamming,
    load_frames,
    select_keyframes,
    sharpness,
)


def scene(seed: int, blur: float = 0.0, size=(320, 240)) -> bytes:
    """A textured synthetic frame; different seeds look different, blur lowers sharpness."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 255, size=(6, 8), dtype=np.uint8)
    img = Image.fromarray(blocks).resize(size, Image.NEAREST).convert("RGB")
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def write_frames(folder, frames):
    folder.mkdir(parents=True, exist_ok=True)
    for i, data in enumerate(frames):
        (folder / f"frame_{i:03d}.jpg").write_bytes(data)
    return folder


def test_sharpness_and_hash():
    a =np.asarray(Image.open(io.BytesIO(scene(1))).convert("L"), dtype=np.float32)
    b = np.asarray(Image.open(io.BytesIO(scene(1, blur=4))).convert("L"), dtype=np.float32)
    assert sharpness(a) > 5 * sharpness(b)
    assert hamming(dhash(a), dhash(a)) == 0
    c = np.asarray(Image.open(io.BytesIO(scene(2))).convert("L"), dtype=np.float32)
    assert hamming(dhash(a), dhash(c)) > 6


def test_select_drops_blur_and_duplicates(tmp_path):
    frames = []
    for seed in range(30):
        frames.append(scene(seed))  # 30 distinct sharp frames
        frames.append(scene(seed))  # exact duplicate
    frames += [scene(100 + i, blur=5) for i in range(10)]  # blurry ones
    loaded = load_frames(write_frames(tmp_path / "clip", frames))
    picked = select_keyframes(loaded)
    assert 20 <= len(picked) <= 30
    assert [f.index for f in picked] == sorted(f.index for f in picked)
    hashes = [dhash(f.gray) for f in picked]
    assert len(set(hashes)) == len(hashes)
    assert all(f.index < 60 for f in picked)  # no blurry frame chosen


def test_small_clip_returns_what_it_can(tmp_path):
    loaded = load_frames(write_frames(tmp_path / "clip", [scene(i) for i in range(5)]))
    assert 1 <= len(select_keyframes(loaded)) <= 5
    assert select_keyframes([]) == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_video_file(tmp_path):
    frames = write_frames(tmp_path / "src", [scene(i % 7) for i in range(20)])
    video = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", "10", "-i",
                    str(frames / "frame_%03d.jpg"), "-pix_fmt", "yuv420p", str(video)],
                   check=True)
    loaded = load_frames(video, fps=5)
    assert 8 <= len(loaded) <= 12
    assert loaded[1].t_s == pytest.approx(0.2)


def test_static_clip_gives_one_keyframe_not_copies(tmp_path):
    loaded = load_frames(write_frames(tmp_path / "clip", [scene(7)] * 30))
    assert len(select_keyframes(loaded)) == 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_decoded_frames_carry_no_encoder_version(tmp_path):
    frames = write_frames(tmp_path / "src", [scene(i % 3) for i in range(6)])
    video = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", "6", "-i",
                    str(frames / "frame_%03d.jpg"), "-pix_fmt", "yuv420p", str(video)],
                   check=True)
    loaded = load_frames(video, fps=3)
    assert loaded and all(b"Lavc" not in f.data for f in loaded)


def test_folder_frames_with_exif_orientation_are_turned_upright(tmp_path):
    upright = Image.new("RGB", (48, 64), (255, 0, 0))
    upright.paste((0, 0, 255), (0, 32, 48, 64))  # red top, blue bottom
    exif = Image.Exif()
    exif[0x0112] = 6  # stored turned, as phones and the glasses do
    (tmp_path / "clip").mkdir()
    upright.transpose(Image.Transpose.ROTATE_90).save(tmp_path / "clip/a.jpg", quality=95,
                                                      exif=exif)
    upright.save(tmp_path / "clip/b.jpg", quality=95)
    frames = load_frames(tmp_path / "clip")
    with Image.open(io.BytesIO(frames[0].data)) as img:
        assert img.size == (48, 64) and img.getexif().get(0x0112, 1) == 1
        assert img.getpixel((24, 8))[0] > 200 and img.getpixel((24, 56))[2] > 200
    assert frames[0].gray.shape == (427, 320)
    assert frames[1].data == (tmp_path / "clip/b.jpg").read_bytes()  # untagged: untouched


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_video_decode_leaves_stdin_alone(tmp_path):
    frames = write_frames(tmp_path / "src", [scene(i % 3) for i in range(6)])
    video = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", "6", "-i",
                    str(frames / "frame_%03d.jpg"), "-pix_fmt", "yuv420p", str(video)],
                   check=True)
    read_end, write_end = os.pipe()
    os.write(write_end, b"y\n" * 50)  # answers meant for the prompts
    os.close(write_end)
    saved = os.dup(0)
    try:
        os.dup2(read_end, 0)
        load_frames(video, fps=3)
        assert os.read(0, 1000) == b"y\n" * 50
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        os.close(read_end)
