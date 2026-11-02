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


@pytest.mark.parametrize("orientation", range(2, 9))
def test_every_exif_orientation_turns_like_pillow_does(tmp_path, orientation):
    from PIL import ImageOps

    stored = Image.new("RGB", (40, 24), (0, 0, 0))
    stored.paste((255, 0, 0), (0, 0, 20, 12))  # one red corner shows any turn or flip
    exif = Image.Exif()
    exif[0x0112] = orientation
    (tmp_path / "clip").mkdir()
    stored.save(tmp_path / "clip/a.png", exif=exif)
    [frame] = load_frames(tmp_path / "clip")
    with Image.open(tmp_path / "clip/a.png") as original:
        expected = ImageOps.exif_transpose(original).convert("RGB")
    with Image.open(io.BytesIO(frame.data)) as img:
        assert img.size == expected.size
        corners = [(1, 1), (img.width - 2, 1), (1, img.height - 2),
                   (img.width - 2, img.height - 2)]
        assert [img.getpixel(c)[0] > 128 for c in corners] == [
            expected.getpixel(c)[0] > 128 for c in corners]


def test_frame_folders_skip_hidden_files_and_read_webp(tmp_path):
    clip = write_frames(tmp_path / "clip", [scene(1), scene(2)])
    (clip / "._frame_000.jpg").write_bytes(b"\x00\x05\x16\x07AppleDouble")  # macOS, on exFAT
    Image.open(io.BytesIO(scene(3))).save(clip / "frame_002.webp")
    (clip / "notes.txt").write_text("take 2")
    frames = load_frames(clip)
    assert len(frames) == 3 and frames[2].data[8:12] == b"WEBP"


def test_unreadable_or_heic_frames_say_what_is_wrong(tmp_path):
    clip = write_frames(tmp_path / "clip", [scene(1)])
    (clip / "frame_001.jpg").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="frame_001.jpg is not a readable image"):
        load_frames(clip)
    (tmp_path / "heic").mkdir()
    (tmp_path / "heic/IMG_0001.HEIC").write_bytes(b"\x00\x00\x00\x18ftypheic")
    with pytest.raises(ValueError, match="HEIC"):
        load_frames(tmp_path / "heic")


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
