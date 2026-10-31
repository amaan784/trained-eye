"""Pick 20-30 sharp, diverse keyframes from a ~10 second teaching clip.

The clip is either a video file (decoded with ffmpeg) or a folder of JPEG, PNG or WebP
frames as the iPhone app streams them (HEIC photos need converting first; hidden files
such as the "._" copies macOS leaves on external drives are ignored). Blurry frames are dropped by the variance of the
Laplacian; near-duplicates by the Hamming distance between difference hashes.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
UNSUPPORTED_SUFFIXES = (".heic", ".heif")
EXIF_ORIENTATION = 0x0112
TURNS = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
         4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
         6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
         8: Image.Transpose.ROTATE_90}


@dataclass
class Frame:
    index: int
    t_s: float
    data: bytes  # encoded image as captured, upright (an EXIF Orientation turn baked in)
    gray: np.ndarray = field(repr=False)  # small grayscale copy for scoring


def _gray(data: bytes, width: int = 320) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("L")
        height = max(1, round(img.height * width / img.width))
        return np.asarray(img.resize((width, height)), dtype=np.float32)


def upright(data: bytes) -> bytes:
    """Bake an EXIF Orientation turn into the pixels, as ffmpeg does for video.

    Phone and glasses photos are often stored sideways with a tag saying how to turn
    them. Untagged images come back byte for byte, so their cache keys stay the same.
    """
    with Image.open(io.BytesIO(data)) as img:
        turn = TURNS.get(img.getexif().get(EXIF_ORIENTATION, 1))
        if turn is None:
            return data
        # Turned by the tag alone: the rest of the EXIF is not rewritten (or kept).
        turned = img.convert("RGB").transpose(turn)
    buf = io.BytesIO()
    turned.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def load_frames(source: Path | str, fps: float = 3.0) -> list[Frame]:
    source = Path(source)
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.is_file() and not p.name.startswith("."))
        paths = [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES]
        if not paths and any(p.suffix.lower() in UNSUPPORTED_SUFFIXES for p in files):
            raise ValueError(f"{source} holds HEIC photos, which cannot be read: export "
                             "them as JPEG (or record a video)")
        frames = []
        for i, path in enumerate(paths):
            try:
                data = upright(path.read_bytes())
                frames.append(Frame(i, i / fps, data, _gray(data)))
            except (UnidentifiedImageError, OSError, SyntaxError) as err:
                raise ValueError(f"{path} is not a readable image ({err})") from None
        return frames
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed to read video files (brew install ffmpeg)")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            # bitexact: no encoder version in the JPEGs, so cache keys survive upgrades
            # -nostdin and no stdin: a backgrounded run must not stop on the terminal, and
            # answers piped to the prompts must reach them
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-vf", f"fps={fps}",
             "-fflags", "+bitexact", "-flags:v", "+bitexact", "-q:v", "2",
             str(Path(tmp) / "f%05d.jpg")],
            stdin=subprocess.DEVNULL, check=True,
        )
        return load_frames(Path(tmp), fps)


def sharpness(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian; low means blurry."""
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
           + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(lap.var())


def dhash(gray: np.ndarray, size: int = 8) -> int:
    img = Image.fromarray(gray.astype(np.uint8)).resize((size + 1, size))
    px = np.asarray(img, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def select_keyframes(
    frames: list[Frame],
    target: int = 24,
    min_count: int = 20,
    max_count: int = 30,
    blur_drop: float = 0.25,
    min_distance: int = 6,
) -> list[Frame]:
    """Sharpest-first greedy selection of distinct frames, returned in time order.

    Drops the blurriest `blur_drop` fraction, then adds frames whose hash differs from
    every chosen frame by at least `min_distance` bits. If that yields fewer than
    `min_count`, the distance is relaxed step by step, but never to zero: frames with
    identical hashes are never both chosen, so a static clip gives few keyframes.
    """
    if not frames:
        return []
    scores = [sharpness(f.gray) for f in frames]
    order = sorted(range(len(frames)), key=lambda i: scores[i], reverse=True)
    keep = order[: max(1, round(len(order) * (1 - blur_drop)))]
    hashes = {i: dhash(frames[i].gray) for i in keep}
    want = min(max(target, min_count), max_count, len(keep))

    chosen: list[int] = []
    for distance in range(min_distance, 0, -1):
        for i in keep:
            if len(chosen) >= want:
                break
            if i in chosen:
                continue
            if all(hamming(hashes[i], hashes[j]) >= distance for j in chosen):
                chosen.append(i)
        if len(chosen) >= min(min_count, len(keep)):
            break
    return [frames[i] for i in sorted(chosen)]
