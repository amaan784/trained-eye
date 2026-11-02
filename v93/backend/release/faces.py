"""Blur faces in anything published: README images, the playground, the video, the test set.

`blur_boxes` blurs given boxes with Pillow alone. `detect_faces` finds faces with the Haar
cascades OpenCV ships (frontal, and profile facing either way); OpenCV is imported only
then: `pip install opencv-python-headless` (on the approved list). Haar cascades miss
small, dark, tilted or covered faces, so look at every published image as well: this is
a first pass, not a guarantee.

Output is rebuilt from the pixels alone: EXIF (GPS, camera, time), XMP, ICC and text
chunks are left out, and an EXIF rotation is applied to the pixels first. Use it on real
footage only: Muse Image outputs are never published, and their provenance metadata must
never be stripped.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from backend.meta.cache import sha256_hex

Box = tuple[int, int, int, int]  # left, top, right, bottom in pixels (right, bottom excluded)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
SOURCES_FILE = "sources.json"  # from blur_folder: SHA-256 of each original -> blurred file
STRENGTH = 0.25  # Gaussian radius as a share of the box's shorter side
BLOCK_PX = 8  # a box is first reduced to this many pixels across its shorter side
MARGIN = 0.2  # a detected face box grows by this share on every side (hair, ears, chin)
CASCADES = ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml")
INSTALL = "pip install opencv-python-headless"
Detector = Callable[[bytes], list[Box]]


def _open(data: bytes) -> tuple[Image.Image, str]:
    with Image.open(io.BytesIO(data)) as img:
        fmt = img.format or "PNG"
        fmt = "JPEG" if fmt == "MPO" else fmt  # iPhone photos: a JPEG with extra frames
        img = ImageOps.exif_transpose(img).convert("RGB")
    # a fresh image from the pixels alone, so no metadata can come along
    return Image.frombytes("RGB", img.size, img.tobytes()), fmt


def _encode(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=95)
    elif fmt == "WEBP":
        img.save(buf, format="WEBP", lossless=True)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def _clip(box, size: tuple[int, int]) -> Box | None:
    left, top, right, bottom = (round(float(v)) for v in box)
    left, top = max(0, left), max(0, top)
    right, bottom = min(size[0], right), min(size[1], bottom)
    return (left, top, right, bottom) if right > left and bottom > top else None


def blur_boxes(image_bytes: bytes, boxes: Iterable[Box], strength: float = STRENGTH) -> bytes:
    """The image with each box blurred beyond recognition and nothing else changed. JPEG
    stays JPEG (quality 95, so other pixels shift by compression only); PNG and WebP are
    written losslessly. Boxes are clipped to the image; empty ones are ignored."""
    if not 0 < strength <= 1:
        raise ValueError("strength must be above 0 and at most 1")
    img, fmt = _open(image_bytes)
    for box in boxes:
        clipped = _clip(box, img.size)
        if clipped is None:
            continue
        region = img.crop(clipped)
        w, h = region.size
        scale = BLOCK_PX / min(w, h)
        if scale < 1:  # the detail is gone, not only softened
            small = region.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                  Image.Resampling.BOX)
            region = small.resize((w, h), Image.Resampling.BILINEAR)
        region = region.filter(ImageFilter.GaussianBlur(max(1.0, strength * min(w, h))))
        img.paste(region, clipped[:2])
    return _encode(img, fmt)


def _cv2():
    try:
        import cv2
    except ImportError:
        raise RuntimeError(f"face detection needs OpenCV: {INSTALL} (on the approved list), "
                           "then run this again") from None
    return cv2


def detect_faces(image_bytes: bytes, min_size: int = 20, margin: float = MARGIN) -> list[Box]:
    """Face boxes from OpenCV's Haar cascades, each grown by `margin`, in the coordinates
    blur_boxes uses (EXIF rotation applied)."""
    cv2 = _cv2()
    img, _ = _open(image_bytes)
    gray = cv2.equalizeHist(np.asarray(img.convert("L")))
    width = gray.shape[1]
    found = set()
    for name in CASCADES:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if cascade.empty():
            raise RuntimeError(f"OpenCV's {name} did not load; reinstall it: {INSTALL}")
        for flipped in (False, True) if "profile" in name else (False,):
            view = np.ascontiguousarray(gray[:, ::-1]) if flipped else gray
            for x, y, w, h in cascade.detectMultiScale(view, scaleFactor=1.1, minNeighbors=5,
                                                       minSize=(min_size, min_size)):
                x = width - x - w if flipped else x
                dx, dy = w * margin, h * margin
                box = _clip((x - dx, y - dy, x + w + dx, y + h + dy), img.size)
                if box:
                    found.add(box)
    return sorted(found)


def blur_image(image_bytes: bytes, strength: float = STRENGTH,
               detector: Detector = detect_faces) -> tuple[bytes, int]:
    """(the image with every detected face blurred, faces found). Always re-encoded, so
    the metadata is left out even when no face is found."""
    boxes = detector(image_bytes)
    return blur_boxes(image_bytes, boxes, strength), len(boxes)


def blur_folder(src: Path, dst: Path, strength: float = STRENGTH,
                detector: Detector = detect_faces) -> dict:
    """Blur every image in `src` into `dst` under the same names and write dst/sources.json,
    {SHA-256 of each original: its blurred file}, which lets `regenerate` use blurred
    keyframes. Returns the faces found per file."""
    src, dst = Path(src), Path(dst)
    if dst.resolve() == src.resolve():
        raise ValueError("the destination must be another folder: the originals are kept")
    dst.mkdir(parents=True, exist_ok=True)
    faces, sources = {}, {}
    for path in sorted(src.iterdir()):
        if (not path.is_file() or path.name.startswith(".")
                or path.suffix.lower() not in IMAGE_SUFFIXES):
            continue
        data = path.read_bytes()
        out, faces[path.name] = blur_image(data, strength, detector)
        (dst / path.name).write_bytes(out)
        sources[sha256_hex(data)] = path.name
    (dst / SOURCES_FILE).write_text(json.dumps(sources, indent=1, sort_keys=True) + "\n")
    return {"images": len(faces), "faces": sum(faces.values()), "per_file": faces}


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args],
                   stdin=subprocess.DEVNULL, check=True)


def blur_video(src: Path, dst: Path, strength: float = STRENGTH, fps: float = 30.0,
               detector: Detector = detect_faces) -> dict:
    """Blur faces in a video frame by frame: ffmpeg decodes it at `fps`, each frame is
    blurred, ffmpeg encodes H.264. The container's metadata (location included) and the
    audio are left out: voices can identify people too, so add sound back after a listen."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed for video (brew install ffmpeg)")
    if Path(dst).resolve() == Path(src).resolve():
        raise ValueError("the destination must be another file: the original is kept")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        _ffmpeg("-i", str(src), "-vf", f"fps={fps}", str(Path(tmp) / "f%06d.png"))
        frames, faces, with_faces = sorted(Path(tmp).glob("f*.png")), 0, 0
        for path in frames:
            out, found = blur_image(path.read_bytes(), strength, detector)
            path.write_bytes(out)
            faces, with_faces = faces + found, with_faces + bool(found)
        _ffmpeg("-framerate", str(fps), "-i", str(Path(tmp) / "f%06d.png"), "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18", "-an", "-map_metadata", "-1", str(dst))
    return {"frames": len(frames), "faces": faces, "frames_with_faces": with_faces}
