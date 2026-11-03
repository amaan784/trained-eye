"""Blur faces in anything published: README images, the playground, the video, the test set.

`blur_boxes` blurs given boxes with Pillow alone. `detect_faces` finds faces with the Haar
cascades OpenCV ships (frontal, and profile facing either way); OpenCV is imported only
then: `pip install opencv-python-headless` (on the approved list). Haar cascades miss
small, dark, tilted or covered faces, so look at every published image as well: this is
a first pass, not a guarantee.

Faces the detector misses are blurred through manual boxes (`te blur --boxes`): per image
file a list of boxes, or for a video a list of {"start_s", "end_s", "box"}, all added to
the detector's. A video's blur record, <output>.blur.json, holds the SHA-256 of the source
and of the output, the fps, the faces found per frame and the manual boxes used, so a
later step (the playground, the test set) can check that a video is this blurred output.

Output is rebuilt from the pixels alone: EXIF (GPS, camera, time), XMP, ICC and text
chunks are left out, and an EXIF rotation is applied to the pixels first. Use it on real
footage only: Muse Image outputs are never published, and their provenance metadata must
never be stripped.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from backend.meta.cache import sha256_hex

Box = tuple[int, int, int, int]  # left, top, right, bottom in pixels (right, bottom excluded)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
SOURCES_FILE = "sources.json"  # from blur_folder: SHA-256 of each original -> blurred file
BLUR_RECORD = ".blur.json"  # blur_video writes <output>.blur.json, the video's blur record
STRENGTH = 0.25  # Gaussian radius as a share of the box's shorter side
BLOCK_PX = 8  # a box is first reduced to this many pixels across its shorter side
MARGIN = 0.2  # a detected face box grows by this share on every side (hair, ears, chin)
CASCADES = ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml")
INSTALL = "pip install opencv-python-headless"
Detector = Callable[[bytes], list[Box]]
SPAN_KEYS = ("start_s", "end_s", "box")  # one manual box over a stretch of a video


def _number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def check_box(box) -> Box:
    """A manual box: [left, top, right, bottom] in pixels, right above left and bottom
    below top. Parts outside the image are clipped when it is blurred."""
    if not (isinstance(box, list | tuple) and len(box) == 4 and all(map(_number, box))):
        raise ValueError(f"a box is [left, top, right, bottom] in pixels, not {box!r}")
    left, top, right, bottom = (round(float(v)) for v in box)
    if right <= left or bottom <= top:
        raise ValueError(f"box {list(box)}: right must be above left and bottom below top")
    return left, top, right, bottom


def image_boxes(raw) -> dict[str, list[Box]]:
    """--boxes for an image folder: {"<file name>": [[left, top, right, bottom], ...]}
    (TypeError for another shape, ValueError for a bad box)."""
    if not isinstance(raw, dict):
        raise TypeError('boxes for a folder are {"<file name>": [[left, top, right, '
                        'bottom], ...]}')
    out = {}
    for name, boxes in raw.items():
        if not isinstance(boxes, list):
            raise TypeError(f"{name}: give a list of boxes")
        out[name] = [check_box(b) for b in boxes]
    return out


def video_boxes(raw) -> list[dict]:
    """--boxes for a video: [{"start_s": 1.5, "end_s": 4.0, "box": [l, t, r, b]}, ...]; the
    box is blurred on every frame from start_s to end_s, both included."""
    if not isinstance(raw, list):
        raise TypeError('boxes for a video are [{"start_s": .., "end_s": .., "box": '
                        '[left, top, right, bottom]}, ...]')
    out = []
    for span in raw:
        if not isinstance(span, dict) or sorted(span) != sorted(SPAN_KEYS):
            raise ValueError(f"each video box has exactly {', '.join(SPAN_KEYS)}: {span!r}")
        start, end = span["start_s"], span["end_s"]
        if not (_number(start) and _number(end) and 0 <= start <= end):
            raise ValueError(f"start_s and end_s are seconds, 0 <= start_s <= end_s: {span!r}")
        out.append({"start_s": float(start), "end_s": float(end),
                    "box": list(check_box(span["box"]))})
    return out


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
               detector: Detector = detect_faces, extra: Iterable[Box] = ()) -> tuple[bytes, int]:
    """(the image with every detected face and every `extra` box blurred, faces detected).
    Always re-encoded, so the metadata is left out even when no face is found."""
    boxes = detector(image_bytes)
    return blur_boxes(image_bytes, [*boxes, *extra], strength), len(boxes)


def _images(src: Path) -> list[Path]:
    return [p for p in sorted(Path(src).iterdir()) if p.is_file() and not p.name.startswith(".")
            and p.suffix.lower() in IMAGE_SUFFIXES]


def outside(boxes: Iterable[Box], size: tuple[int, int]) -> list[Box]:
    """The manual boxes that miss the picture entirely (blur_boxes would skip them)."""
    return [box for box in boxes if _clip(box, size) is None]


def blur_folder(src: Path, dst: Path, strength: float = STRENGTH,
                detector: Detector = detect_faces,
                extra: Mapping[str, Iterable[Box]] | None = None) -> dict:
    """Blur every image in `src` into `dst` under the same names and write dst/sources.json,
    {SHA-256 of each original: its blurred file}, which lets `regenerate` use blurred
    keyframes. `extra` ({file name: boxes}, see image_boxes) adds manual boxes to the
    detector's; a name that is not an image in `src` is refused, so a typo cannot leave a
    face unblurred. Returns the faces found per file and the manual boxes used."""
    src, dst = Path(src), Path(dst)
    if dst.resolve() == src.resolve():
        raise ValueError("the destination must be another folder: the originals are kept")
    images, extra = _images(src), dict(extra or {})
    if unknown := sorted(set(extra) - {p.name for p in images}):
        raise ValueError(f"the boxes name files that are not images in {src}: "
                         f"{', '.join(unknown[:5])}")
    for path in images:  # a box off the picture would be counted but blur nothing
        if boxes := list(extra.get(path.name, ())):
            size = _open(path.read_bytes())[0].size
            if missed := outside(boxes, size):
                raise ValueError(f"{path.name}: box {list(missed[0])} is outside the "
                                 f"{size[0]}x{size[1]} picture (measured on a rotated view?); "
                                 "nothing written")
    dst.mkdir(parents=True, exist_ok=True)
    faces, sources, manual = {}, {}, 0
    for path in images:
        data, boxes = path.read_bytes(), list(extra.get(path.name, ()))
        out, faces[path.name] = blur_image(data, strength, detector, boxes)
        (dst / path.name).write_bytes(out)
        sources[sha256_hex(data)] = path.name
        manual += len(boxes)
    (dst / SOURCES_FILE).write_text(json.dumps(sources, indent=1, sort_keys=True) + "\n")
    return {"images": len(faces), "faces": sum(faces.values()), "per_file": faces,
            "manual_boxes": manual}


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args],
                   stdin=subprocess.DEVNULL, check=True)


def blur_video(src: Path, dst: Path, strength: float = STRENGTH, fps: float = 30.0,
               detector: Detector = detect_faces, extra: Iterable[dict] | None = None) -> dict:
    """Blur faces in a video frame by frame: ffmpeg decodes it at `fps`, each frame is
    blurred, ffmpeg encodes H.264. The container's metadata (location included) and the
    audio are left out: voices can identify people too, so add sound back after a listen.

    `extra` (see video_boxes) adds manual boxes over stretches of time; frame i is at
    i / fps seconds. A box that falls on no frame is refused before anything is written.
    Writes the blur record <dst>.blur.json and returns it."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed for video (brew install ffmpeg)")
    src, dst = Path(src), Path(dst)
    if dst.resolve() == src.resolve():
        raise ValueError("the destination must be another file: the original is kept")
    spans = video_boxes(list(extra or []))
    record_path = Path(f"{dst}{BLUR_RECORD}")
    with tempfile.TemporaryDirectory() as tmp:
        _ffmpeg("-i", str(src), "-vf", f"fps={fps}", str(Path(tmp) / "f%06d.png"))
        frames = sorted(Path(tmp).glob("f*.png"))
        per_frame = [[n for n, s in enumerate(spans) if s["start_s"] <= i / fps <= s["end_s"]]
                     for i in range(len(frames))]
        used = {n for frame in per_frame for n in frame}
        if unused := [s for n, s in enumerate(spans) if n not in used]:
            raise ValueError(f"{len(unused)} manual boxes fall on no frame (the first from "
                             f"{unused[0]['start_s']:g} s to {unused[0]['end_s']:g} s; the "
                             f"video has {len(frames)} frames at {fps:g} fps): nothing written")
        if spans and frames:
            size = _open(frames[0].read_bytes())[0].size
            if missed := outside([s["box"] for s in spans], size):
                raise ValueError(f"box {list(missed[0])} is outside the {size[0]}x{size[1]} "
                                 "frame (measured on a rotated view?): nothing written")
        dst.parent.mkdir(parents=True, exist_ok=True)
        record_path.unlink(missing_ok=True)  # an older record never vouches for this output
        found = []
        for path, spans_here in zip(frames, per_frame, strict=True):
            boxes = [spans[n]["box"] for n in spans_here]
            out, n = blur_image(path.read_bytes(), strength, detector, boxes)
            path.write_bytes(out)
            found.append(n)
        _ffmpeg("-framerate", str(fps), "-i", str(Path(tmp) / "f%06d.png"), "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18", "-an", "-map_metadata", "-1", str(dst))
    record = {"source": src.name, "source_sha256": file_sha256(src), "output": dst.name,
              "output_sha256": file_sha256(dst), "fps": fps, "strength": strength,
              "frames": len(frames), "faces": sum(found),
              "frames_with_faces": sum(map(bool, found)), "faces_per_frame": found,
              "manual_boxes": spans, "frames_with_manual_boxes": sum(map(bool, per_frame)),
              "audio": "removed", "metadata": "removed"}
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    return record
