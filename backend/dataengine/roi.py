"""Region of interest: where the skill's target is in the teaching keyframes (SAM 3.1).

Runs SAM 3.1 on each keyframe with the spec's `target.sam_prompt` (one concept per
request) and keeps, per frame, the box around all its confident matches. The boxes drive
the "outline" visual and the crops: te teach adds each keyframe cropped to that box
(`crop_box`, with a margin) as one more real example, labelled like the keyframe.

SAM's output is special-token text in Meta's SAM 3 protocol. SPEC.md names Meta's
`meta-sam-parser` package; this project reads the boxes with its own implementation of
the protocol instead (backend/dataengine/sam_text.py, no dependency). The raw text is
always stored too (and the call cached), so a run can be parsed again without paying.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from backend.dataengine import sam_text
from backend.dataengine.synth import TooManyFailures, skippable, too_many_failures
from backend.meta.cache import sha256_hex
from backend.meta.client import MetaAPIError, MetaClient
from backend.skills.spec import SkillSpec

MIN_SCORE = 0.5
CROP_MARGIN = 0.25  # of the box's width and height, added on every side
CROP_MIN_SIDE = 64  # pixels; a crop is grown to this, aspect kept, unless the frame is smaller


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in the keyframe's pixels
    score: float | None = None  # SAM's confidence; None when it gave none (not zero)
    object_id: str | None = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class FrameRoi:
    keyframe_sha256: str
    width: int
    height: int
    raw: str
    detections: list[Detection] = field(default_factory=list)
    best: Detection | None = None
    parser: str = "none"
    problems: list[str] = field(default_factory=list)  # what the parser dropped or ignored


PARSER = "sam3-text"


def parse_sam_output(text: str, size: tuple[int, int] | None = None
                     ) -> tuple[list[Detection], str, list[str]]:
    """(detections, parser name, problems) from SAM's text for one image. With `size`
    (the keyframe's width and height), boxes SAM gave in another size are scaled to it."""
    parsed = sam_text.parse(text, "image")
    detections = []
    for record in parsed.records:
        x1, y1, x2, y2 = record.box
        if size and record.size != tuple(size):
            sx, sy = size[0] / record.size[0], size[1] / record.size[1]
            x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
        detections.append(Detection((x1, y1, x2, y2), record.confidence, record.object_id))
    return detections, PARSER, parsed.errors + parsed.warnings


def best_detection(detections: list[Detection], min_score: float = MIN_SCORE) -> Detection | None:
    """The region to watch: the box around every confident match. A target often comes
    in several instances (both doors of a French-door fridge, every burner of a hob) and
    the condition may be on any of them, so the crop has to hold them all. The largest
    match alone was the closed door in most of the first fridge run's keyframes, which
    made a crop of a shut door that was labelled open. The score is the best one among
    the matches, the object ids are joined with "+"."""
    confident = [d for d in detections if d.score is None or d.score >= min_score]
    if len(confident) < 2:
        return confident[0] if confident else None
    boxes = [d.box for d in confident]
    scores = [d.score for d in confident if d.score is not None]
    ids = [d.object_id for d in confident if d.object_id is not None]
    return Detection((min(b[0] for b in boxes), min(b[1] for b in boxes),
                      max(b[2] for b in boxes), max(b[3] for b in boxes)),
                     max(scores) if scores else None, "+".join(ids) or None)


def crop_box(box: tuple[float, float, float, float], size: tuple[int, int],
             margin: float = CROP_MARGIN, min_side: int = CROP_MIN_SIDE
             ) -> tuple[int, int, int, int]:
    """The crop around `box` (x1, y1, x2, y2) in a frame of `size` (width, height), as
    whole pixels (left, top, right, bottom; right and bottom exclusive).

    The box gets `margin` of its width and height on every side, is grown to at least
    `min_side` pixels a side, then shrunk to fit the frame, each step keeping its aspect
    ratio, and is moved inside the frame, centred on the box where it can be. Shrinking
    never cuts the box (it fits the frame), so the frame only wins over `min_side`."""
    width, height = size
    if width < 1 or height < 1:
        raise ValueError(f"frame size {width}x{height}")
    x1, x2 = sorted(min(max(float(v), 0.0), width) for v in (box[0], box[2]))
    y1, y2 = sorted(min(max(float(v), 0.0), height) for v in (box[1], box[3]))
    w, h = max(x2 - x1, 1.0) * (1 + 2 * margin), max(y2 - y1, 1.0) * (1 + 2 * margin)
    scale = max(1.0, min_side / min(w, h))
    scale = min(scale, width / w, height / h)
    cw = min(width, max(1, round(w * scale)))
    ch = min(height, max(1, round(h * scale)))
    left = min(max(round((x1 + x2 - cw) / 2), 0), width - cw)
    top = min(max(round((y1 + y2 - ch) / 2), 0), height - ch)
    return left, top, left + cw, top + ch


def crop_image(image: bytes, crop: tuple[int, int, int, int]) -> bytes:
    """The `crop` (from crop_box) of an encoded image, as PNG bytes."""
    with Image.open(io.BytesIO(image)) as img:
        part = img.convert("RGB").crop(crop)
    buf = io.BytesIO()
    part.save(buf, format="PNG")
    return buf.getvalue()


def locate(client: MetaClient, spec: SkillSpec, keyframes: list[bytes],
           progress: Callable[[bool, float], None] | None = None,
           workers: int = 1) -> list[FrameRoi]:
    """SAM on each keyframe, `workers` at a time, in keyframe order.
    `progress(cached, cost_usd)` is called after each one.

    A frame SAM rejects or answers unreadably gets no box (as in synth and labels, a few
    such failures are skipped; more stop the run with TooManyFailures)."""
    from backend.dataengine import parallel

    results: dict[int, FrameRoi] = {}
    failed = 0

    def ask(image: bytes):
        return client.sam_image(spec.target.sam_prompt, image, skill=spec.name)

    for i, sam, err in parallel.run_all(keyframes, ask, workers):
        image = keyframes[i]
        with Image.open(io.BytesIO(image)) as img:
            width, height = img.size
        if err is not None:
            if not isinstance(err, MetaAPIError) or not skippable(err):
                raise err
            # rejected, or billed but unreadable: no box for this frame
            results[i] = FrameRoi(sha256_hex(image), width, height, "", [], None,
                                  f"error: {str(err)[:200]}")
            failed += 1
            if too_many_failures(failed, len(keyframes)):
                raise TooManyFailures(f"{failed} of {len(keyframes)} SAM calls failed so "
                                      f"far; last: {str(err)[:300]}") from None
            if progress:
                progress(False, err.cost_usd)
            continue
        if progress:
            progress(sam.info.cached, sam.info.cost_usd)
        detections, parser, problems = parse_sam_output(sam.text, (width, height))
        results[i] = FrameRoi(sha256_hex(image), width, height, sam.text, detections,
                              best_detection(detections), parser, problems[:20])
    return [results[i] for i in sorted(results)]


def save(rois: list[FrameRoi], path: Path) -> None:
    path.write_text(json.dumps([asdict(r) for r in rois], indent=1) + "\n", encoding="utf-8")
