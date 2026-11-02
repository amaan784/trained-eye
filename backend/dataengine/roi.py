"""Region of interest: where the skill's target is in the teaching keyframes (SAM 3.1).

Runs SAM 3.1 on each keyframe with the spec's `target.sam_prompt` (one concept per
request) and keeps the largest confident match per frame. The boxes drive the "outline"
visual.

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
    confident = [d for d in detections if d.score is None or d.score >= min_score]
    return max(confident, key=lambda d: d.area) if confident else None


def locate(client: MetaClient, spec: SkillSpec, keyframes: list[bytes],
           progress: Callable[[bool, float], None] | None = None) -> list[FrameRoi]:
    """SAM on each keyframe. `progress(cached, cost_usd)` is called after each one.

    A frame SAM rejects or answers unreadably gets no box (as in synth and labels, a few
    such failures are skipped; more stop the run with TooManyFailures)."""
    results, failed = [], 0
    for image in keyframes:
        with Image.open(io.BytesIO(image)) as img:
            width, height = img.size
        try:
            sam = client.sam_image(spec.target.sam_prompt, image, skill=spec.name)
        except MetaAPIError as err:
            if not skippable(err):
                raise
            # rejected, or billed but unreadable: no box for this frame
            results.append(FrameRoi(sha256_hex(image), width, height, "", [], None,
                                    f"error: {str(err)[:200]}"))
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
        results.append(FrameRoi(sha256_hex(image), width, height, sam.text, detections,
                                best_detection(detections), parser, problems[:20]))
    return results


def save(rois: list[FrameRoi], path: Path) -> None:
    path.write_text(json.dumps([asdict(r) for r in rois], indent=1) + "\n", encoding="utf-8")
