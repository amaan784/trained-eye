"""Region of interest: where the skill's target is in the teaching keyframes (SAM 3.1).

Runs SAM 3.1 on each keyframe with the spec's `target.sam_prompt` (one concept per
request) and keeps the largest confident match per frame. The boxes later drive crops
and the "outline" visual.

SAM's output is streamed special-token text; SPEC.md says to parse it with the
`meta-sam-parser` package. That package and the docs describing the format could not
be reached while this was written, so parsing goes through `parse_sam_output`, which
uses meta_sam_parser when it is installed and otherwise records the raw text with no
boxes. The raw text is always stored (and the call cached), so boxes can be parsed
later without paying again.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from backend.dataengine.synth import TooManyFailures, skippable, too_many_failures
from backend.meta.cache import sha256_hex
from backend.meta.client import MetaAPIError, MetaClient
from backend.skills.spec import SkillSpec

MIN_SCORE = 0.5


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in source pixels
    score: float | None = None

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


def parse_sam_output(text: str) -> tuple[list[Detection], str]:
    """Return (detections, parser name). Empty if no parser is available."""
    try:
        import meta_sam_parser  # noqa: F401
    except ImportError:
        return [], "none"
    # Installed, but its API has not been checked against a real SAM response yet.
    return [], "unwired"


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
        detections, parser = parse_sam_output(sam.text)
        results.append(FrameRoi(sha256_hex(image), width, height, sam.text, detections,
                                best_detection(detections), parser))
    return results


def save(rois: list[FrameRoi], path: Path) -> None:
    path.write_text(json.dumps([asdict(r) for r in rois], indent=1) + "\n", encoding="utf-8")
