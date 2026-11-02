"""Human labels for a recorded clip: the time ranges where the skill's condition is true.

Press [ where it becomes true and ] where it stops, fix the list, Save. `te calibrate
--labels` reads the file: {"clip": "a.mp4", "duration_s": 312.4, "true": [[41.2, 97.0]]}.
Ranges are sorted and merged where they overlap or touch, each 0 <= start < end <=
duration; an existing file is loaded to resume. Test labels come from a person only.
Starlette's FileResponse (0.39 on) answers Range requests with 206, which Safari needs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from backend.labeltool.images import STATIC, write_json


def _number(x) -> bool:
    try:
        return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)
    except OverflowError:  # an integer too large for a float
        return False


def clean_ranges(ranges, duration_s) -> list[list[float]]:
    """The ranges sorted, with overlapping or touching ones merged. Raises ValueError
    unless the duration is a positive number and every range is [start_s, end_s] with
    0 <= start < end <= duration_s (TypeError if `ranges` is not a list)."""
    if not _number(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be a positive number of seconds")
    if not isinstance(ranges, list):
        raise TypeError("true must be a list of [start_s, end_s] ranges")
    checked = []
    for r in ranges:
        if not (isinstance(r, list) and len(r) == 2 and all(_number(x) for x in r)):
            raise ValueError(f"each range must be [start_s, end_s], got {r!r}")
        if not 0 <= r[0] < r[1] <= duration_s:
            raise ValueError(f"range [{r[0]:g}, {r[1]:g}] must have 0 <= start < end <= "
                             f"{duration_s:g} (the clip's length)")
        checked.append([float(r[0]), float(r[1])])
    merged: list[list[float]] = []
    for start, end in sorted(checked):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def create_clip_app(clip: Path, out: Path) -> Starlette:
    """The labelling page for `clip`, saving to `out`. Raises FileNotFoundError if the
    clip is missing and ValueError if `out` holds labels for a different clip."""
    clip, out = Path(clip), Path(out)
    if not clip.is_file():
        raise FileNotFoundError(f"no clip at {clip}")
    saved = {"clip": clip.name, "duration_s": None, "true": []}
    if out.is_file():
        saved = {**saved, **json.loads(out.read_text(encoding="utf-8"))}
        if saved["clip"] != clip.name:
            raise ValueError(f"{out} holds labels for {saved['clip']}, not {clip.name}")

    async def labels(request: Request):
        nonlocal saved
        if request.method == "POST":
            try:
                body = await request.json()
                if not isinstance(body, dict):
                    raise TypeError("send a JSON object with duration_s and true")
                ranges = clean_ranges(body.get("true"), body.get("duration_s"))
            except (ValueError, TypeError) as err:  # JSONDecodeError is a ValueError
                return JSONResponse({"error": str(err)}, status_code=400)
            saved = {"clip": clip.name, "duration_s": float(body["duration_s"]), "true": ranges}
            write_json(out, saved, indent=None)  # one line, as in the docstring
        return JSONResponse({**saved, "out": out.name})

    return Starlette(routes=[
        Route("/", lambda request: FileResponse(STATIC / "clips.html")),
        Route("/video", lambda request: FileResponse(clip)),  # Range: 206, or 416 past the end
        Route("/labels", labels, methods=["GET", "POST"]),
        Mount("/static", StaticFiles(directory=STATIC))])
