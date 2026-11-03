"""Experiment 1's failure galleries, written by te eval next to results.md.

For each scored Experiment 1 evaluation, from its frame logs and labels: the worst false
alarms (frames labelled false that scored over the threshold, highest p first) and the
worst misses (frames inside a true range that stayed under it, lowest p first), up to
`size` of each, at least MIN_GAP_S apart so one long mistake does not fill the row. Each
frame is read again from its clip at the logged t (keyframes.load_frames, at the replay's
frame rate, taken from the log's frame step) and saved as a small JPEG thumbnail, and
each run gets one HTML page with time, p and threshold under every frame:

    <out>/galleries/<run>/index.html       one section per skill (evaluation)
    <out>/galleries/<run>/<evaluation>-<log>-fa-01.jpg, ...-miss-01.jpg, ...

<run> is the evaluation's run (te replay --run), else its name; galleries/ is written
afresh each time. A clip comes from the plan's "clip", else the file or folder next to its
labels with the same name; without one (or without ffmpeg for a video) the page lists the
frames without pictures. Each clip is read once per te eval (however many evaluations
replayed it), keeping only the frames shown.

The thumbnails are frames of the recorded clips, which may show faces: the galleries stay
on this Mac (galleries/ holds a .gitignore of its own, wherever --out points), are never
logged to W&B, and the page says to run te blur on any frame before it is published.
"""

from __future__ import annotations

import html
import io
import itertools
import os
import re
import shutil
import statistics
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from backend.eval import metrics
from backend.eval.timeline import clock

MIN_GAP_S = 5.0
UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")
THUMB_PX = 240
FOLDER = "galleries"
NOTE = ("Local only. These are frames of your recorded clips and may show faces or other "
        "people's things: run te blur on any frame before publishing it, and publish only "
        "footage you approve.")
HOW = ("False alarms: frames labelled false that scored at or over the threshold, the "
       "highest p first. Misses: frames inside a labelled true range that stayed under it, "
       f"the lowest p first. Each at least {MIN_GAP_S:g} s from the others shown.")
STYLE = """
:root{--bg:#fcfcfb;--t1:#0b0b0b;--t2:#52514e;--t3:#898781;--line:#e1e0d9;--acc:#2a78d6}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--t1:#f0efec;--t2:#c3c2b7;
--t3:#898781;--line:#2c2c2a;--acc:#3987e5}}
body{margin:0;padding:24px;background:var(--bg);color:var(--t1);
font:14px/1.45 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
h1{font-size:20px;margin:0 0 8px}h2{font-size:16px;margin:28px 0 4px}
h3{font-size:14px;margin:16px 0 8px;color:var(--t2)}
.note{border-left:3px solid var(--acc);padding:8px 12px;color:var(--t2);max-width:720px}
.grid{display:flex;flex-wrap:wrap;gap:12px}
figure{margin:0;width:240px}figure img{width:240px;border-radius:6px;display:block}
.none{width:240px;height:135px;border:1px dashed var(--line);border-radius:6px;
display:flex;align-items:center;justify-content:center;color:var(--t3)}
figcaption{color:var(--t2);font-size:12px;margin-top:4px}p{color:var(--t2)}
"""


def _over(row: dict) -> bool:
    return bool(row["over"]) if "over" in row else float(row["p"]) >= float(row["threshold"])


def _spread(rows: Iterable[dict], size: int, gap_s: float) -> list[dict]:
    picked: list[dict] = []
    for row in rows:
        if len(picked) == size:
            break
        if all(abs(float(row["t"]) - float(p["t"])) >= gap_s for p in picked):
            picked.append(row)
    return picked


def pick(rows: Iterable[dict], skill: str, true_ranges: Sequence[Sequence[float]],
         size: int, gap_s: float = MIN_GAP_S) -> tuple[list[dict], list[dict]]:
    """(false alarms, misses) of one skill in one clip's log rows, worst first."""
    ranges = metrics.check_ranges(true_ranges)
    scored = [r for r in rows if r.get("skill") == skill and r.get("p") is not None]

    def truth(r: dict) -> bool:
        return any(a <= float(r["t"]) <= b for a, b in ranges)

    alarms = sorted((r for r in scored if _over(r) and not truth(r)),
                    key=lambda r: (-float(r["p"]), float(r["t"])))
    misses = sorted((r for r in scored if truth(r) and not _over(r)),
                    key=lambda r: (float(r["p"]), float(r["t"])))
    return _spread(alarms, size, gap_s), _spread(misses, size, gap_s)


def frame_step(rows: Iterable[dict]) -> float:
    """Seconds between replayed frames: the median gap of the log's times (1 at 1 fps)."""
    ts = sorted({float(r["t"]) for r in rows})
    gaps = [b - a for a, b in itertools.pairwise(ts)]
    return statistics.median(gaps) if gaps else 1.0


def find_clip(clip: str | None, labels: Path, base: Path) -> Path | None:
    """The recorded clip: the plan's, else the file or folder named like its labels."""
    if clip:
        path = base / clip
        return path if path.exists() else None
    labels = base / labels
    stem = labels.name.removesuffix(".labels.json").removesuffix(".json")
    if not labels.parent.is_dir():
        return None
    found = sorted(p for p in labels.parent.iterdir() if p != labels and (
        p.stem == stem or p.name == stem) and p.suffix not in (".json", ".jsonl"))
    return found[0] if found else None


def thumbnail(data: bytes, px: int = THUMB_PX) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((px, px))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=80)
    return out.getvalue()


def _frames(clip: Path | None, step: float,
            wanted: set[int]) -> tuple[dict[int, bytes], str | None]:
    """{frame index: JPEG bytes} of the `wanted` frames of the clip at 1/step fps (the rest
    are dropped as soon as the clip is read), or why there are none."""
    from backend.dataengine.keyframes import load_frames

    if clip is None:
        return {}, "the clip is not on this Mac, so its frames are not shown"
    try:
        return {f.index: f.data for f in load_frames(clip, fps=1 / step)
                if f.index in wanted}, None
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as err:
        # RuntimeError: no ffmpeg; CalledProcessError: ffmpeg could not read the video
        return {}, f"the frames could not be read from {clip.name}: {type(err).__name__}"


def _card(row: dict, src: str | None) -> str:
    threshold = row.get("threshold")
    caption = (f"{clock(float(row['t']))} ({float(row['t']):.1f} s), p {float(row['p']):.2f}, "
               f"threshold {'n/a' if threshold is None else f'{float(threshold):.2f}'}")
    pic = (f'<img src="{html.escape(src)}" alt="frame at {float(row["t"]):.1f} s">' if src
           else '<div class="none">no picture</div>')
    return f"<figure>{pic}<figcaption>{html.escape(caption)}</figcaption></figure>"


@dataclass
class _Picked:
    """One clip of one evaluation: its worst frames and where its pictures come from."""

    evaluation: dict
    log: str
    alarms: list[dict]
    misses: list[dict]
    step: float
    key: tuple  # (the clip, resolved, or None; the frame step): its frames, read once

    def indexes(self) -> set[int]:
        return {round(float(r["t"]) / self.step) for r in self.alarms + self.misses}


def _pick_clip(e: dict, c: dict, base: Path, size: int) -> tuple[_Picked, Path | None]:
    rows = [r for r in metrics.read_log(base / c["log"]) if r.get("skill") == e["skill"]]
    alarms, misses = pick(rows, e["skill"], metrics.read_labels(base / c["labels"]), size)
    step = frame_step(rows)
    clip = find_clip(c.get("clip"), Path(c["labels"]), base) if alarms or misses else None
    return _Picked(e, c["log"], alarms, misses, step,
                   (clip.resolve() if clip else None, step)), clip


def _section(e: dict, picked: list[_Picked], frames: dict, folder: Path) -> list[str]:
    """One evaluation's part of the page, writing its thumbnails into `folder`."""
    parts = [f"<h2>{html.escape(e['skill'])}</h2>",
             f"<p>Evaluation {html.escape(e['name'])} ({html.escape(e['variant'])}).</p>"]
    for p in picked:
        found, why = frames.get(p.key, ({}, None))
        parts.append(f"<h3>{html.escape(Path(p.log).name)}</h3>")
        if why:
            parts.append(f"<p>{html.escape(why)}.</p>")
        for kind, title, rows in (("fa", "False alarms", p.alarms), ("miss", "Misses", p.misses)):
            cards = []
            for n, row in enumerate(rows, 1):
                data, src = found.get(round(float(row["t"]) / p.step)), None
                if data is not None:
                    src = UNSAFE.sub("_", f"{e['name']}-{Path(p.log).stem}-{kind}-{n:02d}.jpg")
                    (folder / src).write_bytes(thumbnail(data))
                cards.append(_card(row, src))
            parts.append(f"<h3>{title}: {len(rows) or 'none'}</h3>")
            if cards:
                parts.append(f'<div class="grid">{"".join(cards)}</div>')
    return parts


def page(run: str, sections: list[str]) -> str:
    return "".join([
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Failure gallery: {html.escape(run)}</title><style>{STYLE}</style></head>",
        f"<body><h1>Failure gallery: {html.escape(run)}</h1>",
        f'<p class="note">{html.escape(NOTE)}</p>',
        f"<p>{html.escape(HOW)}</p>", *sections,
        "</body></html>\n"])


def write_galleries(results: dict, base: Path, out_dir: Path, size: int = 6) -> dict:
    """Write the galleries of the scored Experiment 1 evaluations (out_dir/galleries/ is
    replaced); returns {run: {"page" (relative to out_dir), "false_alarms", "misses"}}.
    Each clip is read once, keeping only the frames some gallery shows."""
    by_run: dict[str, list[dict]] = {}
    for e in results["evaluations"]:
        if e["experiment"] == 1 and e["status"] == "scored":
            by_run.setdefault(e.get("run") or e["name"], []).append(e)
    picked: dict[str, list[_Picked]] = {}
    wanted: dict[tuple, tuple[Path | None, set[int]]] = {}
    for evals in by_run.values():
        for e in evals:
            for c in e["clips"]:
                p, clip = _pick_clip(e, c, base, size)
                picked.setdefault(e["name"], []).append(p)
                if p.alarms or p.misses:
                    wanted.setdefault(p.key, (clip, set()))[1].update(p.indexes())
    frames = {key: _frames(clip, key[1], indexes) for key, (clip, indexes) in wanted.items()}
    out = {}
    shutil.rmtree(Path(out_dir) / FOLDER, ignore_errors=True)  # only this function writes here
    if by_run:  # frames of real footage: git ignores them wherever --out points
        (Path(out_dir) / FOLDER).mkdir(parents=True)
        (Path(out_dir) / FOLDER / ".gitignore").write_text("# may show faces: never commit\n*\n")
    for run, evals in by_run.items():
        folder = Path(out_dir) / FOLDER / run
        folder.mkdir(parents=True)
        sections = [part for e in evals for part in _section(e, picked[e["name"]], frames,
                                                              folder)]
        mine = [p for e in evals for p in picked[e["name"]]]
        (folder / "index.html").write_text(page(run, sections), encoding="utf-8")
        out[run] = {"page": os.path.relpath(folder / "index.html", out_dir).replace(os.sep, "/"),
                    "false_alarms": sum(len(p.alarms) for p in mine),
                    "misses": sum(len(p.misses) for p in mine)}
    return out
