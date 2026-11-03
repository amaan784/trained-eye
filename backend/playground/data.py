"""Local results for the playground, read when a page asks: recorded clips with their
frame logs, the evaluation's charts, and the files those pages link to. Nothing here
calls Meta or writes a file; videos and images are served from where they already are.

A clip is a folder data/playground/clips/<name>/ (under data/, so real footage is never
committed; `te playground-clip` makes it) with log.jsonl (the frame log `te watch`
writes: t, skill, threshold, p, alert), clip.mp4 (or .webm, .mov) and optionally
labels.json ({"true": [[start_s, end_s], ...]}, as for `te calibrate`). A video blurred
for publishing has a blur record next to it, clip.mp4.blur.json, whose output_sha256 is
the video's.

The public copy (public=True) serves only what may be published: no generated images,
no original keyframes, keyframes only from a `te blur` folder listed in its sources.json,
and a clip's video only when its blur record matches it.

URLs are relative ("media/..."), so the site also works under a path prefix.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
from pathlib import Path

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")
FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}")
VIDEOS = {"clip.mp4": "video/mp4", "clip.webm": "video/webm", "clip.mov": "video/quicktime"}
IMAGES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
          ".webp": "image/webp"}
SVG = "image/svg+xml"
CLIPS = Path("playground") / "clips"  # under the data folder
SOURCES = "sources.json"  # te blur's {SHA-256 of an original: its blurred file}
BLUR_RECORD = ".blur.json"  # <video>.blur.json: the video's blur record
BLURRED_KEYFRAMES = "keyframes_blurred"  # per skill: te blur --src keyframes --dst this


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_jsonl(path: Path) -> list[dict]:
    """The lines of a JSONL file that are JSON objects; a torn or bad line is skipped."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def child(root: Path, *parts: str) -> Path | None:
    """The existing file root/parts, if every part is a plain name and it is inside root."""
    if not parts or not all(FILE.fullmatch(p) and ".." not in p for p in parts):
        return None
    try:
        base = Path(root).resolve()
        path = base.joinpath(*parts).resolve()
    except OSError:
        return None
    return path if base in path.parents and path.is_file() else None


def clips_root(data_dir: Path) -> Path:
    return Path(data_dir) / CLIPS


@functools.lru_cache(maxsize=64)
def _sha256(path: str, mtime_ns: int, size: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str | None:
    """SHA-256 of a file, kept while its size and time stay the same (a video is asked for
    in many ranges)."""
    try:
        st = Path(path).stat()
        return _sha256(str(Path(path).resolve()), st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def blurred_files(folder: Path) -> dict[str, str]:
    """{file: SHA-256 of its original} for the images a `te blur` folder lists in its
    sources.json and still holds. Anything else in the folder is not counted as blurred."""
    sources = read_json(Path(folder) / SOURCES)
    out = {}
    for sha, name in sources.items() if isinstance(sources, dict) else []:
        if (isinstance(name, str) and FILE.fullmatch(name)
                and Path(name).suffix.lower() in IMAGES and (Path(folder) / name).is_file()):
            out[name] = sha
    return dict(sorted(out.items()))


def blurred_video(folder: Path) -> str | None:
    """The clip's video if its blur record (<video>.blur.json) names this very file."""
    for name in VIDEOS:
        record = read_json(Path(folder) / f"{name}{BLUR_RECORD}")
        sha = record.get("output_sha256") if isinstance(record, dict) else None
        if isinstance(sha, str) and (Path(folder) / name).is_file() and (
                file_sha256(Path(folder) / name) == sha):
            return name
    return None


def clip_folders(data_dir: Path) -> list[Path]:
    root = clips_root(data_dir)
    if not root.is_dir():
        return []
    return [d for d in sorted(root.iterdir())
            if NAME.fullmatch(d.name) and (d / "log.jsonl").is_file()]


def frame_rows(path: Path) -> list[dict]:
    """A frame log's scored frames, by time. p is None where the student failed; alert is
    the spoken text if the row has it, else whether an alert went out."""
    out = []
    for row in read_jsonl(path):
        t, skill, alert = number(row.get("t")), row.get("skill"), row.get("alert")
        if t is None or not isinstance(skill, str):
            continue
        alert = alert.strip() if isinstance(alert, str) else alert is True
        out.append({"t": t, "skill": skill, "p": number(row.get("p")),
                    "threshold": number(row.get("threshold")), "alert": alert})
    return sorted(out, key=lambda r: r["t"])


def video_file(folder: Path, public: bool = False) -> str | None:
    """The clip's video; in the public copy only a blurred one (blurred_video)."""
    if public:
        return blurred_video(folder)
    return next((name for name in VIDEOS if (folder / name).is_file()), None)


def list_clips(data_dir: Path, *, public: bool = False) -> list[dict]:
    out = []
    for folder in clip_folders(data_dir):
        rows = frame_rows(folder / "log.jsonl")
        out.append({"name": folder.name, "video": video_file(folder, public) is not None,
                    "skills": sorted({r["skill"] for r in rows}), "frames": len(rows),
                    "alerts": sum(1 for r in rows if r["alert"]),
                    "labelled": (folder / "labels.json").is_file()})
    return out


def spoken(data_dir: Path, skill: str) -> str:
    """What the skill says when it fires, from its spec."""
    path = Path(data_dir) / "skills" / skill / "spec.json"
    spec = read_json(path) if NAME.fullmatch(skill) else None
    text = spec.get("alert") if isinstance(spec, dict) else None
    return text.strip() if isinstance(text, str) and text.strip() else "Alert"


def true_ranges(raw) -> list[list[float]]:
    items = raw.get("true") if isinstance(raw, dict) else None
    out = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, list) and len(item) == 2:
            a, b = number(item[0]), number(item[1])
            if a is not None and b is not None and a <= b:
                out.append([a, b])
    return out


def clip_timeline(data_dir: Path, name: str, *, public: bool = False) -> dict | None:
    """One clip for the live timeline: per skill its threshold, [t, p] points and alerts."""
    folder = clips_root(data_dir) / name
    if not NAME.fullmatch(name) or not (folder / "log.jsonl").is_file():
        return None
    rows = frame_rows(folder / "log.jsonl")
    skills: dict[str, dict] = {}
    for r in rows:
        s = skills.setdefault(r["skill"], {"threshold": None, "points": [], "alerts": []})
        s["points"].append([r["t"], r["p"]])
        if r["threshold"] is not None:
            s["threshold"] = r["threshold"]
        if r["alert"]:
            text = r["alert"] if isinstance(r["alert"], str) else spoken(data_dir, r["skill"])
            s["alerts"].append({"t": r["t"], "p": r["p"], "text": text})
    video = video_file(folder, public)
    return {"name": name, "video": f"media/clips/{name}/{video}" if video else None,
            "duration": max((r["t"] for r in rows), default=0.0), "skills": skills,
            "true_ranges": true_ranges(read_json(folder / "labels.json"))}


def results(eval_dir: Path) -> dict:
    """The SVG charts the evaluation wrote into eval_dir, and results.md as text."""
    root = Path(eval_dir)
    charts = [{"name": p.stem.replace("_", " ").replace("-", " "),
               "url": f"media/results/{p.name}"}
              for p in sorted(root.glob("*.svg")) if FILE.fullmatch(p.name) and p.is_file()]
    md = root / "results.md"
    text = md.read_text(encoding="utf-8") if md.is_file() else None
    return {"charts": charts, "results_md": text}


def media_path(data_dir: Path, eval_dir: Path, path: str, *,
               public: bool = False) -> tuple[Path, str] | None:
    """The file behind a media/ URL and its content type. Only what the pages link to
    can be reached: clip videos, keyframes, generated variations, the provided keyframes
    and the evaluation's SVG charts. The public copy serves no generated image and no
    original keyframe: keyframes come from te blur folders, videos only when blurred."""
    data_dir = Path(data_dir)
    kind = IMAGES.get(Path(path).suffix.lower())
    match path.split("/"):
        case ["clips", name, file] if file in VIDEOS:
            found, kind = child(clips_root(data_dir), name, file), VIDEOS[file]
            if public and found and blurred_video(found.parent) != file:
                found = None
        case ["skills", skill, ("keyframes" | "synthetic") as folder, file] if not public:
            found = child(data_dir / "skills", skill, folder, file)
        case ["skills", skill, "keyframes_blurred", file] if public:
            found = child(data_dir / "skills", skill, BLURRED_KEYFRAMES, file)
            found = found if found and file in blurred_files(found.parent) else None
        case ["playground", "keyframes", file]:
            found = child(data_dir / "playground" / "keyframes", file)
            if public and found and file not in blurred_files(found.parent):
                found = None
        case ["results", file] if file.endswith(".svg"):
            found, kind = child(eval_dir, file), SVG
        case _:
            return None
    return (found, kind) if found and kind else None
