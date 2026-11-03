"""What the public copy of the playground may hold, for `te playground --deploy`: the files
create_app(public=True) reads, and nothing it would refuse to serve. Copied to a staging
folder as data/... and eval/..., which the deploy uploads to the private Modal Volume.

Per skill: spec.json, roi.json, label_stats.json, labels.jsonl and the generated images'
manifest (counts and labels only), and the te blur folder keyframes_blurred/ (sources.json
and the files it lists). The curated keyframes in data/playground/keyframes/ the same way.
Per clip: log.jsonl, labels.json, and the video only with a blur record that matches it.
The evaluation's SVG charts and results.md. Never a generated image, an original
keyframe, an unblurred video, the call cache, the ledger or the budget.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from backend.playground import data
from backend.skills.spec import NAME_PATTERN

SKILL_FILES = ("spec.json", "roi.json", "label_stats.json", "labels.jsonl",
               "synthetic/manifest.jsonl")
CLIP_FILES = ("log.jsonl", "labels.json")


def blurred_set(folder: Path, rel: str) -> list[tuple[Path, str]]:
    """A te blur folder: its sources.json and the blurred files it lists."""
    files = data.blurred_files(folder)
    if not files:
        return []
    return [(folder / data.SOURCES, f"{rel}/{data.SOURCES}")] + [
        (folder / name, f"{rel}/{name}") for name in files]


def public_files(data_dir: Path, eval_dir: Path) -> dict:
    """{"files": [(source, path in the staging folder)], "notes": [what was left out]}."""
    data_dir, eval_dir = Path(data_dir), Path(eval_dir)
    files, notes = [], []
    skills = data_dir / "skills"
    for folder in sorted(skills.iterdir()) if skills.is_dir() else []:
        if not (re.fullmatch(NAME_PATTERN, folder.name) and (folder / "spec.json").is_file()):
            continue
        rel = f"data/skills/{folder.name}"
        files += [(folder / f, f"{rel}/{f}") for f in SKILL_FILES if (folder / f).is_file()]
        blurred = blurred_set(folder / data.BLURRED_KEYFRAMES, f"{rel}/{data.BLURRED_KEYFRAMES}")
        files += blurred
        if not blurred:
            notes.append(f"{folder.name}: no keyframes_blurred/ from te blur, so no keyframes")
    curated = blurred_set(data_dir / "playground" / "keyframes", "data/playground/keyframes")
    files += curated
    if not curated:
        notes.append("no blurred keyframes in data/playground/keyframes/ (te blur --dst it), "
                     "so teach it yourself has none to offer")
    for folder in data.clip_folders(data_dir):
        rel = f"data/{data.CLIPS.as_posix()}/{folder.name}"
        files += [(folder / f, f"{rel}/{f}") for f in CLIP_FILES if (folder / f).is_file()]
        video = data.blurred_video(folder)
        if video:
            record = f"{video}{data.BLUR_RECORD}"
            files += [(folder / video, f"{rel}/{video}"), (folder / record, f"{rel}/{record}")]
        else:
            notes.append(f"clip {folder.name}: no matching blur record, so its timeline "
                         "goes without the video (te playground-clip blurs it)")
    for path in [*sorted(eval_dir.glob("*.svg")), eval_dir / "results.md"]:
        if path.is_file() and data.FILE.fullmatch(path.name):
            files.append((path, f"eval/{path.name}"))
    return {"files": files, "notes": notes}


def stage(files: list[tuple[Path, str]], dst: Path) -> int:
    """Copy the files under dst at their staging paths (contents only, no metadata)."""
    for src, rel in files:
        out = Path(dst) / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
    return len(files)
