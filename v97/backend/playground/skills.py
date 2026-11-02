"""How each taught skill was built, read from data/skills/<skill>/ as the teach flow left
it: the read-back, the keyframes, SAM's box on a teaching frame (signature visual 4)
and the labelled Muse Image variations for the training wall (signature visual 1).
Also the precomputed "teach it yourself" examples and the keyframes a visitor may pick.

Images are linked where they are (media/skills/...), never copied.
"""

from __future__ import annotations

import re
from pathlib import Path

from backend.meta.cache import sha256_hex
from backend.playground.data import FILE, IMAGES, child, number, read_json, read_jsonl
from backend.skills.spec import NAME_PATTERN, SkillSpec, read_back
from backend.teach import spread

WALL_TILES = 48  # the wall shows this many, spread over all of a skill's variations
EXAMPLE_TILES = 6  # per answer, as in a live run: 6 yes and 6 no
PROVIDED_PER_SKILL = 3
ROI_NOTE = ("SAM 3.1's output for these keyframes is stored, but reading boxes out of it "
            "is not built yet (the parser is a stub), so no outline can be drawn.")


def taught(data_dir: Path) -> list[tuple[Path, SkillSpec]]:
    """(folder, spec) of every taught skill that was not refused."""
    root = Path(data_dir) / "skills"
    out = []
    for folder in sorted(root.iterdir()) if root.is_dir() else []:
        if re.fullmatch(NAME_PATTERN, folder.name) and (spec := load_spec(folder)):
            out.append((folder, spec))
    return out


def load_spec(folder: Path) -> SkillSpec | None:
    try:
        spec = SkillSpec.model_validate_json((folder / "spec.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return None if spec.refused else spec


def keyframe_files(folder: Path) -> list[str]:
    kf = folder / "keyframes"
    return sorted(p.name for p in kf.iterdir() if p.suffix.lower() in IMAGES and p.is_file()
                  and FILE.fullmatch(p.name)) if kf.is_dir() else []


def variations(folder: Path) -> list[dict]:
    """Generated images that exist, with Spark's labels (from labels.jsonl), or unlabelled
    from the manifest if labelling has not run yet."""
    rows = [r for r in read_jsonl(folder / "labels.jsonl") if r.get("source") == "synthetic"]
    if not rows:
        rows = [{**r, "file": f"synthetic/{r['file']}"}
                for r in read_jsonl(folder / "synthetic" / "manifest.jsonl")
                if isinstance(r.get("file"), str)]
    return [r for r in rows if isinstance(r.get("file"), str)
            and child(folder, *r["file"].split("/"))]


def tile(skill: str, row: dict) -> dict:
    """One wall tile: the image, what it was made to show, and what Spark said."""
    spark = row.get("spark_label")
    return {"url": f"media/skills/{skill}/{row['file']}", "kind": row.get("kind"),
            "label": row.get("label"), "spark_label": spark,
            "kept": spark is not None and spark == row.get("label"),
            "condition": row.get("condition"), "nuisance": row.get("nuisance")}


def list_skills(data_dir: Path) -> list[dict]:
    out = []
    for folder, spec in taught(data_dir):
        stats = read_json(folder / "label_stats.json")
        out.append({"name": folder.name, "read_back": read_back(spec),
                    "keyframes": len(keyframe_files(folder)),
                    "variations": len(variations(folder)),
                    "kept": stats.get("kept") if isinstance(stats, dict) else None})
    return out


def skill_detail(data_dir: Path, name: str) -> dict | None:
    folder = Path(data_dir) / "skills" / name
    spec = load_spec(folder) if re.fullmatch(NAME_PATTERN, name) else None
    if spec is None:
        return None
    real = {Path(r["file"]).name: r for r in read_jsonl(folder / "labels.jsonl")
            if r.get("source") == "real" and isinstance(r.get("file"), str)}
    keyframes = [{"file": f, "url": f"media/skills/{name}/keyframes/{f}",
                  "spark_label": real.get(f, {}).get("spark_label"),
                  "human_label": real.get(f, {}).get("human_label")}
                 for f in keyframe_files(folder)]
    rows = variations(folder)
    return {"name": name, "request": spec.request, "read_back": read_back(spec),
            "target": spec.target.sam_prompt, "positive": spec.positive,
            "negatives": spec.negatives, "alert": spec.alert, "keyframes": keyframes,
            "roi": outline(folder, keyframes, spec.target.sam_prompt),
            "wall": [tile(name, r) for r in spread(rows, WALL_TILES)], "wall_total": len(rows),
            "label_stats": read_json(folder / "label_stats.json")}


def box_of(roi: dict) -> list[float] | None:
    best = roi.get("best")
    box = best.get("box") if isinstance(best, dict) else None
    if not isinstance(box, list) or len(box) != 4:
        return None
    x1, y1, x2, y2 = (number(v) for v in box)
    ok = None not in (x1, y1, x2, y2) and x2 > x1 and y2 > y1
    return [x1, y1, x2, y2] if ok else None


def outline(folder: Path, keyframes: list[dict], prompt: str) -> dict:
    """SAM's box on the first teaching frame that has one, with how many frames had one."""
    raw = read_json(folder / "roi.json", [])
    rois = [r for r in raw if isinstance(r, dict)] if isinstance(raw, list) else []
    boxed = {r.get("keyframe_sha256"): r for r in rois if box_of(r)}
    info = {"prompt": prompt, "frames": len(rois), "with_box": len(boxed), "frame": None,
            "note": None if boxed else ROI_NOTE}
    for kf in keyframes if boxed else []:
        roi = boxed.get(sha256_hex((folder / "keyframes" / kf["file"]).read_bytes()))
        if roi and number(roi.get("width")) and number(roi.get("height")):
            info["frame"] = {"url": kf["url"], "width": roi["width"], "height": roi["height"],
                             "box": box_of(roi), "score": number(roi["best"].get("score"))}
            break
    return info


def examples(data_dir: Path) -> list[dict]:
    """Precomputed "teach it yourself" results: per taught skill, its read-back and the
    first 6 yes and 6 no variations Spark labelled when it was taught."""
    out = []
    for folder, spec in taught(data_dir):
        rows = [r for r in variations(folder) if r.get("spark_label")]
        picked = ([r for r in rows if r.get("kind") == "positive"][:EXAMPLE_TILES]
                  + [r for r in rows if r.get("kind") == "hard_negative"][:EXAMPLE_TILES])
        if picked:
            out.append({"skill": folder.name, "request": spec.request,
                        "read_back": read_back(spec),
                        "variations": [tile(folder.name, r) for r in picked]})
    return out


def provided_keyframes(data_dir: Path) -> list[dict]:
    """Keyframes a visitor may pick: the curated (face-blurred) set in
    data/playground/keyframes/ if there is one, else a few of each taught skill's own."""
    curated = Path(data_dir) / "playground" / "keyframes"
    files = sorted(p.name for p in curated.iterdir() if p.suffix.lower() in IMAGES
                   and FILE.fullmatch(p.name)) if curated.is_dir() else []
    if files:
        return [{"id": f"playground/{f}", "url": f"media/playground/keyframes/{f}"}
                for f in files]
    return [{"id": f"{folder.name}/{f}", "url": f"media/skills/{folder.name}/keyframes/{f}"}
            for folder, _ in taught(data_dir)
            for f in spread(keyframe_files(folder), PROVIDED_PER_SKILL)]


def keyframe_bytes(data_dir: Path, keyframe_id: str) -> bytes | None:
    """The bytes of a provided keyframe, by its id; None for anything not on offer."""
    for kf in provided_keyframes(data_dir):
        if kf["id"] == keyframe_id:
            path = child(data_dir, *kf["url"].split("/")[1:])
            return path.read_bytes() if path else None
    return None
