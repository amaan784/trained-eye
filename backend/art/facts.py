"""The numbers and pictures the README art shows, read from real run outputs only: a
skill's teach outputs (data/skills/<skill>/), its training run (data/train/), and te eval's
results (eval/results.json, written with results.md). Whatever is missing raises
MissingFact, so an image is never rendered with a placeholder number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from backend.config import DEFAULT_MODELS

# What each Meta model does in Trained Eye, keyed by its job in config.DEFAULT_MODELS.
JOBS = {
    "transcribe": ("Muse Voice Transcribe", "Hears the spoken request."),
    "compiler": ("Muse Wonder", ("Turns the request into a skill: what to watch, what counts, "
                 "what does not.")),
    "sam": ("SAM 3.1", "Finds the target in the teaching frames."),
    "image": ("Muse Image", ("Makes the variations: the condition, near-misses, other light "
              "and angles.")),
    "labeler": ("Muse Spark", "Labels every image and keeps the ones it agrees with."),
    "student_base": ("Muse Glimmer 30B", "Fine-tuned, open weights: watches every frame."),
}
ORDER = ("transcribe", "compiler", "sam", "image", "labeler", "student_base")


class MissingFact(LookupError):
    """A number or picture an image needs does not exist yet (not scored, not run)."""


def _json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise MissingFact(f"{path} does not exist yet") from None


def _jsonl(path: Path) -> list[dict]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise MissingFact(f"{path} does not exist yet") from None
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@dataclass
class Model:
    job: str
    name: str
    model_id: str
    what: str
    count: str  # what it did in this skill's runs, e.g. "300 images"
    highlight: bool = False


@dataclass
class SkillFacts:
    skill: str
    request: str
    read_back_target: str
    sam_prompt: str
    keyframes: int
    frames_with_box: int
    made: int  # images Muse Image returned for this skill
    kept: int  # kept after Spark's check
    rejected: int  # Spark said no or unsure
    real_examples: int
    examples: int
    yes: int
    no: int
    frame: Path  # a teaching keyframe SAM found the target in
    box: tuple[float, float, float, float]  # SAM's box on it, in its pixels
    frame_size: tuple[int, int]
    models: list[Model]


def skill_facts(data_dir: Path, skill: str, run: str) -> SkillFacts:
    root = Path(data_dir) / "skills" / skill
    spec = _json(root / "spec.json")
    request = _json(root / "request.json")
    stats = _json(root / "label_stats.json")
    manifest_rows = _jsonl(root / "synthetic" / "manifest.jsonl")
    made = [r for r in manifest_rows if r.get("file")]
    if stats.get("total") is None or stats.get("kept") is None:
        raise MissingFact(f"{root / 'label_stats.json'} has no total or kept")
    dataset = _json(root / "dataset" / f"{skill}-teach.manifest.json")
    index = dataset.get("examples_index") or []
    yes = sum(1 for e in index if e.get("answer") == "yes")
    no = sum(1 for e in index if e.get("answer") == "no")
    real = sum(1 for e in index if e.get("source") == "real")
    rois = _json(root / "roi.json")
    boxed = [r for r in rois if r.get("best")]
    if not boxed:
        raise MissingFact(f"no teaching keyframe with a SAM box in {root / 'roi.json'}")
    from backend.meta.cache import sha256_hex

    by_sha = {sha256_hex(p.read_bytes()): p for p in sorted((root / "keyframes").glob("kf_*"))}
    # the keyframe whose box covers about half of it: the target and the room around it
    best = min((r for r in boxed if r["keyframe_sha256"] in by_sha),
               key=lambda r: abs(_area(r["best"]["box"]) / (r["width"] * r["height"]) - 0.5),
               default=None)
    if best is None:
        raise MissingFact("the keyframes SAM boxed are not in the keyframes folder")
    result = _json(Path(data_dir) / "train" / f"{run}.result.json")
    if result.get("ok") is not True or result.get("smoke"):
        raise MissingFact(f"{run} has no finished training run (data/train/{run}.result.json)")
    image_ids = sorted({r.get("model") for r in made if r.get("model")})
    counts = {
        "transcribe": "the spoken request",
        "compiler": "1 skill from 1 sentence",
        "sam": f"{len(boxed)} of {len(rois)} teaching frames boxed",
        "image": f"{len(made)} images made",
        "labeler": f"{len(_jsonl(root / 'labels.jsonl'))} images labelled",
        "student_base": f"{dataset.get('examples', len(index))} examples, one epoch",
    }
    used = {"compiler", "sam", "labeler", "student_base"}
    if made:
        used.add("image")
    if request.get("from_audio"):
        used.add("transcribe")
    models = []
    for job in ORDER:
        if job not in used:
            continue
        model_id = image_ids[0] if job == "image" and image_ids else DEFAULT_MODELS[job]
        name, what = JOBS[job]
        models.append(Model(job, name, model_id, what, counts[job],
                            highlight=job == "student_base"))
    return SkillFacts(
        skill=skill, request=request.get("request", ""),
        read_back_target=spec.get("positive", ""),
        sam_prompt=(spec.get("target") or {}).get("sam_prompt") or "target", keyframes=len(rois),
        frames_with_box=len(boxed), made=len(made), kept=int(stats["kept"]),
        rejected=int(stats["total"]) - int(stats["kept"]), real_examples=real,
        examples=int(dataset.get("examples", len(index))), yes=yes, no=no,
        frame=by_sha[best["keyframe_sha256"]], box=tuple(best["best"]["box"]),
        frame_size=(int(best["width"]), int(best["height"])), models=models)


def _area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class Headline:
    text: str  # e.g. "Frame F1 0.81 on other kitchens"
    detail: str  # e.g. "fine-tuned Glimmer vs 0.42 base, 0.77 Muse Spark"
    footage: str


def headline(results_path: Path) -> Headline:
    """The banner's result: Experiment 4's three judges pooled (te eval's results.json,
    "experiment4"): fine-tuned Glimmer's frame F1 and latency, against base Glimmer and
    Muse Spark judging every frame. MissingFact until all three are scored."""
    results = _json(results_path)
    judges = {j.get("variant"): j for j in results.get("experiment4") or []}
    need = ("finetuned", "base", "spark")
    if not all(v in judges for v in need):
        raise MissingFact(f"Experiment 4 is not scored for all three judges in {results_path}")
    m = {v: judges[v].get("metrics") or {} for v in need}
    f1 = {v: m[v].get("f1") for v in need}
    lat = {v: m[v].get("latency_median_s") for v in need}
    usd = {v: judges[v].get("usd_per_hour") for v in need}
    if any(x is None for d in (f1, lat, usd) for x in d.values()):
        raise MissingFact(f"frame F1, latency or dollars per hour missing in {results_path}")
    footage = results.get("footage") or "the test footage"
    text = f"Frame F1 {f1['finetuned']:.2f} on {footage.split(' (')[0]}, {lat['finetuned']:.1f} s a frame"
    detail = (f"Fine-tuned Glimmer, against base Glimmer ({f1['base']:.2f}, "
              f"{lat['base']:.1f} s) and Muse Spark on every frame ({f1['spark']:.2f}, "
              f"{lat['spark']:.1f} s), at ${usd['finetuned']:.2f} an hour against "
              f"${usd['spark']:.2f} for Spark.")
    return Headline(text, detail, footage)
