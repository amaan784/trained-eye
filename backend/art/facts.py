"""The numbers and pictures the README art shows, read from real run outputs only: a
skill's teach outputs (data/skills/<skill>/: keyframes.json for each teaching frame's time,
roi.json for SAM's boxes, label_stats.json), its training run (data/train/), a replay that
te playground-clip made (its blurred video and frame log), and te eval's results
(eval/results.json, written with results.md). Whatever is missing raises MissingFact, so an
image is never rendered with a placeholder number.
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
    train_examples: int = 0
    val_examples: int = 0
    gpu: str = ""


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
    bundle = _json(Path(data_dir) / "train" / run / "manifest.json")
    if bundle.get("train_examples") is None:
        raise MissingFact(f"data/train/{run}/manifest.json has no train_examples")
    image_ids = sorted({r.get("model") for r in made if r.get("model")})
    counts = {
        "transcribe": "the spoken request",
        "compiler": "1 skill from 1 sentence",
        "sam": f"{len(boxed)} of {len(rois)} teaching frames boxed",
        "image": f"{len(made)} variations made",
        "labeler": (f"{len(_jsonl(root / 'labels.jsonl'))} images labelled, "
                    f"{int(stats['kept'])} variations kept"),
        "student_base": f"{bundle['train_examples']} training examples, {epochs_text()}",
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
        frame_size=(int(best["width"]), int(best["height"])), models=models,
        train_examples=int(bundle["train_examples"]),
        val_examples=int(bundle.get("val_examples") or 0), gpu=training_gpu(data_dir, run))


def epochs_text() -> str:
    from backend.training.axolotl import RECIPE

    n = RECIPE["num_epochs"]
    return "one epoch" if n == 1 else f"{n} epochs"


def training_gpu(data_dir: Path, run: str) -> str:
    """The GPU of the run's last training job in te train's Modal ledger."""
    path = Path(data_dir) / "modal_ledger.jsonl"
    rows = _jsonl(path) if path.is_file() else []
    gpus = [r.get("gpu") for r in rows if r.get("kind") == "train" and r.get("name") == run]
    if not gpus or not gpus[-1]:
        raise MissingFact(f"no training job for {run} in {path}")
    return str(gpus[-1])


@dataclass
class TeachFrame:
    path: Path
    t_s: float  # its time in the teaching clip
    box: tuple[float, float, float, float] | None  # SAM's box, in its pixels
    size: tuple[int, int]


def teaching_frames(data_dir: Path, skill: str) -> list[TeachFrame]:
    """The skill's teaching keyframes in clip order, each with its time (keyframes.json,
    written by te teach) and SAM's box (roi.json)."""
    from backend.meta.cache import sha256_hex

    root = Path(data_dir) / "skills" / skill
    listed = _json(root / "keyframes.json").get("keyframes") or []
    rois = {r["keyframe_sha256"]: r for r in _json(root / "roi.json")}
    out = []
    for k in listed:
        path = root / "keyframes" / k["name"]
        if not path.is_file() or k.get("t_s") is None:
            raise MissingFact(f"{path} or its time is missing (keyframes.json)")
        roi = rois.get(sha256_hex(path.read_bytes()))
        if roi is None:
            raise MissingFact(f"{k['name']} has no entry in {root / 'roi.json'}")
        best = roi.get("best")
        out.append(TeachFrame(path, float(k["t_s"]), tuple(best["box"]) if best else None,
                              (int(roi["width"]), int(roi["height"]))))
    if not out:
        raise MissingFact(f"no teaching frames in {root / 'keyframes.json'}")
    return sorted(out, key=lambda f: f.t_s)


def _area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class Headline:
    text: str  # the banner's bold line
    detail: str  # the sentence under it, with the comparison
    footage: str


def headline(results_path: Path) -> Headline:
    """The banner's result from Experiment 4's three judges pooled (te eval's results.json,
    "experiment4", the numbers results.md shows): the accuracy when the fine-tuned model is
    clearly ahead, else its speed, with the frame F1 of all three said plainly. MissingFact
    until all three are scored."""
    from backend.eval.readme import CLOSE_F1

    results = _json(results_path)
    judges = {j.get("variant"): j for j in results.get("experiment4") or []}
    need = ("finetuned", "base", "spark")
    if not all(v in judges for v in need):
        raise MissingFact(f"Experiment 4 is not scored for all three judges in {results_path}")
    m = {v: judges[v].get("metrics") or {} for v in need}
    f1 = {v: m[v].get("f1") for v in need}
    lat = {v: m[v].get("latency_median_s") for v in need}
    if any(x is None for d in (f1, lat) for x in d.values()):
        raise MissingFact(f"frame F1 or latency missing in {results_path}")
    footage = results.get("footage") or "the test footage"
    where = footage.split(" (")[0]
    scores = f"frame F1 {f1['finetuned']:.2f}, against {f1['base']:.2f} and {f1['spark']:.2f}"
    others = max(f1["base"], f1["spark"])
    if f1["finetuned"] - others > CLOSE_F1:
        return Headline(f"Spots it more often: frame F1 {f1['finetuned']:.2f}.",
                        f"Base Glimmer scores {f1['base']:.2f} and Muse Spark "
                        f"{f1['spark']:.2f} on {where}, and the fine-tuned model answers in "
                        f"{lat['finetuned']:.1f} s a frame.", footage)
    how = ("at about the same accuracy" if others - f1["finetuned"] <= CLOSE_F1
           else "though less accurate")
    return Headline(f"Fast enough to watch live: {lat['finetuned']:.1f} s a frame.",
                    f"Base Glimmer takes {lat['base']:.1f} s and Muse Spark "
                    f"{lat['spark']:.1f} s, {how} on {where} ({scores}).", footage)


@dataclass
class Replay:
    clip: str
    video: Path  # te blur's video of the clip
    points: list[tuple[float, float]]  # (t, the student's probability) per scored frame
    alert_t: float  # the first alert
    alert_text: str
    hold_s: float


def replay(data_dir: Path, clip_name: str, skill: str) -> Replay:
    """A te playground-clip replay of the author's own footage in which the alert fires:
    its blurred video and its frame log's scores."""
    from backend.playground import data as pg
    from backend.skills.spec import SKILL_HOLDS

    clip = pg.clip_timeline(Path(data_dir), clip_name, public=True)  # blurred video only
    if not clip or skill not in clip["skills"]:
        raise MissingFact(f"no playground clip {clip_name!r} with a {skill} log "
                          "(te playground-clip makes one)")
    sk = clip["skills"][skill]
    if not sk["alerts"]:
        raise MissingFact(f"the alert never fires in playground clip {clip_name!r}")
    if not clip["video"]:
        raise MissingFact(f"playground clip {clip_name!r} has no blurred video (te blur)")
    video = pg.clips_root(Path(data_dir)) / clip_name / Path(clip["video"]).name
    first = sk["alerts"][0]
    return Replay(clip_name, video, [(float(t), float(p)) for t, p in sk["points"]],
                  float(first["t"]), str(first["text"]), float(SKILL_HOLDS.get(skill, 0)))
