"""eval/plan.json, the evaluation plan te eval reads: one evaluation per run of plan.py (and
Experiment 4's three judges) and skill, with its clips, the frame logs to replay them into
and their labels.

The plan (plan_skeleton) also has each replayed model's validation clips (the session 2
clips, which training never sees: realclips.SESSIONS) in "calibration", so te replay
calibrates every run's threshold on its own replays before it replays the test clips.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

from backend import config
from backend.cli_train import GPU
from backend.config import MODAL_GPU_USD_PER_HOUR, Settings
from backend.dataengine.realclips import find_clips, labels_path
from backend.eval.plan import spark_usd_per_check, training_runs

EXP4 = (("exp4-finetuned", "finetuned"), ("exp4-base", "base"), ("exp4-spark", "spark"))
TIMELINE_RUNS = ("exp1-mixed", "exp4-finetuned")  # their first test clip draws the timeline
# before the clips are recorded, the plan names them as docs/recording-protocol.md does
TEST_PLACEHOLDER, VALIDATION_PLACEHOLDER = "session3_test_a.mp4", "session2_a.mp4"


def _spec_alert(skill_dir: Path) -> str | None:
    """The spoken alert in the skill's spec.json, for the timeline's marker."""
    try:
        alert = json.loads((skill_dir / "spec.json").read_text(encoding="utf-8")).get("alert")
    except (OSError, ValueError, AttributeError):
        return None
    return alert if isinstance(alert, str) and alert.strip() else None


def log_runs(runs: dict[str, dict]) -> dict[str, str]:
    """For each arm, the run whose replays its evaluations read (data/watch/<run>/): its
    own, the first run with the same bundle, exp1-mixed's for Experiment 4's fine-tuned
    arm, and exp4-base and exp4-spark for the other two judges."""
    out = {name: run.get("same_bundle_as") or name for name, run in runs.items()}
    out.update({name: out["exp1-mixed"] if arm == "finetuned" else name for name, arm in EXP4})
    return out


def plan_skeleton(skills: list[str], settings: Settings, plan_dir: Path | None = None) -> dict:
    """eval/plan.json for te eval (backend/eval/report.py): one evaluation per run (and
    Experiment 4's three judges) and skill. Each test clip (data/clips/<skill>/
    session3_test_*, or a placeholder, pending until it exists) gets its clip, the frame
    log to replay it into, data/watch/<run>/<skill>/<clip>.jsonl, and its labels,
    <clip>.labels.json, relative to `plan_dir` (eval/). `run` is whose replays those are
    (log_runs: a run with the same bundle as an earlier one reads that one's), `scorer`
    who scores the frames. The first test clip of exp1-mixed and exp4-finetuned draws the
    live timeline, and every evaluation takes its skill's spoken alert (spec.json).
    `calibration` lists each replayed model's validation clips (session 2, or a placeholder)
    per skill, so te replay calibrates its own threshold first; Muse Spark needs none.
    `images` counts generated training images per skill (Experiment 3 without data: the
    target); `generation_usd` is what they cost at config prices (te teach paid for every
    image Spark judged; the unfiltered arm needs no labels). Students are priced on an
    H100 (null for the Mac); exp4-spark pays Spark per frame."""
    data, plan_dir = settings.data_dir, Path(plan_dir or config.PROJECT_ROOT / "eval")
    runs = {r["name"]: r for r in training_runs(skills, skills_root=data / "skills")}
    image_usd = settings.price(settings.model("image")).per_image
    check = spark_usd_per_check(settings)
    arms = [(r["name"], r["experiment"], r["arm"], r) for r in runs.values()]
    arms += [(name, 4, arm, runs["exp1-mixed"] if arm == "finetuned" else {})
             for name, arm in EXP4]
    replays = log_runs(runs)

    def rel(path: Path) -> str:
        return os.path.relpath(path, plan_dir)

    def clip_entry(run: str, skill: str, clip: Path) -> dict:
        return {"clip": rel(clip), "log": rel(data / "watch" / run / skill / f"{clip.stem}.jsonl"),
                "labels": rel(labels_path(clip))}

    entries = []
    for (name, experiment, arm, run), skill in itertools.product(arms, skills):
        counts = (run.get("examples") or {}).get("per_skill", {}).get(skill)
        images = (counts["synthetic"] if counts else run.get("limit")) or None
        stats_path = data / "skills" / skill / "label_stats.json"
        stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
        generation = images * image_usd if images else None
        if images and arm != "unfiltered":  # te teach paid for every image Spark judged
            kept = stats.get("kept")
            # label errors were generated and paid for too, though not judged
            paid = stats.get("total", 0) + stats.get("label_errors", 0)
            generation = images * paid / kept * (image_usd + check) if kept else None
        folder = data / "clips" / skill
        clips = [c for c in sorted(folder.glob("session3_test_*")) if c.suffix != ".json"]
        tests = [clip_entry(replays[name], skill, c)  # a placeholder named as the protocol says
                 for c in clips or [folder / TEST_PLACEHOLDER]]
        if name in TIMELINE_RUNS:
            tests[0]["timeline"] = True
        entry = {"name": f"{name}-{skill}", "experiment": experiment, "skill": skill,
                 "variant": arm, "run": replays[name],
                 "scorer": "spark" if arm == "spark" else "student", "clips": tests,
                 "gpu_usd_per_hour": MODAL_GPU_USD_PER_HOUR[GPU], "spark_usd_per_check": check,
                 "images": images, "highlight": arm == "finetuned",
                 "generation_usd": None if generation is None else round(generation, 4)}
        if alert := _spec_alert(data / "skills" / skill):
            entry["alert_text"] = alert
        if arm == "spark":  # its confirmations repeat the same request: cached, free
            entry.update(gpu_usd_per_hour=0.0, spark_usd_per_check=0.0, scorer_usd_per_frame=check)
        entries.append(entry)
    calibration = []
    for run in dict.fromkeys(e["run"] for e in entries if e["scorer"] == "student"):
        for skill in skills:
            folder = data / "clips" / skill
            valid = [c.path for c in find_clips(folder) if c.split == "val"]
            calibration.append({"run": run, "skill": skill, "clips": [
                clip_entry(run, skill, c) for c in valid or [folder / VALIDATION_PLACEHOLDER]]})
    return {"evaluations": entries, "calibration": calibration}


def write_plan(path: Path, plan: dict, *, force: bool = False) -> Path:
    """Write eval/plan.json, never over one that exists (it is edited by hand) unless forced."""
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; it may hold your edits (pass force to replace)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    return path
