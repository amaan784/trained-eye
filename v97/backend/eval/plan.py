"""The experiment plan: every `te train` run of Experiments 1 to 3, their price before any
is launched, and the skeleton of eval/plan.json for the evaluation.

  exp1-real, exp1-synthetic, exp1-mixed  real keyframes only, generated images only, both
  exp2-filtered, exp2-unfiltered         mixed, with and without the Spark filter
  exp3-25 ... exp3-300                   that many generated images per skill (nested sets)

Steps stay fixed, not epochs. The recipe trains 1 epoch at a global batch of 32, so 25
images of one skill would be one optimizer step (three skills, three steps): too little
to tell from the base model, and the curve would measure steps instead of images. Every
run trains the same S steps, one epoch over 300 images per skill (optimizer_steps(300 *
skills): 10 for one skill, 29 for three), through te train --max-steps (Axolotl's
max_steps). The 300 point and exp1-mixed run a little over an epoch, close to the recipe;
a smaller set repeats (25 images about 13 times), so only the data (and the validation
passes) differ; small sets may overfit, the cost of fewer images. exp1-real does the same.

Runs with the same bundle and steps train the same thing (exp2-filtered is exp1-mixed; an
exp3 point at or over what a skill has is exp1-synthetic): see `same_bundle_as`.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path

from backend import config
from backend.cli import GPU, train_estimate
from backend.config import ASK_ABOVE_USD, MODAL_GPU_USD_PER_HOUR, Settings
from backend.dataengine import labels
from backend.eval.spark_scorer import default_log, sent_summary
from backend.meta.ledger import token_cost
from backend.skills.spec import check_name
from backend.teach import IMAGE_TOKENS
from backend.training.axolotl import GLOBAL_BATCH, optimizer_steps
from backend.training.prepare import MIN_FOR_VAL, VAL_SHARE, VARIANTS, skill_examples

EXP3_IMAGES = (25, 50, 100, 200, 300)
MODAL_CREDIT_USD = 150.0  # HANDOVER, M2: stop and ask if the runs would pass this
ARMS = (  # run, experiment, arm, bundle variant, dataset folder, images per skill
    ("exp1-real", 1, "real", "real", "dataset", None),
    ("exp1-synthetic", 1, "synthetic", "synthetic", "dataset", None),
    ("exp1-mixed", 1, "mixed", "mixed", "dataset", None),
    ("exp2-filtered", 2, "filtered", "mixed", "dataset", None),
    ("exp2-unfiltered", 2, "unfiltered", "mixed", "dataset-unfiltered", None),
    *((f"exp3-{n}", 3, str(n), "synthetic", "dataset", n) for n in EXP3_IMAGES),
)
EXP4 = (("exp4-finetuned", "finetuned"), ("exp4-base", "base"), ("exp4-spark", "spark"))


def fixed_steps(skills: list[str]) -> int:
    return optimizer_steps(EXP3_IMAGES[-1] * len(skills))


def steps_samples(max_steps: int, train: int, val: int) -> int:
    """Samples a fixed-steps run goes through, for train_estimate: its training samples and
    one validation pass per epoch (priced as training samples, so it errs high)."""
    samples = max_steps * GLOBAL_BATCH
    return samples + (math.ceil(samples / train) * val if train and val else 0)


def _counts(root: Path, skills: list[str], variant: str, dataset_dir: str,
            limit: int | None) -> dict:
    """Examples per skill as prepare.bundle picks them; validation is about 10%."""
    per_skill = {}
    for skill in skills:
        sources = [e["source"] for _, e in skill_examples(root / skill, dataset_dir)
                   if e["source"] in VARIANTS[variant]]
        used = min(len(sources), limit) if limit else len(sources)
        synthetic = used if variant == "synthetic" else sources.count("synthetic")
        per_skill[skill] = {"examples": used, "synthetic": synthetic, "available": len(sources)}
    total = sum(c["examples"] for c in per_skill.values())
    val = round(VAL_SHARE * total) if total >= MIN_FOR_VAL else 0
    return {"per_skill": per_skill, "train": total - val, "val": val}


def training_runs(skills: list[str], *, skills_root: Path | None = None) -> list[dict]:
    """Every training run with its `te train` arguments. With `skills_root`
    (data/skills), each run also gets its example counts and what it still needs."""
    names = [check_name(s) for s in skills]
    if not names:
        raise ValueError("give at least one skill")
    steps, runs, bundles = fixed_steps(names), [], {}
    for name, experiment, arm, variant, dataset_dir, limit in ARMS:
        args = ["te", "train", "--skills", ",".join(names), "--run", name, "--variant", variant]
        args += ["--dataset", dataset_dir] if dataset_dir != "dataset" else []
        args += ["--limit", str(limit)] if limit else []
        run = {"name": name, "experiment": experiment, "arm": arm, "variant": variant,
               "dataset": dataset_dir, "limit": limit, "max_steps": steps,
               "args": [*args, "--max-steps", str(steps)], "examples": None, "needs": []}
        effective = limit
        if skills_root is not None:
            root = Path(skills_root)
            missing = [s for s in names
                       if not (root / s / dataset_dir / f"{s}-teach.manifest.json").is_file()]
            if missing:
                how = "te teach" if dataset_dir == "dataset" else "te unfiltered --skill <skill>"
                run["needs"].append(f"no {dataset_dir} for {', '.join(missing)}: run {how}")
            else:
                run["examples"] = counts = _counts(root, names, variant, dataset_dir, limit)
                have = {s: c["available"] for s, c in counts["per_skill"].items()}
                run["needs"] += [f"{s} has {n} generated examples, under {limit}: this point "
                                 f"uses all {n} (teach with more --positives and --negatives)"
                                 for s, n in have.items() if limit and n < limit]
                if limit and all(n <= limit for n in have.values()):
                    effective = None  # the limit takes everything: the same bundle as none
        first = bundles.setdefault((variant, dataset_dir, effective), name)
        run["same_bundle_as"] = None if first == name else first
        runs.append(run)
    return runs


def plan_cost(runs: list[dict], timing: dict | None) -> dict:
    """GPU time and dollars per run and in total, priced like te train from the smoke's
    timing (data/train/smoke-timing.json) and steps_samples."""
    if train_estimate(1, timing) is None:
        return {"priced": False, "note": "no smoke timing yet: run te train with --smoke "
                "first (it saves data/train/smoke-timing.json)"}
    rows, total, repeats = [], 0.0, 0.0
    for run in runs:
        counts = run.get("examples") or {}
        estimate = train_estimate(steps_samples(run["max_steps"], counts.get("train", 0),
                                                counts.get("val", 0)), timing)
        total += estimate["usd"]
        repeats += estimate["usd"] if run.get("same_bundle_as") else 0.0
        rows.append({"name": run["name"], "minutes": round(estimate["seconds"] / 60, 1),
                     "usd": round(estimate["usd"], 2), "same_bundle_as": run.get("same_bundle_as")})
    return {"priced": True, "gpu": GPU, "gpu_usd_per_hour": MODAL_GPU_USD_PER_HOUR[GPU],
            "runs": rows, "total_usd": round(total, 2), "repeats_usd": round(repeats, 2),
            "over_ask": [r["name"] for r in rows if r["usd"] > ASK_ABOVE_USD],
            "over_credit": total > MODAL_CREDIT_USD}


def format_plan(runs: list[dict], cost: dict) -> str:
    """The plan as te experiments prints it."""
    prices = {r["name"]: f"${r['usd']:.2f}" for r in cost.get("runs", [])}
    lines = [f"{runs[0]['max_steps']} optimizer steps per run, fixed (backend/eval/plan.py)"]
    for run in runs:
        price = prices.get(run["name"], "?")
        lines.append(f"{run['name']:<16} {price:>7}  {' '.join(run['args'])}")
        if run["same_bundle_as"]:
            lines.append(f"{'':<25}same bundle as {run['same_bundle_as']}: can be skipped")
        lines += [f"{'':<25}needs: {need}" for need in run["needs"]]
    if not cost["priced"]:
        return "\n".join([*lines, cost["note"]])
    lines.append(f"total about ${cost['total_usd']:.2f} at ${cost['gpu_usd_per_hour']:.2f} an "
                 f"hour of {cost['gpu']} (${cost['repeats_usd']:.2f} of it in repeated bundles)")
    if cost["over_ask"]:
        lines.append(f"over ${ASK_ABOVE_USD:.0f} each, ask first: {', '.join(cost['over_ask'])}")
    if cost["over_credit"]:
        lines.append(f"the total passes ${MODAL_CREDIT_USD:.0f} of Modal credit: ask first")
    return "\n".join(lines)


def spark_usd_per_check(settings: Settings) -> float:
    """The measured mean from te watch --scorer spark, else the estimate te teach uses."""
    measured = sent_summary(default_log(settings.data_dir))["usd_per_check"]
    price = settings.price(settings.model("labeler"))
    return measured or round(token_cost(price, IMAGE_TOKENS + 250, labels.MAX_TOKENS[0]), 6)


def plan_skeleton(skills: list[str], settings: Settings, plan_dir: Path | None = None) -> dict:
    """eval/plan.json for te eval (backend/eval/report.py): one evaluation per run (and
    Experiment 4's three judges) and skill. Each test clip (data/clips/<skill>/
    session3_test_*, or a placeholder, pending until it exists) gets the frame log to
    replay it into, data/watch/<run>/<skill>/<clip>.jsonl, and its labels, <clip>.labels.json,
    relative to `plan_dir` (eval/). `images` counts generated training images per skill
    (Experiment 3 without data: the target); `generation_usd` is what they cost at config
    prices (te teach paid for every image Spark judged; the unfiltered arm needs no labels).
    Students are priced on an H100 (null for the Mac); exp4-spark pays Spark per frame."""
    data, plan_dir = settings.data_dir, Path(plan_dir or config.PROJECT_ROOT / "eval")
    runs = {r["name"]: r for r in training_runs(skills, skills_root=data / "skills")}
    image_usd = settings.price(settings.model("image")).per_image
    check = spark_usd_per_check(settings)
    arms = [(r["name"], r["experiment"], r["arm"], r) for r in runs.values()]
    arms += [(name, 4, arm, runs["exp1-mixed"] if arm == "finetuned" else {})
             for name, arm in EXP4]
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
        entry = {"name": f"{name}-{skill}", "experiment": experiment, "skill": skill,
                 "variant": arm, "clips": [
                     {"log": os.path.relpath(data / "watch" / name / skill / f"{c.stem}.jsonl",
                                             plan_dir),
                      "labels": os.path.relpath(c.with_suffix(".labels.json"), plan_dir)}
                     for c in clips or [folder / "session3_test.mp4"]],
                 "gpu_usd_per_hour": MODAL_GPU_USD_PER_HOUR[GPU], "spark_usd_per_check": check,
                 "images": images, "highlight": arm == "finetuned",
                 "generation_usd": None if generation is None else round(generation, 4)}
        if arm == "spark":  # its confirmations repeat the same request: cached, free
            entry.update(gpu_usd_per_hour=0.0, spark_usd_per_check=0.0, scorer_usd_per_frame=check)
        entries.append(entry)
    return {"evaluations": entries}


def write_plan(path: Path, plan: dict, *, force: bool = False) -> Path:
    """Write eval/plan.json, never over one that exists (it is edited by hand) unless forced."""
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; it may hold your edits (pass force to replace)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    return path
