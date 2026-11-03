"""The experiment plan: every `te train` run of Experiments 1 to 3, their price before any
is launched, and the skeleton of eval/plan.json for the evaluation.

  exp1-real, exp1-synthetic, exp1-mixed  real only (keyframes, their crops and the
                                        session 1 clips), generated images only, both
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
exp3 point at or over what a skill has is exp1-synthetic): see `same_bundle_as`. Such a
run, and Experiment 4's fine-tuned arm (exp1-mixed's model), are never replayed apart:
their evaluations read the first run's frame logs (an evaluation's "run").

The price (plan_cost) is everything the $150 of Modal credit pays for: the training runs,
the final run (every skill, mixed, the recipe's epoch: `final_run`), an export and a
serving estimate per evaluated model (each run with its own bundle, the final run, and
exp4-base's base endpoint, which serves without an export), and the Modal spend already
in data/modal_ledger.jsonl. Runs whose finished result is on this Mac (`progress`) count
in that spend instead of their estimate. An export is priced at te export's limit
(EXPORT_MAX_HOURS) until one is timed; serving at an endpoint-hours guess per model
(te experiments --endpoint-hours). Serving the runs as adapters on the base endpoint
(te serve-student --runs, untried) would need no export per run.

The plan (plan_skeleton) also has each replayed model's validation clips (the session 2
clips, which training never sees: realclips.SESSIONS) in "calibration", so te replay
calibrates every run's threshold on its own replays before it replays the test clips.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path

from backend import config
from backend.cli import EXPORT_MAX_HOURS, GPU, train_estimate
from backend.config import ASK_ABOVE_USD, MODAL_GPU_USD_PER_HOUR, Settings
from backend.dataengine import labels
from backend.dataengine.realclips import find_clips, labels_path
from backend.eval.spark_scorer import default_log, sent_summary
from backend.meta.ledger import image_tokens, token_cost
from backend.skills.spec import check_name
from backend.training.axolotl import GLOBAL_BATCH, optimizer_steps
from backend.training.prepare import (
    CLIP_VARIANTS,
    MIN_FOR_VAL,
    VAL_SHARE,
    VARIANTS,
    clip_examples,
    skill_examples,
)

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
BASE_RUN = "exp4-base"  # the base endpoint (te serve-student --base): served, never exported
FINAL_RUN = "final"  # HANDOVER, M2: about 10 experiment runs plus 1 final
DEFAULT_ENDPOINT_HOURS = 1.0  # our guess per model, not measured: its replays at 1 fps,
# the cold start and the idle minutes before the container stops (te experiments
# --endpoint-hours)
TIMELINE_RUNS = ("exp1-mixed", "exp4-finetuned")  # their first test clip draws the timeline
# before the clips are recorded, the plan names them as docs/recording-protocol.md does
TEST_PLACEHOLDER, VALIDATION_PLACEHOLDER = "session3_test_a.mp4", "session2_a.mp4"


def fixed_steps(skills: list[str]) -> int:
    return optimizer_steps(EXP3_IMAGES[-1] * len(skills))


def steps_samples(max_steps: int, train: int, val: int) -> int:
    """Samples a fixed-steps run goes through, for train_estimate: its training samples and
    one validation pass per epoch (priced as training samples, so it errs high)."""
    samples = max_steps * GLOBAL_BATCH
    return samples + (math.ceil(samples / train) * val if train and val else 0)


def _counts(root: Path, skills: list[str], variant: str, dataset_dir: str,
            limit: int | None) -> dict:
    """Examples per skill as prepare.bundle picks them: the variant's teach examples and,
    for real and mixed, te clips' session clips (session 1 frames train, session 2 frames
    validate). About 10% of the teach examples validate once there are MIN_FOR_VAL."""
    per_skill, teach_total, clip_train, clip_val = {}, 0, 0, 0
    for skill in skills:
        sources = [e["source"] for _, e in skill_examples(root / skill, dataset_dir)
                   if e["source"] in VARIANTS[variant]]
        recorded = clip_examples(root / skill) if variant in CLIP_VARIANTS else None
        splits = [e.get("split") for _, e in recorded[1]] if recorded else []
        session2 = splits.count("val")
        pool = len(sources) + len(splits) - session2  # the limit cuts these, never session 2
        used = min(pool, limit) if limit else pool
        teach = round(used * len(sources) / pool) if pool else 0
        synthetic = teach if variant == "synthetic" else sources.count("synthetic")
        per_skill[skill] = {"examples": used + session2, "synthetic": synthetic,
                            "available": len(sources), "clips_train": used - teach,
                            "clips_val": session2}
        teach_total, clip_train, clip_val = (teach_total + teach, clip_train + used - teach,
                                             clip_val + session2)
    val = round(VAL_SHARE * teach_total) if teach_total >= MIN_FOR_VAL else 0
    return {"per_skill": per_skill, "train": teach_total - val + clip_train,
            "val": val + clip_val}


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


def final_run(skills: list[str], *, skills_root: Path | None = None) -> dict:
    """The final run: every skill, mixed, as te train trains it without --max-steps (the
    recipe's one epoch over every example), shaped like a training_runs entry."""
    names = [check_name(s) for s in skills]
    if not names:
        raise ValueError("give at least one skill")
    run = {"name": FINAL_RUN, "experiment": None, "arm": "final", "variant": "mixed",
           "dataset": "dataset", "limit": None, "max_steps": None, "same_bundle_as": None,
           "skills": names,
           "args": ["te", "train", "--skills", ",".join(names), "--run", FINAL_RUN,
                    "--variant", "mixed"], "examples": None, "needs": []}
    if skills_root is not None:
        root = Path(skills_root)
        missing = [s for s in names
                   if not (root / s / "dataset" / f"{s}-teach.manifest.json").is_file()]
        if missing:
            run["needs"].append(f"no dataset for {', '.join(missing)}: run te teach")
        else:
            run["examples"] = _counts(root, names, "mixed", "dataset", None)
    return run


def _result(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def progress(data_dir: Path) -> dict[str, set[str]]:
    """What is done on this Mac: runs with a finished full training run (data/train/
    <run>.result.json, ok and not a smoke) and runs with a successful export
    (<run>.export.json with its merged_dir). Their cost is in the Modal ledger."""
    folder = Path(data_dir) / "train"
    trained = {p.name.removesuffix(".result.json") for p in folder.glob("*.result.json")
               if (r := _result(p)).get("ok") is True and not r.get("smoke")}
    exported = {p.name.removesuffix(".export.json") for p in folder.glob("*.export.json")
                if (r := _result(p)).get("ok") is not False and r.get("merged_dir")}
    return {"trained": trained, "exported": exported}


def _estimate(run: dict, timing: dict) -> dict:
    counts = run.get("examples") or {}
    train, val = counts.get("train", 0), counts.get("val", 0)
    if run.get("max_steps"):
        samples = steps_samples(run["max_steps"], train, val)
    else:  # te train without --max-steps: one epoch; before the data, the 300-image target
        samples = train + val or fixed_steps(run["skills"]) * GLOBAL_BATCH
    return train_estimate(samples, timing)


def plan_cost(runs: list[dict], timing: dict | None, *, final: dict | None = None,
              endpoint_hours: float = DEFAULT_ENDPOINT_HOURS,
              export_hours: float | None = None, spent_usd: float = 0.0,
              done: dict[str, set[str]] | None = None) -> dict:
    """GPU time and dollars per run, priced like te train from the smoke's timing
    (data/train/smoke-timing.json) and steps_samples, plus the final run, an export
    (`export_hours`, default EXPORT_MAX_HOURS) and `endpoint_hours` of serving per evaluated
    model, and what is spent already. Runs `done` (progress()) are in `spent_usd`, not in
    what remains. The total is compared with MODAL_CREDIT_USD."""
    done = done or {"trained": set(), "exported": set()}
    rate = MODAL_GPU_USD_PER_HOUR[GPU]
    measured = export_hours is not None
    export_hours = float(export_hours if measured else EXPORT_MAX_HOURS)
    models = [{"name": r["name"], "export": True} for r in runs if not r.get("same_bundle_as")]
    models += [{"name": BASE_RUN, "export": False}]
    models += [{"name": final["name"], "export": True}] if final else []
    for m in models:
        m["exported"] = m["export"] and m["name"] in done["exported"]
        m["export_usd"] = (round(export_hours * rate, 2) if m["export"] and not m["exported"]
                           else 0.0)
        m["serving_usd"] = round(endpoint_hours * rate, 2)
    exports = round(sum(m["export_usd"] for m in models), 2)
    serving = round(sum(m["serving_usd"] for m in models), 2)
    base = {"gpu": GPU, "gpu_usd_per_hour": rate, "models": models, "export_usd": exports,
            "export_hours": export_hours, "export_measured": measured,
            "endpoint_hours": endpoint_hours, "serving_usd": serving,
            "spent_usd": round(spent_usd, 2)}
    if train_estimate(1, timing) is None:
        return {**base, "priced": False, "note": "no smoke timing yet: run te train with "
                "--smoke first (it saves data/train/smoke-timing.json)"}
    rows, training, repeats = [], 0.0, 0.0
    for run in [*runs, *([final] if final else [])]:
        estimate = _estimate(run, timing)
        is_done = run["name"] in done["trained"]
        if not is_done:
            training += estimate["usd"]
            repeats += estimate["usd"] if run.get("same_bundle_as") else 0.0
        rows.append({"name": run["name"], "minutes": round(estimate["seconds"] / 60, 1),
                     "usd": round(estimate["usd"], 2), "done": is_done,
                     "same_bundle_as": run.get("same_bundle_as")})
    total = spent_usd + training + exports + serving
    return {**base, "priced": True, "runs": rows, "training_usd": round(training, 2),
            "repeats_usd": round(repeats, 2), "total_usd": round(total, 2),
            "over_ask": [r["name"] for r in rows if r["usd"] > ASK_ABOVE_USD and not r["done"]],
            "over_credit": total > MODAL_CREDIT_USD}


def format_plan(runs: list[dict], cost: dict, final: dict | None = None) -> str:
    """The plan as te experiments prints it."""
    prices = {r["name"]: "done" if r["done"] else f"${r['usd']:.2f}"
              for r in cost.get("runs", [])}
    lines = [f"{runs[0]['max_steps']} optimizer steps per run, fixed (backend/eval/plan.py)"]
    for run in [*runs, *([final] if final else [])]:
        price = prices.get(run["name"], "?")
        lines.append(f"{run['name']:<16} {price:>7}  {' '.join(run['args'])}")
        if run["same_bundle_as"]:
            lines.append(f"{'':<25}same bundle as {run['same_bundle_as']}: can be skipped "
                         "(its evaluations read that run's replays)")
        if run is final:
            lines.append(f"{'':<25}the final run: the recipe's epoch over every example, "
                         "not fixed steps")
        lines += [f"{'':<25}needs: {need}" for need in run["needs"]]
    models = cost["models"]
    to_export = [m["name"] for m in models if m["export"] and not m["exported"]]
    basis = ("the longest export timed so far" if cost["export_measured"] else
             "the most te export can bill, until one is timed")
    done_exports = [m["name"] for m in models if m["exported"]]
    lines.append(f"models to evaluate: {', '.join(m['name'] for m in models)}; each is served "
                 f"(replays, te regress) and exported first, except {BASE_RUN} (the base "
                 "endpoint)"
                 + (f" and {', '.join(done_exports)} (exported already)" if done_exports
                    else ""))
    lines.append(f"exports about ${cost['export_usd']:.2f}: {len(to_export)} models at "
                 f"{cost['export_hours']:.1f} h ({basis}); serving about "
                 f"${cost['serving_usd']:.2f}: {len(models)} models at "
                 f"{cost['endpoint_hours']:g} h each (a guess: te experiments --endpoint-hours)")
    lines.append(f"Modal spend so far: ${cost['spent_usd']:.2f} (te spend)")
    if not cost["priced"]:
        return "\n".join([*lines, cost["note"]])
    left_out = [r["name"] for r in cost["runs"] if r["done"]]
    lines.append(f"training about ${cost['training_usd']:.2f} at "
                 f"${cost['gpu_usd_per_hour']:.2f} an hour of {cost['gpu']} "
                 f"(${cost['repeats_usd']:.2f} of it in repeated bundles"
                 + (f"; trained already, in the spend: {', '.join(left_out)}" if left_out
                    else "") + ")")
    lines.append(f"total about ${cost['total_usd']:.2f} of the ${MODAL_CREDIT_USD:.0f} Modal "
                 "credit: spend so far, training, exports and serving")
    if cost["over_ask"]:
        lines.append(f"over ${ASK_ABOVE_USD:.0f} each, ask first: {', '.join(cost['over_ask'])}")
    if cost["over_credit"]:
        lines.append(f"the total passes ${MODAL_CREDIT_USD:.0f} of Modal credit: ask first "
                     "(serving the runs as adapters, te serve-student --runs, needs no export "
                     "each; untried)")
    return "\n".join(lines)


def spark_usd_per_check(settings: Settings) -> float:
    """The measured mean from te watch --scorer spark, else the estimate te teach uses (at
    the ledger's measured tokens per image, ledger.image_tokens)."""
    measured = sent_summary(default_log(settings.data_dir))["usd_per_check"]
    price = settings.price(settings.model("labeler"))
    per_image = image_tokens(settings).per_image
    return measured or round(token_cost(price, per_image + 250, labels.MAX_TOKENS[0]), 6)


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
