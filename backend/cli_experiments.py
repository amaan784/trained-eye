"""te commands for M3's experiments (registered by backend/cli.py).

  te experiments    print every training run of Experiments 1 to 3 with its price
  te replay         replay a run's pending clips of eval/plan.json into the plan's logs
  te unfiltered     write the Experiment 2 arm without the Spark filter (free)
  te clips          labelled session 1 and 2 clips as real examples, dataset-clips (free)
  te spark-errors   Spark's error rate against the author's own image labels (free)
  te regress        the regression check: base vs fine-tuned on general questions
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _hours(text: str) -> float:
    value = float(text)
    if not value > 0 or value == float("inf"):
        raise argparse.ArgumentTypeError(f"must be a number of hours over 0, got {text}")
    return value


def _per_second(text: str) -> float:
    value = float(text)
    if not 0 < value <= 30:
        raise argparse.ArgumentTypeError(f"must be over 0 and at most 30 frames a second, "
                                         f"got {text}")
    return value


def _seconds(text: str) -> float:
    value = float(text)
    if not 0 <= value < float("inf"):
        raise argparse.ArgumentTypeError(f"must be 0 or more seconds, got {text}")
    return value


def _fraction(text: str) -> float:
    from backend.cli import fraction

    return fraction(text)


def _positive(text: str) -> int:
    from backend.cli import positive

    return positive(text)


def _skills(text: str) -> list[str]:
    from backend.skills.spec import check_name

    names = [check_name(n.strip()) for n in text.split(",") if n.strip()]
    if not names:
        raise ValueError("--skills needs at least one skill name")
    return names


def cmd_experiments(args) -> int:
    from backend.cli import load_settings, shown_path
    from backend.config import PROJECT_ROOT
    from backend.eval.plan import (
        DEFAULT_ENDPOINT_HOURS,
        final_run,
        format_plan,
        plan_cost,
        plan_skeleton,
        progress,
        training_runs,
        write_plan,
    )
    from backend.training import gpu_ledger

    settings = load_settings()
    try:
        names = _skills(args.skills)
        runs = training_runs(names, skills_root=settings.data_dir / "skills")
        final = final_run(names, skills_root=settings.data_dir / "skills")
    except (OSError, ValueError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    timing_path = settings.data_dir / "train" / "smoke-timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.is_file() else None
    exports = [r["seconds"] for r in gpu_ledger.read(settings.data_dir) if r["kind"] == "export"
               and isinstance(r.get("seconds"), (int, float))]
    cost = plan_cost(runs, timing, final=final,
                     endpoint_hours=args.endpoint_hours or DEFAULT_ENDPOINT_HOURS,
                     export_hours=max(exports) / 3600 if exports else None,
                     spent_usd=gpu_ledger.spent_usd(settings.data_dir),
                     done=progress(settings.data_dir))
    print(format_plan(runs, cost, final))
    if args.write_plan:
        plan = plan_skeleton(names, settings)
        try:
            path = write_plan(PROJECT_ROOT / "eval" / "plan.json", plan, force=args.force)
        except FileExistsError as err:
            print(f"te: {err} (--force replaces it)", file=sys.stderr)
            return 2
        replayed = list(dict.fromkeys(e["run"] for e in plan["evaluations"]))
        print(f"evaluation plan: {shown_path(path)}; with each model served, te replay --plan "
              f"{shown_path(path)} --run <run> fills its logs, for runs {', '.join(replayed)}")
    return 0


def _clips(n: int) -> str:
    return f"{n} clip{'' if n == 1 else 's'}"


def _replay_note(job, failed: dict, held: dict) -> str | None:
    """Why a clip of te replay is still pending (None: replayed and labelled)."""
    if job.log in failed:
        return f"replay failed: {failed[job.log]}"
    if job.log in held:
        return held[job.log]
    if not job.log.is_file():
        return (f"no clip yet at {job.named}: record it (docs/recording-protocol.md); "
                "one named otherwise needs the plan written again (te experiments "
                "--write-plan --force)" if job.clip is None else "not replayed yet")
    if not job.labels.is_file():
        return f"no labels yet: te label --clip {job.clip or job.named}"
    return None


HELD = ("held back: {run} has no threshold for {skill} yet, and its validation clips are "
        "not all recorded, labelled and replayed (or calibration found none); or pass "
        "--threshold")


def _held(jobs, todo, calibrates, settings, run: str, threshold) -> dict:
    """Test clips te replay leaves for later: their skill is calibrated for the run in the
    plan but has no run threshold, and this replay cannot calibrate it either."""
    from backend.watch.watcher import run_threshold_file

    def can_calibrate(skill: str) -> bool:
        valid = [j for j in jobs if j.kind == "validation" and j.skill == skill]
        return bool(valid) and all(j.labels.is_file() and (j.log.is_file() or j in todo)
                                   for j in valid)
    return {j.log: HELD.format(run=run, skill=j.skill) for j in todo
            if j.kind == "test" and j.skill in calibrates and threshold is None
            and not run_threshold_file(settings.data_dir, run, j.skill).is_file()
            and not can_calibrate(j.skill)}


def _stale(jobs, calibrates, settings, run: str, held: dict) -> dict:
    """Test logs scored at another threshold than the run's calibrated one (it was
    calibrated again, or the log came from te watch): replayed again only by hand."""
    from backend.watch.watcher import read_threshold, run_threshold_file

    out = {}
    for job in jobs:
        if job.kind != "test" or job.skill not in calibrates or not job.log.is_file() or (
                job.log in held):
            continue
        try:
            now = read_threshold(run_threshold_file(settings.data_dir, run, job.skill))
            used = {round(float(r["threshold"]), 4) for r in map(json.loads, job.log.read_text(
                encoding="utf-8").splitlines()) if r.get("skill") == job.skill}
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if now is not None and used and used != {round(now, 4)}:
            out[job.log] = (f"scored at {', '.join(f'{u:.3f}' for u in sorted(used))}, but "
                            f"{run}'s threshold for {job.skill} is {now:.3f} now: delete this "
                            "log and run te replay again to score it at that")
    return out


def serving_note(data_dir: Path, run: str, model: str) -> str | None:
    """A warning when te serve-student last deployed another run's merged model and the
    model asked for is not named after this run (te serve-student --runs) or the base."""
    from backend.cli import BASE_MODEL_NAME
    from backend.training import runs

    served = runs.serving(data_dir)
    if served is None or served["run"] == run or model in (run, BASE_MODEL_NAME):
        return None
    return (f"note: te serve-student last deployed run {served['run']}, not {run}: make sure "
            f"TE_STUDENT_URL serves {run}, or its replays score another model")


def _replay_scorer(args, settings, scorer: str, todo, folders, frames_of, client, run: str,
                   folded: bool = False):
    """The frame scorer for te replay, after Spark's price and question or the student's
    warm-up; None when the replay should stop (exit 1)."""
    from backend.cli import ask
    from backend.config import ASK_ABOVE_USD
    from backend.dataengine.keyframes import load_frames
    from backend.eval import replay
    from backend.inference import student
    from backend.watch.watcher import load_skill

    if scorer == "spark":
        from backend.eval.plan import spark_usd_per_check
        from backend.eval.spark_scorer import SparkScorer, default_log

        settings.secret("MODEL_API_KEY")
        frames = sum(len(f) for f in frames_of.values())
        cost = frames * spark_usd_per_check(settings)
        print(f"Muse Spark on {frames} frames of {_clips(len(todo))}: about ${cost:.2f} before "
              "cache hits")
        if (cost > ASK_ABOVE_USD or not args.yes) and not ask("Spend that?"):
            return None
        return SparkScorer(client, latency_log=default_log(settings.data_dir))
    model = args.model or settings.student_model
    if note := serving_note(settings.data_dir, run, model):
        print(note)
    scoring = student.OpenAIStudent(
        settings.student_url, model, api_key=settings.secrets.get("STUDENT_API_KEY"),
        max_tokens=args.max_tokens or settings.student_max_tokens,
        system_in_user=folded)
    print(f"{run}: asking {model} at {settings.student_url}")
    warm = replay.DEFAULT_WARM_UP_S if args.warm_up_s is None else args.warm_up_s
    if not warm:
        return scoring
    first = todo[0]
    frame = load_frames(first.clip, fps=args.fps)[0]
    spec = load_skill(folders[first.skill]).spec
    return scoring if replay.warm_up(lambda: scoring.score(frame.data, spec), warm) else None


def cmd_replay(args) -> int:
    from argparse import Namespace

    from backend.cli import cmd_calibrate, load_settings, make_client, shown_path
    from backend.dataengine.keyframes import load_frames
    from backend.dataengine.labels import label_image
    from backend.eval import replay
    from backend.eval.report import Plan
    from backend.training import runs
    from backend.watch.watcher import load_skill, run_threshold_file, system_in_user

    settings = load_settings()
    try:
        run = runs.check_run(args.run)
        plan_path = Path(args.plan)
        plan = Plan.model_validate_json(plan_path.read_text(encoding="utf-8"))
        base = plan_path.resolve().parent
        jobs = replay.plan_jobs(plan, base, run)
        scorer = replay.plan_scorer(plan, run, args.scorer)
    except (OSError, ValueError) as err:  # pydantic's ValidationError is a ValueError
        print(f"te: {err}", file=sys.stderr)
        return 2
    calibrates = replay.calibrated_skills(plan, run) if scorer == "student" else set()
    todo = [j for j in jobs if not j.log.is_file() and j.clip is not None]
    held = _held(jobs, todo, calibrates, settings, run, args.threshold)
    todo = [j for j in todo if j.log not in held]
    failed: dict[Path, str] = {}
    replayed: set[Path] = set()
    scoring = confirm = None
    folders: dict[str, Path] = {}
    frames_of: dict[Path, list] = {}
    if todo:
        if scorer == "student" and not settings.student_url:
            print(f"te: replay needs the endpoint serving {run}: set TE_STUDENT_URL (te "
                  "serve-student prints it)", file=sys.stderr)
            return 2
        print(f"{run}: {sum(j.kind == 'validation' for j in todo)} validation and "
              f"{sum(j.kind == 'test' for j in todo)} test clips to replay ({scorer})")
        folders = {j.skill: settings.data_dir / "skills" / j.skill for j in todo}
        try:
            for name, folder in folders.items():
                if not (folder / "spec.json").is_file():
                    raise ValueError(f"no taught skill named {name!r}")
            folded = scorer == "student" and system_in_user(
                folders.values(), bundle=runs.bundle_manifest(settings.data_dir, run),
                default=settings.system_in_user)
            if scorer == "spark":  # priced before anything is sent: load every clip now
                frames_of = {j.log: load_frames(j.clip, fps=args.fps) for j in todo}
            client = make_client(settings)
            scoring = _replay_scorer(args, settings, scorer, todo, folders, frames_of, client,
                                     run, folded=folded)
        except (ValueError, RuntimeError, IndexError, subprocess.CalledProcessError) as err:
            print(f"te: {err}", file=sys.stderr)  # IndexError: a clip without frames
            return 2
        if scoring is None:
            return 1
        if not args.no_cloud:
            def confirm(image, spec):
                return label_image(client, spec, image)

    def replay_all(kind: str) -> None:
        for job in [j for j in todo if j.kind == kind]:
            if (kind == "test" and job.skill in calibrates and args.threshold is None
                    and not run_threshold_file(settings.data_dir, run, job.skill).is_file()):
                held[job.log] = HELD.format(run=run, skill=job.skill)
                continue
            try:
                frames = frames_of.pop(job.log, None) or load_frames(job.clip, fps=args.fps)
                skill = load_skill(folders[job.skill], run=run, threshold=args.threshold)
            except (ValueError, RuntimeError, subprocess.CalledProcessError) as err:
                failed[job.log] = str(err)
                continue
            if scorer == "spark" and args.threshold is None:  # only a yes counts
                skill.trigger.threshold, skill.threshold_source = 1.0, "spark"
            scored, errors = replay.replay_clip(job, skill, scoring, frames, run=run,
                                                confirm=confirm if kind == "test" else None)
            print(f"  {kind} {shown_path(job.log)}: {scored} frames scored, {errors} student "
                  f"errors, threshold {skill.trigger.threshold:.3f} ({skill.threshold_source})")
            if scored:
                replayed.add(job.log)
            else:
                failed[job.log] = "every frame was a student error"

    replay_all("validation")
    for cal in [c for c in plan.calibration if c.run == run and c.skill in calibrates]:
        logs = [(base / c.log).resolve() for c in cal.clips]
        labels = [(base / c.labels).resolve() for c in cal.clips]
        target = run_threshold_file(settings.data_dir, run, cal.skill)
        fresh = any(log in replayed for log in logs)
        if (target.is_file() and not fresh) or not all(p.is_file() for p in logs + labels):
            continue
        cmd_calibrate(Namespace(skill=cal.skill, run=run, log=[str(p) for p in logs],
                                labels=[str(p) for p in labels]))
    replay_all("test")
    if args.threshold is None:
        held.update(_stale(jobs, calibrates, settings, run, held))
    notes = [(j, note) for j in jobs if (note := _replay_note(j, failed, held))]
    print(f"{run}: {_clips(len(replayed))} replayed now; {len(jobs) - len(notes)} of {len(jobs)} "
          "are replayed and labelled, ready for make eval")
    if notes:
        print("still pending:")
        for job, note in notes:
            print(f"  {job.kind} {job.skill} {shown_path(job.log)}: {note}")
    return 1 if failed else 0


def cmd_unfiltered(args) -> int:
    from backend.cli import load_settings
    from backend.eval.spark_errors import write_unfiltered_dataset
    from backend.skills.spec import check_name

    settings = load_settings()
    try:
        manifest = write_unfiltered_dataset(
            settings.data_dir / "skills" / check_name(args.skill),
            train_on_model_outputs=settings.train_on_model_outputs,
            system_in_user=settings.system_in_user)
    except (OSError, ValueError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    counts = ", ".join(f"{k} {v}" for k, v in manifest["counts"].items())
    print(f"{args.skill}: dataset-unfiltered with {manifest['examples']} examples ({counts}); "
          f"train it with te train --dataset dataset-unfiltered")
    return 0


def cmd_clips(args) -> int:
    from backend.cli import load_settings, shown_path
    from backend.dataengine.realclips import DATASET_DIR, FPS, write_clip_dataset
    from backend.skills.spec import check_name

    settings = load_settings()
    try:
        name = check_name(args.skill)
        clips_dir = Path(args.clips) if args.clips else settings.data_dir / "clips" / name
        print(f"{name}: reading the session 1 and 2 clips in {shown_path(clips_dir)} at "
              f"{FPS:g} fps", flush=True)
        manifest = write_clip_dataset(settings.data_dir / "skills" / name, clips_dir,
                                      system_in_user=settings.system_in_user)
    except (OSError, ValueError, RuntimeError) as err:  # RuntimeError: no ffmpeg
        print(f"te: {err}", file=sys.stderr)
        return 2
    for c in manifest["clips"]:
        print(f"  {c['clip']} ({c['split']}): {c['examples']} of {c['frames']} frames, "
              f"{c['yes']} yes and {c['no']} no; {c['near_a_change']} near a change left out")
    for skipped in manifest["skipped"]:
        print(f"  left out {skipped['clip']}: {skipped['reason']}")
    train, val = manifest["splits"]["train"], manifest["splits"]["val"]
    out = settings.data_dir / "skills" / name / DATASET_DIR
    print(f"{name}: {DATASET_DIR} with {manifest['examples']} examples ({train} from session 1 "
          f"to train, {val} from session 2 to validate) -> {shown_path(out)}")
    print("te train --variant real or mixed adds them; session 2 frames are never trained on")
    if not val:
        print("note: no session 2 frames, so the clips hold nothing out for validation")
    if not train:
        print("note: no session 1 frames, so the clips add nothing to train on")
    return 0


def _rate(part: dict) -> str:
    if not part["n"]:
        return "none judged"
    low, high = part["ci95"]
    return f"{part['errors']} of {part['n']} ({part['rate']:.1%}, 95% CI {low:.1%} to {high:.1%})"


def cmd_spark_errors(args) -> int:
    from backend.cli import load_settings, shown_path
    from backend.eval.spark_errors import spark_error_rate
    from backend.skills.spec import check_name

    settings = load_settings()
    try:
        folder = settings.data_dir / "skills" / check_name(args.skill)
        human = Path(args.human) if args.human else folder / "human_labels.json"
        result = spark_error_rate(folder / "labels.jsonl", human)
    except (OSError, ValueError, TypeError, KeyError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    out = folder / "spark_errors.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{args.skill}: {result['hand_labelled']} hand-labelled images")
    print(f"  Spark wrong: {_rate(result['spark'])}; unsure on {result['spark']['unsure']}")
    print(f"  generated-for label wrong (unfiltered arm): {_rate(result['intended'])}")
    print(f"  kept by the filter but wrong (filtered arm): {_rate(result['kept'])}")
    print(f"  left out: {result['human_unsure']} you marked unsure, {result['spark_failed']} "
          f"Spark could not label, {result['unmatched']} not in labels.jsonl")
    print(f"saved: {shown_path(out)}")
    return 0


def cmd_regress(args) -> int:
    from backend.cli import load_settings, shown_path
    from backend.config import PROJECT_ROOT
    from backend.eval.regression import Endpoint, load_questions, run_regression
    from backend.inference.student import REASONING

    settings = load_settings()
    key = settings.secrets.get("STUDENT_API_KEY")
    try:
        questions = load_questions(Path(args.questions))
    except (OSError, ValueError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    # The same sampling as the student scorer: the checkpoint's published settings (no
    # temperature) and short reasoning, for both models.
    sides = [Endpoint(url, model, api_key=key, max_tokens=args.max_tokens, temperature=None,
                      extra_body=REASONING)
             for url, model in ((args.base_url, args.base_model), (args.ft_url, args.ft_model))]
    report = run_regression(questions, *sides, precision=args.precision)
    out = PROJECT_ROOT / "eval"
    out.mkdir(exist_ok=True)
    (out / "regression.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "regression.md").write_text(report["markdown"] + "\n")
    print(report["markdown"])
    print(f"saved: {shown_path(out / 'regression.md')}")
    return 1 if report["base"]["failed"] or report["finetuned"]["failed"] else 0


def add_parsers(sub) -> None:
    exp = sub.add_parser("experiments", help="print the training runs of Experiments 1 to 3")
    exp.add_argument("--skills", required=True, help="comma-separated, e.g. washer_done")
    exp.add_argument("--write-plan", action="store_true", help="also write eval/plan.json")
    exp.add_argument("--force", action="store_true", help="replace an existing eval/plan.json")
    exp.add_argument("--endpoint-hours", type=_hours, default=None,
                     help="hours of serving to price per evaluated model (default: "
                     "backend/eval/plan.py's DEFAULT_ENDPOINT_HOURS)")
    exp.set_defaults(func=cmd_experiments)

    rep = sub.add_parser("replay", help="replay a run's pending clips into eval/plan.json's logs")
    rep.add_argument("--plan", default="eval/plan.json", help="the evaluation plan")
    rep.add_argument("--run", required=True, help="whose clips: a run of the plan, e.g. "
                     "exp1-mixed (the model TE_STUDENT_URL serves), exp4-base or exp4-spark")
    rep.add_argument("--scorer", choices=["student", "spark"], help="who scores the frames "
                     "(default: what the plan says for the run; spark is paid and asks)")
    rep.add_argument("--model", help="the served model name (default TE_STUDENT_MODEL), "
                     "e.g. muse-glimmer for exp4-base, or the run's name with te "
                     "serve-student --runs")
    rep.add_argument("--fps", type=_per_second, default=1.0, help="frames scored per second "
                     "(default 1, as te watch)")
    rep.add_argument("--threshold", type=_fraction, help="one threshold for every skill, "
                     "instead of the run's calibrated one")
    rep.add_argument("--max-tokens", type=_positive, help="the student's reply budget "
                     "(default TE_STUDENT_MAX_TOKENS)")
    rep.add_argument("--no-cloud", action="store_true", help="skip the Muse Spark check on "
                     "test clip alerts")
    rep.add_argument("--warm-up-s", type=_seconds, default=None, help="how long to wait for "
                     "a cold endpoint to answer (default 900; 0 skips the warm-up)")
    rep.add_argument("--yes", action="store_true", help="with the spark scorer: no question "
                     "(not over $15)")
    rep.set_defaults(func=cmd_replay)

    unf = sub.add_parser("unfiltered", help="write Experiment 2's arm without the Spark filter")
    unf.add_argument("--skill", required=True)
    unf.set_defaults(func=cmd_unfiltered)

    clips = sub.add_parser("clips", help="labelled session 1 and 2 clips as real examples")
    clips.add_argument("--skill", required=True)
    clips.add_argument("--clips", help="the clip folder (default data/clips/<skill>)")
    clips.set_defaults(func=cmd_clips)

    err = sub.add_parser("spark-errors", help="Spark's error rate against your image labels")
    err.add_argument("--skill", required=True)
    err.add_argument("--human", help="labels from te label --skill (default: "
                     "data/skills/<skill>/human_labels.json)")
    err.set_defaults(func=cmd_spark_errors)

    reg = sub.add_parser("regress", help="base vs fine-tuned on general questions")
    reg.add_argument("--questions", required=True, help="JSONL: id, image, question, answer, "
                     "match (backend/eval/regression.py)")
    reg.add_argument("--base-url", required=True)
    reg.add_argument("--base-model", required=True)
    reg.add_argument("--ft-url", required=True)
    reg.add_argument("--ft-model", default="trained-eye")
    reg.add_argument("--precision", required=True, help="e.g. bf16 or nf4, the same for both")
    reg.add_argument("--max-tokens", type=int, default=2048)
    reg.set_defaults(func=cmd_regress)
