"""te replay (registered by backend/cli.py): replay a run's pending clips of eval/plan.json
into the plan's logs. The run's threshold is calibrated on its own validation clips first,
then its test clips are scored at it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


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


def _spec_hold(settings, skill: str) -> float | None:
    """The hold the skill's spec has now (data/skills/<skill>/spec.json), None if unreadable."""
    from backend.skills.spec import SkillSpec

    try:
        path = settings.data_dir / "skills" / skill / "spec.json"
        return float(SkillSpec.model_validate_json(path.read_text(encoding="utf-8")).hold_s)
    except (OSError, ValueError):
        return None


def _stale(jobs, calibrates, settings, run: str, held: dict) -> dict:
    """Test logs scored at another threshold than the run's calibrated one (it was
    calibrated again, or the log came from te watch), or fired with another hold than the
    skill's spec has now: replayed again only by hand."""
    from backend.eval.metrics import hold_of
    from backend.watch.watcher import read_threshold, run_threshold_file

    out = {}
    for job in jobs:
        if job.kind != "test" or job.skill not in calibrates or not job.log.is_file() or (
                job.log in held):
            continue
        try:
            now = read_threshold(run_threshold_file(settings.data_dir, run, job.skill))
            rows = [json.loads(line) for line in job.log.read_text(
                encoding="utf-8").splitlines() if line.strip()]
            used = {round(float(r["threshold"]), 4) for r in rows if r.get("skill") == job.skill}
            hold_used = hold_of(rows, job.skill)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        hold_now = _spec_hold(settings, job.skill)
        if hold_now is not None and used and hold_used != hold_now:
            out[job.log] = (f"fired with a {hold_used:g} s hold, but {job.skill}'s spec has "
                            f"{hold_now:g} s now: delete this log and run te replay again")
        elif now is not None and used and used != {round(now, 4)}:
            out[job.log] = (f"scored at {', '.join(f'{u:.3f}' for u in sorted(used))}, but "
                            f"{run}'s threshold for {job.skill} is {now:.3f} now: delete this "
                            "log and run te replay again to score it at that")
    return out


def serving_note(data_dir: Path, run: str, model: str) -> str | None:
    """A warning when te serve-student last deployed another run's merged model and the
    model asked for is not named after this run (te serve-student --runs) or the base."""
    from backend.cli_serving import BASE_MODEL_NAME
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

    from backend.cli import load_settings, make_client, shown_path
    from backend.cli_watch import cmd_calibrate
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
    uncalibrated: set[str] = set()
    for cal in [c for c in plan.calibration if c.run == run and c.skill in calibrates]:
        logs = [(base / c.log).resolve() for c in cal.clips]
        labels = [(base / c.labels).resolve() for c in cal.clips]
        target = run_threshold_file(settings.data_dir, run, cal.skill)
        if not all(p.is_file() for p in logs + labels):
            continue
        if target.is_file() and not (any(log in replayed for log in logs)
                                     or _calibration_changed(
                                         target, logs, labels, _spec_hold(settings, cal.skill))):
            continue
        if cmd_calibrate(Namespace(skill=cal.skill, run=run, log=[str(p) for p in logs],
                                   labels=[str(p) for p in labels])) != 0:
            uncalibrated.add(cal.skill)  # an older threshold must not score the test clips
    for job in [j for j in todo if j.kind == "test" and j.skill in uncalibrated]:
        held[job.log] = (f"te calibrate --run {run} failed for {job.skill} (see above): fix "
                         "the validation labels, then te replay again")
    todo = [j for j in todo if j.log not in held]
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
    return 1 if failed or uncalibrated else 0


def _calibration_changed(target: Path, logs: list[Path], labels: list[Path],
                         hold_s: float | None = None) -> bool:
    """Whether a run threshold was computed on other validation clips than the plan's now,
    before one of their logs or labels changed (labels added after a replay), or with
    another hold than the skill's spec has now (`hold_s`, when known)."""
    try:
        saved = json.loads(target.read_text(encoding="utf-8"))
        used = saved.get("clips") or []
        recorded = {(c.get("log"), c.get("labels")) for c in used}
        hold_then = saved.get("hold_s")
    except (OSError, ValueError, AttributeError):
        return True
    if hold_s is not None and (hold_then is None or float(hold_then) != hold_s):
        return True
    if recorded != {(log.name, lab.name) for log, lab in zip(logs, labels, strict=True)}:
        return True
    made = target.stat().st_mtime
    return any(p.stat().st_mtime > made for p in [*logs, *labels])


def add_parsers(sub) -> None:
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
