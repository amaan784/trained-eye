"""te watch and te calibrate (registered by backend/cli.py): replay a clip through watch
mode, scored by the trained model or by Muse Spark, and set a skill's threshold from the
frame logs of replayed, labelled validation clips.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from backend import cli
from backend.cli import ASK_ABOVE_USD, fraction, load_settings, positive, shown_path
from backend.training import runs


def cmd_watch(args) -> int:
    from backend.dataengine.keyframes import load_frames
    from backend.dataengine.labels import label_image
    from backend.inference.student import OpenAIStudent
    from backend.skills.spec import check_name
    from backend.watch.watcher import FrameLog, Watcher, load_skill, system_in_user

    settings = load_settings()
    if args.scorer == "student" and not settings.student_url:
        print("te: watch needs the trained model's endpoint: set TE_STUDENT_URL (or use "
              "--scorer spark)", file=sys.stderr)
        return 2
    bundle, folded = None, False
    try:
        if args.run is not None:
            runs.check_run(args.run)
            bundle = runs.bundle_manifest(settings.data_dir, args.run)
        names = [check_name(n.strip()) for n in args.skills.split(",") if n.strip()]
        skills, folders = [], []
        for name in names:
            folder = settings.data_dir / "skills" / name
            if not (folder / "spec.json").is_file():
                raise ValueError(f"no taught skill named {name!r}")
            skills.append(load_skill(folder, run=args.run, threshold=args.threshold))
            folders.append(folder)
        if args.scorer == "student":  # ask the student the way it was trained
            folded = system_in_user(folders, bundle=bundle, default=settings.system_in_user)
        frames = load_frames(args.clip, fps=args.fps)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    for skill in skills if args.scorer == "student" else ():
        name, threshold = skill.spec.name, skill.trigger.threshold
        if skill.threshold_source == "default":
            print(f"note: {name} has no calibrated threshold yet; using {threshold}")
        elif args.run and skill.threshold_source == "skill":
            print(f"note: {name} has no threshold for run {args.run} yet (te calibrate --run "
                  f"{args.run}); using the skill-wide {threshold:.3f}")
        if args.threshold is None and (note := served_run_note(settings, name, args.run)):
            print(note)
    if bundle is not None and (untrained := sorted(set(names) - set(bundle.get("skills", {})))):
        print(f"note: run {args.run} was not trained on {', '.join(untrained)}")
    if folded:
        print("the student is asked with the system prompt in the user text, as its training "
              "data was written")
    client = cli.make_client(settings)
    confirm = None if args.no_cloud else (lambda image, spec: label_image(client, spec, image))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.log:  # a rerun replays every frame (cached ones are free): start the log afresh
        Path(args.log).unlink(missing_ok=True)
    log = FrameLog(Path(args.log) if args.log else
                   settings.data_dir / "watch" / f"replay-{stamp}-{Path(args.clip).stem}.jsonl",
                   {"run": args.run} if args.run else None)
    if args.scorer == "spark":  # Experiment 4: Muse Spark judges every frame, paid per frame
        from backend.eval.plan import spark_usd_per_check
        from backend.eval.spark_scorer import SparkScorer, default_log

        settings.secret("MODEL_API_KEY")
        frames = list(frames)
        cost = len(frames) * len(skills) * spark_usd_per_check(settings)
        print(f"Muse Spark on {len(frames)} frames x {len(skills)} skills: about ${cost:.2f} "
              "before cache hits")
        over = cost > ASK_ABOVE_USD
        if (over or not args.yes) and not cli.ask("Spend that?"):
            return 1
        student = SparkScorer(client, latency_log=default_log(settings.data_dir))
        for skill in skills:  # Spark answers yes, no or unsure (0.5): only a yes counts
            if args.threshold is None:
                skill.trigger.threshold, skill.threshold_source = 1.0, "spark"
    else:
        student = OpenAIStudent(settings.student_url, settings.student_model,
                                api_key=settings.secrets.get("STUDENT_API_KEY"),
                                max_tokens=args.max_tokens or settings.student_max_tokens,
                                system_in_user=folded)
    watcher = Watcher(skills, student, confirm=confirm, log=log)
    count = 0
    for frame in frames:
        _, alerts = watcher.on_frame(frame.data, t=frame.t_s)
        count += 1
        for alert in alerts:
            cloud = {True: "confirmed", None: "not checked"}.get(alert.confirmed, "no")
            print(f"{alert.t:8.1f} s  {alert.skill}: \"{alert.text}\"  "
                  f"(p {alert.p:.2f}, cloud {cloud})")
    print(f"{count} frames scored at {args.fps:g} fps; frame log: {shown_path(log.path)}")
    if args.scorer == "spark":
        print(f"Spark: {json.dumps(student.summary())}")
    return 0


def served_run_note(settings, skill: str, run: str | None) -> str | None:
    """Without --run: a note when te serve scores this skill at the served run's own
    threshold, which a skill-wide calibrate or replay does not change."""
    from backend.training import runs
    from backend.watch.watcher import run_threshold_file

    served = runs.serving(settings.data_dir)
    if run or served is None or skill not in served["skills"]:
        return None
    if not run_threshold_file(settings.data_dir, served["run"], skill).is_file():
        return None
    return (f"note: te serve scores {skill} with run {served['run']}'s own threshold, not "
            f"the skill-wide one: pass --run {served['run']} to change or test what it uses")


def cmd_calibrate(args) -> int:
    from backend.skills.spec import SkillSpec, check_name
    from backend.watch.trigger import calibrate, labelled_frames
    from backend.watch.watcher import run_threshold_file

    settings = load_settings()
    try:
        check_name(args.skill)
        folder = settings.data_dir / "skills" / args.skill
        target = (run_threshold_file(settings.data_dir, args.run, args.skill) if args.run
                  else folder / "threshold.json")
        spec = SkillSpec.model_validate_json((folder / "spec.json").read_text())
        if len(args.log) != len(args.labels):
            raise ValueError("give one --labels file for each --log")
        clips, sources = [], []
        for log, labels in zip(args.log, args.labels, strict=True):
            rows = [json.loads(line) for line in Path(log).read_text().splitlines() if line]
            replayed = sorted({r["run"] for r in rows
                               if r.get("skill") == args.skill and r.get("run")})
            if args.run and set(replayed) - {args.run}:
                raise ValueError(f"{log} was replayed for run {', '.join(replayed)}, not "
                                 f"{args.run}")
            if replayed and not args.run:
                print(f"note: {log} was replayed for run {', '.join(replayed)}; te calibrate "
                      "--run keeps that run's threshold apart from the skill-wide one")
            ranges = json.loads(Path(labels).read_text())["true"]
            frames = labelled_frames(rows, args.skill, ranges)
            if not frames:
                raise ValueError(f"{log} has no scored frames for {args.skill}")
            clips.append(frames)
            sources.append({"log": Path(log).name, "labels": Path(labels).name,
                            "frames": len(frames)})
        result = calibrate(clips, float(spec.cooldown_s), hold_s=float(spec.hold_s))
    except (OSError, ValueError, KeyError, TypeError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    minutes = result["hours"] * 60
    if result.get("never_fires"):
        print(f"te: no threshold keeps false alarms under one per hour on these "
              f"{minutes:.0f} minutes: the student needs more training", file=sys.stderr)
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({**result, **({"run": args.run} if args.run else {}),
                                  "clips": sources}, indent=1) + "\n")
    delay = result["median_delay_s"]
    for_run = f" for run {args.run}" if args.run else ""
    print(f"{args.skill}{for_run}: threshold {result['threshold']:.3f} on {minutes:.0f} minutes of "
          f"footage: {result['false_alarms']} false alarms, {result['caught']} of "
          f"{result['events']} events caught"
          + (f", median delay {delay:.0f} s" if delay is not None else "")
          + (f" (after the {spec.hold_s} s hold)" if spec.hold_s and delay is not None else ""))
    print(f"saved: {shown_path(target)}")
    if note := served_run_note(settings, args.skill, args.run):
        print(note)
    if minutes < 60:
        print("note: under an hour of footage, so even one false alarm is too many here")
    return 0


def add_parsers(sub) -> None:
    watch = sub.add_parser("watch", help="replay a clip through watch mode")
    watch.add_argument("--skills", required=True, help="comma-separated, e.g. burner_left_on")
    watch.add_argument("--clip", required=True, help="video file or folder of frames")
    watch.add_argument("--fps", type=float, default=1.0, help="frames scored per second")
    watch.add_argument("--no-cloud", action="store_true", help="skip the Muse Spark check")
    watch.add_argument("--scorer", choices=["student", "spark"], default="student",
                       help="who scores each frame: the trained model, or Muse Spark (paid)")
    watch.add_argument("--log", help="frame log path (the evaluation plan names one per clip)")
    watch.add_argument("--run", help="the trained run TE_STUDENT_URL serves, e.g. exp1-mixed: "
                       "its threshold (te calibrate --run) comes before the skill's, and the "
                       "frame log names it")
    watch.add_argument("--threshold", type=fraction, help="one threshold for every skill, "
                       "over the run's and the skill's")
    watch.add_argument("--max-tokens", type=positive, help="the student's reply budget "
                       "(default TE_STUDENT_MAX_TOKENS, 512); base Glimmer may need more")
    watch.add_argument("--yes", action="store_true", help="with --scorer spark: no question "
                       "(not over $15)")
    watch.set_defaults(func=cmd_watch)

    cal = sub.add_parser("calibrate", help="set a skill's threshold from validation clips")
    cal.add_argument("--skill", required=True)
    cal.add_argument("--log", action="append", required=True,
                     help="a frame log from te watch (data/watch/replay-*.jsonl); repeatable")
    cal.add_argument("--labels", action="append", required=True,
                     help='JSON {"true": [[start_s, end_s], ...]} for the clip of that log')
    cal.add_argument("--run", help="the run these logs replayed (te watch --run): writes "
                     "data/watch/<run>/<skill>/threshold.json, not the skill-wide file")
    cal.set_defaults(func=cmd_calibrate)
