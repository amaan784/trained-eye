"""te commands for M3's experiments (registered by backend/cli.py).

  te experiments    print every training run of Experiments 1 to 3 with its price
  te unfiltered     write the Experiment 2 arm without the Spark filter (free)
  te clips          labelled session 1 and 2 clips as real examples, dataset-clips (free)
  te spark-errors   Spark's error rate against the author's own image labels (free)
  te regress        the regression check: base vs fine-tuned on general questions

te replay, which replays a run's pending clips into the plan's logs, is in cli_replay.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _hours(text: str) -> float:
    value = float(text)
    if not value > 0 or value == float("inf"):
        raise argparse.ArgumentTypeError(f"must be a number of hours over 0, got {text}")
    return value


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
        progress,
        training_runs,
    )
    from backend.eval.plan_file import plan_skeleton, write_plan
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
    exp.add_argument("--skills", required=True, help="comma-separated, e.g. burner_left_on")
    exp.add_argument("--write-plan", action="store_true", help="also write eval/plan.json")
    exp.add_argument("--force", action="store_true", help="replace an existing eval/plan.json")
    exp.add_argument("--endpoint-hours", type=_hours, default=None,
                     help="hours of serving to price per evaluated model (default: "
                     "backend/eval/plan.py's DEFAULT_ENDPOINT_HOURS)")
    exp.set_defaults(func=cmd_experiments)

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
