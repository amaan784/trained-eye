"""te train (registered by backend/cli.py): bundle taught skills and train the student on
Modal, with the two-step smoke, the processor preflight (--preflight), the real run priced
from the smoke's timing, --detach and --collect, the W&B tags and artifacts, and the GPU
time recorded in the Modal ledger.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from backend import cli
from backend.cli import (
    ASK_ABOVE_USD,
    MODAL_GPU_USD_PER_HOUR,
    PROJECT_ROOT,
    count,
    load_settings,
    shown_path,
)
from backend.config import STUDENT_BASE_REVISION
from backend.training import gpu_ledger, runs
from backend.training.runs import RUN_NAME

GPU = "H100"


def train_estimate(examples: int, timing: dict | None) -> dict | None:
    """H100 time and dollars for one epoch over `examples`, from a smoke run: its setup
    (container start, base model load) plus examples at its measured samples per second.
    None without a usable smoke timing."""
    rate = (timing or {}).get("train_samples_per_second")
    if not rate:
        return None
    setup = max(0.0, timing["seconds"] - timing.get("train_runtime", 0.0))
    seconds = setup + examples / rate
    return {"seconds": seconds, "usd": seconds / 3600 * MODAL_GPU_USD_PER_HOUR[GPU]}


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                         capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


TRAIN_VOLUME = "trained-eye"  # backend/training/modal_train.py's VOLUME
# modal_train.base_marker(): written on the volume once the pinned base is downloaded
BASE_MARKER = f"base-{STUDENT_BASE_REVISION}.json"


def base_on_volume() -> bool:
    """Whether download_base() finished for the pinned revision (its marker file)."""
    with tempfile.TemporaryDirectory() as tmp:
        code = subprocess.run(["modal", "volume", "get", TRAIN_VOLUME, f"/hf/{BASE_MARKER}",
                               tmp], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              check=False).returncode
        return code == 0 and any(Path(tmp).rglob(BASE_MARKER))


def ensure_base() -> int:
    """Before a GPU run: the 30B base on the volume, downloaded by a CPU job if missing,
    so the H100 is not billed while it downloads."""
    if base_on_volume():
        return 0
    print("the 30B base is not on the volume yet: a CPU job on Modal downloads it first "
          "(about 60 GB), so the H100 does not wait for it")
    code = subprocess.run(["modal", "run", "backend/training/modal_train.py", "--fetch-base"],
                          cwd=PROJECT_ROOT, check=False).returncode
    if code != 0 or not base_on_volume():
        print(f"te: the base model download failed (exit {code}); nothing was trained",
              file=sys.stderr)
        return code or 1
    return 0


def read_result(path: Path) -> dict | None:
    """A Modal job's local result file, or None (missing or half written)."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def record_train_spend(settings, result: dict) -> None:
    """Add a training result's container seconds to the Modal ledger (once per run end)."""
    seconds = result.get("seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return
    run = str(result.get("run") or "?")
    key = f"train:{run}:{result['finished']}" if result.get("finished") else None
    gpu_ledger.record(settings.data_dir, "train", run, seconds, gpu=GPU, key=key,
                      upper_bound=bool(result.get("seconds_upper_bound")))


def finish_train(settings, run: str, result: dict, smoke: bool) -> int:
    """What te train prints after a run, also after te train --collect."""
    if result.get("ok") is False:
        print(f"te: training failed; the full log is on the Modal volume under "
              f"runs/{run}/train.log", file=sys.stderr)
        return 1
    if smoke and result.get("train_samples_per_second"):
        timing_path = settings.data_dir / "train" / "smoke-timing.json"
        timing_path.write_text(json.dumps(result, indent=1) + "\n")
        print(f"smoke timing saved: {shown_path(timing_path)}")
    print(f"done: {result['seconds']:.0f} s in the training container; the adapter is on the "
          f"volume under runs/{run}/out")
    return 0


def collect_train(settings, run: str) -> int:
    """te train --collect: fetch a run's result.json from the Modal volume (a run launched
    with --detach, whose terminal closed), then save and print it as te train does."""
    try:
        runs.check_run(run)
    except ValueError as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    if not cli.modal_ready():
        return 2
    train_dir = settings.data_dir / "train"
    fetched, result = train_dir / f".collect-{run}", None
    shutil.rmtree(fetched, ignore_errors=True)
    fetched.mkdir(parents=True)
    try:
        code = subprocess.run(["modal", "volume", "get", TRAIN_VOLUME,
                               f"/runs/{run}/result.json", str(fetched)],
                              cwd=PROJECT_ROOT, check=False).returncode
        found = next((f for f in fetched.rglob("result.json") if f.is_file()), None)
        result = json.loads(found.read_text()) if code == 0 and found else None
    except (OSError, ValueError):
        result = None
    finally:
        shutil.rmtree(fetched, ignore_errors=True)
    if not isinstance(result, dict) or result.get("run") != run:
        print(f"te: no result for {run} on the Modal volume yet: the run may still be going "
              f"(modal app list shows it), or it stopped before writing one (its log is "
              f"runs/{run}/train.log on the volume)", file=sys.stderr)
        return 1
    status = runs.read_status(settings.data_dir, run) or {}
    started = status.get("started")
    if started and str(result.get("finished") or "") < started:
        print(f"te: the result on the volume is from an earlier run of {run} (finished "
              f"{result.get('finished')}; te train started it again at {started}): that run "
              "has not finished yet", file=sys.stderr)
        return 1
    result_path = train_dir / f"{run}.result.json"
    result_path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"collected: {shown_path(result_path)}")
    smoke = bool(result.get("smoke"))
    runs.write_status(settings.data_dir, run, "failed" if result.get("ok") is False else "done",
                      status.get("skills") or runs.run_skills(settings.data_dir, run),
                      started=started or result.get("started") or runs.now_iso(),
                      finished=result.get("finished") or runs.now_iso(), smoke=smoke)
    record_train_spend(settings, result)
    return finish_train(settings, run, result, smoke)


def preflight_train(settings, args, bundle_dir: Path) -> int:
    """te train --preflight: render a few bundle examples with the pinned processor on
    Modal (backend/training/preflight.py), before the paid smoke."""
    from backend.training.preflight import format_report

    path = bundle_dir.parent / f"{args.run}.preflight.json"
    print("processor preflight: a CPU job on Modal (no GPU) loads the pinned processor, no "
          "weights, and renders a few examples of this bundle in both prompt layouts. The "
          "first one also builds the training image (flash-attn compiles), which the smoke "
          "then reuses.")
    if args.estimate_only:
        return 0
    if not args.yes and not cli.ask("Run the preflight on Modal?"):
        return 1
    if not cli.modal_ready():
        return 2
    path.unlink(missing_ok=True)
    command = ["modal", "run", "backend/training/modal_train.py", "--bundle", str(bundle_dir),
               "--run", args.run, "--preflight", "--result", str(path)]
    code = subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
    if code != 0 or not path.is_file():
        print(f"te: the preflight failed (exit {code})", file=sys.stderr)
        return code or 1
    report = json.loads(path.read_text())
    print(format_report(report))
    print(f"saved: {shown_path(path)}")
    return 0 if report.get("ok") else 1


def ask_artifacts(settings, args, manifest: dict, bundle_dir: Path) -> bool:
    """te train --wandb-artifacts: say what becomes visible and ask, even with --yes. A yes
    copies the skills' dataset manifests (the provenance index of every image) into the
    bundle, so they go along."""
    from backend.training.axolotl import WANDB_PROJECT, data_hash

    where = (f"team {args.wandb_entity}" if args.wandb_entity else "your own W&B account")
    counts = manifest.get("counts", {})
    images = sum(counts.values())
    synthetic = sum(n for key, n in counts.items() if key.split("/")[2:3] == ["synthetic"])
    print(f"--wandb-artifacts uploads to W&B project {WANDB_PROJECT} ({where}): this bundle as "
          f"the dataset artifact {args.run}-{data_hash(manifest)[:12]} ({images} images with "
          f"their prompts and yes/no labels; {synthetic} are Muse Image outputs, padded copies "
          "whose provenance index goes along in provenance/) and, at the end of training, the "
          "adapter weights. Everyone who can see the project sees them, and everyone once it "
          "is public. Faces in these images are not blurred.")
    if not cli.ask("Upload them to W&B? (asked even with --yes)"):
        print("te: nothing uploaded or trained; without --wandb-artifacts no data or weights "
              "go to W&B", file=sys.stderr)
        return False
    target = bundle_dir / "provenance"
    target.mkdir(exist_ok=True)
    for skill in manifest.get("skills", {}):
        name = f"{skill}-teach.manifest.json"
        shutil.copyfile(settings.data_dir / "skills" / skill / args.dataset / name,
                        target / name)
    return True


def cmd_train(args) -> int:
    from backend.skills.spec import check_name
    from backend.training.axolotl import optimizer_steps, wandb_tags
    from backend.training.prepare import bundle

    settings = load_settings()
    if args.collect:
        return collect_train(settings, args.collect)
    try:
        if not args.skills or not args.run:
            raise ValueError("pass --skills and --run (or --collect <run>)")
        names = [check_name(n.strip()) for n in args.skills.split(",") if n.strip()]
        if not names:
            raise ValueError("--skills needs at least one skill name")
        if not RUN_NAME.fullmatch(args.run):
            raise ValueError(f"run name {args.run!r}: lower-case letters, digits and -")
        out = settings.data_dir / "train" / args.run
        manifest = bundle([settings.data_dir / "skills" / n for n in names], out,
                          f"/vol/bundles/{args.run}", variant=args.variant, limit=args.limit,
                          dataset_dir=args.dataset)
    except ValueError as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    train_n, val_n = manifest["train_examples"], manifest.get("val_examples", 0)
    print(f"bundle {args.run}: {train_n} training and {val_n} validation examples "
          f"({args.variant}) -> {shown_path(out)}")
    if manifest["system_in_user"] != settings.system_in_user:
        print(f"note: these datasets were written with TE_SYSTEM_IN_USER="
              f"{int(manifest['system_in_user'])}, not {int(settings.system_in_user)} as now; "
              + ("run te unfiltered --skill <skill> again" if args.dataset == "dataset-unfiltered"
                 else "run te teach again") + " to rewrite them the new way")
    if args.preflight:
        return preflight_train(settings, args, out)
    timing_path = settings.data_dir / "train" / "smoke-timing.json"
    estimate = None
    if args.smoke:
        print(f"two-step smoke on one {GPU} on Modal. The first one may also build the image "
              "(flash-attn compiles; the preflight builds it too), so its time is only known "
              "once it has run. The 30B base is downloaded before it by a CPU job.")
        question = "Run the smoke on Modal?"
    else:
        timing = json.loads(timing_path.read_text()) if timing_path.is_file() else None
        from backend.eval.plan import steps_samples

        samples = (steps_samples(args.max_steps, train_n, val_n) if args.max_steps
                   else train_n + val_n)
        estimate = train_estimate(samples, timing)
        if estimate is None:
            print("te: no smoke timing yet: run the same command with --smoke first, so this "
                  "run can be priced", file=sys.stderr)
            return 2
        print(f"{args.max_steps or optimizer_steps(train_n)} optimizer steps; estimate: about "
              f"{estimate['seconds'] / 60:.0f} min of {GPU}, about ${estimate['usd']:.2f} at "
              f"${MODAL_GPU_USD_PER_HOUR[GPU]:.2f} an hour (price not checked; see config.py)")
        question = f"Train {args.run} on Modal for about ${estimate['usd']:.2f}?"
    if args.estimate_only:
        return 0
    over = estimate is not None and estimate["usd"] > ASK_ABOVE_USD
    if over:
        print(f"te: this is over ${ASK_ABOVE_USD:.0f}, so it needs your yes even with --yes")
    credit = gpu_ledger.credit_check(settings.data_dir, estimate["usd"] if estimate else 0.0)
    if credit:
        print(f"te: {credit}")
    if args.wandb_artifacts and not ask_artifacts(settings, args, manifest, out):
        return 1
    if (over or credit or not args.yes) and not cli.ask(question):
        return 1
    if shutil.which("modal") is None:
        print('te: the modal command is missing: pip install -e ".[train]", then modal token '
              "set", file=sys.stderr)
        return 2
    fetched = ensure_base()
    if fetched:
        return fetched
    result_path = out.parent / f"{args.run}.result.json"
    result_path.unlink(missing_ok=True)
    tags = [f"exp:{args.run.split('-')[0]}", f"skills:{'+'.join(names)}",
            f"mix:{args.variant}", f"examples:{train_n}", "seed:42", f"commit:{git_commit()}",
            *wandb_tags(manifest)]  # data:<hash>, base:<revision>, dataset:, images:
    if args.max_steps and not args.smoke:
        tags.append(f"steps:{args.max_steps}")
    command = ["modal", "run", *(["--detach"] if args.detach else []),
               "backend/training/modal_train.py", "--bundle", str(out),
               "--run", args.run, "--tags", ",".join(tags), "--result", str(result_path)]
    if args.wandb_entity:
        command += ["--wandb-entity", args.wandb_entity]
    if args.wandb_artifacts:
        command.append("--wandb-artifacts")
    if args.detach:  # the entrypoint then leaves the result to te train --collect
        command.append("--detached")
    if args.smoke:
        command.append("--smoke")
    elif args.max_steps:
        command += ["--max-steps", str(args.max_steps)]
    skills, started, code = sorted(manifest["skills"]), runs.now_iso(), None
    runs.write_status(settings.data_dir, args.run, "running", skills, started=started,
                      smoke=args.smoke, detached=args.detach)  # GET /skills: training
    if args.detach:
        print(f"detached: if this terminal closes or the Mac sleeps, the run goes on in Modal; "
              f"te train --collect {args.run} then fetches its result")
    try:
        code = subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
    finally:  # also when interrupted: then the run is marked failed, unless it is detached
        if not (args.detach and not result_path.is_file()):
            runs.write_status(settings.data_dir, args.run,
                              "done" if code == 0 and result_path.is_file() else "failed",
                              skills, started=started, finished=runs.now_iso(),
                              smoke=args.smoke)
        elif code is None:
            print(f"\nte: stopped here; the detached run goes on: te train --collect "
                  f"{args.run} fetches its result when it is done", file=sys.stderr)
        result = read_result(result_path)
        if result is not None:  # also a failed or interrupted run: its GPU time was billed
            record_train_spend(settings, {"run": args.run, **result})
    if result is None and args.detach:
        print(f"te: the Modal client stopped (exit {code}); the detached run may still be "
              f"going: te train --collect {args.run} fetches its result when it is done",
              file=sys.stderr)
        return code or 1
    if code != 0 or result is None:
        print(f"te: training failed (exit {code}); the full log is on the Modal volume under "
              f"runs/{args.run}/train.log", file=sys.stderr)
        return code or 1
    return finish_train(settings, args.run, result, args.smoke)


def add_parsers(sub) -> None:
    train = sub.add_parser("train", help="train the student on Modal from taught skills")
    train.add_argument("--skills", help="comma-separated, e.g. burner_left_on (needed but with "
                       "--collect)")
    train.add_argument("--run", help="run name, e.g. exp1-mixed (needed but with --collect)")
    train.add_argument("--variant", choices=["mixed", "real", "synthetic"], default="mixed")
    train.add_argument("--limit", type=int, help="at most this many examples per skill")
    train.add_argument("--max-steps", type=count, help="fixed optimizer steps for Experiments 1 "
                       "to 3 (backend/eval/plan.py)")
    train.add_argument("--dataset", choices=["dataset", "dataset-unfiltered"], default="dataset",
                       help="the skill's dataset folder (Experiment 2: dataset-unfiltered)")
    train.add_argument("--smoke", action="store_true", help="the recipe's two-step check")
    train.add_argument("--preflight", action="store_true", help="instead of training: render "
                       "a few examples with the pinned processor on Modal (CPU): lengths, the "
                       "system turn, the yes/no target")
    train.add_argument("--detach", action="store_true", help="modal run --detach: the run "
                       "goes on if this Mac sleeps or disconnects (then te train --collect)")
    train.add_argument("--collect", metavar="RUN", help="fetch a detached run's result from "
                       "the Modal volume and print it as te train does")
    train.add_argument("--wandb-entity", help="the W&B team for the run (default: your own "
                       "account)")
    train.add_argument("--wandb-artifacts", action="store_true", help="also upload the bundle "
                       "and the adapter as W&B Artifacts (asks first, even with --yes)")
    train.add_argument("--estimate-only", action="store_true", help="bundle and price only")
    train.add_argument("--yes", action="store_true", help="no question (not over $15)")
    train.set_defaults(func=cmd_train)
