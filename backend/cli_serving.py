"""te export and te serve-student (registered by backend/cli.py): merge a trained adapter
on Modal and check it answers as before, and deploy or stop the trained model's endpoint
(or the base model, alone or with the runs' adapters).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

from backend import cli
from backend.cli import (
    ASK_ABOVE_USD,
    MODAL_GPU_USD_PER_HOUR,
    PROJECT_ROOT,
    load_settings,
    positive,
    shown_path,
)
from backend.cli_train import GPU, read_result
from backend.training import gpu_ledger, runs
from backend.training.runs import RUN_NAME

EXPORT_MAX_HOURS = 3  # backend/training/export.py's TIMEOUT_S: the most an export can bill
STUDENT_APP = "trained-eye-student"  # backend/inference/modal_serve.py's APP
BASE_APP = "trained-eye-base"  # modal_serve.py's BASE_APP: the base, and --runs adapters
BASE_MODEL_NAME = "muse-glimmer"  # modal_serve.py's BASE_MODEL_NAME
SERVE_ENVS = ("TE_SERVE_RUN", "TE_SERVE_BASE", "TE_SERVE_LORAS")  # modal_serve.py reads them


def cmd_export(args) -> int:
    settings = load_settings()
    bundle = settings.data_dir / "train" / args.run
    if not RUN_NAME.fullmatch(args.run) or not (bundle / "manifest.json").is_file():
        print(f"te: no training bundle for run {args.run!r} in {shown_path(bundle)}",
              file=sys.stderr)
        return 2
    worst = EXPORT_MAX_HOURS * MODAL_GPU_USD_PER_HOUR[GPU]
    print(f"export {args.run}: merge on one {GPU}, then compare {args.limit} answers "
          f"({args.reference} base + adapter vs merged). Not timed yet: at most "
          f"{EXPORT_MAX_HOURS} h, ${worst:.2f} at ${MODAL_GPU_USD_PER_HOUR[GPU]:.2f} an hour "
          "(price not checked; see config.py)")
    timed = [r for r in gpu_ledger.read(settings.data_dir) if r["kind"] == "export"]
    if timed:
        print(f"the last export took {timed[-1]['seconds'] / 60:.0f} min, about "
              f"${timed[-1]['usd']:.2f} (data/{gpu_ledger.LEDGER_FILE})")
    over = worst > ASK_ABOVE_USD
    credit = gpu_ledger.credit_check(settings.data_dir, worst)
    if credit:
        print(f"te: {credit}")
    if (over or credit or not args.yes) and not cli.ask("Run the export on Modal?"):
        return 1
    if not cli.modal_ready():
        return 2
    result_path = bundle.parent / f"{args.run}.export.json"
    result_path.unlink(missing_ok=True)
    command = ["modal", "run", "backend/training/export.py", "--run", args.run, "--bundle",
               str(bundle), "--limit", str(args.limit), "--reference", args.reference,
               "--result", str(result_path)]
    began, code, result = time.monotonic(), None, None
    try:
        code = subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
    finally:  # also on Ctrl+C: the container was billed while it ran
        result = read_result(result_path)
        seconds = (result or {}).get("seconds")
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
            gpu_ledger.record(settings.data_dir, "export", args.run, seconds, gpu=GPU)
        elif code is None:  # interrupted before a result: count the wall time, at most
            gpu_ledger.record(settings.data_dir, "export", args.run,
                              time.monotonic() - began, gpu=GPU, upper_bound=True)
    if code != 0 or result is None or result.get("ok") is False:
        print(f"te: the export failed (exit {code})", file=sys.stderr)
        return code or 1
    print(f"merged: {result.get('merged_dir')}; the merged model gave the same answer on "
          f"{result['agreement']:.0%} of {result['examples']} examples")
    problems = []
    if result.get("tensor_names_match") is False:
        problems.append("its tensor names differ from the base model's")
    if result.get("bin_files"):
        problems.append(f"it has {result['bin_files']} .bin files instead of safetensors")
    if not result.get("all_agree"):
        problems.append(f"answers differ on {', '.join(result.get('disagreements', [])[:10])}")
    for problem in problems:
        print(f"te: check the merged model: {problem}", file=sys.stderr)
    return 1 if problems else 0


def serve_runs(text: str) -> list[str]:
    """te serve-student --runs a,b,c: the run names, checked."""
    names = [runs.check_run(n.strip()) for n in text.split(",") if n.strip()]
    if not names:
        raise ValueError("--runs needs at least one run name")
    if len(set(names)) != len(names):
        raise ValueError("--runs names a run twice")
    return names


def cmd_serve_student(args) -> int:
    import os

    from backend.config import DEFAULT_MODELS, STUDENT_BASE_REVISION, check_model_allowed

    settings = load_settings()
    base = bool(args.base or args.runs)
    app = BASE_APP if base else STUDENT_APP
    if args.stop:
        if not cli.modal_ready():
            return 2
        code = subprocess.run(["modal", "app", "stop", "--yes", app], cwd=PROJECT_ROOT,
                              check=False).returncode
        if code == 0:
            if not base:
                runs.clear_serving(settings.data_dir)  # GET /skills: no skill is served now
            stopped = gpu_ledger.record_stop(settings.data_dir, app)
            if stopped:
                print(f"{app} was deployed for {stopped['seconds'] / 3600:.1f} h: at most "
                      f"${stopped['usd']:.2f} of {GPU}, billed only while a container was up "
                      f"(recorded in data/{gpu_ledger.LEDGER_FILE})")
        return code
    base_model, env, names = DEFAULT_MODELS["student_base"], dict.fromkeys(SERVE_ENVS, ""), []
    try:
        if args.run and base:
            raise ValueError("--run serves one merged model; --base and --runs serve the base")
        if args.runs:
            names = serve_runs(args.runs)
            env["TE_SERVE_LORAS"] = ",".join(names)
            what = (f"the base model as {BASE_MODEL_NAME} with the adapters of "
                    f"{', '.join(names)} (runs/<run>/out), each under its run name")
        elif args.base:
            env["TE_SERVE_BASE"] = "1"
            what = (f"the base model {base_model} at revision {STUDENT_BASE_REVISION[:12]} as "
                    f"{BASE_MODEL_NAME}")
        elif args.run and RUN_NAME.fullmatch(args.run):
            env["TE_SERVE_RUN"] = args.run
            what = f"runs/{args.run}/merged"
        else:
            raise ValueError("pass --run <run> (the merged model under runs/<run>/merged), "
                             "--base, --runs <run,run,...> or --stop")
    except ValueError as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    if base:
        check_model_allowed(base_model)
    print(f"deploys {app} serving {what} on one {GPU}. A container is billed at "
          f"${MODAL_GPU_USD_PER_HOUR[GPU]:.2f} an hour (not checked) while it is up; it stops "
          f"after a few idle minutes, and te serve-student --stop{' --base' if base else ''} "
          "stops it now.")
    if names:
        print("note: LoRA serving is not documented for Muse Glimmer (the cookbook's vllm.md), "
              "so it is untried: send one image request per run before relying on it")
    for name in names:
        try:
            result = json.loads((settings.data_dir / "train" / f"{name}.result.json").read_text())
        except (OSError, ValueError):
            result = {}
        if not isinstance(result, dict) or not result or result.get("ok") is False or (
                result.get("smoke")):
            print(f"note: data/train/{name}.result.json shows no finished full run of {name}; "
                  f"vLLM fails to start if runs/{name}/out on the volume has no adapter")
    if not args.yes and not cli.ask("Deploy the endpoint?"):
        return 1
    if not cli.modal_ready():
        return 2
    code = subprocess.run(["modal", "deploy", "backend/inference/modal_serve.py"],
                          cwd=PROJECT_ROOT, check=False, env={**os.environ, **env}).returncode
    if code != 0:
        return code
    gpu_ledger.record_deploy(settings.data_dir, app, what, gpu=GPU)
    if not base:
        served = runs.write_serving(settings.data_dir, args.run)
        print(f"recorded in {shown_path(runs.serving_path(settings.data_dir))}: run {args.run} "
              f"serves {', '.join(served['skills']) or 'no skill te train knows of'}")
        print("then set TE_STUDENT_URL to the URL modal printed, plus /v1, and STUDENT_API_KEY "
              'in .env to the value in the Modal Secret "student"')
        return 0
    print(f"then, for the exp4-base replay, TE_STUDENT_URL=<the URL modal printed>/v1 and "
          f"TE_STUDENT_MODEL={BASE_MODEL_NAME} (te watch --max-tokens if replies stop before "
          f"yes or no); te regress --base-url <that URL>/v1 --base-model {BASE_MODEL_NAME}. "
          'STUDENT_API_KEY is the value in the Modal Secret "student", as for the student.')
    if names:
        print(f"each run is a model name on that endpoint: TE_STUDENT_MODEL={names[0]} with te "
              f"watch --run {names[0]}, and te regress --ft-url <that URL>/v1 --ft-model "
              f"{names[0]}")
    return 0


def add_parsers(sub) -> None:
    export = sub.add_parser("export", help="merge a trained adapter on Modal and check it")
    export.add_argument("--run", required=True, help="the training run, e.g. exp1-mixed")
    export.add_argument("--reference", choices=["bf16", "nf4"], default="bf16",
                        help="base precision the merged model is compared against")
    export.add_argument("--limit", type=positive, default=50, help="examples to compare")
    export.add_argument("--yes", action="store_true", help="no question (not over $15)")
    export.set_defaults(func=cmd_export)

    serve_student = sub.add_parser("serve-student", help="deploy or stop the student endpoint")
    serve_student.add_argument("--run", help="the run whose merged model to serve")
    serve_student.add_argument("--base", action="store_true", help="serve the base model "
                               "(app trained-eye-base, model muse-glimmer); with --stop, stop "
                               "that app")
    serve_student.add_argument("--runs", help="comma-separated runs: the base with each run's "
                               "adapter, each run a model name (vLLM --enable-lora; untried)")
    serve_student.add_argument("--stop", action="store_true", help="stop the endpoint")
    serve_student.add_argument("--yes", action="store_true", help="no question")
    serve_student.set_defaults(func=cmd_serve_student)
