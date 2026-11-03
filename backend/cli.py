"""The `te` command line.

  te spend          print Meta API spend from the ledger, and Modal GPU spend
  te models         list the models your MODEL_API_KEY can use (free)
  te api-smoke      one tiny paid Muse Spark call through the cache and ledger
  te serve          run the backend (WebSocket at /ws)
  te ws-selftest    stream synthetic frames through the WebSocket, in process
  te policy         check a request against the local policy rules (free, offline)
  te teach          teach a skill: request + 10 s clip -> spec, generated data, labels, dataset
  te train          bundle taught skills and train the student on Modal (asks first)
  te watch          replay a clip through watch mode: scores, alerts and the frame log
  te calibrate      set a skill's threshold from replayed, labelled validation clips
  te label          mark where a clip's condition is true, or hand-label generated images
  te eval           score replayed, labelled clips: results.md and the charts (make eval)
  te export         merge a trained adapter on Modal and check it answers as before (asks)
  te serve-student  deploy or stop the trained model's endpoint on Modal (asks)
  te experiments, te unfiltered, te spark-errors, te regress: M3 (backend/cli_experiments.py)
  te manifests, te regenerate, te blur, te release, te playground: M4 (backend/cli_release.py)
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from backend.config import (
    ASK_ABOVE_USD,
    MODAL_GPU_USD_PER_HOUR,
    PROJECT_ROOT,
    ConfigError,
    load_settings,
    shown_path,
)
from backend.meta.client import MetaAPIError, MetaClient, OutputError
from backend.meta.ledger import BudgetExceeded, Ledger, format_report, token_cost
from backend.training import gpu_ledger, runs
from backend.training.runs import RUN_NAME

SMOKE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def make_ledger(settings) -> Ledger:
    return Ledger(settings.ledger_path, settings.skill_cap_usd, settings.global_cap_usd)


def make_client(settings) -> MetaClient:
    return MetaClient(settings)


def cmd_spend(args) -> int:
    settings = load_settings()
    print(format_report(make_ledger(settings)))
    print(gpu_ledger.format_report(settings.data_dir))  # data/modal_ledger.jsonl
    return 0


def cmd_models(args) -> int:
    settings = load_settings()
    offered = make_client(settings).list_models()
    for model_id in offered:
        print(model_id)
    wanted = {m for job, m in settings.models.items() if job != "student_base"}
    missing = sorted(wanted - set(offered))
    print(f"configured models missing for this key: {', '.join(missing) or 'none'}")
    return 1 if missing else 0


def check_smoke_reply(data) -> None:
    """Raise for any reply other than {"ok": true}, so a failed smoke check is never cached."""
    if not (isinstance(data, dict) and data.get("ok") is True):
        raise OutputError(f"unexpected smoke reply: {json.dumps(data)[:100]}")


def cmd_api_smoke(args) -> int:
    settings = load_settings()
    model = settings.model("labeler")
    bound = token_cost(settings.price(model), 200, args.max_tokens)
    print(f"one structured call to {model}, at most about ${bound:.4f}")
    result = make_client(settings).structured(
        model, "Reply with ok set to true.", SMOKE_SCHEMA, name="smoke", skill="smoke",
        max_tokens=args.max_tokens, validate=check_smoke_reply,
    )
    note = "  (cache hit: no call made)" if result.info.cached else ""
    print(f"result: {json.dumps(result.data)}")
    print(f"cost: ${result.info.cost_usd:.6f}{note}")
    print(f"usage: {json.dumps(result.info.usage, sort_keys=True)}")
    return 0


def lan_address() -> str | None:
    """This machine's address on the local network, as the phone would reach it. No
    packet is sent: connecting a UDP socket only picks the outgoing interface."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            address = s.getsockname()[0]
    except OSError:
        return None
    return None if address.startswith("127.") or address == "0.0.0.0" else address


def server_log_config() -> dict:
    """uvicorn's logging, plus the server's own lines (session opened, frames, closed)."""
    import uvicorn.config

    config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    config["loggers"]["trained_eye"] = {"handlers": ["default"], "level": "INFO",
                                        "propagate": False}
    return config


def cmd_serve(args) -> int:
    import uvicorn

    host = args.host
    if host in ("0.0.0.0", "::"):
        host = lan_address() or "<this Mac's Wi-Fi IP>"
    print(f"Trained Eye backend: ws://{host}:{args.port}/ws  (check: "
          f"http://{host}:{args.port}/health; Ctrl+C to stop)")
    try:
        settings = load_settings()
        teach = "off (set SERVER_TOKEN in .env to turn it on)"
        if settings.has_secret("SERVER_TOKEN"):
            settings.secret("SERVER_TOKEN")  # a token with spaces or smart quotes stops here
            teach = "on (the app needs SERVER_TOKEN)"
    except ConfigError as err:
        teach = f"off ({err})"
    print(f"teaching from the phone: {teach}")
    uvicorn.run("backend.server:app", host=args.host, port=args.port, log_level="info",
                log_config=server_log_config())
    return 0


def synthetic_jpeg(i: int, size: tuple[int, int] = (640, 480)) -> bytes:
    from PIL import Image

    shade = 40 + (i * 50) % 200
    buf = io.BytesIO()
    Image.new("RGB", size, (shade, shade, shade)).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def cmd_ws_selftest(args) -> int:
    from starlette.testclient import TestClient

    from backend.server import create_app

    acks = 0
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "client": "cli", "source": "mock", "fps": 1})
        welcome = ws.receive_json()
        print(f"connected: session {welcome['session']}, protocol {welcome['protocol']}")
        for i in range(args.frames):
            ws.send_bytes(synthetic_jpeg(i))
            reply = ws.receive_json()
            if reply["type"] == "ack":
                acks += 1
                print(f"frame {reply['seq']}: ack {reply['width']}x{reply['height']}, "
                      f"{reply['bytes']} bytes")
            else:
                print(f"frame {i + 1}: {reply}")
        ws.send_json({"type": "bye"})
    print(f"sent {args.frames} frames, got {acks} acks")
    return 0 if acks == args.frames else 1


def cmd_policy(args) -> int:
    from backend.skills.spec import policy_check

    refusal = policy_check(args.request)
    if refusal:
        print(f"refused: {refusal}")
        return 1
    print("ok: no local policy objection (the skill compiler checks again)")
    return 0


def audio_duration_s(path: Path) -> float | None:
    """Duration via ffprobe, for audio formats the client cannot measure itself."""
    if shutil.which("ffprobe") is None:
        return None
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(path)], capture_output=True, text=True, check=False,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def ask(question: str) -> bool:
    try:
        answer = input(f"{question} [y/N] ")
    except (EOFError, RuntimeError, OSError):  # stdin empty, closed or not readable: "no"
        print()
        return False
    return answer.strip().lower() in ("y", "yes")


RESUME_HINT = ("finished calls are cached, so running the same command again continues "
               "without paying for them twice")
FAILURES_HINT = ("failed calls are not cached and are paid for again on a rerun: look at "
                 "the reason above (and the skill folder) before running it again")


class Stopped(ValueError):
    """te teach cannot go on with these inputs; found before anything is spent."""


class Refused(Stopped):
    """The local policy check refused the request text."""


def _unique_keys(pairs: list[tuple]) -> dict:
    keys = [k for k, _ in pairs]
    repeated = sorted({k for k in keys if keys.count(k) > 1})
    if repeated:
        raise ValueError(f"--labels names {', '.join(map(repr, repeated))} more than once")
    return dict(pairs)


def teach_inputs(args, settings) -> tuple[list, dict[str, str], float]:
    """Check everything te teach needs before any money is mentioned: (keyframes, your
    labels, audio seconds). Raises ValueError with a message for the user."""
    from backend.dataengine.keyframes import load_frames, select_keyframes
    from backend.skills.compiler import MAX_REQUEST
    from backend.skills.spec import check_name, policy_check
    from backend.teach import (
        check_clip,
        keyframe_name,
        labels_problem,
        normalise_labels,
        save_keyframes,
    )

    check_name(args.skill)
    request = " ".join((args.request or "").split())
    if not (args.audio or request):
        raise ValueError("pass --audio or --request")
    if args.request and len(request) > MAX_REQUEST:
        raise ValueError(f"--request is {len(request)} characters, over the {MAX_REQUEST} "
                         "allowed: say it in one sentence")
    if args.request and policy_check(request):
        raise Refused(policy_check(request))
    if args.audio and not Path(args.audio).is_file():
        raise ValueError(f"no audio file at {args.audio}")
    clip = Path(args.clip)
    if not clip.exists():
        raise ValueError(f"no clip at {clip}")
    skill_root = settings.data_dir / "skills" / args.skill
    check_clip(clip, skill_root)
    try:
        frames = select_keyframes(load_frames(clip))
    except (RuntimeError, subprocess.CalledProcessError) as err:
        raise ValueError(f"cannot read the clip {clip}: {err}") from None
    if not frames:
        raise ValueError(f"no usable frames in {clip} (a video, or a folder of .jpg, .png or "
                         ".webp frames)")
    names = [keyframe_name(i, f.data) for i, f in enumerate(frames)]
    labels = {}
    if args.labels:  # checked before the estimate, so a typo never costs anything
        try:
            raw = json.loads(Path(args.labels).read_text(), object_pairs_hook=_unique_keys)
            labels = normalise_labels(raw, names)
        except (OSError, TypeError) as err:
            raise ValueError(str(err)) from None
    problem = labels_problem(settings.train_on_model_outputs, labels, skill_root / "keyframes")
    if problem:
        save_keyframes(skill_root / "keyframes", frames)  # so they can be labelled, for free
        raise Stopped(problem)
    audio_s = (audio_duration_s(Path(args.audio)) or 10.0) if args.audio else 0.0
    return frames, labels, audio_s


def cmd_teach(args) -> int:
    from backend.dataengine.synth import TooManyFailures
    from backend.meta.ledger import image_tokens
    from backend.teach import TeachOptions, estimate, run_teach

    settings = load_settings()
    try:
        frames, labels, audio_s = teach_inputs(args, settings)
    except Refused as err:
        print(f"I can't learn that one. {err}")
        print('stopped: refused (nothing was spent). te policy "<request>" checks a wording '
              "for free.")
        return 1
    except Stopped as err:
        print(f"stopped: {err} (nothing was spent)")
        return 1
    except ValueError as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    client = make_client(settings)
    parts = estimate(client, audio_s=audio_s, keyframes=len(frames),
                     positives=args.positives, negatives=args.negatives)
    at_most = parts.pop("at_most")
    print(f"estimate for {args.skill} (before cache hits): " + ", ".join(
        f"{k} ${v:.2f}" for k, v in parts.items()))
    print(f"  at most ${at_most:.2f} if every label has to be asked again with a larger budget")
    print(f"  {image_tokens(settings, client.ledger).describe()}")  # what estimate() used
    print("train on model outputs: " + ("yes" if settings.train_on_model_outputs else
                                        "no (the dataset uses real frames with your own labels)"))
    ledger = make_ledger(settings)
    room = min(settings.skill_cap_usd - ledger.spent(args.skill),
               settings.global_cap_usd - ledger.spent())
    if parts["total"] > room:
        print(f"te: warning: only ${max(room, 0):.2f} is left under the spend caps for "
              f"{args.skill}. Cached steps are free, but uncached calls past that stop the "
              "run partway, before the dataset. Raise TE_SKILL_CAP_USD or TE_GLOBAL_CAP_USD, "
              "or lower --positives and --negatives.", file=sys.stderr)
    elif at_most > room:
        print(f"te: note: ${max(room, 0):.2f} is left under the spend caps for {args.skill}. "
              "If many labels need the larger budget the run could pass that, and the cap "
              "would stop it partway (a rerun with a higher cap continues).", file=sys.stderr)
    if args.estimate_only:
        return 0
    settings.secret("MODEL_API_KEY")  # a missing key stops here, before the question
    over = at_most > ASK_ABOVE_USD
    if over:
        print(f"te: this could cost up to ${at_most:.2f}, over ${ASK_ABOVE_USD:.0f}, so it needs "
              "your yes even with --yes")
    if (over or not args.yes) and not ask(f"Spend up to about ${parts['total']:.2f} "
                                          f"(at most ${at_most:.2f})?"):
        return 1
    opts = TeachOptions(skill=args.skill, clip=Path(args.clip),
                        audio=Path(args.audio) if args.audio else None, request=args.request,
                        positives=args.positives, negatives=args.negatives, real_labels=labels,
                        audio_s=audio_s or None, correction=args.correct)
    confirm = (lambda sentence: True) if args.yes else (lambda sentence: ask("Is that right?"))
    try:
        report = run_teach(client, opts, settings.data_dir / "skills", confirm=confirm)
    except KeyboardInterrupt:
        print(f"\nte: stopped; {RESUME_HINT}", file=sys.stderr)
        return 130
    except TooManyFailures as err:
        print(f"te: {err}\nte: teach stopped; {FAILURES_HINT}", file=sys.stderr)
        return 2
    except MetaAPIError:
        print(f"te: teach stopped; {RESUME_HINT}", file=sys.stderr)
        raise
    except ValueError as err:  # e.g. a spoken request over 500 characters
        print(f"te: {err}", file=sys.stderr)
        return 2
    if report.get("stopped"):
        print(f"stopped: {report['stopped']}")
        if report["stopped"] == "not confirmed":
            print('To fix the read-back, run the same command with --correct "what is wrong" '
                  "(or a reworded --request); the clip and transcript are reused for free.")
        elif report["stopped"] == "refused":
            print('To try another wording first, te policy "<request>" checks it for free.')
        return 1
    print(format_report(make_ledger(settings)))
    return 0


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
    if not modal_ready():
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
    if not args.yes and not ask("Run the preflight on Modal?"):
        return 1
    if not modal_ready():
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
    if not ask("Upload them to W&B? (asked even with --yes)"):
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
              "run te teach again to rewrite them the new way")
    if args.preflight:
        return preflight_train(settings, args, out)
    timing_path = settings.data_dir / "train" / "smoke-timing.json"
    estimate = None
    if args.smoke:
        print(f"two-step smoke on one {GPU} on Modal. The first one also builds the image "
              "(flash-attn compiles) and downloads the 30B base to the volume, so its time is "
              "only known once it has run.")
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
    if (over or credit or not args.yes) and not ask(question):
        return 1
    if shutil.which("modal") is None:
        print('te: the modal command is missing: pip install -e ".[train]", then modal token '
              "set", file=sys.stderr)
        return 2
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
    result = json.loads(result_path.read_text()) if result_path.is_file() else None
    if result is not None:  # also a failed run: its GPU time was billed
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
    if bundle is not None and (untrained := sorted(set(names) - set(bundle.get("skills", {})))):
        print(f"note: run {args.run} was not trained on {', '.join(untrained)}")
    if folded:
        print("the student is asked with the system prompt in the user text, as its training "
              "data was written")
    client = make_client(settings)
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
        if (over or not args.yes) and not ask("Spend that?"):
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
        result = calibrate(clips, float(spec.cooldown_s))
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
          + (f", median delay {delay:.0f} s" if delay is not None else ""))
    print(f"saved: {shown_path(target)}")
    if minutes < 60:
        print("note: under an hour of footage, so even one false alarm is too many here")
    return 0


EXPORT_MAX_HOURS = 3  # backend/training/export.py's TIMEOUT_S: the most an export can bill
STUDENT_APP = "trained-eye-student"  # backend/inference/modal_serve.py's APP
BASE_APP = "trained-eye-base"  # modal_serve.py's BASE_APP: the base, and --runs adapters
BASE_MODEL_NAME = "muse-glimmer"  # modal_serve.py's BASE_MODEL_NAME
SERVE_ENVS = ("TE_SERVE_RUN", "TE_SERVE_BASE", "TE_SERVE_LORAS")  # modal_serve.py reads them


def modal_ready() -> bool:
    if shutil.which("modal") is None:
        print('te: the modal command is missing: pip install -e ".[train]", then modal token '
              "set", file=sys.stderr)
        return False
    return True


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
    if (over or credit or not args.yes) and not ask("Run the export on Modal?"):
        return 1
    if not modal_ready():
        return 2
    result_path = bundle.parent / f"{args.run}.export.json"
    result_path.unlink(missing_ok=True)
    command = ["modal", "run", "backend/training/export.py", "--run", args.run, "--bundle",
               str(bundle), "--limit", str(args.limit), "--reference", args.reference,
               "--result", str(result_path)]
    code = subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
    result = json.loads(result_path.read_text()) if result_path.is_file() else None
    seconds = (result or {}).get("seconds")
    if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
        gpu_ledger.record(settings.data_dir, "export", args.run, seconds, gpu=GPU)
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
        if not modal_ready():
            return 2
        code = subprocess.run(["modal", "app", "stop", app], cwd=PROJECT_ROOT,
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
    if not args.yes and not ask("Deploy the endpoint?"):
        return 1
    if not modal_ready():
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


def cmd_eval(args) -> int:
    from backend.eval.readme import write_readme
    from backend.eval.report import run_eval

    settings = load_settings()
    try:
        if args.wandb_entity and not args.wandb:
            raise ValueError("--wandb-entity needs --wandb")
        if args.readme and not Path(args.readme).is_file():
            raise ValueError(f"--readme: no file {args.readme}")
        results = run_eval(Path(args.plan), Path(args.out), log_wandb=args.wandb,
                           wandb_entity=args.wandb_entity, data_dir=settings.data_dir,
                           regression=PROJECT_ROOT / "eval" / "regression.json")
    except (OSError, ValueError) as err:  # pydantic's ValidationError is a ValueError
        print(f"te: {err}", file=sys.stderr)
        return 2
    evals = results["evaluations"]
    scored = sum(e["status"] == "scored" for e in evals)
    print(f"{scored} of {len(evals)} evaluations scored ({len(evals) - scored} waiting for "
          f"clips); report: {shown_path(Path(args.out) / 'results.md')}")
    if results.get("galleries"):
        print(f"failure galleries (local only, may show faces: te blur before publishing): "
              f"{shown_path(Path(args.out) / 'galleries')}")
    if args.wandb:
        print(f"W&B: {results['wandb']}")
    if args.readme:
        try:
            write_readme(Path(args.readme), Path(args.out) / "results.json")
        except (OSError, ValueError) as err:
            print(f"te: --readme: {err}; {args.readme} was left as it is", file=sys.stderr)
            return 1
        print(f"results block written into {args.readme} (from results.json only)")
    return 0


def cmd_label(args) -> int:
    import threading
    import webbrowser

    import uvicorn

    from backend.skills.spec import check_name

    settings = load_settings()
    try:
        if args.clip:
            from backend.labeltool.clips import create_clip_app

            clip = Path(args.clip)
            out = Path(args.out) if args.out else clip.with_suffix(".labels.json")
            app = create_clip_app(clip, out)
        else:
            from backend.labeltool.images import create_image_app, sample_images

            root = settings.data_dir / "skills" / check_name(args.skill)
            out = Path(args.out) if args.out else root / "human_labels.json"
            images = sample_images(root / "labels.jsonl", args.sample, args.seed)
            app = create_image_app(images, out)
    except (OSError, ValueError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    url = f"http://127.0.0.1:{args.port}/"
    print(f"labelling tool: {url}  (labels: {shown_path(out)}; Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def positive(text: str) -> int:
    """An argparse type: a whole number, 1 or more."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


def fraction(text: str) -> float:
    """An argparse type: a number from 0 to 1."""
    value = float(text)
    if not 0.0 <= value <= 1.0:  # also false for nan
        raise argparse.ArgumentTypeError(f"must be from 0 to 1, got {text}")
    return value


def count(text: str) -> int:
    """An argparse type: a whole number, 0 or more."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="te", description="Trained Eye")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("spend", help="print Meta API spend").set_defaults(func=cmd_spend)
    sub.add_parser("models", help="list models for this key").set_defaults(func=cmd_models)

    smoke = sub.add_parser("api-smoke", help="one tiny paid call to check the key")
    smoke.add_argument("--max-tokens", type=int, default=512)
    smoke.set_defaults(func=cmd_api_smoke)

    serve = sub.add_parser("serve", help="run the backend server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=cmd_serve)

    selftest = sub.add_parser("ws-selftest", help="send synthetic frames over the WebSocket")
    selftest.add_argument("--frames", type=int, default=3)
    selftest.set_defaults(func=cmd_ws_selftest)

    policy = sub.add_parser("policy", help="check a request against the local policy rules")
    policy.add_argument("request")
    policy.set_defaults(func=cmd_policy)

    teach = sub.add_parser("teach", help="teach a skill from a request and a 10 s clip")
    teach.add_argument("--skill", required=True, help="skill name, e.g. washer_done")
    teach.add_argument("--clip", required=True, help="video file or folder of JPEG frames")
    said = teach.add_mutually_exclusive_group()
    said.add_argument("--audio", help="the spoken request (wav, m4a, ...)")
    said.add_argument("--request", help="the request as text, instead of --audio")
    teach.add_argument("--correct", help='fix a wrong read-back, e.g. "the light on the lid"')
    teach.add_argument("--positives", type=count, default=150)
    teach.add_argument("--negatives", type=count, default=150)
    teach.add_argument("--labels", help='JSON of your own keyframe labels {"kf_00_1a2b3c4d.jpg": '
                       '"yes" or "no"}, names as in data/skills/<skill>/keyframes/')
    teach.add_argument("--estimate-only", action="store_true", help="print the cost estimate")
    teach.add_argument("--yes", action="store_true", help="skip both confirmations")
    teach.set_defaults(func=cmd_teach)

    train = sub.add_parser("train", help="train the student on Modal from taught skills")
    train.add_argument("--skills", help="comma-separated, e.g. washer_done (needed but with "
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

    watch = sub.add_parser("watch", help="replay a clip through watch mode")
    watch.add_argument("--skills", required=True, help="comma-separated, e.g. washer_done")
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

    ev = sub.add_parser("eval", help="score replayed, labelled clips: results.md and charts")
    ev.add_argument("--plan", required=True, help="the evaluation plan (backend/eval/report.py)")
    ev.add_argument("--out", required=True, help="folder for results.json, results.md, charts")
    ev.add_argument("--wandb", action="store_true", help="also log to W&B (needs wandb)")
    ev.add_argument("--wandb-entity", help="with --wandb: the W&B team (default: your own "
                    "account)")
    ev.add_argument("--readme", help="also write the results block between the te eval "
                    "markers in this file, e.g. README.md (backend/eval/readme.py)")
    ev.set_defaults(func=cmd_eval)

    label = sub.add_parser("label", help="mark where a clip's condition is true, or label images")
    what = label.add_mutually_exclusive_group(required=True)
    what.add_argument("--clip", help="a recorded clip, e.g. data/clips/<skill>/session3_test_a.mp4")
    what.add_argument("--skill", help="hand-label sampled generated images of this skill")
    label.add_argument("--sample", type=count, default=200, help="with --skill: images to label")
    label.add_argument("--seed", type=int, default=0, help="with --skill: same seed, same images")
    label.add_argument("--out", help="labels file; default <clip>.labels.json next to the clip, or "
                       "data/skills/<skill>/human_labels.json")
    label.add_argument("--port", type=int, default=8766)
    label.add_argument("--no-browser", action="store_true", help="print the address only")
    label.set_defaults(func=cmd_label)

    from backend.cli_experiments import add_parsers as experiment_parsers

    experiment_parsers(sub)

    from backend.cli_release import add_parsers as release_parsers

    release_parsers(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BudgetExceeded as err:
        print(f"te: {err}\nte: raise TE_SKILL_CAP_USD (or TE_GLOBAL_CAP_USD) and run the same "
              f"command to continue; {RESUME_HINT}", file=sys.stderr)
        return 2
    except (ConfigError, MetaAPIError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
