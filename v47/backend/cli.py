"""The `te` command line.

  te spend          print Meta API spend from the ledger
  te models         list the models your MODEL_API_KEY can use (free)
  te api-smoke      one tiny paid Muse Spark call through the cache and ledger
  te serve          run the backend (WebSocket at /ws)
  te ws-selftest    stream synthetic frames through the WebSocket, in process
  te policy         check a request against the local policy rules (free, offline)
  te teach          teach a skill: request + 10 s clip -> spec, generated data, labels, dataset
  te train          bundle taught skills and train the student on Modal (asks first)
  te watch          replay a clip through watch mode: scores, alerts and the frame log
  te calibrate      set a skill's threshold from replayed, labelled validation clips
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import re
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
    print(format_report(make_ledger(load_settings())))
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
                 "the reasons above (and in the skill folder) before running it again")


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
    if not (args.audio or args.request):
        raise ValueError("pass --audio or --request")
    request = " ".join((args.request or "").split())
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
    if not args.yes and not ask(f"Spend up to about ${parts['total']:.2f} "
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
    except TooManyFailures:
        print(f"te: teach stopped; {FAILURES_HINT}", file=sys.stderr)
        raise
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


RUN_NAME = re.compile(r"[a-z0-9][a-z0-9-]{1,39}")
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


def cmd_train(args) -> int:
    from backend.skills.spec import check_name
    from backend.training.axolotl import optimizer_steps
    from backend.training.prepare import bundle

    settings = load_settings()
    try:
        names = [check_name(n.strip()) for n in args.skills.split(",") if n.strip()]
        if not names:
            raise ValueError("--skills needs at least one skill name")
        if not RUN_NAME.fullmatch(args.run):
            raise ValueError(f"run name {args.run!r}: lower-case letters, digits and -")
        out = settings.data_dir / "train" / args.run
        manifest = bundle([settings.data_dir / "skills" / n for n in names], out,
                          f"/vol/bundles/{args.run}", variant=args.variant, limit=args.limit)
    except ValueError as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    train_n, val_n = manifest["train_examples"], manifest.get("val_examples", 0)
    print(f"bundle {args.run}: {train_n} training and {val_n} validation examples "
          f"({args.variant}) -> {shown_path(out)}")
    timing_path = settings.data_dir / "train" / "smoke-timing.json"
    estimate = None
    if args.smoke:
        print(f"two-step smoke on one {GPU} on Modal. The first one also builds the image "
              "(flash-attn compiles) and downloads the 30B base to the volume, so its time is "
              "only known once it has run.")
        question = "Run the smoke on Modal?"
    else:
        timing = json.loads(timing_path.read_text()) if timing_path.is_file() else None
        estimate = train_estimate(train_n + val_n, timing)
        if estimate is None:
            print("te: no smoke timing yet: run the same command with --smoke first, so this "
                  "run can be priced", file=sys.stderr)
            return 2
        print(f"{optimizer_steps(train_n)} optimizer steps; estimate: about "
              f"{estimate['seconds'] / 60:.0f} min of {GPU}, about ${estimate['usd']:.2f} at "
              f"${MODAL_GPU_USD_PER_HOUR[GPU]:.2f} an hour (price not checked; see config.py)")
        question = f"Train {args.run} on Modal for about ${estimate['usd']:.2f}?"
    if args.estimate_only:
        return 0
    over = estimate is not None and estimate["usd"] > ASK_ABOVE_USD
    if over:
        print(f"te: this is over ${ASK_ABOVE_USD:.0f}, so it needs your yes even with --yes")
    if (over or not args.yes) and not ask(question):
        return 1
    if shutil.which("modal") is None:
        print('te: the modal command is missing: pip install -e ".[train]", then modal token '
              "set", file=sys.stderr)
        return 2
    result_path = out.parent / f"{args.run}.result.json"
    result_path.unlink(missing_ok=True)
    tags = [f"exp:{args.run.split('-')[0]}", f"skills:{'+'.join(names)}",
            f"mix:{args.variant}", f"examples:{train_n}", "seed:42", f"commit:{git_commit()}"]
    command = ["modal", "run", "backend/training/modal_train.py", "--bundle", str(out),
               "--run", args.run, "--tags", ",".join(tags), "--result", str(result_path)]
    if args.smoke:
        command.append("--smoke")
    code = subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
    if code != 0 or not result_path.is_file():
        print(f"te: training failed (exit {code}); the full log is on the Modal volume under "
              f"runs/{args.run}/train.log", file=sys.stderr)
        return code or 1
    result = json.loads(result_path.read_text())
    if args.smoke and result.get("train_samples_per_second"):
        timing_path.write_text(json.dumps(result, indent=1) + "\n")
        print(f"smoke timing saved: {shown_path(timing_path)}")
    print(f"done: {result['seconds']:.0f} s in the training container; the adapter is on the "
          f"volume under runs/{args.run}/out")
    return 0


def cmd_watch(args) -> int:
    from backend.dataengine.keyframes import load_frames
    from backend.dataengine.labels import label_image
    from backend.inference.student import OpenAIStudent
    from backend.skills.spec import check_name
    from backend.watch.watcher import FrameLog, Watcher, load_skill

    settings = load_settings()
    if not settings.student_url:
        print("te: watch needs the trained model's endpoint: set TE_STUDENT_URL",
              file=sys.stderr)
        return 2
    try:
        names = [check_name(n.strip()) for n in args.skills.split(",") if n.strip()]
        skills = []
        for name in names:
            folder = settings.data_dir / "skills" / name
            if not (folder / "spec.json").is_file():
                raise ValueError(f"no taught skill named {name!r}")
            skills.append(load_skill(folder))
        frames = load_frames(args.clip, fps=args.fps)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    for skill in skills:
        if not skill.calibrated:
            print(f"note: {skill.spec.name} has no calibrated threshold yet; using "
                  f"{skill.trigger.threshold}")
    client = make_client(settings)
    confirm = None if args.no_cloud else (lambda image, spec: label_image(client, spec, image))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log = FrameLog(settings.data_dir / "watch" / f"replay-{stamp}-{Path(args.clip).stem}.jsonl")
    watcher = Watcher(skills, OpenAIStudent(settings.student_url, settings.student_model),
                      confirm=confirm, log=log)
    count = 0
    for frame in frames:
        _, alerts = watcher.on_frame(frame.data, t=frame.t_s)
        count += 1
        for alert in alerts:
            cloud = {True: "confirmed", None: "not checked"}.get(alert.confirmed, "no")
            print(f"{alert.t:8.1f} s  {alert.skill}: \"{alert.text}\"  "
                  f"(p {alert.p:.2f}, cloud {cloud})")
    print(f"{count} frames scored at {args.fps:g} fps; frame log: {shown_path(log.path)}")
    return 0


def cmd_calibrate(args) -> int:
    from backend.skills.spec import SkillSpec, check_name
    from backend.watch.trigger import calibrate, labelled_frames

    settings = load_settings()
    try:
        check_name(args.skill)
        folder = settings.data_dir / "skills" / args.skill
        spec = SkillSpec.model_validate_json((folder / "spec.json").read_text())
        if len(args.log) != len(args.labels):
            raise ValueError("give one --labels file for each --log")
        clips, sources = [], []
        for log, labels in zip(args.log, args.labels, strict=True):
            rows = [json.loads(line) for line in Path(log).read_text().splitlines() if line]
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
    (folder / "threshold.json").write_text(json.dumps({**result, "clips": sources}, indent=1)
                                           + "\n")
    delay = result["median_delay_s"]
    print(f"{args.skill}: threshold {result['threshold']:.3f} on {minutes:.0f} minutes of "
          f"footage: {result['false_alarms']} false alarms, {result['caught']} of "
          f"{result['events']} events caught"
          + (f", median delay {delay:.0f} s" if delay is not None else ""))
    if minutes < 60:
        print("note: under an hour of footage, so even one false alarm is too many here")
    return 0


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
    teach.add_argument("--positives", type=int, default=150)
    teach.add_argument("--negatives", type=int, default=150)
    teach.add_argument("--labels", help='JSON of your own keyframe labels {"kf_00_1a2b3c4d.jpg": '
                       '"yes" or "no"}, names as in data/skills/<skill>/keyframes/')
    teach.add_argument("--estimate-only", action="store_true", help="print the cost estimate")
    teach.add_argument("--yes", action="store_true", help="skip both confirmations")
    teach.set_defaults(func=cmd_teach)

    train = sub.add_parser("train", help="train the student on Modal from taught skills")
    train.add_argument("--skills", required=True, help="comma-separated, e.g. washer_done")
    train.add_argument("--run", required=True, help="run name, e.g. exp1-mixed")
    train.add_argument("--variant", choices=["mixed", "real", "synthetic"], default="mixed")
    train.add_argument("--limit", type=int, help="at most this many examples per skill")
    train.add_argument("--smoke", action="store_true", help="the recipe's two-step check")
    train.add_argument("--estimate-only", action="store_true", help="bundle and price only")
    train.add_argument("--yes", action="store_true", help="no question (not over $15)")
    train.set_defaults(func=cmd_train)

    watch = sub.add_parser("watch", help="replay a clip through watch mode")
    watch.add_argument("--skills", required=True, help="comma-separated, e.g. washer_done")
    watch.add_argument("--clip", required=True, help="video file or folder of frames")
    watch.add_argument("--fps", type=float, default=1.0, help="frames scored per second")
    watch.add_argument("--no-cloud", action="store_true", help="skip the Muse Spark check")
    watch.set_defaults(func=cmd_watch)

    cal = sub.add_parser("calibrate", help="set a skill's threshold from validation clips")
    cal.add_argument("--skill", required=True)
    cal.add_argument("--log", action="append", required=True,
                     help="a frame log from te watch (data/watch/replay-*.jsonl); repeatable")
    cal.add_argument("--labels", action="append", required=True,
                     help='JSON {"true": [[start_s, end_s], ...]} for the clip of that log')
    cal.set_defaults(func=cmd_calibrate)
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
