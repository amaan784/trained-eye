"""The `te` command line.

  te spend          print Meta API spend from the ledger
  te models         list the models your MODEL_API_KEY can use (free)
  te api-smoke      one tiny paid Muse Spark call through the cache and ledger
  te serve          run the backend (WebSocket at /ws)
  te ws-selftest    stream synthetic frames through the WebSocket, in process
  te policy         check a request against the local policy rules (free, offline)
  te teach          teach a skill: request + 10 s clip -> spec, generated data, labels, dataset
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
from pathlib import Path

from backend.config import ConfigError, load_settings
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
