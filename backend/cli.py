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
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

from backend.config import ConfigError, load_settings
from backend.meta.client import MetaAPIError, MetaClient
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


def cmd_api_smoke(args) -> int:
    settings = load_settings()
    model = settings.model("labeler")
    bound = token_cost(settings.price(model), 200, args.max_tokens)
    print(f"one structured call to {model}, at most about ${bound:.4f}")
    result = make_client(settings).structured(
        model, "Reply with ok set to true.", SMOKE_SCHEMA, name="smoke", skill="smoke",
        max_tokens=args.max_tokens,
    )
    note = "  (cache hit: no call made)" if result.info.cached else ""
    print(f"result: {json.dumps(result.data)}")
    print(f"cost: ${result.info.cost_usd:.6f}{note}")
    print(f"usage: {json.dumps(result.info.usage, sort_keys=True)}")
    return 0 if result.data.get("ok") is True else 1


def cmd_serve(args) -> int:
    import uvicorn

    print(f"Trained Eye backend: ws://{args.host}:{args.port}/ws  (Ctrl+C to stop)")
    uvicorn.run("backend.server:app", host=args.host, port=args.port, log_level="info")
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
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def cmd_teach(args) -> int:
    from backend.dataengine.keyframes import load_frames, select_keyframes
    from backend.skills.spec import check_name
    from backend.teach import TeachOptions, estimate, run_teach

    try:
        check_name(args.skill)
    except ValueError as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    settings = load_settings()
    client = make_client(settings)
    if not (args.audio or args.request):
        print("te: pass --audio or --request", file=sys.stderr)
        return 2
    audio_s = (audio_duration_s(Path(args.audio)) or 10.0) if args.audio else 0.0
    keyframes = len(select_keyframes(load_frames(args.clip)))
    parts = estimate(client, audio_s=audio_s, keyframes=keyframes,
                     positives=args.positives, negatives=args.negatives)
    print(f"estimate for {args.skill} (before cache hits): " + ", ".join(
        f"{k} ${v:.2f}" for k, v in parts.items()))
    if args.estimate_only:
        return 0
    if not args.yes and not ask(f"Spend up to about ${parts['total']:.2f}?"):
        return 1
    labels = json.loads(Path(args.labels).read_text()) if args.labels else {}
    opts = TeachOptions(skill=args.skill, clip=Path(args.clip),
                        audio=Path(args.audio) if args.audio else None, request=args.request,
                        positives=args.positives, negatives=args.negatives, real_labels=labels,
                        audio_s=audio_s or None)
    confirm = (lambda sentence: True) if args.yes else (lambda sentence: ask("Is that right?"))
    report = run_teach(client, opts, settings.data_dir / "skills", confirm=confirm)
    if report.get("stopped"):
        print(f"stopped: {report['stopped']}")
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
    teach.add_argument("--audio", help="the spoken request (wav, m4a, ...)")
    teach.add_argument("--request", help="the request as text, instead of --audio")
    teach.add_argument("--positives", type=int, default=150)
    teach.add_argument("--negatives", type=int, default=150)
    teach.add_argument("--labels", help="JSON of your own keyframe labels {file: yes|no}")
    teach.add_argument("--estimate-only", action="store_true", help="print the cost estimate")
    teach.add_argument("--yes", action="store_true", help="skip both confirmations")
    teach.set_defaults(func=cmd_teach)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, MetaAPIError, BudgetExceeded) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
