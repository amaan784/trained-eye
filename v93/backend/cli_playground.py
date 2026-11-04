"""te commands for M4's web playground (registered by backend/cli.py).

  te playground     run the web playground locally (--public: the public copy;
                    --deploy: publish the public copy on Modal, asks first)
  te playground-clip  add a test clip to the playground, face-blurred
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from backend import cli, config
from backend.config import load_settings, shown_path

PLAYGROUND_APP = "trained-eye-playground"  # backend/playground/modal_app.py's APP
PLAYGROUND_VOLUME = "trained-eye-playground"  # and its VOLUME
# What a deploy publishes on the Volume, cleared before each copy (see deploy_playground)
PUBLISHED_TREES = ("/data/skills", "/data/playground/clips", "/data/playground/keyframes",
                   "/eval")
PLAYGROUND_SECRET = "meta"  # and its API_SECRET, which holds MODEL_API_KEY
CAP_ENVS = ("TE_PLAYGROUND_VISITOR_CAP", "TE_PLAYGROUND_DAILY_CAP")  # modal_app.py reads them
LOG_SLACK_S = 1.0  # a frame log may end this far past the video (frame times are rounded)


def cmd_playground(args) -> int:
    import uvicorn

    from backend.playground.app import create_app

    if args.deploy:
        return deploy_playground(args)
    if "*" in [ip.strip() for ip in args.forwarded_allow_ips.split(",")]:
        print('te: --forwarded-allow-ips "*" would let any visitor pick their own address '
              "and reset their spend cap: give the proxy's address", file=sys.stderr)
        return 2
    settings = load_settings()
    app = create_app(settings.data_dir, config.PROJECT_ROOT / args.eval_dir,
                     lambda: cli.make_client(load_settings()),
                     visitor_cap_usd=args.visitor_cap, daily_cap_usd=args.daily_cap,
                     public=args.public)
    live = "on" if settings.has_secret("MODEL_API_KEY") else "off (MODEL_API_KEY is not set)"
    print(f"playground: http://{args.host}:{args.port}/  live teaching {live}", flush=True)
    if args.public:
        print("public copy: no generated images; keyframes and clip videos only face-blurred",
              flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", proxy_headers=True,
                forwarded_allow_ips=args.forwarded_allow_ips)
    return 0


def deploy_playground(args) -> int:
    """Copy the public files to the Modal Volume and deploy modal_app.py (asks first)."""
    from backend.playground.publish import public_files, stage

    total = config.Settings.skill_cap_usd if args.total_cap is None else args.total_cap
    if min(args.visitor_cap, args.daily_cap, total) <= 0:
        print("te: the caps must be above $0", file=sys.stderr)
        return 2
    picked = public_files(load_settings().data_dir, config.PROJECT_ROOT / args.eval_dir)
    for note in picked["notes"]:
        print(f"note: {note}")
    print(f"{len(picked['files'])} files to copy to the private Modal Volume "
          f"{PLAYGROUND_VOLUME} (no generated image, original keyframe or unblurred video). "
          f"Live teaching uses the Modal Secret {PLAYGROUND_SECRET!r} (MODEL_API_KEY) and "
          f"spends at most ${args.visitor_cap:.2f} per visitor a day, ${args.daily_cap:.2f} "
          f"a day and ${total:.2f} in all (the ledger's cap for the skill \"playground\").")
    over = total > config.ASK_ABOVE_USD
    if over:
        print(f"te: ${total:.2f} in all is over ${config.ASK_ABOVE_USD:.0f}, so it needs your "
              "yes even with --yes")
    if (over or not args.yes) and not cli.ask("Publish the playground at a public URL "
                                               f"(Modal app {PLAYGROUND_APP})?"):
        print("nothing was copied or deployed")
        return 1
    if not cli.modal_ready():
        return 2
    caps = {**dict(zip(CAP_ENVS, (args.visitor_cap, args.daily_cap), strict=True)),
            "TE_SKILL_CAP_USD": total, "TE_GLOBAL_CAP_USD": total}  # the Volume's ledger
    env = {**os.environ, **{k: str(v) for k, v in caps.items()}}

    def modal(*command: str, quiet: bool = False) -> int:
        out = subprocess.DEVNULL if quiet else None
        return subprocess.run(["modal", *command], cwd=config.PROJECT_ROOT, check=False,
                              env=env, stdout=out, stderr=out).returncode

    modal("volume", "create", PLAYGROUND_VOLUME, quiet=True)  # fails when it exists: fine
    # `volume put` only adds and overwrites, so a file removed here (a clip pulled for a face
    # the detector missed) would stay public: clear the published trees first. The budget,
    # the ledger and the call cache on the Volume are kept.
    for published in PUBLISHED_TREES:
        modal("volume", "rm", "-r", PLAYGROUND_VOLUME, published, quiet=True)  # may not exist
    with tempfile.TemporaryDirectory() as tmp:
        stage(picked["files"], Path(tmp))
        for part in ("data", "eval"):
            (Path(tmp) / part).mkdir(exist_ok=True)
            if code := modal("volume", "put", "--force", PLAYGROUND_VOLUME,
                             str(Path(tmp) / part), f"/{part}"):
                print(f"te: copying {part}/ to the Volume failed (exit {code}); nothing was "
                      "deployed", file=sys.stderr)
                return code
    code = modal("deploy", "backend/playground/modal_app.py")
    if code == 0:
        print(f"deployed: the URL is above; modal app stop {PLAYGROUND_APP} takes it down")
    return code


def clip_problem(clip: Path, rows: list[dict], ranges, duration: float | None) -> str | None:
    """Why the log or the labels do not fit the video, or None."""
    if duration is None:
        return f"could not read how long {clip} is (needs ffprobe: brew install ffmpeg)"
    first, last = rows[0]["t"], rows[-1]["t"]
    if first < -LOG_SLACK_S or last > duration + LOG_SLACK_S:
        return (f"the log runs from {first:g} s to {last:g} s but the video is {duration:g} s "
                "long: is it this clip's log?")
    if ranges and ranges[-1][1] > duration + LOG_SLACK_S:
        return (f"the labels run to {ranges[-1][1]:g} s but the video is {duration:g} s long: "
                "are they this clip's?")
    return None


def cmd_playground_clip(args) -> int:
    from backend.eval.metrics import read_labels
    from backend.playground import data
    from backend.release import faces

    if not data.NAME.fullmatch(args.name):
        print("te: --name takes letters, digits, _ and -, starting with a letter or digit "
              "(at most 80)", file=sys.stderr)
        return 2
    root = data.clips_root(load_settings().data_dir)
    folder, clip = root / args.name, Path(args.clip)
    if folder.exists():
        print(f"te: {shown_path(folder)} already exists: pick another --name", file=sys.stderr)
        return 2
    if not clip.is_file():
        print(f"te: no clip at {clip}", file=sys.stderr)
        return 2
    rows = data.frame_rows(Path(args.log))
    if not rows:
        print(f"te: no scored frames in {args.log} (te watch --log writes them)", file=sys.stderr)
        return 2
    try:
        ranges = read_labels(Path(args.labels)) if args.labels else []
    except (OSError, ValueError, KeyError, TypeError) as err:
        print(f"te: {args.labels}: not a labels file from te label ({err})", file=sys.stderr)
        return 2
    video, record, problem = "clip.mp4", None, None
    if args.blurred:  # te blur's output: its blur record must name this very file
        video = f"clip{clip.suffix.lower()}"
        record = data.read_json(Path(f"{clip}{data.BLUR_RECORD}"))
        sha = record.get("output_sha256") if isinstance(record, dict) else None
        if video not in data.VIDEOS:
            problem = f"{clip.name}: the playground plays .mp4, .webm or .mov"
        elif not sha or sha != data.file_sha256(clip):
            problem = (f"no blur record that matches {clip} ({clip.name}{data.BLUR_RECORD}): "
                       "leave out --blurred and this command blurs it")
    problem = problem or clip_problem(clip, rows, ranges, cli.audio_duration_s(clip))
    if problem:
        print(f"te: {problem}", file=sys.stderr)
        return 2
    root.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{args.name}-", dir=root))  # not listed: a dot
    try:
        if args.blurred:
            shutil.copyfile(clip, partial / video)
            report = record
        else:
            report = faces.blur_video(clip, partial / video, fps=args.fps,
                                      detector=faces.detect_faces)
            own = data.read_json(partial / f"{video}{data.BLUR_RECORD}")  # if te blur wrote one
            report = {**(own if isinstance(own, dict) else {}),
                      "source_sha256": data.file_sha256(clip), "fps": args.fps, **report,
                      "output_sha256": data.file_sha256(partial / video),
                      "by": "te playground-clip"}
        (partial / f"{video}{data.BLUR_RECORD}").write_text(json.dumps(report, indent=1) + "\n")
        shutil.copyfile(args.log, partial / "log.jsonl")
        if args.labels:
            shutil.copyfile(args.labels, partial / "labels.json")
        partial.rename(folder)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as err:
        shutil.rmtree(partial, ignore_errors=True)  # this command's own half-made folder
        print(f"te: {err}", file=sys.stderr)
        return 2
    skills = ", ".join(sorted({r["skill"] for r in rows}))
    blurred = ("copied as blurred by te blur" if args.blurred else
               f"{report['faces']} faces blurred in {report['frames_with_faces']} frames")
    print(f"{args.name}: {len(rows)} scored frames ({skills}), {blurred} -> "
          f"{shown_path(folder)} (audio left out)")
    print("look at every frame before publishing: the detector misses some faces")
    return 0


def add_parsers(sub) -> None:
    play = sub.add_parser("playground", help="run the web playground")
    play.add_argument("--host", default="127.0.0.1")
    play.add_argument("--port", type=int, default=8100)
    play.add_argument("--eval-dir", default="eval", help="*.svg and results.md from te eval")
    play.add_argument("--public", action="store_true",
                      help="the public copy: no generated images, only blurred keyframes")
    play.add_argument("--deploy", action="store_true",
                      help="publish the public copy on Modal (asks first)")
    play.add_argument("--total-cap", type=float,
                      help="--deploy: USD in all for live teaching (default TE_SKILL_CAP_USD's "
                           f"${config.Settings.skill_cap_usd:.0f})")
    play.add_argument("--yes", action="store_true", help="--deploy: the author's OK")
    play.add_argument("--visitor-cap", type=float, default=0.50, help="USD per visitor a day")
    play.add_argument("--daily-cap", type=float, default=10.0, help="USD a day for everyone")
    play.add_argument("--forwarded-allow-ips", default="127.0.0.1",
                      help="the proxies trusted to give the visitor's address, comma "
                           'separated (never "*")')
    play.set_defaults(func=cmd_playground)

    pclip = sub.add_parser("playground-clip", help="add a test clip to the playground, blurred")
    pclip.add_argument("--clip", required=True, help="the test clip (a video)")
    pclip.add_argument("--log", required=True, help="its frame log from te watch")
    pclip.add_argument("--labels", help="its labels from te label (<clip>.labels.json)")
    pclip.add_argument("--name", required=True, help="the clip's name on the page")
    pclip.add_argument("--blurred", action="store_true",
                       help="the clip is a te blur output with its blur record: copy it")
    pclip.add_argument("--fps", type=float, default=30.0, help="frames per second, blurred")
    pclip.set_defaults(func=cmd_playground_clip)
