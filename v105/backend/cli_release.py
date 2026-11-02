"""te commands for M4's release (registered by backend/cli.py).

  te manifests      copy the prompt manifests out of data/ into a committed folder
  te regenerate     rebuild a skill's generated images from its manifest (asks first)
  te blur           blur faces in an image folder or a video before publishing
  te release        assemble the Hugging Face release folder; --upload asks first
  te playground     run the web playground locally
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from backend import cli, config
from backend.cli import FAILURES_HINT, RESUME_HINT
from backend.config import load_settings, shown_path


def cmd_manifests(args) -> int:
    from backend.regenerate import export_manifests

    settings = load_settings()
    try:
        written = export_manifests(settings.data_dir,
                                   Path(args.out or config.PROJECT_ROOT / "manifests"))
    except (OSError, ValueError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    if not written:
        print("te: no skill has generated images yet (te teach makes them)", file=sys.stderr)
        return 1
    for path in written:
        print(f"{shown_path(path)}: {len(path.read_text().splitlines())} prompts")
    print("commit these; the images themselves stay in data/ and are never committed")
    return 0


def cmd_regenerate(args) -> int:
    from backend.dataengine.synth import TooManyFailures
    from backend.regenerate import regenerate
    from backend.skills.spec import check_name

    settings = load_settings()
    try:
        skill = check_name(args.skill)
    except ValueError as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    root = settings.data_dir / "skills" / skill
    manifest = Path(args.manifest or config.PROJECT_ROOT / "manifests" / f"{skill}.jsonl")
    out = Path(args.out or root / "regenerated")
    spent = []

    def confirm(plan: dict) -> bool:
        print(f"{plan['planned']} images to regenerate from {plan['keyframes']} keyframes "
              f"({plan['skipped_failed_originally']} failed originally and are skipped); "
              f"estimate before cache hits: ${plan['estimate_usd']:.2f}")
        if plan["mapped_keyframes"]:
            print("note: face-blurred keyframes, so the images will differ from the manifest's")
        if plan["manifest_models"] not in ([], [plan["model"]]):
            print(f"note: made with {', '.join(plan['manifest_models'])}, now {plan['model']}")
        if args.estimate_only:
            return False
        settings.secret("MODEL_API_KEY")  # a missing key stops here, before the question
        over = plan["estimate_usd"] > config.ASK_ABOVE_USD
        if over:
            print(f"te: this is over ${config.ASK_ABOVE_USD:.0f}, so it needs your yes even "
                  "with --yes")
        return (args.yes and not over) or cli.ask(
            f"Spend up to about ${plan['estimate_usd']:.2f}?")

    def progress(cached: bool, cost_usd: float) -> None:
        spent.append(cost_usd)
        if len(spent) % 10 == 0:
            print(f"  images {len(spent)}  ${sum(spent):.2f} spent", flush=True)

    try:
        report = regenerate(cli.make_client(settings), manifest,
                            Path(args.keyframes or root / "keyframes"), out, progress,
                            mapping=Path(args.map) if args.map else None, confirm=confirm)
    except KeyboardInterrupt:
        print(f"\nte: stopped; {RESUME_HINT}", file=sys.stderr)
        return 130
    except TooManyFailures as err:
        print(f"te: {err}\nte: regenerate stopped; {FAILURES_HINT}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    if report.get("stopped"):
        return 0 if args.estimate_only else 1
    print(f"regenerated {report['generated']} of {report['planned']} images -> "
          f"{shown_path(out)} ({report['cached']} from the cache, "
          f"${report['spent_usd']:.2f} spent)")
    print(f"{report['matching']} match the manifest's image hashes, {report['differing']} differ"
          + (" (Muse Image may not be deterministic: new calls give new images)"
             if report["differing"] else ""))
    return 0


def cmd_blur(args) -> int:
    from backend.release.faces import blur_folder, blur_video

    src, dst = Path(args.src), Path(args.dst)
    try:
        if src.is_dir():
            report = blur_folder(src, dst)
            print(f"{report['images']} images, {report['faces']} faces blurred -> "
                  f"{shown_path(dst)}")
        else:
            report = blur_video(src, dst, fps=args.fps)
            print(f"{report['frames']} frames, {report['faces']} faces in "
                  f"{report['frames_with_faces']} frames blurred -> {shown_path(dst)} "
                  "(audio left out)")
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    print("look at every frame before publishing: the detector misses some faces")
    return 0


def cmd_release(args) -> int:
    from backend.release.model_card import prepare_release, upload

    settings = load_settings()
    out = Path(args.out)
    result = settings.data_dir / "train" / f"{args.run}.result.json"  # written by te train
    steps = json.loads(result.read_text()).get("steps") if result.is_file() else None
    try:
        release = prepare_release(
            out, run=args.run, bundle_manifest=settings.data_dir / "train" / args.run /
            "manifest.json", manifests_dir=Path(args.manifests or config.PROJECT_ROOT / "manifests"),
            weights_dir=Path(args.weights) if args.weights else None, kind=args.kind,
            results=Path(args.results) if args.results else None, license_id=args.license,
            models=settings.models, max_steps=steps)
    except (OSError, ValueError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    print(f"release folder: {shown_path(out)} ({len(release['files'])} files; not included: "
          f"{', '.join(release['skipped']) or 'nothing'})")
    if not args.license:
        print("no license yet: check the Glimmer model card, then run again with --license")
    if not args.upload:
        return 0
    if not args.repo:
        print("te: --upload needs --repo owner/name", file=sys.stderr)
        return 2
    visibility = "public" if args.public else "private"
    if not (args.yes or cli.ask(f"Publish {shown_path(out)} to huggingface.co/{args.repo} "
                            f"({visibility})?")):
        print("nothing was uploaded")
        return 1
    try:
        url = upload(out, args.repo, confirmed=True, private=not args.public)
    except (RuntimeError, ValueError) as err:  # MissingSecret is a RuntimeError
        print(f"te: {err}", file=sys.stderr)
        return 2
    print(f"uploaded ({visibility}): {url}")
    return 0


def cmd_playground(args) -> int:
    import uvicorn

    from backend.playground.app import create_app

    if "*" in [ip.strip() for ip in args.forwarded_allow_ips.split(",")]:
        print('te: --forwarded-allow-ips "*" would let any visitor pick their own address '
              "and reset their spend cap: give the proxy's address", file=sys.stderr)
        return 2
    settings = load_settings()
    app = create_app(settings.data_dir, config.PROJECT_ROOT / args.eval_dir,
                     lambda: cli.make_client(load_settings()),
                     visitor_cap_usd=args.visitor_cap, daily_cap_usd=args.daily_cap)
    live = "on" if settings.has_secret("MODEL_API_KEY") else "off (MODEL_API_KEY is not set)"
    print(f"playground: http://{args.host}:{args.port}/  live teaching {live}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", proxy_headers=True,
                forwarded_allow_ips=args.forwarded_allow_ips)
    return 0


def add_parsers(sub) -> None:
    manifests = sub.add_parser("manifests", help="copy the prompt manifests out of data/")
    manifests.add_argument("--out", help="default: manifests/ in the project")
    manifests.set_defaults(func=cmd_manifests)

    regen = sub.add_parser("regenerate", help="rebuild a skill's generated images")
    regen.add_argument("--skill", required=True)
    regen.add_argument("--manifest", help="default: manifests/<skill>.jsonl")
    regen.add_argument("--keyframes", help="default: the skill's original keyframes")
    regen.add_argument("--map", help="sources.json from te blur, for face-blurred keyframes")
    regen.add_argument("--out", help="default: data/skills/<skill>/regenerated")
    regen.add_argument("--estimate-only", action="store_true", help="print the estimate")
    regen.add_argument("--yes", action="store_true", help="no question")
    regen.set_defaults(func=cmd_regenerate)

    blur = sub.add_parser("blur", help="blur faces in an image folder or a video")
    blur.add_argument("--src", required=True, help="a folder of images or a video file")
    blur.add_argument("--dst", required=True, help="another folder, or the output video")
    blur.add_argument("--fps", type=float, default=30.0, help="video only")
    blur.set_defaults(func=cmd_blur)

    release = sub.add_parser("release", help="assemble the Hugging Face release folder")
    release.add_argument("--run", required=True, help="the training run, e.g. final")
    release.add_argument("--out", required=True, help="the release folder to write")
    release.add_argument("--kind", choices=["adapter", "merged"], default="adapter")
    release.add_argument("--weights", help="folder with the adapter or merged model files")
    release.add_argument("--manifests", help="default: manifests/ in the project")
    release.add_argument("--results", help="results.json from te eval")
    release.add_argument("--license", help="Hugging Face license id, after checking the card")
    release.add_argument("--upload", action="store_true", help="publish (asks first)")
    release.add_argument("--repo", help="owner/name on Hugging Face")
    release.add_argument("--public", action="store_true", help="a public repo (default private)")
    release.add_argument("--yes", action="store_true", help="the author's OK, no question")
    release.set_defaults(func=cmd_release)

    play = sub.add_parser("playground", help="run the web playground")
    play.add_argument("--host", default="127.0.0.1")
    play.add_argument("--port", type=int, default=8100)
    play.add_argument("--eval-dir", default="eval", help="clips/, *.svg and results.md")
    play.add_argument("--visitor-cap", type=float, default=0.50, help="USD per visitor a day")
    play.add_argument("--daily-cap", type=float, default=10.0, help="USD a day for everyone")
    play.add_argument("--forwarded-allow-ips", default="127.0.0.1",
                      help="the proxies trusted to give the visitor's address, comma "
                           'separated (never "*")')
    play.set_defaults(func=cmd_playground)
