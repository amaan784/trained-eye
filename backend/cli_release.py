"""te commands for M4's release (registered by backend/cli.py).

  te manifests      copy the prompt manifests out of data/ into a committed folder
  te regenerate     rebuild a skill's generated images from its manifest (asks first)
  te blur           blur faces in an image folder or a video before publishing
                    (--boxes: faces the detector missed; a video gets a blur record)
  te fetch-weights  download a run's adapter or merged files from the Modal volume
  te release        assemble the Hugging Face release folder; --upload asks first
  te release-test-set  the blurred, human-labelled session 3 clips as a dataset folder;
                    --upload asks first (a private dataset repo)
  te release-collection  group the release repos into a Hugging Face collection (asks)
  te wandb-report   save the W&B report "Trained Eye: experiments" as a draft (asks)
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
    from backend.release import faces

    src, dst = Path(args.src), Path(args.dst)
    try:
        raw = json.loads(Path(args.boxes).read_text(encoding="utf-8")) if args.boxes else None
        if src.is_dir():
            extra = faces.image_boxes(raw) if raw is not None else None
            report = faces.blur_folder(src, dst, detector=faces.detect_faces, extra=extra)
            print(f"{report['images']} images, {report['faces']} faces found and "
                  f"{report['manual_boxes']} manual boxes blurred -> {shown_path(dst)}")
        else:
            extra = faces.video_boxes(raw) if raw is not None else None
            report = faces.blur_video(src, dst, fps=args.fps, detector=faces.detect_faces,
                                      extra=extra)
            print(f"{report['frames']} frames, {report['faces']} faces in "
                  f"{report['frames_with_faces']} frames and {len(report['manual_boxes'])} "
                  f"manual boxes on {report['frames_with_manual_boxes']} frames blurred -> "
                  f"{shown_path(dst)} (audio left out); blur record: "
                  f"{shown_path(Path(f'{dst}{faces.BLUR_RECORD}'))}")
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.CalledProcessError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    print("look at every frame before publishing: the detector misses some faces")
    return 0


def pairs(values: list[str] | None, flag: str) -> dict[str, Path]:
    """{skill: folder} from repeated `--flag <skill>=<folder>`."""
    from backend.skills.spec import check_name

    out: dict[str, Path] = {}
    for value in values or []:
        skill, sep, folder = value.partition("=")
        if not sep or not folder or skill in out:
            raise ValueError(f"{flag} takes <skill>=<folder>, once per skill: {value!r}")
        out[check_name(skill)] = Path(folder)
    return out


def cmd_fetch_weights(args) -> int:
    from backend.release.weights import VOLUME, default_dir, fetch_weights

    out = Path(args.out) if args.out else default_dir(load_settings().data_dir, args.run,
                                                       args.kind)
    try:
        report = fetch_weights(args.run, out, args.kind)
    except (OSError, RuntimeError, ValueError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    print(f"{len(report['files'])} files, {report['bytes'] / 1e6:.1f} MB from the Modal volume "
          f"{VOLUME} {report['remote']} -> {shown_path(out)} (left out: "
          f"{', '.join(report['skipped']) or 'nothing'})")
    print(f"next: te release --run {args.run} --kind {args.kind} --out <folder>"
          + (f" --weights {shown_path(out)}" if args.out else ""))
    return 0


def cmd_release(args) -> int:
    from backend.release.model_card import prepare_release, upload
    from backend.release.weights import default_dir

    settings = load_settings()
    out = Path(args.out)
    result = settings.data_dir / "train" / f"{args.run}.result.json"  # written by te train
    steps = json.loads(result.read_text()).get("steps") if result.is_file() else None
    weights = Path(args.weights) if args.weights else default_dir(settings.data_dir, args.run,
                                                                  args.kind)
    if not args.weights and not weights.is_dir():
        print(f"note: no weights at {shown_path(weights)} (te fetch-weights --run {args.run} "
              f"--kind {args.kind}), so the folder cannot be uploaded yet")
        weights = None
    try:
        release = prepare_release(
            out, run=args.run, bundle_manifest=settings.data_dir / "train" / args.run /
            "manifest.json",
            manifests_dir=Path(args.manifests or config.PROJECT_ROOT / "manifests"),
            weights_dir=weights, keyframes=pairs(args.keyframes, "--keyframes"), kind=args.kind,
            results=Path(args.results) if args.results else None, license_id=args.license,
            models=settings.models, max_steps=steps)
    except (OSError, ValueError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    print(f"release folder: {shown_path(out)} ({len(release['files'])} files; not included: "
          f"{', '.join(release['skipped']) or 'nothing'})")
    for skill, n in release["keyframes"].items():
        print(f"{skill}: {n} face-blurred keyframes for te regenerate; look at each before "
              "publishing")
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


def cmd_release_test_set(args) -> int:
    from backend.release.model_card import upload
    from backend.release.test_set import prepare_test_set

    settings = load_settings()
    out = Path(args.out)
    try:
        release = prepare_test_set(
            out, pairs(args.clips, "--clips"), license_id=args.license or "",
            labels_root=Path(args.labels_root) if args.labels_root else settings.data_dir
            / "clips")
    except (OSError, ValueError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    for name, why in release["skipped"]:
        print(f"left out {name}: {why}")
    clips = [c["path"] for c in release["clips"]]
    print(f"test-set folder: {shown_path(out)} ({len(clips)} clips, license {args.license}): "
          + ", ".join(clips))
    print("look at every frame of every clip before publishing: the detector misses some faces")
    if not args.upload:
        return 0
    if not args.repo:
        print("te: --upload needs --repo owner/name", file=sys.stderr)
        return 2
    if not (args.yes or cli.ask(f"Publish these {len(clips)} clips to "
                                f"huggingface.co/datasets/{args.repo} (private)?")):
        print("nothing was uploaded")
        return 1
    try:
        url = upload(out, args.repo, confirmed=True, private=True)
    except (RuntimeError, ValueError) as err:  # MissingSecret is a RuntimeError
        print(f"te: {err}", file=sys.stderr)
        return 2
    print(f"uploaded (private): {url}")
    return 0


def cmd_release_collection(args) -> int:
    from backend.release.model_card import collection_item, make_collection

    try:
        items = [collection_item(spec) for spec in args.repos]
    except ValueError as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    visibility = "public" if args.public else "private"
    print(f"collection {args.title!r} ({visibility}): "
          + ", ".join(f"{repo} ({kind})" for kind, repo in items))
    if not (args.yes or cli.ask(f"Make this {visibility} Hugging Face collection?")):
        print("nothing was made")
        return 1
    try:
        url = make_collection(args.title, args.repos, confirmed=True, private=not args.public,
                              namespace=args.namespace, description=args.description)
    except (RuntimeError, ValueError) as err:  # MissingSecret is a RuntimeError
        print(f"te: {err}", file=sys.stderr)
        return 2
    print(f"collection ({visibility}): {url}")
    return 0


def cmd_wandb_report(args) -> int:
    from backend.eval import wandb_report

    folder = config.PROJECT_ROOT / args.eval_dir
    try:
        results = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        markdown = (folder / "results.md").read_text(encoding="utf-8")
        spec = wandb_report.plan(results, markdown, args.entity)
    except FileNotFoundError as err:
        print(f"te: {err.filename} is missing: make eval writes it", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    print(f"W&B report {spec['title']!r} in {spec['entity']}/{spec['project']}: "
          f"{len(spec['sections'])} sections, the eval run {spec['eval_run']['url']}; saved "
          "as a draft, which only you can see")
    if not (args.yes or cli.ask("Save the draft report on W&B?")):
        print("nothing was saved")
        return 1
    try:
        url = wandb_report.save_report(spec, confirmed=True)
    except RuntimeError as err:
        print(f"te: {err}", file=sys.stderr)
        return 2
    print(f"draft report: {url}")
    print("publishing it is a separate step in the W&B UI: ask the author first")
    return 0


PLAYGROUND_APP = "trained-eye-playground"  # backend/playground/modal_app.py's APP
PLAYGROUND_VOLUME = "trained-eye-playground"  # and its VOLUME
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
    with tempfile.TemporaryDirectory() as tmp:
        stage(picked["files"], Path(tmp))
        for part in ("data", "eval"):
            if (Path(tmp) / part).is_dir() and (code := modal(
                    "volume", "put", "--force", PLAYGROUND_VOLUME, str(Path(tmp) / part),
                    f"/{part}")):
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
    blur.add_argument("--boxes", help='JSON of boxes the detector missed: for a folder '
                      '{"<file>": [[left, top, right, bottom], ...]}, for a video '
                      '[{"start_s": .., "end_s": .., "box": [left, top, right, bottom]}, ...]')
    blur.set_defaults(func=cmd_blur)

    fetch = sub.add_parser("fetch-weights", help="download a run's weights from Modal")
    fetch.add_argument("--run", required=True, help="the training run, e.g. final")
    fetch.add_argument("--kind", choices=["adapter", "merged"], default="adapter",
                       help="merged: the folder te export writes")
    fetch.add_argument("--out", help="default: data/weights/<run>/<kind>, where te release "
                       "looks")
    fetch.set_defaults(func=cmd_fetch_weights)

    release = sub.add_parser("release", help="assemble the Hugging Face release folder")
    release.add_argument("--run", required=True, help="the training run, e.g. final")
    release.add_argument("--out", required=True, help="the release folder to write")
    release.add_argument("--kind", choices=["adapter", "merged"], default="adapter")
    release.add_argument("--weights", help="folder with the adapter or merged model files "
                         "(default: data/weights/<run>/<kind> from te fetch-weights)")
    release.add_argument("--keyframes", action="append", metavar="SKILL=FOLDER",
                         help="a skill's face-blurred keyframes (a te blur output folder), "
                              "for te regenerate; once per skill")
    release.add_argument("--manifests", help="default: manifests/ in the project")
    release.add_argument("--results", help="results.json from te eval")
    release.add_argument("--license", default="apache-2.0",
                         help="Hugging Face license id (default: the base model's, apache-2.0)")
    release.add_argument("--upload", action="store_true", help="publish (asks first)")
    release.add_argument("--repo", help="owner/name on Hugging Face")
    release.add_argument("--public", action="store_true", help="a public repo (default private)")
    release.add_argument("--yes", action="store_true", help="the author's OK, no question")
    release.set_defaults(func=cmd_release)

    tests = sub.add_parser("release-test-set", help="the labelled test clips as a dataset")
    tests.add_argument("--clips", action="append", required=True, metavar="SKILL=FOLDER",
                       help="a te blur output folder of the skill's session 3 test clips; "
                            "once per skill")
    tests.add_argument("--out", required=True, help="the dataset folder to write")
    tests.add_argument("--license", help="Hugging Face license id the author chose (needed: "
                       "no default)")
    tests.add_argument("--labels-root", help="where the original clips and their labels are "
                       "(default: data/clips, with <skill>/<clip>.labels.json)")
    tests.add_argument("--upload", action="store_true", help="publish, private (asks first)")
    tests.add_argument("--repo", help="owner/name of the dataset repo on Hugging Face")
    tests.add_argument("--yes", action="store_true", help="the author's OK, no question")
    tests.set_defaults(func=cmd_release_test_set)

    coll = sub.add_parser("release-collection", help="group release repos in a collection")
    coll.add_argument("--title", required=True, help='e.g. "Trained Eye"')
    coll.add_argument("--repos", nargs="+", required=True,
                      help="owner/name (a model repo) or datasets/owner/name")
    coll.add_argument("--namespace", help="user or organisation (default: the token's user)")
    coll.add_argument("--description")
    coll.add_argument("--public", action="store_true", help="a public collection (default "
                      "private); every repo must then be public")
    coll.add_argument("--yes", action="store_true", help="the author's OK, no question")
    coll.set_defaults(func=cmd_release_collection)

    report = sub.add_parser("wandb-report", help="the W&B report of the experiments (asks)")
    report.add_argument("--eval-dir", default="eval", help="results.json and results.md of "
                        "te eval --wandb")
    report.add_argument("--entity", help="W&B entity (default: the eval run's)")
    report.add_argument("--yes", action="store_true", help="the author's OK, no question")
    report.set_defaults(func=cmd_wandb_report)

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
