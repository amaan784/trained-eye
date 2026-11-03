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

te playground and te playground-clip, the web playground's commands, are in
cli_playground.py.
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
    from backend.release.package import prepare_release, upload
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
    from backend.release.package import upload
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
    from backend.release.package import collection_item, make_collection

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
