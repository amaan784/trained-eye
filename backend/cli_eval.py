"""te eval and te label (registered by backend/cli.py): score replayed, labelled clips into
results.md and the charts, and the local labelling tool for clips and generated images.
"""

from __future__ import annotations

import sys
from pathlib import Path

from backend.cli import PROJECT_ROOT, count, load_settings, shown_path


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
            write_readme(Path(args.readme), Path(args.out) / "results.json",
                         section=args.readme_section)
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


def add_parsers(sub) -> None:
    ev = sub.add_parser("eval", help="score replayed, labelled clips: results.md and charts")
    ev.add_argument("--plan", required=True, help="the evaluation plan (backend/eval/report.py)")
    ev.add_argument("--out", required=True, help="folder for results.json, results.md, charts")
    ev.add_argument("--wandb", action="store_true", help="also log to W&B (needs wandb)")
    ev.add_argument("--wandb-entity", help="with --wandb: the W&B team (default: your own "
                    "account)")
    ev.add_argument("--readme", help="also write the results block between the te eval "
                    "markers in this file, e.g. README.md (backend/eval/readme.py)")
    ev.add_argument("--readme-section", help="with --readme: a second block under its own "
                    "markers and this heading, e.g. \"Results on EPIC-KITCHENS-100\"")
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
