"""The README art, rendered from HTML with headless Chrome in the app's look and filled
only from real run outputs:

    python -m backend.art.build                       # every image whose facts exist
    python -m backend.art.build --only strip,cards    # some of them

banner.png      the name, the promise, a real frame with the spoken alert, the headline
                result from te eval (eval/results.json, written with results.md) and a
                chip per Meta model the runs used
hero.gif/.mp4   the playground's live timeline replaying a real clip until the alert fires
                (a te playground-clip folder: its video is already blurred)
strip.png       how a skill is built: the teaching frame with SAM's box, then the counts
cards.png       one card per Meta model the runs used, fine-tuned Glimmer highlighted

An image whose numbers or pictures do not exist yet is skipped with the reason, never
drawn with a placeholder. Every real frame goes through te blur's face blurring first.
Generated images are never drawn: the training wall belongs in the video only.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from backend.art import facts as F
from backend.art.render import RenderError, fill, gif, mp4, screenshot
from backend.config import PROJECT_ROOT, load_settings

FONT_DIR = PROJECT_ROOT / "ios" / "TrainedEye" / "Fonts"
TIMELINE_JS = PROJECT_ROOT / "backend" / "playground" / "static" / "timeline.js"
ALERT = "The fridge is open."
PARTS = ("banner", "strip", "cards", "hero")
HERO_SECONDS, HERO_FPS, HERO_MAX_BYTES = 10.0, 10, 8_000_000


def css() -> str:
    import string

    return string.Template((F_TEMPLATES / "base.css").read_text()).substitute(
        font_dir=FONT_DIR.resolve().as_uri())


F_TEMPLATES = Path(__file__).with_name("templates")


def blurred_copy(src: Path, dst: Path) -> Path:
    """`src` with every face te blur finds blurred (and its metadata dropped)."""
    from backend.release.faces import blur_image

    data, _ = blur_image(Path(src).read_bytes())
    dst.write_bytes(data)
    return dst


def cover(size: tuple[int, int], frame: tuple[int, int], focus: tuple[float, float]):
    """Scale and offset (CSS px) for a frame covering a `size` panel, centred on `focus`
    (in frame pixels) as far as the edges allow."""
    s = max(size[0] / frame[0], size[1] / frame[1])
    w, h = frame[0] * s, frame[1] * s
    left = min(0.0, max(size[0] - w, size[0] / 2 - focus[0] * s))
    top = min(0.0, max(size[1] - h, size[1] / 2 - focus[1] * s))
    return s, left, top


def strip(sk: F.SkillFacts, out: Path, work: Path) -> Path:
    frame = blurred_copy(sk.frame, work / "strip-frame.jpg")
    x1, y1, x2, y2 = sk.box
    panel = (248, 330)
    s, left, top = cover(panel, sk.frame_size, ((x1 + x2) / 2, (y1 + y2) / 2))
    box = (f"left:{left + x1 * s:.1f}px;top:{top + y1 * s:.1f}px;"
           f"width:{(x2 - x1) * s:.1f}px;height:{(y2 - y1) * s:.1f}px")
    page = fill("strip.html", {
        "css": css(), "request": html.escape(sk.request), "frame": frame.as_uri(),
        "frame_style": "", "img_style": (f"width:{sk.frame_size[0] * s:.1f}px;"
                                          f"left:{left:.1f}px;top:{top:.1f}px"),
        "box_style": box, "target": html.escape(sk.sam_prompt),
        "boxed": sk.frames_with_box, "keyframes": sk.keyframes,
        "made": sk.made, "kept": sk.kept, "rejected": sk.rejected, "examples": sk.examples,
        "yes": sk.yes, "no": sk.no, "real": sk.real_examples})
    return screenshot(page, out, (1280, 470), folder=work)


def cards(sk: F.SkillFacts, out: Path, work: Path) -> Path:
    items = []
    for m in sk.models:
        badge = '<div class="badge">OPEN WEIGHTS</div>' if m.highlight else ""
        items.append(f'<div class="card{" on" if m.highlight else ""}">{badge}'
                     f'<div class="name">{html.escape(m.name)}</div>'
                     f'<div class="id">{html.escape(m.model_id)}</div>'
                     f'<div class="what">{html.escape(m.what)}</div>'
                     f'<div class="count num">{html.escape(m.count)}</div></div>')
    page = fill("cards.html", {"css": css(), "cards": "".join(items), "height": "420px"})
    return screenshot(page, out, (1280, 420), folder=work)


def banner(sk: F.SkillFacts, head: F.Headline, frame_src: Path, out: Path, work: Path) -> Path:
    frame = blurred_copy(frame_src, work / "banner-frame.jpg")
    chips = "".join(f'<span class="chip{" on" if m.highlight else ""}">'
                    f'{"<span class=dot></span>" if m.highlight else ""}{html.escape(m.name)}'
                    "</span>" for m in sk.models)
    page = fill("banner.html", {
        "css": css(), "headline": html.escape(head.text), "detail": html.escape(head.detail),
        "chips": chips, "frame": frame.as_uri(), "focus": "50% 45%", "alert": ALERT})
    return screenshot(page, out, (1280, 640), folder=work)


def hero(data_dir: Path, clip_name: str, skill: str, out_gif: Path, out_mp4: Path,
         work: Path) -> Path:
    from backend.playground import data as pg

    clip = pg.clip_timeline(data_dir, clip_name)
    if not clip or skill not in clip["skills"]:
        raise F.MissingFact(f"no playground clip {clip_name!r} with a {skill} log "
                            "(te playground-clip makes one)")
    sk = clip["skills"][skill]
    if not sk["alerts"]:
        raise F.MissingFact(f"the alert never fires in playground clip {clip_name!r}")
    video = pg.clips_root(data_dir) / clip_name / Path(clip["video"] or "").name
    if not clip["video"] or not video.is_file():
        raise F.MissingFact(f"playground clip {clip_name!r} has no video")
    end = min(clip["duration"], sk["alerts"][0]["t"] + 2.5)  # until the alert, and a beat
    n = int(HERO_SECONDS * HERO_FPS)
    times = [end * i / (n - 1) for i in range(n)]
    stills = work / "stills"
    stills.mkdir()
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video), "-vf",
                    f"fps={(n - 1) / end:.6f},scale=-2:720", str(stills / "s_%05d.jpg")],
                   check=True)
    shots = sorted(stills.glob("s_*.jpg"))
    frames_dir = work / "frames"
    frames_dir.mkdir()
    frames = []
    for i, t in enumerate(times):
        fired = any(a["t"] <= t for a in sk["alerts"])
        page = fill("hero.html", {
            "css": css(), "frame": shots[min(i, len(shots) - 1)].as_uri(),
            "spoken_hidden": "" if fired else "hidden",
            "alert": html.escape(sk["alerts"][0]["text"]),
            "title": f"Watching for {skill.replace('_', ' ')}",
            "subtitle": ("The fine-tuned student scores one frame a second; the alert fires "
                         "once the condition has held for the skill's hold."),
            "timeline_src": TIMELINE_JS.resolve().as_uri(),
            "clip_json": json.dumps(clip), "t": f"{t:.3f}", "skill": skill})
        frames.append(screenshot(page, frames_dir / f"frame_{i:05d}.png", (1100, 560),
                                 scale=1, folder=work))
    mp4(frames, out_mp4, HERO_FPS)
    return gif(frames, out_gif, HERO_FPS, 960, HERO_MAX_BYTES)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m backend.art.build", description=__doc__.split(
        "\n\n")[0])
    p.add_argument("--skill", default="fridge_left_open")
    p.add_argument("--run", default="exp1-mixed", help="the skill's training run")
    p.add_argument("--results", default=str(PROJECT_ROOT / "eval" / "results.json"))
    p.add_argument("--banner-frame", help="a real first-person frame for the banner "
                   "(default: the teaching frame the strip uses)")
    p.add_argument("--hero-clip", help="a playground clip name (te playground-clip)")
    p.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "img"))
    p.add_argument("--video-out", default=None, help="where the hero MP4 goes (default: "
                   "data/art/, not committed)")
    p.add_argument("--only", default=",".join(PARTS))
    args = p.parse_args(argv)
    settings = load_settings()
    out = Path(args.out)
    video_out = Path(args.video_out or settings.data_dir / "art")
    out.mkdir(parents=True, exist_ok=True)
    video_out.mkdir(parents=True, exist_ok=True)
    wanted = [x.strip() for x in args.only.split(",") if x.strip()]
    made, skipped = [], []
    try:
        sk = F.skill_facts(settings.data_dir, args.skill, args.run)
    except F.MissingFact as err:
        print(f"art: nothing to draw: {err}", file=sys.stderr)
        return 1
    for part in wanted:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            try:
                if part == "strip":
                    made.append(strip(sk, out / "strip.png", work))
                elif part == "cards":
                    made.append(cards(sk, out / "cards.png", work))
                elif part == "banner":
                    head = F.headline(Path(args.results))
                    made.append(banner(sk, head, Path(args.banner_frame or sk.frame),
                                       out / "banner.png", work))
                elif part == "hero":
                    if not args.hero_clip:
                        raise F.MissingFact("no --hero-clip given")
                    made.append(hero(settings.data_dir, args.hero_clip, args.skill,
                                     out / "hero.gif", video_out / "hero.mp4", work))
                else:
                    raise F.MissingFact(f"unknown part {part!r}")
            except F.MissingFact as err:
                skipped.append(f"{part}: {err}")
                for stale in (out / f"{part}.png", out / f"{part}.gif"):
                    stale.unlink(missing_ok=True)  # never leave an older image behind
    for path in made:
        print(f"art: {path}")
    for why in skipped:
        print(f"art: skipped {why}", file=sys.stderr)
    return 0 if made else 1


if __name__ == "__main__":
    shutil.which("ffmpeg") or sys.exit("art: ffmpeg is needed")
    try:
        sys.exit(main())
    except RenderError as err:
        sys.exit(f"art: {err}")
