"""The README art, rendered from HTML with headless Chrome in the contact-sheet design
(DESIGN.md: real frames on black film, edge print, one grease-pencil ring) and filled only
from real run outputs:

    python -m backend.art.build                       # every part whose facts exist
    python -m backend.art.build --only strip,models   # some of them
    python -m backend.art.build --only banner --replay-clip <te playground-clip name>

banner.png      the name, the promise, a film strip of a real replay of the author's own
                footage, edge-printed with each frame's time and score, the ring on the
                frame where the alert fired, the spoken line, one result sentence from te
                eval (eval/results.json, written with results.md) and the models it uses
strip.png       "How it learned": 12 teaching frames as a two-row contact sheet with SAM's
                boxes (roi.json), and the four steps with the teach run's numbers
models          the README's table of Meta models, each with its job and what it did in
                this run, written between the README's "art: models" markers
hero.gif/.mp4   the playground's live timeline replaying a real clip until the alert fires
                (a te playground-clip folder: its video is already blurred)

A part whose numbers or pictures do not exist yet is skipped with the reason, never drawn
with a placeholder. Every teaching frame goes through te blur's face blurring first, and
replays use te blur's video. Generated images and EPIC-KITCHENS frames are never drawn.
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
from backend.art import film
from backend.art.render import RenderError, fill, gif, mp4, screenshot
from backend.config import PROJECT_ROOT, load_settings

FONT_DIR = PROJECT_ROOT / "backend" / "playground" / "static" / "fonts"
TIMELINE_JS = PROJECT_ROOT / "backend" / "playground" / "static" / "timeline.js"
README = PROJECT_ROOT / "README.md"
PARTS = ("banner", "strip", "models", "hero")
HERO_SECONDS, HERO_FPS, HERO_MAX_BYTES = 10.0, 10, 8_000_000
MODELS_START = ("<!-- art: models start (written by python -m backend.art.build; edits here "
                "are lost) -->")
MODELS_END = "<!-- art: models end -->"
SHEET = 12  # teaching frames on the strip, in two rows
BANNER_BEFORE, BANNER_AFTER = 4, 3  # replay frames either side of the alert frame


def css() -> str:
    import string

    return string.Template((F_TEMPLATES / "base.css").read_text()).substitute(
        font_dir=FONT_DIR.resolve().as_uri())


F_TEMPLATES = Path(__file__).with_name("templates")


def blurred_copy(src: Path, dst: Path) -> tuple[Path, int]:
    """`src` with every face te blur finds blurred (and its metadata dropped), and how many
    it found."""
    from backend.release.faces import blur_image

    data, found = blur_image(Path(src).read_bytes())
    dst.write_bytes(data)
    return dst, found


def _and(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def pick_teaching(frames: list[F.TeachFrame], work: Path, n: int = SHEET):
    """`n` frames spread over the teaching clip, blurred. Where te blur's detector marks
    something on a picked frame, a near neighbour is used when that one is clean."""
    step = len(frames) / n
    picked, used = [], set()
    for k in range(min(n, len(frames))):
        i = int(k * step)
        options = [j for j in (i, i + 1, i - 1, i + 2) if 0 <= j < len(frames) and j not in used]
        best = None
        for j in options:
            path, found = blurred_copy(frames[j].path, work / f"teach_{j:02d}.jpg")
            if best is None:
                best = (j, path)
            if not found:
                best = (j, path)
                break
        used.add(best[0])
        picked.append((frames[best[0]], best[1]))
    return sorted(picked, key=lambda x: x[0].t_s)


def strip(sk: F.SkillFacts, frames: list[F.TeachFrame], out: Path, work: Path) -> Path:
    picked = pick_teaching(frames, work)
    times = [f.t_s for f, _ in picked]
    tenths = len({int(t) for t in times}) < len(times)
    cells = [film.Cell(path.as_uri(), film.edge_time(f.t_s, tenths), f.size, f.box)
             for f, path in picked]
    half = (len(cells) + 1) // 2
    geo = {"fw": 96, "fh": 128, "gap": 8, "rebate": 26}  # room for the edge print
    rows = (film.strip(cells[:half], left=64, top=132, sam=True, **geo)
            + film.strip(cells[half:], left=64, top=132 + 180 + 18, sam=True, **geo))
    steps = [f"<b>{html.escape(x['lead'])}</b>"
             + (f" <q>{html.escape(x['quote'])}</q>" if x["quote"] else "")
             + html.escape(x["rest"]) for x in F.teach_steps(sk)]
    page = fill("strip.html", {
        "css": css(), "title": f"How it learned this {sk.skill.split('_')[0]}",
        "strips": rows, "steps": "".join(f"<li>{x}</li>" for x in steps),
        "note": (f"{len(cells)} of the {len(frames)} frames kept from the teaching clip, "
                 "faces blurred, with SAM's boxes in red.")})
    return screenshot(page, out, (1280, 560), folder=work)


def banner_times(rp: F.Replay) -> list[tuple[float, float]]:
    """Scored frames of the replay: a few before the alert, the alert frame, a few after."""
    points = sorted(rp.points)
    if not points:
        raise F.MissingFact(f"replay {rp.clip} has no scored frames")
    step = max(1.0, rp.alert_t / (BANNER_BEFORE + 1))
    wanted = ([rp.alert_t - step * k for k in range(BANNER_BEFORE, 0, -1)] + [rp.alert_t]
              + [rp.alert_t + step * k for k in range(1, BANNER_AFTER + 1)])
    end = points[-1][0]
    chosen = []
    for t in wanted:
        if t < 0 or t > end:
            continue
        near = min(points, key=lambda pt: abs(pt[0] - t))
        if near not in chosen:
            chosen.append(near)
    return chosen


def still(video: Path, t: float, dst: Path) -> Path:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i",
                    str(video), "-frames:v", "1", "-q:v", "3", str(dst)], check=True)
    if not dst.is_file():
        raise F.MissingFact(f"no frame at {t:.1f} s in {video}")
    return dst


def banner(sk: F.SkillFacts, head: F.Headline, rp: F.Replay, out: Path, work: Path) -> Path:
    from PIL import Image

    shots = banner_times(rp)
    alert_i = min(range(len(shots)), key=lambda i: abs(shots[i][0] - rp.alert_t))
    cells = []
    for i, (t, p) in enumerate(shots):
        path = still(rp.video, t, work / f"replay_{i:02d}.jpg")
        with Image.open(path) as im:
            size = im.size
        cells.append(film.Cell(path.as_uri(), f"{film.edge_time(t)} {p:.2f}", size))
    fw, fh, gap, rebate, top, pad = 156, 208, 12, 36, 236, -40
    start = pad + (BANNER_BEFORE - alert_i) * (fw + gap)  # the alert frame stays in place
    strip_html = film.strip(cells, left=0, top=top, fw=fw, fh=fh, gap=gap, rebate=rebate,
                            pad=start, width=1280, hole_every=22)
    cx = start + alert_i * (fw + gap) + fw / 2
    height = rebate * 2 + fh
    cy = top + height / 2
    lead = (cx + fw * 0.68 * 0.5 + 40, top - 64)
    ring = film.ring(cx, cy, fw * 0.68, height * 0.55, lead, (1280, 640))
    data_models = [m.name for m in sk.models if not m.highlight]
    student = next(m.name for m in sk.models if m.highlight)
    page = fill("banner.html", {
        "css": css(), "strip": strip_html, "ring": ring,
        "note_left": f"{lead[0] + 12:.0f}", "note_top": f"{lead[1] - 40:.0f}",
        "alert": html.escape(rp.alert_text),
        "when": f"Spoken on the glasses once this has lasted {rp.hold_s:g} seconds.",
        "headline": html.escape(head.text), "detail": html.escape(head.detail),
        "credits": (f"Built on {html.escape(_and(data_models))}, with a fine-tuned "
                    f"{html.escape(student)} watching.<br>Meta Global AI Developer Hackathon "
                    "2026.")})
    return screenshot(page, out, (1280, 640), folder=work)


def models_markdown(sk: F.SkillFacts) -> str:
    """The README's table of Meta models: each one's job and what it did in this run."""
    rows = ["| Model | Its job | In this run |", "|---|---|---|"]
    for m in sk.models:
        name = (f"**{m.name}** (`{m.model_id}`), fine-tuned" if m.highlight
                else f"{m.name} (`{m.model_id}`)")
        rows.append(f"| {name} | {m.what} | {m.count} |")
    return "\n".join(["## Meta models, each with a job", "", *rows])


def write_models(readme: Path, sk: F.SkillFacts) -> Path:
    text = readme.read_text(encoding="utf-8")
    if MODELS_START not in text or MODELS_END not in text:
        raise F.MissingFact(f"{readme} has no art: models markers")
    head, rest = text.split(MODELS_START, 1)
    _, tail = rest.split(MODELS_END, 1)
    readme.write_text(f"{head}{MODELS_START}\n{models_markdown(sk)}\n{MODELS_END}{tail}",
                      encoding="utf-8")
    return readme


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
    p.add_argument("--replay-clip", help="the banner's replay: a te playground-clip name "
                   "of the author's own footage in which the alert fires")
    p.add_argument("--hero-clip", help="a playground clip name (te playground-clip)")
    p.add_argument("--readme", default=str(README), help="the README the models table goes in")
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
                    frames = F.teaching_frames(settings.data_dir, args.skill)
                    made.append(strip(sk, frames, out / "strip.png", work))
                elif part == "models":
                    made.append(write_models(Path(args.readme), sk))
                elif part == "banner":
                    if not args.replay_clip:
                        raise F.MissingFact("no --replay-clip given")
                    head = F.headline(Path(args.results))
                    rp = F.replay(settings.data_dir, args.replay_clip, args.skill)
                    made.append(banner(sk, head, rp, out / "banner.png", work))
                elif part == "hero":
                    if not args.hero_clip:
                        raise F.MissingFact("no --hero-clip given")
                    made.append(hero(settings.data_dir, args.hero_clip, args.skill,
                                     out / "hero.gif", video_out / "hero.mp4", work))
                else:
                    raise F.MissingFact(f"unknown part {part!r}")
            except F.MissingFact as err:
                skipped.append(f"{part}: {err}")
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
