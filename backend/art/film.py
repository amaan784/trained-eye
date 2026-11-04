"""The contact sheet's pieces as HTML (DESIGN.md): film strips of real frames on black,
sprocket holes in both rebates, edge print in its own band, SAM's boxes as thin marker
rectangles, and the grease-pencil ring around the frame where the real alert fired.

Ported from the design kit's make_banner.py and make_strip.py, which rendered fixed preview
frames with Playwright; here every frame, time and score comes from a run, and the pages
render with the headless Chrome of render.py.
"""

from __future__ import annotations

import html
import math
import random
from dataclasses import dataclass


@dataclass
class Cell:
    src: str  # a file URI of the (face-blurred) frame
    edge: str  # the edge print: its time, and its score where a replay exists
    size: tuple[int, int]  # the frame's pixels
    box: tuple[float, float, float, float] | None = None  # SAM's box, in its pixels


def edge_time(t: float, tenths: bool = False) -> str:
    """0:16, or 0:05.7 when two frames share a second."""
    minutes, seconds = divmod(t, 60)
    return (f"{int(minutes)}:{seconds:04.1f}" if tenths
            else f"{int(minutes)}:{int(seconds):02d}")


def cover(panel: tuple[float, float], size: tuple[int, int]) -> tuple[float, float, float]:
    """Scale and offset of a frame filling `panel` from the centre, as object-fit: cover."""
    s = max(panel[0] / size[0], panel[1] / size[1])
    return s, (panel[0] - size[0] * s) / 2, (panel[1] - size[1] * s) / 2


def holes(width: float, spacing: float, cls: str) -> str:
    return (f'<div class="{cls}">' + "".join(
        f'<span class="hole" style="left:{x:.0f}px"></span>'
        for x in range(6, int(width) - 8, int(spacing))) + "</div>")


def strip(cells: list[Cell], *, left: float, top: float, fw: float, fh: float, gap: float,
          rebate: float, pad: float = 12, width: float | None = None, hole_every: float = 16,
          sam: bool = False) -> str:
    """One strip of film: frames left to right, each with its edge print under it."""
    run = pad * 2 + len(cells) * fw + (len(cells) - 1) * gap
    width = width or run
    parts = []
    for i, c in enumerate(cells):
        x = pad + i * (fw + gap)
        s, ox, oy = cover((fw, fh), c.size)
        box = ""
        if sam and c.box:
            x1, y1, x2, y2 = c.box
            box = (f'<span class="sam" style="left:{ox + x1 * s:.1f}px;top:{oy + y1 * s:.1f}px;'
                   f'width:{(x2 - x1) * s:.1f}px;height:{(y2 - y1) * s:.1f}px"></span>')
        parts.append(f'<figure class="frame" style="left:{x:.1f}px;top:{rebate:.0f}px;'
                     f'width:{fw:.0f}px;height:{fh:.0f}px"><img src="{html.escape(c.src)}" '
                     f'alt="">{box}</figure>'
                     f'<span class="edge" style="left:{x:.1f}px;top:{rebate + fh + 3:.0f}px">'
                     f"{html.escape(c.edge)}</span>")
    height = rebate * 2 + fh
    return (f'<div class="strip" style="left:{left:.0f}px;top:{top:.0f}px;width:{width:.0f}px;'
            f'height:{height:.0f}px">{holes(width, hole_every, "holes top")}{"".join(parts)}'
            f'{holes(width, hole_every, "holes bottom")}</div>')


def hand_ellipse(cx: float, cy: float, rx: float, ry: float, rot_deg: float, seed: int,
                 overshoot: float = 0.18) -> str:
    """An SVG path for an ellipse drawn quickly by hand: it wobbles, opens up as it comes
    round again, and overshoots its start (the design kit's ring)."""
    rnd = random.Random(seed)
    rot = math.radians(rot_deg)
    start = math.radians(rnd.uniform(195, 215))
    sweep = 2 * math.pi * (1 + overshoot)
    wobble = [rnd.uniform(-1, 1) for _ in range(8)]
    n, pts = 90, []
    for i in range(n + 1):
        a = start + sweep * i / n
        r_noise = 1 + 0.035 * sum(w * math.sin((k + 1) * a + k) for k, w in enumerate(wobble)) / 4
        drift = 1 + 0.06 * i / n
        x0, y0 = rx * r_noise * drift * math.cos(a), ry * r_noise * drift * math.sin(a)
        pts.append((cx + x0 * math.cos(rot) - y0 * math.sin(rot),
                    cy + x0 * math.sin(rot) + y0 * math.cos(rot)))
    return f"M {pts[0][0]:.1f} {pts[0][1]:.1f} " + " ".join(f"L {x:.1f} {y:.1f}" for x, y in pts[1:])


def ring(cx: float, cy: float, rx: float, ry: float, leader: tuple[float, float] | None,
         size: tuple[int, int]) -> str:
    """The marker ring (two passes, a little wax) and its leader line to `leader`."""
    d1 = hand_ellipse(cx, cy, rx, ry, -7, seed=3)
    d2 = hand_ellipse(cx + 2, cy - 1, rx * 1.03, ry * 1.02, -5, seed=11, overshoot=0.08)
    line = ""
    if leader:
        lx0, ly0 = cx + rx * 0.5, cy - ry * 0.96
        lx1, ly1 = leader
        line = (f'<path class="ring" d="M {lx0:.1f} {ly0:.1f} C {lx0 + 2:.1f} {ly0 - 26:.1f}, '
                f'{lx1 - 22:.1f} {ly1 + 12:.1f}, {lx1:.1f} {ly1:.1f}" stroke-width="4"/>')
    return (f'<svg class="marks" viewBox="0 0 {size[0]} {size[1]}" width="{size[0]}" '
            f'height="{size[1]}"><defs><filter id="wax" x="-10%" y="-10%" width="120%" '
            'height="120%"><feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" '
            'seed="4" result="n"/><feDisplacementMap in="SourceGraphic" in2="n" scale="2.2"/>'
            '</filter></defs><g filter="url(#wax)">'
            f'<path class="ring" d="{d1}" stroke-width="5.5" opacity=".95"/>'
            f'<path class="ring" d="{d2}" stroke-width="2.6" opacity=".7"/>{line}</g></svg>')
