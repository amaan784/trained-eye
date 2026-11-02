"""The live timeline (signature visual 2) as an SVG, from one skill's rows of a frame log.

The student's yes-probability is the accent line, the threshold is dashed, the labelled
true ranges are shaded, and each alert is a dot with the text the glasses spoke above
it. A frame the student failed on leaves a gap in the line. Same look as the other charts (svg.py).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from backend.eval.svg import Box, baseline, document, dot, legend, text, text_width, y_axis

TIME_STEPS_S = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600)


def clock(seconds: float) -> str:
    """m:ss from the start of the clip."""
    s = round(seconds)
    return f"{s // 60}:{s % 60:02d}"


def _segments(points: Sequence[tuple[float, float | None]]) -> list[list[tuple[float, float]]]:
    out: list[list[tuple[float, float]]] = [[]]
    for x, y in points:
        if y is None:
            out.append([])
        else:
            out[-1].append((x, y))
    return [s for s in out if s]


def timeline(title: str, subtitle: str, rows: Sequence[dict],
             true_ranges: Sequence[Sequence[float]], alert_text: str = "alert") -> str:
    """The SVG for rows of one skill (t, p, threshold, alert, cloud) and its true ranges."""
    rows = sorted(rows, key=lambda r: float(r["t"]))
    if not rows:
        raise ValueError("the timeline needs at least one frame")
    t0, t1 = float(rows[0]["t"]), float(rows[-1]["t"])
    span = (t1 - t0) or 1.0
    box = Box(56, 140, 872, 160)

    def px(t: float) -> float:
        return box.x((min(max(t, t0), t1) - t0) / span)

    threshold = float(rows[-1]["threshold"])
    body = legend(24, 76, [("line", "acc-s", "yes-probability"),
                           ("dash", "t2-s", f"threshold {threshold:.2f}"),
                           ("box", "wash", "labelled true"), ("dot", "t1", "alert")])
    body += y_axis(box)
    for a, b in sorted((float(a), float(b)) for a, b in true_ranges):
        if b >= t0 and a <= t1:
            body.append(f'<rect class="wash" x="{px(a):.1f}" y="{box.top}" '
                        f'width="{max(px(b) - px(a), 1):.1f}" height="{box.height}">'
                        f"<title>labelled true from {clock(a)} to {clock(b)}</title></rect>")
    limit = " ".join(f"{px(float(r['t'])):.1f},{box.y(float(r['threshold'])):.1f}" for r in rows)
    body.append(f'<polyline points="{limit}" fill="none" class="t2-s" stroke-width="1.5" '
                'stroke-dasharray="5 4"/>')
    for seg in _segments([(px(float(r["t"])), r.get("p")) for r in rows]):
        coords = " ".join(f"{x:.1f},{box.y(float(p)):.1f}" for x, p in seg)
        body.append(f'<polyline points="{coords}" fill="none" class="acc-s" stroke-width="2" '
                    'stroke-linejoin="round" stroke-linecap="round"/>' if len(seg) > 1 else
                    f'<circle cx="{seg[0][0]:.1f}" cy="{box.y(float(seg[0][1])):.1f}" r="1.5" '
                    'class="acc"/>')
    ends = [-math.inf, -math.inf]  # where each of the two label rows is taken up to
    for r in rows:
        if not r.get("alert") or r.get("p") is None:
            continue
        t, x, y = float(r["t"]), px(float(r["t"])), box.y(float(r["p"]))
        label = f'"{alert_text}" at {clock(t)}'
        w = text_width(label, 11)
        left, right = (x, x + w) if x + w <= box.x(1) + 24 else (x - w, x)
        row = next((i for i in (0, 1) if left > ends[i] + 8), min((0, 1), key=ends.__getitem__))
        ends[row] = right
        ly = box.top - 32 + 16 * row
        cloud = f" (Spark: {r['cloud']})" if r.get("cloud") else ""
        body += [(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{ly + 4:.1f}" y2="{y - 7:.1f}" '
                  'class="t2-s" stroke-width="1"/>'),
                 dot(x, y, "t1", f"alert at {clock(t)}: {alert_text}{cloud}", 5),
                 text(x, ly, label, "t1 small", "start" if left == x else "end")]
    step = next((s for s in TIME_STEPS_S if span / s <= 8), TIME_STEPS_S[-1])
    tick = math.ceil(t0 / step) * step
    while tick <= t1 + 1e-9:
        body.append(text(px(tick), box.bottom + 18, clock(tick), "t3 small", "middle"))
        tick += step
    body += [baseline(box), text(box.x(0.5), box.bottom + 38,
                                 "time in the clip (minutes:seconds)", "t3", "middle")]
    return document(960, box.bottom + 54, title, subtitle, body)
