"""The evaluation charts as SVG, written by hand: grouped bars (Experiments 1 and 2), F1
against the number of synthetic images (Experiment 3) and accuracy against dollars per
hour (Experiment 4, signature visual 3). The look and the shared pieces are in svg.py.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from backend.eval.svg import (
    Box,
    bar_path,
    baseline,
    document,
    dot,
    esc,
    legend,
    nice_ticks,
    text,
    usd,
    y_axis,
)


def grouped_bars(title: str, subtitle: str, groups: Sequence[str],
                 series: Sequence[tuple[str, Sequence[float | None]]]) -> str:
    """Bars from 0 to 1, one group per variant, one bar per series: the first series in the
    accent colour, the second in grey. A missing value draws no bar and reads n/a."""
    if not 1 <= len(series) <= 2 or any(len(v) != len(groups) for _, v in series):
        raise ValueError("one or two series, each with one value per group")
    width = max(560, 80 + 140 * len(groups))
    box = Box(56, 96, width - 80, 200)
    keys = [("box", cls, name) for cls, (name, _) in zip(("acc", "neu"), series, strict=False)]
    body = (legend(24, 76, keys) if len(series) > 1 else []) + y_axis(box)
    band, bar, gap = box.width / max(len(groups), 1), 24, 8
    for i, group in enumerate(groups):
        x = box.left + band * i + (band - len(series) * (bar + gap) + gap) / 2
        for cls, (name, values) in zip(("acc", "neu"), series, strict=False):
            v = values[i]
            label = "n/a" if v is None else f"{v:.2f}"
            top = box.y(v or 0.0)
            if v:  # zero and missing values draw no bar
                body.append(f'<path class="{cls}" d="{bar_path(x, top, bar, box.bottom)}">'
                            f"<title>{esc(f'{group}, {name}: {label}')}</title></path>")
            body.append(text(x + bar / 2, top - 6, label, "t1 small", "middle"))
            x += bar + gap
        body.append(text(box.left + band * (i + 0.5), box.bottom + 20, group, "t2", "middle"))
    return document(width, box.bottom + 40, title, subtitle, [*body, baseline(box)])


def line_chart(title: str, subtitle: str, points: Sequence[tuple[float, float | None, str]],
               x_name: str, y_name: str = "F1") -> str:
    """One series of 0 to 1 values against counts over 0 on a log scale. Each point shows
    its value above it and its note (such as the generation cost) below it."""
    points = sorted(points, key=lambda p: p[0])
    if not points or points[0][0] <= 0:
        raise ValueError("a line chart needs points with counts over 0")
    box = Box(56, 80, 544, 210)
    lo, hi = math.log(points[0][0]), math.log(points[-1][0])

    def px(x: float) -> float:
        return box.x(0.5 if hi == lo else 0.04 + 0.92 * (math.log(x) - lo) / (hi - lo))

    body = y_axis(box)
    shown = [(px(x), box.y(y)) for x, y, _ in points if y is not None]
    if len(shown) > 1:
        coords = " ".join(f"{a:.1f},{b:.1f}" for a, b in shown)
        body.append(f'<polyline points="{coords}" fill="none" class="acc-s" stroke-width="2" '
                    'stroke-linejoin="round" stroke-linecap="round"/>')
    for x, y, note in points:
        cx = px(x)
        body.append(text(cx, box.bottom + 20, f"{x:g}", "t3 small", "middle"))
        if y is None:
            body.append(text(cx, box.bottom - 8, "n/a", "t3 small", "middle"))
            continue
        cy = box.y(y)
        body.append(dot(cx, cy, "acc", f"{x:g} {x_name}: {y_name} {y:.2f}, {note}"))
        low = cy + 22 > box.bottom - 4  # no room under the point: the note goes on top
        body.append(text(cx, cy - (28 if low else 12), f"{y:.2f}", "t1 small strong", "middle"))
        body.append(text(cx, cy - 12 if low else cy + 22, note, "t3 small", "middle"))
    body += [baseline(box), text(box.x(0.5), box.bottom + 40, f"{x_name} (log scale)", "t3",
                                 "middle")]
    return document(640, box.bottom + 56, title, subtitle, body)


def scatter(title: str, subtitle: str, points: Sequence[dict], y_name: str = "F1") -> str:
    """Points {"label", "x": dollars per hour, "y": 0 to 1, "highlight"}, each labelled
    with its name and both values. Highlighted points (the fine-tuned model, the point of
    the chart) take the marker colour."""
    shown = [p for p in points if p.get("x") is not None and p.get("y") is not None]
    missing = [str(p["label"]) for p in points if p not in shown]
    box = Box(56, 80, 544, 240)
    ticks = nice_ticks(max([p["x"] for p in shown], default=0) * 1.15)
    body = y_axis(box)
    for v in ticks:
        body.append(text(box.x(v / ticks[-1]), box.bottom + 20, f"${v:g}", "t3 small", "middle"))
    placed: list[tuple[float, float]] = []
    for p in sorted(shown, key=lambda p: -p["y"]):
        cx, cy = box.x(p["x"] / ticks[-1]), box.y(p["y"])
        right = cx < box.x(0.6)
        lx, ly = (cx + 12 if right else cx - 12), cy - 2
        while any(abs(ox - lx) < 200 and abs(oy - ly) < 32 for ox, oy in placed):
            ly += 32  # keep two-line labels apart
        placed.append((lx, ly))
        on = bool(p.get("highlight"))
        value = f"{y_name} {p['y']:.2f}, {usd(p['x'])} per hour"
        body.append(dot(cx, cy, "mark" if on else "neu", f"{p['label']}: {value}", 6 if on else 5))
        anchor = "start" if right else "end"
        body.append(text(lx, ly, p["label"], "t1 strong" if on else "t1", anchor))
        body.append(text(lx, ly + 15, value, "t2 small", anchor))
    body += [baseline(box), text(box.x(0.5), box.bottom + 40, "dollars per hour of watching",
                                 "t3", "middle")]
    if missing:
        body.append(text(24, box.bottom + 64, "not plotted, no numbers yet: " + ", ".join(missing),
                         "t3 small"))
    return document(640, box.bottom + (76 if missing else 56), title, subtitle, body)
