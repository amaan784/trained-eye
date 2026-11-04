"""The SVG pieces every evaluation chart shares: palette, style, escaping, axes, marks.

One look for every chart, the contact sheet's (DESIGN.md): flat marks with no gradients,
Archivo with the system sans as fallback, Ink for the series a chart is about and
Graphite for the rest, Marker only for the fine-tuned model where it is the point of the
chart, values written on the marks, hairline grids. Each SVG carries its own <style>:
Paper by default and Film under @media (prefers-color-scheme: dark), so it reads on a
light or a dark page. Marks carry a <title>, which browsers show on hover. All text is
escaped.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

FONT = 'Archivo, system-ui, -apple-system, "Segoe UI", sans-serif'
PAPER, FILM, MARKER, INK, GRAPHITE = "#F4F5F2", "#000000", "#D1332B", "#1D1E20", "#6E7176"
# Surface, three inks, grid, axis, the series a chart is about (Ink), the rest (Graphite),
# the fine-tuned model when it is the point (Marker) and the shading for ranges.
LIGHT = {"bg": PAPER, "t1": INK, "t2": GRAPHITE, "t3": GRAPHITE,
         "grid": "#DCDDD9", "axis": GRAPHITE, "acc": INK, "neu": GRAPHITE, "mark": MARKER,
         "wash": f"{INK};fill-opacity:.08"}
DARK = {"bg": FILM, "t1": PAPER, "t2": "#A4A7AB", "t3": "#A4A7AB",
        "grid": "#26272A", "axis": GRAPHITE, "acc": PAPER, "neu": GRAPHITE, "mark": MARKER,
        "wash": f"{PAPER};fill-opacity:.1"}


def _rules(c: dict) -> str:
    fills = "".join(f".{k}{{fill:{c[k]}}}"
                    for k in ("bg", "t1", "t2", "t3", "acc", "neu", "mark", "wash"))
    lines = "".join(f".{k}-s{{stroke:{c[k]}}}"
                    for k in ("t1", "t2", "grid", "axis", "acc", "neu", "mark"))
    return fills + lines + f".ring{{stroke:{c['bg']}}}"


STYLE = (f"text{{font-family:{FONT};font-size:12px}}.title{{font-size:15px;font-weight:650}}"
         ".small{font-size:11px}.strong{font-weight:650}" + _rules(LIGHT)
         + "@media (prefers-color-scheme: dark){" + _rules(DARK) + "}")
TICKS_01 = [0.0, 0.25, 0.5, 0.75, 1.0]
_NOT_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


def esc(value: object) -> str:
    """Text made safe inside SVG: escaped, without characters XML cannot hold."""
    return html.escape(_NOT_XML.sub("", str(value)), quote=True)


def usd(value: float) -> str:
    return f"${value:.3f}" if 0 < abs(value) < 0.1 else f"${value:.2f}"


def text_width(value: str, size: float = 12) -> float:
    return 0.6 * size * len(value)  # a safe guess for Archivo and the system sans


def text(x: float, y: float, value: object, cls: str = "t2", anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">'
            f"{esc(value)}</text>")


@dataclass
class Box:
    """The plot area; x() and y() turn 0 to 1 fractions into pixels (y grows upwards)."""

    left: float
    top: float
    width: float
    height: float

    @property
    def bottom(self) -> float:
        return self.top + self.height

    def x(self, frac: float) -> float:
        return self.left + frac * self.width

    def y(self, frac: float) -> float:
        return self.top + (1 - min(max(frac, 0.0), 1.0)) * self.height


def document(width: float, height: float, title: str, subtitle: str, body: Sequence[str]) -> str:
    """A whole SVG: the style, the background, the title and subtitle, then `body`."""
    size = f'width="{width:.0f}" height="{height:.0f}"'
    return "".join([
        (f'<svg xmlns="http://www.w3.org/2000/svg" {size} viewBox="0 0 {width:.0f} {height:.0f}"'
         ' role="img" aria-labelledby="title desc">'),
        f'<title id="title">{esc(title)}</title><desc id="desc">{esc(subtitle)}</desc>',
        f"<style>{STYLE}</style>", f'<rect class="bg" {size} rx="8"/>',
        text(24, 32, title, "t1 title"), text(24, 52, subtitle, "t2"), *body, "</svg>\n"])


def nice_ticks(hi: float, count: int = 4) -> list[float]:
    """Round tick values from 0 to at least `hi`."""
    if not hi > 0:
        return [0.0, 1.0]
    raw = hi / count
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw * (1 - 1e-9))
    return [round(i * step, 10) for i in range(math.ceil(hi / step - 1e-9) + 1)]


def y_axis(box: Box, ticks: Sequence[float] = TICKS_01) -> list[str]:
    out = []
    for v in ticks:
        y = box.y(v / ticks[-1])
        out.append(f'<line x1="{box.left}" x2="{box.left + box.width}" y1="{y:.1f}" '
                   f'y2="{y:.1f}" class="grid-s"/>')
        out.append(text(box.left - 8, y + 4, f"{v:g}", "t3 small", "end"))
    return out


def baseline(box: Box) -> str:
    return (f'<line x1="{box.left}" x2="{box.left + box.width}" y1="{box.bottom}" '
            f'y2="{box.bottom}" class="axis-s"/>')


def legend(x: float, y: float, items: Sequence[tuple[str, str, str]]) -> list[str]:
    """One row of keys and names. Items are (kind, class, name), kind one of box, dot,
    line or dash."""
    out = []
    for kind, cls, name in items:
        if kind in ("box", "dot"):
            out.append(f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" class="{cls}"/>'
                       if kind == "box" else f'<circle cx="{x + 5}" cy="{y - 4}" r="4" '
                       f'class="{cls}"/>')
            key = 10
        else:
            dash = ' stroke-dasharray="5 4"' if kind == "dash" else ""
            out.append(f'<line x1="{x}" x2="{x + 18}" y1="{y - 4}" y2="{y - 4}" class="{cls}" '
                       f'stroke-width="2"{dash}/>')
            key = 18
        out.append(text(x + key + 6, y, name))
        x += key + 6 + text_width(name) + 18
    return out


def dot(x: float, y: float, cls: str, tip: str, r: float = 4) -> str:
    """A marker with a surface ring and a 24 px hover target."""
    return (f'<g><title>{esc(tip)}</title><circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="#000" '
            f'fill-opacity="0"/><circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="{cls} ring" '
            'stroke-width="2"/></g>')


def bar_path(x: float, top: float, width: float, bottom: float, r: float = 4) -> str:
    """A bar rounded at its data end and square at the baseline."""
    r = min(r, width / 2, bottom - top)
    return (f"M{x:.1f},{bottom:.1f}V{top + r:.1f}Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f}"
            f"H{x + width - r:.1f}Q{x + width:.1f},{top:.1f} {x + width:.1f},{top + r:.1f}"
            f"V{bottom:.1f}Z")
