"""Render HTML to PNG with headless Chrome, and PNG frames to a GIF and an MP4 with ffmpeg.

Every page is filled from a template with string.Template; a page that still holds a
$name or {{...}} after filling is refused, so no image carries a placeholder.
"""

from __future__ import annotations

import re
import shutil
import string
import subprocess
import tempfile
from pathlib import Path

CHROME_PATHS = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Chromium.app/Contents/MacOS/Chromium")
TEMPLATES = Path(__file__).with_name("templates")
LEFTOVER = re.compile(r"\$[A-Za-z_]\w*|\$\{\w+\}|\{\{.*?\}\}")


class RenderError(RuntimeError):
    pass


def chrome() -> str:
    for path in (shutil.which("google-chrome"), shutil.which("chromium"), *CHROME_PATHS):
        if path and Path(path).exists():
            return path
    raise RenderError("headless Chrome is needed (Google Chrome or Chromium)")


def fill(template: str, values: dict) -> str:
    """The template with every $name replaced; RenderError if any is left over."""
    # safe_substitute leaves a script's own ${...} alone; outside scripts, anything left
    # unfilled is refused below
    page = string.Template((TEMPLATES / template).read_text(encoding="utf-8")).safe_substitute(
        values)
    left = LEFTOVER.findall(page.split("<script", 1)[0])
    if left:
        raise RenderError(f"{template}: unfilled {sorted(set(left))}")
    return page


def screenshot(html: str, out: Path, size: tuple[int, int], *, scale: int = 2,
               folder: Path | None = None) -> Path:
    """Render `html` (written into `folder`, so relative links resolve) at `size` CSS
    pixels and `scale` device pixels per CSS pixel."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(folder or tmp) / f".render-{out.stem}.html"
        page.write_text(html, encoding="utf-8")
        profile = Path(tmp) / "profile"
        try:
            done = subprocess.run(
                [chrome(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                 "--no-first-run", "--no-default-browser-check", "--mute-audio",
                 f"--user-data-dir={profile}", "--allow-file-access-from-files",
                 f"--force-device-scale-factor={scale}", "--virtual-time-budget=4000",
                 f"--window-size={size[0]},{size[1]}", f"--screenshot={out.resolve()}",
                 page.resolve().as_uri()],
                capture_output=True, text=True, timeout=120, check=False)
        finally:
            page.unlink(missing_ok=True)
    if done.returncode != 0 or not out.is_file():
        raise RenderError(f"Chrome did not render {out.name}: {done.stderr[-300:]}")
    return out


def gif(frames: list[Path], out: Path, fps: float, width: int, max_bytes: int) -> Path:
    """An optimised GIF of `frames` (one palette for all), smaller than `max_bytes`; the
    width steps down until it fits."""
    pattern = _sequence(frames)
    for w in (width, int(width * 0.85), int(width * 0.7), int(width * 0.6)):
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-framerate", str(fps), "-i",
               str(pattern), "-vf",
               (f"scale={w}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:"
               "stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:"
               "diff_mode=rectangle"), str(out)]
        subprocess.run(cmd, check=True)
        if out.stat().st_size <= max_bytes:
            return out
    raise RenderError(f"{out.name} is {out.stat().st_size} bytes, over {max_bytes}")


def mp4(frames: list[Path], out: Path, fps: float) -> Path:
    pattern = _sequence(frames)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-framerate", str(fps), "-i",
                    str(pattern), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", str(out)], check=True)
    return out


def _sequence(frames: list[Path]) -> Path:
    """The frames as one numbered sequence (they are written as frame_00000.png ...)."""
    folder = frames[0].parent
    for i, f in enumerate(frames):
        if f.name != f"frame_{i:05d}.png" or f.parent != folder:
            raise RenderError("frames must be frame_00000.png, frame_00001.png, ... in one folder")
    return folder / "frame_%05d.png"
