"""Make ios/TrainedEye/Resources/mock_feed.mp4: the synthetic camera feed for DAT's mock
glasses in the simulator, which has no camera.

A gas stove burner seen from above, drawn with Pillow: off with its knob at 0 for the
first 3 seconds, then lit, a ring of blue flame with no pan on it, for the rest of the
clip, so the burner_left_on skill (with its 10 s hold) has something to watch for.
Nothing in it is recorded, so it shows no person and no home. HEVC (libx265, tagged hvc1
so Apple's decoders accept it), 360x640 like the glasses' low resolution stream, 10 fps,
16 seconds.

    python ios/make_mock_feed.py          # needs Pillow and ffmpeg with libx265
"""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).parent / "TrainedEye" / "Resources" / "mock_feed.mp4"
SIZE, FPS, SECONDS, LIT_AT = (360, 640), 10, 16, 3
FLAMES = 18


def frame(t: float) -> Image.Image:
    """The burner at t seconds: a slow drift keeps consecutive frames from being identical."""
    img = Image.new("RGB", SIZE, (22, 23, 26))
    draw = ImageDraw.Draw(img)
    dx = round(4 * (t % 4 - 2))
    cx, cy = 180 + dx, 270
    draw.rounded_rectangle((20 + dx, 90, 340 + dx, 560), 28, fill=(38, 40, 45))  # the hob
    for angle in range(0, 360, 45):  # the grate's arms
        a = math.radians(angle)
        draw.line((cx + 70 * math.cos(a), cy + 70 * math.sin(a),
                   cx + 125 * math.cos(a), cy + 125 * math.sin(a)), fill=(70, 72, 78), width=9)
    draw.ellipse((cx - 125, cy - 125, cx + 125, cy + 125), outline=(70, 72, 78), width=9)
    lit = t >= LIT_AT
    if lit:  # a ring of blue flames around the cap, flickering a little
        for i in range(FLAMES):
            a = 2 * math.pi * i / FLAMES
            reach = 64 + 6 * math.sin(7 * t + 2.3 * i)
            tip = (cx + reach * math.cos(a), cy + reach * math.sin(a))
            side = (math.cos(a + math.pi / 2) * 7, math.sin(a + math.pi / 2) * 7)
            base = (cx + 44 * math.cos(a), cy + 44 * math.sin(a))
            draw.polygon([(base[0] - side[0], base[1] - side[1]), tip,
                          (base[0] + side[0], base[1] + side[1])], fill=(60, 120, 255))
        draw.ellipse((cx - 50, cy - 50, cx + 50, cy + 50), outline=(150, 200, 255), width=5)
    draw.ellipse((cx - 44, cy - 44, cx + 44, cy + 44), fill=(28, 28, 30))  # the burner cap
    kx, ky = 180 + dx, 480  # the knob, turned when lit
    draw.ellipse((kx - 32, ky - 32, kx + 32, ky + 32), fill=(190, 192, 198))
    turn = math.radians(-90 + (120 if lit else 0))
    draw.line((kx, ky, kx + 26 * math.cos(turn), ky + 26 * math.sin(turn)),
              fill=(30, 30, 34), width=6)
    return img


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(FPS * SECONDS):
            frame(i / FPS).save(Path(tmp) / f"f{i:04d}.png")
        command = ["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i",
                   str(Path(tmp) / "f%04d.png"), "-c:v", "libx265", "-tag:v", "hvc1",
                   "-pix_fmt", "yuv420p", "-x265-params", "log-level=error", "-crf", "30",
                   "-map_metadata", "-1", str(OUT)]
        if subprocess.run(command, check=False).returncode != 0:
            print("ffmpeg with libx265 is needed", file=sys.stderr)
            return 1
    print(f"{OUT}: {OUT.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
