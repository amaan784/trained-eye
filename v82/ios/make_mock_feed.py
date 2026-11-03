"""Make ios/TrainedEye/Resources/mock_feed.mp4: the synthetic camera feed for DAT's mock
glasses in the simulator, which has no camera.

A washer's control panel drawn with Pillow: the display counts down 0:09 to 0:01, then
shows "End" for the rest of the clip, so the washer_done skill has something to watch for.
Nothing in it is recorded, so it shows no person and no home. HEVC (libx265, tagged hvc1
so Apple's decoders accept it), 360x640 like the glasses' low resolution stream, 10 fps,
16 seconds.

    python ios/make_mock_feed.py          # needs Pillow and ffmpeg with libx265
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "TrainedEye" / "Resources" / "mock_feed.mp4"
SIZE, FPS, SECONDS, COUNTDOWN = (360, 640), 10, 16, 9
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")


def font(size: int) -> ImageFont.ImageFont:
    return ImageFont.truetype(str(FONT), size) if FONT.is_file() else ImageFont.load_default(size)


def frame(t: float) -> Image.Image:
    """The panel at t seconds: a slow drift keeps consecutive frames from being identical."""
    img = Image.new("RGB", SIZE, (28, 30, 34))
    draw = ImageDraw.Draw(img)
    dx = round(4 * (t % 4 - 2))
    draw.rounded_rectangle((30 + dx, 170, 330 + dx, 470), 24, fill=(214, 216, 220))  # panel
    draw.rounded_rectangle((70 + dx, 210, 290 + dx, 300), 10, fill=(12, 16, 14))  # display
    left = COUNTDOWN - int(t)
    text, colour = (f"0:{left:02d}", (90, 230, 140)) if left > 0 else ("End", (120, 255, 170))
    box = draw.textbbox((0, 0), text, font=font(54))
    draw.text((180 + dx - (box[2] - box[0]) / 2, 228), text, font=font(54), fill=colour)
    for i, x in enumerate((95, 180, 265)):  # three buttons
        lit = i == 1 and left <= 0
        draw.ellipse((x - 22 + dx, 360, x + 22 + dx, 404), fill=(90, 230, 140) if lit
                     else (150, 153, 160))
    draw.ellipse((150 + dx, 480, 210 + dx, 540), outline=(110, 112, 118), width=6)  # drum edge
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
