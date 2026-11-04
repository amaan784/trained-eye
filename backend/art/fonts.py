"""Archivo's static instances for the iPhone app, cut from the variable web font with
fontTools' instancer (DESIGN.md: iOS bundles static instances, not the variable font):

    pip install "fonttools[woff]"
    python -m backend.art.fonts

Writes ios/TrainedEye/Fonts/Archivo-<Name>.ttf and Archivo-OFL.txt. Archivo is under the
SIL Open Font License 1.1 with no Reserved Font Name, so the instances keep its name.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from backend.config import PROJECT_ROOT

SOURCE = PROJECT_ROOT / "backend" / "playground" / "static" / "fonts"
VARIABLE = SOURCE / "archivo-latin-wdth-normal.woff2"
OUT = PROJECT_ROOT / "ios" / "TrainedEye" / "Fonts"
# name: (width, weight, style name); Theme.swift loads each by its PostScript name
INSTANCES = {
    "Display": (125, 800, "Expanded ExtraBold"),
    "Text": (100, 400, "Regular"),
    "TextSemibold": (100, 650, "Semibold"),
    "Edge": (62, 500, "ExtraCondensed Medium"),
}


def postscript_name(name: str) -> str:
    return f"Archivo-{name}"


def cut(name: str, width: int, weight: int, style: str, out_dir: Path = OUT) -> Path:
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    font = instantiateVariableFont(TTFont(VARIABLE), {"wdth": width, "wght": weight})
    font.flavor = None  # a plain TrueType file: iOS does not read woff2
    family = f"Archivo {name}"
    names = font["name"]
    for record in list(names.names):
        if record.nameID in (1, 2, 3, 4, 6, 16, 17, 25):
            names.removeNames(nameID=record.nameID)
    for name_id, value in ((1, family), (2, "Regular"), (3, f"{postscript_name(name)};{style}"),
                           (4, family), (6, postscript_name(name))):
        names.setName(value, name_id, 3, 1, 0x409)
        names.setName(value, name_id, 1, 0, 0)
    font["OS/2"].usWeightClass = weight
    path = Path(out_dir) / f"{postscript_name(name)}.ttf"
    font.save(path)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (width, weight, style) in INSTANCES.items():
        print(f"fonts: {cut(name, width, weight, style)} (width {width}, weight {weight})")
    shutil.copyfile(SOURCE / "Archivo-OFL.txt", OUT / "Archivo-OFL.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
