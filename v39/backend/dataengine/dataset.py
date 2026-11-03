"""Write a skill's labeled images as an Axolotl multimodal SFT dataset.

Format, following Meta's cookbook recipe (muse-glimmer/post-training/sft/axolotl):
OpenAI-style `messages`; user content is an ordered list of typed parts, the image part
first (`{"type": "image", "path": <absolute path to an RGB PNG>}`), then the text part;
the assistant content is a plain string, "yes" or "no". Images are padded to a 512x512
square (aspect kept, black fill, bilinear), the recipe's default, and the watcher uses
the same function so training and inference see the same pixels.

The cookbook's examples have no system turn. SPEC.md asks for one, so it is on by
default and `system_in_user=True` folds it into the user text instead, in case the
processor preflight shows the template mishandles system turns.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_SIZE = 512
SYSTEM_PROMPT = (
    "You watch first-person camera frames for one condition. Answer only yes or no."
)


def pad_image(data: bytes, size: int = IMAGE_SIZE) -> Image.Image:
    with Image.open(io.BytesIO(data)) as img:
        return ImageOps.pad(img.convert("RGB"), (size, size),
                            method=Image.Resampling.BILINEAR, color=(0, 0, 0))


def user_prompt(positive: str, negatives: Iterable[str]) -> str:
    nots = "; ".join(n.strip().rstrip(".") for n in negatives)
    return (f"Condition: {positive.strip().rstrip('.')}. Not: {nots}. "
            "Is the condition true in this frame?")


@dataclass
class Example:
    id: str
    image: bytes  # source image bytes (keyframe, real frame or generated image)
    answer: str  # "yes" or "no"
    source: str  # "real" or "synthetic"
    skill: str
    positive: str
    negatives: list[str]
    # Where it came from, added to its provenance index entry: a crop's keyframe and box,
    # a clip frame's clip, time and session.
    provenance: dict = field(default_factory=dict)


def build_record(example: Example, image_path: Path, system_in_user: bool = False) -> dict:
    text = user_prompt(example.positive, example.negatives)
    messages = []
    if system_in_user:
        text = f"{SYSTEM_PROMPT}\n\n{text}"
    else:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages += [
        {"role": "user", "content": [
            {"type": "image", "path": str(image_path.resolve())},
            {"type": "text", "text": text},
        ]},
        {"role": "assistant", "content": example.answer},
    ]
    return {"id": example.id, "messages": messages}


def write_dataset(
    examples: list[Example], out_dir: Path, name: str, system_in_user: bool = False,
) -> dict:
    """Write PNGs and `<name>.jsonl`; return the manifest (also written as JSON)."""
    if not examples:
        raise ValueError("no examples to write")
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    lines, counts, index = [], {}, []
    for ex in examples:
        if ex.answer not in ("yes", "no"):
            raise ValueError(f"{ex.id}: answer must be yes or no, got {ex.answer!r}")
        path = images_dir / f"{ex.id}.png"
        pad_image(ex.image).save(path, format="PNG")  # rewritten so a rerun never goes stale
        lines.append(json.dumps(build_record(ex, path, system_in_user), sort_keys=True))
        key = f"{ex.skill}/{ex.source}/{ex.answer}"
        counts[key] = counts.get(key, 0) + 1
        # Provenance: the padded PNG is a derived copy; this points back to the original.
        index.append({**ex.provenance, "id": ex.id, "source": ex.source, "answer": ex.answer,
                      "original_sha256": hashlib.sha256(ex.image).hexdigest()})
    jsonl = out_dir / f"{name}.jsonl"
    body = "\n".join(lines) + "\n"
    jsonl.write_text(body, encoding="utf-8")
    manifest = {
        "name": name,
        "examples": len(examples),
        "counts": dict(sorted(counts.items())),
        "image_size": IMAGE_SIZE,
        "system_in_user": system_in_user,
        "jsonl_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "ids_sha256": hashlib.sha256("\n".join(e.id for e in examples).encode()).hexdigest(),
        "examples_index": index,
    }
    (out_dir / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest
