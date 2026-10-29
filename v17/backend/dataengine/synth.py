"""Synthetic variations of the teaching keyframes with Muse Image edits.

Each variation is an edit of one real keyframe (anchored generation), so the scene
stays the user's own:
  - positives: the skill's `positive` condition rendered realistically
  - hard negatives: one of the skill's `negatives` (near-misses)
  - nuisance variation on both: lighting, angle, blur, distance, partial occlusion

The full prompt manifest is written as JSONL next to the images. Images are saved
byte for byte as Muse Image returned them, keeping the watermark and provenance
metadata; they stay in data/ and are never committed. The manifest plus this module
regenerate them (cached calls are free).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.meta.cache import sha256_hex
from backend.meta.client import EXTENSIONS, MetaClient, sniff_mime
from backend.skills.spec import SkillSpec, lower_first

NUISANCE = (
    "in the same lighting",
    "in dim evening light",
    "in bright daylight with some glare",
    "under warm indoor lighting",
    "seen from a slightly lower angle",
    "seen from a slightly higher angle, tilted a little",
    "from a step further away",
    "from a little closer",
    "slightly out of focus",
    "with mild motion blur, as if the head is turning",
    "with the edge of a hand or an object partly in front",
)

STYLE = ("Keep it a realistic first-person photo of the same scene and the same object. "
         "No people or faces in view. No added text, borders or captions.")


@dataclass
class Variation:
    id: str
    skill: str
    label: str  # the label this image is generated to have: "yes" or "no"
    kind: str  # "positive" or "hard_negative"
    condition: str
    nuisance: str
    keyframe_sha256: str
    prompt: str


def plan_variations(
    spec: SkillSpec, keyframe_hashes: list[str], positives: int = 150, negatives: int = 150,
) -> list[Variation]:
    """Deterministic plan: same spec and keyframes give the same prompts and ids.

    Every (condition, keyframe, nuisance) combination is used once, in a fixed shuffled
    order, before any repeats; a repeat gets a "Variation N." suffix so no two requests
    are identical (identical requests would come back from the cache as one image).
    """
    if spec.refused:
        raise ValueError("cannot generate data for a refused skill")
    if not keyframe_hashes:
        raise ValueError("need at least one keyframe")
    plan = []
    for kind, label, count in (("positive", "yes", positives),
                               ("hard_negative", "no", negatives)):
        conditions = [spec.positive] if kind == "positive" else list(spec.negatives)
        combos = [(c, k, n) for c in conditions for k in keyframe_hashes for n in NUISANCE]
        combos.sort(key=lambda combo: hashlib.sha256(
            f"{spec.name}|{kind}|{combo}".encode()).hexdigest())
        for i in range(count):
            condition, keyframe, nuisance = combos[i % len(combos)]
            if kind == "positive":
                change = f"Edit this photo so that {_clause(condition)}."
            else:
                change = (f"Edit this photo so that {_clause(condition)}, and it is clearly "
                          f"not the case that {_clause(spec.positive)}.")
            repeat = i // len(combos)
            suffix = f" Variation {repeat + 1}." if repeat else ""
            prompt = f"{change} Show it {nuisance}. {STYLE}{suffix}"
            digest = hashlib.sha256(f"{spec.name}|{kind}|{i}|{keyframe}|{prompt}".encode())
            plan.append(Variation(
                id=f"{spec.name}-{kind[:3]}-{i:03d}-{digest.hexdigest()[:8]}",
                skill=spec.name, label=label, kind=kind, condition=condition,
                nuisance=nuisance, keyframe_sha256=keyframe, prompt=prompt,
            ))
    return plan


def _clause(text: str) -> str:
    return lower_first(text)


def estimate_cost(client: MetaClient, count: int) -> float:
    return count * client.settings.price(client.settings.model("image")).per_image


def generate(
    client: MetaClient,
    plan: list[Variation],
    keyframes: dict[str, bytes],
    out_dir: Path,
) -> list[dict]:
    """Run the plan (cached), save images untouched, write manifest.jsonl."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in plan:
        result = client.image_edit(keyframes[v.keyframe_sha256], v.prompt, skill=v.skill)
        image = result.images[0]
        path = out_dir / f"{v.id}.{EXTENSIONS.get(sniff_mime(image), 'bin')}"
        path.write_bytes(image)  # always the bytes the manifest describes, untouched
        rows.append({**asdict(v), "file": path.name, "image_sha256": sha256_hex(image),
                     "model": result.info.model, "cache_key": result.info.cache_key,
                     "cost_usd": result.info.cost_usd})
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return rows
