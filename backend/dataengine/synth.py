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
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.dataengine import parallel
from backend.meta.cache import sha256_hex
from backend.meta.client import EXTENSIONS, MetaAPIError, MetaClient, OutputError, sniff_mime
from backend.skills.spec import SkillSpec, distinct, lower_first

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
    # Muse Image's reasoning_strength ("low"), or None for the API's default: recorded in
    # the manifest, so te regenerate asks the same way
    reasoning_strength: str | None = None


def plan_variations(
    spec: SkillSpec, keyframe_hashes: list[str], positives: int = 150, negatives: int = 150,
    reasoning_strength: str | None = None,
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
        conditions = [spec.positive] if kind == "positive" else distinct(spec.negatives)
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
                reasoning_strength=reasoning_strength,
            ))
    return plan


def _clause(text: str) -> str:
    return lower_first(text)


# Errors that concern one image (a rejected prompt, no image returned): that variation is
# skipped and recorded. Anything else (auth, rate limits, server, budget) stops the run.
SKIPPABLE_STATUSES = frozenset({400, 422})
MAX_FAILED_SHARE = 0.2


class TooManyFailures(MetaAPIError):
    """Too many items failed: something systematic is wrong. Failed calls are not cached,
    so a rerun pays for them again; fix the cause first."""

    def __init__(self, message: str):
        super().__init__(None, message)


def skippable(err: MetaAPIError) -> bool:
    return isinstance(err, OutputError) or err.status in SKIPPABLE_STATUSES


def too_many_failures(failed: int, total: int) -> bool:
    """More than a few items failing means something systematic is wrong: stop paying."""
    return failed > max(5, MAX_FAILED_SHARE * total)


def estimate_cost(client: MetaClient, count: int) -> float:
    return count * client.settings.price(client.settings.model("image")).per_image


def generate(
    client: MetaClient,
    plan: list[Variation],
    keyframes: dict[str, bytes],
    out_dir: Path,
    progress: Callable[[bool, float], None] | None = None,
    workers: int = 1,
) -> list[dict]:
    """Run the plan (cached), `workers` edits at a time, save images untouched, write
    manifest.jsonl in plan order.

    A variation whose edit is rejected is recorded in the manifest with file null and the
    error, and left out of the returned rows; failed calls are not cached, so a rerun
    tries it again. The manifest is written even when the run stops, so it always
    describes this run. `progress(cached, cost_usd)` is called after each item.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, failed, order = [], [], {v.id: i for i, v in enumerate(plan)}

    def edit(v: Variation):
        return client.image_edit(keyframes[v.keyframe_sha256], v.prompt, skill=v.skill,
                                 reasoning_strength=v.reasoning_strength)

    try:
        for i, result, err in parallel.run_all(plan, edit, workers):
            v = plan[i]
            if err is not None:
                if not isinstance(err, MetaAPIError) or not skippable(err):
                    raise err
                failed.append({**asdict(v), "file": None, "error": str(err)[:300],
                               "cost_usd": err.cost_usd})
                if too_many_failures(len(failed), len(plan)):
                    raise TooManyFailures(f"{len(failed)} of {len(plan)} image edits failed "
                                          f"so far; last: {str(err)[:300]}") from None
                if progress:
                    progress(False, err.cost_usd)
                continue
            image = result.images[0]
            path = out_dir / f"{v.id}.{EXTENSIONS.get(sniff_mime(image), 'bin')}"
            path.write_bytes(image)  # always the bytes the manifest describes, untouched
            rows.append({**asdict(v), "file": path.name, "image_sha256": sha256_hex(image),
                         "model": result.info.model, "cache_key": result.info.cache_key,
                         # what the image cost when it was made, also on a cached rerun
                         "cost_usd": result.info.billed_usd, "cached": result.info.cached})
            if progress:
                progress(result.info.cached, result.info.cost_usd)
    finally:
        rows.sort(key=lambda row: order[row["id"]])  # plan order, however they finished
        failed.sort(key=lambda row: order[row["id"]])
        tmp = out_dir / "manifest.jsonl.tmp"
        with tmp.open("w", encoding="utf-8") as fh:
            for row in rows + failed:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        tmp.replace(out_dir / "manifest.jsonl")
    return rows
