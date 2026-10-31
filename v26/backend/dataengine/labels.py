"""Muse Spark labels for every training image, real or generated.

Each image gets {label: yes|no|unsure, reason} against the skill's positive condition.
A generated image is kept only if Spark's label matches the label it was generated
for; "unsure" is dropped. The agreement rates are Experiment 2's raw data.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.dataengine.dataset import user_prompt
from backend.meta.client import MetaClient, image_part, text_part
from backend.skills.spec import SkillSpec

LABELS = ("yes", "no", "unsure")
LABEL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["label", "reason"],
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "reason": {"type": "string"},
    },
}
SYSTEM = (
    "You check training images for a camera-glasses watch skill. Decide whether the "
    "condition is true in this single image. Answer unsure if the target is not "
    "clearly visible, the image looks unrealistic, or it could go either way. Give a "
    "one-sentence reason."
)


@dataclass
class Label:
    label: str
    reason: str
    cost_usd: float
    cached: bool


def label_image(client: MetaClient, spec: SkillSpec, image: bytes) -> Label:
    prompt = user_prompt(spec.positive, spec.negatives)
    result = client.structured(
        client.settings.model("labeler"), [image_part(image), text_part(prompt)], LABEL_SCHEMA,
        name="frame_label", system=SYSTEM, skill=spec.name, max_tokens=1024,
    )
    label = result.data.get("label")
    if label not in LABELS:
        label = "unsure"
    return Label(label, str(result.data.get("reason", ""))[:300], result.info.cost_usd,
                 result.info.cached)


def filter_generated(rows: list[dict]) -> tuple[list[dict], dict]:
    """Keep rows whose `spark_label` equals the intended `label`. Returns (kept, stats).

    Each row needs `label` (intended), `spark_label` and `kind`.
    """
    stats = {"total": len(rows), "kept": 0, "unsure": 0, "disagree": 0, "by_kind": {}}
    kept = []
    for row in rows:
        kind = stats["by_kind"].setdefault(row["kind"], {"total": 0, "agree": 0, "unsure": 0})
        kind["total"] += 1
        if row["spark_label"] == "unsure":
            stats["unsure"] += 1
            kind["unsure"] += 1
        elif row["spark_label"] == row["label"]:
            kept.append(row)
            kind["agree"] += 1
        else:
            stats["disagree"] += 1
    stats["kept"] = len(kept)
    stats["agreement"] = round(len(kept) / len(rows), 4) if rows else None
    for kind in stats["by_kind"].values():
        kind["agreement"] = round(kind["agree"] / kind["total"], 4)
    return kept, stats
