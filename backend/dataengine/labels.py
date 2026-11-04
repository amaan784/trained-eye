"""Muse Spark labels for every training image, real or generated.

Each image gets {label: yes|no|unsure, reason} against the skill's positive condition.
A generated image is kept only if Spark's label matches the label it was generated
for; "unsure" is dropped. The agreement rates are Experiment 2's raw data.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.dataengine.dataset import user_prompt
from backend.meta.client import (
    CacheMiss,
    MetaAPIError,
    MetaClient,
    OutputError,
    image_part,
    text_part,
)
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
# Spark reasons before answering and shares max_tokens with the reasoning; only a reply
# cut off for length is asked again with the larger budget. The first budget was 1024
# until the first live runs (Nov 4): with real keyframes (about 3,500 image tokens) 11 of
# 36 labels used all of it on reasoning and were paid twice. Billing follows the tokens
# used, not the budget, so a larger first budget costs nothing when it is not needed.
MAX_TOKENS = (2048, 4096)
SKIPPABLE_STATUSES = frozenset({400, 422})


@dataclass
class Label:
    label: str
    reason: str
    cost_usd: float
    cached: bool
    error: bool = False  # no usable label: the call failed or the reply was unusable


def label_image(client: MetaClient, spec: SkillSpec, image: bytes) -> Label:
    """Spark's label for one image. A failure for this image alone (a refusal, an
    unusable or rejected reply) gives an "unsure" Label with error=True instead of
    stopping the whole run; the failed call is billed but not cached.

    A reply that needed the larger budget is looked up in the cache first, so a rerun
    does not pay for the cut-off attempt again. `cost_usd` includes failed attempts."""
    prompt = user_prompt(spec.positive, spec.negatives)

    def ask(max_tokens: int, cache_only: bool = False):
        return client.structured(
            client.settings.model("labeler"), [image_part(image), text_part(prompt)],
            LABEL_SCHEMA, name="frame_label", system=SYSTEM, skill=spec.name,
            max_tokens=max_tokens, cache_only=cache_only,
        )

    spent = 0.0
    try:
        result = ask(MAX_TOKENS[-1], cache_only=True)
    except CacheMiss:
        for max_tokens in MAX_TOKENS:
            try:
                result = ask(max_tokens)
                break
            except MetaAPIError as err:
                spent += err.cost_usd
                if not (isinstance(err, OutputError) or err.status in SKIPPABLE_STATUSES):
                    raise
                cut_off = isinstance(err, OutputError) and err.finish_reason == "length"
                if cut_off and max_tokens != MAX_TOKENS[-1]:
                    continue
                return Label("unsure", f"no label: {err}"[:300], spent, False, error=True)
    data = result.data if isinstance(result.data, dict) else {}
    label = data.get("label")
    if label not in LABELS:
        label = "unsure"
    return Label(label, str(data.get("reason", ""))[:300], spent + result.info.cost_usd,
                 result.info.cached and not spent)


def filter_generated(rows: list[dict]) -> tuple[list[dict], dict]:
    """Keep rows whose `spark_label` equals the intended `label`. Returns (kept, stats).

    Each row needs `label` (intended), `spark_label` and `kind`. Rows whose labelling
    failed (`label_error`) are dropped and counted apart, so the agreement rates cover only
    images Spark actually judged.
    """
    errors = [row for row in rows if row.get("label_error")]
    rows = [row for row in rows if not row.get("label_error")]
    stats = {"total": len(rows), "kept": 0, "unsure": 0, "disagree": 0, "by_kind": {}}
    if errors:
        stats["label_errors"] = len(errors)
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
