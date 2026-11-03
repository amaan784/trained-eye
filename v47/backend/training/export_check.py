"""Which examples export.py checks, and how the two models' answers are compared.

Plain Python with no Modal or torch: export.py's local entrypoint runs it on the Mac, before
and after the GPU work. The selection and the answer cut follow the cookbook's evaluator
(meta-oss-cookbook b30ca2c, muse-glimmer/post-training/sft/axolotl/scripts/evaluate.py,
Apache 2.0): examples sorted by a hash of seed and id, and the reply read after "to=user".
"""

from __future__ import annotations

import hashlib
import json
import re

LIMIT = 50  # HANDOVER.md M2: the same yes/no decisions as the adapter on 50 frames
SELECTION_SEED = 17  # evaluate.py's default --selection-seed
CHANNEL_MARKS = ("assistant to=", "user to=", "to=self")  # where evaluate.py ends the reply


def pick(lines: list[str], limit: int = LIMIT, seed: int = SELECTION_SEED) -> list[dict]:
    """Up to `limit` JSONL examples in evaluate.py's fixed order, whatever the file order."""
    examples = [json.loads(line) for line in lines if line.strip()]
    examples.sort(key=lambda e: (hashlib.sha256(f"{seed}\0{e['id']}".encode()).hexdigest(),
                                 e["id"]))
    return examples[:limit]


def yes_no(generated: str) -> str | None:
    """The first yes or no of the user-directed reply, cut as evaluate.py's
    extract_user_response cuts it; None when the reply has neither (or there is no reply)."""
    text = generated.strip()
    if (start := text.find("to=user")) >= 0:
        text = text[start + len("to=user"):]
    cuts = [i for mark in CHANNEL_MARKS if (i := text.find(mark)) >= 0]
    words = re.findall(r"[a-z]+", (text[:min(cuts)] if cuts else text).lower())
    return next((w for w in words if w in ("yes", "no")), None)


def compare(ids: list[str], labels: list[str], before: list, after: list) -> dict:
    """How often the merged model gives the reference's answer. Unreadable never agrees."""
    n = len(ids)
    if not n or not len(labels) == len(before) == len(after) == n:
        raise ValueError("compare needs one label and two answers per example")
    same = [a is not None and a == b for a, b in zip(before, after, strict=True)]
    return {"examples": n, "agreement": sum(same) / n, "all_agree": all(same),
            "disagreements": [i for i, ok in zip(ids, same, strict=True) if not ok],
            "unreadable": {"reference": before.count(None), "merged": after.count(None)},
            "label_match": {side: sum(a == y for a, y in zip(answers, labels, strict=True)) / n
                            for side, answers in (("reference", before), ("merged", after))}}
