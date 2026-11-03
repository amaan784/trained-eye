"""Experiment 2: how often Muse Spark's labels are wrong, and the arm without its filter.

`spark_error_rate` compares Spark's label (labels.jsonl, written by te teach) with the
author's hand labels from the labelling tool (backend/labeltool/images.py):

    {"labels": {"<image file name>": "yes" | "no" | "unsure", ...}}

Keys are file names as in labels.jsonl's "file" (its last part). Images the author marked
unsure are left out and counted, as are images Spark could not label (`label_error`).
- `spark`: Spark's error rate over the images it said yes or no to, with a 95% Wilson
  interval; Spark's unsure answers are counted apart (the filter drops those images).
- `confusion`: counts as confusion[human][spark].
- `intended`: the error rate of the label each generated image was made to have: the
  label noise of the unfiltered arm.
- `kept`: the error rate among the images the filter keeps (Spark agreed with the
  intended label): the label noise of the filtered arm, which is what te teach trains on.

`write_unfiltered_dataset` writes that unfiltered arm: a dataset written exactly like te
teach's (same writer, name and file names) into dataset-unfiltered/, with every generated
image and the label it was generated for, whatever Spark said, plus the real keyframes
with the same answers te teach used, and the system prompt placed as in te teach's
dataset (its manifest's system_in_user). `te train --dataset dataset-unfiltered` trains on
it.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from backend.dataengine.dataset import write_dataset
from backend.skills.spec import SkillSpec
from backend.teach import _examples

HUMAN = ("yes", "no", "unsure")
Z95 = 1.959964


def wilson(errors: int, n: int, z: float = Z95) -> list[float] | None:
    """The Wilson score interval for errors / n (None for n = 0)."""
    if n == 0:
        return None
    p = errors / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _rate(errors: int, n: int) -> dict:
    return {"n": n, "errors": errors, "rate": round(errors / n, 4) if n else None,
            "ci95": wilson(errors, n)}


def read_rows(labels_jsonl: Path) -> dict[str, dict]:
    """labels.jsonl rows by image file name."""
    rows = [json.loads(line) for line in Path(labels_jsonl).read_text("utf-8").splitlines()
            if line.strip()]
    by_name = {}
    for row in rows:
        name = Path(row["file"]).name
        if name in by_name:
            raise ValueError(f"{labels_jsonl} names the image {name} twice")
        by_name[name] = row
    return by_name


def read_human(path: Path) -> dict[str, str]:
    raw = json.loads(Path(path).read_text("utf-8"))
    labels = raw.get("labels") if isinstance(raw, dict) else None
    if not isinstance(labels, dict):
        raise TypeError(f'{path}: expected {{"labels": {{"<file name>": "yes"}}}}')
    bad = [f"{k}: {v!r}" for k, v in labels.items() if v not in HUMAN]
    if bad:
        raise ValueError(f"{path}: labels must be yes, no or unsure: {', '.join(bad[:5])}")
    return {Path(k).name: v for k, v in labels.items()}


def spark_error_rate(labels_jsonl: Path, human: Path) -> dict:
    """Spark's and the intended labels' error rates against the author's (see above)."""
    rows, truth = read_rows(labels_jsonl), read_human(human)
    unmatched = sorted(name for name in truth if name not in rows)
    if len(unmatched) == len(truth):
        raise ValueError(f"none of the {len(truth)} hand-labelled images is in {labels_jsonl}")
    confusion = {h: dict.fromkeys(HUMAN, 0) for h in ("yes", "no")}
    human_unsure = spark_failed = 0
    intended = [0, 0]  # errors, n
    kept = [0, 0]
    for name, answer in sorted(truth.items()):
        row = rows.get(name)
        if row is None:
            continue
        if answer == "unsure":
            human_unsure += 1
            continue
        if row.get("source") == "synthetic" and row.get("label") in ("yes", "no"):
            intended[0] += row["label"] != answer
            intended[1] += 1
        if row.get("label_error"):
            spark_failed += 1
            continue
        confusion[answer][row["spark_label"]] += 1
        if row.get("source") == "synthetic" and row["spark_label"] == row.get("label"):
            kept[0] += row["label"] != answer
            kept[1] += 1
    decided = sum(confusion[h][s] for h in confusion for s in ("yes", "no"))
    wrong = confusion["yes"]["no"] + confusion["no"]["yes"]
    unsure = confusion["yes"]["unsure"] + confusion["no"]["unsure"]
    judged = decided + unsure
    return {
        "hand_labelled": len(truth), "unmatched": len(unmatched),
        "unmatched_files": unmatched[:20], "human_unsure": human_unsure,
        "spark_failed": spark_failed,
        "spark": {**_rate(wrong, decided), "unsure": unsure,
                  "unsure_rate": round(unsure / judged, 4) if judged else None},
        "confusion": confusion,
        "intended": _rate(*intended),
        "kept": _rate(*kept),
    }


def write_unfiltered_dataset(skill_dir: Path, *, train_on_model_outputs: bool = True,
                             system_in_user: bool | None = None) -> dict:
    """Write skill_dir/dataset-unfiltered (replacing it) and return its manifest. Needs a
    finished te teach run (its dataset/ shows the labels.jsonl is complete). The system
    prompt goes where te teach's dataset put it; `system_in_user` (TE_SYSTEM_IN_USER),
    when given, must agree, so the two arms differ only in the filter."""
    skill_dir = Path(skill_dir)
    name = skill_dir.name
    if not train_on_model_outputs:
        raise ValueError("training on model outputs is off: the unfiltered arm trains on "
                         "generated images, so there is nothing to compare")
    teach_manifest = skill_dir / "dataset" / f"{name}-teach.manifest.json"
    if not teach_manifest.is_file():
        raise ValueError(f"{name} has no finished teach run (no dataset/): run te teach "
                         "for it first")
    taught = bool(json.loads(teach_manifest.read_text("utf-8")).get("system_in_user", False))
    if system_in_user is not None and system_in_user != taught:
        raise ValueError(f"{name}'s dataset/ was written with TE_SYSTEM_IN_USER="
                         f"{int(taught)}, but it is {int(system_in_user)} now: run te teach "
                         "again first (cached calls are free), so both arms match")
    spec = SkillSpec.model_validate_json((skill_dir / "spec.json").read_text("utf-8"))
    rows = [json.loads(line) for line in (skill_dir / "labels.jsonl").read_text("utf-8")
            .splitlines() if line.strip()]
    generated = [row for row in rows if row["source"] == "synthetic"]
    if not generated:
        raise ValueError(f"{name} has no generated images in labels.jsonl")
    examples = _examples(spec, skill_dir, rows, generated, True)
    out = skill_dir / "dataset-unfiltered"
    shutil.rmtree(out, ignore_errors=True)  # only this function writes here
    return write_dataset(examples, out, f"{name}-teach", system_in_user=taught)
