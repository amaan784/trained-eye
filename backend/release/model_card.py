"""The Hugging Face model card: README.md with its front matter, its training, data and
results sections and its license text. The release folder and the upload are in
package.py.

Every number in the card comes from a file: axolotl.RECIPE, the training bundle's
manifest.json, and only the eval harness's results.json for results. The weights take the
base model's license: the Muse Glimmer model card at the pinned revision (read on
2026-11-02) says apache-2.0, with Meta's Usage Policy beside it, so the card says so and
states that the weights are a modification, and the release folder carries the base
model's LICENSE (BASE_LICENSE.txt, copied from that revision, by package.prepare_release).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from backend.config import DEFAULT_MODELS, STUDENT_BASE_REVISION
from backend.eval.writeup import on_time_text
from backend.training.axolotl import GLOBAL_BATCH, RECIPE

RECIPE_KEYS = ("adapter", "load_in_4bit", "lora_r", "lora_alpha", "lora_dropout",
               "lora_target_modules", "sequence_len", "micro_batch_size", "num_epochs",
               "optimizer", "lr_scheduler", "learning_rate", "warmup_ratio", "weight_decay",
               "bf16", "attn_implementation", "seed")
DATA_JOBS = ("transcribe", "compiler", "sam", "image", "labeler")
COLUMNS = (("real", "yes"), ("real", "no"), ("synthetic", "yes"), ("synthetic", "no"))
METRICS = ("hours", "precision", "recall", "f1", "event_recall_in_window",
           "false_alarms_per_hour", "time_to_alert_median_s", "latency_median_s")
BASE_LICENSE = "apache-2.0"  # the Muse Glimmer card's license at STUDENT_BASE_REVISION
USAGE_POLICY = "https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/USAGE_POLICY.md"
KINDS = {  # title, base_model_relation, front matter, what it is
    "adapter": ("LoRA adapter", "adapter", "library_name: peft\n", "A QLoRA adapter (PEFT) for"),
    "merged": ("merged build", "finetune", "",
               "The final adapter merged (`merge_and_unload()`) into"),
}
LICENSE_ID = re.compile(r"[a-z0-9][a-z0-9.+-]*")
CARD = """\
---
base_model: {base}
base_model_relation: {relation}
{front}pipeline_tag: image-text-to-text
tags: [trained-eye, qlora, axolotl, smart-glasses]
---

# Trained Eye watch skills ({title})

{what} `{base}` at revision `{revision}`. It answers yes or no to one watch condition,
given in the prompt, about a first-person camera frame. Run `{run}`; skills: {skills}.

## Training

Meta's cookbook QLoRA recipe for Muse Glimmer (meta-oss-cookbook at commit b30ca2c,
muse-glimmer/post-training/sft/axolotl) with Axolotl 0.19.0, unchanged except the paths, the
W&B names and the gradient accumulation that keeps its global batch of {batch} on one GPU{steps}.
The training data was made through the Meta Model API:
{models}.

| Setting | Value |
|---|---|
{recipe}

## Training data

{data}

{generated}

## Evaluation

{results}

## Uses

Intended: yes/no watch skills on first-person frames from smart glasses or a phone, each
taught by its user from their own footage: the state of an object, a display, a sign or a
vehicle (has the fridge been left open? is the tap still running?), with an alert threshold
calibrated on the user's own validation clips and a hold before each alert. An alert is a
reminder: it never replaces a smoke alarm or turning a burner off.

Out of scope: people (identifying, recognising or tracking anyone), bodies, health,
medication and other sensitive traits, covert monitoring, sensitive places, sports analysis
or coaching, and anything where a missed or false alert could put someone at risk.

## Limitations

- Each skill comes from one user's scenes and camera; others are untested beyond the
  evaluation above. {mix}
- It answers per frame: alerts need the watcher's smoothing, each skill's hold, a
  calibrated threshold and a cooldown. The vision tower stayed frozen, so small, distant or glary targets stay hard.
- Scoring reads the server's top logprobs for the yes or no token, so the server must return them.

## Privacy and provenance

- Faces are blurred in everything published with this model. Watch mode does not store
  frames unless the user saves a clip, and never hides the glasses' capture LED.
- Muse Image outputs keep their watermark and provenance metadata, byte for byte, and are
  never published.

## License

{license}
"""


def _cell(value) -> str:
    text = "n/a" if value is None else value if isinstance(value, str) else json.dumps(value)
    return text.replace("|", "\\|")


def data_section(manifest: dict) -> str:
    cells: dict[tuple[str, str], dict] = {}
    for key, n in manifest["counts"].items():
        split, skill, source, answer = key.split("/")
        cell = cells.setdefault((skill, split), {})
        cell[source, answer] = cell.get((source, answer), 0) + n
    spark = ("kept only where Spark agreed with the label it was made for"
             if manifest.get("dataset_dir", "dataset") == "dataset" else "not filtered")
    limit = f", at most {manifest['limit']} per skill" if manifest.get("limit") else ""
    intro = (f"Bundle `{manifest['variant']}`{limit}: {manifest.get('train_examples', 0)} "
             f"training and {manifest.get('val_examples', 0)} validation examples (train.jsonl "
             f"SHA-256 `{manifest.get('train_sha256')}`). Real: keyframes of the user's own "
             "teaching clips, labelled by the user or else by Muse Spark.")
    if source_counts(manifest)[1]:
        intro += (f" Synthetic: Muse Image edits of those keyframes, labelled by Muse Spark and "
                  f"{spark}.")
    head = "| Skill | Split | Real yes | Real no | Synthetic yes | Synthetic no |"
    return "\n".join([intro, "", head, "|---|---|---|---|---|---|"] + [
        f"| {skill} | {split} | " + " | ".join(str(c.get(k, 0)) for k in COLUMNS) + " |"
        for (skill, split), c in sorted(cells.items())])


def synthetic_skills(manifest: dict) -> list[str]:
    """The skills with generated images in a bundle manifest."""
    return sorted({key.split("/")[1] for key in manifest["counts"]
                   if key.split("/")[2] == "synthetic"})


def source_counts(manifest: dict) -> tuple[int, int]:
    """(real, synthetic) examples in a bundle manifest, both splits."""
    real = synthetic = 0
    for key, n in manifest["counts"].items():
        source = key.split("/")[2]
        real += n if source == "real" else 0
        synthetic += n if source == "synthetic" else 0
    return real, synthetic


def generated_text(real: int, synthetic: int, skills: Iterable[str] = (),
                   shipped: Iterable[str] = ()) -> tuple[str, str]:
    """The card's paragraph on generated images and its limitation, from the counts.
    `skills` have generated images; `shipped` of them have face-blurred keyframes here."""
    if not synthetic:
        about = ("There are no generated images in this bundle: it holds the user's own "
                 "keyframes only.")
        limit = "The training images are the user's own keyframes, so there were few of them."
        return about, limit
    share = "Most" if synthetic > real else "Some"
    shipped = sorted(shipped)
    unshipped = sorted(set(skills) - set(shipped))
    about = ("The generated images are not included. `manifests/` lists every edit prompt with "
             "the SHA-256 of its keyframe and of its image.")
    if shipped:
        about += (" `keyframes/` holds the face-blurred keyframes the edits start from (for "
                  f"{', '.join(shipped)}), and `te regenerate --skill <skill> --manifest "
                  "manifests/<skill>.jsonl --keyframes keyframes/<skill>` needs them to rebuild "
                  "the images. The edits then start from blurred pixels, so the rebuilt images "
                  "can differ from the ones trained on.")
    if unshipped:
        about += (f" For {', '.join(unshipped)}, `te regenerate` needs the original keyframes, "
                  "which are not published.")
    limit = (f"{share} training images are Muse Image edits with Muse Spark's labels, which "
             "can be wrong.")
    return about, limit


EPIC_CREDIT = ("EPIC-KITCHENS-100 (Damen et al., \"Rescaling Egocentric Vision\", IJCV 2022, "
               "epic-kitchens.github.io) is under CC BY-NC 4.0; it was used to test this model "
               "only, and the model was never trained on it.")


def results_section(results: Path | None, run: str | None = None) -> str:
    """The scored evaluations of `te eval`'s results.json, its numbers copied as they are:
    this run's, and the judges it is compared with (Experiment 4's base Glimmer and Muse
    Spark), not other trained runs. The test footage is named, and credited if it is
    EPIC-KITCHENS-100."""
    if results is None:
        return "Not evaluated yet: no results from the eval harness were given."
    body = Path(results).read_bytes()
    data = json.loads(body)
    evaluations = [e for e in data["evaluations"] if run is None or e.get("run") in (None, run)
                   or (e.get("experiment") == 4 and e.get("variant") in ("base", "spark"))]
    scored = [e for e in evaluations if e["status"] == "scored"]
    if not scored:
        return f"Not evaluated yet: `{Path(results).name}` has no scored evaluation."
    on_time = on_time_text(json.loads(body))
    source = (f"From `te eval`'s `{Path(results).name}` (SHA-256 "
              f"`{hashlib.sha256(body).hexdigest()}`); evaluations scored: {len(scored)}, pending: "
              f"{len(evaluations) - len(scored)}. n/a means there was nothing to compute it from. "
              "Each skill alerts only once its condition has stayed true for its hold: "
              "event_recall_in_window counts an alert within the skill's window after the hold "
              "ends, and time_to_alert_median_s is counted from the end of the hold. Holds and "
              f"windows: {on_time}")
    if footage := data.get("footage"):
        source = f"Test footage: {footage}. " + source
        if "EPIC-KITCHENS" in footage:
            source += " " + EPIC_CREDIT
    if data.get("threshold_objective") == "frames":
        source += (" Thresholds were calibrated for frame F1 (this footage has too few events "
                   "past the hold to judge alerts).")
    columns = ("evaluation", "experiment", "skill", "variant", *METRICS, "usd_per_hour")
    rows = [[e["name"], e["experiment"], e["skill"], e["variant"],
             *(e["metrics"].get(k) for k in METRICS), e["cost_per_hour"].get("total_usd")]
            for e in scored]
    return "\n".join([source, "", "| " + " | ".join(columns) + " |", "|---" * len(columns) + "|"]
                     + ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows])


def model_card(out_path: Path, *, run: str, bundle_manifest: Path, kind: str = "adapter",
               results: Path | None = None, license_id: str = BASE_LICENSE,
               models: Mapping[str, str] = DEFAULT_MODELS, max_steps: int | None = None,
               keyframes: Iterable[str] = (), spoken: bool = True) -> Path:
    """Write the model card (README.md with YAML front matter) to out_path. `max_steps` is
    the run's fixed step count (te train --max-steps), if it had one; `keyframes` are the
    skills whose face-blurred keyframes the release holds; `spoken` False leaves Voice
    Transcribe out of the models that made the data (every request was typed)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if not LICENSE_ID.fullmatch(license_id or ""):
        raise ValueError(f"license {license_id!r}: use a Hugging Face license id")
    manifest = json.loads(Path(bundle_manifest).read_text())
    title, relation, front, what = KINDS[kind]
    base = models["student_base"]
    generated, mix = generated_text(*source_counts(manifest), synthetic_skills(manifest),
                                    keyframes)
    rows = [f"| `{k}` | `{_cell(RECIPE[k])}` |" for k in RECIPE_KEYS]
    steps = ""
    if max_steps:
        rows.append(f"| `max_steps` | `{max_steps}` |")
        steps = (f", and a fixed {max_steps} optimizer steps in place of one epoch (the "
                 "experiments keep the steps the same for every run)")
    Path(out_path).write_text(CARD.format(
        base=base, relation=relation, title=title, what=what,
        front=front + f"license: {license_id}\n",
        revision=STUDENT_BASE_REVISION, run=run, skills=", ".join(sorted(manifest["skills"])),
        batch=GLOBAL_BATCH, models=", ".join(f"{job} `{models[job]}`" for job in DATA_JOBS
                                             if spoken or job != "transcribe"),
        recipe="\n".join(rows), steps=steps, generated=generated, mix=mix,
        data=data_section(manifest), results=results_section(results, run),
        license=license_text(license_id, base)),
        encoding="utf-8")
    return Path(out_path)


def license_text(license_id: str, base: str) -> str:
    """The card's License section: the license, that these weights modify the base model,
    and Meta's Usage Policy, which applies to any use of Muse Glimmer."""
    same = (" as the base model's" if license_id == BASE_LICENSE
            else f"; the base model's is `{BASE_LICENSE}`")
    return (f"`{license_id}`{same} ({base} at revision `{STUDENT_BASE_REVISION}`; its "
            "license text is in `LICENSE`). These weights are a modification of that model: "
            "QLoRA fine-tuning on the data above (for a merged build, that adapter merged "
            f"into the base). Any use is also subject to Meta's [Muse Glimmer Usage Policy]"
            f"({USAGE_POLICY}).")
