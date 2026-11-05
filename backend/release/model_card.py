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
from backend.dataengine.dataset import IMAGE_SIZE, SYSTEM_PROMPT
from backend.eval.readme import per_skill, summary
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
WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}
CODE_URL = "https://github.com/amaan784/trained-eye"
PLAYGROUND_URL = "https://aas2438--trained-eye-playground-web.us-east.modal.run"
WANDB_URL = "https://wandb.ai/amaan784-columbia-university/trained-eye"
JUDGES = {"finetuned": "this adapter", "base": "base Muse Glimmer 30B",
          "spark": "Muse Spark (API)"}
FILES = (  # (name, what it is), in the card's order; shown when the release has the file
    ("adapter_model.safetensors", ("The LoRA weights: rank {rank} on every attention and MLP "
     "projection of the language model's layers; the vision tower is unchanged. {size}")),
    ("adapter_config.json", ("PEFT's settings for those weights: the base model, the rank, "
     "alpha and the target modules.")),
    ("config.json", "The model configuration the training run saved with the adapter."),
    ("processor_config.json", "The image processor's settings, saved with the adapter."),
    ("tokenizer.json", "The tokenizer, saved with the adapter."),
    ("tokenizer_config.json", "The tokenizer's settings, saved with the adapter."),
    ("chat_template.jinja", "The chat template the prompts were built with."),
    ("manifests/", ("One line per generated training image ({manifests}): Muse Image's edit "
     "prompt, the answer it was made for (yes, or no for a near-miss), and the SHA-256 of the "
     "frame it edited and of the image. The images themselves are not here.")),
    ("keyframes/", "Face-blurred teaching frames, for rebuilding the generated images."),
    ("LICENSE", "The base model's license text (Apache-2.0)."),
    ("release.json", "Every file here with its size and SHA-256."),
    ("README.md", "This card."),
)
LICENSE_ID = re.compile(r"[a-z0-9][a-z0-9.+-]*")
CARD = """\
---
base_model: {base}
base_model_relation: {relation}
{front}language: [en]
pipeline_tag: image-text-to-text
tags: [trained-eye, qlora, lora, axolotl, smart-glasses, egocentric-vision]
---

# Trained Eye: watch skills for Muse Glimmer 30B ({title})

**What this is.** {what} `{base}` (revision `{revision}`), from
[Trained Eye]({code}), which lets Meta glasses watch for something and say when it
happens. Given one first-person camera frame and a condition in the prompt, it answers
yes or no. It was trained on {n_skills} skill{plural} at once (the training run is called
`{run}`): {skill_list}. {standalone}

[Code]({code}) | [Playground]({playground}) | [Training runs (W&B)]({wandb}) |
[How it works]({playground}/how-it-works)

## What is in this repository

| File | What it is |
|---|---|
{files}

Not in this repository:

- {base_note}
- The training images. The generated images are never published (they keep Muse Image's
  provenance and stay with the author), and neither are the teaching frames they were
  edited from. `manifests/` says how each generated image was made.
- The test clips and their labels, and the training code, which is on [GitHub]({code}).

## How to use it

Send each frame the way the training examples were built:

- System message: `{system}`
- User message: the frame, padded (not cropped) to {size} x {size} pixels with black
  bars, then this text: `Condition: <what to watch for>. Not: <a near-miss>; <another>.
  Is the condition true in this frame?`

{conditions}

Glimmer writes a short reasoning before it answers, so read the last `yes` or `no` token it
generates, and take the chance of yes from that token's top logprobs (yes over yes plus
no). An alert should wait until that chance has stayed over a threshold, calibrated on
your own labelled clips, for the skill's hold (Trained Eye's watcher does this).

What ran on Nov 4: the adapter merged into the bf16 base with PEFT's
`merge_and_unload()` on one 80 GB H100, then served with vLLM's
`vllm/vllm-openai:muse-glimmer` image, each request with `logprobs` on, `top_logprobs`
10, up to 512 new tokens and `chat_template_kwargs` `{{"reasoning_strength": "low"}}`.
Loading the adapter with vLLM's `--enable-lora` did not start for this model on that
image. The merge, as the project's `te export` runs it:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base, revision = "{base}", "{revision}"
model = AutoModelForImageTextToText.from_pretrained(
    base, revision=revision, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "{repo}").merge_and_unload()
model.save_pretrained("trained-eye-merged")
AutoProcessor.from_pretrained(base, revision=revision).save_pretrained("trained-eye-merged")
```

## Results

{results}

The full tables, the EPIC-KITCHENS-100 results and how it was measured are on GitHub:
[results]({code}/blob/main/eval/results.md), [EPIC-KITCHENS]({code}/blob/main/eval/epic/results.md),
[method]({code}/blob/main/docs/evaluation.md).

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
- In plain words: overall it is about as accurate as base Glimmer and faster; it is the best
  of the three on the fridge and the weakest on the burner, so do not read it as an accuracy
  gain from fine-tuning on every skill.
- It watches only what the glasses can see: look away and it cannot tell whether the
  condition is still true.
- If Muse Spark's check fails outright (an error, not a no), the alert is still spoken and
  logged as unconfirmed, on purpose, so a cloud outage never silences an alert; a no from
  Spark holds it back.
- The test labels were drafted by an AI assistant, too few events lasted past the hold to
  judge alerts well, and Experiments 2 and 3 did not run.
- Validation during training was split by example, not by source frame, so related images
  can sit on both sides; the separate test footage is not affected.

## Privacy and provenance

- Faces are blurred in everything published with this model. Watch mode does not store
  frames unless the user saves a clip, and never hides the glasses' capture LED.
- Muse Image outputs keep their watermark and provenance metadata, byte for byte, and are
  never published.

## License

{license}

## Citation

```bibtex
@software{{sheikh2026trainedeye,
  author = {{Sheikh, Amaan}},
  title  = {{Trained Eye: Teach Your Glasses to Notice Anything}},
  year   = {{2026}},
  month  = nov,
  url    = {{{code}}}
}}
```
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
        intro += (f" Generated: images Muse Image made by editing those same keyframes (the "
                  f"condition, near-misses, other light and angles), labelled by Muse Spark and "
                  f"{spark}.")
    head = "| Skill | Split | Real yes | Real no | Generated yes | Generated no |"
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
    full = "\n".join(["| " + " | ".join(columns) + " |", "|---" * len(columns) + "|"]
                     + ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows])
    short = summary(data)
    skills = per_skill({"evaluations": evaluations})
    return "\n".join([*([f"In short: {short}", ""] if short else []),
                      *([skills, ""] if skills else []),
                      plain_table(scored), "", source, "",
                      "<details><summary>Every scored evaluation</summary>", "", full, "",
                      "</details>"])


def plain_table(scored: list[dict]) -> str:
    """Per skill and model: frame precision, recall, F1 and accuracy, the median seconds a
    frame and the dollars an hour, copied from results.json."""
    def num(value, spec=".2f"):
        return "n/a" if value is None else f"{value:{spec}}"

    rows = [("| Skill | Model | Frame precision | Frame recall | Frame F1 | Frame accuracy | "
            "Seconds per frame | Dollars per hour |"), "|---|---|---|---|---|---|---|---|"]
    order = list(JUDGES)
    for e in sorted(scored, key=lambda e: (e["skill"], order.index(e["variant"])
                                           if e["variant"] in order else len(order))):
        m, usd = e["metrics"], e["cost_per_hour"].get("total_usd")
        rows.append(f"| {e['skill']} | {_cell(JUDGES.get(e['variant'], e['variant']))} | "
                    f"{num(m.get('precision'))} | {num(m.get('recall'))} | {num(m.get('f1'))} | "
                    f"{num(m.get('accuracy'))} | {num(m.get('latency_median_s'))} | "
                    f"{'n/a' if usd is None else f'${usd:.2f}'} |")
    return "\n".join(rows)


def model_card(out_path: Path, *, run: str, bundle_manifest: Path, kind: str = "adapter",
               results: Path | None = None, license_id: str = BASE_LICENSE,
               models: Mapping[str, str] = DEFAULT_MODELS, max_steps: int | None = None,
               keyframes: Iterable[str] = (), spoken: bool = True,
               specs: Mapping[str, dict] | None = None, repo: str | None = None) -> Path:
    """Write the model card (README.md with YAML front matter) to out_path. `max_steps` is
    the run's fixed step count (te train --max-steps), if it had one; `keyframes` are the
    skills whose face-blurred keyframes the release holds; `spoken` False leaves Voice
    Transcribe out of the models that made the data (every request was typed); `specs`
    are the skills' spec.json ({skill: spec}), for the conditions word for word; `repo` is
    the Hugging Face repo, for the merge example."""
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
    skills = sorted(manifest["skills"])
    asked = {k: (v or {}).get("request") for k, v in (specs or {}).items()}
    skill_list = ", ".join(f"`{k}`" + (f" (\"{asked[k]}\")" if asked.get(k) else "")
                           for k in skills)
    standalone = ("It is not a model on its own: it needs the base model's weights."
                  if kind == "adapter" else "The adapter is already merged into the base.")
    base_note = (f"The base model's weights: get `{base}` at revision `{STUDENT_BASE_REVISION}` "
                 "from Hugging Face, under its own terms." if kind == "adapter" else
                 "The adapter on its own: this is the merged build.")
    Path(out_path).write_text(CARD.format(
        code=CODE_URL, playground=PLAYGROUND_URL, wandb=WANDB_URL,
        n_skills=WORDS.get(len(skills), str(len(skills))), plural="" if len(skills) == 1 else "s", skill_list=skill_list,
        standalone=standalone, base_note=base_note, files=files_table(Path(out_path).parent),
        system=SYSTEM_PROMPT, size=IMAGE_SIZE, conditions=conditions_text(specs or {}),
        repo=repo or "<owner>/<repo>",
        base=base, relation=relation, title=title, what=what,
        front=front + f"license: {license_id}\n",
        revision=STUDENT_BASE_REVISION, run=run,
        batch=GLOBAL_BATCH, models=", ".join(f"{job} `{models[job]}`" for job in DATA_JOBS
                                             if spoken or job != "transcribe"),
        recipe="\n".join(rows), steps=steps, generated=generated, mix=mix,
        data=data_section(manifest), results=results_section(results, run),
        license=license_text(license_id, base)),
        encoding="utf-8")
    return Path(out_path)


def files_table(folder: Path) -> str:
    """The card's file table: each known file the release folder has (LICENSE, release.json
    and README.md always, as prepare_release writes them)."""
    rows = []
    adapter = folder / "adapter_model.safetensors"
    size = f"{adapter.stat().st_size / 1e6:,.0f} MB." if adapter.is_file() else ""
    counts = [f"{len(p.read_text().splitlines())} for {p.stem}"
              for p in sorted((folder / "manifests").glob("*.jsonl"))]
    always = ("LICENSE", "release.json", "README.md")
    for name, what in FILES:
        if name in always or (folder / name.rstrip("/")).exists():
            text = what.format(rank=RECIPE["lora_r"], size=size,
                               manifests=", ".join(counts) or "none")
            shown = f"`{name}<skill>.jsonl`" if name == "manifests/" else f"`{name}`"
            rows.append(f"| {shown} | {text.strip()} |")
    return "\n".join(rows)


def conditions_text(specs: Mapping[str, dict]) -> str:
    """The conditions each skill was trained with, word for word, from its spec.json."""
    if not specs:
        return "The skills' own conditions are in each skill's spec.json from te teach."
    rows = ["The conditions it was trained with, word for word:", "",
            "| Skill | Asked as | Condition | Not |", "|---|---|---|---|"]
    for skill, spec in sorted(specs.items()):
        nots = "; ".join(n.strip().rstrip(".") for n in spec.get("negatives") or [])
        rows.append(f"| {skill} | {_cell(spec.get('request') or '')} | "
                    f"{_cell(spec['positive'].strip().rstrip('.'))} | {_cell(nots)} |")
    return "\n".join(rows)


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
