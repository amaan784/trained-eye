"""The Hugging Face release: the model card, the release folder and the upload.

Every number in the card comes from a file: axolotl.RECIPE, the training bundle's
manifest.json, and only the eval harness's results.json for results. The weights take the
base model's license: the Muse Glimmer model card at the pinned revision (read on
2026-11-02) says apache-2.0, with Meta's Usage Policy beside it, so the card says so,
states that the weights are a modification, and the release folder carries the base
model's LICENSE (BASE_LICENSE.txt, copied from that revision). `upload` imports
huggingface_hub only when called and refuses without confirmed=True (the author's yes).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path

from backend.config import DEFAULT_MODELS, STUDENT_BASE_REVISION, load_settings
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
BASE_LICENSE_FILE = Path(__file__).with_name("BASE_LICENSE.txt")
USAGE_POLICY = "https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/USAGE_POLICY.md"
KINDS = {  # title, base_model_relation, front matter, what it is
    "adapter": ("LoRA adapter", "adapter", "library_name: peft\n", "A QLoRA adapter (PEFT) for"),
    "merged": ("merged build", "finetune", "", "The final adapter merged (`merge_and_unload()`) into"),
}
WEIGHT_SUFFIXES = (".safetensors", ".json", ".jinja", ".model", ".txt", ".gguf")
MEDIA_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".mp4", ".mov")
REPO_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*")
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
vehicle (has the washer's display reached the end of the cycle?), with an alert threshold
calibrated on the user's own validation clips.

Out of scope: people (identifying, recognising or tracking anyone), bodies, health,
medication and other sensitive traits, covert monitoring, sensitive places, sports analysis
or coaching, and anything where a missed or false alert could put someone at risk.

## Limitations

- Each skill comes from one user's scenes and camera; others are untested beyond the
  evaluation above. {mix}
- It answers per frame: alerts need the watcher's smoothing, a calibrated threshold and a
  cooldown. The vision tower stayed frozen, so small, distant or glary targets stay hard.
- Glimmer reasons before it answers: scoring needs the server's top logprobs for yes and no.

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


def source_counts(manifest: dict) -> tuple[int, int]:
    """(real, synthetic) examples in a bundle manifest, both splits."""
    real = synthetic = 0
    for key, n in manifest["counts"].items():
        source = key.split("/")[2]
        real += n if source == "real" else 0
        synthetic += n if source == "synthetic" else 0
    return real, synthetic


def generated_text(real: int, synthetic: int) -> tuple[str, str]:
    """The card's paragraph on generated images and its limitation, from the counts."""
    if not synthetic:
        about = ("There are no generated images in this bundle: it holds the user's own "
                 "keyframes only.")
        limit = "The training images are the user's own keyframes, so there were few of them."
        return about, limit
    share = "Most" if synthetic > real else "Some"
    about = ("The generated images are not included. `manifests/` lists every edit prompt with "
             "the SHA-256 of its keyframe and of its image; `te regenerate` rebuilds them from "
             "the original keyframes.")
    limit = (f"{share} training images are Muse Image edits with Muse Spark's labels, which "
             "can be wrong.")
    return about, limit


def results_section(results: Path | None) -> str:
    """The scored evaluations of `te eval`'s results.json, its numbers copied as they are."""
    if results is None:
        return "Not evaluated yet: no results from the eval harness were given."
    body = Path(results).read_bytes()
    evaluations = json.loads(body)["evaluations"]
    scored = [e for e in evaluations if e["status"] == "scored"]
    if not scored:
        return f"Not evaluated yet: `{Path(results).name}` has no scored evaluation."
    source = (f"From `te eval`'s `{Path(results).name}` (SHA-256 "
              f"`{hashlib.sha256(body).hexdigest()}`); evaluations scored: {len(scored)}, pending: "
              f"{len(evaluations) - len(scored)}. n/a means there was nothing to compute it from.")
    columns = ("evaluation", "experiment", "skill", "variant", *METRICS, "usd_per_hour")
    rows = [[e["name"], e["experiment"], e["skill"], e["variant"],
             *(e["metrics"].get(k) for k in METRICS), e["cost_per_hour"].get("total_usd")]
            for e in scored]
    return "\n".join([source, "", "| " + " | ".join(columns) + " |", "|---" * len(columns) + "|"]
                     + ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows])


def model_card(out_path: Path, *, run: str, bundle_manifest: Path, kind: str = "adapter",
               results: Path | None = None, license_id: str = BASE_LICENSE,
               models: Mapping[str, str] = DEFAULT_MODELS, max_steps: int | None = None) -> Path:
    """Write the model card (README.md with YAML front matter) to out_path. `max_steps` is
    the run's fixed step count (te train --max-steps), if it had one."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if not LICENSE_ID.fullmatch(license_id or ""):
        raise ValueError(f"license {license_id!r}: use a Hugging Face license id")
    manifest = json.loads(Path(bundle_manifest).read_text())
    title, relation, front, what = KINDS[kind]
    base = models["student_base"]
    generated, mix = generated_text(*source_counts(manifest))
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
        batch=GLOBAL_BATCH, models=", ".join(f"{job} `{models[job]}`" for job in DATA_JOBS),
        recipe="\n".join(rows), steps=steps, generated=generated, mix=mix,
        data=data_section(manifest), results=results_section(results),
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


def _sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def prepare_release(out_dir: Path, *, run: str, bundle_manifest: Path, manifests_dir: Path,
                    weights_dir: Path | None = None, **card) -> dict:
    """Assemble the folder to upload (uploading nothing): README.md (the card; `card` holds
    kind, results, license_id, models), weights_dir's top-level weight and config files (no
    pickles, checkpoints or trainer README), the manifests of the skills with synthetic data,
    and release.json listing every file with its SHA-256."""
    out_dir, manifests_dir = Path(out_dir), Path(manifests_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not (out_dir / "release.json").is_file():
        raise ValueError(f"{out_dir} is not empty and not a release folder: pick another")
    keys = [key.split("/") for key in json.loads(Path(bundle_manifest).read_text())["counts"]]
    synthetic = sorted({k[1] for k in keys if k[2] == "synthetic"})
    if missing := [s for s in synthetic if not (manifests_dir / f"{s}.jsonl").is_file()]:
        raise ValueError(f"no prompt manifest for {', '.join(missing)} in {manifests_dir}: "
                         "run te manifests first")
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "manifests").mkdir(parents=True)
    skipped = []
    for path in sorted(Path(weights_dir).iterdir()) if weights_dir else []:
        if not path.is_file() or path.suffix not in WEIGHT_SUFFIXES or path.name == "README.md":
            skipped.append(path.name)
            continue
        shutil.copy2(path, out_dir / path.name)
    for skill in synthetic:
        shutil.copyfile(manifests_dir / f"{skill}.jsonl", out_dir / "manifests" / f"{skill}.jsonl")
    model_card(out_dir / "README.md", run=run, bundle_manifest=bundle_manifest, **card)
    shutil.copyfile(BASE_LICENSE_FILE, out_dir / "LICENSE")  # Apache-2.0 4(a): ship the text
    release = {"run": run, "kind": card.get("kind", "adapter"), "skipped": skipped,
               "files": [{"path": p.relative_to(out_dir).as_posix(), "bytes": p.stat().st_size,
                          "sha256": _sha256(p)} for p in sorted(out_dir.rglob("*")) if p.is_file()]}
    (out_dir / "release.json").write_text(json.dumps(release, indent=1) + "\n")
    return release


def upload(folder: Path, repo_id: str, token_env: str = "HF_TOKEN", *, confirmed: bool = False,
           private: bool = True) -> str:
    """Publish a prepare_release folder to `repo_id` (private unless private=False). Refuses
    without confirmed=True, without the base model's LICENSE, or with an image or video in
    it."""
    if not confirmed:
        raise PermissionError("publishing needs the author's yes: nothing was uploaded")
    folder = Path(folder)
    if not REPO_ID.fullmatch(repo_id) or not (folder / "release.json").is_file():
        raise ValueError(f"need a repo id like owner/name and a folder from te release: "
                         f"{repo_id!r}, {folder}")
    if not (folder / "LICENSE").is_file():
        raise ValueError("the release folder has no LICENSE (the base model's): run te "
                         "release again")
    if media := sorted(p.name for p in folder.rglob("*") if p.suffix.lower() in MEDIA_SUFFIXES):
        raise ValueError(f"images or video in the release folder ({', '.join(media[:5])}): "
                         "generated images are never published")
    release = json.loads((folder / "release.json").read_text())
    names = {f["path"] for f in release["files"]}
    needed = "adapter_config.json" if release.get("kind", "adapter") == "adapter" else "config.json"
    if needed not in names or not any(n.endswith(".safetensors") for n in names):
        raise ValueError(f"the release folder has no {release.get('kind', 'adapter')} weights "
                         f"({needed} and .safetensors): run te release with --weights")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise RuntimeError("uploading needs huggingface_hub, which is not installed and not on "
                           "the approved downloads list: ask the author, then "
                           "pip install huggingface_hub") from None
    token = load_settings().secret(token_env)  # from the environment or .env
    def failed(err: Exception) -> RuntimeError:  # its text must not carry the token
        return RuntimeError(f"upload failed ({type(err).__name__}): "
                            f"{str(err).replace(token, '[redacted]')[:300]}")

    try:
        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        # an existing repo keeps its visibility: never publish after a "private" question
        actual = getattr(api.repo_info(repo_id, repo_type="model"), "private", None)
    except Exception as err:  # noqa: BLE001
        raise failed(err) from None
    if actual is not private:
        raise ValueError(f"{repo_id} already exists and is "
                         f"{'private' if actual else 'public'}, not "
                         f"{'private' if private else 'public'}: nothing was uploaded")
    try:
        api.upload_folder(repo_id=repo_id, folder_path=str(folder), repo_type="model",
                          commit_message=f"Trained Eye release {folder.name}")
    except Exception as err:  # noqa: BLE001
        raise failed(err) from None
    return f"https://huggingface.co/{repo_id}"
