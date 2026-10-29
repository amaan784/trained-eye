"""The teach flow for one skill, end to end.

  voice (Muse Voice Transcribe) -> skill spec (Muse Wonder) -> read-back and confirm
  -> region of interest (SAM 3.1) -> synthetic variations (Muse Image)
  -> labels (Muse Spark) -> Axolotl dataset

Every step writes its output under data/skills/<skill>/ and every Meta call is cached,
so a rerun after a crash or a change further down costs nothing for the steps that
already ran.

If training on Meta model outputs is not allowed (settings.train_on_model_outputs is
False, see SPEC.md "Fallback"), generated images and Spark labels are still produced
for evaluation, but the dataset holds only real frames with the user's own labels.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from backend.dataengine import dataset, labels, roi, synth
from backend.dataengine.keyframes import load_frames, select_keyframes
from backend.meta.cache import sha256_hex
from backend.meta.client import EXTENSIONS, MetaClient, sniff_mime, wav_duration_s
from backend.meta.ledger import token_cost, unit_cost
from backend.skills.compiler import compile_skill
from backend.skills.spec import SkillSpec, check_name, read_back

IMAGE_TOKENS = 1500  # guess per image for estimates only


@dataclass
class TeachOptions:
    skill: str
    clip: Path
    audio: Path | None = None
    request: str | None = None
    positives: int = 150
    negatives: int = 150
    audio_s: float | None = None  # length of the audio, if the caller measured it
    real_labels: dict[str, str] = field(default_factory=dict)  # keyframe file -> yes/no


def estimate(client: MetaClient, *, audio_s: float, keyframes: int, positives: int,
             negatives: int) -> dict[str, float]:
    s = client.settings
    images = positives + negatives
    parts = {
        "transcribe": unit_cost(s.price(s.model("transcribe")), audio_s=audio_s),
        "compile": token_cost(s.price(s.model("compiler")), 3 * IMAGE_TOKENS + 900, 4096),
        "sam": unit_cost(s.price(s.model("sam")), images=keyframes),
        "images": unit_cost(s.price(s.model("image")), images=images),
        "labels": (keyframes + images)
        * token_cost(s.price(s.model("labeler")), IMAGE_TOKENS + 250, 1024),
    }
    parts["total"] = sum(parts.values())
    return parts


def spread(items: list, count: int) -> list:
    """`count` items evenly spread over the list (first and last included)."""
    if len(items) <= count:
        return list(items)
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


def run_teach(
    client: MetaClient,
    opts: TeachOptions,
    out_root: Path,
    confirm: Callable[[str], bool],
    say: Callable[[str], None] = print,
) -> dict:
    check_name(opts.skill)
    root = Path(out_root) / opts.skill
    root.mkdir(parents=True, exist_ok=True)
    report: dict = {"skill": opts.skill}

    # 1. keyframes (free)
    frames = select_keyframes(load_frames(opts.clip))
    if not frames:
        raise ValueError(f"no frames found in {opts.clip}")
    kf_dir = root / "keyframes"
    kf_dir.mkdir(exist_ok=True)
    keyframes = {}
    for i, frame in enumerate(frames):
        name = f"kf_{i:02d}.{EXTENSIONS.get(sniff_mime(frame.data), 'jpg')}"
        (kf_dir / name).write_bytes(frame.data)
        keyframes[name] = frame.data
    say(f"keyframes: {len(frames)} kept")
    if not client.settings.train_on_model_outputs and not any(
            opts.real_labels.get(name) in ("yes", "no") for name in keyframes):
        report["stopped"] = ("training on model outputs is off, so the dataset needs your own "
                             f"labels: label the frames in {kf_dir} and pass --labels")
        return report

    # 2. request text
    if opts.audio:
        audio = Path(opts.audio).read_bytes()
        duration = wav_duration_s(audio) or opts.audio_s or 10.0
        transcript = client.transcribe(audio, filename=Path(opts.audio).name, skill=opts.skill,
                                       duration_s=duration)
        request = transcript.text.strip()
    else:
        request = (opts.request or "").strip()
    if not request:
        raise ValueError("no request: pass audio or request text")
    _write_json(root / "request.json", {"request": request, "from_audio": bool(opts.audio)})
    say(f'request: "{request}"')

    # 3. skill spec, read back, confirm
    spec = compile_skill(client, request, spread(list(keyframes.values()), 3), name=opts.skill)
    _write_json(root / "spec.json", spec.model_dump())
    sentence = read_back(spec)
    say(sentence)
    report.update(refused=spec.refused, read_back=sentence)
    if spec.refused or not confirm(sentence):
        report["stopped"] = "refused" if spec.refused else "not confirmed"
        return report

    # 4. region of interest
    rois = roi.locate(client, spec, list(keyframes.values()))
    roi.save(rois, root / "roi.json")
    report["roi_frames_with_box"] = sum(1 for r in rois if r.best)

    # 5. synthetic variations
    by_hash = {sha256_hex(data): data for data in keyframes.values()}
    plan = synth.plan_variations(spec, list(by_hash), opts.positives, opts.negatives)
    generated = synth.generate(client, plan, by_hash, root / "synthetic")
    say(f"generated: {len(generated)} images")

    # 6. labels for every image, real and generated
    label_rows = []
    for name, data in keyframes.items():
        lab = labels.label_image(client, spec, data)
        label_rows.append({"file": f"keyframes/{name}", "source": "real", "spark_label": lab.label,
                           "reason": lab.reason, "human_label": opts.real_labels.get(name)})
    for row in generated:
        data = (root / "synthetic" / row["file"]).read_bytes()
        lab = labels.label_image(client, spec, data)
        label_rows.append({**row, "file": f"synthetic/{row['file']}", "source": "synthetic",
                           "spark_label": lab.label, "reason": lab.reason})
    _write_jsonl(root / "labels.jsonl", label_rows)
    kept, stats = labels.filter_generated([r for r in label_rows if r["source"] == "synthetic"])
    _write_json(root / "label_stats.json", stats)
    say(f"labels: kept {stats['kept']} of {stats['total']} generated "
        f"(agreement {stats['agreement']})")

    # 7. dataset
    examples = _examples(spec, root, label_rows, kept, client.settings.train_on_model_outputs)
    if not examples:
        report["stopped"] = "no usable examples"
        return report
    manifest = dataset.write_dataset(examples, root / "dataset", f"{opts.skill}-teach")
    report.update(dataset=manifest, label_stats=stats)
    say(f"dataset: {manifest['examples']} examples -> {root / 'dataset'}")
    return report


def _examples(spec: SkillSpec, root: Path, rows: list[dict], kept: list[dict],
              allow_model_outputs: bool) -> list[dataset.Example]:
    out = []
    for row in rows:
        if row["source"] != "real":
            continue
        answer = row.get("human_label") or (row["spark_label"] if allow_model_outputs else None)
        if answer in ("yes", "no"):
            out.append(_example(spec, root, row, answer))
    if allow_model_outputs:
        out += [_example(spec, root, row, row["label"]) for row in kept]
    return out


def _example(spec: SkillSpec, root: Path, row: dict, answer: str) -> dataset.Example:
    stem = Path(row["file"]).stem
    return dataset.Example(
        id=f"{spec.name}-{row['source'][:4]}-{stem}", image=(root / row["file"]).read_bytes(),
        answer=answer, source=row["source"], skill=spec.name, positive=spec.positive,
        negatives=list(spec.negatives),
    )


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), "utf-8")
