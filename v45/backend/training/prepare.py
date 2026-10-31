"""Bundle taught skills into one training set for Modal.

One adapter learns every skill: the condition is in the prompt, so examples from all
skills go into one file. The bundle is a folder with train.jsonl, val.jsonl (when there
are enough examples), the images, and manifest.json. Its image paths point where the
folder will be on the Modal volume (`image_root`): Meta's recipe reads absolute paths,
and paths from the Mac do not exist in the container.

Variants for Experiment 1: "real" keeps only real frames, "synthetic" only Muse Image
edits, "mixed" both. `limit` keeps a fixed subset per skill, for the image-count curve.
Everything is deterministic: the same inputs give the same bundle and hashes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

VARIANTS = {"real": {"real"}, "synthetic": {"synthetic"}, "mixed": {"real", "synthetic"}}
VAL_SHARE = 0.1
MIN_FOR_VAL = 50  # below this every example trains, and evaluation uses the test footage


def _order(example_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{example_id}".encode()).hexdigest()


def skill_examples(skill_dir: Path) -> list[tuple[dict, dict]]:
    """(record, index entry) for each example of a skill's teach dataset."""
    skill_dir = Path(skill_dir)
    name = skill_dir.name
    dataset = skill_dir / "dataset"
    manifest_path = dataset / f"{name}-teach.manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{name} has no dataset yet: run te teach for it first")
    manifest = json.loads(manifest_path.read_text())
    body = (dataset / f"{name}-teach.jsonl").read_text(encoding="utf-8")
    if hashlib.sha256(body.encode("utf-8")).hexdigest() != manifest["jsonl_sha256"]:
        raise ValueError(f"{name}: the dataset does not match its manifest; run te teach again")
    index = {e["id"]: e for e in manifest["examples_index"]}
    return [(record, index[record["id"]]) for record in map(json.loads, body.splitlines())]


def bundle(skill_dirs: list[Path], out_dir: Path, image_root: str, *, variant: str = "mixed",
           limit: int | None = None, salt: str = "trained-eye") -> dict:
    """Write the bundle to `out_dir` (replacing it) and return its manifest."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {', '.join(VARIANTS)}")
    if not image_root.startswith("/"):
        raise ValueError("image_root must be an absolute path in the container")
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "images").mkdir(parents=True)
    chosen, sources = [], {}
    for skill_dir in sorted(Path(d) for d in skill_dirs):
        examples = [(r, e) for r, e in skill_examples(skill_dir)
                    if e["source"] in VARIANTS[variant]]
        examples.sort(key=lambda pair: _order(pair[0]["id"], salt))
        if limit is not None:
            examples = examples[:limit]
        if not examples:
            raise ValueError(f"{skill_dir.name} has no {variant} examples")
        sources[skill_dir.name] = json.loads(
            (skill_dir / "dataset" / f"{skill_dir.name}-teach.manifest.json").read_text()
        )["jsonl_sha256"]
        chosen += [(skill_dir, r, e) for r, e in examples]

    use_val = len(chosen) >= MIN_FOR_VAL
    splits: dict[str, list[str]] = {"train": [], "val": []}
    counts: dict[str, int] = {}
    for skill_dir, record, entry in sorted(chosen, key=lambda c: _order(c[1]["id"], salt)):
        image_part = record["messages"][-2]["content"][0]
        source_png = skill_dir / "dataset" / "images" / Path(image_part["path"]).name
        target = out_dir / "images" / f"{record['id']}.png"
        if not source_png.is_file():
            raise ValueError(f"{skill_dir.name}: missing image {source_png}; run te teach again")
        shutil.copyfile(source_png, target)
        image_part["path"] = f"{image_root.rstrip('/')}/images/{target.name}"
        val = use_val and int(_order(record["id"], salt + "|val")[:8], 16) / 16**8 < VAL_SHARE
        split = "val" if val else "train"
        splits[split].append(json.dumps(record, sort_keys=True))
        key = f"{split}/{skill_dir.name}/{entry['source']}/{entry['answer']}"
        counts[key] = counts.get(key, 0) + 1

    manifest = {"variant": variant, "limit": limit, "image_root": image_root,
                "skills": sources, "counts": dict(sorted(counts.items()))}
    for split, lines in splits.items():
        if not lines:
            continue
        text = "\n".join(lines) + "\n"
        (out_dir / f"{split}.jsonl").write_text(text, encoding="utf-8")
        manifest[f"{split}_examples"] = len(lines)
        manifest[f"{split}_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest
