"""Bundle taught skills into one training set for Modal.

One adapter learns every skill: the condition is in the prompt, so examples from all
skills go into one file. The bundle is a folder with train.jsonl, val.jsonl (when there
are enough examples), the images, and manifest.json. Its image paths point where the
folder will be on the Modal volume (`image_root`): Meta's recipe reads absolute paths,
and paths from the Mac do not exist in the container.

Variants for Experiment 1: "real" keeps only real frames, "synthetic" only Muse Image
edits, "mixed" both. Real frames are the teach dataset's keyframes (and their crops) and,
when te clips has written the skill's dataset-clips (backend/dataengine/realclips.py), the
frames of the labelled session 1 and 2 clips: session 1 frames always train and session 2
frames are always in the val split, never trained on. The rest of the val split is a
fixed 10% of the teach examples, once there are MIN_FOR_VAL of them. `limit` keeps a fixed
subset per skill (session 2 frames not counted), for the image-count curve. Everything is
deterministic: the same inputs give the same bundle and hashes.

All skills in a bundle must have been written the same way (the dataset manifests'
`system_in_user`, from TE_SYSTEM_IN_USER); the bundle records it, and watch mode asks the
trained student the same way.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

VARIANTS = {"real": {"real"}, "synthetic": {"synthetic"}, "mixed": {"real", "synthetic"}}
CLIPS_DIR = "dataset-clips"  # te clips (backend/dataengine/realclips.py)
CLIP_VARIANTS = ("real", "mixed")
VAL_SHARE = 0.1
MIN_FOR_VAL = 50  # below this every teach example trains; evaluation uses the test footage


def _order(example_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{example_id}".encode()).hexdigest()


def _read(skill_dir: Path, dataset_dir: str, kind: str, rerun: str
          ) -> tuple[dict, list[tuple[dict, dict]]]:
    """(manifest, [(record, index entry)]) of <skill>-<kind> in a skill's dataset folder,
    checked against its manifest."""
    name, dataset = skill_dir.name, skill_dir / dataset_dir
    manifest = json.loads((dataset / f"{name}-{kind}.manifest.json").read_text())
    body = (dataset / f"{name}-{kind}.jsonl").read_text(encoding="utf-8")
    if hashlib.sha256(body.encode("utf-8")).hexdigest() != manifest["jsonl_sha256"]:
        raise ValueError(f"{name}: {dataset_dir} does not match its manifest; run {rerun} "
                         "again")
    index = {e["id"]: e for e in manifest["examples_index"]}
    return manifest, [(r, index[r["id"]]) for r in map(json.loads, body.splitlines())]


def skill_examples(skill_dir: Path, dataset_dir: str = "dataset") -> list[tuple[dict, dict]]:
    """(record, index entry) for each example of a skill's teach dataset."""
    skill_dir = Path(skill_dir)
    name = skill_dir.name
    if not (skill_dir / dataset_dir / f"{name}-teach.manifest.json").is_file():
        step = ("te teach for it first" if dataset_dir == "dataset"
                else f"te unfiltered --skill {name} first")
        raise ValueError(f"{name} has no {dataset_dir} yet: run {step}")
    rerun = "te teach" if dataset_dir == "dataset" else f"te unfiltered --skill {name}"
    return _read(skill_dir, dataset_dir, "teach", rerun)[1]


def clip_examples(skill_dir: Path) -> tuple[dict, list[tuple[dict, dict]]] | None:
    """(manifest, [(record, index entry)]) of a skill's dataset-clips, or None without one.
    Each index entry's split is "train" (session 1) or "val" (session 2)."""
    skill_dir = Path(skill_dir)
    if not (skill_dir / CLIPS_DIR / f"{skill_dir.name}-clips.manifest.json").is_file():
        return None
    return _read(skill_dir, CLIPS_DIR, "clips", f"te clips --skill {skill_dir.name}")


def _prompts(pairs: list[tuple[dict, dict]]) -> set[str]:
    """The system turn and user text of each record (the image path left out)."""
    return {json.dumps([r["messages"][:-2], r["messages"][-2]["content"][1:]], sort_keys=True)
            for r, _ in pairs}


def prompt_layouts(skill_dirs: list[Path], dataset_dir: str = "dataset") -> dict[str, bool]:
    """{skill: system_in_user} from each skill's dataset manifest that exists."""
    layouts = {}
    for skill_dir in map(Path, skill_dirs):
        path = skill_dir / dataset_dir / f"{skill_dir.name}-teach.manifest.json"
        if path.is_file():
            layouts[skill_dir.name] = bool(json.loads(path.read_text()).get("system_in_user",
                                                                            False))
    return layouts


def bundle(skill_dirs: list[Path], out_dir: Path, image_root: str, *, variant: str = "mixed",
           limit: int | None = None, salt: str = "trained-eye",
           dataset_dir: str = "dataset") -> dict:
    """Write the bundle to `out_dir` (replacing it) and return its manifest. `dataset_dir`
    picks the skill's dataset folder: "dataset" from te teach, or "dataset-unfiltered"
    for Experiment 2's arm without the Spark filter. The real and mixed variants also take
    each skill's dataset-clips when it exists; the manifest's "clips" records them."""
    if dataset_dir not in ("dataset", "dataset-unfiltered"):
        raise ValueError("dataset_dir must be dataset or dataset-unfiltered")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {', '.join(VARIANTS)}")
    if not image_root.startswith("/"):
        raise ValueError("image_root must be an absolute path in the container")
    layouts = prompt_layouts(skill_dirs, dataset_dir)
    if len(set(layouts.values())) > 1:
        folded = ", ".join(sorted(n for n, v in layouts.items() if v))
        apart = ", ".join(sorted(n for n, v in layouts.items() if not v))
        raise ValueError(f"the datasets put the system prompt differently (folded into the "
                         f"user text: {folded}; its own turn: {apart}). Set TE_SYSTEM_IN_USER "
                         "and run te teach again for the odd ones (cached calls are free)")
    if dataset_dir != "dataset":  # Experiment 2's arms must differ in their labels only
        teach = prompt_layouts(skill_dirs, "dataset")
        if odd := sorted(n for n, v in layouts.items() if n in teach and teach[n] != v):
            raise ValueError(f"{dataset_dir} puts the system prompt differently from the "
                             f"teach dataset for {', '.join(odd)}: run te unfiltered --skill "
                             f"{odd[0]} again, so both arms share one layout")
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "images").mkdir(parents=True)
    chosen, sources, clips = [], {}, {}
    for skill_dir in sorted(Path(d) for d in skill_dirs):
        name = skill_dir.name
        taught = skill_examples(skill_dir, dataset_dir)
        pool = [(dataset_dir, r, e) for r, e in taught if e["source"] in VARIANTS[variant]]
        held_out = []
        recorded = clip_examples(skill_dir) if variant in CLIP_VARIANTS else None
        if recorded:
            clip_manifest, frames = recorded
            if _prompts(frames) != _prompts(taught):
                raise ValueError(f"{name}: {CLIPS_DIR} was written for another spec or prompt "
                                 f"layout than {dataset_dir}: run te clips --skill {name} "
                                 "again (free)")
            pool += [(CLIPS_DIR, r, e) for r, e in frames if e.get("split") != "val"]
            held_out = [(CLIPS_DIR, r, e) for r, e in frames if e.get("split") == "val"]
            clips[name] = {"jsonl_sha256": clip_manifest["jsonl_sha256"], "train": 0, "val": 0}
        pool.sort(key=lambda c: _order(c[1]["id"], salt))
        if limit is not None:
            pool = pool[:limit]
        if not pool:
            raise ValueError(f"{name} has no {variant} examples")
        sources[name] = json.loads(
            (skill_dir / dataset_dir / f"{name}-teach.manifest.json").read_text()
        )["jsonl_sha256"]
        chosen += [(skill_dir, folder, r, e) for folder, r, e in pool + held_out]

    use_val = sum(folder != CLIPS_DIR for _, folder, _, _ in chosen) >= MIN_FOR_VAL
    splits: dict[str, list[str]] = {"train": [], "val": []}
    counts: dict[str, int] = {}
    for skill_dir, folder, record, entry in sorted(chosen,
                                                   key=lambda c: _order(c[2]["id"], salt)):
        image_part = record["messages"][-2]["content"][0]
        source_png = skill_dir / folder / "images" / Path(image_part["path"]).name
        target = out_dir / "images" / f"{record['id']}.png"
        if not source_png.is_file():
            rerun = "te teach" if folder != CLIPS_DIR else f"te clips --skill {skill_dir.name}"
            raise ValueError(f"{skill_dir.name}: missing image {source_png}; run {rerun} again")
        if target.exists():
            raise ValueError(f"two examples share the id {record['id']}")
        shutil.copyfile(source_png, target)
        image_part["path"] = f"{image_root.rstrip('/')}/images/{target.name}"
        if folder == CLIPS_DIR:  # by session: 1 trains, 2 validates
            split = "val" if entry.get("split") == "val" else "train"
            clips[skill_dir.name][split] += 1
        else:
            val = use_val and int(_order(record["id"], salt + "|val")[:8], 16) / 16**8 < VAL_SHARE
            split = "val" if val else "train"
        splits[split].append(json.dumps(record, sort_keys=True))
        key = f"{split}/{skill_dir.name}/{entry['source']}/{entry['answer']}"
        counts[key] = counts.get(key, 0) + 1

    manifest = {"variant": variant, "limit": limit, "image_root": image_root,
                "dataset_dir": dataset_dir, "skills": sources, "clips": clips,
                "system_in_user": next(iter(layouts.values()), False),
                "counts": dict(sorted(counts.items()))}
    for split, lines in splits.items():
        if not lines:
            continue
        text = "\n".join(lines) + "\n"
        (out_dir / f"{split}.jsonl").write_text(text, encoding="utf-8")
        manifest[f"{split}_examples"] = len(lines)
        manifest[f"{split}_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest
