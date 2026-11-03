"""Rebuild a skill's generated images from its committed prompt manifest.

Generated images are never published or committed: they stay in data/ with Muse Image's
watermark and provenance metadata. What ships instead is each skill's prompt manifest,
copied out of data/ by `export_manifests` into a committed folder (manifests/<skill>.jsonl),
and `regenerate`, which runs the same edits again through synth.generate. Every call goes
through MetaClient, so it is cached, checked against the spend caps and recorded in the
ledger; on the machine that made the images every call is a free cache hit.

The edits start from the teaching keyframes, found by the SHA-256 each row records, so
regeneration needs the original keyframes (data/skills/<skill>/keyframes/). Face-blurred
copies have other hashes: they work only through a mapping {original SHA-256: file name},
the sources.json that blur_folder writes next to them, and then the edits start from other
pixels, so the images differ from the manifest's.

Muse Image may not be deterministic, so the report counts how many images match the
manifest's image_sha256: a mismatch is reported, never an error.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

from backend.dataengine import synth
from backend.meta.cache import sha256_hex
from backend.meta.client import MetaClient
from backend.release.faces import IMAGE_SUFFIXES, SOURCES_FILE

PLAN_FIELDS = tuple(f.name for f in fields(synth.Variation))  # what regenerates an image
PROVENANCE_FIELDS = ("image_sha256", "model", "cache_key")  # what checks it


def read_rows(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [f for f in PLAN_FIELDS if not isinstance(row.get(f), str)]
        if missing:
            raise ValueError(f"{path} line {n}: no {', '.join(missing)}")
        rows.append(row)
    return rows


def failed(row: dict) -> bool:
    """The edit failed when the manifest was made, so there is no image to rebuild."""
    return not row.get("image_sha256")


def export_row(row: dict) -> dict:
    return {k: row.get(k) for k in PLAN_FIELDS + (("error",) if failed(row)
                                                  else PROVENANCE_FIELDS)}


def export_manifests(data_dir: Path, out_dir: Path) -> list[Path]:
    """Copy each taught skill's synthetic/manifest.jsonl to out_dir/<skill>.jsonl with only
    the fields that regenerate an image and check its provenance: no image bytes, file
    names or costs. Failed rows keep their error. Sorted by id, so reruns give the same
    file. Returns the files written."""
    written = []
    for source in sorted(Path(data_dir).glob("skills/*/synthetic/manifest.jsonl")):
        skill = source.parent.parent.name
        rows = read_rows(source)
        if others := {r["skill"] for r in rows} - {skill}:
            raise ValueError(f"{source} has rows of other skills: {', '.join(sorted(others))}")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        target = Path(out_dir) / f"{skill}.jsonl"
        target.write_text("".join(json.dumps(export_row(r), sort_keys=True) + "\n"
                                  for r in sorted(rows, key=lambda r: r["id"])), "utf-8")
        written.append(target)
    return written


def find_keyframes(keyframes_dir: Path, needed: set[str],
                   mapping: Path | None = None) -> tuple[dict[str, bytes], int]:
    """({original SHA-256: bytes to edit}, how many came through the mapping). Files match
    by their own SHA-256, or through `mapping` (default: the folder's sources.json)."""
    folder = Path(keyframes_dir)
    if not folder.is_dir():
        raise ValueError(f"no keyframes folder at {folder}")
    found = {}
    for path in sorted(folder.iterdir()):
        if (path.is_file() and not path.name.startswith(".")
                and path.suffix.lower() in IMAGE_SUFFIXES):
            data = path.read_bytes()
            found[sha256_hex(data)] = data
    mapped = 0
    if mapping is None and (folder / SOURCES_FILE).is_file():
        mapping = folder / SOURCES_FILE
    if mapping is not None:
        for original, name in json.loads(Path(mapping).read_text()).items():
            path = folder / Path(str(name)).name
            if original in needed and original not in found and path.is_file():
                found[original], mapped = path.read_bytes(), mapped + 1
    if missing := needed - found.keys():
        raise ValueError(
            f"{len(missing)} of {len(needed)} keyframes are not in {folder}: they are found "
            "by SHA-256, so regeneration needs the original keyframes "
            "(data/skills/<skill>/keyframes/). For face-blurred copies, keep the "
            f"{SOURCES_FILE} that te blur writes next to them, or pass --map")
    return {h: found[h] for h in needed}, mapped


def regenerate(client: MetaClient, manifest: Path, keyframes_dir: Path, out_dir: Path,
               progress: Callable[[bool, float], None] | None = None, *,
               mapping: Path | None = None,
               confirm: Callable[[dict], bool] | None = None) -> dict:
    """Run the manifest's edits again (cached, capped, in the ledger) into out_dir.

    Rows that failed originally are skipped. `confirm(report)` sees the plan and the
    estimate (before cache hits) before any call; if it says no, nothing is spent. The
    report counts the images made and how many match the manifest's image_sha256."""
    manifest, out_dir = Path(manifest), Path(out_dir)
    if (out_dir / "manifest.jsonl").resolve() == manifest.resolve():
        raise ValueError(f"{out_dir} holds the manifest being read: pick another folder")
    rows = read_rows(manifest)
    todo = [r for r in rows if not failed(r)]
    plan = [synth.Variation(**{f: r[f] for f in PLAN_FIELDS}) for r in todo]
    keyframes, mapped = find_keyframes(keyframes_dir, {v.keyframe_sha256 for v in plan},
                                       mapping)
    report = {"manifest": str(manifest), "planned": len(plan),
              "skipped_failed_originally": len(rows) - len(todo),
              "keyframes": len(keyframes), "mapped_keyframes": mapped,
              "estimate_usd": synth.estimate_cost(client, len(plan)),
              "model": client.settings.model("image"),
              "manifest_models": sorted({r["model"] for r in todo if r.get("model")})}
    if confirm is not None and not confirm(report):
        return {**report, "stopped": "not confirmed"}
    spent, cached = 0.0, 0

    def tick(was_cached: bool, cost_usd: float) -> None:
        nonlocal spent, cached
        spent, cached = spent + cost_usd, cached + was_cached
        if progress:
            progress(was_cached, cost_usd)

    made = synth.generate(client, plan, keyframes, out_dir, progress=tick)
    expected = {r["id"]: r["image_sha256"] for r in todo}
    differing = sorted(r["id"] for r in made if r["image_sha256"] != expected[r["id"]])
    report.update(generated=len(made), failed=len(plan) - len(made),
                  matching=len(made) - len(differing), differing=len(differing),
                  differing_ids=differing, cached=cached, spent_usd=spent,
                  out_dir=str(out_dir))
    return report
