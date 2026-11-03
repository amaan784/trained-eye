"""The human-labelled test set as a Hugging Face dataset (`te release-test-set`).

Only session 3 clips go in: the held-out test footage of docs/recording-protocol.md
(session3_test_<anything>, recorded on another day or angle than the training clips), and
only videos that `te blur` wrote, which carry a blur record (<video>.blur.json) whose
output SHA-256 is the video's. Each clip needs its labels from `te label --clip` (the
author's own marks, never a model's), looked for next to the blurred video and then next
to the original in data/clips/<skill>/.

prepare_test_set writes the folder to upload (uploading nothing): <skill>/<clip>.mp4 with
<skill>/<clip>.labels.json, metadata.jsonl (one row per clip: file_name, skill, session,
duration, true ranges, SHA-256), the dataset card README.md and release.json, which lists
every clip with its blur record. The license is the author's choice: there is no default.
package.upload sends the folder to a dataset repo after check_folder: no image at all,
and a video only when release.json lists it with a matching blur record.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

from backend.eval.metrics import check_ranges
from backend.release.faces import BLUR_RECORD, file_sha256
from backend.release.model_card import LICENSE_ID
from backend.release.package import MEDIA_SUFFIXES, TEST_SET_KIND
from backend.skills.spec import check_name

SESSION3 = re.compile(r"session3_test_[A-Za-z0-9][A-Za-z0-9_.-]*")  # a file stem
VIDEO_SUFFIXES = (".mp4", ".mov")  # what upload treats as video
SUFFIX = ".mp4"  # te blur writes H.264; the test set holds .mp4 only
SLACK_S = 1.0  # labels may end this far past the video (re-encoding rounds the length)
BLUR_KEYS = ("source_sha256", "output_sha256", "fps", "frames", "faces", "frames_with_faces",
             "frames_with_manual_boxes", "audio", "metadata")
CARD = """\
---
license: {license}
pretty_name: Trained Eye test clips
task_categories: [video-classification]
tags: [trained-eye, smart-glasses, first-person-video]
---

# Trained Eye test clips

The held-out test clips of the Trained Eye experiments: first-person video from smart
glasses, recorded in a session of its own (another day, angle or lighting than the clips
the skills were taught and trained on), one folder per skill. For each clip, the labels
give the stretches of time when the skill's condition is true.

| Skill | Clips | Minutes | Minutes labelled true |
|---|---|---|---|
{rows}

## Files

- `<skill>/<clip>.mp4`: the clip, H.264, no audio.
- `<skill>/<clip>.labels.json`: `{{"clip", "duration_s", "true": [[start_s, end_s], ...]}}`.
- `metadata.jsonl`: one row per clip with `file_name`, `skill`, `session`, `duration_s`,
  `true_ranges` and the video's `sha256`.
- `release.json`: every file with its SHA-256, and each clip's blur record.

## Labels

Human labels only: the author marked every true stretch with `te label`. No model labelled
anything in this dataset.

## Privacy

- Faces were blurred with `te blur` before upload: OpenCV's face detector on every frame,
  plus any boxes the author added by hand where it missed one. Each clip's blur record in
  `release.json` gives the counts.
- The audio track and the container metadata (location, camera, time) were removed.
- Only session 3 test footage is included, published with the author's approval.

## Uses

Intended: scoring yes/no watch skills on first-person video, as `te eval` does with the
frame logs of `te watch`. Out of scope: identifying, recognising or tracking people, and
training any model to do so.

## License

`{license}`, chosen by the author.
"""


def video_duration_s(path: Path) -> float | None:
    """How long a video is, from ffprobe; None without ffprobe or for an unreadable file."""
    if shutil.which("ffprobe") is None:
        return None
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "csv=p=0", str(path)], capture_output=True, text=True, check=False)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def _json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def blur_record(video: Path) -> dict | None:
    """The video's blur record if it names this very file (its output_sha256)."""
    record = _json(Path(f"{video}{BLUR_RECORD}"))
    if isinstance(record, dict) and record.get("output_sha256") == file_sha256(video):
        return record
    return None


def find_clips(folder: Path) -> tuple[list[tuple[Path, dict]], list[tuple[str, str]]]:
    """([(video, its blur record)] of the blurred session 3 clips in a te blur output
    folder, [(file, why it was left out)])."""
    clips, skipped = [], []
    for path in sorted(Path(folder).iterdir()):
        if (not path.is_file() or path.name.startswith(".")
                or path.suffix.lower() not in VIDEO_SUFFIXES):
            continue
        record = blur_record(path)
        source = Path(str(record.get("source") or path.name)) if record else path
        if path.suffix.lower() != SUFFIX:
            skipped.append((path.name, f"not {SUFFIX} (te blur --dst <name>{SUFFIX})"))
        elif record is None:
            skipped.append((path.name, "no blur record from te blur that matches it"))
        elif not SESSION3.fullmatch(source.stem):
            skipped.append((path.name, f"not a session 3 test clip (from {source.name})"))
        else:
            clips.append((path, record))
    return clips, skipped


def _labels(video: Path, source: str, skill: str, labels_root: Path | None) -> Path:
    stem = Path(source).stem
    places = [video.parent / f"{stem}.labels.json", video.with_suffix(".labels.json")]
    if labels_root is not None:
        places.append(Path(labels_root) / skill / f"{stem}.labels.json")
    for place in places:
        if place.is_file():
            return place
    raise ValueError(f"{video}: no labels (looked for {', '.join(str(p) for p in places)}): "
                     "te label --clip <the original clip> marks them")


def clip_entry(skill: str, video: Path, record: dict, labels_root: Path | None,
               duration: Callable[[Path], float | None]) -> dict:
    """One clip's labels, duration and record, checked against each other."""
    source = str(record.get("source") or video.name)
    labels_path = _labels(video, source, skill, labels_root)
    labels = _json(labels_path)
    if not isinstance(labels, dict) or labels.get("clip") not in (source, video.name):
        raise ValueError(f"{labels_path} is not te label's file for {source}")
    try:
        ranges = [list(r) for r in check_ranges(labels["true"])]
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(f"{labels_path}: bad true ranges ({err})") from None
    seconds = duration(video)
    if seconds is None:
        raise ValueError(f"could not read how long {video} is (needs ffprobe: brew install "
                         "ffmpeg)")
    labelled = labels.get("duration_s")
    if isinstance(labelled, int | float) and abs(labelled - seconds) > SLACK_S:
        raise ValueError(f"{labels_path} is for a {labelled:g} s clip but {video.name} is "
                         f"{seconds:g} s long: are they this clip's?")
    if ranges and ranges[-1][1] > seconds + SLACK_S:
        raise ValueError(f"{labels_path} runs to {ranges[-1][1]:g} s but {video.name} is "
                         f"{seconds:g} s long: are they this clip's?")
    original = Path(labels_root) / skill / source if labels_root is not None else None
    if original is not None and original.is_file() and (
            file_sha256(original) != record.get("source_sha256")):
        raise ValueError(f"{video.name}'s blur record is not of {original}: blur it again")
    name = f"{skill}/{Path(source).stem}{SUFFIX}"
    return {"path": name, "skill": skill, "session": 3, "duration_s": round(seconds, 3),
            "true_ranges": ranges, "sha256": record["output_sha256"], "video": video,
            "blur": {**{k: record.get(k) for k in BLUR_KEYS},
                     "manual_boxes": len(record.get("manual_boxes") or [])}}


def card_rows(entries: list[dict]) -> str:
    rows = []
    for skill in sorted({e["skill"] for e in entries}):
        mine = [e for e in entries if e["skill"] == skill]
        minutes = sum(e["duration_s"] for e in mine) / 60
        true = sum(b - a for e in mine for a, b in e["true_ranges"]) / 60
        rows.append(f"| {skill} | {len(mine)} | {minutes:.1f} | {true:.1f} |")
    return "\n".join(rows)


def prepare_test_set(out_dir: Path, clips: Mapping[str, Path], *, license_id: str,
                     labels_root: Path | None = None,
                     duration: Callable[[Path], float | None] | None = None) -> dict:
    """Assemble the test-set folder from {skill: te blur output folder} (uploading nothing).
    `duration` measures a video (default video_duration_s). Returns release.json's contents
    plus "skipped": [(file, why)]."""
    duration = duration or video_duration_s
    if not LICENSE_ID.fullmatch(license_id or ""):
        raise ValueError("the test set needs the author's license, a Hugging Face license id "
                         "(for example cc-by-4.0): there is no default")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not (out_dir / "release.json").is_file():
        raise ValueError(f"{out_dir} is not empty and not a release folder: pick another")
    entries, skipped = [], []
    for skill, folder in sorted(clips.items()):
        check_name(skill)
        if not Path(folder).is_dir():
            raise ValueError(f"no folder at {folder} (give the te blur output folder)")
        found, left = find_clips(folder)
        skipped += [(f"{skill}/{name}", why) for name, why in left]
        entries += [clip_entry(skill, video, record, labels_root, duration)
                    for video, record in found]
    if not entries:
        raise ValueError("no blurred session 3 test clip with a blur record was found"
                         + "".join(f"\n  {name}: {why}" for name, why in skipped))
    names = [e["path"] for e in entries]
    if twice := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"two clips would both be {', '.join(twice)}")
    shutil.rmtree(out_dir, ignore_errors=True)
    rows = []
    for e in entries:
        target = out_dir / e["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(e.pop("video"), target)
        labels = {"clip": target.name, "duration_s": e["duration_s"], "true": e["true_ranges"]}
        target.with_suffix(".labels.json").write_text(json.dumps(labels, indent=1) + "\n")
        rows.append({"file_name": e["path"], "skill": e["skill"], "session": e["session"],
                     "duration_s": e["duration_s"], "true_ranges": e["true_ranges"],
                     "sha256": e["sha256"], "labels": target.with_suffix(".labels.json")
                     .relative_to(out_dir).as_posix()})
    (out_dir / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out_dir / "README.md").write_text(CARD.format(license=license_id, rows=card_rows(entries)),
                                       encoding="utf-8")
    release = {"kind": TEST_SET_KIND, "license": license_id,
               "clips": [{k: e[k] for k in ("path", "skill", "session", "sha256", "blur")}
                         for e in entries],
               "files": [{"path": p.relative_to(out_dir).as_posix(), "bytes": p.stat().st_size,
                          "sha256": file_sha256(p)}
                         for p in sorted(out_dir.rglob("*")) if p.is_file()]}
    (out_dir / "release.json").write_text(json.dumps(release, indent=1) + "\n")
    return {**release, "skipped": skipped}


def check_folder(folder: Path, release: dict) -> None:
    """Raise ValueError unless the folder may go to a dataset repo: the card with the
    author's license, no image anywhere, and every video listed in release.json as a
    session 3 clip whose SHA-256 is its blur record's output."""
    folder = Path(folder)
    stop = ": nothing was uploaded"
    if release.get("kind") != TEST_SET_KIND:
        raise ValueError(f"{folder} is not a test-set folder from te release-test-set{stop}")
    license_id = release.get("license")
    card = (folder / "README.md").read_text(encoding="utf-8") if (
        folder / "README.md").is_file() else ""
    if not LICENSE_ID.fullmatch(str(license_id or "")) or (
            f"\nlicense: {license_id}\n" not in card.split("\n---\n")[0] + "\n"):
        raise ValueError(f"the dataset card does not carry the author's license{stop}")
    listed = {c["path"]: c for c in release.get("clips", [])}
    for path in sorted(folder.rglob("*")):
        rel = path.relative_to(folder).as_posix()
        suffix = path.suffix.lower()
        if suffix in MEDIA_SUFFIXES and suffix not in VIDEO_SUFFIXES:
            raise ValueError(f"an image in the test set ({rel}): only blurred clips are "
                             f"published{stop}")
        if suffix in VIDEO_SUFFIXES:
            clip = listed.get(rel)
            sha = file_sha256(path) if path.is_file() else None
            if (clip is None or clip.get("session") != 3 or sha != clip.get("sha256")
                    or sha != (clip.get("blur") or {}).get("output_sha256")):
                raise ValueError(f"{rel} is not a listed, blurred session 3 clip: unblurred "
                                 f"footage is never published{stop}")
    if missing := sorted(p for p in listed if not (folder / p).is_file()):
        raise ValueError(f"release.json lists clips that are not there ({', '.join(missing)})"
                         f"{stop}")
    rows = [json.loads(line) for line in (folder / "metadata.jsonl").read_text().splitlines()
            if line.strip()] if (folder / "metadata.jsonl").is_file() else []
    if sorted(r.get("file_name") for r in rows) != sorted(listed):
        raise ValueError(f"metadata.jsonl does not list exactly the clips of release.json"
                         f"{stop}")
