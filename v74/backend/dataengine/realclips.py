"""Labelled session clips as real training and validation examples (Experiment 1).

The recording protocol (docs/recording-protocol.md) names a skill's clips

    data/clips/<skill>/session1_<anything>       training
    data/clips/<skill>/session2_<anything>       validation: never trained on (te calibrate)
    data/clips/<skill>/session3_test_<anything>  the test set: never read here

and te label --clip marks each one's true ranges in <clip>.labels.json, {"clip":
"a.mp4", "duration_s": 312.4, "true": [[41.2, 97.0]]} (te calibrate reads the same file).
A clip is a video or a folder of frames, as everywhere else.

Frames are sampled at 1 fps with keyframes.load_frames (frame i of a folder is at i
seconds). A frame inside a true range answers yes, outside no, and frames within 1.5 s
of a range's start or end are left out: the moment of change is unclear in the footage
and marked by hand (a range starting at 0 or ending at the clip's end has no change
there). Each kept frame is a source "real" example, written with dataset.write_dataset into
data/skills/<skill>/dataset-clips/ (<skill>-clips.jsonl and its manifest), each with
kind "clip", its clip's name, time, session and split ("train" for session 1, "val" for
session 2) in the provenance index. The prompt and its layout are the teach dataset's,
so prepare.bundle can check the two still match; it adds these examples to the real and
mixed variants, session 2 always in the val split.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from backend.dataengine.dataset import Example, write_dataset
from backend.dataengine.keyframes import load_frames
from backend.labeltool.clips import clean_ranges
from backend.skills.spec import SkillSpec

DATASET_DIR = "dataset-clips"
FPS = 1.0
GUARD_S = 1.5  # frames this close to a range's start or end are left out
SESSIONS = {"session1": "train", "session2": "val"}
UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


@dataclass
class Clip:
    path: Path
    session: str  # "session1" or "session2"

    @property
    def split(self) -> str:
        return SESSIONS[self.session]

    @property
    def labels(self) -> Path:
        return labels_path(self.path)


def labels_path(clip: Path) -> Path:
    """Where te label --clip saves a clip's labels: <clip>.labels.json beside it."""
    return Path(clip).with_suffix(".labels.json")


def find_clips(folder: Path) -> list[Clip]:
    """The session 1 and session 2 clips in `folder` (videos or frame folders), sorted;
    labels files are not clips."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    clips = []
    for path in sorted(folder.iterdir()):
        session = path.name.split("_", 1)[0]
        if (session in SESSIONS and path.name.startswith(f"{session}_")
                and path.suffix.lower() != ".json"):
            clips.append(Clip(path, session))
    return clips


def frame_answers(times: list[float], ranges: list[list[float]], duration_s: float,
                  guard_s: float = GUARD_S) -> list[str | None]:
    """For each frame time: "yes" or "no", or None within `guard_s` of a change, which is
    a range's start (unless 0) or end (unless the clip's end). Ranges are half-open: a
    frame at a range's end is outside it."""
    changes = [s for s, _ in ranges if s > 0] + [e for _, e in ranges if e < duration_s]
    answers = []
    for t in times:
        if any(abs(t - c) <= guard_s for c in changes):
            answers.append(None)
        else:
            answers.append("yes" if any(s <= t < e for s, e in ranges) else "no")
    return answers


def read_labels(clip: Clip, end_s: float) -> tuple[list[list[float]], float]:
    """(true ranges, duration) from the clip's labels file. The duration is the file's
    duration_s, or else `end_s` (just past the last sampled frame). Raises ValueError or
    TypeError when the file cannot be used."""
    labels = json.loads(clip.labels.read_text(encoding="utf-8"))
    if not isinstance(labels, dict):
        raise TypeError("the labels file must be an object with true ranges")
    if labels.get("clip") not in (None, clip.path.name):
        raise ValueError(f"its labels are for {labels['clip']}")
    duration = labels.get("duration_s")
    duration = end_s if duration is None else duration
    return clean_ranges(labels.get("true"), duration), float(duration)


def clip_examples(clip: Clip, spec: SkillSpec, fps: float = FPS,
                  guard_s: float = GUARD_S) -> tuple[list[Example], dict]:
    """The clip's frames as examples, and a summary of what was kept."""
    frames = load_frames(clip.path, fps=fps)
    if not frames:
        raise ValueError("no frames in it")
    ranges, duration = read_labels(clip, frames[-1].t_s + 1 / fps)
    answers = frame_answers([f.t_s for f in frames], ranges, duration, guard_s)
    token = UNSAFE.sub("-", clip.path.name)
    examples = [
        Example(id=f"{spec.name}-real-{token}-{frame.t_s:07.1f}s", image=frame.data,
                answer=answer, source="real", skill=spec.name, positive=spec.positive,
                negatives=list(spec.negatives),
                provenance={"kind": "clip", "clip": clip.path.name, "t_s": frame.t_s,
                            "session": clip.session, "split": clip.split})
        for frame, answer in zip(frames, answers, strict=True) if answer]
    summary = {"clip": clip.path.name, "session": clip.session, "split": clip.split,
               "labels": clip.labels.name, "frames": len(frames), "examples": len(examples),
               "yes": sum(e.answer == "yes" for e in examples),
               "no": sum(e.answer == "no" for e in examples),
               "near_a_change": answers.count(None)}
    return examples, summary


def teach_layout(skill_dir: Path, system_in_user: bool | None) -> bool:
    """Where the system prompt goes: as in the skill's teach dataset when there is one
    (`system_in_user`, TE_SYSTEM_IN_USER, must then agree), else `system_in_user`."""
    name = Path(skill_dir).name
    manifest = Path(skill_dir) / "dataset" / f"{name}-teach.manifest.json"
    if not manifest.is_file():
        return bool(system_in_user)
    taught = bool(json.loads(manifest.read_text("utf-8")).get("system_in_user", False))
    if system_in_user is not None and system_in_user != taught:
        raise ValueError(f"{name}'s dataset/ was written with TE_SYSTEM_IN_USER="
                         f"{int(taught)}, but it is {int(system_in_user)} now: run te teach "
                         "again first (cached calls are free), so both datasets match")
    return taught


def write_clip_dataset(skill_dir: Path, clips_dir: Path, *, system_in_user: bool | None = None,
                       fps: float = FPS, guard_s: float = GUARD_S) -> dict:
    """Write skill_dir/dataset-clips (replacing it) from the labelled session 1 and 2
    clips in `clips_dir`, and return its manifest. Clips without usable labels are left
    out and listed under "skipped"; with none left, ValueError."""
    skill_dir, clips_dir = Path(skill_dir), Path(clips_dir)
    name = skill_dir.name
    spec_path = skill_dir / "spec.json"
    if not spec_path.is_file():
        raise ValueError(f"{name} has no spec yet: run te teach for it first")
    spec = SkillSpec.model_validate_json(spec_path.read_text("utf-8"))
    if spec.refused:
        raise ValueError(f"{name} was refused, so it has nothing to train")
    layout = teach_layout(skill_dir, system_in_user)
    clips = find_clips(clips_dir)
    if not clips:
        raise ValueError(f"no session1_* or session2_* clips in {clips_dir} (see "
                         "docs/recording-protocol.md)")
    examples, used, skipped = [], [], []
    for clip in clips:
        if not clip.labels.is_file():
            skipped.append({"clip": clip.path.name, "reason": f"no {clip.labels.name}: "
                            f"te label --clip {clip.path} marks it"})
            continue
        try:
            found, summary = clip_examples(clip, spec, fps, guard_s)
        except (OSError, ValueError, TypeError) as err:  # JSONDecodeError is a ValueError
            skipped.append({"clip": clip.path.name, "reason": str(err)})
            continue
        except subprocess.CalledProcessError:
            skipped.append({"clip": clip.path.name, "reason": "ffmpeg cannot read it"})
            continue
        if not found:
            skipped.append({"clip": clip.path.name, "reason": "every frame is near a change"})
            continue
        examples += found
        used.append(summary)
    if not examples:
        reasons = "; ".join(f"{s['clip']}: {s['reason']}" for s in skipped)
        raise ValueError(f"no usable labelled clips for {name}: {reasons}")
    clash = [i for i, n in Counter(e.id for e in examples).items() if n > 1]
    if clash:
        raise ValueError(f"two clips give the same example id {min(clash)}: rename one")
    out = skill_dir / DATASET_DIR
    shutil.rmtree(out, ignore_errors=True)  # only this function writes here
    manifest = write_dataset(examples, out, f"{name}-clips", system_in_user=layout)
    manifest.update(fps=fps, guard_s=guard_s, clips=used, skipped=skipped,
                    splits={split: sum(c["examples"] for c in used if c["split"] == split)
                            for split in ("train", "val")})
    (out / f"{name}-clips.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest
