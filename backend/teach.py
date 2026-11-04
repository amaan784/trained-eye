"""The teach flow for one skill, end to end.

  voice (Muse Voice Transcribe) -> skill spec (Muse Wonder) -> read-back and confirm
  -> region of interest (SAM 3.1) -> synthetic variations (Muse Image)
  -> labels (Muse Spark) -> Axolotl dataset

The dataset's real examples are the keyframes and, for each keyframe SAM found a box in,
the keyframe cropped to that box (roi.crop_box), labelled like the keyframe and recorded
with the keyframe's hash and the box in the dataset's provenance index.

Every step writes its output under data/skills/<skill>/ and every Meta call is cached,
so a rerun after a crash or a change further down costs nothing for the steps that
already ran.

If training on Meta model outputs is not allowed (settings.train_on_model_outputs is
False, see SPEC.md "Fallback"), generated images and Spark labels are still produced
for evaluation, but the dataset holds only real frames with the user's own labels (and
no crops, as SAM chose their pixels).
"""

from __future__ import annotations

import functools
import json
import re
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from backend.dataengine import dataset, labels, parallel, roi, synth
from backend.dataengine.keyframes import load_frames, select_keyframes
from backend.meta.cache import sha256_hex
from backend.meta.client import EXTENSIONS, MetaClient, sniff_mime, to_wav, wav_duration_s
from backend.meta.ledger import (
    IMAGE_TOKENS,  # noqa: F401 - the guess, still importable from here
    image_tokens,
    token_cost,
    unit_cost,
)
from backend.skills.compiler import compile_skill
from backend.skills.spec import SkillSpec, check_name, read_back

LABEL_WORDS = {"yes": "yes", "y": "yes", "true": "yes", "no": "no", "n": "no", "false": "no"}
KEYFRAME_FILE = re.compile(r"kf_\d{2}(?:_[0-9a-f]{8})?\.(?:jpg|png|webp)")  # written here
MIN_KEYFRAMES = 20  # SPEC asks for 20 to 30; select_keyframes keeps at most 24
MIN_CLASS_SHARE = 0.1  # below this, warn that the dataset is unbalanced
_print = functools.partial(print, flush=True)  # progress reaches a pipe or a log as it happens


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
    correction: str | None = None  # the user's fix for a wrong read-back
    workers: int = 1  # image edits and labels at a time (te teach: config.PARALLEL_CALLS)


def estimate(client: MetaClient, *, audio_s: float, keyframes: int, positives: int,
             negatives: int) -> dict[str, float]:
    """Cost parts and their total, before cache hits. "at_most" is the total if every
    label is cut off and asked again with the larger budget. Images are priced at the
    input tokens per image measured from the ledger (ledger.image_tokens; the guess,
    IMAGE_TOKENS, before there are such calls)."""
    s = client.settings
    images = positives + negatives
    label_price = s.price(s.model("labeler"))
    per_image = image_tokens(s, client.ledger).per_image
    parts = {
        "transcribe": unit_cost(s.price(s.model("transcribe")), audio_s=audio_s),
        "compile": token_cost(s.price(s.model("compiler")), 3 * per_image + 900, 4096),
        "sam": unit_cost(s.price(s.model("sam")), images=keyframes),
        "images": unit_cost(s.price(s.model("image")), images=images),
        "labels": (keyframes + images)
        * token_cost(label_price, per_image + 250, labels.MAX_TOKENS[0]),
    }
    parts["total"] = sum(parts.values())
    retries = (keyframes + images) * sum(token_cost(label_price, per_image + 250, m)
                                         for m in labels.MAX_TOKENS[1:])
    parts["at_most"] = parts["total"] + retries
    return parts


def keyframe_name(i: int, data: bytes) -> str:
    """kf_<position>_<content hash>: a --labels entry only matches the image it was made for."""
    return f"kf_{i:02d}_{sha256_hex(data)[:8]}.{EXTENSIONS.get(sniff_mime(data), 'jpg')}"


def save_keyframes(kf_dir: Path, frames: list) -> dict[str, bytes]:
    """Write the keyframes as kf_<position>_<hash> files, removing those of an earlier take
    (only files named that way). Returns {name: bytes}."""
    kf_dir.mkdir(parents=True, exist_ok=True)
    keyframes = {keyframe_name(i, frame.data): frame.data for i, frame in enumerate(frames)}
    for old in kf_dir.iterdir():
        if old.is_file() and KEYFRAME_FILE.fullmatch(old.name) and old.name not in keyframes:
            old.unlink()  # from an earlier take; keyframes are rebuilt from the clip each run
    for name, data in keyframes.items():
        (kf_dir / name).write_bytes(data)
    return keyframes


def normalise_labels(raw, names: list[str]) -> dict[str, str]:
    """The user's own labels as {keyframe name: yes|no}. Keys may be bare names or paths
    ending in one; values yes/no in any case (y, true, n, false also work). Raises
    TypeError if `raw` is not an object, ValueError naming every entry that cannot be used."""
    if not isinstance(raw, dict):
        raise TypeError('--labels must be a JSON object like {"kf_00_1a2b3c4d.jpg": "yes"}')
    out, bad = {}, []
    for key, value in raw.items():
        name, label = Path(str(key)).name, LABEL_WORDS.get(str(value).strip().lower())
        if name in names and label and out.get(name, label) == label:
            out[name] = label
        elif label and name in out:
            bad.append(f"{key!r}: {value!r} (says the opposite of an earlier entry)")
        else:
            bad.append(f"{key!r}: {value!r}")
    if bad:
        valid = f"{names[0]} to {names[-1]}" if names else "none (the clip gave no keyframes)"
        raise ValueError(f"--labels entries that cannot be used: {', '.join(bad)}. Keys must "
                         f"be keyframe names of this clip ({valid}), values yes or no")
    return out


def check_clip(clip: Path, skill_root: Path) -> None:
    """The clip must not be this skill's own output folder: keyframes there are rewritten."""
    clip, root = Path(clip).resolve(), Path(skill_root).resolve()
    if clip == root or root in clip.parents:
        raise ValueError(f"--clip {clip} is inside this skill's own folder {skill_root}: pass "
                         "the original video or frame folder")


def labels_problem(train_on_model_outputs: bool, human: dict[str, str],
                   kf_dir: Path) -> str | None:
    """Why the user's labels cannot make a dataset when training on model outputs is off
    (the dataset is then only the labelled keyframes), or None."""
    if train_on_model_outputs:
        return None
    if not human:
        return ("training on model outputs is off, so the dataset needs your own labels: "
                f"label the frames in {kf_dir} yes or no (both are needed) and pass --labels")
    if len(set(human.values())) < 2:
        only = next(iter(human.values()))
        return ("training on model outputs is off, so the dataset is only your labelled "
                f"keyframes, and all {len(human)} say {only!r}. A yes/no check needs both: "
                "record a clip in which the condition becomes true, and label frames from "
                "before and after")
    return None


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
    say: Callable[[str], None] = _print,
) -> dict:
    check_name(opts.skill)
    root = Path(out_root) / opts.skill
    check_clip(opts.clip, root)
    root.mkdir(parents=True, exist_ok=True)
    report: dict = {"skill": opts.skill}

    # 1. keyframes (free)
    frames = select_keyframes(load_frames(opts.clip))
    if not frames:
        raise ValueError(f"no frames found in {opts.clip}")
    kf_dir = root / "keyframes"
    keyframes = save_keyframes(kf_dir, frames)
    say(f"keyframes: {len(frames)} kept")
    if len(frames) < MIN_KEYFRAMES:
        say(f"warning: fewer than {MIN_KEYFRAMES} keyframes; a 10 s clip with more movement "
            "gives more")
    try:
        human = normalise_labels(opts.real_labels, list(keyframes)) if opts.real_labels else {}
    except (TypeError, ValueError) as err:
        report["stopped"] = str(err)
        return report
    if human:
        say(f"your labels: {len(human)} of {len(keyframes)} keyframes")
    problem = labels_problem(client.settings.train_on_model_outputs, human, kf_dir)
    if problem:
        report["stopped"] = problem
        return report

    # 2. request text
    if opts.audio:
        audio = to_wav(Path(opts.audio))  # Voice Transcribe takes WAV only
        duration = wav_duration_s(audio) or opts.audio_s or 10.0
        transcript = client.transcribe(audio, filename=f"{Path(opts.audio).stem}.wav",
                                       skill=opts.skill, duration_s=duration)
        request = transcript.text.strip()
    else:
        request = (opts.request or "").strip()
    if not request:
        raise ValueError("no request: pass audio or request text")
    _write_json(root / "request.json", {"request": request, "from_audio": bool(opts.audio),
                                        "correction": opts.correction})
    say(f'request: "{request}"')
    if opts.correction:
        say(f'correction: "{opts.correction}"')

    # 3. skill spec, read back, confirm
    spec = compile_skill(client, request, spread(list(keyframes.values()), 3), name=opts.skill,
                         correction=opts.correction)
    _write_json(root / "spec.json", spec.model_dump())
    # A dataset from an earlier spec must not outlive it; it is rebuilt for free if the
    # spec is the same.
    shutil.rmtree(root / "dataset", ignore_errors=True)
    sentence = read_back(spec)
    say(sentence)
    report.update(refused=spec.refused, read_back=sentence)
    if spec.refused or not confirm(sentence):
        report["stopped"] = "refused" if spec.refused else "not confirmed"
        return report

    # 4. region of interest
    say(f"region of interest: SAM on {len(keyframes)} keyframes")
    rois = roi.locate(client, spec, list(keyframes.values()),
                      progress=_progress(say, "sam", len(keyframes)))
    roi.save(rois, root / "roi.json")
    report["roi_frames_with_box"] = sum(1 for r in rois if r.best)
    report["roi_frames_without_box"] = len(rois) - report["roi_frames_with_box"]
    say(f"region of interest: {report['roi_frames_with_box']} of {len(rois)} keyframes "
        "with a box" + (f" ({report['roi_frames_without_box']} without one get no crop)"
                        if report["roi_frames_without_box"] else ""))

    # 5. synthetic variations
    by_hash = {sha256_hex(data): data for data in keyframes.values()}
    plan = synth.plan_variations(spec, list(by_hash), opts.positives, opts.negatives)
    say(f"generating {len(plan)} images with Muse Image (finished calls are cached, so "
        "stopping and rerunning costs nothing for them)")
    generated = synth.generate(client, plan, by_hash, root / "synthetic",
                               progress=_progress(say, "images", len(plan)),
                               workers=opts.workers)
    report["generation_failed"] = len(plan) - len(generated)
    failed = f", {report['generation_failed']} failed" if report["generation_failed"] else ""
    say(f"generated: {len(generated)} images{failed}")

    # 6. labels for every image, real and generated
    jobs = [({"file": f"keyframes/{name}", "source": "real", "human_label": human.get(name)},
             data) for name, data in keyframes.items()]
    jobs += [({**row, "file": f"synthetic/{row['file']}", "source": "synthetic"}, None)
             for row in generated]
    tick = _progress(say, "labels", len(jobs))
    done, label_errors = {}, 0

    def label(job):
        base, data = job
        return labels.label_image(client, spec, data or (root / base["file"]).read_bytes())

    try:
        for i, lab, err in parallel.run_all(jobs, label, opts.workers):
            if err is not None:
                raise err
            tick(lab.cached, lab.cost_usd)
            done[i] = {**jobs[i][0], "spark_label": lab.label, "reason": lab.reason,
                       "label_error": lab.error}
            label_errors += lab.error
            if lab.error and synth.too_many_failures(label_errors, len(jobs)):
                raise synth.TooManyFailures(f"{label_errors} of {len(jobs)} labels failed so "
                                            f"far; last: {lab.reason}")
    finally:  # also when the run stops, so the failures so far can be looked at
        label_rows = [done[i] for i in sorted(done)]  # in job order, however they finished
        _write_jsonl(root / "labels.jsonl", label_rows)
    kept, stats = labels.filter_generated([r for r in label_rows if r["source"] == "synthetic"])
    _write_json(root / "label_stats.json", stats)
    errors = f", {label_errors} could not be labelled" if label_errors else ""
    say(f"labels: kept {stats['kept']} of {stats['total']} generated "
        f"(agreement {stats['agreement']}){errors}")

    # 7. dataset: needs both answers to teach a yes/no check
    examples = _examples(spec, root, label_rows, kept, client.settings.train_on_model_outputs)
    answers = Counter(e.answer for e in examples)
    report["label_stats"] = stats
    if not examples:
        report["stopped"] = "no usable examples"
        return report
    if len(answers) < 2:
        report["stopped"] = (
            f"the dataset would hold only {answers.most_common(1)[0][0]!r} examples "
            f"({len(examples)}), so it cannot teach a yes/no check. Spark's verdicts per kind "
            f"are in {root / 'label_stats.json'}")
        return report
    manifest = dataset.write_dataset(examples, root / "dataset", f"{opts.skill}-teach",
                                     system_in_user=client.settings.system_in_user)
    report["dataset"] = manifest
    report["crop_examples"] = sum(e.provenance.get("kind") == "crop" for e in examples)
    say(f"dataset: {manifest['examples']} examples ({answers['yes']} yes, {answers['no']} no; "
        f"{report['crop_examples']} of them keyframes cropped to SAM's box) -> {root / 'dataset'}")
    if min(answers.values()) < MIN_CLASS_SHARE * len(examples):
        report["warning"] = "unbalanced dataset"
        say(f"warning: unbalanced dataset, under {MIN_CLASS_SHARE:.0%} of the examples say "
            f"{min(answers, key=answers.get)!r}; see {root / 'label_stats.json'}")
    return report


def _examples(spec: SkillSpec, root: Path, rows: list[dict], kept: list[dict],
              allow_model_outputs: bool) -> list[dataset.Example]:
    """The dataset: the real keyframes with an answer, each followed by its crop to SAM's
    box when roi.json has one (not in the fallback), then the generated images kept."""
    out = []
    boxes = _boxes(root) if allow_model_outputs else {}
    for row in rows:
        if row["source"] != "real":
            continue
        human = row.get("human_label")
        if human in ("yes", "no"):
            answer = human
        else:
            answer = row["spark_label"] if allow_model_outputs else None
        if answer in ("yes", "no"):
            out.append(_example(spec, root, row, answer))
            crop = _crop_example(out[-1], boxes.get(sha256_hex(out[-1].image)))
            out += [crop] if crop else []
    if allow_model_outputs:
        out += [_example(spec, root, row, row["label"]) for row in kept]
    return out


def _boxes(root: Path) -> dict[str, dict]:
    """{keyframe sha256: its roi.json entry} for the keyframes SAM found a box in."""
    path = root / "roi.json"
    frames = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    return {f["keyframe_sha256"]: f for f in frames if f.get("best")}


def _crop_example(keyframe: dataset.Example, frame: dict | None) -> dataset.Example | None:
    """The keyframe cropped to its best box, labelled like it; None without a box, or when
    the crop would be the whole frame (no new view)."""
    if frame is None:
        return None
    box = tuple(frame["best"]["box"])
    crop = roi.crop_box(box, (frame["width"], frame["height"]))
    if crop == (0, 0, frame["width"], frame["height"]):
        return None
    return dataset.Example(
        id=f"{keyframe.id}-crop", image=roi.crop_image(keyframe.image, crop),
        answer=keyframe.answer, source=keyframe.source, skill=keyframe.skill,
        positive=keyframe.positive, negatives=list(keyframe.negatives),
        provenance={"kind": "crop", "keyframe_sha256": frame["keyframe_sha256"],
                    "box": [round(float(v), 2) for v in box], "crop": list(crop)})


def _example(spec: SkillSpec, root: Path, row: dict, answer: str) -> dataset.Example:
    stem = Path(row["file"]).stem
    return dataset.Example(
        id=f"{spec.name}-{row['source'][:4]}-{stem}", image=(root / row["file"]).read_bytes(),
        answer=answer, source=row["source"], skill=spec.name, positive=spec.positive,
        negatives=list(spec.negatives),
    )


def _progress(say: Callable[[str], None], what: str, total: int,
              every: int = 10) -> Callable[[bool, float], None]:
    """A callback for the paid loops: a line every `every` items and at the end."""
    done = cached = 0
    spent = 0.0

    def tick(was_cached: bool, cost_usd: float) -> None:
        nonlocal done, cached, spent
        done += 1
        cached += was_cached
        spent += cost_usd
        if done % every == 0 or done == total:
            say(f"  {what} {done}/{total}  ${spent:.2f} spent  ({cached} from cache)")
    return tick


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), "utf-8")
