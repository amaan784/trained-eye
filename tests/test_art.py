"""The README art (backend/art): its numbers come from real run outputs only, and nothing
is drawn while one is missing. Nothing here launches Chrome."""

import io
import json

import pytest
from PIL import Image

from backend.art import facts as F
from backend.art.render import RenderError, fill


def jpeg(color=(40, 60, 80), size=(160, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def taught(data, *, from_audio=False, smoke=False, stats=True):
    from backend.meta.cache import sha256_hex

    root = data / "skills" / "fridge_left_open"
    (root / "keyframes").mkdir(parents=True)
    (root / "synthetic").mkdir()
    (root / "dataset").mkdir()
    frames = [jpeg((i * 40, 60, 80)) for i in range(3)]
    for i, f in enumerate(frames):
        (root / "keyframes" / f"kf_{i:02d}_{sha256_hex(f)[:8]}.jpg").write_bytes(f)
    (root / "spec.json").write_text(json.dumps({"positive": "the fridge door is open"}))
    (root / "request.json").write_text(json.dumps(
        {"request": "Tell me when the fridge door is left open", "from_audio": from_audio}))
    if stats:
        (root / "label_stats.json").write_text(json.dumps({"total": 4, "kept": 3}))
    rows = [{"id": f"g{i}", "file": f"g{i}.webp", "model": "muse-image-1.0"} for i in range(4)]
    rows.append({"id": "g4", "file": None, "error": "rejected"})
    (root / "synthetic" / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    (root / "labels.jsonl").write_text("".join("{}\n" for _ in range(7)))
    index = ([{"source": "real", "answer": "yes"}] * 2 + [{"source": "synthetic", "answer": "yes"}]
             + [{"source": "synthetic", "answer": "no"}] * 2)
    (root / "dataset" / "fridge_left_open-teach.manifest.json").write_text(
        json.dumps({"examples": 5, "examples_index": index}))
    rois = [{"keyframe_sha256": sha256_hex(f), "width": 160, "height": 120,
             "best": {"box": [10, 10, 10 + 30 * (i + 1), 90]} if i else None}
            for i, f in enumerate(frames)]
    (root / "roi.json").write_text(json.dumps(rois))
    (data / "train").mkdir()
    (data / "train" / "exp1-mixed.result.json").write_text(
        json.dumps({"run": "exp1-mixed", "ok": True, "smoke": smoke}))
    return root


def test_the_facts_come_from_the_runs_and_list_only_models_called(tmp_path):
    taught(tmp_path)
    sk = F.skill_facts(tmp_path, "fridge_left_open", "exp1-mixed")
    assert (sk.made, sk.kept, sk.rejected) == (4, 3, 1)  # the failed edit made nothing
    assert (sk.examples, sk.yes, sk.no, sk.real_examples) == (5, 3, 2, 2)
    assert (sk.keyframes, sk.frames_with_box) == (3, 2)
    assert sk.box == (10, 10, 100, 90)  # the box nearest half the frame (0.375, not 0.25)
    names = [m.name for m in sk.models]
    assert names == ["Muse Wonder", "SAM 3.1", "Muse Image", "Muse Spark", "Muse Glimmer 30B"]
    assert [m.highlight for m in sk.models] == [False] * 4 + [True]
    counts = {m.job: m.count for m in sk.models}
    assert counts["labeler"] == "7 images labelled" and counts["image"] == "4 images made"


def test_voice_transcribe_is_listed_only_when_the_request_was_spoken(tmp_path):
    taught(tmp_path, from_audio=True)
    sk = F.skill_facts(tmp_path, "fridge_left_open", "exp1-mixed")
    assert sk.models[0].name == "Muse Voice Transcribe" and len(sk.models) == 6


@pytest.mark.parametrize("change", ["no stats", "smoke only", "no run"])
def test_nothing_is_drawn_while_a_fact_is_missing(tmp_path, change):
    taught(tmp_path, stats=change != "no stats", smoke=change == "smoke only")
    if change == "no run":
        (tmp_path / "train" / "exp1-mixed.result.json").unlink()
    with pytest.raises(F.MissingFact):
        F.skill_facts(tmp_path, "fridge_left_open", "exp1-mixed")


def results(*judges):
    variants = {"finetuned": (0.81, 3.95), "base": (0.42, 3.95), "spark": (0.77, 9.6)}
    return {"footage": "other kitchens (EPIC-KITCHENS-100, head-mounted camera)",
            "evaluations": [{"experiment": 4, "variant": v, "status": "scored",
                             "metrics": {"f1": variants[v][0]},
                             "usd_per_hour": variants[v][1]} for v in judges]}


def test_the_headline_waits_for_all_three_judges(tmp_path):
    path = tmp_path / "results.json"
    with pytest.raises(F.MissingFact):
        F.headline(path)  # not evaluated yet
    path.write_text(json.dumps(results("finetuned", "spark")))
    with pytest.raises(F.MissingFact, match="all three"):
        F.headline(path)
    path.write_text(json.dumps(results("finetuned", "base", "spark")))
    head = F.headline(path)
    assert head.text == "Frame F1 0.81 on other kitchens"
    assert "0.42 for base Glimmer and 0.77 for Muse Spark" in head.detail
    assert "$3.95 an hour against $9.60" in head.detail


def test_a_page_with_an_unfilled_field_is_refused(tmp_path, monkeypatch):
    from backend.art import render

    monkeypatch.setattr(render, "TEMPLATES", tmp_path)
    (tmp_path / "t.html").write_text("<p>$a</p><script>var x = `${y}` + $a;</script>")
    assert fill("t.html", {"a": "1"}) == "<p>1</p><script>var x = `${y}` + 1;</script>"
    (tmp_path / "u.html").write_text("<p>$a {{b}}</p>")
    with pytest.raises(RenderError, match="unfilled"):
        fill("u.html", {"a": "1"})
