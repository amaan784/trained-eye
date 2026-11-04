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
    (root / "keyframes.json").write_text(json.dumps({"keyframes": [
        {"name": f"kf_{i:02d}_{sha256_hex(f)[:8]}.jpg", "t_s": [6.5, 5.0, 7.25][i]}
        for i, f in enumerate(frames)]}))
    (data / "train" / "exp1-mixed").mkdir(parents=True)
    (data / "train" / "exp1-mixed.result.json").write_text(
        json.dumps({"run": "exp1-mixed", "ok": True, "smoke": smoke}))
    (data / "train" / "exp1-mixed" / "manifest.json").write_text(
        json.dumps({"train_examples": 4, "val_examples": 1}))
    (data / "modal_ledger.jsonl").write_text(
        json.dumps({"kind": "train", "name": "exp1-mixed", "gpu": "A100"}) + "\n"
        + json.dumps({"kind": "train", "name": "exp1-mixed", "gpu": "H100"}) + "\n"
        + json.dumps({"kind": "train", "name": "other", "gpu": "L4"}) + "\n")
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
    assert counts["labeler"] == "7 images labelled, 3 variations kept"
    assert counts["image"] == "4 variations made"
    assert counts["student_base"] == "4 training examples, one epoch"  # the bundle's, not 5
    assert (sk.train_examples, sk.val_examples, sk.gpu) == (4, 1, "H100")  # the run's last job


def test_voice_transcribe_is_listed_only_when_the_request_was_spoken(tmp_path):
    taught(tmp_path, from_audio=True)
    sk = F.skill_facts(tmp_path, "fridge_left_open", "exp1-mixed")
    assert sk.models[0].name == "Muse Voice Transcribe" and len(sk.models) == 6


@pytest.mark.parametrize("change", ["no stats", "smoke only", "no run", "no bundle",
                                    "no ledger"])
def test_nothing_is_drawn_while_a_fact_is_missing(tmp_path, change):
    taught(tmp_path, stats=change != "no stats", smoke=change == "smoke only")
    if change == "no run":
        (tmp_path / "train" / "exp1-mixed.result.json").unlink()
    if change == "no bundle":
        (tmp_path / "train" / "exp1-mixed" / "manifest.json").unlink()
    if change == "no ledger":
        (tmp_path / "modal_ledger.jsonl").unlink()
    with pytest.raises(F.MissingFact):
        F.skill_facts(tmp_path, "fridge_left_open", "exp1-mixed")


def results(*judges):
    values = {"finetuned": (0.88, 0.98, 3.95), "base": (0.89, 2.07, 3.95),
              "spark": (0.86, 11.16, 16.32)}
    return {"footage": "other kitchens (EPIC-KITCHENS-100, head-mounted camera)",
            "experiment4": [{"variant": v, "usd_per_hour": values[v][2],
                             "metrics": {"f1": values[v][0], "latency_median_s": values[v][1]}}
                            for v in judges]}


def test_the_headline_waits_for_all_three_judges(tmp_path):
    path = tmp_path / "results.json"
    with pytest.raises(F.MissingFact):
        F.headline(path)  # not evaluated yet
    path.write_text(json.dumps(results("finetuned", "spark")))
    with pytest.raises(F.MissingFact, match="all three"):
        F.headline(path)
    path.write_text(json.dumps(results("finetuned", "base", "spark")))
    head = F.headline(path)
    assert head.text == "Fast enough to watch live: 1.0 s a frame."
    assert head.detail == ("Base Glimmer takes 2.1 s and Muse Spark 11.2 s, at about the same "
                           "accuracy on other kitchens (frame F1 0.88, against 0.89 and 0.86).")


def test_the_headline_says_accuracy_only_when_the_fine_tuned_model_is_clearly_ahead(tmp_path):
    path = tmp_path / "results.json"
    data = results("finetuned", "base", "spark")
    data["experiment4"][0]["metrics"]["f1"] = 0.97  # 0.08 over the best other: clearly ahead
    path.write_text(json.dumps(data))
    assert F.headline(path).text == "Spots it more often: frame F1 0.97."
    data["experiment4"][0]["metrics"]["f1"] = 0.70  # behind: still speed, said plainly
    path.write_text(json.dumps(data))
    head = F.headline(path)
    assert head.text.startswith("Fast enough") and "though less accurate" in head.detail


def test_teaching_frames_carry_their_time_and_sams_box_in_clip_order(tmp_path):
    taught(tmp_path)
    frames = F.teaching_frames(tmp_path, "fridge_left_open")
    assert [f.t_s for f in frames] == [5.0, 6.5, 7.25]
    assert [f.box for f in frames] == [(10, 10, 70, 90), None, (10, 10, 100, 90)]
    (tmp_path / "skills" / "fridge_left_open" / "keyframes.json").unlink()
    with pytest.raises(F.MissingFact):  # made before te teach wrote times: nothing to print
        F.teaching_frames(tmp_path, "fridge_left_open")


def test_the_models_table_goes_between_the_readme_markers_only(tmp_path):
    from backend.art import build

    taught(tmp_path)
    sk = F.skill_facts(tmp_path, "fridge_left_open", "exp1-mixed")
    readme = tmp_path / "README.md"
    readme.write_text(f"# Top\n\n{build.MODELS_START}\nold\n{build.MODELS_END}\n\nEnd\n")
    build.write_models(readme, sk)
    text = readme.read_text()
    assert text.startswith("# Top\n") and text.endswith("\n\nEnd\n") and "old" not in text
    assert "| SAM 3.1 (`sam-3.1`) | Finds the target in the teaching frames. | 2 of 3 " in text
    assert ("| **Muse Glimmer 30B** (`meta-models/Muse-Glimmer-30B`), fine-tuned |" in text)
    build.write_models(readme, sk)
    assert readme.read_text() == text  # rewriting changes nothing
    readme.write_text("# no markers\n")
    with pytest.raises(F.MissingFact, match="markers"):
        build.write_models(readme, sk)


def test_the_banner_frames_lead_up_to_the_alert_and_print_time_and_score():
    from backend.art import build, film

    rp = F.Replay("c", None, [(float(t), t / 100) for t in range(61)], 35.0, "Shut it.", 30)
    shots = build.banner_times(rp)
    assert [t for t, _ in shots] == [7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0]
    assert shots[4] == (35.0, 0.35)
    rp.points = rp.points[:40]  # the clip ends at 39 s: nothing after it is invented
    assert [t for t, _ in build.banner_times(rp)][-2:] == [28.0, 35.0]
    assert film.edge_time(16.4) == "0:16" and film.edge_time(65.67, tenths=True) == "1:05.7"


def test_a_page_with_an_unfilled_field_is_refused(tmp_path, monkeypatch):
    from backend.art import render

    monkeypatch.setattr(render, "TEMPLATES", tmp_path)
    (tmp_path / "t.html").write_text("<p>$a</p><script>var x = `${y}` + $a;</script>")
    assert fill("t.html", {"a": "1"}) == "<p>1</p><script>var x = `${y}` + 1;</script>"
    (tmp_path / "u.html").write_text("<p>$a {{b}}</p>")
    with pytest.raises(RenderError, match="unfilled"):
        fill("u.html", {"a": "1"})
