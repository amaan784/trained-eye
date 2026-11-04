"""How each skill was built, read from a skill folder that the real teach flow wrote
(against the fake Meta API in test_teach.py), plus the examples and provided keyframes."""

import json

from backend.playground import data, skills
from backend.release import faces
from backend.teach import TeachOptions, run_teach
from tests.test_teach import FakeMeta, client_for, clip, png


def make_skill(tmp_path, name="burner_left_on"):
    """data/skills/<name>/ as `te teach` leaves it, with 6 + 6 variations. Returns data/."""
    client = client_for(tmp_path, FakeMeta())
    opts = TeachOptions(skill=name, clip=clip(tmp_path / f"clip_{name}"),
                        request="tell me when the washer says done", positives=6, negatives=6)
    data_dir = tmp_path / "data"
    run_teach(client, opts, data_dir / "skills", confirm=lambda s: True, say=lambda s: None)
    return data_dir


def test_list_skills_skips_refused_and_odd_folders(tmp_path):
    data_dir = make_skill(tmp_path)
    refused = data_dir / "skills" / "sam_arrives"
    refused.mkdir()
    (refused / "spec.json").write_text(json.dumps({
        "name": "sam_arrives", "request": "tell me when Sam gets here", "refused": True,
        "refusal_reason": "no people"}))
    (data_dir / "skills" / "Bad-Name").mkdir()
    listed = skills.list_skills(data_dir)
    assert [s["name"] for s in listed] == ["burner_left_on"]
    stats = json.loads((data_dir / "skills/burner_left_on/label_stats.json").read_text())
    assert listed[0]["read_back"].startswith("I'll watch the digital display")
    assert listed[0]["variations"] == 12 and listed[0]["kept"] == stats["kept"]
    assert listed[0]["keyframes"] == len(list((data_dir / "skills/burner_left_on/keyframes")
                                              .iterdir()))
    assert skills.list_skills(tmp_path / "nothing") == []


def test_skill_detail_wall_and_keyframes(tmp_path):
    data_dir = make_skill(tmp_path)
    detail = skills.skill_detail(data_dir, "burner_left_on")
    assert detail["target"] == "washing machine display"
    assert detail["keyframes"][0]["spark_label"] == "no"  # the fake labels real frames no
    assert detail["wall_total"] == 12 and len(detail["wall"]) == 12
    kinds = {t["kind"] for t in detail["wall"]}
    assert kinds == {"positive", "hard_negative"}
    for t in detail["wall"] + detail["keyframes"]:
        assert data.media_path(data_dir, tmp_path, t["url"].removeprefix("media/"))
    for t in detail["wall"]:
        assert t["kept"] == (t["spark_label"] == t["label"])
    assert any(not t["kept"] for t in detail["wall"])  # the fake's deliberate mistake
    assert detail["roi"]["with_box"] == 0 and detail["roi"]["frame"] is None
    assert detail["roi"]["frames"] == len(detail["keyframes"])
    assert "found no box" in detail["roi"]["note"]
    for name in ("missing", "../burner_left_on", "sam_arrives"):
        assert skills.skill_detail(data_dir, name) is None


def test_wall_is_spread_over_all_variations(tmp_path, monkeypatch):
    data_dir = make_skill(tmp_path)
    monkeypatch.setattr(skills, "WALL_TILES", 5)
    detail = skills.skill_detail(data_dir, "burner_left_on")
    assert len(detail["wall"]) == 5 and detail["wall_total"] == 12
    assert detail["wall"][0]["kind"] == "positive"
    assert detail["wall"][-1]["kind"] == "hard_negative"


def test_outline_on_the_first_frame_with_a_box(tmp_path):
    data_dir = make_skill(tmp_path)
    folder = data_dir / "skills" / "burner_left_on"
    rois = json.loads((folder / "roi.json").read_text())
    rois[2]["best"] = {"box": [10, 20, 200, 180], "score": 0.9}
    rois[3]["best"] = {"box": [5, 5, 1, 1], "score": 0.9}  # empty box: ignored
    (folder / "roi.json").write_text(json.dumps(rois))
    roi = skills.skill_detail(data_dir, "burner_left_on")["roi"]
    assert roi["with_box"] == 1 and roi["note"] is None
    frame = roi["frame"]
    assert frame["box"] == [10.0, 20.0, 200.0, 180.0] and frame["score"] == 0.9
    assert (frame["width"], frame["height"]) == (rois[2]["width"], rois[2]["height"])
    keyframe = data.media_path(data_dir, tmp_path, frame["url"].removeprefix("media/"))[0]
    assert skills.sha256_hex(keyframe.read_bytes()) == rois[2]["keyframe_sha256"]


def test_a_skill_taught_before_the_parser_gets_its_box_from_the_stored_text(tmp_path):
    data_dir = make_skill(tmp_path)
    folder = data_dir / "skills" / "burner_left_on"
    rois = json.loads((folder / "roi.json").read_text())
    w, h = rois[0]["width"], rois[0]["height"]
    rois[0]["raw"] = (f"<0f>0<|box;x1=1;y1=2;x2={w - 2};y2={h - 3};w={w};h={h};c=0.7|>"
                      "<|mask;x=0;y=0;data=2,2,!!!!#Kq|>\n")
    (folder / "roi.json").write_text(json.dumps(rois))
    roi = skills.skill_detail(data_dir, "burner_left_on")["roi"]
    assert roi["with_box"] == 1 and roi["frame"]["box"] == [1.0, 2.0, w - 1.0, h - 2.0]
    assert roi["frame"]["score"] == 0.7


def test_wall_before_labelling_comes_from_the_manifest(tmp_path):
    data_dir = make_skill(tmp_path)
    (data_dir / "skills" / "burner_left_on" / "labels.jsonl").unlink()
    wall = skills.skill_detail(data_dir, "burner_left_on")["wall"]
    assert len(wall) == 12 and all(t["spark_label"] is None and not t["kept"] for t in wall)
    assert skills.examples(data_dir) == []  # nothing labelled, nothing to show


def test_examples(tmp_path):
    data_dir = make_skill(tmp_path)
    (example,) = skills.examples(data_dir)
    assert example["skill"] == "burner_left_on"
    assert example["request"] == "tell me when the washer says done"
    kinds = [t["kind"] for t in example["variations"]]
    assert kinds == ["positive"] * 6 + ["hard_negative"] * 6


def test_provided_keyframes(tmp_path):
    data_dir = make_skill(tmp_path)
    offered = skills.provided_keyframes(data_dir)
    assert len(offered) == skills.PROVIDED_PER_SKILL
    first = offered[0]
    path = data.media_path(data_dir, tmp_path, first["url"].removeprefix("media/"))[0]
    assert skills.keyframe_bytes(data_dir, first["id"]) == path.read_bytes()
    for bad in ("burner_left_on/../spec.json", "burner_left_on/missing.jpg", "", "x"):
        assert skills.keyframe_bytes(data_dir, bad) is None

    curated = data_dir / "playground" / "keyframes"
    curated.mkdir(parents=True)
    (curated / "kettle.png").write_bytes(png((10, 10, 10)))
    (curated / "notes.txt").write_text("not an image")
    assert skills.provided_keyframes(data_dir) == [
        {"id": "playground/kettle.png", "url": "media/playground/keyframes/kettle.png"}]
    assert skills.keyframe_bytes(data_dir, "playground/kettle.png") == png((10, 10, 10))
    assert skills.keyframe_bytes(data_dir, first["id"]) is None  # no longer on offer


def blur(src, dst):
    """te blur's output folder (sources.json included), with a fake detector."""
    return faces.blur_folder(src, dst, detector=lambda b: [(0, 0, 8, 8)])


def test_public_copy_shows_blurred_keyframes_and_no_generated_images(tmp_path):
    data_dir = make_skill(tmp_path)
    folder = data_dir / "skills" / "burner_left_on"
    rois = json.loads((folder / "roi.json").read_text())
    rois[1]["best"] = {"box": [10, 20, 200, 180], "score": 0.8}
    (folder / "roi.json").write_text(json.dumps(rois))
    bare = skills.skill_detail(data_dir, "burner_left_on", public=True)
    assert bare["keyframes"] == [] and bare["roi"]["frame"] is None  # not blurred yet
    blur(folder / "keyframes", folder / "keyframes_blurred")
    for f in (folder / "synthetic").glob("*.png"):
        f.unlink()  # the public copy's data has no generated image: counts stay
    detail = skills.skill_detail(data_dir, "burner_left_on", public=True)
    assert detail["wall"] == [] and detail["wall_note"] == skills.WALL_NOTE
    assert detail["wall_total"] == 12
    originals = skills.keyframe_files(folder)
    assert [k["file"] for k in detail["keyframes"]] == originals
    assert detail["keyframes"][0]["spark_label"] == "no"  # same names: labels still match
    for k in detail["keyframes"]:
        assert k["url"].startswith("media/skills/burner_left_on/keyframes_blurred/")
        assert data.media_path(data_dir, tmp_path, k["url"][6:], public=True)
    frame = detail["roi"]["frame"]  # found through sources.json: the bytes differ
    assert frame["url"] == detail["keyframes"][1]["url"] and frame["box"][0] == 10.0
    blurred = (folder / "keyframes_blurred" / originals[1]).read_bytes()
    assert skills.sha256_hex(blurred) != rois[1]["keyframe_sha256"]
    (listed,) = skills.list_skills(data_dir, public=True)
    assert listed["variations"] == 12 and listed["keyframes"] == len(originals)
    assert skills.list_skills(data_dir)[0]["variations"] == 0  # locally: only files that exist
    assert skills.examples(data_dir, public=True) == []
    assert skills.skill_detail(data_dir, "burner_left_on")["wall_note"] is None


def test_public_copy_offers_only_curated_blurred_keyframes(tmp_path):
    data_dir = make_skill(tmp_path)
    assert skills.provided_keyframes(data_dir, public=True) == []  # no fallback to originals
    first = skills.provided_keyframes(data_dir)[0]["id"]
    assert skills.keyframe_bytes(data_dir, first, public=True) is None
    curated = data_dir / "playground" / "keyframes"
    curated.mkdir(parents=True)
    (curated / "by_hand.png").write_bytes(png((10, 10, 10)))
    assert skills.provided_keyframes(data_dir, public=True) == []  # not from te blur
    picked = tmp_path / "picked"
    picked.mkdir()
    (picked / "kettle.png").write_bytes(png((40, 50, 60)))
    blur(picked, curated)
    assert skills.provided_keyframes(data_dir, public=True) == [
        {"id": "playground/kettle.png", "url": "media/playground/keyframes/kettle.png"}]
    made = (curated / "kettle.png").read_bytes()
    assert skills.keyframe_bytes(data_dir, "playground/kettle.png", public=True) == made
    assert skills.keyframe_bytes(data_dir, "playground/by_hand.png", public=True) is None
