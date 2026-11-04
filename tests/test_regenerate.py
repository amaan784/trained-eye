"""Exporting prompt manifests and regenerating images from them, against the fake Meta API
of test_teach (nothing is sent or billed)."""

import json

import pytest

from backend.dataengine import synth
from backend.meta.cache import sha256_hex
from backend.regenerate import PLAN_FIELDS, export_manifests, regenerate
from backend.release.faces import SOURCES_FILE, blur_folder
from backend.skills.spec import SkillSpec, Target
from tests.test_teach import FakeMeta, client_for, png

SPEC = SkillSpec(name="burner_left_on", request="tell me when the washer is done",
                 target=Target(sam_prompt="washer display", description="the display"),
                 positive="the display says End", negatives=["the display shows time left"],
                 alert="Laundry's done.")


def taught(tmp_path):
    """A skill as te teach leaves it: keyframes, and synthetic/ with one rejected edit."""
    keyframes = [png((40 * i, 90, 160), serial=i) for i in range(3)]
    plan = synth.plan_variations(SPEC, [sha256_hex(k) for k in keyframes], 3, 3)
    fake = FakeMeta(reject_edit=lambda body: plan[0].prompt.encode() in body)
    client = client_for(tmp_path, fake)
    skill = client.settings.data_dir / "skills" / "burner_left_on"
    (skill / "keyframes").mkdir(parents=True)
    for i, data in enumerate(keyframes):
        (skill / "keyframes" / f"kf_{i:02d}.png").write_bytes(data)
    synth.generate(client, plan, {sha256_hex(k): k for k in keyframes}, skill / "synthetic")
    manifest = export_manifests(client.settings.data_dir, tmp_path / "manifests")[0]
    return client, fake, plan, skill, manifest


def test_export_keeps_what_regenerates_and_checks_provenance(tmp_path):
    client, _, plan, skill, manifest = taught(tmp_path)
    (client.settings.data_dir / "skills" / "tap_left_running").mkdir()  # nothing generated yet
    assert export_manifests(client.settings.data_dir, tmp_path / "manifests") == [manifest]
    assert manifest == tmp_path / "manifests" / "burner_left_on.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert [r["id"] for r in rows] == sorted(v.id for v in plan)
    ok = [r for r in rows if r.get("image_sha256")]
    bad = [r for r in rows if not r.get("image_sha256")]
    assert len(ok) == 5 and len(bad) == 1 and "content_policy" in bad[0]["error"]
    assert all(set(r) == set(PLAN_FIELDS) | {"image_sha256", "model", "cache_key"} for r in ok)
    assert set(bad[0]) == set(PLAN_FIELDS) | {"error"}
    images = {sha256_hex(p.read_bytes()) for p in (skill / "synthetic").glob("*.png")}
    assert {r["image_sha256"] for r in ok} == images


def test_regenerating_where_the_images_were_made_is_free_and_matches(tmp_path):
    client, fake, _, skill, manifest = taught(tmp_path)
    edits = fake.edits
    report = regenerate(client, manifest, skill / "keyframes", tmp_path / "again")
    assert fake.edits == edits  # every edit came from the cache
    assert report["planned"] == 5 and report["skipped_failed_originally"] == 1
    assert report["matching"] == 5 and report["differing"] == 0 and report["cached"] == 5
    assert report["spent_usd"] == 0 and report["estimate_usd"] == synth.estimate_cost(client, 5)
    assert report["manifest_models"] == [report["model"]]


def test_a_fresh_regeneration_reports_differences_without_failing(tmp_path):
    _, _, _, skill, manifest = taught(tmp_path)
    other = FakeMeta()
    other.edits = 100  # a model that is not deterministic: other bytes for the same prompts
    fresh = client_for(tmp_path / "elsewhere", other)
    costs = []
    report = regenerate(fresh, manifest, skill / "keyframes", tmp_path / "again",
                        progress=lambda cached, usd: costs.append(usd))
    assert other.edits == 105 and report["generated"] == 5 and report["failed"] == 0
    assert report["matching"] == 0 and report["differing"] == 5
    assert len(report["differing_ids"]) == 5 and report["cached"] == 0
    assert report["spent_usd"] == pytest.approx(sum(costs)) and sum(costs) > 0
    assert len((tmp_path / "again" / "manifest.jsonl").read_text().splitlines()) == 5


def test_face_blurred_keyframes_need_the_mapping(tmp_path):
    client, _, _, skill, manifest = taught(tmp_path)
    blurred = tmp_path / "published"
    blur_folder(skill / "keyframes", blurred, detector=lambda data: [(0, 0, 8, 8)])
    mapping = (blurred / SOURCES_FILE).rename(tmp_path / "map.json")
    with pytest.raises(ValueError, match="needs the original keyframes"):
        regenerate(client, manifest, blurred, tmp_path / "again")
    report = regenerate(client, manifest, blurred, tmp_path / "again", mapping=mapping)
    assert report["mapped_keyframes"] == report["keyframes"] > 0 and report["generated"] == 5
    mapping.rename(blurred / SOURCES_FILE)  # next to the keyframes, it is found on its own
    report = regenerate(client, manifest, blurred, tmp_path / "again2")
    assert report["mapped_keyframes"] == report["keyframes"]


def test_nothing_is_spent_without_a_yes_and_the_input_is_never_overwritten(tmp_path):
    client, _, _, skill, manifest = taught(tmp_path)
    other = FakeMeta()
    fresh = client_for(tmp_path / "elsewhere", other)
    seen = []
    report = regenerate(fresh, manifest, skill / "keyframes", tmp_path / "again",
                        confirm=lambda plan: seen.append(plan) or False)
    assert report["stopped"] == "not confirmed" and other.edits == 0
    assert seen[0]["estimate_usd"] == synth.estimate_cost(fresh, 5)
    assert not (tmp_path / "again").exists()
    with pytest.raises(ValueError, match="pick another folder"):
        regenerate(client, skill / "synthetic" / "manifest.jsonl", skill / "keyframes",
                   skill / "synthetic")
    broken = tmp_path / "broken.jsonl"
    broken.write_text(json.dumps({"id": "x", "skill": "burner_left_on"}) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        regenerate(client, broken, skill / "keyframes", tmp_path / "again")
