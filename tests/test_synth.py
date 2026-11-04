import pytest

from backend.dataengine.synth import plan_variations
from backend.skills.spec import SkillSpec, Target

SPEC = SkillSpec(
    name="burner_left_on",
    request="tell me if I leave the burner on",
    target=Target(sam_prompt="stove burner", description="the stove burner"),
    positive="the burner is lit with no pan on it",
    negatives=["the burner is off", "a pan being stirred on it", "a different appliance"],
    alert="The burner is still on.",
)


def test_plan_is_deterministic_and_balanced():
    hashes = ["a" * 64, "b" * 64, "c" * 64]
    plan = plan_variations(SPEC, hashes, positives=10, negatives=12)
    assert plan == plan_variations(SPEC, hashes, positives=10, negatives=12)
    assert sum(v.label == "yes" for v in plan) == 10
    assert sum(v.label == "no" for v in plan) == 12
    assert len({v.id for v in plan}) == 22
    assert {v.condition for v in plan if v.label == "no"} == set(SPEC.negatives)
    assert {v.keyframe_sha256 for v in plan} == set(hashes)
    assert all("No people or faces" in v.prompt for v in plan)
    with pytest.raises(ValueError):
        plan_variations(SPEC, [], 1, 1)


def test_no_two_requests_are_identical():
    """Identical (keyframe, prompt) pairs would come back from the cache as one image."""
    hashes = [f"{i:064x}" for i in range(22)]
    plan = plan_variations(SPEC, hashes, positives=150, negatives=150)
    pairs = {(v.keyframe_sha256, v.prompt) for v in plan}
    assert len(pairs) == 300
    assert len({v.keyframe_sha256 for v in plan if v.label == "yes"}) == 22

    one = plan_variations(SPEC, ["a" * 64], positives=30, negatives=0)
    assert len({v.prompt for v in one}) == 30  # 11 nuisances, then "Variation N." repeats
    assert sum(v.prompt.endswith("Variation 2.") for v in one) == 11
    assert sum(v.prompt.endswith("Variation 3.") for v in one) == 8


def test_negatives_that_differ_only_in_case_or_a_period_are_planned_once():
    spec = SPEC.model_copy(update={"negatives": ["The burner is off", "the burner is off.",
                                                 "the burner is off",
                                                 "a pan being stirred on it"]})
    plan = plan_variations(spec, ["a" * 64], positives=0, negatives=40)
    assert {v.condition for v in plan} == {"The burner is off", "a pan being stirred on it"}
    assert len({v.prompt for v in plan}) == 40


def test_the_manifest_records_this_run_even_when_it_stops(tmp_path):
    import json

    from backend.dataengine.synth import TooManyFailures, generate
    from backend.meta.client import OutputError

    class Rejects:
        def image_edit(self, image, prompt, skill=None):
            raise OutputError("no images returned")

    (tmp_path / "manifest.jsonl").write_text('{"id": "from an older plan"}\n')
    plan = plan_variations(SPEC, ["a" * 64], positives=4, negatives=3)
    with pytest.raises(TooManyFailures, match="6 of 7 image edits failed"):
        generate(Rejects(), plan, {"a" * 64: b"jpeg"}, tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and all(r["file"] is None and r["error"] for r in rows)
    assert [r["id"] for r in rows] == [v.id for v in plan[:6]]
