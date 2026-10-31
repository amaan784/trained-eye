import pytest

from backend.dataengine.synth import plan_variations
from backend.skills.spec import SkillSpec, Target

SPEC = SkillSpec(
    name="washer_done",
    request="let me know when the washer says done",
    target=Target(sam_prompt="washing machine display", description="the washer display"),
    positive="the display shows the cycle has finished (e.g. 'End' or '0:00')",
    negatives=["display shows time remaining", "display off", "a different appliance"],
    alert="Your laundry's done.",
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
