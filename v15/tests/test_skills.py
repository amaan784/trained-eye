import pytest
from pydantic import ValidationError

from backend.cli import main
from backend.skills.spec import SkillSpec, Target, policy_check, read_back, slugify

WASHER = {
    "name": "washer_done",
    "request": "let me know when the washer says done",
    "target": Target(sam_prompt="washing machine display",
                     description="the digital display on the front of the washer"),
    "positive": "the display shows the cycle has finished (e.g. 'End' or '0:00')",
    "negatives": ["display shows time remaining", "display off", "a different appliance"],
    "alert": "Your laundry's done.",
}


def test_spec_matches_spec_md_example():
    spec = SkillSpec(**WASHER)
    data = spec.model_dump()
    assert data["version"] == 1 and data["cooldown_s"] == 600
    assert data["confirm_with_cloud"] is True and data["refused"] is False
    assert data["target"]["sam_prompt"] == "washing machine display"


def test_read_back():
    assert read_back(SkillSpec(**WASHER)) == (
        "I'll watch the digital display on the front of the washer and tell you when the "
        "display shows the cycle has finished (e.g. 'End' or '0:00'). "
        "I'll say: \"Your laundry's done.\""
    )
    refused = SkillSpec(name="sam_arrives", request="tell me when Sam gets here", refused=True,
                        refusal_reason="I can't watch for or recognize specific people.")
    assert read_back(refused).startswith("I can't learn that one. I can't watch for")


@pytest.mark.parametrize("change", [
    {"name": "Washer Done"},
    {"negatives": []},
    {"target": None},
    {"cooldown_s": -1},
    {"extra": 1},
])
def test_invalid_specs(change):
    with pytest.raises(ValidationError):
        SkillSpec(**{**WASHER, **change})


def test_refused_needs_reason():
    with pytest.raises(ValidationError):
        SkillSpec(name="x_y", request="r", refused=True)


@pytest.mark.parametrize("request_text", [
    "Tell me when Sam gets here",
    "tell me when Jake's car pulls in",
    "let me know when my wife arrives",
    "who is at the door?",
    "tell me if someone is at the door",
    "tell me when a stranger walks in",
    "remind me when I forget my pills",
    "tell me how many calories are on my plate",
    "watch my neighbor secretly",
    "check my golf swing",
])
def test_policy_refuses(request_text):
    assert policy_check(request_text) is not None


@pytest.mark.parametrize("request_text", [
    "let me know when the washer says done",
    "tell me before this pot boils over",
    "Tell me when the M4 bus shows up",
    "tell me when the kettle light turns off",
    "let me know when the oven timer says 0:00",
    "When The oven beeps, tell me",
    "tell me when the manual timer on the oven hits zero",
    "let me know when the clock face shows noon",
    "tell me when the dog bowl is empty",
])
def test_policy_allows(request_text):
    assert policy_check(request_text) is None


def test_slugify():
    assert slugify("Let me know when the washer says done!") == (
        "let_me_know_when_the_washer_says_done")
    assert slugify("4 o'clock bus") == "skill_4_o_clock_bus"
    assert slugify("") == "skill"


def test_policy_command(capsys):
    assert main(["policy", "Tell me when Sam gets here"]) == 1
    assert "refused: I can't watch for or recognize specific people" in capsys.readouterr().out
    assert main(["policy", "let me know when the washer says done"]) == 0
    assert capsys.readouterr().out.startswith("ok:")


def test_read_back_handles_capitalised_model_phrasing():
    spec = SkillSpec(**{**WASHER, "target": Target(sam_prompt="washer display",
                                                   description="The digital display on the washer."),
                        "positive": "A green light shows next to End."})
    assert read_back(spec).startswith(
        "I'll watch the digital display on the washer and tell you when a green light "
        "shows next to End.")
    keeps_acronym = SkillSpec(**{**WASHER, "positive": "LED shows END"})
    assert "tell you when LED shows END." in read_back(keeps_acronym)


@pytest.mark.parametrize("name", ["washer-done", "Washer", "x", "1washer", "a" * 41])
def test_check_name_rejects(name):
    from backend.skills.spec import check_name

    with pytest.raises(ValueError, match="snake_case"):
        check_name(name)


@pytest.mark.parametrize("request_text", [
    "When Sam gets here, tell me",
    "If Dave leaves the house, tell me",
    "Tell me when Dave's home",
    "Tell me when Sarah pulls into the driveway",
    "Let me know as soon as Mike gets home",
    "Notify me when Mike's truck arrives",
    "Tell me when she gets home",
    "Tell me when he leaves the house",
    "Tell me when my landlord shows up",
    "recognize her when she walks past",
    "Tell me when the line at the abortion clinic is short",
    "Tell me when the locker room light turns on",
    "Tell me when the door of the mosque opens",
    "Remind me when I take my meds",
    "Tell me when my glucose monitor reads below 70",
    "Tell me when the thermometer shows a fever",
    "Watch the house across the street without her knowing",
])
def test_policy_refuses_review_cases(request_text):
    assert policy_check(request_text) is not None


@pytest.mark.parametrize("request_text", [
    "Learn to recognize when the washer is done",
    "Recognize when the pot is about to boil over",
    "Tell me when the crossing shows the green man",
    "Tell me when the kitchen scale shows the right weight",
    "Tell me when the chicken skin is crispy",
    "Tell me when the celery stalks are soft",
    "Tell me when the meat thermometer reads 70",
    "Tell me when the Amazon truck arrives",
    "Tell me when Roomba is stuck",
    "Tell me when Alexa lights up",
    "Tell me when Instant Pot shows done",
    "let me know when Bosch dishwasher is finished",
    "Tell me when Tesla starts charging",
    "The countdown is still running until End lights up",
])
def test_policy_allows_object_skills(request_text):
    assert policy_check(request_text) is None
