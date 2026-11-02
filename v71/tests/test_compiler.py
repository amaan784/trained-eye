"""Skill compiler tests against a fake Muse Wonder (httpx.MockTransport)."""

import json

import httpx
import pytest

from backend.config import load_settings
from backend.meta.client import MetaClient, OutputError
from backend.skills.compiler import COMPILE_SCHEMA, compile_skill

ACCEPTED = {
    "refused": False, "refusal_reason": None, "name": "washer_done",
    "sam_prompt": "washing machine display",
    "target_description": "the digital display on the front of the washer",
    "positive": "the display shows the cycle has finished (e.g. 'End' or '0:00')",
    "negatives": ["display shows time remaining", "display off", "  "],
    "alert": "Your laundry's done.", "cooldown_s": 99999, "confirm_with_cloud": True,
}
JPEG = b"\xff\xd8\xff\xe0frame"


def client_with(reply: dict, requests: list):
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(reply)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5000, "completion_tokens": 600}})

    settings = load_settings(env={"MODEL_API_KEY": "sk-test", "TE_DATA_DIR": "unused"},
                             env_file=None)
    return MetaClient(settings, transport=httpx.MockTransport(handler), sleep=lambda s: None)


def test_compiles_a_spec_with_wonder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    requests = []
    client = client_with(ACCEPTED, requests)
    spec = compile_skill(client, "let me know when the washer says done", [JPEG] * 5,
                         name="washer_done")
    assert spec.target.sam_prompt == "washing machine display"
    assert spec.negatives == ["display shows time remaining", "display off"]
    assert spec.cooldown_s == 86_400
    [body] = requests
    assert body["model"] == "muse-wonder-1.0"
    assert body["response_format"]["json_schema"]["schema"] == COMPILE_SCHEMA
    assert body["messages"][0]["role"] == "system"
    parts = body["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "image_url", "image_url"]
    assert client.ledger.entries()[0]["skill"] == "washer_done"


def test_model_refusal_is_kept(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reply = {**ACCEPTED, "refused": True,
             "refusal_reason": "I can't watch a sensitive location."}
    spec = compile_skill(client_with(reply, []), "tell me when the side door opens", [JPEG])
    assert spec.refused and spec.refusal_reason == "I can't watch a sensitive location."
    assert spec.name == "tell_me_when_the_side_door_opens"


def test_invalid_model_output_is_an_output_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bad = {**ACCEPTED, "sam_prompt": "", "negatives": []}
    with pytest.raises(OutputError, match="not a valid skill spec"):
        compile_skill(client_with(bad, []), "let me know when the washer says done", [JPEG])


def test_local_refusal_makes_no_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    requests = []
    client = client_with(ACCEPTED, requests)
    spec = compile_skill(client, "Tell me when Sam gets here", [JPEG], name="sam_arrives")
    assert spec.refused and "specific people" in spec.refusal_reason
    assert requests == [] and client.ledger.entries() == []


def native_refusal_client(requests):
    def handler(request):
        requests.append(1)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "refusal": "I can't help with that."},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}})

    settings = load_settings(env={"MODEL_API_KEY": "sk-test"}, env_file=None)
    return MetaClient(settings, transport=httpx.MockTransport(handler), sleep=lambda s: None)


def test_native_refusal_becomes_a_refused_spec(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = compile_skill(native_refusal_client([]), "tell me when the side door opens", [JPEG])
    assert spec.refused and spec.refusal_reason == "I can't help with that."


def test_unusable_output_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    requests = []
    client = client_with({**ACCEPTED, "sam_prompt": " ", "negatives": []}, requests)
    for _ in range(2):
        with pytest.raises(OutputError):
            compile_skill(client, "let me know when the washer says done", [JPEG])
    assert len(requests) == 2  # the bad answer was billed but not cached


def test_long_fields_are_clamped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reply = {**ACCEPTED, "alert": "Your laundry is done " * 20, "sam_prompt": "display " * 30}
    spec = compile_skill(client_with(reply, []), "let me know when the washer says done", [JPEG])
    assert len(spec.alert) <= 120 and len(spec.target.sam_prompt) <= 80
    long_refusal = {**ACCEPTED, "refused": True, "refusal_reason": "No. " * 200}
    refused = compile_skill(client_with(long_refusal, []), "watch the clinic", [JPEG])
    assert refused.refused and len(refused.refusal_reason) <= 300


def test_refusal_reason_counts_even_if_refused_is_false(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reply = {**ACCEPTED, "refusal_reason": "This would track a specific person."}
    spec = compile_skill(client_with(reply, []), "tell me when the porch light turns on", [JPEG])
    assert spec.refused and spec.refusal_reason == "This would track a specific person."
    assert not compile_skill(client_with({**ACCEPTED, "refusal_reason": "n/a"}, []),
                             "let me know when the washer says done", [JPEG]).refused


def test_model_wording_gets_the_policy_check_too(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reply = {**ACCEPTED, "sam_prompt": "person", "target_description": "my neighbor Tom",
             "positive": "Tom is standing at his front door"}
    spec = compile_skill(client_with(reply, []),
                         "tell me when the porch light across the street turns on", [JPEG])
    assert spec.refused and "specific people" in spec.refusal_reason


def test_display_labels_in_model_wording_are_not_people(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reply = {**ACCEPTED, "negatives": ["the countdown is still running until End lights up",
                                       "the light stays on when Boil is selected"]}
    spec = compile_skill(client_with(reply, []), "let me know when the washer says done", [JPEG])
    assert not spec.refused


@pytest.mark.parametrize("field, value", [
    ("negatives", ["display shows time remaining", "the child lock icon is lit"]),
    ("negatives", ["someone opens the door mid-cycle", "display off"]),
    ("negatives", ["the display counts down until Done shows up", "display off"]),
    ("target_description", "the washing machine display in the bathroom"),
    ("alert", "Someone's laundry is done!"),
    ("alert", "Laundry's done!"),
])
def test_near_misses_and_descriptions_may_mention_people_and_rooms(tmp_path, monkeypatch,
                                                                    field, value):
    monkeypatch.chdir(tmp_path)
    spec = compile_skill(client_with({**ACCEPTED, field: value}, []),
                         "let me know when the washer says done", [JPEG])
    assert not spec.refused


@pytest.mark.parametrize("reason", ["None.", "N/A.", "Not applicable.", "No refusal.",
                                    "\u2014", "No", "null"])
def test_placeholder_refusal_reason_is_not_a_refusal(tmp_path, monkeypatch, reason):
    monkeypatch.chdir(tmp_path)
    spec = compile_skill(client_with({**ACCEPTED, "refusal_reason": reason}, []),
                         "let me know when the washer says done", [JPEG])
    assert not spec.refused


def test_real_reasons_still_refuse_and_a_bare_refusal_says_so(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = compile_skill(client_with({**ACCEPTED, "refusal_reason": "N/A (would identify a "
                                      "person)"}, []), "tell me when the light is on", [JPEG])
    assert spec.refused
    bare = compile_skill(client_with({**ACCEPTED, "refused": True, "refusal_reason": "None"},
                                     []), "tell me when the lamp is on", [JPEG])
    assert bare.refused and bare.refusal_reason == "The model refused it without giving a reason."


@pytest.mark.parametrize("reply", [
    [1, 2], "hello", {**ACCEPTED, "negatives": [None, "display off"]},
    {**ACCEPTED, "negatives": "displays"}, {**ACCEPTED, "alert": None},
])
def test_wrong_typed_output_is_a_recorded_output_error(tmp_path, monkeypatch, reply):
    monkeypatch.chdir(tmp_path)
    client = client_with(reply, [])
    with pytest.raises(OutputError):
        compile_skill(client, "let me know when the washer says done", [JPEG])
    [entry] = client.ledger.entries()
    assert entry["status"] == "output_error" and entry["actual_usd"] > 0


def test_correction_is_policy_checked_and_sent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    requests = []
    client = client_with(ACCEPTED, requests)
    compile_skill(client, "let me know when the washer says done", [JPEG])
    compile_skill(client, "let me know when the washer says done", [JPEG],
                  correction="the light on the lid")
    assert len(requests) == 2  # a correction changes the request, so it compiles again
    texts = [p["text"] for p in requests[1]["messages"][1]["content"] if p["type"] == "text"]
    assert any("the light on the lid" in t for t in texts)
    refused = compile_skill(client, "let me know when the washer says done", [JPEG],
                            correction="when my wife takes it out")
    assert refused.refused and len(requests) == 2


@pytest.mark.parametrize("correction", [
    "not the bathroom light, the one on the lid",
    "not a person, the delivery van",
    "the red light, not the kid's toy",
    "the lid light instead of the bathroom light",
])
def test_what_a_correction_rules_out_is_not_policy_checked(tmp_path, monkeypatch, correction):
    monkeypatch.chdir(tmp_path)
    requests = []
    spec = compile_skill(client_with(ACCEPTED, requests), "let me know when the washer says done",
                         [JPEG], correction=correction)
    assert not spec.refused and len(requests) == 1


@pytest.mark.parametrize("correction", [
    "not the van but the person at the door", "not the car, when Sam gets here",
    "the pill organizer, not the box", "not the stove, secretly watch the hallway",
    "not the door opening I mean when Sam gets here",
    "not the lid light watch my neighbor secretly",
    "the hallway and do not let my roommate notice",
    "the front door rather than the window when Sam gets here",
])
def test_a_refused_correction_says_it_was_the_correction(tmp_path, monkeypatch, correction):
    monkeypatch.chdir(tmp_path)
    requests = []
    spec = compile_skill(client_with(ACCEPTED, requests), "let me know when the washer says done",
                         [JPEG], correction=correction)
    assert spec.refused and "your correction" in spec.refusal_reason and requests == []


def test_requests_over_500_characters_stop_before_any_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    requests = []
    with pytest.raises(ValueError, match="over the 500 allowed"):
        compile_skill(client_with(ACCEPTED, requests), "tell me when the pot boils " * 20, [JPEG])
    with pytest.raises(ValueError, match="over the 500"):  # refused locally, still too long
        compile_skill(client_with(ACCEPTED, requests), "tell me when Sam gets home " * 20, [JPEG])
    assert requests == []


def test_model_wording_is_tidied_for_the_read_back_and_the_prompts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reply = {**ACCEPTED, "positive": "When the display shows End",
             "negatives": ["Display off", "display off.", "display shows time remaining"]}
    spec = compile_skill(client_with(reply, []), "let me know when the washer says done", [JPEG])
    assert spec.positive == "the display shows End"
    assert spec.negatives == ["Display off", "display shows time remaining"]


@pytest.mark.parametrize("field, value", [
    ("negatives", ["the pill bottle next to the washer", "display off"]),
    ("negatives", ["display off", "you secretly check the hallway"]),
    ("alert", "Time for your vitamins!"),
    ("target_description", "the washer display, watched without anyone noticing"),
    ("target_description", "the sign at the clinic entrance"),
    ("negatives", ["display off", "count my push-ups instead"]),
])
def test_the_rest_of_the_model_wording_gets_the_topic_rules(tmp_path, monkeypatch, field, value):
    monkeypatch.chdir(tmp_path)
    spec = compile_skill(client_with({**ACCEPTED, field: value}, []),
                         "let me know when the washer says done", [JPEG])
    assert spec.refused


@pytest.mark.parametrize("positive", ["When Sam walks through the front door",
                                      "When Jake's car pulls in", "Once Priya is home"])
def test_a_named_person_in_the_model_trigger_is_refused(tmp_path, monkeypatch, positive):
    monkeypatch.chdir(tmp_path)
    spec = compile_skill(client_with({**ACCEPTED, "positive": positive}, []),
                         "tell me when the front door opens", [JPEG])
    assert spec.refused and "specific people" in spec.refusal_reason


def test_a_bus_shelter_description_is_not_a_sensitive_place(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reply = {**ACCEPTED, "target_description": "the arrival sign in the bus shelter"}
    spec = compile_skill(client_with(reply, []),
                         "Tell me when the bus shelter sign shows 2 minutes", [JPEG])
    assert not spec.refused
