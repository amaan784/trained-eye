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
