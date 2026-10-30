"""MetaClient tests. Every request goes to an httpx.MockTransport; nothing reaches the
network and nothing is billed."""

import base64
import json

import httpx
import pytest

from backend.config import ConfigError, MissingSecret, load_settings
from backend.meta.client import (
    MetaAPIError,
    MetaClient,
    OutputError,
    image_part,
    text_part,
)
from backend.meta.ledger import BudgetExceeded

KEY = "sk-test-not-a-real-key"
SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["yes", "no", "unsure"]}},
    "required": ["label"],
    "additionalProperties": False,
}
JPEG = b"\xff\xd8\xff\xe0fake-jpeg"


def chat_reply(content, usage=None, finish_reason="stop", refusal=None):
    message = {"role": "assistant", "content": content}
    if refusal:
        message["refusal"] = refusal
    body = {"choices": [{"message": message, "finish_reason": finish_reason}]}
    if usage is not False:
        body["usage"] = usage or {"prompt_tokens": 1000, "completion_tokens": 200,
                                  "completion_tokens_details": {"reasoning_tokens": 150}}
    return body


class Recorder:
    """A MockTransport handler that replays queued responses and keeps the requests."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, json=reply)


def make_client(tmp_path, handler, key=KEY, **env):
    settings = load_settings(
        env={"TE_DATA_DIR": str(tmp_path), **({"MODEL_API_KEY": key} if key else {}), **env},
        env_file=None,
    )
    sleeps = []
    client = MetaClient(settings, transport=httpx.MockTransport(handler), sleep=sleeps.append)
    return client, sleeps


def test_structured_call_is_billed_then_cached(tmp_path):
    rec = Recorder(chat_reply('{"label": "yes"}'))
    client, _ = make_client(tmp_path, rec)
    first = client.structured("muse-spark-1.3", [text_part("Is it done?"), image_part(JPEG)],
                              SCHEMA, name="label", skill="washer_done")
    assert first.data == {"label": "yes"}
    assert not first.info.cached and first.reasoning_tokens == 150
    assert first.info.cost_usd == pytest.approx((1000 * 1.25 + 200 * 4.25) / 1e6)

    request = rec.requests[0]
    assert request.url == "https://api.meta.ai/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert body["response_format"]["json_schema"] == {"name": "label", "schema": SCHEMA,
                                                      "strict": True}
    image_url = body["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url == "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()
    assert "reasoning_effort" not in body

    again = client.structured("muse-spark-1.3", [text_part("Is it done?"), image_part(JPEG)],
                              SCHEMA, name="label", skill="washer_done")
    assert again.data == {"label": "yes"} and again.info.cached and again.info.cost_usd == 0
    assert len(rec.requests) == 1
    entries = client.ledger.entries()
    assert len(entries) == 1 and entries[0]["skill"] == "washer_done"
    assert entries[0]["actual_usd"] == pytest.approx(first.info.cost_usd)
    assert entries[0]["estimate_usd"] >= entries[0]["actual_usd"]


def test_cached_fixture_needs_no_key_and_no_network(tmp_path):
    seed, _ = make_client(tmp_path, Recorder(chat_reply('{"label": "no"}')))
    seed.structured("muse-spark-1.3", "frame?", SCHEMA)
    rec = Recorder()
    offline, _ = make_client(tmp_path, rec, key=None)
    assert offline.structured("muse-spark-1.3", "frame?", SCHEMA).data == {"label": "no"}
    assert rec.requests == []
    with pytest.raises(MissingSecret):
        offline.structured("muse-spark-1.3", "a different frame?", SCHEMA)
    assert len(offline.ledger.entries()) == 1


def test_retries_429_with_retry_after_then_succeeds(tmp_path):
    rec = Recorder(httpx.Response(429, headers={"retry-after": "2"}, text="slow down"),
                   httpx.Response(503, text="busy"),
                   chat_reply('{"label": "no"}'))
    client, sleeps = make_client(tmp_path, rec)
    assert client.structured("muse-spark-1.3", "x", SCHEMA).data == {"label": "no"}
    assert len(rec.requests) == 3
    assert sleeps[0] == 2.0 and 2.0 <= sleeps[1] <= 2.5


def test_retries_timeouts(tmp_path):
    rec = Recorder(httpx.ReadTimeout("slow"), httpx.ConnectError("down"),
                   chat_reply('{"label": "no"}'))
    client, sleeps = make_client(tmp_path, rec)
    assert client.structured("muse-spark-1.3", "x", SCHEMA).data == {"label": "no"}
    assert len(sleeps) == 2


def test_gives_up_after_max_retries(tmp_path):
    rec = Recorder(*[httpx.Response(503, text="busy")] * 3)
    client, sleeps = make_client(tmp_path, rec, TE_MAX_RETRIES="2")
    with pytest.raises(MetaAPIError, match="gave up after 3 tries"):
        client.structured("muse-spark-1.3", "x", SCHEMA)
    assert len(rec.requests) == 3 and len(sleeps) == 2
    [entry] = client.ledger.entries()
    assert entry["status"] == "error" and entry["actual_usd"] == 0


def test_attempts_that_may_have_run_are_charged(tmp_path):
    """A read timeout or a 500/502/504 may have been processed and billed upstream."""
    rec = Recorder(httpx.ReadTimeout("slow"), httpx.Response(502, text="bad gateway"),
                   httpx.ConnectError("refused"), chat_reply('{"label": "yes"}'))
    client, _ = make_client(tmp_path, rec)
    result = client.structured("muse-spark-1.3", "x", SCHEMA)
    [entry] = client.ledger.entries()
    reported = (1000 * 1.25 + 200 * 4.25) / 1e6
    assert entry["actual_usd"] == pytest.approx(reported + 2 * entry["estimate_usd"])
    assert result.info.usage["maybe_billed_retries"] == 2

    rec = Recorder(*[httpx.ReadTimeout("slow")] * 3)
    client, _ = make_client(tmp_path / "b", rec, TE_MAX_RETRIES="2")
    with pytest.raises(MetaAPIError, match="network error after 3 tries"):
        client.structured("muse-spark-1.3", "x", SCHEMA)
    [entry] = client.ledger.entries()
    assert entry["actual_usd"] == pytest.approx(3 * entry["estimate_usd"])


def test_secrets_are_redacted_from_errors(tmp_path):
    rec = Recorder(httpx.Response(401, json={"error": f"Invalid API key: {KEY}"}))
    client, _ = make_client(tmp_path, rec)
    with pytest.raises(MetaAPIError) as err:
        client.structured("muse-spark-1.3", "x", SCHEMA)
    assert KEY not in str(err.value) and "[redacted]" in str(err.value)


def test_config_mistakes_are_not_retried(tmp_path):
    rec = Recorder(httpx.UnsupportedProtocol("no scheme"))
    client, sleeps = make_client(tmp_path, rec)
    with pytest.raises(MetaAPIError, match="TE_BASE_URL"):
        client.list_models()
    assert sleeps == []


def test_retry_after_http_date(tmp_path):
    import email.utils
    import time

    when = email.utils.formatdate(time.time() + 30, usegmt=True)
    rec = Recorder(httpx.Response(429, headers={"retry-after": when}),
                   chat_reply('{"label": "no"}'))
    client, sleeps = make_client(tmp_path, rec)
    client.structured("muse-spark-1.3", "x", SCHEMA)
    assert 25 <= sleeps[0] <= 31


@pytest.mark.parametrize("reply", [
    httpx.Response(307, headers={"location": "https://elsewhere.test/v1/chat/completions"}),
    httpx.Response(200, text="<html>captive portal</html>"),
    httpx.Response(200, json=["not", "an", "object"]),
])
def test_non_json_or_redirect_is_a_clean_error(tmp_path, reply):
    client, _ = make_client(tmp_path, Recorder(reply))
    with pytest.raises(MetaAPIError):
        client.structured("muse-spark-1.3", "x", SCHEMA)
    assert client.ledger.entries()[0]["actual_usd"] == 0


@pytest.mark.parametrize("usage, expect", [
    ({"input_tokens": 1000, "output_tokens": 200}, (1000 * 1.25 + 200 * 4.25) / 1e6),
    ({"prompt_tokens": 1000, "completion_tokens": None}, 1000 * 1.25 / 1e6),
    ({"total_tokens": 1200}, None),  # unpriceable: charged at the estimate
])
def test_usage_variants(tmp_path, usage, expect):
    client, _ = make_client(tmp_path, Recorder(chat_reply('{"label": "no"}', usage=usage)))
    result = client.structured("muse-spark-1.3", "x", SCHEMA)
    [entry] = client.ledger.entries()
    assert entry["actual_usd"] == pytest.approx(expect if expect else entry["estimate_usd"])
    assert result.info.cost_usd == entry["actual_usd"]


def test_contributor_models_refused_at_call_time(tmp_path):
    rec = Recorder()
    client, _ = make_client(tmp_path, rec)
    with pytest.raises(ConfigError, match="contributor"):
        client.structured("muse-spark-1.3-Contributor", "x", SCHEMA)
    assert rec.requests == []


def test_schema_property_order_is_part_of_the_key(tmp_path):
    reason_first = {**SCHEMA, "properties": {"reason": {"type": "string"},
                                              "label": SCHEMA["properties"]["label"]}}
    label_first = {**SCHEMA, "properties": {"label": SCHEMA["properties"]["label"],
                                             "reason": {"type": "string"}}}
    rec = Recorder(chat_reply('{"label": "no"}'), chat_reply('{"label": "yes"}'))
    client, _ = make_client(tmp_path, rec)
    assert client.structured("muse-spark-1.3", "x", reason_first).data == {"label": "no"}
    assert client.structured("muse-spark-1.3", "x", label_first).data == {"label": "yes"}
    assert len(rec.requests) == 2


def test_400_is_never_retried(tmp_path):
    rec = Recorder(httpx.Response(400, json={"error": {"message": "bad schema"}}))
    client, sleeps = make_client(tmp_path, rec)
    with pytest.raises(MetaAPIError, match="400") as err:
        client.structured("muse-spark-1.3", "x", SCHEMA)
    assert err.value.status == 400
    assert len(rec.requests) == 1 and sleeps == []


def test_budget_checked_before_sending(tmp_path):
    rec = Recorder()
    client, _ = make_client(tmp_path, rec, TE_SKILL_CAP_USD="0.001")
    with pytest.raises(BudgetExceeded):
        client.structured("muse-spark-1.3", "x", SCHEMA, skill="washer_done", max_tokens=4096)
    assert rec.requests == [] and client.ledger.entries() == []


@pytest.mark.parametrize(
    "reply, message",
    [
        (chat_reply("", finish_reason="length"), "raise max_tokens"),
        (chat_reply(None, refusal="I can't help with that."), "refused"),
        (chat_reply("the answer is yes"), "not JSON"),
    ],
)
def test_unusable_output_is_billed_but_not_cached(tmp_path, reply, message):
    rec = Recorder(reply, reply)
    client, _ = make_client(tmp_path, rec)
    with pytest.raises(OutputError, match=message):
        client.structured("muse-spark-1.3", "x", SCHEMA)
    with pytest.raises(OutputError):
        client.structured("muse-spark-1.3", "x", SCHEMA)
    assert len(rec.requests) == 2
    entries = client.ledger.entries()
    assert all(e["actual_usd"] > 0 and e["status"] == "output_error" for e in entries)
    assert client.ledger.totals()["errors"] == 2


def test_fenced_json_and_missing_usage(tmp_path):
    rec = Recorder(chat_reply('```json\n{"label": "unsure"}\n```', usage=False))
    client, _ = make_client(tmp_path, rec)
    result = client.structured("muse-spark-1.3", "x", SCHEMA, reasoning_effort="low")
    assert result.data == {"label": "unsure"}
    assert result.info.usage == {"missing": True}
    [entry] = client.ledger.entries()
    assert entry["actual_usd"] == entry["estimate_usd"] > 0
    assert json.loads(rec.requests[0].content)["reasoning_effort"] == "low"


def test_list_models(tmp_path):
    rec = Recorder({"data": [{"id": "sam-3.1"}, {"id": "muse-spark-1.3"}]})
    client, _ = make_client(tmp_path, rec)
    assert client.list_models() == ["muse-spark-1.3", "sam-3.1"]
    assert rec.requests[0].method == "GET" and rec.requests[0].url.path == "/v1/models"
    assert client.ledger.entries() == []

