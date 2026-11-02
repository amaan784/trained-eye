"""MetaClient tests. Every request goes to an httpx.MockTransport; nothing reaches the
network and nothing is billed."""

import base64
import io
import json
import wave

import httpx
import pytest

from backend.config import ConfigError, MissingSecret, load_settings
from backend.meta.client import (
    CacheMiss,
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
        if isinstance(reply, BaseException):
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
    assert again.info.billed_usd == pytest.approx(first.info.cost_usd)  # kept with the reply
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


def test_retries_stop_at_the_spend_cap(tmp_path):
    rec = Recorder(*[httpx.Response(502, text="bad gateway")] * 6)
    client, sleeps = make_client(tmp_path, rec, TE_SKILL_CAP_USD="0.10")
    with pytest.raises(BudgetExceeded, match="skill 'washer_done'"):
        client.image_generate("x", skill="washer_done", n=4)  # $0.04 per attempt
    assert len(rec.requests) == 2 and len(sleeps) == 1
    [entry] = client.ledger.entries()
    assert entry["status"] == "error" and entry["actual_usd"] == pytest.approx(0.08)
    assert client.ledger.spent("washer_done") <= 0.10


@pytest.mark.parametrize("body", [
    {**chat_reply('{"label": "yes"}'), "usage": "n/a"},
    {**chat_reply('{"label": "yes"}'), "usage": [1, 2]},
    {"choices": [None], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    chat_reply([{"type": "text", "text": '{"label": "yes"}'}]),
    httpx.Response(200, content=json.dumps(chat_reply('{"label": "yes"}')).replace(
        '"prompt_tokens": 1000', '"prompt_tokens": 1e999').encode()),  # parses as inf
])
def test_odd_200_replies_are_always_recorded(tmp_path, body):
    client, _ = make_client(tmp_path, Recorder(body))
    try:
        client.structured("muse-spark-1.3", "x", SCHEMA)
    except OutputError:
        pass
    [entry] = client.ledger.entries()
    assert entry["actual_usd"] > 0  # served, so counted against the caps either way


@pytest.mark.parametrize("duration", ["NaN", "-9e9", "1e999"])
def test_a_reported_cost_that_is_not_a_cost_is_charged_at_the_estimate(tmp_path, duration):
    body = ('{"text": "tell me when the pot boils", "duration": ' + duration + "}").encode()
    rec = Recorder(httpx.Response(200, content=body), chat_reply('{"label": "yes"}'))
    client, _ = make_client(tmp_path, rec)
    client.transcribe(wav_bytes(), skill="pot_boils")
    [entry] = client.ledger.entries()
    assert entry["actual_usd"] == pytest.approx(0.18 * 2 / 3600) and entry["usage"]["unpriced"]
    client.structured("muse-spark-1.3", "x", SCHEMA, skill="pot_boils")  # the caps still work
    assert client.ledger.spent() > entry["actual_usd"]


def test_ctrl_c_during_a_call_is_charged(tmp_path):
    rec = Recorder(KeyboardInterrupt(), chat_reply('{"label": "yes"}'))
    client, _ = make_client(tmp_path, rec)
    with pytest.raises(KeyboardInterrupt):
        client.structured("muse-spark-1.3", "x", SCHEMA, skill="washer_done")
    [entry] = client.ledger.entries()
    assert entry["status"] == "error" and entry["actual_usd"] > 0
    assert client.ledger.spent("washer_done") == entry["actual_usd"]


def test_failed_calls_carry_their_charge(tmp_path):
    rec = Recorder(chat_reply("not json"), httpx.Response(400, text="bad image"))
    client, _ = make_client(tmp_path, rec)
    with pytest.raises(OutputError) as unusable:
        client.structured("muse-spark-1.3", "x", SCHEMA)
    with pytest.raises(MetaAPIError) as rejected:
        client.structured("muse-spark-1.3", "y", SCHEMA)
    billed, free = client.ledger.entries()
    assert unusable.value.cost_usd == billed["actual_usd"] > 0
    assert rejected.value.cost_usd == free["actual_usd"] == 0


@pytest.mark.parametrize("reply", [
    chat_reply("", finish_reason="length"),
    chat_reply('{"label": "ye', finish_reason="length"),  # cut off mid-answer
])
def test_a_reply_cut_off_for_length_says_so(tmp_path, reply):
    client, _ = make_client(tmp_path, Recorder(reply))
    with pytest.raises(OutputError, match="finish_reason=length") as err:
        client.structured("muse-spark-1.3", "x", SCHEMA)
    assert err.value.finish_reason == "length"


def test_cache_only_never_sends(tmp_path):
    rec = Recorder(chat_reply('{"label": "no"}'))
    client, _ = make_client(tmp_path, rec)
    with pytest.raises(CacheMiss):
        client.structured("muse-spark-1.3", "x", SCHEMA, cache_only=True)
    assert rec.requests == [] and client.ledger.entries() == []
    client.structured("muse-spark-1.3", "x", SCHEMA)
    hit = client.structured("muse-spark-1.3", "x", SCHEMA, cache_only=True)
    assert hit.info.cached and hit.data == {"label": "no"}


def test_a_cached_reply_that_no_longer_passes_is_fetched_again(tmp_path):
    rec = Recorder(chat_reply('{"label": "maybe"}'), chat_reply('{"label": "yes"}'))
    client, _ = make_client(tmp_path, rec)
    client.structured("muse-spark-1.3", "x", SCHEMA)  # cached under looser checks

    def strict(data):
        if data["label"] not in ("yes", "no"):
            raise OutputError("label must be yes or no")

    fresh = client.structured("muse-spark-1.3", "x", SCHEMA, validate=strict)
    assert fresh.data == {"label": "yes"} and not fresh.info.cached
    again = client.structured("muse-spark-1.3", "x", SCHEMA, validate=strict)
    assert again.info.cached and len(rec.requests) == 2


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


# -- media endpoints ---------------------------------------------------------------

PNG = b"\x89PNG\r\n\x1a\n" + b"generated-with-provenance-metadata"


def wav_bytes(seconds=2.0, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(seconds * rate))
    return buf.getvalue()


def test_transcribe(tmp_path):
    rec = Recorder({"text": "let me know when the washer says done", "duration": 2.0,
                    "segments": [{"start": 0.0, "end": 2.0, "text": "let me know..."}]})
    client, _ = make_client(tmp_path, rec)
    t = client.transcribe(wav_bytes(), skill="washer_done")
    assert t.text.startswith("let me know") and t.duration_s == 2.0 and len(t.segments) == 1
    assert t.info.cost_usd == pytest.approx(0.18 * 2 / 3600)
    [entry] = client.ledger.entries()
    assert entry["skill"] == "washer_done" and entry["actual_usd"] == t.info.cost_usd
    request = rec.requests[0]
    assert request.url.path == "/v1/audio/transcriptions"
    assert b'name="model"' in request.content and b"muse-voice-transcribe-1.0" in request.content
    assert client.transcribe(wav_bytes(), skill="washer_done").info.cached
    assert len(client.ledger.entries()) == 1
    with pytest.raises(ValueError, match="duration_s"):
        client.transcribe(b"not a wav")


@pytest.mark.parametrize("reply", [{"text": ""}, {"text": "  "}, {}, {"text": None}])
def test_an_empty_transcript_is_billed_but_not_cached(tmp_path, reply):
    rec = Recorder(reply, {"text": "tell me when the pot boils"})
    client, _ = make_client(tmp_path, rec)
    with pytest.raises(OutputError, match="no words.*--request"):
        client.transcribe(wav_bytes(), skill="pot_boils")
    assert client.transcribe(wav_bytes(), skill="pot_boils").text == "tell me when the pot boils"
    assert [e["status"] for e in client.ledger.entries()] == ["output_error", "ok"]


def sse(*events):
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)


LANE = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
SAM_LINE = "<0f>0<|box;x1=10;y1=20;x2=299;y2=199;w=640;h=480;c=0.91|><|mask;x=0;y=0;data=2,2,!!!!#Kq|>"


def test_sam_streamed(tmp_path):
    rec = Recorder(sse(
        {"type": "response.output_text.delta", "delta": SAM_LINE[:40], **LANE},
        {"type": "response.output_text.delta", "delta": SAM_LINE[40:] + "\n", **LANE},
        {"type": "response.content_part.done", **LANE,
         "part": {"type": "output_text", "text": SAM_LINE + "\n"}},
        {"type": "response.completed", "response": {"usage": {"input_tokens": 900}}},
    ))
    client, _ = make_client(tmp_path, rec)
    result = client.sam_image("washing machine display", JPEG, skill="washer_done")
    assert result.text == SAM_LINE + "\n"
    assert result.info.cost_usd == pytest.approx(0.0025)
    body = json.loads(rec.requests[0].content)
    assert body["model"] == "sam-3.1" and body["stream"] is True
    assert body["metadata"] == {"include_confidence": "true"}
    assert [p["type"] for p in body["input"][0]["content"]] == ["input_text", "input_image"]
    assert client.sam_image("washing machine display", JPEG, skill="washer_done").info.cached
    [entry] = client.ledger.entries()
    assert entry["model"] == "sam-3.1" and entry["actual_usd"] == pytest.approx(0.0025)


def test_sam_stream_with_crlf_and_line_separator(tmp_path):
    events = [
        {"type": "response.output_text.delta", "delta": "<obj>pot\u2028lid"},
        {"type": "response.completed", "response": {"usage": {}}},
    ]
    body = "".join(f"event: {e['type']}\r\ndata: {json.dumps(e, ensure_ascii=False)}\r\n\r\n"
                   for e in events)
    rec = Recorder(httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body))
    client, _ = make_client(tmp_path, rec)
    assert client.sam_image("pot", JPEG).text == "<obj>pot\u2028lid"


@pytest.mark.parametrize("last", [
    {"type": "response.incomplete", "response": {"status": "incomplete"}},
    {"type": "response.failed", "response": {"status": "failed"}},
    {"type": "error", "message": "boom"},
])
def test_sam_unfinished_stream_is_billed_not_cached(tmp_path, last):
    delta = {"type": "response.output_text.delta", "delta": "<obj>pot<box>10 2"}
    rec = Recorder(sse(delta, last), sse(delta, last))
    client, _ = make_client(tmp_path, rec)
    for _ in range(2):
        with pytest.raises(OutputError, match="SAM stream ended with"):
            client.sam_image("pot", JPEG)
    assert len(rec.requests) == 2
    assert all(e["status"] == "output_error" and e["actual_usd"] == 0.0025
               for e in client.ledger.entries())


def test_sam_non_streamed_and_empty(tmp_path):
    rec = Recorder({"output": [{"content": [{"type": "output_text", "text": "<none/>"}]}]},
                   {"output": []})
    client, _ = make_client(tmp_path, rec)
    assert client.sam_image("pot", JPEG).text == "<none/>"
    assert client.sam_image("lid", JPEG).text == ""  # SAM found nothing: an answer


@pytest.mark.parametrize("events, message", [
    ([{"type": "response.output_text.delta", "delta": "<0f>", **LANE},
      {"type": "response.output_text.delta", "delta": "0", **LANE, "item_id": "msg_2"},
      {"type": "response.completed", "response": {}}], "more than one output text lane"),
    ([{"type": "response.output_text.delta", "delta": "<0f>0", **LANE},
      {"type": "response.output_text.done", "text": "<0f>9", **LANE},
      {"type": "response.completed", "response": {}}], "does not match"),
    ([{"type": "response.refusal.done", "refusal": "no", **LANE}], "refused"),
    ([{"type": "response.output_text.delta", "delta": "<0f>0", **LANE}], "did not complete"),
])
def test_sam_stream_lanes_follow_the_protocol(tmp_path, events, message):
    client, _ = make_client(tmp_path, Recorder(sse(*events)))
    with pytest.raises(OutputError, match=message):
        client.sam_image("pot", JPEG)


def test_sam_done_text_alone_is_the_whole_output(tmp_path):
    rec = Recorder(sse({"type": "response.output_text.done", "text": SAM_LINE, **LANE},
                       {"type": "response.completed", "response": {"usage": {}}}))
    client, _ = make_client(tmp_path, rec)
    assert client.sam_image("pot", JPEG).text == SAM_LINE


def test_image_edit_keeps_bytes_exactly(tmp_path):
    rec = Recorder({"data": [{"b64_json": base64.b64encode(PNG).decode(),
                              "revised_prompt": "washer display showing End"}]})
    client, _ = make_client(tmp_path, rec)
    result = client.image_edit(JPEG, "make the display show End", skill="washer_done")
    assert result.images == [PNG] and result.revised_prompts == ["washer display showing End"]
    assert result.info.cost_usd == pytest.approx(0.01)
    request = rec.requests[0]
    assert request.url.path == "/v1/images/edits"
    assert b'name="image"' in request.content and b'name="prompt"' in request.content
    cached = client.image_edit(JPEG, "make the display show End", skill="washer_done")
    assert cached.info.cached and cached.images == [PNG]
    [entry] = client.ledger.entries()
    assert entry["skill"] == "washer_done" and entry["actual_usd"] == pytest.approx(0.01)
    assert b"b64_json" not in (tmp_path / "cache").joinpath(
        result.info.cache_key[:2], f"{result.info.cache_key}.json").read_bytes()


def test_image_url_download_never_sends_the_key(tmp_path):
    rec = Recorder({"data": [{"url": "https://cdn.example.test/img/1.png"}]},
                   httpx.Response(302, headers={"location": "https://store.example.test/1.png"}),
                   httpx.Response(200, content=PNG))
    client, _ = make_client(tmp_path, rec)
    assert client.image_generate("a washer display showing End").images == [PNG]
    generate, download, redirected = rec.requests
    assert generate.url.path == "/v1/images/generations"
    assert json.loads(generate.content)["model"] == "muse-image-1.0"
    assert download.url.host == "cdn.example.test" and "authorization" not in download.headers
    assert redirected.url.host == "store.example.test"
    assert "authorization" not in redirected.headers


def test_a_failed_image_download_is_a_billed_output_error(tmp_path):
    rec = Recorder({"data": [{"url": "https://cdn.example.test/img/1.png"}]},
                   httpx.Response(503), httpx.ReadTimeout("slow cdn"))
    client, sleeps = make_client(tmp_path, rec)
    with pytest.raises(OutputError, match="image download failed: ReadTimeout") as err:
        client.image_edit(JPEG, "make the display show End", skill="washer_done")
    assert len(rec.requests) == 3 and sleeps == [1.0]  # the download is tried twice
    [entry] = client.ledger.entries()
    assert entry["status"] == "output_error" and entry["actual_usd"] == pytest.approx(0.01)
    assert err.value.cost_usd == pytest.approx(0.01)


def test_an_image_reply_without_images_is_charged_for_what_was_asked(tmp_path):
    rec = Recorder({"data": [{"image": "in a field the client does not know"}],
                    "usage": {"images": 2}})
    client, _ = make_client(tmp_path, rec)
    with pytest.raises(OutputError, match="no images"):
        client.image_generate("a washer display showing End", n=2)
    [entry] = client.ledger.entries()
    assert entry["status"] == "output_error" and entry["actual_usd"] == pytest.approx(0.02)


def test_image_generate_with_refs_uses_edits(tmp_path):
    rec = Recorder({"data": [{"b64_json": base64.b64encode(PNG).decode()}] * 2})
    client, _ = make_client(tmp_path, rec)
    result = client.image_generate("same washer, dim light", refs=[JPEG, JPEG + b"2"], n=2)
    assert len(result.images) == 2 and result.info.cost_usd == pytest.approx(0.02)
    assert rec.requests[0].url.path == "/v1/images/edits"
    assert rec.requests[0].content.count(b'name="image[]"') == 2
