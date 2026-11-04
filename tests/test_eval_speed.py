"""backend/eval/speed.py: per-frame tokens and seconds from a student endpoint (faked)."""

import json

import httpx

from backend.eval.speed import measure, summarize
from backend.skills.spec import SkillSpec, Target

SPEC = SkillSpec(name="fridge_left_open", request="tell me when the fridge door is left open",
                 target=Target(sam_prompt="refrigerator door", description="the fridge"),
                 positive="the fridge door is open", negatives=["the doors are closed"],
                 alert="The fridge is open.")


class Frame:
    def __init__(self, t):
        self.t_s, self.data = t, b"\xff\xd8\xff\xe0frame"


def test_it_records_tokens_reasoning_and_seconds_per_frame(monkeypatch):
    from backend.eval import speed

    monkeypatch.setattr(speed, "frame_message", lambda image, spec: [{"role": "user"}])
    sent = []

    def endpoint(request):
        body = json.loads(request.content)
        sent.append(body)
        reasoning = "The door is open." if len(sent) % 2 else ""
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "Yes", "reasoning": reasoning}}],
            "usage": {"completion_tokens": 53 if reasoning else 4, "prompt_tokens": 474}})

    http = httpx.Client(base_url="http://x/v1", transport=httpx.MockTransport(endpoint))
    rows = measure(http, "muse-glimmer", SPEC, [Frame(0.0), Frame(1.0), Frame(2.0)])
    assert [r["completion_tokens"] for r in rows] == [53, 4, 53]
    assert [bool(r["reasoning_chars"]) for r in rows] == [True, False, True]
    assert sent[0]["logprobs"] is True and sent[0]["chat_template_kwargs"] == {
        "reasoning_strength": "low"}  # the student's own request
    result = summarize(rows)
    assert (result["frames"], result["median_completion_tokens"],
            result["frames_with_reasoning"]) == (3, 53, 2)
