import base64
import io
import json
import math

import httpx
import pytest
from PIL import Image

from backend.dataengine.dataset import SYSTEM_PROMPT, build_record, user_prompt
from backend.inference.student import (
    OpenAIStudent,
    StudentError,
    answer_probability,
    frame_message,
)
from backend.skills.spec import SkillSpec, Target

SPEC = SkillSpec(name="washer_done", request="tell me when the washer says End",
                 target=Target(sam_prompt="washer display", description="the washer display"),
                 positive="the display shows End", negatives=["display off", "time left"],
                 alert="Laundry's done.")


def jpeg(size=(1280, 720)):
    buf = io.BytesIO()
    Image.new("RGB", size, (30, 200, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def token(text, logprob, tops=()):
    return {"token": text, "logprob": logprob,
            "top_logprobs": [{"token": t, "logprob": lp} for t, lp in tops]}


def reply(content_tokens):
    return {"choices": [{"message": {"role": "assistant", "content": "yes"},
                         "logprobs": {"content": content_tokens}, "finish_reason": "stop"}]}


def test_the_answer_token_gives_the_probability_after_any_reasoning():
    tokens = [token("The", -0.1), token(" display", -0.2), token(" says", -0.1),
              token(" no", -0.3),  # inside the reasoning: not the answer
              token(" End", -0.1), token("yes", math.log(0.6),
                                         [("yes", math.log(0.6)), ("Yes", math.log(0.1)),
                                          ("no", math.log(0.2)), ("maybe", math.log(0.1))])]
    p, answer = answer_probability(tokens)
    assert answer == "yes" and p == pytest.approx(0.7 / 0.9)


@pytest.mark.parametrize("tokens", [[], [token("maybe", -0.1)], None])
def test_no_answer_token_is_an_error_not_a_guess(tokens):
    with pytest.raises(StudentError, match="no yes or no"):
        answer_probability(tokens)


def test_frames_are_sent_as_the_training_examples_were_built(tmp_path):
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(200, json=reply([token("no", math.log(0.9),
                                                     [("yes", math.log(0.1))])]))

    student = OpenAIStudent("http://mac.local:8080/v1/", "trained-eye", api_key="k",
                            transport=httpx.MockTransport(handler))
    score = student.score(jpeg(), SPEC)
    assert score.answer == "no" and score.p == pytest.approx(0.1)
    [request] = sent
    assert str(request.url) == "http://mac.local:8080/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer k"
    body = json.loads(request.content)
    assert body["logprobs"] is True and body["temperature"] == 0
    system, user = body["messages"]
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    image_part, text_part = user["content"]
    assert text_part["text"] == user_prompt(SPEC.positive, SPEC.negatives)
    png = base64.b64decode(image_part["image_url"]["url"].split(",", 1)[1])
    assert Image.open(io.BytesIO(png)).size == (512, 512)
    # the same text and system turn as the dataset
    (tmp_path / "a.png").write_bytes(png)
    from backend.dataengine.dataset import Example
    record = build_record(Example("x", png, "no", "real", SPEC.name, SPEC.positive,
                                  list(SPEC.negatives)), tmp_path / "a.png")
    assert record["messages"][0]["content"] == system["content"]
    assert record["messages"][1]["content"][1]["text"] == text_part["text"]


def test_system_prompt_can_be_folded_into_the_user_text():
    [user] = frame_message(jpeg(), SPEC, system_in_user=True)
    assert user["content"][1]["text"].startswith(SYSTEM_PROMPT)


@pytest.mark.parametrize("response", [
    httpx.Response(500, text="oom"),
    httpx.Response(200, json={"choices": [{"message": {"content": "yes"}}]}),
    httpx.Response(200, text="not json"),
])
def test_endpoint_failures_are_student_errors(response):
    student = OpenAIStudent("http://x/v1", "m", transport=httpx.MockTransport(lambda r: response))
    with pytest.raises(StudentError):
        student.score(jpeg(), SPEC)
