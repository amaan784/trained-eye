"""The regression check, against two fake OpenAI-compatible endpoints."""

import json

import httpx
import pytest

from backend.eval.regression import Endpoint, grade, load_questions, run_regression
from tests.test_teach import png

QUESTIONS = [
    {"id": "q1", "image": "img/red.png", "question": "What colour is the square?",
     "answer": "red", "match": "contains"},
    {"id": "q2", "image": "img/red.png", "question": "Is it a square? yes or no",
     "answer": "yes", "match": "exact"},
    {"id": "q3", "question": "Which is a fruit? A) chair B) apple", "answer": "B",
     "match": "choice"},
    {"id": "q4", "question": "What is 2 + 2?", "answer": "4", "match": "exact"},
]


def questions_file(tmp_path, rows=QUESTIONS):
    (tmp_path / "img").mkdir(exist_ok=True)
    (tmp_path / "img/red.png").write_bytes(png((220, 0, 0)))
    path = tmp_path / "questions.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def endpoint(replies, sent, status=200):
    """A fake endpoint answering each question with replies[text of the question]."""

    def handler(request):
        body = json.loads(request.content)
        sent.append((request.url.host, request.headers.get("authorization"), body))
        if status != 200:
            return httpx.Response(status, json={"error": "down"})
        content = body["messages"][0]["content"]
        text = content if isinstance(content, str) else content[-1]["text"]
        reply, finish = replies[text]
        return httpx.Response(200, json={"choices": [
            {"message": {"role": "assistant", "content": reply}, "finish_reason": finish}]})
    return handler


@pytest.mark.parametrize("reply, answer, match, ok", [
    ("Yes.", "yes", "exact", True), ("yes, it is", "yes", "exact", False),
    ("It looks Red to me", "red", "contains", True), ("bored", "red", "contains", False),
    ("B) apple", "B", "choice", True), ("(b)", "B", "choice", True),
    ("Answer: B", "B", "choice", True), ("A, not B", "B", "choice", False),
    ("", "yes", "exact", False), ("12", "2", "contains", False), ("2.5", "2", "contains", False),
    ("about 2 of them", "2", "contains", True), ("**B**. apple", "B", "choice", True),
])
def test_grading(reply, answer, match, ok):
    assert grade(reply, answer, match) is ok


def test_base_and_fine_tuned_are_asked_the_same_and_compared(tmp_path):
    questions = load_questions(questions_file(tmp_path))
    texts = [q["question"] for q in QUESTIONS]
    sent = []
    base = Endpoint("http://base:8000/v1", "glimmer-base", api_key="secret-token",
                    transport=httpx.MockTransport(endpoint(dict(zip(texts, [
                        ("Red.", "stop"), ("yes", "stop"), ("B", "stop"), ("4", "stop")])),
                        sent)))
    tuned = Endpoint("http://tuned:8000/v1", "trained-eye",
                     transport=httpx.MockTransport(endpoint(dict(zip(texts, [
                         ("red", "stop"), ("no", "stop"), ("A", "stop"), ("", "length")])),
                         sent)))
    report = run_regression(questions, base, tuned, precision="bf16")
    assert (report["base"]["correct"], report["base"]["accuracy"]) == (4, 1.0)
    assert (report["finetuned"]["correct"], report["finetuned"]["accuracy"]) == (1, 0.25)
    assert report["finetuned"]["with_image"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert report["finetuned"]["cut_off"] == ["q4"]
    assert [c["id"] for c in report["changed"]] == ["q2", "q3", "q4"]
    for host, auth, body in sent:
        assert body["temperature"] == 0 and body["model"] == {
            "base": "glimmer-base", "tuned": "trained-eye"}[host]
        assert auth == ("Bearer secret-token" if host == "base" else None)
    image_part = sent[0][2]["messages"][0]["content"][0]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert isinstance(sent[4][2]["messages"][0]["content"], str)  # text only: plain text
    table = report["markdown"]
    assert "| All (4) | 4/4 (100%) | 1/4 (25%) |" in table
    assert "| Text only (2) | 2/2 (100%) | 0/2 (0%) |" in table
    assert "q2 (right to wrong)" in table and "1 replies cut off" in table
    assert "secret-token" not in json.dumps(report)


def test_a_failed_request_counts_as_wrong_and_is_listed(tmp_path):
    questions = load_questions(questions_file(tmp_path))
    sent = []
    down = Endpoint("http://tuned/v1", "trained-eye",
                    transport=httpx.MockTransport(endpoint({}, sent, status=503)))
    report = run_regression(questions[:1], down, down, precision="nf4")
    assert "temperature" not in Endpoint("http://x/v1", "m", temperature=None).extra
    assert report["base"]["failed"] == ["q1"] and report["base"]["correct"] == 0
    assert "503" in report["rows"][0]["base"]["error"] and not report["changed"]
    assert "1 failed requests" in report["markdown"]


@pytest.mark.parametrize("row, message", [
    ({"id": "x", "question": "?", "answer": "a", "match": "fuzzy"}, "match must be one of"),
    ({"id": "x", "question": "?", "answer": "a"}, "KeyError"),
    ({"id": "x", "image": "img/none.png", "question": "?", "answer": "a", "match": "exact"},
     "no image at"),
    ({"id": "x", "image": "questions.jsonl", "question": "?", "answer": "a", "match": "exact"},
     "not a JPEG, PNG or WebP"),
    ({"id": "q1", "question": "?", "answer": "a", "match": "exact"}, "ids used twice: q1"),
])
def test_unusable_question_rows_are_named(tmp_path, row, message):
    with pytest.raises(ValueError, match=message):
        load_questions(questions_file(tmp_path, [QUESTIONS[0], row]))
